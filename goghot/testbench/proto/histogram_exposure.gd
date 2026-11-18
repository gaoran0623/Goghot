# Histogram auto exposure (GDScript prototype; to be ported into the goghot GDExtension).
#
# POST_TRANSPARENT compositor effect, GPU only (no readback, no frame latency):
#   1. histogram: log2 luminance of the HDR buffer into BINS bins over [MIN_LOG2, MAX_LOG2], center weighted
#   2. reduce:    weighted mean log2 luminance between low/high percentiles -> target EV,
#                 temporal adaptation (separate speeds for brightening / darkening), stored on GPU
#   3. apply:     color *= exposure scale, in place (glow/tonemap downstream see exposed HDR)
#
# Defaults are scene-independent on purpose (CLAUDE.md: no per-scene tuning).
@tool

extends CompositorEffect

const BINS := 128
const MIN_LOG2 := -14.0
const MAX_LOG2 := 6.0

## Percentile window over the (weighted) histogram. Pixels outside it are ignored.
@export_range(0.0, 1.0) var low_percentile := 0.6
@export_range(0.0, 1.0) var high_percentile := 0.95
## Final image middle-grey target (linear).
@export var key := 0.18
## Exposure compensation in EV (+ = brighter).
@export var compensation_ev := 0.0
@export var min_ev := -12.0
@export var max_ev := 12.0
## Adaptation speed in 1/s; brightening (scene got darker) is slower, like eyes.
@export var speed_up := 1.5
@export var speed_down := 3.0
## Center weighting: weight = 1 + center_weight * exp(-r^2 / (2 * center_sigma^2)), r = distance from center in screen heights.
@export var center_weight := 3.0
@export var center_sigma := 0.25
## Meter one pixel per METER_STRIDE x METER_STRIDE block (metering does not need full resolution,
## and it avoids atomic contention on uniform images).
const METER_STRIDE := 4
## Fixed timestep for deterministic captures (0 = use real frame delta).
var fixed_dt := 0.0
## Set to true to jump straight to the target exposure on the next frame (camera cuts, scene loads).
var snap := false

var _rd: RenderingDevice
var _shaders := {}
var _pipelines := {}
var _hist_buf: RID
var _state_buf: RID # float ev, float initialized
var _last_usec := 0

const COMMON := """
#version 450
#define BINS 128
layout(push_constant, std430) uniform Params {
	ivec2 size;
	float min_log2;
	float max_log2;
	float low_pct;
	float high_pct;
	float key;
	float comp_ev;
	float min_ev;
	float max_ev;
	float speed_up;
	float speed_down;
	float dt;
	float center_weight;
	float center_sigma;
	float snap;
	int stride;
	int pad0;
	int pad1;
	int pad2;
} params;
layout(set = 0, binding = 1, std430) restrict buffer Hist { uint hist[BINS]; };
layout(set = 0, binding = 2, std430) restrict buffer State { float ev; float initialized; };
float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
"""

const HIST := """
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, set = 0, binding = 0) uniform restrict readonly image2D color_img;
shared uint local_hist[BINS];
void main() {
	uint li = gl_LocalInvocationIndex;
	if (li < BINS) { local_hist[li] = 0u; }
	barrier();
	ivec2 p = ivec2(gl_GlobalInvocationID.xy) * params.stride + params.stride / 2;
	float l = all(lessThan(p, params.size)) ? lum(imageLoad(color_img, p).rgb) : 0.0;
	float lg = log2(max(l, 1e-12));
	// Pixels darker than the histogram range (e.g. unlit black) are ignored, not counted as darkest bin.
	if (all(lessThan(p, params.size)) && lg >= params.min_log2) {
		float t = clamp((lg - params.min_log2) / (params.max_log2 - params.min_log2), 0.0, 1.0);
		uint bin = min(uint(t * float(BINS)), uint(BINS - 1));
		vec2 uv = (vec2(p) + 0.5) / vec2(params.size) - 0.5;
		uv.x *= float(params.size.x) / float(params.size.y); // distance in units of screen height
		float w = 1.0 + params.center_weight * exp(-dot(uv, uv) / (2.0 * params.center_sigma * params.center_sigma));
		atomicAdd(local_hist[bin], uint(w * 16.0 + 0.5));
	}
	barrier();
	if (li < BINS && local_hist[li] > 0u) { atomicAdd(hist[li], local_hist[li]); }
}
"""

const REDUCE := """
layout(local_size_x = 1) in;
void main() {
	float total = 0.0;
	for (int i = 0; i < BINS; i++) { total += float(hist[i]); }
	float lo = total * params.low_pct;
	float hi = total * params.high_pct;
	float acc = 0.0, sum_w = 0.0, sum_log = 0.0;
	for (int i = 0; i < BINS; i++) {
		float c = float(hist[i]);
		// portion of this bin inside [lo, hi]
		float a = clamp(lo - acc, 0.0, c);
		float b = clamp(hi - acc, 0.0, c);
		float w = b - a;
		float center = params.min_log2 + (float(i) + 0.5) / float(BINS) * (params.max_log2 - params.min_log2);
		sum_w += w;
		sum_log += w * center;
		acc += c;
		hist[i] = 0u; // clear for next frame
	}
	float avg_log2 = sum_w > 0.0 ? sum_log / sum_w : log2(params.key);
	// EV here = -log2(scale); scale = key / avg luminance * 2^comp
	float target_ev = clamp(avg_log2 - log2(params.key) - params.comp_ev, params.min_ev, params.max_ev);
	if (initialized < 0.5 || params.snap > 0.5) {
		ev = target_ev;
		initialized = 1.0;
	} else {
		float speed = target_ev > ev ? params.speed_down : params.speed_up;
		ev += (target_ev - ev) * (1.0 - exp(-params.dt * speed));
	}
}
"""

const APPLY := """
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba16f, set = 0, binding = 0) uniform restrict image2D color_img;
void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(p, params.size))) { return; }
	vec4 c = imageLoad(color_img, p);
	imageStore(color_img, p, vec4(c.rgb * exp2(-ev), c.a));
}
"""


func _init() -> void:
	effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
	_rd = RenderingServer.get_rendering_device()
	for name in ["hist", "reduce", "apply"]:
		var body: String = {"hist": HIST, "reduce": REDUCE, "apply": APPLY}[name]
		var src := RDShaderSource.new()
		src.source_compute = COMMON + body
		var spirv := _rd.shader_compile_spirv_from_source(src)
		if not spirv.compile_error_compute.is_empty():
			push_error("histogram_exposure %s: %s" % [name, spirv.compile_error_compute])
			return
		_shaders[name] = _rd.shader_create_from_spirv(spirv)
		_pipelines[name] = _rd.compute_pipeline_create(_shaders[name])
	_hist_buf = _rd.storage_buffer_create(BINS * 4, PackedByteArray())
	_state_buf = _rd.storage_buffer_create(16, PackedFloat32Array([0, 0, 0, 0]).to_byte_array())


func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE and _rd:
		for s in _shaders.values():
			_rd.free_rid(s)
		for b in [_hist_buf, _state_buf]:
			if b.is_valid():
				_rd.free_rid(b)


func _uniform(type: int, binding: int, rid: RID) -> RDUniform:
	var u := RDUniform.new()
	u.uniform_type = type
	u.binding = binding
	u.add_id(rid)
	return u


func _render_callback(callback_type: int, render_data: RenderData) -> void:
	if callback_type != EFFECT_CALLBACK_TYPE_POST_TRANSPARENT or _pipelines.size() != 3:
		return
	var rb: RenderSceneBuffersRD = render_data.get_render_scene_buffers()
	if rb == null:
		return
	var size := rb.get_internal_size()
	var now := Time.get_ticks_usec()
	var dt := fixed_dt if fixed_dt > 0.0 else (clampf((now - _last_usec) / 1e6, 0.0, 0.25) if _last_usec > 0 else 0.0)
	_last_usec = now

	var pc := PackedInt32Array([size.x, size.y]).to_byte_array()
	pc.append_array(PackedFloat32Array([MIN_LOG2, MAX_LOG2, low_percentile, high_percentile, key, compensation_ev,
			min_ev, max_ev, speed_up, speed_down, dt, center_weight, center_sigma, 1.0 if snap else 0.0]).to_byte_array())
	pc.append_array(PackedInt32Array([METER_STRIDE, 0, 0, 0]).to_byte_array())
	snap = false

	var u_hist := _uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 1, _hist_buf)
	var u_state := _uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 2, _state_buf)
	# Meter view 0 only; apply the same exposure to all views.
	var color0 := rb.get_color_layer(0)

	var cl := _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(cl, _pipelines.hist)
	_rd.compute_list_bind_uniform_set(cl, UniformSetCacheRD.get_cache(_shaders.hist, 0,
			[_uniform(RenderingDevice.UNIFORM_TYPE_IMAGE, 0, color0), u_hist, u_state]), 0)
	_rd.compute_list_set_push_constant(cl, pc, pc.size())
	var msize := (size + Vector2i(METER_STRIDE - 1, METER_STRIDE - 1)) / METER_STRIDE
	_rd.compute_list_dispatch(cl, (msize.x + 15) / 16, (msize.y + 15) / 16, 1)
	_rd.compute_list_add_barrier(cl)

	_rd.compute_list_bind_compute_pipeline(cl, _pipelines.reduce)
	_rd.compute_list_bind_uniform_set(cl, UniformSetCacheRD.get_cache(_shaders.reduce, 0, [u_hist, u_state]), 0)
	_rd.compute_list_set_push_constant(cl, pc, pc.size())
	_rd.compute_list_dispatch(cl, 1, 1, 1)
	_rd.compute_list_add_barrier(cl)

	_rd.compute_list_bind_compute_pipeline(cl, _pipelines.apply)
	_rd.compute_list_set_push_constant(cl, pc, pc.size())
	for view in rb.get_view_count():
		_rd.compute_list_bind_uniform_set(cl, UniformSetCacheRD.get_cache(_shaders.apply, 0,
				[_uniform(RenderingDevice.UNIFORM_TYPE_IMAGE, 0, rb.get_color_layer(view)), u_hist, u_state]), 0)
		_rd.compute_list_dispatch(cl, (size.x + 7) / 8, (size.y + 7) / 8, 1)
	_rd.compute_list_end()
