# Validation effect for engine patch 1 (CompositorEffect.provides_indirect_light).
# Fills the GI ambient buffer with a constant linear color and the reflection buffer with alpha 0.
# - identity test: color = environment ambient color -> image must equal the baseline.
# - color test: any other color -> must show up as the ambient term on every opaque surface.
@tool
class_name GoghotGIInjectTest
extends CompositorEffect

var ambient := Color(1, 0, 0)

var _rd: RenderingDevice
var _shader: RID
var _pipeline: RID

const SRC := """
#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba16f, set = 0, binding = 0) uniform restrict writeonly image2D ambient_img;
layout(rgba16f, set = 0, binding = 1) uniform restrict writeonly image2D reflection_img;
layout(push_constant, std430) uniform Params {
	vec4 ambient;
	ivec2 size;
	ivec2 pad;
} params;
void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(p, params.size))) {
		return;
	}
	imageStore(ambient_img, p, vec4(params.ambient.rgb, 1.0));
	imageStore(reflection_img, p, vec4(0.0));
}
"""


func _init() -> void:
	effect_callback_type = EFFECT_CALLBACK_TYPE_PRE_OPAQUE
	provides_indirect_light = true
	_rd = RenderingServer.get_rendering_device()
	var src := RDShaderSource.new()
	src.source_compute = SRC
	var spirv := _rd.shader_compile_spirv_from_source(src)
	if not spirv.compile_error_compute.is_empty():
		push_error(spirv.compile_error_compute)
		return
	_shader = _rd.shader_create_from_spirv(spirv)
	_pipeline = _rd.compute_pipeline_create(_shader)


func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE and _shader.is_valid():
		_rd.free_rid(_shader)


func _render_callback(callback_type: int, render_data: RenderData) -> void:
	if callback_type != EFFECT_CALLBACK_TYPE_PRE_OPAQUE or not _pipeline.is_valid():
		return
	var rb: RenderSceneBuffersRD = render_data.get_render_scene_buffers()
	if rb == null or not rb.has_texture("rbgi", "ambient"):
		push_error("GI buffers missing: engine patch 1 not active?")
		return
	var size := rb.get_internal_size()
	var pc := PackedFloat32Array([ambient.r, ambient.g, ambient.b, 1.0]).to_byte_array()
	pc.append_array(PackedInt32Array([size.x, size.y, 0, 0]).to_byte_array())
	for view in rb.get_view_count():
		var u0 := RDUniform.new()
		u0.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
		u0.binding = 0
		u0.add_id(rb.get_texture_slice("rbgi", "ambient", view, 0, 1, 1))
		var u1 := RDUniform.new()
		u1.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
		u1.binding = 1
		u1.add_id(rb.get_texture_slice("rbgi", "reflection", view, 0, 1, 1))
		var uset := UniformSetCacheRD.get_cache(_shader, 0, [u0, u1])
		var cl := _rd.compute_list_begin()
		_rd.compute_list_bind_compute_pipeline(cl, _pipeline)
		_rd.compute_list_bind_uniform_set(cl, uset, 0)
		_rd.compute_list_set_push_constant(cl, pc, pc.size())
		_rd.compute_list_dispatch(cl, (size.x + 7) / 8, (size.y + 7) / 8, 1)
		_rd.compute_list_end()
