#include "screen_probe_gi.h"

#include <godot_cpp/classes/rd_sampler_state.hpp>
#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/render_scene_data.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/uniform_set_cache_rd.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "gi/gi_common.glsl.gen.h"
#include "gi/gi_copy.glsl.gen.h"
#include "gi/gi_filter.glsl.gen.h"
#include "gi/gi_hzb_copy.glsl.gen.h"
#include "gi/gi_hzb_down.glsl.gen.h"
#include "gi/gi_probe_adaptive.glsl.gen.h"
#include "gi/gi_probe_place.glsl.gen.h"
#include "gi/gi_resolve.glsl.gen.h"
#include "gi/gi_temporal.glsl.gen.h"
#include "gi/gi_trace.glsl.gen.h"

using namespace godot;

namespace goghot {

using RD = RenderingDevice;

namespace {

// std140 layout of the Frame UBO in gi_common.glsl (368 bytes).
struct FrameUBO {
	float proj[16];
	float inv_proj[16];
	float cur_to_prev_view[16];
	float prev_proj[16];
	int32_t size[2];
	int32_t probe_grid[2];
	int32_t adaptive_grid[2];
	int32_t spacing;
	int32_t frame;
	float sky[4];
	int32_t rays;
	int32_t max_adaptive;
	int32_t hzb_levels;
	int32_t debug_mode;
	float thickness;
	float max_distance;
	float plane_threshold;
	float normal_power;
	float temporal_max_frames;
	float near_plane;
	int32_t has_history;
	int32_t max_steps;
	float pad_clip;
	float response_sigma;
	float response_min_frames;
	int32_t jitter;
};
static_assert(sizeof(FrameUBO) == 368, "Frame UBO layout");

void store(const Projection &p_m, float *r_out) {
	for (int c = 0; c < 4; c++) {
		for (int r = 0; r < 4; r++) {
			r_out[c * 4 + r] = p_m.columns[c][r];
		}
	}
}

void store(const Transform3D &p_t, float *r_out) {
	for (int c = 0; c < 3; c++) {
		Vector3 col = p_t.basis.get_column(c);
		r_out[c * 4 + 0] = col.x;
		r_out[c * 4 + 1] = col.y;
		r_out[c * 4 + 2] = col.z;
		r_out[c * 4 + 3] = 0.0f;
	}
	r_out[12] = p_t.origin.x;
	r_out[13] = p_t.origin.y;
	r_out[14] = p_t.origin.z;
	r_out[15] = 1.0f;
}

Ref<RDUniform> uni(RD::UniformType p_type, int p_binding, const RID &p_a, const RID &p_b = RID()) {
	Ref<RDUniform> u;
	u.instantiate();
	u->set_uniform_type(p_type);
	u->set_binding(p_binding);
	u->add_id(p_a);
	if (p_b.is_valid()) {
		u->add_id(p_b);
	}
	return u;
}

} // namespace

void GoghotScreenProbeGI::_create_pipelines() {
	rd = RenderingServer::get_singleton()->get_rendering_device();
	if (rd == nullptr) {
		return;
	}
	const char *bodies[PASS_MAX] = { GI_HZB_COPY_GLSL, GI_HZB_DOWN_GLSL, GI_PROBE_PLACE_GLSL, GI_PROBE_ADAPTIVE_GLSL,
		GI_TRACE_GLSL, GI_FILTER_GLSL, GI_RESOLVE_GLSL, GI_TEMPORAL_GLSL, GI_COPY_GLSL };
	for (int i = 0; i < PASS_MAX; i++) {
		Ref<RDShaderSource> src;
		src.instantiate();
		src->set_stage_source(RD::SHADER_STAGE_COMPUTE, String(GI_COMMON_GLSL) + String(bodies[i]));
		Ref<RDShaderSPIRV> spirv = rd->shader_compile_spirv_from_source(src);
		String err = spirv->get_stage_compile_error(RD::SHADER_STAGE_COMPUTE);
		if (!err.is_empty()) {
			UtilityFunctions::push_error("Goghot GI shader ", i, ": ", err);
			return;
		}
		shaders[i] = rd->shader_create_from_spirv(spirv);
		pipelines[i] = rd->compute_pipeline_create(shaders[i]);
	}
	Ref<RDSamplerState> ss;
	ss.instantiate();
	ss->set_repeat_u(RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE);
	ss->set_repeat_v(RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE);
	nearest_sampler = rd->sampler_create(ss);
	ss->set_min_filter(RD::SAMPLER_FILTER_LINEAR);
	ss->set_mag_filter(RD::SAMPLER_FILTER_LINEAR);
	linear_sampler = rd->sampler_create(ss);
	frame_ubo = rd->uniform_buffer_create(sizeof(FrameUBO));
	ready = true;
}

RID GoghotScreenProbeGI::_tex(RD::DataFormat p_format, const Vector2i &p_size, int p_mips, uint32_t p_usage) {
	Ref<RDTextureFormat> tf;
	tf.instantiate();
	tf->set_format(p_format);
	tf->set_width(p_size.x);
	tf->set_height(p_size.y);
	tf->set_mipmaps(p_mips);
	tf->set_usage_bits(p_usage);
	Ref<RDTextureView> tv;
	tv.instantiate();
	return rd->texture_create(tf, tv);
}

void GoghotScreenProbeGI::_free_sized() {
	if (rd == nullptr) {
		return;
	}
	for (const RID &m : hzb_mips) {
		rd->free_rid(m);
	}
	hzb_mips.clear();
	RID *rids[] = { &hzb, &probes, &counter, &adaptive_map, &radiance, &filtered, &resolved, &history[0], &history[1], &aux[0], &aux[1], &moments[0], &moments[1], &prev_color };
	for (RID *r : rids) {
		if (r->is_valid()) {
			rd->free_rid(*r);
		}
		*r = RID();
	}
	size = Vector2i();
	has_history = false;
	prev_color_valid = false;
}

void GoghotScreenProbeGI::_free_all() {
	if (rd == nullptr) {
		return;
	}
	_free_sized();
	for (int i = 0; i < PASS_MAX; i++) {
		if (shaders[i].is_valid()) {
			rd->free_rid(shaders[i]);
		}
		shaders[i] = RID();
		pipelines[i] = RID();
	}
	RID *rids[] = { &linear_sampler, &nearest_sampler, &frame_ubo };
	for (RID *r : rids) {
		if (r->is_valid()) {
			rd->free_rid(*r);
		}
		*r = RID();
	}
	ready = false;
	rd = nullptr;
}

void GoghotScreenProbeGI::_ensure_size(const Vector2i &p_size) {
	int spacing = CLAMP(probe_spacing, 4, 128) & ~1;
	Vector2i pg = (p_size + Vector2i(spacing - 1, spacing - 1)) / spacing;
	Vector2i ag = (p_size + Vector2i(spacing / 2 - 1, spacing / 2 - 1)) / (spacing / 2);
	if (p_size == size && pg == probe_grid && ag == adaptive_grid) {
		return;
	}
	_free_sized();
	size = p_size;
	probe_grid = pg;
	adaptive_grid = ag;
	max_adaptive = pg.x * pg.y; // budget: as many adaptive as uniform probes
	int max_dim = MAX(size.x, size.y);
	hzb_levels = 1;
	while ((1 << hzb_levels) <= max_dim && hzb_levels < 14) {
		hzb_levels++;
	}

	const uint32_t storage_sample = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
	hzb = _tex(RD::DATA_FORMAT_R32_SFLOAT, size, hzb_levels, storage_sample);
	for (int i = 0; i < hzb_levels; i++) {
		Ref<RDTextureView> tv;
		tv.instantiate();
		hzb_mips.push_back(rd->texture_create_shared_from_slice(tv, hzb, 0, i, 1, RD::TEXTURE_SLICE_2D));
	}
	int total_probes = pg.x * pg.y + max_adaptive;
	probes = rd->storage_buffer_create(total_probes * 32);
	radiance = rd->storage_buffer_create(total_probes * 16);
	filtered = rd->storage_buffer_create(total_probes * 16);
	counter = rd->storage_buffer_create(16);
	adaptive_map = rd->storage_buffer_create(ag.x * ag.y * 4);
	resolved = _tex(RD::DATA_FORMAT_R16G16B16A16_SFLOAT, size, 1, storage_sample);
	for (int i = 0; i < 2; i++) {
		history[i] = _tex(RD::DATA_FORMAT_R16G16B16A16_SFLOAT, size, 1, storage_sample);
		aux[i] = _tex(RD::DATA_FORMAT_R32G32B32A32_SFLOAT, size, 1, storage_sample);
		moments[i] = _tex(RD::DATA_FORMAT_R32G32_SFLOAT, size, 1, storage_sample);
	}
	prev_color = _tex(RD::DATA_FORMAT_R16G16B16A16_SFLOAT, size, 1, storage_sample);
	has_history = false;
	prev_color_valid = false;
}

void GoghotScreenProbeGI::_dispatch(int64_t p_list, Pass p_pass, const TypedArray<RDUniform> &p_uniforms, uint32_t p_x, uint32_t p_y) {
	TypedArray<RDUniform> us;
	us.push_back(uni(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, frame_ubo));
	us.append_array(p_uniforms);
	rd->compute_list_bind_compute_pipeline(p_list, pipelines[p_pass]);
	rd->compute_list_bind_uniform_set(p_list, UniformSetCacheRD::get_cache(shaders[p_pass], 0, us), 0);
	rd->compute_list_dispatch(p_list, MAX(p_x, 1u), MAX(p_y, 1u), 1);
	rd->compute_list_add_barrier(p_list);
}

void GoghotScreenProbeGI::_render_callback(int32_t p_effect_callback_type, RenderData *p_render_data) {
	if (p_effect_callback_type != EFFECT_CALLBACK_TYPE_PRE_OPAQUE || !ready || p_render_data == nullptr) {
		return;
	}
	RenderSceneBuffersRD *rb = Object::cast_to<RenderSceneBuffersRD>(p_render_data->get_render_scene_buffers().ptr());
	RenderSceneData *sd = p_render_data->get_render_scene_data();
	if (rb == nullptr || sd == nullptr) {
		return;
	}
	if (!rb->has_texture("rbgi", "ambient")) {
		WARN_PRINT_ONCE("GoghotScreenProbeGI: GI buffers missing (engine patch 1 not active, or SDFGI/VoxelGI enabled).");
		return;
	}
	if (rb->get_view_count() != 1) {
		WARN_PRINT_ONCE("GoghotScreenProbeGI: multiview is not supported yet.");
		return;
	}
	_ensure_size(rb->get_internal_size());

	const Projection proj = sd->get_cam_projection();
	const Transform3D cam = sd->get_cam_transform();
	const Projection inv_proj = proj.inverse();
	const Vector4 near_h = inv_proj.xform(Vector4(0, 0, 1, 1)); // reverse Z: NDC z = 1 is the near plane
	const float near_plane = -(near_h.z / near_h.w);
	if (!has_history) {
		prev_proj = proj;
		prev_cam = cam;
	}

	FrameUBO f = {};
	store(proj, f.proj);
	store(inv_proj, f.inv_proj);
	store(prev_cam.affine_inverse() * cam, f.cur_to_prev_view);
	store(prev_proj, f.prev_proj);
	f.size[0] = size.x;
	f.size[1] = size.y;
	f.probe_grid[0] = probe_grid.x;
	f.probe_grid[1] = probe_grid.y;
	f.adaptive_grid[0] = adaptive_grid.x;
	f.adaptive_grid[1] = adaptive_grid.y;
	f.spacing = CLAMP(probe_spacing, 4, 128) & ~1;
	f.frame = frame;
	f.sky[0] = sky_color.r * sky_energy;
	f.sky[1] = sky_color.g * sky_energy;
	f.sky[2] = sky_color.b * sky_energy;
	f.rays = CLAMP(rays_per_probe, 1, 32);
	f.max_adaptive = max_adaptive;
	f.hzb_levels = hzb_levels;
	f.debug_mode = debug_mode;
	f.thickness = thickness;
	f.max_distance = max_distance;
	f.plane_threshold = plane_threshold;
	f.normal_power = normal_power;
	f.temporal_max_frames = float(MAX(temporal_max_frames, 1));
	f.near_plane = near_plane;
	f.has_history = (has_history && prev_color_valid) ? 1 : 0;
	f.max_steps = max_steps;
	f.response_sigma = response_sigma;
	f.response_min_frames = float(MAX(response_min_frames, 1));
	f.jitter = jitter_probes ? 1 : 0;
	PackedByteArray ubo;
	ubo.resize(sizeof(f));
	memcpy(ubo.ptrw(), &f, sizeof(f));
	rd->buffer_update(frame_ubo, 0, sizeof(f), ubo);

	const RID depth = rb->get_depth_layer(0);
	const RID normal = rb->get_texture_slice("forward_clustered", "normal_roughness", 0, 0, 1, 1);
	const RID ambient = rb->get_texture_slice("rbgi", "ambient", 0, 0, 1, 1);
	const RID reflection = rb->get_texture_slice("rbgi", "reflection", 0, 0, 1, 1);
	const Vector2i groups8 = (size + Vector2i(7, 7)) / 8;
	const int total_probes = probe_grid.x * probe_grid.y + max_adaptive;
	auto depth_u = [&](int b) { return uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, b, nearest_sampler, depth); };
	auto normal_u = [&](int b) { return uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, b, nearest_sampler, normal); };
	auto buf = [&](int b, const RID &r) { return uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, b, r); };
	auto img = [&](int b, const RID &r) { return uni(RD::UNIFORM_TYPE_IMAGE, b, r); };

	// GPU timing markers (read by the test bench: RenderingDevice.get_captured_timestamp_*).
	rd->capture_timestamp("goghot_gi_begin");
	const int64_t cl = rd->compute_list_begin();
	{ // HZB
		TypedArray<RDUniform> u;
		u.push_back(depth_u(1));
		u.push_back(img(2, hzb_mips[0]));
		_dispatch(cl, PASS_HZB_COPY, u, groups8.x, groups8.y);
		Vector2i ms = size;
		for (int i = 1; i < hzb_levels; i++) {
			ms = Vector2i(MAX(ms.x / 2, 1), MAX(ms.y / 2, 1));
			TypedArray<RDUniform> ud;
			ud.push_back(img(1, hzb_mips[i - 1]));
			ud.push_back(img(2, hzb_mips[i]));
			_dispatch(cl, PASS_HZB_DOWN, ud, (ms.x + 7) / 8, (ms.y + 7) / 8);
		}
	}
	{ // Probes
		TypedArray<RDUniform> u;
		u.push_back(depth_u(1));
		u.push_back(normal_u(2));
		u.push_back(buf(3, probes));
		u.push_back(buf(4, counter));
		_dispatch(cl, PASS_PROBE_PLACE, u, (probe_grid.x + 7) / 8, (probe_grid.y + 7) / 8);
		u.push_back(buf(5, adaptive_map));
		_dispatch(cl, PASS_PROBE_ADAPTIVE, u, (adaptive_grid.x + 7) / 8, (adaptive_grid.y + 7) / 8);
	}
	{ // Trace
		TypedArray<RDUniform> u;
		u.push_back(uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, nearest_sampler, hzb));
		u.push_back(uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 2, linear_sampler, prev_color));
		u.push_back(buf(3, probes));
		u.push_back(buf(4, counter));
		u.push_back(buf(5, radiance));
		_dispatch(cl, PASS_TRACE, u, total_probes, 1);
	}
	{ // Probe filter
		TypedArray<RDUniform> u;
		u.push_back(buf(1, probes));
		u.push_back(buf(2, counter));
		u.push_back(buf(3, radiance));
		u.push_back(buf(4, filtered));
		_dispatch(cl, PASS_FILTER, u, (total_probes + 63) / 64, 1);
	}
	{ // Resolve
		TypedArray<RDUniform> u;
		u.push_back(depth_u(1));
		u.push_back(normal_u(2));
		u.push_back(buf(3, probes));
		u.push_back(buf(4, filtered));
		u.push_back(buf(5, adaptive_map));
		u.push_back(img(6, resolved));
		_dispatch(cl, PASS_RESOLVE, u, groups8.x, groups8.y);
	}
	{ // Temporal + output
		TypedArray<RDUniform> u;
		u.push_back(depth_u(1));
		u.push_back(normal_u(2));
		u.push_back(img(3, resolved));
		u.push_back(uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 4, nearest_sampler, history[ping ^ 1]));
		u.push_back(uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 5, nearest_sampler, aux[ping ^ 1]));
		u.push_back(img(6, history[ping]));
		u.push_back(img(7, aux[ping]));
		u.push_back(img(8, ambient));
		u.push_back(img(9, reflection));
		u.push_back(uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 10, nearest_sampler, moments[ping ^ 1]));
		u.push_back(img(11, moments[ping]));
		_dispatch(cl, PASS_TEMPORAL, u, groups8.x, groups8.y);
	}
	rd->compute_list_end();
	rd->capture_timestamp("goghot_gi_end");

	prev_proj = proj;
	prev_cam = cam;
	has_history = true;
	ping ^= 1;
	frame++;
}

void GoghotScreenProbeGI::_capture_color(RenderSceneBuffersRD *p_rb) {
	if (!ready || !prev_color.is_valid() || p_rb->get_internal_size() != size) {
		return;
	}
	TypedArray<RDUniform> u;
	u.push_back(uni(RD::UNIFORM_TYPE_IMAGE, 1, p_rb->get_color_layer(0)));
	u.push_back(uni(RD::UNIFORM_TYPE_IMAGE, 2, prev_color));
	rd->capture_timestamp("goghot_capture_begin");
	const int64_t cl = rd->compute_list_begin();
	_dispatch(cl, PASS_COPY, u, (size.x + 7) / 8, (size.y + 7) / 8);
	rd->compute_list_end();
	rd->capture_timestamp("goghot_capture_end");
	prev_color_valid = true;
}

Ref<GoghotGIColorCapture> GoghotScreenProbeGI::create_color_capture() {
	Ref<GoghotGIColorCapture> c;
	c.instantiate();
	c->set_gi(Ref<GoghotScreenProbeGI>(this));
	return c;
}

void GoghotScreenProbeGI::reset_history() {
	has_history = false;
}

void GoghotScreenProbeGI::_notification(int p_what) {
	if (p_what == NOTIFICATION_PREDELETE) {
		_free_all();
	}
}

GoghotScreenProbeGI::GoghotScreenProbeGI() {
	set_effect_callback_type(EFFECT_CALLBACK_TYPE_PRE_OPAQUE);
	set_provides_indirect_light(true);
	_create_pipelines();
}

GoghotScreenProbeGI::~GoghotScreenProbeGI() {
	_free_all();
}

void GoghotScreenProbeGI::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_color_capture"), &GoghotScreenProbeGI::create_color_capture);
	ClassDB::bind_method(D_METHOD("reset_history"), &GoghotScreenProbeGI::reset_history);
#define GOGHOT_BIND(m_vtype, m_name, m_hint, m_hint_str)                                                \
	ClassDB::bind_method(D_METHOD("set_" #m_name, "value"), &GoghotScreenProbeGI::set_##m_name);       \
	ClassDB::bind_method(D_METHOD("get_" #m_name), &GoghotScreenProbeGI::get_##m_name);                \
	ADD_PROPERTY(PropertyInfo(m_vtype, #m_name, m_hint, m_hint_str), "set_" #m_name, "get_" #m_name);
	GOGHOT_BIND(Variant::INT, probe_spacing, PROPERTY_HINT_RANGE, "8,64,2,suffix:px")
	GOGHOT_BIND(Variant::INT, rays_per_probe, PROPERTY_HINT_RANGE, "4,32,1")
	GOGHOT_BIND(Variant::FLOAT, thickness, PROPERTY_HINT_RANGE, "0.001,1,0.001")
	GOGHOT_BIND(Variant::FLOAT, max_distance, PROPERTY_HINT_RANGE, "1,1000,1,suffix:m")
	GOGHOT_BIND(Variant::FLOAT, plane_threshold, PROPERTY_HINT_RANGE, "0.001,1,0.001")
	GOGHOT_BIND(Variant::FLOAT, normal_power, PROPERTY_HINT_RANGE, "0,64,0.5")
	GOGHOT_BIND(Variant::INT, temporal_max_frames, PROPERTY_HINT_RANGE, "1,64,1")
	GOGHOT_BIND(Variant::INT, max_steps, PROPERTY_HINT_RANGE, "8,256,1")
	GOGHOT_BIND(Variant::FLOAT, response_sigma, PROPERTY_HINT_RANGE, "0,8,0.05")
	GOGHOT_BIND(Variant::INT, response_min_frames, PROPERTY_HINT_RANGE, "1,64,1")
	GOGHOT_BIND(Variant::BOOL, jitter_probes, PROPERTY_HINT_NONE, "")
	GOGHOT_BIND(Variant::COLOR, sky_color, PROPERTY_HINT_NONE, "")
	GOGHOT_BIND(Variant::FLOAT, sky_energy, PROPERTY_HINT_RANGE, "0,16,0.01")
	GOGHOT_BIND(Variant::INT, debug_mode, PROPERTY_HINT_ENUM, "Off,No Temporal,No Spatial Filter")
#undef GOGHOT_BIND
}

// ---------------------------------------------------------------------------

void GoghotGIColorCapture::_render_callback(int32_t p_effect_callback_type, RenderData *p_render_data) {
	if (p_effect_callback_type != EFFECT_CALLBACK_TYPE_POST_SKY || gi.is_null() || p_render_data == nullptr) {
		return;
	}
	RenderSceneBuffersRD *rb = Object::cast_to<RenderSceneBuffersRD>(p_render_data->get_render_scene_buffers().ptr());
	if (rb != nullptr) {
		gi->_capture_color(rb);
	}
}

GoghotGIColorCapture::GoghotGIColorCapture() {
	set_effect_callback_type(EFFECT_CALLBACK_TYPE_POST_SKY);
}

void GoghotGIColorCapture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_gi", "gi"), &GoghotGIColorCapture::set_gi);
	ClassDB::bind_method(D_METHOD("get_gi"), &GoghotGIColorCapture::get_gi);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "gi", PROPERTY_HINT_RESOURCE_TYPE, "GoghotScreenProbeGI"), "set_gi", "get_gi");
}

} // namespace goghot
