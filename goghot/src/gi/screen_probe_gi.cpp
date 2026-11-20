#include "screen_probe_gi.h"

#include <godot_cpp/classes/time.hpp>

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
#include "gi/gi_debug.glsl.gen.h"
#include "gi/gi_filter.glsl.gen.h"
#include "gi/gi_voxel_common.glsl.gen.h"
#include "gi/gi_fog_accumulate.glsl.gen.h"
#include "gi/gi_gather.glsl.gen.h"
#include "gi/gi_hzb_copy.glsl.gen.h"
#include "gi/gi_hzb_down.glsl.gen.h"
#include "gi/gi_probe_adaptive.glsl.gen.h"
#include "gi/gi_probe_place.glsl.gen.h"
#include "gi/gi_refl_resolve.glsl.gen.h"
#include "gi/gi_refl_trace.glsl.gen.h"
#include "gi/gi_screen_trace.glsl.gen.h"
#include "gi/gi_resolve.glsl.gen.h"
#include "gi/gi_temporal.glsl.gen.h"
#include "gi/gi_trace.glsl.gen.h"

using namespace godot;

namespace goghot {

using RD = RenderingDevice;

namespace {

// std140 layout of the Frame UBO in gi_common.glsl (624 bytes).
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
	float view_to_world[16];
	int32_t light_count;
	int32_t cache_mask;
	int32_t rt_enabled;
	int32_t cache_mature_frames;
	float cache_min_cell;
	float cache_cell_scale;
	float cache_max_frames;
	float pad3;
	int32_t fog_grid[4];
	float fog_origin[4];
	int32_t fog_shift[4];
	float fog_max_frames;
	float pad4[3];
	int32_t refl_grid[4];
	float refl_params[4];
	float vox_origin[3][4];
	int32_t vox_params[4];
	float sky_sh[9][4];
};
static_assert(sizeof(FrameUBO) == 768, "Frame UBO layout");

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
		GI_TRACE_GLSL, GI_GATHER_GLSL, GI_FILTER_GLSL, GI_RESOLVE_GLSL, GI_TEMPORAL_GLSL, GI_FOG_ACCUMULATE_GLSL, GI_REFL_TRACE_GLSL, GI_REFL_RESOLVE_GLSL, GI_COPY_GLSL };
	for (int i = 0; i < PASS_MAX; i++) {
		Ref<RDShaderSource> src;
		src.instantiate();
		const bool screen_trace = i == PASS_TRACE || i == PASS_REFL_TRACE;
		src->set_stage_source(RD::SHADER_STAGE_COMPUTE, String("#version 450\n") + String(GI_COMMON_GLSL) + (screen_trace ? String(GI_SCREEN_TRACE_GLSL) : String()) + String(bodies[i]));
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
	for (int i = 0; i < 3; i++) { // debug views
		String src_str = String("#version 450\n") + String(GI_COMMON_GLSL) + "#define DEBUG_PASS " + itos(i) + "\n";
		if (i == 2) {
			src_str += String("#define VOX_OCC_BINDING 6\n#define VOX_NORMAL_BINDING 7\n") + String(GI_VOXEL_COMMON_GLSL);
		}
		Ref<RDShaderSource> src;
		src.instantiate();
		src->set_stage_source(RD::SHADER_STAGE_COMPUTE, src_str + String(GI_DEBUG_GLSL));
		Ref<RDShaderSPIRV> spirv = rd->shader_compile_spirv_from_source(src);
		String err = spirv->get_stage_compile_error(RD::SHADER_STAGE_COMPUTE);
		if (!err.is_empty()) {
			UtilityFunctions::push_error("Goghot debug shader ", i, ": ", err);
			continue;
		}
		debug_shaders[i] = rd->shader_create_from_spirv(spirv);
		debug_pipelines[i] = rd->compute_pipeline_create(debug_shaders[i]);
	}
	if (!world.init(rd, String(), String())) {
		UtilityFunctions::print_verbose("Goghot: hardware ray tracing unavailable, world tracer disabled (screen + sky only).");
	}
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
	RID *rids[] = { &hzb, &probes, &counter, &adaptive_map, &rays, &radiance, &filtered, &resolved, &history[0], &history[1], &aux[0], &aux[1], &moments[0], &moments[1], &prev_color, &refl_rays, &refl_history[0], &refl_history[1] };
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

void GoghotScreenProbeGI::_free_fog() {
	if (rd == nullptr) {
		return;
	}
	if (fog_env_bound.is_valid()) {
		RenderingServer::get_singleton()->environment_set_volumetric_fog_gi_volume(fog_env_bound, RID(), 0.0f);
		fog_env_bound = RID();
	}
	for (RID *r : { &fog_rays, &fog_volume, &fog_history }) {
		if (r->is_valid()) {
			rd->free_rid(*r);
		}
		*r = RID();
	}
	fog_res_alloc = 0;
	fog_has_history = false;
}

void GoghotScreenProbeGI::_free_all() {
	if (rd == nullptr) {
		return;
	}
	_free_sized();
	_free_fog();
	world.free();
	for (RID &s : debug_shaders) {
		if (s.is_valid()) {
			rd->free_rid(s);
		}
		s = RID();
	}
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
	rays = rd->storage_buffer_create(total_probes * 32 * 16);
	radiance = rd->storage_buffer_create(total_probes * 16);
	filtered = rd->storage_buffer_create(total_probes * 16);
	counter = rd->storage_buffer_create(16);
	adaptive_map = rd->storage_buffer_create(ag.x * ag.y * 4);
	resolved = _tex(RD::DATA_FORMAT_R16G16B16A16_SFLOAT, size, 1, storage_sample);
	for (int i = 0; i < 2; i++) {
		history[i] = _tex(RD::DATA_FORMAT_R16G16B16A16_SFLOAT, size, 1, storage_sample);
		aux[i] = _tex(RD::DATA_FORMAT_R32G32B32A32_SFLOAT, size, 1, storage_sample);
		moments[i] = _tex(RD::DATA_FORMAT_R32G32B32A32_SFLOAT, size, 1, storage_sample);
	}
	prev_color = _tex(RD::DATA_FORMAT_R16G16B16A16_SFLOAT, size, 1, storage_sample);
	refl_size = (size + Vector2i(1, 1)) / 2;
	refl_rays = rd->storage_buffer_create(refl_size.x * refl_size.y * 48);
	for (int i = 0; i < 2; i++) {
		refl_history[i] = _tex(RD::DATA_FORMAT_R16G16B16A16_SFLOAT, size, 1, storage_sample);
	}
	has_history = false;
	prev_color_valid = false;
}

void GoghotScreenProbeGI::_dispatch(int64_t p_list, Pass p_pass, const TypedArray<RDUniform> &p_uniforms, uint32_t p_x, uint32_t p_y, uint32_t p_z) {
	TypedArray<RDUniform> us;
	us.push_back(uni(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, frame_ubo));
	us.append_array(p_uniforms);
	rd->compute_list_bind_compute_pipeline(p_list, pipelines[p_pass]);
	rd->compute_list_bind_uniform_set(p_list, UniformSetCacheRD::get_cache(shaders[p_pass], 0, us), 0);
	rd->compute_list_dispatch(p_list, MAX(p_x, 1u), MAX(p_y, 1u), MAX(p_z, 1u));
	rd->compute_list_add_barrier(p_list);
}

void GoghotScreenProbeGI::_render_callback(int32_t p_effect_callback_type, RenderData *p_render_data) {
	if (p_effect_callback_type != EFFECT_CALLBACK_TYPE_PRE_OPAQUE || !ready || p_render_data == nullptr) {
		return;
	}
	const uint64_t cpu_t0 = Time::get_singleton()->get_ticks_usec();
	_render_gi(p_render_data);
	render_cpu_ms = double(Time::get_singleton()->get_ticks_usec() - cpu_t0) / 1000.0;
}

void GoghotScreenProbeGI::_render_gi(RenderData *p_render_data) {
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

	const int total_probes = probe_grid.x * probe_grid.y + max_adaptive;
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
	f.sky[3] = sky_sh.size() == 9 ? 1.0f : 0.0f;
	for (int i = 0; i < 9 && sky_sh.size() == 9; i++) {
		f.sky_sh[i][0] = sky_sh[i].r * sky_energy;
		f.sky_sh[i][1] = sky_sh[i].g * sky_energy;
		f.sky_sh[i][2] = sky_sh[i].b * sky_energy;
	}
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
	{
		std::lock_guard<std::mutex> lock(scene_mutex);
		if (pending_scene) {
			scene = pending_scene;
			pending_scene.reset();
		}
	}
	// Tracer tier: A = hardware RT, B = voxels, C = none (screen + sky). Auto picks A when supported.
	const int requested = tracer == TRACER_AUTO ? (world.is_rt_supported() ? TRACER_A : TRACER_B) : tracer;
	const bool tier_b = requested == TRACER_B || (requested == TRACER_A && !world.is_rt_supported());
	world.set_deformed_blas_cache(deformed_blas_cache, uint32_t(MAX(deformed_triangle_budget, 1)));
	const bool world_active = world_tracing && requested != TRACER_C && world.is_ready() && scene &&
			world.update_scene(*scene, deformed_geometry, tier_b ? GIWorldTracer::TIER_B_VOXEL : GIWorldTracer::TIER_A_RT);
	active_tier = !world_active ? TRACER_C : (tier_b ? TRACER_B : TRACER_A);
	if (world_active && tier_b) {
		world.voxel_setup(cam.origin, CLAMP(voxel_resolution, 32, 256) & ~3, MAX(voxel_size, 0.01f), f.vox_origin);
		f.vox_params[0] = CLAMP(voxel_resolution, 32, 256) & ~3;
		f.vox_params[1] = GIWorldTracer::VOXEL_CASCADES;
		f.vox_params[2] = 1;
		f.vox_params[3] = CLAMP(voxel_bounce_rays, 1, 16);
	}
	store(cam, f.view_to_world);
	f.light_count = world_active ? int32_t(world.light_count()) : 0;
	f.cache_mask = int32_t(GIWorldTracer::CACHE_ENTRIES - 1);
	f.rt_enabled = world_active ? 1 : 0;
	f.cache_mature_frames = cache_mature_frames;
	f.cache_min_cell = cache_min_cell;
	f.cache_cell_scale = cache_cell_scale;
	f.cache_max_frames = float(MAX(active_tier == TRACER_B ? voxel_cache_max_frames : cache_max_frames, 1));

	// Fog GI volume: centered on the camera, snapped to whole voxels, half size = fog length.
	const bool fog_active = world_active && fog_gi && scene->fog_enabled && scene->fog_length > 0.0f;
	const int fog_n = CLAMP(fog_resolution, 8, 128) & ~1;
	const int fog_rpv = CLAMP(fog_rays_per_voxel, 1, 64);
	Vector3i fog_shift;
	if (fog_active) {
		if (fog_res_alloc != fog_n) {
			_free_fog();
			fog_rays = rd->storage_buffer_create(fog_n * fog_n * fog_n * fog_rpv * 16 + 16);
			Ref<RDTextureFormat> tf;
			tf.instantiate();
			tf->set_texture_type(RD::TEXTURE_TYPE_3D);
			tf->set_format(RD::DATA_FORMAT_R16G16B16A16_SFLOAT);
			tf->set_width(fog_n);
			tf->set_height(fog_n);
			tf->set_depth(fog_n);
			tf->set_usage_bits(RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT);
			Ref<RDTextureView> tv;
			tv.instantiate();
			fog_volume = rd->texture_create(tf, tv);
			fog_history = rd->texture_create(tf, tv);
			rd->texture_clear(fog_volume, Color(0, 0, 0, 0), 0, 1, 0, 1);
			fog_res_alloc = fog_n;
			fog_has_history = false;
		}
		const float voxel = 2.0f * scene->fog_length / float(fog_n);
		const Vector3 cp = cam.origin / voxel;
		const Vector3i cell(int(Math::floor(cp.x)), int(Math::floor(cp.y)), int(Math::floor(cp.z)));
		fog_shift = fog_has_history ? cell - fog_center_cell : Vector3i();
		fog_center_cell = cell;
		const Vector3 origin = Vector3(cell - Vector3i(fog_n / 2, fog_n / 2, fog_n / 2)) * voxel;
		f.fog_origin[0] = origin.x;
		f.fog_origin[1] = origin.y;
		f.fog_origin[2] = origin.z;
		f.fog_origin[3] = voxel;
	} else if (fog_env_bound.is_valid()) {
		_free_fog();
	}
	f.fog_grid[0] = fog_n;
	f.fog_grid[1] = fog_rpv;
	f.fog_grid[2] = fog_active ? 1 : 0;
	f.fog_grid[3] = total_probes * 32;
	f.fog_shift[0] = fog_shift.x;
	f.fog_shift[1] = fog_shift.y;
	f.fog_shift[2] = fog_shift.z;
	f.fog_max_frames = float(MAX(fog_max_frames, 1));
	const uint32_t fog_ray_total = fog_active ? uint32_t(fog_n * fog_n * fog_n * fog_rpv) : 0u;
	f.refl_grid[0] = refl_size.x;
	f.refl_grid[1] = refl_size.y;
	f.refl_grid[2] = reflections ? 1 : 0;
	f.refl_grid[3] = total_probes * 32 + int32_t(fog_ray_total);
	f.refl_params[0] = CLAMP(reflection_max_roughness, 0.0f, 1.0f);
	f.refl_params[1] = float(MAX(reflection_min_frames, 1));
	f.refl_params[2] = float(MAX(reflection_max_frames, 1));
	f.refl_params[3] = 32.0f;
	PackedByteArray ubo;
	ubo.resize(sizeof(f));
	memcpy(ubo.ptrw(), &f, sizeof(f));
	rd->buffer_update(frame_ubo, 0, sizeof(f), ubo);

	const RID depth = rb->get_depth_layer(0);
	const RID normal = rb->get_texture_slice("forward_clustered", "normal_roughness", 0, 0, 1, 1);
	const RID ambient = rb->get_texture_slice("rbgi", "ambient", 0, 0, 1, 1);
	const RID reflection = rb->get_texture_slice("rbgi", "reflection", 0, 0, 1, 1);
	const Vector2i groups8 = (size + Vector2i(7, 7)) / 8;
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
		u.push_back(buf(5, rays));
		_dispatch(cl, PASS_TRACE, u, total_probes, 1);
	}
	if (reflections) { // Reflection rays (half resolution), screen trace
		TypedArray<RDUniform> u;
		u.push_back(uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, nearest_sampler, hzb));
		u.push_back(uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 2, linear_sampler, prev_color));
		u.push_back(depth_u(3));
		u.push_back(normal_u(4));
		u.push_back(buf(5, refl_rays));
		_dispatch(cl, PASS_REFL_TRACE, u, (refl_size.x + 7) / 8, (refl_size.y + 7) / 8);
	}
	rd->compute_list_end();
	if (world_active) {
		const uint32_t refl_ray_count = reflections ? uint32_t(refl_size.x * refl_size.y) : 0u;
		const uint32_t launch = uint32_t(total_probes) * 32u + fog_ray_total + refl_ray_count;
		if (active_tier == TRACER_B) {
			world.voxel_update(frame_ubo);
			world.voxel_trace(frame_ubo, probes, rays, counter, fog_active ? fog_rays : rays, refl_rays, launch);
		} else {
			world.trace(frame_ubo, probes, rays, counter, fog_active ? fog_rays : rays, refl_rays, launch);
		}
	}
	const int64_t cl2 = rd->compute_list_begin();
	{ // Gather rays -> probe radiance
		TypedArray<RDUniform> u;
		u.push_back(buf(1, counter));
		u.push_back(buf(2, rays));
		u.push_back(buf(3, radiance));
		_dispatch(cl2, PASS_GATHER, u, (total_probes + 63) / 64, 1);
	}
	{ // Probe filter
		TypedArray<RDUniform> u;
		u.push_back(buf(1, probes));
		u.push_back(buf(2, counter));
		u.push_back(buf(3, radiance));
		u.push_back(buf(4, filtered));
		_dispatch(cl2, PASS_FILTER, u, (total_probes + 63) / 64, 1);
	}
	{ // Resolve
		TypedArray<RDUniform> u;
		u.push_back(depth_u(1));
		u.push_back(normal_u(2));
		u.push_back(buf(3, probes));
		u.push_back(buf(4, filtered));
		u.push_back(buf(5, adaptive_map));
		u.push_back(img(6, resolved));
		_dispatch(cl2, PASS_RESOLVE, u, groups8.x, groups8.y);
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
		_dispatch(cl2, PASS_TEMPORAL, u, groups8.x, groups8.y);
	}
	if (reflections) { // Reflection resolve (full resolution) -> rbgi/reflection
		TypedArray<RDUniform> u;
		u.push_back(depth_u(1));
		u.push_back(normal_u(2));
		u.push_back(buf(3, refl_rays));
		u.push_back(uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 4, nearest_sampler, refl_history[ping ^ 1]));
		u.push_back(uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 5, nearest_sampler, aux[ping ^ 1]));
		u.push_back(img(6, refl_history[ping]));
		u.push_back(img(7, ambient));
		u.push_back(img(8, reflection));
		_dispatch(cl2, PASS_REFL_RESOLVE, u, groups8.x, groups8.y);
	}
	if (fog_active) { // Fog GI volume accumulate (history -> volume), then keep a copy as next history.
		TypedArray<RDUniform> u;
		u.push_back(buf(1, fog_rays));
		u.push_back(uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 2, nearest_sampler, fog_history));
		u.push_back(img(3, fog_volume));
		_dispatch(cl2, PASS_FOG_ACCUMULATE, u, (fog_n + 3) / 4, (fog_n + 3) / 4, (fog_n + 3) / 4);
	}
	rd->compute_list_end();
	if (fog_active) {
		rd->texture_copy(fog_volume, fog_history, Vector3(), Vector3(), Vector3(fog_n, fog_n, fog_n), 0, 0, 0, 0);
		fog_has_history = true;
		const RID env = p_render_data->get_environment();
		if (env != fog_env_bound || scene->fog_length != fog_extent_bound) {
			if (fog_env_bound.is_valid() && env != fog_env_bound) {
				RenderingServer::get_singleton()->environment_set_volumetric_fog_gi_volume(fog_env_bound, RID(), 0.0f);
			}
			RenderingServer::get_singleton()->environment_set_volumetric_fog_gi_volume(env, fog_volume, scene->fog_length);
			fog_env_bound = env;
			fog_extent_bound = scene->fog_length;
		}
	}
	if (world_active) {
		world.resolve_cache(frame_ubo);
	}
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

void GoghotScreenProbeGI::_render_debug(RenderSceneBuffersRD *p_rb, int p_mode) {
	if (!ready || !probes.is_valid() || p_rb->get_internal_size() != size) {
		return;
	}
	const int pass = p_mode == 3 ? 1 : (p_mode == 4 ? 2 : 0);
	if (!debug_shaders[pass].is_valid() || (pass == 2 && active_tier != TRACER_B)) {
		return; // voxel view only exists in tier B
	}
	TypedArray<RDUniform> u;
	u.push_back(uni(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, frame_ubo));
	u.push_back(uni(RD::UNIFORM_TYPE_IMAGE, 1, p_rb->get_color_layer(0)));
	if (pass == 0) {
		u.push_back(uni(RD::UNIFORM_TYPE_IMAGE, 2, p_rb->get_texture_slice("rbgi", "ambient", 0, 0, 1, 1)));
		u.push_back(uni(RD::UNIFORM_TYPE_IMAGE, 3, p_rb->get_texture_slice("rbgi", "reflection", 0, 0, 1, 1)));
	} else if (pass == 1) {
		u.push_back(uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 4, probes));
		u.push_back(uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 5, counter));
	} else {
		u.push_back(uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 6, nearest_sampler, world.voxel_debug_texture(0)));
		u.push_back(uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 7, nearest_sampler, world.voxel_debug_texture(1)));
		u.push_back(uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 8, nearest_sampler, world.voxel_debug_texture(2)));
		u.push_back(uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 9, nearest_sampler, world.voxel_debug_texture(3)));
		u.push_back(uni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 10, nearest_sampler, world.voxel_debug_texture(4)));
	}
	const int64_t cl = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(cl, debug_pipelines[pass]);
	rd->compute_list_bind_uniform_set(cl, UniformSetCacheRD::get_cache(debug_shaders[pass], 0, u), 0);
	PackedInt32Array pc;
	pc.resize(4);
	pc.fill(0);
	pc[0] = p_mode;
	PackedByteArray pcb = pc.to_byte_array();
	rd->compute_list_set_push_constant(cl, pcb, pcb.size());
	const uint32_t n = pass == 1 ? uint32_t(probe_grid.x * probe_grid.y + max_adaptive) : uint32_t(size.x * size.y);
	rd->compute_list_dispatch(cl, (n + 63) / 64, 1, 1);
	rd->compute_list_end();
}

Ref<GoghotGIColorCapture> GoghotScreenProbeGI::create_color_capture() {
	Ref<GoghotGIColorCapture> c;
	c.instantiate();
	c->set_gi(Ref<GoghotScreenProbeGI>(this));
	return c;
}

Dictionary GoghotScreenProbeGI::get_stats() const {
	const GIWorldTracer::Stats &s = world.stats;
	Dictionary d;
	d["tier"] = active_tier;
	d["render_cpu_ms"] = render_cpu_ms;
	d["update_ms"] = s.update_ms;
	d["rebuild_ms"] = s.rebuild_ms;
	d["rebuilds"] = int64_t(s.rebuilds);
	d["frames"] = int64_t(s.frames);
	d["meshes"] = s.meshes;
	d["instances"] = s.instances;
	d["surfaces"] = s.surfaces;
	d["triangles"] = int64_t(s.triangles);
	d["deformed"] = s.deformed;
	d["deformed_triangles"] = int64_t(s.deformed_triangles);
	d["deformed_triangles_built"] = int64_t(s.deformed_triangles_built);
	d["last_rebuild_added"] = s.last_rebuild_added;
	d["last_rebuild_removed"] = s.last_rebuild_removed;
	d["last_rebuild_deformed_changed"] = s.last_rebuild_deformed_changed;
	d["last_rebuild_new_blas"] = s.last_rebuild_new_blas;
	return d;
}

void GoghotScreenProbeGI::set_scene_snapshot(std::shared_ptr<const GISceneSnapshot> p_scene) {
	std::lock_guard<std::mutex> lock(scene_mutex);
	pending_scene = std::move(p_scene);
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
	GOGHOT_BIND(Variant::PACKED_COLOR_ARRAY, sky_sh, PROPERTY_HINT_NONE, "")
	GOGHOT_BIND(Variant::BOOL, deformed_blas_cache, PROPERTY_HINT_NONE, "")
	GOGHOT_BIND(Variant::INT, deformed_triangle_budget, PROPERTY_HINT_RANGE, "10000,4000000,1000")
	GOGHOT_BIND(Variant::INT, debug_mode, PROPERTY_HINT_ENUM, "Off,No Temporal,No Spatial Filter")
	GOGHOT_BIND(Variant::BOOL, world_tracing, PROPERTY_HINT_NONE, "")
	GOGHOT_BIND(Variant::FLOAT, cache_min_cell, PROPERTY_HINT_RANGE, "0.005,1,0.005,suffix:m")
	GOGHOT_BIND(Variant::FLOAT, cache_cell_scale, PROPERTY_HINT_RANGE, "0.001,0.2,0.001")
	GOGHOT_BIND(Variant::INT, cache_max_frames, PROPERTY_HINT_RANGE, "1,256,1")
	GOGHOT_BIND(Variant::INT, cache_mature_frames, PROPERTY_HINT_RANGE, "1,256,1")
	GOGHOT_BIND(Variant::BOOL, fog_gi, PROPERTY_HINT_NONE, "")
	GOGHOT_BIND(Variant::INT, fog_resolution, PROPERTY_HINT_RANGE, "8,128,2")
	GOGHOT_BIND(Variant::INT, fog_rays_per_voxel, PROPERTY_HINT_RANGE, "1,64,1")
	GOGHOT_BIND(Variant::INT, fog_max_frames, PROPERTY_HINT_RANGE, "1,256,1")
	GOGHOT_BIND(Variant::BOOL, reflections, PROPERTY_HINT_NONE, "")
	GOGHOT_BIND(Variant::BOOL, deformed_geometry, PROPERTY_HINT_NONE, "")
	GOGHOT_BIND(Variant::INT, tracer, PROPERTY_HINT_ENUM, "Auto,A Hardware RT,B Voxels,C Screen Only")
	GOGHOT_BIND(Variant::INT, voxel_resolution, PROPERTY_HINT_RANGE, "32,256,4")
	GOGHOT_BIND(Variant::FLOAT, voxel_size, PROPERTY_HINT_RANGE, "0.01,2,0.005,suffix:m")
	GOGHOT_BIND(Variant::INT, voxel_bounce_rays, PROPERTY_HINT_RANGE, "1,16,1")
	GOGHOT_BIND(Variant::INT, voxel_cache_max_frames, PROPERTY_HINT_RANGE, "1,256,1")
	ClassDB::bind_method(D_METHOD("get_active_tier"), &GoghotScreenProbeGI::get_active_tier);
	ClassDB::bind_method(D_METHOD("get_stats"), &GoghotScreenProbeGI::get_stats);
	GOGHOT_BIND(Variant::FLOAT, reflection_max_roughness, PROPERTY_HINT_RANGE, "0,1,0.01")
	GOGHOT_BIND(Variant::INT, reflection_min_frames, PROPERTY_HINT_RANGE, "1,64,1")
	GOGHOT_BIND(Variant::INT, reflection_max_frames, PROPERTY_HINT_RANGE, "1,64,1")
	ClassDB::bind_method(D_METHOD("is_world_tracer_supported"), &GoghotScreenProbeGI::is_world_tracer_supported);
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
