#pragma once

#include <godot_cpp/classes/compositor_effect.hpp>
#include <godot_cpp/classes/render_data.hpp>
#include <godot_cpp/classes/render_scene_buffers_rd.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/templates/hash_map.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/projection.hpp>
#include <godot_cpp/variant/transform3d.hpp>

#include "gi_scene.h"
#include "world_tracer.h"

#include <memory>
#include <mutex>

namespace goghot {

class GoghotGIColorCapture;

// Screen-space probe final gather: PRE_OPAQUE CompositorEffect with provides_indirect_light
// (engine patch 1). Writes irradiance / PI into rbgi/ambient.
//   HZB -> uniform probes (1 per spacing^2 tile) + adaptive probes on depth/normal discontinuities
//   -> screen trace (cosine rays, previous frame color at hits) -> world tracer for rays that left
//   the screen (tier A hardware RT + radiance cache; needs a GoghotGIScene feeding the scene, sky
//   otherwise) -> gather -> 3x3 bilateral probe filter -> per-pixel interpolation -> temporal.
// Needs a GoghotGIColorCapture (POST_SKY) in the same compositor to supply previous-frame color;
// create it with create_color_capture(). Single view, Forward+ only.
class GoghotScreenProbeGI : public godot::CompositorEffect {
	GDCLASS(GoghotScreenProbeGI, godot::CompositorEffect);
	friend class GoghotGIColorCapture;

	enum Pass {
		PASS_HZB_COPY,
		PASS_HZB_DOWN,
		PASS_PROBE_PLACE,
		PASS_PROBE_ADAPTIVE,
		PASS_TRACE,
		PASS_GATHER,
		PASS_FILTER,
		PASS_RESOLVE,
		PASS_TEMPORAL,
		PASS_FOG_ACCUMULATE,
		PASS_REFL_TRACE,
		PASS_REFL_RESOLVE,
		PASS_COPY,
		PASS_MAX
	};

	// Settings (scene independent defaults).
	int probe_spacing = 16; // default changed from the 32 px spec by the user (2026-11-19, phase 3 data)
	int rays_per_probe = 32;
	float thickness = 0.05f;
	float max_distance = 100.0f;
	float plane_threshold = 0.05f;
	float normal_power = 8.0f;
	int temporal_max_frames = 48;
	int max_steps = 64;
	float response_sigma = 4.0f; // per-channel max deviation; 3 triggers on noise in colored light
	int response_min_frames = 8;
	bool jitter_probes = false; // measured: jitter adds flicker and does not improve accuracy
	godot::Color sky_color = godot::Color(0, 0, 0);
	float sky_energy = 1.0f;
	int debug_mode = 0;
	bool world_tracing = true;
	float cache_min_cell = 0.05f;
	float cache_cell_scale = 0.02f;
	int cache_max_frames = 16; // measured: 32 converges multi-bounce too slowly, 8 flickers more
	int cache_mature_frames = 8;
	bool fog_gi = true; // feed volumetric fog (engine patch 3) when the environment has it enabled
	int fog_resolution = 32;
	int fog_rays_per_voxel = 4;
	int fog_max_frames = 32;
	bool reflections = true;
	float reflection_max_roughness = 0.7f;
	int reflection_min_frames = 8; // measured: 4 flickers (2.1% of pixels), 12 lags
	int reflection_max_frames = 32;

	// GPU state.
	godot::RenderingDevice *rd = nullptr;
	godot::RID shaders[PASS_MAX];
	godot::RID pipelines[PASS_MAX];
	godot::RID linear_sampler;
	godot::RID nearest_sampler;
	godot::RID frame_ubo;
	bool ready = false;

	godot::Vector2i size;
	int hzb_levels = 0;
	godot::RID hzb;
	godot::Vector<godot::RID> hzb_mips;
	godot::RID probes, counter, adaptive_map, rays, radiance, filtered;
	godot::RID resolved;
	godot::RID history[2], aux[2], moments[2];
	godot::RID refl_rays, refl_history[2];
	godot::Vector2i refl_size;
	godot::RID prev_color;
	// Fog GI volume (world aligned, follows the camera in whole voxels).
	godot::RID fog_rays, fog_volume, fog_history;
	int fog_res_alloc = 0;
	godot::Vector3i fog_center_cell;
	bool fog_has_history = false;
	godot::RID fog_env_bound;
	float fog_extent_bound = 0.0f;
	void _free_fog();
	godot::Vector2i probe_grid, adaptive_grid;
	int max_adaptive = 0;

	int frame = 0;
	int ping = 0;
	bool has_history = false;
	bool prev_color_valid = false;
	godot::Projection prev_proj;
	godot::Transform3D prev_cam;

	// World tracer (render thread) and the scene snapshot handed over from the main thread.
	GIWorldTracer world;
	std::mutex scene_mutex;
	std::shared_ptr<const GISceneSnapshot> pending_scene;
	std::shared_ptr<const GISceneSnapshot> scene;

	void _create_pipelines();
	void _free_all();
	void _free_sized();
	void _ensure_size(const godot::Vector2i &p_size);
	godot::RID _tex(godot::RenderingDevice::DataFormat p_format, const godot::Vector2i &p_size, int p_mips, uint32_t p_usage);
	void _dispatch(int64_t p_list, Pass p_pass, const godot::TypedArray<godot::RDUniform> &p_uniforms, uint32_t p_x, uint32_t p_y, uint32_t p_z = 1);
	void _capture_color(godot::RenderSceneBuffersRD *p_rb);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void _render_callback(int32_t p_effect_callback_type, godot::RenderData *p_render_data) override;
	godot::Ref<GoghotGIColorCapture> create_color_capture();
	void set_scene_snapshot(std::shared_ptr<const GISceneSnapshot> p_scene); // any thread
	bool is_world_tracer_supported() const { return world.is_ready(); }
	void reset_history();

#define GOGHOT_PROP(m_type, m_name)            \
	void set_##m_name(m_type p_v) { m_name = p_v; } \
	m_type get_##m_name() const { return m_name; }
	GOGHOT_PROP(int, probe_spacing)
	GOGHOT_PROP(int, rays_per_probe)
	GOGHOT_PROP(float, thickness)
	GOGHOT_PROP(float, max_distance)
	GOGHOT_PROP(float, plane_threshold)
	GOGHOT_PROP(float, normal_power)
	GOGHOT_PROP(int, temporal_max_frames)
	GOGHOT_PROP(int, max_steps)
	GOGHOT_PROP(float, response_sigma)
	GOGHOT_PROP(int, response_min_frames)
	GOGHOT_PROP(bool, jitter_probes)
	GOGHOT_PROP(godot::Color, sky_color)
	GOGHOT_PROP(float, sky_energy)
	GOGHOT_PROP(int, debug_mode)
	GOGHOT_PROP(bool, world_tracing)
	GOGHOT_PROP(float, cache_min_cell)
	GOGHOT_PROP(float, cache_cell_scale)
	GOGHOT_PROP(int, cache_max_frames)
	GOGHOT_PROP(int, cache_mature_frames)
	GOGHOT_PROP(bool, fog_gi)
	GOGHOT_PROP(int, fog_resolution)
	GOGHOT_PROP(int, fog_rays_per_voxel)
	GOGHOT_PROP(int, fog_max_frames)
	GOGHOT_PROP(bool, reflections)
	GOGHOT_PROP(float, reflection_max_roughness)
	GOGHOT_PROP(int, reflection_min_frames)
	GOGHOT_PROP(int, reflection_max_frames)
#undef GOGHOT_PROP

	GoghotScreenProbeGI();
	~GoghotScreenProbeGI();
};

// POST_SKY helper: stores the lit HDR color (no transparents, before exposure) for the next frame.
class GoghotGIColorCapture : public godot::CompositorEffect {
	GDCLASS(GoghotGIColorCapture, godot::CompositorEffect);

	godot::Ref<GoghotScreenProbeGI> gi;

protected:
	static void _bind_methods();

public:
	void set_gi(const godot::Ref<GoghotScreenProbeGI> &p_gi) { gi = p_gi; }
	godot::Ref<GoghotScreenProbeGI> get_gi() const { return gi; }
	void _render_callback(int32_t p_effect_callback_type, godot::RenderData *p_render_data) override;

	GoghotGIColorCapture();
};

} // namespace goghot
