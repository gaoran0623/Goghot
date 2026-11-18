#pragma once

#include <godot_cpp/classes/compositor_effect.hpp>
#include <godot_cpp/classes/render_data.hpp>
#include <godot_cpp/classes/render_scene_buffers_rd.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/templates/hash_map.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/projection.hpp>
#include <godot_cpp/variant/transform3d.hpp>

namespace goghot {

class GoghotGIColorCapture;

// Screen-space probe final gather (phase 2): PRE_OPAQUE CompositorEffect with
// provides_indirect_light (engine patch 1). Writes irradiance / PI into rbgi/ambient.
//   HZB -> uniform probes (1 per spacing^2 tile, jittered) + adaptive probes on depth/normal
//   discontinuities -> screen trace (rays cosine-distributed, previous frame color at hits, sky on
//   miss) -> 3x3 bilateral probe filter -> per-pixel interpolation -> temporal accumulation.
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
		PASS_FILTER,
		PASS_RESOLVE,
		PASS_TEMPORAL,
		PASS_COPY,
		PASS_MAX
	};

	// Settings (scene independent defaults).
	int probe_spacing = 32;
	int rays_per_probe = 32;
	float thickness = 0.05f;
	float max_distance = 100.0f;
	float plane_threshold = 0.05f;
	float normal_power = 8.0f;
	int temporal_max_frames = 48;
	int max_steps = 64;
	float response_sigma = 3.0f;
	int response_min_frames = 8;
	bool jitter_probes = false; // measured: jitter adds flicker and does not improve accuracy
	godot::Color sky_color = godot::Color(0, 0, 0);
	float sky_energy = 1.0f;
	int debug_mode = 0;

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
	godot::RID probes, counter, adaptive_map, radiance, filtered;
	godot::RID resolved;
	godot::RID history[2], aux[2], moments[2];
	godot::RID prev_color;
	godot::Vector2i probe_grid, adaptive_grid;
	int max_adaptive = 0;

	int frame = 0;
	int ping = 0;
	bool has_history = false;
	bool prev_color_valid = false;
	godot::Projection prev_proj;
	godot::Transform3D prev_cam;

	void _create_pipelines();
	void _free_all();
	void _free_sized();
	void _ensure_size(const godot::Vector2i &p_size);
	godot::RID _tex(godot::RenderingDevice::DataFormat p_format, const godot::Vector2i &p_size, int p_mips, uint32_t p_usage);
	void _dispatch(int64_t p_list, Pass p_pass, const godot::TypedArray<godot::RDUniform> &p_uniforms, uint32_t p_x, uint32_t p_y);
	void _capture_color(godot::RenderSceneBuffersRD *p_rb);

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void _render_callback(int32_t p_effect_callback_type, godot::RenderData *p_render_data) override;
	godot::Ref<GoghotGIColorCapture> create_color_capture();
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
