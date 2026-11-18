#pragma once

#include <godot_cpp/classes/compositor_effect.hpp>
#include <godot_cpp/classes/render_data.hpp>
#include <godot_cpp/classes/rendering_device.hpp>

namespace goghot {

// Histogram auto exposure as a POST_TRANSPARENT CompositorEffect, GPU only (no readback, no frame latency):
//   1. histogram of log2 luminance (one pixel per METER_STRIDE^2 block, center weighted, black ignored)
//   2. percentile-trimmed mean -> target EV, temporal adaptation, state kept on the GPU
//   3. scale the HDR color buffer in place
// Defaults are scene independent on purpose (no per-scene tuning).
// Behaviour must match goghot/testbench/proto/histogram_exposure.gd (the bench checks they agree).
class GoghotHistogramExposure : public godot::CompositorEffect {
	GDCLASS(GoghotHistogramExposure, godot::CompositorEffect);

public:
	static constexpr int BINS = 128;
	static constexpr float MIN_LOG2 = -14.0f;
	static constexpr float MAX_LOG2 = 6.0f;
	static constexpr int METER_STRIDE = 4;

private:
	float low_percentile = 0.6f;
	float high_percentile = 0.95f;
	float key = 0.18f;
	float compensation_ev = 0.0f;
	float min_ev = -12.0f;
	float max_ev = 12.0f;
	float speed_up = 1.5f;
	float speed_down = 3.0f;
	float center_weight = 3.0f;
	float center_sigma = 0.25f;
	float fixed_dt = 0.0f;
	bool snap = false;

	godot::RenderingDevice *rd = nullptr;
	godot::RID shaders[3];
	godot::RID pipelines[3];
	godot::RID hist_buffer;
	godot::RID state_buffer;
	uint64_t last_usec = 0;
	bool ready = false;

	void _create_resources();
	void _free_resources();
	godot::PackedByteArray _push_constants(const godot::Vector2i &p_size, float p_dt) const;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void _render_callback(int32_t p_effect_callback_type, godot::RenderData *p_render_data) override;

#define GOGHOT_PROP(m_type, m_name)            \
	void set_##m_name(m_type p_v) { m_name = p_v; } \
	m_type get_##m_name() const { return m_name; }
	GOGHOT_PROP(float, low_percentile)
	GOGHOT_PROP(float, high_percentile)
	GOGHOT_PROP(float, key)
	GOGHOT_PROP(float, compensation_ev)
	GOGHOT_PROP(float, min_ev)
	GOGHOT_PROP(float, max_ev)
	GOGHOT_PROP(float, speed_up)
	GOGHOT_PROP(float, speed_down)
	GOGHOT_PROP(float, center_weight)
	GOGHOT_PROP(float, center_sigma)
	GOGHOT_PROP(float, fixed_dt)
	GOGHOT_PROP(bool, snap)
#undef GOGHOT_PROP

	GoghotHistogramExposure();
	~GoghotHistogramExposure();
};

} // namespace goghot
