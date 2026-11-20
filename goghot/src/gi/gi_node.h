#pragma once

#include "gi_scene.h"
#include "screen_probe_gi.h"

#include <godot_cpp/classes/compositor.hpp>
#include <godot_cpp/classes/compositor_effect.hpp>
#include <godot_cpp/classes/environment.hpp>

namespace goghot {

class GoghotHistogramExposure;

// POST_TRANSPARENT helper of GoghotGI: debug views drawn over the HDR color.
class GoghotGIDebug : public godot::CompositorEffect {
	GDCLASS(GoghotGIDebug, godot::CompositorEffect);

	godot::Ref<GoghotScreenProbeGI> gi;
	int mode = 0;

protected:
	static void _bind_methods();

public:
	void set_gi(const godot::Ref<GoghotScreenProbeGI> &p_gi) { gi = p_gi; }
	void set_mode(int p_mode) { mode = p_mode; }
	int get_mode() const { return mode; }
	void _render_callback(int32_t p_effect_callback_type, godot::RenderData *p_render_data) override;

	GoghotGIDebug();
};

// One node to enable Goghot GI. Add it anywhere in the scene:
//   - creates the GI effects (GoghotScreenProbeGI + color capture, optional histogram exposure and
//     debug views) and adds them to the Compositor of the scene's WorldEnvironment (created if
//     missing), or of the current Camera3D when there is no WorldEnvironment; removes them on exit;
//   - collects the scene for the world tracer (GoghotGIScene) and derives the sky radiance used for
//     rays that miss from the active Environment;
//   - quality presets set the GI parameters (still editable on the `effect` afterwards).
class GoghotGI : public GoghotGIScene {
	GDCLASS(GoghotGI, GoghotGIScene);

public:
	enum Quality {
		QUALITY_LOW, // tier C (screen only), sparse probes, no world reflections / fog GI
		QUALITY_MEDIUM, // auto tier, 32 px probes
		QUALITY_HIGH, // auto tier, 16 px probes (default)
		QUALITY_ULTRA, // auto tier, 16 px probes, finer tier B voxels, finer fog GI volume
	};
	enum DebugView {
		DEBUG_OFF,
		DEBUG_GI,
		DEBUG_REFLECTIONS,
		DEBUG_PROBES,
		DEBUG_VOXELS,
	};

private:
	bool enabled = true;
	int quality = QUALITY_HIGH;
	int tracer = 0; // GoghotScreenProbeGI::TRACER_AUTO
	bool auto_exposure = false;
	int debug_view = DEBUG_OFF;
	bool sky_from_environment = true;
	bool print_stats = false;
	uint64_t stats_next_usec = 0;

	godot::Ref<GoghotGIColorCapture> capture;
	godot::Ref<GoghotHistogramExposure> exposure;
	godot::Ref<GoghotGIDebug> debug;
	godot::Ref<godot::Compositor> attached; // compositor our effects are in
	godot::ObjectID attached_owner; // WorldEnvironment or Camera3D holding it
	int applied_quality = -1;

	void _attach();
	void _detach();
	void _sync_effects();
	void _apply_quality();
	// Panorama SH (unscaled) cached per panorama texture: the image readback and resize are slow.
	mutable uint64_t pano_sh_id = 0;
	mutable godot::PackedColorArray pano_sh;
	godot::Color _sky_radiance(const godot::Ref<godot::Environment> &p_env, godot::PackedColorArray &r_sh) const;

protected:
	static void _bind_methods();
	void _notification(int p_what);
	void _validate_property(godot::PropertyInfo &p_property) const;

public:
	void _process(double p_delta) override;
	void _process_gi(double p_delta);
	double process_ms = 0.0; // CPU time of the last _process (main thread)

	void set_enabled(bool p_enabled);
	bool is_enabled() const { return enabled; }
	void set_quality(int p_quality);
	int get_quality() const { return quality; }
	void set_tracer(int p_tracer);
	int get_tracer() const { return tracer; }
	void set_auto_exposure(bool p_enable);
	bool get_auto_exposure() const { return auto_exposure; }
	void set_debug_view(int p_view);
	int get_debug_view() const { return debug_view; }
	void set_sky_from_environment(bool p_enable) { sky_from_environment = p_enable; }
	bool get_sky_from_environment() const { return sky_from_environment; }
	godot::Ref<GoghotScreenProbeGI> get_effect() const { return gi; }
	void set_print_stats(bool p_enable) { print_stats = p_enable; }
	void set_deformed_blas_cache(bool p_enable) { gi->set_deformed_blas_cache(p_enable); }
	bool get_deformed_blas_cache() const { return gi->get_deformed_blas_cache(); }
	bool get_print_stats() const { return print_stats; }

	GoghotGI();
};

} // namespace goghot

