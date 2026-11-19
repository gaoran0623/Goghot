#include "gi_node.h"

#include "../exposure/histogram_exposure.h"
#include "screen_probe_gi.h"

#include <godot_cpp/classes/camera3d.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/panorama_sky_material.hpp>
#include <godot_cpp/classes/physical_sky_material.hpp>
#include <godot_cpp/classes/procedural_sky_material.hpp>
#include <godot_cpp/classes/render_scene_buffers_rd.hpp>
#include <godot_cpp/classes/scene_tree.hpp>
#include <godot_cpp/classes/sky.hpp>
#include <godot_cpp/classes/viewport.hpp>
#include <godot_cpp/classes/world3d.hpp>
#include <godot_cpp/classes/world_environment.hpp>
#include <godot_cpp/core/class_db.hpp>

using namespace godot;

namespace goghot {

// ---------------------------------------------------------------------------- GoghotGIDebug

void GoghotGIDebug::_render_callback(int32_t p_effect_callback_type, RenderData *p_render_data) {
	if (p_effect_callback_type != EFFECT_CALLBACK_TYPE_POST_TRANSPARENT || gi.is_null() || mode == 0 || p_render_data == nullptr) {
		return;
	}
	RenderSceneBuffersRD *rb = Object::cast_to<RenderSceneBuffersRD>(p_render_data->get_render_scene_buffers().ptr());
	if (rb != nullptr) {
		gi->_render_debug(rb, mode);
	}
}

GoghotGIDebug::GoghotGIDebug() {
	set_effect_callback_type(EFFECT_CALLBACK_TYPE_POST_TRANSPARENT);
}

void GoghotGIDebug::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &GoghotGIDebug::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &GoghotGIDebug::get_mode);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Off,GI,Reflections,Probes,Voxels"), "set_mode", "get_mode");
}

// ---------------------------------------------------------------------------- GoghotGI

GoghotGI::GoghotGI() {
	Ref<GoghotScreenProbeGI> effect;
	effect.instantiate();
	set_gi(effect);
	capture = effect->create_color_capture();
	exposure.instantiate();
	debug.instantiate();
	debug->set_gi(effect);
}

void GoghotGI::_apply_quality() {
	if (gi.is_null() || applied_quality == quality) {
		return;
	}
	applied_quality = quality;
	// Presets only touch the parameters they name; scene independent, like all defaults.
	switch (quality) {
		case QUALITY_LOW:
			gi->set_tracer(3); // C
			gi->set_probe_spacing(32);
			gi->set_rays_per_probe(16);
			gi->set_reflections(false);
			gi->set_fog_gi(false);
			break;
		case QUALITY_MEDIUM:
			gi->set_tracer(tracer);
			gi->set_probe_spacing(32);
			gi->set_rays_per_probe(32);
			gi->set_reflections(true);
			gi->set_fog_gi(true);
			gi->set_voxel_resolution(96);
			break;
		case QUALITY_HIGH:
			gi->set_tracer(tracer);
			gi->set_probe_spacing(16);
			gi->set_rays_per_probe(32);
			gi->set_reflections(true);
			gi->set_fog_gi(true);
			gi->set_voxel_resolution(128);
			gi->set_voxel_size(0.125f);
			break;
		case QUALITY_ULTRA:
			gi->set_tracer(tracer);
			gi->set_probe_spacing(16);
			gi->set_rays_per_probe(32);
			gi->set_reflections(true);
			gi->set_fog_gi(true);
			gi->set_voxel_resolution(160);
			gi->set_voxel_size(0.1f);
			gi->set_fog_resolution(48);
			break;
	}
}

void GoghotGI::_sync_effects() {
	if (attached.is_null()) {
		return;
	}
	TypedArray<CompositorEffect> effects = attached->get_compositor_effects();
	TypedArray<CompositorEffect> out;
	for (int i = 0; i < effects.size(); i++) { // keep the user's own effects, drop ours
		Ref<CompositorEffect> e = effects[i];
		if (e != gi && e != capture && e != exposure && e != debug) {
			out.push_back(e);
		}
	}
	if (enabled) {
		out.push_front(capture);
		out.push_front(gi);
		if (auto_exposure) {
			out.push_back(exposure);
		}
		if (debug_view != DEBUG_OFF) {
			out.push_back(debug);
		}
	}
	attached->set_compositor_effects(out);
}

void GoghotGI::_attach() {
	Node *root = _scene_root();
	if (root == nullptr) {
		return;
	}
	Node *holder = nullptr;
	Ref<Compositor> comp;
	TypedArray<Node> wes = root->find_children("*", "WorldEnvironment", true, false);
	if (Object::cast_to<WorldEnvironment>(root)) {
		wes.push_front(root);
	}
	if (!wes.is_empty()) {
		WorldEnvironment *we = Object::cast_to<WorldEnvironment>(Object::cast_to<Node>(wes[0]));
		comp = we->get_compositor();
		if (comp.is_null()) {
			comp.instantiate();
			we->set_compositor(comp);
		}
		holder = we;
	} else if (Viewport *vp = get_viewport()) {
		if (Camera3D *cam = vp->get_camera_3d()) {
			comp = cam->get_compositor();
			if (comp.is_null()) {
				comp.instantiate();
				cam->set_compositor(comp);
			}
			holder = cam;
		}
	}
	if (holder == nullptr) {
		return; // nothing to attach to yet (no camera); retried every frame
	}
	if (attached.is_valid() && attached != comp) {
		_detach();
	}
	attached = comp;
	attached_owner = ObjectID(holder->get_instance_id());
	_sync_effects();
}

void GoghotGI::_detach() {
	if (attached.is_null()) {
		return;
	}
	const bool was = enabled;
	enabled = false;
	_sync_effects();
	enabled = was;
	attached.unref();
	attached_owner = ObjectID();
}

Color GoghotGI::_sky_radiance(const Ref<Environment> &p_env) const {
	if (p_env.is_null()) {
		return Color(0, 0, 0);
	}
	// Uniform sky radiance for rays that miss (directional sky sampling is not implemented yet).
	if (p_env->get_ambient_source() == Environment::AMBIENT_SOURCE_COLOR) {
		return p_env->get_ambient_light_color().srgb_to_linear() * p_env->get_ambient_light_energy();
	}
	switch (p_env->get_background()) {
		case Environment::BG_COLOR:
			return p_env->get_bg_color().srgb_to_linear() * p_env->get_bg_energy_multiplier();
		case Environment::BG_SKY: {
			Ref<Sky> sky = p_env->get_sky();
			if (sky.is_valid()) {
				Ref<ProceduralSkyMaterial> psm = sky->get_material();
				if (psm.is_valid()) { // average of the upper hemisphere colors, roughly
					Color c = (psm->get_sky_top_color().srgb_to_linear() + psm->get_sky_horizon_color().srgb_to_linear() * 2.0f) / 3.0f;
					return c * psm->get_sky_energy_multiplier() * p_env->get_bg_energy_multiplier();
				}
				Ref<PhysicalSkyMaterial> phy = sky->get_material();
				if (phy.is_valid()) {
					return Color(0.35f, 0.5f, 0.8f) * phy->get_energy_multiplier() * p_env->get_bg_energy_multiplier();
				}
			}
			return Color(0.3f, 0.35f, 0.4f) * p_env->get_bg_energy_multiplier();
		}
		default:
			return Color(0, 0, 0);
	}
}

void GoghotGI::_process(double p_delta) {
	const bool editor = Engine::get_singleton()->is_editor_hint();
	if (editor && !editor_preview) {
		if (attached.is_valid()) {
			_detach();
		}
		return;
	}
	Object *holder = attached_owner.is_valid() ? ObjectDB::get_instance(uint64_t(attached_owner)) : nullptr;
	if (attached.is_null() || holder == nullptr) {
		_attach();
	}
	_apply_quality();
	if (sky_from_environment && gi.is_valid()) {
		Ref<Environment> env;
		if (WorldEnvironment *we = Object::cast_to<WorldEnvironment>(holder)) {
			env = we->get_environment();
		} else if (Viewport *vp = get_viewport()) {
			if (Camera3D *cam = vp->get_camera_3d()) {
				env = cam->get_environment();
			}
			if (env.is_null() && vp->find_world_3d().is_valid()) {
				env = vp->find_world_3d()->get_environment();
			}
		}
		gi->set_sky_color(_sky_radiance(env));
	}
	GoghotGIScene::_process(p_delta);
}

void GoghotGI::_validate_property(PropertyInfo &p_property) const {
	// The node owns its effect (created in the constructor, reachable through get_effect()): it must
	// not be stored in the scene or shown as an editable default of the inherited `gi` property.
	if (p_property.name == StringName("gi")) {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

void GoghotGI::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY:
			set_process(true);
			break;
		case NOTIFICATION_EXIT_TREE:
			_detach();
			break;
	}
}

void GoghotGI::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	_sync_effects();
}

void GoghotGI::set_quality(int p_quality) {
	quality = CLAMP(p_quality, 0, 3);
	_apply_quality();
}

void GoghotGI::set_tracer(int p_tracer) {
	tracer = CLAMP(p_tracer, 0, 3);
	applied_quality = -1; // presets carry the tracer
	_apply_quality();
}

void GoghotGI::set_auto_exposure(bool p_enable) {
	auto_exposure = p_enable;
	_sync_effects();
}

void GoghotGI::set_debug_view(int p_view) {
	debug_view = CLAMP(p_view, 0, 4);
	debug->set_mode(debug_view);
	_sync_effects();
}

void GoghotGI::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &GoghotGI::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &GoghotGI::is_enabled);
	ClassDB::bind_method(D_METHOD("set_quality", "quality"), &GoghotGI::set_quality);
	ClassDB::bind_method(D_METHOD("get_quality"), &GoghotGI::get_quality);
	ClassDB::bind_method(D_METHOD("set_tracer", "tracer"), &GoghotGI::set_tracer);
	ClassDB::bind_method(D_METHOD("get_tracer"), &GoghotGI::get_tracer);
	ClassDB::bind_method(D_METHOD("set_auto_exposure", "enable"), &GoghotGI::set_auto_exposure);
	ClassDB::bind_method(D_METHOD("get_auto_exposure"), &GoghotGI::get_auto_exposure);
	ClassDB::bind_method(D_METHOD("set_debug_view", "view"), &GoghotGI::set_debug_view);
	ClassDB::bind_method(D_METHOD("get_debug_view"), &GoghotGI::get_debug_view);
	ClassDB::bind_method(D_METHOD("set_sky_from_environment", "enable"), &GoghotGI::set_sky_from_environment);
	ClassDB::bind_method(D_METHOD("get_sky_from_environment"), &GoghotGI::get_sky_from_environment);
	ClassDB::bind_method(D_METHOD("get_effect"), &GoghotGI::get_effect);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "quality", PROPERTY_HINT_ENUM, "Low,Medium,High,Ultra"), "set_quality", "get_quality");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tracer", PROPERTY_HINT_ENUM, "Auto,A Hardware RT,B Voxels,C Screen Only"), "set_tracer", "get_tracer");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_exposure"), "set_auto_exposure", "get_auto_exposure");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "debug_view", PROPERTY_HINT_ENUM, "Off,GI,Reflections,Probes,Voxels"), "set_debug_view", "get_debug_view");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sky_from_environment"), "set_sky_from_environment", "get_sky_from_environment");
}

} // namespace goghot
