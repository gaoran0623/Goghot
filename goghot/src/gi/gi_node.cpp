#include "gi_node.h"

#include "../exposure/histogram_exposure.h"
#include "screen_probe_gi.h"

#include <godot_cpp/classes/camera3d.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/core/math.hpp>
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

static constexpr double Math_PI_CONST = 3.14159265358979323846;

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

namespace {

// Real L2 SH basis (same order and constants as sky_radiance() in gi_common.glsl).
void sh_basis(const Vector3 &d, float r_y[9]) {
	r_y[0] = 0.282095f;
	r_y[1] = 0.488603f * d.y;
	r_y[2] = 0.488603f * d.z;
	r_y[3] = 0.488603f * d.x;
	r_y[4] = 1.092548f * d.x * d.y;
	r_y[5] = 1.092548f * d.y * d.z;
	r_y[6] = 0.315392f * (3.0f * d.z * d.z - 1.0f);
	r_y[7] = 1.092548f * d.x * d.z;
	r_y[8] = 0.546274f * (d.x * d.x - d.y * d.y);
}

// Projects radiance(direction) over an equirectangular grid (Godot's panorama mapping:
// u = atan2(x, -z) / 2pi, v = acos(y) / pi) into 9 SH coefficients.
template <typename F>
PackedColorArray sh_project(int p_w, int p_h, F p_radiance) {
	double acc[9][3] = {};
	for (int j = 0; j < p_h; j++) {
		const float theta = (float(j) + 0.5f) / float(p_h) * float(Math_PI_CONST);
		const float dw = (2.0f * float(Math_PI_CONST) / p_w) * (float(Math_PI_CONST) / p_h) * Math::sin(theta);
		for (int i = 0; i < p_w; i++) {
			const float phi = (float(i) + 0.5f) / float(p_w) * 2.0f * float(Math_PI_CONST);
			const Vector3 d(Math::sin(theta) * Math::sin(phi), Math::cos(theta), -Math::sin(theta) * Math::cos(phi));
			const Color l = p_radiance(d, i, j);
			float y[9];
			sh_basis(d, y);
			for (int k = 0; k < 9; k++) {
				acc[k][0] += l.r * y[k] * dw;
				acc[k][1] += l.g * y[k] * dw;
				acc[k][2] += l.b * y[k] * dw;
			}
		}
	}
	PackedColorArray out;
	out.resize(9);
	for (int k = 0; k < 9; k++) {
		out.set(k, Color(acc[k][0], acc[k][1], acc[k][2]));
	}
	return out;
}

} // namespace

// Sky radiance for rays that miss, in the units the scene is shaded in. Returns the average and, for
// Panorama and Procedural skies, their L2 SH in r_sh (else r_sh is emptied: uniform sky).
// Units (as Godot draws them): sky / background * bg_energy (* bg_intensity with physical units) *
// camera exposure normalization; an ambient light color is used as is (Godot applies it unscaled).
Color GoghotGI::_sky_radiance(const Ref<Environment> &p_env, PackedColorArray &r_sh) const {
	r_sh.clear();
	if (p_env.is_null()) {
		return Color(0, 0, 0);
	}
	if (p_env->get_ambient_source() == Environment::AMBIENT_SOURCE_COLOR) {
		return p_env->get_ambient_light_color().srgb_to_linear() * p_env->get_ambient_light_energy();
	}
	const float scale = p_env->get_bg_energy_multiplier() * (physical_units ? p_env->get_bg_intensity() : 1.0f) * exposure_scale;
	switch (p_env->get_background()) {
		case Environment::BG_COLOR:
			return p_env->get_bg_color().srgb_to_linear() * scale;
		case Environment::BG_SKY: {
			Ref<Sky> sky = p_env->get_sky();
			if (sky.is_valid()) {
				Ref<ProceduralSkyMaterial> psm = sky->get_material();
				if (psm.is_valid()) {
					// Godot's procedural sky shader without the sun disks (the sun is traced as a light).
					const Color top = psm->get_sky_top_color().srgb_to_linear() * psm->get_sky_energy_multiplier();
					const Color hor = psm->get_sky_horizon_color().srgb_to_linear() * psm->get_sky_energy_multiplier();
					const Color gbot = psm->get_ground_bottom_color().srgb_to_linear() * psm->get_ground_energy_multiplier();
					const Color ghor = psm->get_ground_horizon_color().srgb_to_linear() * psm->get_ground_energy_multiplier();
					const float inv_sky = 0.6f / MAX(psm->get_sky_curve(), 1e-4f), inv_ground = 0.6f / MAX(psm->get_ground_curve(), 1e-4f);
					const float k = scale * psm->get_energy_multiplier();
					r_sh = sh_project(64, 32, [&](const Vector3 &d, int, int) {
						const float v = CLAMP(d.y, -1.0f, 1.0f);
						const Color sk = top.lerp(hor, CLAMP(Math::pow(1.0f - v, inv_sky), 0.0f, 1.0f));
						const Color gr = gbot.lerp(ghor, CLAMP(Math::pow(1.0f + v, inv_ground), 0.0f, 1.0f));
						return (d.y >= 0.0f ? sk : gr) * k;
					});
					return Color(r_sh[0].r, r_sh[0].g, r_sh[0].b) * 0.282095f;
				}
				Ref<PanoramaSkyMaterial> pano = sky->get_material();
				if (pano.is_valid() && pano->get_panorama().is_valid()) {
					const uint64_t id = pano->get_panorama()->get_rid().get_id();
					if (id == pano_sh_id && pano_sh.size() == 9) {
						const float k = scale * pano->get_energy_multiplier();
						for (int i = 0; i < 9; i++) {
							r_sh.push_back(pano_sh[i] * k);
						}
						return Color(r_sh[0].r, r_sh[0].g, r_sh[0].b) * 0.282095f;
					}
					Ref<Image> img = pano->get_panorama()->get_image();
					if (img.is_valid() && !img->is_empty()) {
						img = img->duplicate();
						if (img->is_compressed()) {
							img->decompress();
						}
						const bool hdr = img->get_format() >= Image::FORMAT_RF && img->get_format() <= Image::FORMAT_RGBE9995;
						img->convert(Image::FORMAT_RGBAF);
						img->resize(128, 64, Image::INTERPOLATE_BILINEAR);
						pano_sh = sh_project(128, 64, [&](const Vector3 &, int i, int j) {
							Color c = img->get_pixel(i, j);
							return hdr ? c : c.srgb_to_linear();
						});
						pano_sh_id = id;
						const float k = scale * pano->get_energy_multiplier();
						for (int i = 0; i < 9; i++) {
							r_sh.push_back(pano_sh[i] * k);
						}
						return Color(r_sh[0].r, r_sh[0].g, r_sh[0].b) * 0.282095f;
					}
				}
				Ref<PhysicalSkyMaterial> phy = sky->get_material();
				if (phy.is_valid()) { // not evaluated: rough uniform blue sky
					return Color(0.35f, 0.5f, 0.8f) * phy->get_energy_multiplier() * scale;
				}
			}
			return Color(0.3f, 0.35f, 0.4f) * scale; // custom sky shader: unknown, uniform estimate
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
		_update_units();
		PackedColorArray sh;
		gi->set_sky_color(_sky_radiance(env, sh));
		gi->set_sky_energy(1.0f);
		gi->set_sky_sh(sh);
	}
	GoghotGIScene::_process(p_delta);
	if (print_stats && gi.is_valid()) {
		const uint64_t now = Time::get_singleton()->get_ticks_usec();
		if (now >= stats_next_usec) {
			stats_next_usec = now + 2000000;
			const Dictionary w = gi->get_stats();
			const Dictionary c = get_collect_stats();
			UtilityFunctions::print(vformat("GoghotGI: tier %s | collect %.2f ms (%d nodes, %d new meshes, cache %d, %d culled) | tracer update %.2f ms, %d rebuilds (last %.1f ms) | %d instances, %d meshes, %d surfaces, %d tris | %d deformed (%d tris), %d tris rebuilt this frame",
					w["tier"], c["collect_ms"], c["nodes_visited"], c["new_meshes"], c["mesh_cache"], c["culled"], w["update_ms"], w["rebuilds"], w["rebuild_ms"],
					w["instances"], w["meshes"], w["surfaces"], w["triangles"], w["deformed"], w["deformed_triangles"], w["deformed_triangles_built"]));
		}
	}
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
	ClassDB::bind_method(D_METHOD("set_print_stats", "enable"), &GoghotGI::set_print_stats);
	ClassDB::bind_method(D_METHOD("get_print_stats"), &GoghotGI::get_print_stats);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "print_stats"), "set_print_stats", "get_print_stats");
}

} // namespace goghot
