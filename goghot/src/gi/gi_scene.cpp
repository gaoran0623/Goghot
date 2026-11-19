#include "gi_scene.h"

#include "screen_probe_gi.h"

#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/base_material3d.hpp>
#include <godot_cpp/classes/skin.hpp>
#include <godot_cpp/classes/camera3d.hpp>
#include <godot_cpp/classes/camera_attributes.hpp>
#include <godot_cpp/classes/camera_attributes_physical.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/math.hpp>
#include <godot_cpp/classes/environment.hpp>
#include <godot_cpp/classes/viewport.hpp>
#include <godot_cpp/classes/world3d.hpp>
#include <godot_cpp/classes/directional_light3d.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/geometry_instance3d.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/classes/multi_mesh.hpp>
#include <godot_cpp/classes/multi_mesh_instance3d.hpp>
#include <godot_cpp/classes/csg_shape3d.hpp>
#include <godot_cpp/classes/grid_map.hpp>
#include <godot_cpp/classes/spot_light3d.hpp>
#include <godot_cpp/classes/omni_light3d.hpp>
#include <godot_cpp/classes/scene_tree.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/window.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>

using namespace godot;

static constexpr double Math_PI_CONST = 3.14159265358979323846;

namespace goghot {

std::shared_ptr<const GIMeshCPU> GoghotGIScene::_get_mesh(const Ref<Resource> &p_res) {
	Ref<Mesh> mesh = p_res;
	if (mesh.is_null()) {
		return nullptr;
	}
	const uint64_t id = mesh->get_rid().get_id();
	auto it = mesh_cache.find(id);
	if (it != mesh_cache.end()) {
		return it->second;
	}
	auto m = std::make_shared<GIMeshCPU>();
	m->id = id;
	for (int s = 0; s < mesh->get_surface_count(); s++) {
		Array arrays = mesh->surface_get_arrays(s);
		PackedVector3Array pos = arrays[Mesh::ARRAY_VERTEX];
		PackedVector3Array nrm = arrays[Mesh::ARRAY_NORMAL];
		PackedInt32Array idx = arrays[Mesh::ARRAY_INDEX];
		PackedVector2Array uv = arrays[Mesh::ARRAY_TEX_UV];
		GIMeshCPU::Surface surf;
		surf.source_surface = s;
		surf.has_normal = !nrm.is_empty();
		surf.has_tangent = !PackedFloat32Array(arrays[Mesh::ARRAY_TANGENT]).is_empty();
		surf.vertex_base = uint32_t(m->positions.size() / 3);
		surf.vertex_count = uint32_t(pos.size());
		surf.index_base = uint32_t(m->indices.size());
		for (int i = 0; i < pos.size(); i++) {
			m->positions.insert(m->positions.end(), { pos[i].x, pos[i].y, pos[i].z });
			Vector3 n = i < nrm.size() ? nrm[i] : Vector3(0, 1, 0);
			m->normals.insert(m->normals.end(), { n.x, n.y, n.z, 0.0f });
			Vector2 t = i < uv.size() ? uv[i] : Vector2();
			m->uvs.insert(m->uvs.end(), { t.x, t.y });
		}
		// Godot front faces are clockwise; the tracer culls nothing and decides facing by normal,
		// so winding does not matter here.
		if (idx.is_empty()) {
			for (int i = 0; i + 2 < pos.size(); i += 3) {
				m->indices.insert(m->indices.end(), { uint32_t(i), uint32_t(i + 1), uint32_t(i + 2) });
			}
		} else {
			for (int i = 0; i + 2 < idx.size(); i += 3) {
				m->indices.insert(m->indices.end(), { uint32_t(idx[i]), uint32_t(idx[i + 1]), uint32_t(idx[i + 2]) });
			}
		}
		surf.index_count = uint32_t(m->indices.size()) - surf.index_base;
		if (surf.index_count > 0) {
			m->surfaces.push_back(surf);
		}
	}
	mesh_cache[id] = m;
	return m;
}

GoghotGIScene::TextureStats GoghotGIScene::_texture_stats(const Ref<Texture2D> &p_tex) {
	const uint64_t id = p_tex->get_rid().get_id();
	auto it = texture_stats.find(id);
	if (it != texture_stats.end()) {
		return it->second;
	}
	// Average color (linear, alpha weighted) and alpha-test coverage, from a downsampled CPU copy.
	TextureStats st = { Color(1, 1, 1), 1.0f };
	Ref<Image> img = p_tex->get_image();
	if (img.is_valid() && !img->is_empty()) {
		img = img->duplicate();
		if (img->is_compressed()) {
			img->decompress();
		}
		img->convert(Image::FORMAT_RGBA8);
		img->resize(MIN(img->get_width(), 64), MIN(img->get_height(), 64), Image::INTERPOLATE_BILINEAR);
		double r = 0, g = 0, b = 0, wsum = 0, cov = 0;
		const int n = img->get_width() * img->get_height();
		for (int y = 0; y < img->get_height(); y++) {
			for (int x = 0; x < img->get_width(); x++) {
				Color px = img->get_pixel(x, y);
				Color l = Color(px.r, px.g, px.b).srgb_to_linear();
				r += l.r * px.a;
				g += l.g * px.a;
				b += l.b * px.a;
				wsum += px.a;
				cov += px.a >= 0.5f ? 1.0 : 0.0;
			}
		}
		if (wsum > 0) {
			st.average = Color(r / wsum, g / wsum, b / wsum);
		}
		st.coverage_at_half = n > 0 ? float(cov / n) : 1.0f;
	}
	texture_stats.emplace(id, st);
	return st;
}

GIMaterial GoghotGIScene::_material_of(const Ref<Material> &p_mat) {
	GIMaterial out;
	out.albedo = Color(0.5f, 0.5f, 0.5f);
	out.emission = Color(0, 0, 0);
	Ref<BaseMaterial3D> bm = p_mat;
	if (bm.is_valid()) {
		// Hit points are shaded as diffuse. Metals have no diffuse, but treating them as black makes them
		// vanish from GI and reflections; their specular reflectance (= base color) is used as a diffuse
		// stand-in instead, which is close for rough metals and too dull for mirrors.
		Color a = bm->get_albedo().srgb_to_linear();
		out.albedo = Color(a.r, a.g, a.b);
		Ref<Texture2D> tex = bm->get_texture(BaseMaterial3D::TEXTURE_ALBEDO);
		if (tex.is_valid()) {
			out.albedo_texture = tex->get_rid();
			const TextureStats st = _texture_stats(tex);
			out.texture_average = st.average;
			out.coverage = st.coverage_at_half;
		}
		const BaseMaterial3D::Transparency tr = bm->get_transparency();
		if (tr == BaseMaterial3D::TRANSPARENCY_ALPHA || tr == BaseMaterial3D::TRANSPARENCY_ALPHA_DEPTH_PRE_PASS) {
			out.opacity = CLAMP(a.a, 0.0f, 1.0f);
		}
		if (tr == BaseMaterial3D::TRANSPARENCY_ALPHA_SCISSOR) {
			out.alpha_scissor = bm->get_alpha_scissor_threshold();
		} else if (tr == BaseMaterial3D::TRANSPARENCY_ALPHA_HASH) {
			out.alpha_scissor = 0.5f; // stochastic in the raster, a fixed test for the tracer
		}
		out.double_sided = bm->get_cull_mode() == BaseMaterial3D::CULL_DISABLED;
		if (out.alpha_scissor < 0.0f) {
			out.coverage = 1.0f; // opaque: the texture alpha does not cut holes
		}
		if (bm->get_feature(BaseMaterial3D::FEATURE_EMISSION)) {
			// As Godot shades it: add: (color + texture) * energy, multiply: color * texture * energy;
			// energy = multiplier (* intensity in nits with physical units) * exposure normalization.
			out.emission_energy = bm->get_emission_energy_multiplier() * (physical_units ? bm->get_emission_intensity() : 1.0f) * exposure_scale;
			out.emission = bm->get_emission().srgb_to_linear() * out.emission_energy;
			Ref<Texture2D> etex = bm->get_texture(BaseMaterial3D::TEXTURE_EMISSION);
			if (etex.is_valid()) {
				out.emission_texture = etex->get_rid();
				out.emission_multiply = bm->get_emission_operator() == BaseMaterial3D::EMISSION_OP_MULTIPLY;
				out.emission_texture_average = _texture_stats(etex).average;
			}
		}
	}
	return out;
}

void GoghotGIScene::_add_instance(GISceneSnapshot &r_snap, std::unordered_map<uint64_t, uint32_t> &r_mesh_index, const Ref<Mesh> &p_mesh,
		const Transform3D &p_xform, const Ref<Material> &p_override, uint64_t p_key, bool p_cast_shadow) {
	std::shared_ptr<const GIMeshCPU> m = _get_mesh(p_mesh);
	if (!m || m->surfaces.empty()) {
		return;
	}
	auto it = r_mesh_index.find(m->id);
	uint32_t mesh_index;
	if (it == r_mesh_index.end()) {
		mesh_index = uint32_t(r_snap.meshes.size());
		r_mesh_index[m->id] = mesh_index;
		r_snap.meshes.push_back(m);
	} else {
		mesh_index = it->second;
	}
	GIInstance inst;
	inst.mesh = mesh_index;
	inst.xform = p_xform;
	inst.aabb = p_xform.xform(p_mesh->get_aabb());
	inst.key = p_key;
	inst.cast_shadow = p_cast_shadow;
	for (const GIMeshCPU::Surface &surf : m->surfaces) {
		inst.materials.push_back(_material_of(p_override.is_valid() ? p_override : p_mesh->surface_get_material(surf.source_surface)));
	}
	r_snap.instances.push_back(std::move(inst));
}

static uint64_t instance_key(const Object *p_node, uint64_t p_index) {
	uint64_t h = (p_node->get_instance_id() ^ (p_index * 0x9E3779B97F4A7C15ull)) * 0xBF58476D1CE4E5B9ull;
	return h ^ (h >> 31);
}

void GoghotGIScene::_update_units() {
	physical_units = bool(ProjectSettings::get_singleton()->get_setting("rendering/lights_and_shadows/use_physical_light_units", false));
	Ref<CameraAttributes> ca;
	if (Viewport *vp = get_viewport()) {
		if (Camera3D *cam = vp->get_camera_3d()) {
			ca = cam->get_attributes();
		}
		if (ca.is_null() && vp->find_world_3d().is_valid()) {
			ca = vp->find_world_3d()->get_camera_attributes();
		}
	}
	exposure_scale = 1.0f;
	if (ca.is_valid()) {
		// Same as CameraAttributes::_update_exposure + calculate_exposure_normalization.
		exposure_scale = ca->get_exposure_multiplier();
		if (physical_units) {
			if (Ref<CameraAttributesPhysical> phys = ca; phys.is_valid()) {
				const float e = phys->get_aperture() * phys->get_aperture() * phys->get_shutter_speed() * (100.0f / phys->get_exposure_sensitivity());
				exposure_scale *= 1.0f / (e * 1.2f);
			} else {
				exposure_scale *= ca->get_exposure_sensitivity() / 3072007.0f; // CameraAttributesPractical
			}
		}
	}
}

// Light color as the tracer shades it (diffuse = albedo * this * n.l * attenuation), matching Godot's
// light energy: non-physical: color * energy; physical: color * energy * intensity (lux, or lumens
// converted to candela: / 4 pi omni, / pi spot) / pi; both times the camera exposure normalization.
static Color light_color(const Light3D *p_light, GILight::Type p_type, bool p_physical, float p_exposure) {
	float e = p_light->get_param(Light3D::PARAM_ENERGY) * p_exposure;
	if (p_physical) {
		e *= p_light->get_param(Light3D::PARAM_INTENSITY) / float(Math_PI_CONST);
		if (p_type == GILight::OMNI) {
			e /= 4.0f * float(Math_PI_CONST);
		} else if (p_type == GILight::SPOT) {
			e /= float(Math_PI_CONST);
		}
	}
	Color c = p_light->get_color().srgb_to_linear();
	if (p_physical) {
		c *= p_light->get_correlated_color().srgb_to_linear(); // color temperature, as Light3D::set_color
	}
	return c * e;
}

void GoghotGIScene::_collect(Node *p_node, GISceneSnapshot &r_snap, std::unordered_map<uint64_t, uint32_t> &r_mesh_index) {
	GeometryInstance3D *geo = Object::cast_to<GeometryInstance3D>(p_node);
	const bool geo_on = geo != nullptr && geo->is_visible_in_tree() && geo->get_gi_mode() != GeometryInstance3D::GI_MODE_DISABLED;
	if (MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(p_node)) {
		Ref<Mesh> mesh = mi->get_mesh();
		if (geo_on && mesh.is_valid()) {
			std::shared_ptr<const GIMeshCPU> m = _get_mesh(mesh);
			if (m && !m->surfaces.empty()) {
				auto it = r_mesh_index.find(m->id);
				uint32_t mesh_index;
				if (it == r_mesh_index.end()) {
					mesh_index = uint32_t(r_snap.meshes.size());
					r_mesh_index[m->id] = mesh_index;
					r_snap.meshes.push_back(m);
				} else {
					mesh_index = it->second;
				}
				GIInstance inst;
				inst.mesh = mesh_index;
				inst.xform = mi->get_global_transform();
				Ref<ArrayMesh> am = mesh;
				inst.deformed = (mi->get_skin().is_valid() && !mi->get_skeleton_path().is_empty()) || (am.is_valid() && am->get_blend_shape_count() > 0);
				inst.rs_instance = mi->get_instance();
				inst.key = instance_key(mi, 0);
				inst.cast_shadow = mi->get_cast_shadows_setting() != GeometryInstance3D::SHADOW_CASTING_SETTING_OFF;
				inst.aabb = inst.xform.xform(mi->get_aabb());
				if (inst.deformed) {
					// Animated poses can leave the rest pose bounds: grow them (tier B dirty regions).
					inst.aabb = inst.aabb.grow(0.25f + 0.5f * inst.aabb.get_longest_axis_size());
				}
				for (const GIMeshCPU::Surface &surf : m->surfaces) {
					inst.materials.push_back(_material_of(mi->get_active_material(surf.source_surface)));
				}
				r_snap.instances.push_back(std::move(inst));
			}
		}
	} else if (MultiMeshInstance3D *mmi = Object::cast_to<MultiMeshInstance3D>(p_node)) {
		Ref<MultiMesh> mm = mmi->get_multimesh();
		if (geo_on && mm.is_valid() && mm->get_mesh().is_valid() && mm->get_transform_format() == MultiMesh::TRANSFORM_3D) {
			// Every instance becomes a world tracer instance (transforms read back from the
			// RenderingServer buffer: 12 floats per instance, then optional color / custom data).
			const int count = mm->get_visible_instance_count() < 0 ? mm->get_instance_count() : mm->get_visible_instance_count();
			const PackedFloat32Array buf = mm->get_buffer();
			const int stride = 12 + (mm->is_using_colors() ? 4 : 0) + (mm->is_using_custom_data() ? 4 : 0);
			const Transform3D base = mmi->get_global_transform();
			const Ref<Material> over = mmi->get_material_override();
			for (int i = 0; i < count && (i + 1) * stride <= buf.size(); i++) {
				const float *f = buf.ptr() + i * stride;
				const Transform3D t(Basis(f[0], f[1], f[2], f[4], f[5], f[6], f[8], f[9], f[10]), Vector3(f[3], f[7], f[11]));
				_add_instance(r_snap, r_mesh_index, mm->get_mesh(), base * t, over, instance_key(mmi, uint64_t(i)), mmi->get_cast_shadows_setting() != GeometryInstance3D::SHADOW_CASTING_SETTING_OFF);
			}
		}
	} else if (CSGShape3D *csg = Object::cast_to<CSGShape3D>(p_node)) {
		if (geo_on && csg->is_root_shape()) {
			const Array arr = csg->get_meshes(); // [local transform, baked mesh]
			if (arr.size() >= 2) {
				_add_instance(r_snap, r_mesh_index, arr[1], csg->get_global_transform() * Transform3D(arr[0]), csg->get_material_override(), instance_key(csg, 0), csg->get_cast_shadows_setting() != GeometryInstance3D::SHADOW_CASTING_SETTING_OFF);
			}
		}
	} else if (GridMap *gm = Object::cast_to<GridMap>(p_node)) {
		if (gm->is_visible_in_tree()) {
			const Array arr = gm->get_meshes(); // [local transform, mesh] per non-empty cell
			const Transform3D base = gm->get_global_transform();
			for (int i = 0; i + 1 < arr.size(); i += 2) {
				_add_instance(r_snap, r_mesh_index, arr[i + 1], base * Transform3D(arr[i]), Ref<Material>(), instance_key(gm, uint64_t(i / 2)));
			}
		}
	} else if (DirectionalLight3D *dl = Object::cast_to<DirectionalLight3D>(p_node)) {
		if (dl->is_visible_in_tree()) {
			GILight l;
			l.type = GILight::DIRECTIONAL;
			l.vec = -dl->get_global_transform().basis.get_column(2).normalized();
			l.color = light_color(dl, GILight::DIRECTIONAL, physical_units, exposure_scale);
			l.shadow = dl->has_shadow();
			r_snap.lights.push_back(l);
		}
	} else if (OmniLight3D *ol = Object::cast_to<OmniLight3D>(p_node)) {
		if (ol->is_visible_in_tree()) {
			GILight l;
			l.type = GILight::OMNI;
			l.vec = ol->get_global_position();
			l.color = light_color(ol, GILight::OMNI, physical_units, exposure_scale);
			l.range = ol->get_param(Light3D::PARAM_RANGE);
			l.attenuation = ol->get_param(Light3D::PARAM_ATTENUATION);
			l.shadow = ol->has_shadow();
			r_snap.lights.push_back(l);
		}
	} else if (SpotLight3D *sl = Object::cast_to<SpotLight3D>(p_node)) {
		if (sl->is_visible_in_tree()) {
			// Godot: omni distance attenuation * (1 - rim ^ (1 / spot_attenuation)),
			// rim = (1 - cos) / (1 - cos(spot_angle)).
			GILight l;
			l.type = GILight::SPOT;
			l.vec = sl->get_global_position();
			l.spot_dir = -sl->get_global_transform().basis.get_column(2).normalized();
			l.color = light_color(sl, GILight::SPOT, physical_units, exposure_scale);
			l.range = sl->get_param(Light3D::PARAM_RANGE);
			l.attenuation = sl->get_param(Light3D::PARAM_ATTENUATION);
			l.spot_cos = Math::cos(Math::deg_to_rad(sl->get_param(Light3D::PARAM_SPOT_ANGLE)));
			l.spot_exp = 1.0f / MAX(sl->get_param(Light3D::PARAM_SPOT_ATTENUATION), 1e-4f);
			l.shadow = sl->has_shadow();
			r_snap.lights.push_back(l);
		}
	}
	for (int i = 0; i < p_node->get_child_count(); i++) {
		_collect(p_node->get_child(i), r_snap, r_mesh_index);
	}
}

void GoghotGIScene::update_now() {
	if (gi.is_null() || !is_inside_tree()) {
		return;
	}
	_update_units();
	auto snap = std::make_shared<GISceneSnapshot>();
	std::unordered_map<uint64_t, uint32_t> mesh_index;
	Node *root = _scene_root();
	if (root == nullptr) {
		return;
	}
	_collect(root, *snap, mesh_index);
	uint64_t h = 1469598103934665603ull;
	for (const auto &m : snap->meshes) {
		h = (h ^ m->id) * 1099511628211ull;
	}
	for (const GIInstance &inst : snap->instances) {
		if (inst.deformed) {
			h = (h ^ (inst.rs_instance.get_id() * 31 + 7)) * 1099511628211ull;
		}
	}
	snap->mesh_set_hash = h;
	Ref<Environment> env;
	if (Viewport *vp = get_viewport()) {
		if (Camera3D *cam = vp->get_camera_3d()) {
			env = cam->get_environment();
		}
		if (env.is_null() && vp->find_world_3d().is_valid()) {
			env = vp->find_world_3d()->get_environment();
		}
	}
	if (env.is_valid() && env->is_volumetric_fog_enabled()) {
		snap->fog_enabled = true;
		snap->fog_length = env->get_volumetric_fog_length();
	}
	gi->set_scene_snapshot(snap);
}

Node *GoghotGIScene::_scene_root() const {
	if (Engine::get_singleton()->is_editor_hint()) {
		return get_tree()->get_edited_scene_root(); // not the editor's own UI
	}
	return get_tree()->get_root();
}

void GoghotGIScene::set_editor_preview(bool p_enable) {
	editor_preview = p_enable;
}

bool GoghotGIScene::get_editor_preview() const {
	return editor_preview;
}

void GoghotGIScene::_process(double p_delta) {
	if (Engine::get_singleton()->is_editor_hint() && !editor_preview) {
		return;
	}
	update_now();
}

void GoghotGIScene::set_gi(const Ref<GoghotScreenProbeGI> &p_gi) {
	gi = p_gi;
	set_process(gi.is_valid());
}

Ref<GoghotScreenProbeGI> GoghotGIScene::get_gi() const {
	return gi;
}

void GoghotGIScene::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_gi", "gi"), &GoghotGIScene::set_gi);
	ClassDB::bind_method(D_METHOD("get_gi"), &GoghotGIScene::get_gi);
	ClassDB::bind_method(D_METHOD("update_now"), &GoghotGIScene::update_now);
	ClassDB::bind_method(D_METHOD("set_editor_preview", "enable"), &GoghotGIScene::set_editor_preview);
	ClassDB::bind_method(D_METHOD("get_editor_preview"), &GoghotGIScene::get_editor_preview);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_preview"), "set_editor_preview", "get_editor_preview");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "gi", PROPERTY_HINT_RESOURCE_TYPE, "GoghotScreenProbeGI"), "set_gi", "get_gi");
}

} // namespace goghot
