#include "gi_scene.h"

#include "screen_probe_gi.h"

#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/base_material3d.hpp>
#include <godot_cpp/classes/skin.hpp>
#include <godot_cpp/classes/camera3d.hpp>
#include <godot_cpp/classes/environment.hpp>
#include <godot_cpp/classes/viewport.hpp>
#include <godot_cpp/classes/world3d.hpp>
#include <godot_cpp/classes/directional_light3d.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/geometry_instance3d.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/classes/omni_light3d.hpp>
#include <godot_cpp/classes/scene_tree.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/window.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>

using namespace godot;

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
			auto it = texture_stats.find(out.albedo_texture.get_id());
			if (it == texture_stats.end()) {
				// Average color (linear) and alpha-test coverage, from a downsampled CPU copy.
				TextureStats st = { Color(1, 1, 1), 1.0f };
				Ref<Image> img = tex->get_image();
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
				it = texture_stats.emplace(out.albedo_texture.get_id(), st).first;
			}
			out.texture_average = it->second.average;
			out.coverage = it->second.coverage_at_half;
		}
		const BaseMaterial3D::Transparency tr = bm->get_transparency();
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
			out.emission = bm->get_emission().srgb_to_linear() * bm->get_emission_energy_multiplier();
		}
	}
	return out;
}

void GoghotGIScene::_collect(Node *p_node, GISceneSnapshot &r_snap, std::unordered_map<uint64_t, uint32_t> &r_mesh_index) {
	if (MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(p_node)) {
		Ref<Mesh> mesh = mi->get_mesh();
		if (mi->is_visible_in_tree() && mesh.is_valid() && mi->get_gi_mode() != GeometryInstance3D::GI_MODE_DISABLED) {
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
				for (const GIMeshCPU::Surface &surf : m->surfaces) {
					inst.materials.push_back(_material_of(mi->get_active_material(surf.source_surface)));
				}
				r_snap.instances.push_back(std::move(inst));
			}
		}
	} else if (DirectionalLight3D *dl = Object::cast_to<DirectionalLight3D>(p_node)) {
		if (dl->is_visible_in_tree()) {
			GILight l;
			l.type = GILight::DIRECTIONAL;
			l.vec = -dl->get_global_transform().basis.get_column(2).normalized();
			l.color = dl->get_color().srgb_to_linear() * dl->get_param(Light3D::PARAM_ENERGY);
			l.shadow = dl->has_shadow();
			r_snap.lights.push_back(l);
		}
	} else if (OmniLight3D *ol = Object::cast_to<OmniLight3D>(p_node)) {
		if (ol->is_visible_in_tree()) {
			GILight l;
			l.type = GILight::OMNI;
			l.vec = ol->get_global_position();
			l.color = ol->get_color().srgb_to_linear() * ol->get_param(Light3D::PARAM_ENERGY);
			l.range = ol->get_param(Light3D::PARAM_RANGE);
			l.attenuation = ol->get_param(Light3D::PARAM_ATTENUATION);
			l.shadow = ol->has_shadow();
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
