#include "gi_scene.h"

#include "screen_probe_gi.h"

#include <godot_cpp/classes/base_material3d.hpp>
#include <godot_cpp/classes/camera3d.hpp>
#include <godot_cpp/classes/environment.hpp>
#include <godot_cpp/classes/viewport.hpp>
#include <godot_cpp/classes/world3d.hpp>
#include <godot_cpp/classes/directional_light3d.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/classes/omni_light3d.hpp>
#include <godot_cpp/classes/scene_tree.hpp>
#include <godot_cpp/classes/window.hpp>
#include <godot_cpp/core/class_db.hpp>
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
		GIMeshCPU::Surface surf;
		surf.source_surface = s;
		surf.vertex_base = uint32_t(m->positions.size() / 3);
		surf.vertex_count = uint32_t(pos.size());
		surf.index_base = uint32_t(m->indices.size());
		for (int i = 0; i < pos.size(); i++) {
			m->positions.insert(m->positions.end(), { pos[i].x, pos[i].y, pos[i].z });
			Vector3 n = i < nrm.size() ? nrm[i] : Vector3(0, 1, 0);
			m->normals.insert(m->normals.end(), { n.x, n.y, n.z, 0.0f });
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

static GIMaterial material_of(const Ref<Material> &p_mat) {
	GIMaterial out;
	out.albedo = Color(0.5f, 0.5f, 0.5f);
	out.emission = Color(0, 0, 0);
	Ref<BaseMaterial3D> bm = p_mat;
	if (bm.is_valid()) {
		Color a = bm->get_albedo().srgb_to_linear();
		float metallic = bm->get_metallic();
		out.albedo = Color(a.r, a.g, a.b) * (1.0f - metallic); // diffuse part only
		if (bm->get_feature(BaseMaterial3D::FEATURE_EMISSION)) {
			out.emission = bm->get_emission().srgb_to_linear() * bm->get_emission_energy_multiplier();
		}
	}
	return out;
}

void GoghotGIScene::_collect(Node *p_node, GISceneSnapshot &r_snap, std::unordered_map<uint64_t, uint32_t> &r_mesh_index) {
	if (MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(p_node)) {
		Ref<Mesh> mesh = mi->get_mesh();
		if (mi->is_visible_in_tree() && mesh.is_valid()) {
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
				for (const GIMeshCPU::Surface &surf : m->surfaces) {
					inst.materials.push_back(material_of(mi->get_active_material(surf.source_surface)));
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
	_collect(get_tree()->get_root(), *snap, mesh_index);
	uint64_t h = 1469598103934665603ull;
	for (const auto &m : snap->meshes) {
		h = (h ^ m->id) * 1099511628211ull;
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

void GoghotGIScene::_process(double p_delta) {
	if (Engine::get_singleton()->is_editor_hint()) {
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
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "gi", PROPERTY_HINT_RESOURCE_TYPE, "GoghotScreenProbeGI"), "set_gi", "get_gi");
}

} // namespace goghot
