#include "gi_scene.h"
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/callable_method_pointer.hpp>
#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/classes/light3d.hpp>

#include <algorithm>

#include "screen_probe_gi.h"

#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/base_material3d.hpp>
#include <godot_cpp/classes/skin.hpp>
#include <godot_cpp/classes/camera3d.hpp>
#include <godot_cpp/classes/camera_attributes.hpp>
#include <godot_cpp/classes/camera_attributes_physical.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/classes/time.hpp>
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
#include <godot_cpp/classes/area_light3d.hpp>
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
static constexpr uint64_t MM_REFRESH_FRAMES = 60;
static constexpr uint64_t NODE_REFRESH_FRAMES = 30; // per node attribute refresh period (staggered)
static constexpr uint32_t NODE_SLICE_FRAMES = 4; // static nodes re-read every 4 frames (staggered)
static constexpr int MM_AGGREGATE_MIN = 32; // MultiMeshes with at least this many instances are merged
static constexpr uint64_t MM_AGGREGATE_MAX_TRIS = 8000000; // ... unless the merged mesh would exceed this

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
	new_meshes++;
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

const GIMaterial &GoghotGIScene::_material_cached(const Ref<Material> &p_mat) {
	const uint64_t id = p_mat.is_valid() ? p_mat->get_rid().get_id() : 0;
	auto it = frame_materials.find(id);
	if (it != frame_materials.end()) {
		return it->second;
	}
	return frame_materials.emplace(id, _material_of(p_mat)).first->second;
}

std::shared_ptr<const std::vector<GIMaterial>> GoghotGIScene::_materials_of(const Ref<Mesh> &p_mesh, const GIMeshCPU &p_cpu, const Ref<Material> &p_override,
		const GeometryInstance3D *p_geo) {
	auto v = std::make_shared<std::vector<GIMaterial>>();
	v->reserve(p_cpu.surfaces.size());
	for (const GIMeshCPU::Surface &surf : p_cpu.surfaces) {
		Ref<Material> m = p_override;
		if (m.is_null()) {
			const MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(p_geo);
			m = mi ? mi->get_active_material(surf.source_surface) : p_mesh->surface_get_material(surf.source_surface);
		}
		v->push_back(_material_cached(m));
	}
	return v;
}

bool GoghotGIScene::_culled(const AABB &p_world, const GIMeshCPU &p_cpu, const std::vector<GIMaterial> &p_mats) const {
	if (!cull_valid || trace_min_size <= 0.0f) {
		return false;
	}
	for (const GIMaterial &m : p_mats) {
		if (m.emission.r + m.emission.g + m.emission.b > 0.0f || m.emission_texture.is_valid()) {
			return false; // small emitters still light the scene
		}
	}
	const float d = MAX(p_world.get_center().distance_to(cull_origin) - 0.5f * p_world.size.length(), 0.0f);
	return p_world.get_longest_axis_size() < trace_min_size * d;
}

void GoghotGIScene::_add_instance(GISceneSnapshot &r_snap, std::unordered_map<uint64_t, uint32_t> &r_mesh_index, const Ref<Mesh> &p_mesh,
		const Transform3D &p_xform, const Ref<Material> &p_override, uint64_t p_key, bool p_cast_shadow, const std::shared_ptr<const std::vector<GIMaterial>> &p_materials) {
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
	inst.aabb = p_xform.xform(p_mesh->get_aabb());
	inst.materials = p_materials ? p_materials : _materials_of(p_mesh, *m, p_override, nullptr);
	if (_culled(inst.aabb, *m, *inst.materials)) {
		culled++;
		return;
	}
	inst.mesh = mesh_index;
	inst.xform = p_xform;
	inst.key = p_key;
	inst.cast_shadow = p_cast_shadow;
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
		} else if (p_type == GILight::AREA) {
			e /= 2.0f * float(Math_PI_CONST);
		}
	}
	Color c = p_light->get_color().srgb_to_linear();
	if (p_physical) {
		c *= p_light->get_correlated_color().srgb_to_linear(); // color temperature, as Light3D::set_color
	}
	return c * e;
}

void GoghotGIScene::_collect(Node *p_node, GISceneSnapshot &r_snap, std::unordered_map<uint64_t, uint32_t> &r_mesh_index) {
	nodes_visited++;
	if (_collect_node(p_node, r_snap, r_mesh_index)) {
		if (relevant_set.insert(p_node->get_instance_id()).second) {
			relevant_nodes.push_back(p_node->get_instance_id());
		}
	}
	for (int i = 0; i < p_node->get_child_count(); i++) {
		_collect(p_node->get_child(i), r_snap, r_mesh_index);
	}
}

bool GoghotGIScene::_is_relevant(Node *p_node) {
	return Object::cast_to<GeometryInstance3D>(p_node) != nullptr || Object::cast_to<GridMap>(p_node) != nullptr || Object::cast_to<Light3D>(p_node) != nullptr;
}

bool GoghotGIScene::_refresh_due(uint64_t p_id) const {
	return (update_frame + p_id) % NODE_REFRESH_FRAMES == 0;
}

void GoghotGIScene::_on_tree_changed(Node *p_node) {
	// node_added fires for every node of an added subtree: relevant ones are appended directly; removed
	// nodes are dropped lazily when found outside the tree (no re-walk of the whole tree).
	if (!_is_relevant(p_node)) {
		return;
	}
	const uint64_t id = p_node->get_instance_id();
	node_records.erase(id); // a re-added node is re-read
	if (relevant_set.insert(id).second) {
		relevant_nodes.push_back(id);
	}
}

void GoghotGIScene::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		tree_dirty = true;
		SceneTree *tree = get_tree();
		const Callable cb = callable_mp(this, &GoghotGIScene::_on_tree_changed);
		if (!tree->is_connected("node_added", cb)) {
			tree->connect("node_added", cb);
		}
	} else if (p_what == NOTIFICATION_EXIT_TREE) {
		SceneTree *tree = get_tree();
		const Callable cb = callable_mp(this, &GoghotGIScene::_on_tree_changed);
		if (tree && tree->is_connected("node_added", cb)) {
			tree->disconnect("node_added", cb);
		}
		relevant_nodes.clear();
		relevant_set.clear();
		node_records.clear();
	}
}

// Collects one node (no recursion). Returns whether the node is of a kind that contributes (geometry or
// light), whatever its current visibility: such nodes are re-read every update from relevant_nodes.
bool GoghotGIScene::_collect_node(Node *p_node, GISceneSnapshot &r_snap, std::unordered_map<uint64_t, uint32_t> &r_mesh_index) {
	if (!_is_relevant(p_node)) {
		return false;
	}
	const uint64_t nid = p_node->get_instance_id();
	GeometryInstance3D *geo = Object::cast_to<GeometryInstance3D>(p_node);
	if (MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(p_node)) {
		NodeRecord &rec = node_records[nid];
		if (!rec.valid || _refresh_due(nid)) {
			rec = NodeRecord();
			rec.valid = true;
			Ref<Mesh> mesh = mi->get_mesh();
			std::shared_ptr<const GIMeshCPU> m = mesh.is_valid() ? _get_mesh(mesh) : nullptr;
			rec.off = !m || m->surfaces.empty() || mi->get_gi_mode() == GeometryInstance3D::GI_MODE_DISABLED;
			if (!rec.off) {
				rec.mesh = m;
				rec.mesh_res = mesh;
				Ref<ArrayMesh> am = mesh;
				rec.deformed = (mi->get_skin().is_valid() && !mi->get_skeleton_path().is_empty()) || (am.is_valid() && am->get_blend_shape_count() > 0);
				rec.rs_instance = mi->get_instance();
				rec.cast_shadow = mi->get_cast_shadows_setting() != GeometryInstance3D::SHADOW_CASTING_SETTING_OFF;
				rec.local_aabb = mi->get_aabb();
				rec.materials = _materials_of(mesh, *m, Ref<Material>(), mi);
			}
		}
		if (!rec.off && mi->is_visible_in_tree()) {
			auto it = r_mesh_index.find(rec.mesh->id);
			uint32_t mesh_index;
			if (it == r_mesh_index.end()) {
				mesh_index = uint32_t(r_snap.meshes.size());
				r_mesh_index[rec.mesh->id] = mesh_index;
				r_snap.meshes.push_back(rec.mesh);
			} else {
				mesh_index = it->second;
			}
			GIInstance inst;
			inst.mesh = mesh_index;
			inst.xform = mi->get_global_transform();
			inst.deformed = rec.deformed;
			inst.rs_instance = rec.rs_instance;
			inst.key = instance_key(mi, 0);
			inst.cast_shadow = rec.cast_shadow;
			inst.aabb = inst.xform.xform(rec.local_aabb);
			if (inst.deformed) {
				// Animated poses can leave the rest pose bounds: grow them (tier B dirty regions).
				inst.aabb = inst.aabb.grow(0.25f + 0.5f * inst.aabb.get_longest_axis_size());
			}
			inst.materials = rec.materials;
			if (!inst.deformed && _culled(inst.aabb, *rec.mesh, *inst.materials)) {
				culled++;
			} else {
				r_snap.instances.push_back(std::move(inst));
			}
		}
		return true;
	}
	const bool geo_on = geo != nullptr && geo->is_visible_in_tree() && geo->get_gi_mode() != GeometryInstance3D::GI_MODE_DISABLED;
	if (false) {
	} else if (MultiMeshInstance3D *mmi = Object::cast_to<MultiMeshInstance3D>(p_node)) {
		Ref<MultiMesh> mm = mmi->get_multimesh();
		if (geo_on && mm.is_valid() && mm->get_mesh().is_valid() && mm->get_transform_format() == MultiMesh::TRANSFORM_3D) {
			// Every instance becomes a world tracer instance (transforms read back from the
			// RenderingServer buffer: 12 floats per instance, then optional color / custom data).
			const int count = mm->get_visible_instance_count() < 0 ? mm->get_instance_count() : mm->get_visible_instance_count();
			MMCache &mc = mm_cache[mm->get_rid().get_id()];
			if (mc.count != count || update_frame - mc.frame >= MM_REFRESH_FRAMES || mc.buffer.is_empty()) {
				mc.buffer = mm->get_buffer();
				mc.count = count;
				mc.frame = update_frame;
			}
			const PackedFloat32Array &buf = mc.buffer;
			const int stride = 12 + (mm->is_using_colors() ? 4 : 0) + (mm->is_using_custom_data() ? 4 : 0);
			const Transform3D base = mmi->get_global_transform();
			const Ref<Material> over = mmi->get_material_override();
			std::shared_ptr<const GIMeshCPU> mcpu = _get_mesh(mm->get_mesh());
			const int valid = MIN(count, stride > 0 ? int(buf.size() / stride) : 0);
			const uint64_t src_tris = mcpu ? mcpu->indices.size() / 3 : 0;
			if (mcpu && !mcpu->surfaces.empty() && valid >= MM_AGGREGATE_MIN && src_tris * uint64_t(valid) <= MM_AGGREGATE_MAX_TRIS) {
				const auto mats = _materials_of(mm->get_mesh(), *mcpu, over, nullptr);
				MMAggregate &ag = mm_aggregates[mm->get_rid().get_id()];
				if (mc.frame == update_frame || !ag.mesh || ag.src_mesh != mcpu->id) { // buffer (re)read this frame: re-merge if it changed
					uint64_t h = 1469598103934665603ull ^ uint64_t(valid);
					const uint32_t *bits = reinterpret_cast<const uint32_t *>(buf.ptr());
					for (int64_t k = 0; k < int64_t(valid) * stride; k++) {
						h = (h ^ bits[k]) * 1099511628211ull;
					}
					if (!ag.mesh || ag.src_mesh != mcpu->id || ag.hash != h) {
						auto out = std::make_shared<GIMeshCPU>();
						out->id = 0x8000000000000000ull | ((mm->get_rid().get_id() * 1315423911ull + ++ag.version) & 0x7FFFFFFFFFFFFFFFull);
						AABB box;
						bool first = true;
						for (const GIMeshCPU::Surface &src : mcpu->surfaces) {
							GIMeshCPU::Surface surf = src;
							surf.vertex_base = uint32_t(out->positions.size() / 3);
							surf.vertex_count = src.vertex_count * uint32_t(valid);
							surf.index_base = uint32_t(out->indices.size());
							for (int i = 0; i < valid; i++) {
								const float *f = buf.ptr() + i * stride;
								const Transform3D t(Basis(f[0], f[1], f[2], f[4], f[5], f[6], f[8], f[9], f[10]), Vector3(f[3], f[7], f[11]));
								const uint32_t vb = uint32_t(out->positions.size() / 3) - surf.vertex_base;
								for (uint32_t v = 0; v < src.vertex_count; v++) {
									const uint32_t sv = src.vertex_base + v;
									const Vector3 p = t.xform(Vector3(mcpu->positions[sv * 3], mcpu->positions[sv * 3 + 1], mcpu->positions[sv * 3 + 2]));
									const Vector3 n = t.basis.xform(Vector3(mcpu->normals[sv * 4], mcpu->normals[sv * 4 + 1], mcpu->normals[sv * 4 + 2])).normalized();
									out->positions.insert(out->positions.end(), { p.x, p.y, p.z });
									out->normals.insert(out->normals.end(), { n.x, n.y, n.z, 0.0f });
									out->uvs.insert(out->uvs.end(), { mcpu->uvs[sv * 2], mcpu->uvs[sv * 2 + 1] });
									if (first) {
										box = AABB(p, Vector3());
										first = false;
									} else {
										box.expand_to(p);
									}
								}
								for (uint32_t k = 0; k < src.index_count; k++) {
									out->indices.push_back(mcpu->indices[src.index_base + k] + vb);
								}
							}
							surf.index_count = src.index_count * uint32_t(valid);
							out->surfaces.push_back(surf);
						}
						ag.mesh = out;
						ag.src_mesh = mcpu->id;
						ag.hash = h;
						ag.local_aabb = box;
					}
				}
				GIInstance inst;
				auto it = r_mesh_index.find(ag.mesh->id);
				if (it == r_mesh_index.end()) {
					inst.mesh = uint32_t(r_snap.meshes.size());
					r_mesh_index[ag.mesh->id] = inst.mesh;
					r_snap.meshes.push_back(ag.mesh);
				} else {
					inst.mesh = it->second;
				}
				inst.xform = base;
				inst.aabb = base.xform(ag.local_aabb);
				inst.key = instance_key(mmi, 0xFFFFFFFFull);
				inst.cast_shadow = mmi->get_cast_shadows_setting() != GeometryInstance3D::SHADOW_CASTING_SETTING_OFF;
				inst.materials = mats;
				r_snap.instances.push_back(std::move(inst));
			} else if (mcpu && !mcpu->surfaces.empty()) {
				const auto mats = _materials_of(mm->get_mesh(), *mcpu, over, nullptr);
				const bool cast = mmi->get_cast_shadows_setting() != GeometryInstance3D::SHADOW_CASTING_SETTING_OFF;
				for (int i = 0; i < count && (i + 1) * stride <= buf.size(); i++) {
					const float *f = buf.ptr() + i * stride;
					const Transform3D t(Basis(f[0], f[1], f[2], f[4], f[5], f[6], f[8], f[9], f[10]), Vector3(f[3], f[7], f[11]));
					_add_instance(r_snap, r_mesh_index, mm->get_mesh(), base * t, over, instance_key(mmi, uint64_t(i)), cast, mats);
				}
			}
		}
	} else if (CSGShape3D *csg = Object::cast_to<CSGShape3D>(p_node)) {
		NodeRecord &rec = node_records[nid];
		if (!rec.valid || _refresh_due(nid)) {
			rec = NodeRecord();
			rec.valid = true;
			rec.off = !csg->is_root_shape();
			if (!rec.off) {
				rec.meshes = csg->get_meshes(); // [local transform, baked mesh]
				rec.override_mat = csg->get_material_override();
				rec.cast_shadow = csg->get_cast_shadows_setting() != GeometryInstance3D::SHADOW_CASTING_SETTING_OFF;
			}
		}
		if (geo_on && !rec.off && rec.meshes.size() >= 2) {
			_add_instance(r_snap, r_mesh_index, rec.meshes[1], csg->get_global_transform() * Transform3D(rec.meshes[0]), rec.override_mat, instance_key(csg, 0), rec.cast_shadow);
		}
	} else if (GridMap *gm = Object::cast_to<GridMap>(p_node)) {
		NodeRecord &rec = node_records[nid];
		if (!rec.valid || _refresh_due(nid)) {
			rec = NodeRecord();
			rec.valid = true;
			rec.off = false;
			rec.meshes = gm->get_meshes(); // [local transform, mesh] per non-empty cell
		}
		if (gm->is_visible_in_tree()) {
			const Array &arr = rec.meshes;
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
	} else if (AreaLight3D *al = Object::cast_to<AreaLight3D>(p_node)) {
		const Vector2 sz = al->get_area_size();
		if (al->is_visible_in_tree() && sz.x > 0.0f && sz.y > 0.0f) {
			// Godot: a one-sided rectangle (local X width, Y height) of uniform radiance emitting along
			// -Z, radiance = color * energy (/ area with area_normalize_energy), LTC integrated; the
			// distance window is Godot's omni attenuation at the closest point of the rectangle times d^2.
			const Basis b = al->get_global_transform().basis;
			GILight l;
			l.type = GILight::AREA;
			l.vec = al->get_global_position();
			l.spot_dir = -b.get_column(2).normalized();
			l.area_a = b.get_column(0).normalized() * (0.5f * sz.x);
			l.area_b = b.get_column(1).normalized() * (0.5f * sz.y);
			l.color = light_color(al, GILight::AREA, physical_units, exposure_scale) / (al->is_area_normalizing_energy() ? sz.x * sz.y : 1.0f);
			l.range = al->get_param(Light3D::PARAM_RANGE);
			l.attenuation = al->get_param(Light3D::PARAM_ATTENUATION);
			l.shadow = al->has_shadow();
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
	return true;
}

void GoghotGIScene::update_now() {
	if (gi.is_null() || !is_inside_tree()) {
		return;
	}
	const uint64_t t0 = Time::get_singleton()->get_ticks_usec();
	nodes_visited = 0;
	new_meshes = 0;
	culled = 0;
	update_frame++;
	frame_materials.clear();
	cull_valid = false;
	if (Viewport *vp = get_viewport()) {
		if (Camera3D *cam = vp->get_camera_3d()) {
			cull_origin = cam->get_global_position();
			cull_valid = true;
		}
	}
	_update_units();
	auto snap = std::make_shared<GISceneSnapshot>();
	std::unordered_map<uint64_t, uint32_t> mesh_index;
	Node *root = _scene_root();
	if (root == nullptr) {
		return;
	}
	// Full tree walk only when 3D nodes were added / removed (or the root changed); otherwise only the
	// geometry / light nodes found by the last walk are read (12k-node scenes: the walk itself cost ms).
	if (tree_dirty || relevant_root != root->get_instance_id()) {
		relevant_nodes.clear();
		relevant_set.clear();
		node_out.clear();
		relevant_root = root->get_instance_id();
		tree_dirty = false;
		_collect(root, *snap, mesh_index);
	} else {
		const bool check_ancestor = root != get_tree()->get_root(); // editor: only the edited scene
		for (size_t i = 0; i < relevant_nodes.size();) {
			const uint64_t id = relevant_nodes[i];
			Node *n = Object::cast_to<Node>(ObjectDB::get_instance(id));
			if (n == nullptr || !n->is_inside_tree() || (check_ancestor && n != root && !root->is_ancestor_of(n))) {
				// Removed (or outside the collected root): drop it (re-added nodes come back via node_added).
				relevant_set.erase(id);
				node_records.erase(id);
				node_out.erase(id);
				relevant_nodes[i] = relevant_nodes.back();
				relevant_nodes.pop_back();
				continue;
			}
			NodeOut &o = node_out[id];
			if (!o.valid || o.always || o.hot > 0 || (update_frame + id) % NODE_SLICE_FRAMES == 0) {
				nodes_visited++;
				GISceneSnapshot tmp;
				std::unordered_map<uint64_t, uint32_t> tmp_index;
				_collect_node(n, tmp, tmp_index);
				o.instances.clear();
				o.meshes = tmp.meshes;
				o.always = !tmp.lights.empty();
				for (const GIInstance &inst : tmp.instances) {
					o.instances.emplace_back(tmp.meshes[inst.mesh], inst);
					o.always = o.always || inst.deformed;
				}
				o.lights = std::move(tmp.lights);
				const Transform3D x = o.instances.empty() ? Transform3D() : o.instances[0].second.xform;
				if (o.valid && x != o.xform) {
					o.hot = NODE_SLICE_FRAMES * 2; // moving: read every frame for a while
				} else if (o.hot > 0) {
					o.hot--;
				}
				o.xform = x;
				o.valid = true;
			}
			for (const auto &m : o.meshes) {
				if (mesh_index.find(m->id) == mesh_index.end()) {
					mesh_index[m->id] = uint32_t(snap->meshes.size());
					snap->meshes.push_back(m);
				}
			}
			for (const auto &pi : o.instances) {
				snap->instances.push_back(pi.second);
				snap->instances.back().mesh = mesh_index[pi.first->id];
			}
			snap->lights.insert(snap->lights.end(), o.lights.begin(), o.lights.end());
			i++;
		}
	}
	// Canonical mesh order (by id): the same set of meshes always gives the same order and hash, so the
	// world tracer does not rebuild all geometry when only the scene tree order changed.
	{
		std::vector<uint32_t> order(snap->meshes.size());
		for (uint32_t i = 0; i < order.size(); i++) {
			order[i] = i;
		}
		std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return snap->meshes[a]->id < snap->meshes[b]->id; });
		std::vector<uint32_t> remap(order.size());
		std::vector<std::shared_ptr<const GIMeshCPU>> sorted(order.size());
		for (uint32_t i = 0; i < order.size(); i++) {
			sorted[i] = snap->meshes[order[i]];
			remap[order[i]] = i;
		}
		snap->meshes = std::move(sorted);
		for (GIInstance &inst : snap->instances) {
			inst.mesh = remap[inst.mesh];
		}
	}
	std::vector<uint64_t> deformed_ids;
	for (const GIInstance &inst : snap->instances) {
		if (inst.deformed) {
			deformed_ids.push_back(inst.rs_instance.get_id());
		}
	}
	std::sort(deformed_ids.begin(), deformed_ids.end());
	uint64_t h = 1469598103934665603ull;
	for (const auto &m : snap->meshes) {
		h = (h ^ m->id) * 1099511628211ull;
	}
	for (const uint64_t id : deformed_ids) {
		h = (h ^ (id * 31 + 7)) * 1099511628211ull;
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
	collect_ms = double(Time::get_singleton()->get_ticks_usec() - t0) / 1000.0;
}

Dictionary GoghotGIScene::get_collect_stats() const {
	Dictionary d;
	d["collect_ms"] = collect_ms;
	d["nodes_visited"] = int64_t(nodes_visited);
	d["new_meshes"] = int64_t(new_meshes);
	d["mesh_cache"] = int64_t(mesh_cache.size());
	d["culled"] = int64_t(culled);
	return d;
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
	ClassDB::bind_method(D_METHOD("get_collect_stats"), &GoghotGIScene::get_collect_stats);
	ClassDB::bind_method(D_METHOD("set_trace_min_size", "size"), &GoghotGIScene::set_trace_min_size);
	ClassDB::bind_method(D_METHOD("get_trace_min_size"), &GoghotGIScene::get_trace_min_size);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "trace_min_size", PROPERTY_HINT_RANGE, "0,0.1,0.001"), "set_trace_min_size", "get_trace_min_size");
	ClassDB::bind_method(D_METHOD("set_editor_preview", "enable"), &GoghotGIScene::set_editor_preview);
	ClassDB::bind_method(D_METHOD("get_editor_preview"), &GoghotGIScene::get_editor_preview);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_preview"), "set_editor_preview", "get_editor_preview");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "gi", PROPERTY_HINT_RESOURCE_TYPE, "GoghotScreenProbeGI"), "set_gi", "get_gi");
}

} // namespace goghot
