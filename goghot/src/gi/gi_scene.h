#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/geometry_instance3d.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/variant/aabb.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/classes/material.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/transform3d.hpp>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace goghot {

class GoghotScreenProbeGI;

// CPU-side scene description for the world tracer. Built on the main thread by GoghotGIScene,
// consumed on the render thread by GoghotScreenProbeGI (immutable once published).
struct GIMeshCPU {
	struct Surface {
		uint32_t vertex_base, vertex_count, index_base, index_count;
		int32_t source_surface; // surface index in the Mesh (empty surfaces are skipped)
		bool has_normal, has_tangent; // layout of Godot's deformed vertex buffer
	};
	uint64_t id = 0; // Mesh RID id
	std::vector<float> positions; // xyz per vertex (BLAS input)
	std::vector<float> normals; // xyz0 per vertex (shading)
	std::vector<float> uvs; // uv per vertex (texture lookups at hits)
	std::vector<uint32_t> indices; // per surface, relative to the surface vertex_base
	std::vector<Surface> surfaces;
};

struct GIMaterial {
	godot::Color albedo; // linear; alpha unused (multiplied with the albedo texture)
	godot::Color emission; // linear, energy applied
	godot::RID albedo_texture; // RenderingServer texture, invalid = none
	float alpha_scissor = -1.0f; // < 0: opaque
	bool double_sided = false;
	godot::Color texture_average = godot::Color(1, 1, 1); // linear average of the albedo texture (voxel tier)
	float coverage = 1.0f; // fraction of the surface that is solid (alpha tested foliage < 1)
	float opacity = 1.0f; // alpha blended (glass, ...): albedo alpha; rays pass with probability 1 - opacity
	// Emission texture: emission = (emission + texture * energy) (add) or emission * texture (multiply).
	godot::RID emission_texture;
	bool emission_multiply = false;
	float emission_energy = 1.0f;
	godot::Color emission_texture_average = godot::Color(1, 1, 1); // linear (voxel tier)
};

struct GIInstance {
	uint32_t mesh; // index into GISceneSnapshot::meshes
	godot::Transform3D xform;
	std::shared_ptr<const std::vector<GIMaterial>> materials; // per surface (shared by MultiMesh instances)
	bool deformed = false; // skinned / blend shapes: geometry comes from Godot's per-instance buffer
	godot::RID rs_instance; // RenderingServer instance (for instance_surface_get_vertex_buffer_rd_rid)
	godot::AABB aabb; // world bounds (deformed: rest pose bounds, grown)
	uint64_t key = 0; // stable identity across frames (node, and cell / instance index)
	bool cast_shadow = true; // GeometryInstance3D.cast_shadow != OFF: blocks shadow rays
};

struct GILight {
	enum Type { DIRECTIONAL = 0, OMNI = 1, SPOT = 2, AREA = 3 };
	Type type;
	godot::Vector3 vec; // DIRECTIONAL: direction light travels; OMNI / SPOT: position
	godot::Vector3 spot_dir; // SPOT: direction the cone points; AREA: direction the rectangle emits to
	godot::Vector3 area_a, area_b; // AREA: half width / half height vectors of the rectangle
	float spot_cos = -1.0f; // SPOT: cos of the cone half angle
	float spot_exp = 1.0f; // SPOT: Godot's 1 / spot_angle_attenuation
	godot::Color color; // linear * energy (non-physical units, as Godot shades it)
	float range = 0.0f;
	float attenuation = 1.0f;
	bool shadow = true;
};

struct GISceneSnapshot {
	std::vector<std::shared_ptr<const GIMeshCPU>> meshes;
	uint64_t mesh_set_hash = 0; // changes when the set of meshes or deformed instances changes (GPU rebuild)
	std::vector<GIInstance> instances;
	std::vector<GILight> lights;
	bool fog_enabled = false; // Environment volumetric fog of the active camera / world
	float fog_length = 0.0f;
};

// Collects meshes, materials and lights of the scene tree for the world tracer every frame and
// hands them to its GoghotScreenProbeGI: MeshInstance3D (incl. skinned), MultiMeshInstance3D, CSG
// root shapes, GridMap cells (BaseMaterial3D albedo / emission colors and textures, alpha scissor,
// cull mode; other materials are grey), DirectionalLight3D, OmniLight3D and SpotLight3D.
// Geometry with gi_mode = Disabled is excluded (it then neither occludes nor bounces light).
// In the editor it only runs with editor_preview and only collects the edited scene.
class GoghotGIScene : public godot::Node {
	GDCLASS(GoghotGIScene, godot::Node);

protected:
	godot::Ref<GoghotScreenProbeGI> gi;
	bool editor_preview = false;
	godot::Node *_scene_root() const;
	// Light units, refreshed every update: physical light units project setting and the active camera
	// attributes' exposure normalization (exposure_multiplier, times the physical camera / sensitivity
	// normalization when physical units are on), which Godot applies to lights and emission.
	// Collection statistics of the last update (CPU, main thread).
	double collect_ms = 0.0;
	uint64_t nodes_visited = 0;
	uint64_t new_meshes = 0; // meshes read from the Mesh resource this frame (mesh cache misses)
	bool physical_units = false;
	float exposure_scale = 1.0f;
	void _update_units();
	std::unordered_map<uint64_t, std::shared_ptr<const GIMeshCPU>> mesh_cache;
	struct TextureStats {
		godot::Color average;
		float coverage_at_half;
	};
	std::unordered_map<uint64_t, TextureStats> texture_stats; // per texture RID, computed once
	TextureStats _texture_stats(const godot::Ref<godot::Texture2D> &p_tex);
	GIMaterial _material_of(const godot::Ref<godot::Material> &p_mat);
	// Per update: materials by RID (each distinct material is evaluated once per frame).
	std::unordered_map<uint64_t, GIMaterial> frame_materials;
	const GIMaterial &_material_cached(const godot::Ref<godot::Material> &p_mat);
	std::shared_ptr<const std::vector<GIMaterial>> _materials_of(const godot::Ref<godot::Mesh> &p_mesh, const GIMeshCPU &p_cpu,
			const godot::Ref<godot::Material> &p_override, const godot::GeometryInstance3D *p_geo);
	// World tracer culling of small, distant instances (their GI contribution is negligible; on screen
	// the screen probes still see them): skipped when bounds size / distance < trace_min_size.
	float trace_min_size = 0.01f;
	bool cull_valid = false;
	godot::Vector3 cull_origin;
	uint64_t culled = 0;
	bool _culled(const godot::AABB &p_world, const GIMeshCPU &p_cpu, const std::vector<GIMaterial> &p_mats) const;
	// MultiMesh transform buffers, read back from the RenderingServer only every MM_REFRESH_FRAMES
	// (a readback can stall the CPU on the GPU) or when the instance count changes.
	struct MMCache {
		godot::PackedFloat32Array buffer;
		int count = -1;
		uint64_t frame = 0;
	};
	std::unordered_map<uint64_t, MMCache> mm_cache;
	// MultiMeshes with many instances (grass, debris) are merged into one static mesh in the
	// MultiMesh's space: one BLAS and one TLAS instance instead of thousands, re-merged only when the
	// transforms (or the mesh) change.
	struct MMAggregate {
		std::shared_ptr<const GIMeshCPU> mesh;
		uint64_t src_mesh = 0, hash = 0, version = 0;
		godot::AABB local_aabb;
	};
	std::unordered_map<uint64_t, MMAggregate> mm_aggregates;
	uint64_t update_frame = 0;
	// One instance of p_mesh (materials: p_override, else the mesh's surface materials).
	void _add_instance(GISceneSnapshot &r_snap, std::unordered_map<uint64_t, uint32_t> &r_mesh_index, const godot::Ref<godot::Mesh> &p_mesh,
			const godot::Transform3D &p_xform, const godot::Ref<godot::Material> &p_override, uint64_t p_key, bool p_cast_shadow = true,
			const std::shared_ptr<const std::vector<GIMaterial>> &p_materials = nullptr);

	void _collect(godot::Node *p_node, GISceneSnapshot &r_snap, std::unordered_map<uint64_t, uint32_t> &r_mesh_index);
	std::shared_ptr<const GIMeshCPU> _get_mesh(const godot::Ref<godot::Resource> &p_mesh);

	static void _bind_methods();

public:
	void _process(double p_delta) override;
	void update_now();
	void set_editor_preview(bool p_enable);
	void set_trace_min_size(float p_size) { trace_min_size = p_size; }
	float get_trace_min_size() const { return trace_min_size; }
	godot::Dictionary get_collect_stats() const;
	bool get_editor_preview() const;

	void set_gi(const godot::Ref<GoghotScreenProbeGI> &p_gi);
	godot::Ref<GoghotScreenProbeGI> get_gi() const;
};

} // namespace goghot
