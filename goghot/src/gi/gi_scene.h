#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/material.hpp>
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
};

struct GIInstance {
	uint32_t mesh; // index into GISceneSnapshot::meshes
	godot::Transform3D xform;
	std::vector<GIMaterial> materials; // per surface
	bool deformed = false; // skinned / blend shapes: geometry comes from Godot's per-instance buffer
	godot::RID rs_instance; // RenderingServer instance (for instance_surface_get_vertex_buffer_rd_rid)
};

struct GILight {
	enum Type { DIRECTIONAL = 0, OMNI = 1 };
	Type type;
	godot::Vector3 vec; // DIRECTIONAL: direction light travels; OMNI: position
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
// hands them to its GoghotScreenProbeGI. Phase 3 scope: MeshInstance3D with BaseMaterial3D
// (albedo / emission colors, textures not sampled yet), DirectionalLight3D and OmniLight3D.
class GoghotGIScene : public godot::Node {
	GDCLASS(GoghotGIScene, godot::Node);

	godot::Ref<GoghotScreenProbeGI> gi;
	std::unordered_map<uint64_t, std::shared_ptr<const GIMeshCPU>> mesh_cache;
	struct TextureStats {
		godot::Color average;
		float coverage_at_half;
	};
	std::unordered_map<uint64_t, TextureStats> texture_stats; // per albedo texture RID, computed once
	GIMaterial _material_of(const godot::Ref<godot::Material> &p_mat);

	void _collect(godot::Node *p_node, GISceneSnapshot &r_snap, std::unordered_map<uint64_t, uint32_t> &r_mesh_index);
	std::shared_ptr<const GIMeshCPU> _get_mesh(const godot::Ref<godot::Resource> &p_mesh);

protected:
	static void _bind_methods();

public:
	void _process(double p_delta) override;
	void update_now();

	void set_gi(const godot::Ref<GoghotScreenProbeGI> &p_gi);
	godot::Ref<GoghotScreenProbeGI> get_gi() const;
};

} // namespace goghot
