#pragma once

#include "gi_scene.h"

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace goghot {

// Tier A world tracer (hardware ray tracing) + world-space radiance cache, render thread only.
// Owns the GPU copy of the scene (global vertex/index/normal buffers, one BLAS per mesh, TLAS per
// frame, per instance-surface material table, light table), the RT pipeline and the cache.
class GIWorldTracer {
public:
	static constexpr uint32_t CACHE_ENTRIES = 1u << 19; // 48 bytes each -> 24 MiB

	bool init(godot::RenderingDevice *p_rd, const godot::String &p_rt_header, const godot::String &p_compute_header);
	void free();
	bool is_ready() const { return ready; }

	// Uploads/rebuilds what changed. Returns true when there is something to trace against.
	bool update_scene(const GISceneSnapshot &p_snap, bool p_deformed = true);
	uint32_t light_count() const { return lights_used; }

	// Traces rays[] entries with w == 0 (see gi_rt_raygen.glsl). p_ray_count = probes * 32.
	void trace(const godot::RID &p_frame_ubo, const godot::RID &p_probes, const godot::RID &p_rays, const godot::RID &p_counter, const godot::RID &p_fog_rays, const godot::RID &p_refl, uint32_t p_ray_count);
	void resolve_cache(const godot::RID &p_frame_ubo);
	void clear_cache();

private:
	godot::RenderingDevice *rd = nullptr;
	bool ready = false;

	// Pipelines.
	godot::RID rt_shader, shadow_miss_shader, rt_pipeline, hit_sbt;
	godot::RID cache_resolve_shader, cache_resolve_pipeline;
	godot::RID cache;

	// Scene geometry (rebuilt when the mesh set changes).
	uint64_t mesh_set_hash = 0;
	godot::RID vertex_buffer, normal_storage, index_storage, uv_storage;
	std::vector<godot::RID> blas_index_buffers; // one per surface: RD validates indices against the whole buffer
	static constexpr uint32_t MAX_TEXTURES = 128;
	godot::RID white_texture, texture_sampler;
	godot::TypedArray<godot::RID> texture_table;
	struct MeshGPU {
		godot::RID blas;
		int64_t sbt_range = -1;
		uint32_t surface_count = 0;
		std::vector<godot::RID> index_buffers; // per surface (BLAS input)
		uint32_t vertex_base = 0, index_base = 0; // into the global buffers
	};
	std::vector<MeshGPU> meshes;
	// Deformed instances: own BLAS over Godot's per-instance deformed vertex buffers (engine patch 2),
	// rebuilt every frame; normals decoded into their own region of the normal buffer.
	struct DeformedGPU {
		uint32_t mesh = 0;
		godot::RID rs_instance;
		uint32_t region_base = 0;
		std::vector<godot::RID> sources;
		godot::RID blas;
	};
	std::vector<DeformedGPU> deformed;
	std::unordered_set<uint64_t> alive_deformed; // deformed instances present in the next snapshot
	std::unordered_map<uint64_t, size_t> deformed_by_instance;
	godot::RID skin_normals_shader, skin_normals_pipeline;
	void _update_deformed(const GISceneSnapshot &p_snap);

	// Per frame.
	godot::RID tlas;
	uint32_t tlas_capacity = 0;
	uint32_t instances_used = 0;
	godot::RID geo_buffer, light_buffer;
	uint32_t geo_capacity = 0, light_capacity = 0;
	uint32_t lights_used = 0;

	void _free_geometry();
	void _rebuild_geometry(const GISceneSnapshot &p_snap);
	godot::RID _storage(const godot::PackedByteArray &p_data);
};

} // namespace goghot
