#pragma once

#include "gi_scene.h"

#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rd_acceleration_structure_instance.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace goghot {

// World tracer + world-space radiance cache, render thread only.
// Tier A: hardware ray tracing (BLAS per mesh, TLAS per frame, RT pipeline).
// Tier B: voxel clipmap (re-voxelized every frame) traced with a DDA in compute (world_tracer_voxel.cpp),
// for GPUs without ray tracing.
// Both share the GPU copy of the scene (global index/normal/uv/position buffers, per instance-surface
// material table, light table) and the radiance cache.
class GIWorldTracer {
public:
	// Per frame statistics (render thread, read by GoghotScreenProbeGI::get_stats).
	struct Stats {
		double update_ms = 0.0; // CPU time of update_scene
		double deformed_ms = 0.0, lights_ms = 0.0; // update_scene phases
		double rebuild_ms = 0.0; // CPU time of the last geometry rebuild
		uint64_t rebuilds = 0; // geometry rebuilds (BLAS / buffers) since start
		uint64_t frames = 0;
		uint32_t instances = 0, surfaces = 0, meshes = 0, deformed = 0;
		uint64_t triangles = 0, deformed_triangles = 0; // per frame: all instances / deformed
		uint64_t deformed_triangles_built = 0; // deformed BLAS triangles rebuilt this frame (budget)
		uint32_t last_rebuild_added = 0, last_rebuild_removed = 0; // meshes vs the previous build
		bool last_rebuild_deformed_changed = false;
		double rows_ms = 0.0, tlas_ms = 0.0, upload_ms = 0.0; // update_scene phases
		uint64_t rows_fast = 0, rows_full = 0; // fast / full rows builds since start
		uint32_t last_rebuild_new_blas = 0; // BLASes built by the last rebuild (others reused)
	};
	Stats stats;
	std::vector<uint64_t> built_mesh_ids, built_deformed_ids; // sorted, of the current build (diagnostics)
	enum Tier {
		TIER_A_RT,
		TIER_B_VOXEL,
	};
	static constexpr int VOXEL_CASCADES = 3;
	static constexpr uint32_t CACHE_ENTRIES = 1u << 19; // 48 bytes each -> 24 MiB

	bool init(godot::RenderingDevice *p_rd, const godot::String &p_rt_header, const godot::String &p_compute_header);
	void free();
	bool is_ready() const { return ready; }
	bool is_rt_supported() const { return rt_supported; }

	// Uploads/rebuilds what changed. Returns true when there is something to trace against.
	bool update_scene(const GISceneSnapshot &p_snap, bool p_deformed, Tier p_tier);
	uint32_t light_count() const { return lights_used; }

	// Traces rays[] entries with w == 0 (see gi_rt_raygen.glsl). p_ray_count = probes * 32.
	void trace(const godot::RID &p_frame_ubo, const godot::RID &p_probes, const godot::RID &p_rays, const godot::RID &p_counter, const godot::RID &p_fog_rays, const godot::RID &p_refl, uint32_t p_ray_count);
	void resolve_cache(const godot::RID &p_frame_ubo);
	// Tier B. voxel_setup() fills the clipmap placement (camera-centered, voxel snapped) for the UBO;
	// voxel_update() voxelizes + injects light (needs the UBO), voxel_trace() resolves the rays.
	void voxel_setup(const godot::Vector3 &p_camera, int p_resolution, float p_finest_size, float r_origin[VOXEL_CASCADES][4]);
	void voxel_update(const godot::RID &p_frame_ubo);
	void voxel_trace(const godot::RID &p_frame_ubo, const godot::RID &p_probes, const godot::RID &p_rays, const godot::RID &p_counter, const godot::RID &p_fog_rays, const godot::RID &p_refl, uint32_t p_ray_count);
	void clear_cache();
	// Debug view: 0 occupancy, 1/2 face normals, 3/4 face lit (the volume the last trace used).
	godot::RID voxel_debug_texture(int p_which) const {
		switch (p_which) {
			case 0:
				return vox_occ;
			case 1:
				return vox_normal[0];
			case 2:
				return vox_normal[1];
			case 3:
				return vox_lit[0][vox_ping ^ 1];
			default:
				return vox_lit[1][vox_ping ^ 1];
		}
	}

private:
	godot::RenderingDevice *rd = nullptr;
	bool ready = false;
	bool rt_supported = false;
	Tier built_tier = TIER_A_RT;

	// Pipelines.
	godot::RID rt_shader, shadow_miss_shader, rt_pipeline, hit_sbt;
	godot::RID cache_resolve_shader, cache_resolve_pipeline;
	godot::RID cache;

	// Scene geometry (rebuilt when the mesh set changes).
	uint64_t mesh_set_hash = 0;
	godot::RID vertex_buffer, normal_storage, index_storage, uv_storage;
	godot::RID position_storage; // packed float3, bindable as storage (voxelization)
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
	// Tier A: BLAS (+ its own vertex / index buffers and SBT range) per mesh id, kept across geometry
	// rebuilds: a rebuild (mesh added / removed, e.g. streaming) only builds the new meshes' BLASes.
	struct CachedMesh {
		godot::RID vertex_buffer;
		std::vector<godot::RID> index_buffers;
		godot::RID blas;
		int64_t sbt_range = -1;
		uint32_t surface_count = 0;
		uint64_t last_used = 0; // stats.frames when last in a build
	};
	std::unordered_map<uint64_t, CachedMesh> mesh_gpu_cache;
	void _free_cached_mesh(CachedMesh &p_mesh);
	void _free_mesh_cache();
	// Deformed instances: own BLAS over Godot's per-instance deformed vertex buffers (engine patch 2),
	// rebuilt every frame; normals decoded into their own region of the normal buffer.
	struct DeformedGPU {
		uint32_t mesh = 0; // index into the snapshot's meshes (updated by every rebuild)
		uint64_t mesh_id = 0;
		godot::RID rs_instance;
		uint32_t region_base = 0;
		std::vector<godot::RID> sources; // vertex buffers the active BLAS / normals were built from
		godot::RID blas; // active (= slot_blas[k] whose sources match)
		// Godot alternates two deformed buffers per instance when motion vectors are on: one BLAS per
		// buffer, so a swap reuses a BLAS instead of recreating it.
		godot::RID slot_blas[2];
		std::vector<godot::RID> slot_src[2];
		uint32_t next_evict = 0;
		uint32_t triangles = 0;
	};
	uint32_t deformed_cursor = 0; // round robin start of the BLAS rebuild budget
	// TLAS instances as RenderingDevice.tlas_build_packed takes them (engine patch 5), kept across frames:
	// the fast path only rewrites the instances that changed, and builds without creating objects.
	struct TlasInstance {
		float xform[12]; // basis rows, origin
		uint32_t id;
		uint32_t mask;
		int64_t hit_sbt_range;
		uint32_t flags;
		uint32_t pad;
		uint64_t blas;
	};
	static_assert(sizeof(TlasInstance) == 80, "tlas_build_packed layout");
	std::vector<TlasInstance> tlas_instances;
	godot::PackedByteArray tlas_bytes;
	static void _pack_xform(const godot::Transform3D &p_xform, float r_out[12]);
	bool tlas_dirty = true; // TLAS must be built even if no instance changed (new TLAS, full rows build)
	std::unordered_map<uint64_t, godot::RID> rd_texture_cache; // texture RID -> RD texture (sRGB view)
	uint64_t rows_signature = 0; // instance structure of the last full rows build (fast path when equal)
	size_t cached_instance_count = 0;
	void _update_lights(const GISceneSnapshot &p_snap);
	uint32_t deformed_triangle_budget = 250000; // deformed BLAS triangles rebuilt per frame (round robin)
	bool deformed_blas_cache = true; // false: one BLAS per deformed instance, recreated on buffer swaps and rebuilt every frame (pre-cache behavior)
public:
	void set_deformed_blas_cache(bool p_enable, uint32_t p_budget) {
		deformed_blas_cache = p_enable;
		deformed_triangle_budget = p_budget;
	}
private:
	std::vector<DeformedGPU> deformed;
	std::unordered_set<uint64_t> alive_deformed; // deformed instances present in the next snapshot
	std::unordered_map<uint64_t, size_t> deformed_by_instance;
	godot::RID skin_normals_shader, skin_normals_pipeline;
	void _update_deformed(const GISceneSnapshot &p_snap, bool p_rt);

	// Per frame.
	godot::RID tlas;
	uint32_t tlas_capacity = 0;
	uint32_t instances_used = 0;
	godot::RID geo_buffer, light_buffer;
	uint32_t geo_capacity = 0, light_capacity = 0;
	uint32_t lights_used = 0;

	void _free_geometry();
	// Global vertex / index buffers as arenas: each mesh (and each deformed instance's private normal / uv
	// region) keeps its range across rebuilds, so a mesh streamed in only uploads its own data. Ranges of
	// meshes no longer in the scene stay allocated for BLAS_EVICT_FRAMES (they may come back), and the
	// arenas are repacked from scratch only when they are full or more than half holes.
	struct Range {
		uint32_t vbase = 0, vcount = 0, ibase = 0, icount = 0;
		uint64_t last_used = 0;
	};
	std::unordered_map<uint64_t, Range> mesh_ranges; // mesh id
	std::unordered_map<uint64_t, Range> deformed_ranges; // RS instance id (vertex region only)
	uint32_t vtx_used = 0, idx_used = 0, vtx_cap = 0, idx_cap = 0;
	uint64_t repacks = 0;
	void _arena_alloc(Range &r_range, uint32_t p_vcount, uint32_t p_icount);
	void _arena_upload(const Range &p_range, const GIMeshCPU &p_mesh, bool p_geometry);

public:
	uint64_t get_repacks() const { return repacks; }

private:
	// Tier B state.
	struct VoxDraw {
		godot::Transform3D xform;
		godot::RID positions;
		uint32_t vertex_base, index_base, tri_count;
		float albedo[4];
		float emission[4];
		godot::AABB aabb; // world bounds
		uint64_t key = 0; // instance + surface, stable across frames
		bool deformed = false; // vertices change every frame
		bool same_as(const VoxDraw &p_o) const;
	};
	std::vector<VoxDraw> vox_draws;
	// Incremental voxelization: last frame's draws (by key) and clipmap origins.
	std::unordered_map<uint64_t, VoxDraw> vox_prev;
	float vox_origin_cur[3][4] = {};
	float vox_origin_prev[3][4] = {};
	bool vox_valid = false; // volume content matches vox_origin_prev / vox_prev
	int vox_boxes_last = 0; // boxes rebuilt last frame (stats)
	godot::RID vox_clear_shader, vox_clear_pipeline;
	// Voxel attributes (N x N x N*cascades). Two face slots per voxel (0 / 1), see gi_voxel_common.glsl.
	godot::RID vox_occ, vox_key[2], vox_albedo[2], vox_normal[2], vox_emission[2], vox_ecov, vox_lit[2][2]; // vox_lit[face][ping]
	int vox_res = 0;
	int vox_ping = 0;
	godot::RID vox_voxelize_shader, vox_voxelize_pipeline, vox_voxelize2_shader, vox_voxelize2_pipeline, vox_light_shader, vox_light_pipeline, vox_rays_shader, vox_rays_pipeline;
	godot::RID vox_sampler;
	bool _init_voxel();
	bool _init_rt();
	void _free_voxel();
	void _ensure_voxel_textures(int p_res);
	void _voxel_dispatch_pc(int64_t p_list, const VoxDraw *p_draw, uint32_t p_draw_id, const int p_lo[4], const int p_hi[4]);
	void _rebuild_geometry(const GISceneSnapshot &p_snap);
	bool _update_scene(const GISceneSnapshot &p_snap, bool p_deformed, Tier p_tier);
	godot::RID _storage(const godot::PackedByteArray &p_data);
};

} // namespace goghot
