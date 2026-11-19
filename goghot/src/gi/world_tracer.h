#pragma once

#include "gi_scene.h"

#include <godot_cpp/classes/rendering_device.hpp>
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
	void _update_deformed(const GISceneSnapshot &p_snap, bool p_rt);

	// Per frame.
	godot::RID tlas;
	uint32_t tlas_capacity = 0;
	uint32_t instances_used = 0;
	godot::RID geo_buffer, light_buffer;
	uint32_t geo_capacity = 0, light_capacity = 0;
	uint32_t lights_used = 0;

	void _free_geometry();
	// Tier B state.
	struct VoxDraw {
		godot::Transform3D xform;
		godot::RID positions;
		uint32_t vertex_base, index_base, tri_count;
		float albedo[4];
		float emission[4];
	};
	std::vector<VoxDraw> vox_draws;
	// Voxel attributes (N x N x N*cascades). Two face slots per voxel (0 / 1), see gi_voxel_common.glsl.
	godot::RID vox_occ, vox_key[2], vox_albedo[2], vox_normal[2], vox_emission[2], vox_lit[2][2]; // vox_lit[face][ping]
	int vox_res = 0;
	int vox_ping = 0;
	godot::RID vox_voxelize_shader, vox_voxelize_pipeline, vox_voxelize2_shader, vox_voxelize2_pipeline, vox_light_shader, vox_light_pipeline, vox_rays_shader, vox_rays_pipeline;
	godot::RID vox_sampler;
	bool _init_voxel();
	bool _init_rt();
	void _free_voxel();
	void _ensure_voxel_textures(int p_res);
	void _rebuild_geometry(const GISceneSnapshot &p_snap);
	godot::RID _storage(const godot::PackedByteArray &p_data);
};

} // namespace goghot
