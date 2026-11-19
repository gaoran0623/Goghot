#include "world_tracer.h"

#include <godot_cpp/classes/time.hpp>

#include <godot_cpp/classes/rd_acceleration_structure_geometry.hpp>
#include <godot_cpp/classes/rd_acceleration_structure_instance.hpp>
#include <godot_cpp/classes/rd_hit_group.hpp>
#include <godot_cpp/classes/rd_pipeline_shader.hpp>
#include <godot_cpp/classes/rd_sampler_state.hpp>
#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/uniform_set_cache_rd.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "gi/gi_cache.glsl.gen.h"
#include "gi/gi_cache_resolve.glsl.gen.h"
#include "gi/gi_common.glsl.gen.h"
#include "gi/gi_rt_ahit.glsl.gen.h"
#include "gi/gi_rt_chit.glsl.gen.h"
#include "gi/gi_rt_common.glsl.gen.h"
#include "gi/gi_rt_miss.glsl.gen.h"
#include "gi/gi_rt_raygen.glsl.gen.h"
#include "gi/gi_rt_shadow_miss.glsl.gen.h"
#include "gi/gi_skin_normals.glsl.gen.h"

using namespace godot;

namespace goghot {

using RD = RenderingDevice;

namespace {

Ref<RDUniform> uni(RD::UniformType p_type, int p_binding, const RID &p_rid) {
	Ref<RDUniform> u;
	u.instantiate();
	u->set_uniform_type(p_type);
	u->set_binding(p_binding);
	u->add_id(p_rid);
	return u;
}

Ref<RDPipelineShader> pipeline_shader(const RID &p_shader) {
	Ref<RDPipelineShader> ps;
	ps.instantiate();
	ps->set_shader(p_shader);
	return ps;
}

template <typename T>
PackedByteArray bytes_of(const std::vector<T> &p_v) {
	PackedByteArray b;
	b.resize(int64_t(p_v.size() * sizeof(T)));
	if (!p_v.empty()) {
		memcpy(b.ptrw(), p_v.data(), p_v.size() * sizeof(T));
	}
	return b;
}

const char *RT_HEADER = "#version 460\n#extension GL_EXT_ray_tracing : require\n#extension GL_EXT_nonuniform_qualifier : require\n";

} // namespace

RID GIWorldTracer::_storage(const PackedByteArray &p_data) {
	return rd->storage_buffer_create(MAX(p_data.size(), 16), p_data);
}

bool GIWorldTracer::_init_rt() {
	const String rt_prefix = String(RT_HEADER) + String(GI_COMMON_GLSL) + "#define CACHE_BINDING 9\n" + String(GI_CACHE_GLSL) + String(GI_RT_COMMON_GLSL);
	Ref<RDShaderSource> src;
	src.instantiate();
	src->set_stage_source(RD::SHADER_STAGE_RAYGEN, rt_prefix + String(GI_RT_RAYGEN_GLSL));
	src->set_stage_source(RD::SHADER_STAGE_MISS, rt_prefix + String(GI_RT_MISS_GLSL));
	src->set_stage_source(RD::SHADER_STAGE_CLOSEST_HIT, rt_prefix + String(GI_RT_CHIT_GLSL));
	src->set_stage_source(RD::SHADER_STAGE_ANY_HIT, rt_prefix + String(GI_RT_AHIT_GLSL));
	Ref<RDShaderSPIRV> spirv = rd->shader_compile_spirv_from_source(src);
	for (RD::ShaderStage st : { RD::SHADER_STAGE_RAYGEN, RD::SHADER_STAGE_MISS, RD::SHADER_STAGE_CLOSEST_HIT, RD::SHADER_STAGE_ANY_HIT }) {
		String err = spirv->get_stage_compile_error(st);
		if (!err.is_empty()) {
			UtilityFunctions::push_error("Goghot world tracer shader stage ", int(st), ": ", err);
			return false; // tier A unavailable
		}
	}
	rt_shader = rd->shader_create_from_spirv(spirv);

	Ref<RDShaderSource> ssrc;
	ssrc.instantiate();
	ssrc->set_stage_source(RD::SHADER_STAGE_MISS, String(RT_HEADER) + String(GI_RT_SHADOW_MISS_GLSL));
	Ref<RDShaderSPIRV> sspirv = rd->shader_compile_spirv_from_source(ssrc);
	if (!sspirv->get_stage_compile_error(RD::SHADER_STAGE_MISS).is_empty()) {
		UtilityFunctions::push_error("Goghot shadow miss shader: ", sspirv->get_stage_compile_error(RD::SHADER_STAGE_MISS));
		return false; // tier A unavailable
	}
	shadow_miss_shader = rd->shader_create_from_spirv(sspirv);

	Ref<RDHitGroup> hg;
	hg.instantiate();
	hg->set_closest_hit_shader(pipeline_shader(rt_shader));
	hg->set_any_hit_shader(pipeline_shader(rt_shader));
	TypedArray<RDPipelineShader> raygens, misses;
	TypedArray<RDHitGroup> hits;
	raygens.push_back(pipeline_shader(rt_shader));
	misses.push_back(pipeline_shader(rt_shader));
	misses.push_back(pipeline_shader(shadow_miss_shader));
	hits.push_back(hg);
	rt_pipeline = rd->raytracing_pipeline_create(raygens, misses, hits, 2);
	if (!rt_pipeline.is_valid()) {
		UtilityFunctions::push_error("Goghot: raytracing_pipeline_create failed");
		return false; // tier A unavailable
	}
	hit_sbt = rd->hit_sbt_create(rt_pipeline, 64);

	return true;
}

bool GIWorldTracer::init(RenderingDevice *p_rd, const String &, const String &) {
	rd = p_rd;
	if (rd == nullptr) {
		return false;
	}
	rt_supported = rd->has_feature(RD::SUPPORTS_RAYTRACING_PIPELINE) && _init_rt();
	Ref<RDShaderSource> csrc;
	csrc.instantiate();
	csrc->set_stage_source(RD::SHADER_STAGE_COMPUTE, String("#version 450\n") + String(GI_COMMON_GLSL) + "#define CACHE_BINDING 1\n" + String(GI_CACHE_GLSL) + String(GI_CACHE_RESOLVE_GLSL));
	Ref<RDShaderSPIRV> cspirv = rd->shader_compile_spirv_from_source(csrc);
	if (!cspirv->get_stage_compile_error(RD::SHADER_STAGE_COMPUTE).is_empty()) {
		UtilityFunctions::push_error("Goghot cache resolve shader: ", cspirv->get_stage_compile_error(RD::SHADER_STAGE_COMPUTE));
		return false;
	}
	cache_resolve_shader = rd->shader_create_from_spirv(cspirv);
	cache_resolve_pipeline = rd->compute_pipeline_create(cache_resolve_shader);

	cache = rd->storage_buffer_create(CACHE_ENTRIES * 48);
	{
		Ref<RDShaderSource> nsrc;
		nsrc.instantiate();
		nsrc->set_stage_source(RD::SHADER_STAGE_COMPUTE, String(GI_SKIN_NORMALS_GLSL));
		Ref<RDShaderSPIRV> nsp = rd->shader_compile_spirv_from_source(nsrc);
		if (!nsp->get_stage_compile_error(RD::SHADER_STAGE_COMPUTE).is_empty()) {
			UtilityFunctions::push_error("Goghot skin normals shader: ", nsp->get_stage_compile_error(RD::SHADER_STAGE_COMPUTE));
			return false;
		}
		skin_normals_shader = rd->shader_create_from_spirv(nsp);
		skin_normals_pipeline = rd->compute_pipeline_create(skin_normals_shader);
	}
	{ // 1x1 white for unused texture slots, linear repeat sampler
		Ref<RDTextureFormat> tf;
		tf.instantiate();
		tf->set_format(RD::DATA_FORMAT_R8G8B8A8_UNORM);
		tf->set_width(1);
		tf->set_height(1);
		tf->set_usage_bits(RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT);
		PackedByteArray px;
		px.resize(4);
		px.fill(255);
		TypedArray<PackedByteArray> data;
		data.push_back(px);
		Ref<RDTextureView> tv;
		tv.instantiate();
		white_texture = rd->texture_create(tf, tv, data);
		Ref<RDSamplerState> ss;
		ss.instantiate();
		ss->set_min_filter(RD::SAMPLER_FILTER_LINEAR);
		ss->set_mag_filter(RD::SAMPLER_FILTER_LINEAR);
		ss->set_repeat_u(RD::SAMPLER_REPEAT_MODE_REPEAT);
		ss->set_repeat_v(RD::SAMPLER_REPEAT_MODE_REPEAT);
		texture_sampler = rd->sampler_create(ss);
	}
	if (!_init_voxel()) {
		return false;
	}
	clear_cache();
	ready = true;
	return true;
}

void GIWorldTracer::clear_cache() {
	if (cache.is_valid()) {
		rd->buffer_clear(cache, 0, CACHE_ENTRIES * 48);
	}
}

void GIWorldTracer::_free_geometry() {
	for (MeshGPU &m : meshes) {
		if (m.blas.is_valid()) {
			rd->free_rid(m.blas);
		}
		if (m.sbt_range >= 0) {
			rd->hit_sbt_range_free(hit_sbt, m.sbt_range);
		}
	}
	meshes.clear();
	// A deformed BLAS depends on Godot's deformed vertex buffer: when that buffer is freed (instance
	// deleted, engine shutdown) RD frees the BLAS with it, and there is no RD query for BLAS validity.
	// So only free the ones whose instance is still alive (see p_alive), leave the rest to RD.
	for (DeformedGPU &d : deformed) {
		if (alive_deformed.count(d.rs_instance.get_id())) {
			for (RID &b : d.slot_blas) {
				if (b.is_valid()) {
					rd->free_rid(b);
				}
			}
		}
	}
	deformed.clear();
	deformed_by_instance.clear();
	for (const RID &b : blas_index_buffers) {
		rd->free_rid(b);
	}
	blas_index_buffers.clear();
	for (RID *r : { &vertex_buffer, &normal_storage, &index_storage, &uv_storage, &position_storage }) {
		if (r->is_valid()) {
			rd->free_rid(*r);
		}
		*r = RID();
	}
	mesh_set_hash = 0;
}

void GIWorldTracer::free() {
	if (rd == nullptr) {
		return;
	}
	alive_deformed.clear(); // shutdown / effect freed: deformed BLASes go with Godot's buffers
	_free_geometry();
	_free_voxel();
	for (RID *r : { &tlas, &geo_buffer, &light_buffer, &cache, &cache_resolve_shader, &rt_shader, &shadow_miss_shader, &white_texture, &texture_sampler, &skin_normals_shader }) {
		if (r->is_valid()) {
			rd->free_rid(*r); // shaders free their pipelines / SBT
		}
		*r = RID();
	}
	rt_pipeline = RID();
	hit_sbt = RID();
	cache_resolve_pipeline = RID();
	tlas_capacity = geo_capacity = light_capacity = 0;
	ready = false;
	rd = nullptr;
}

void GIWorldTracer::_rebuild_geometry(const GISceneSnapshot &p_snap) {
	const bool build_rt = built_tier == TIER_A_RT && rt_supported;
	alive_deformed.clear();
	for (const GIInstance &gi : p_snap.instances) {
		if (gi.deformed) {
			alive_deformed.insert(gi.rs_instance.get_id());
		}
	}
	_free_geometry();
	std::vector<float> positions, normals, uvs;
	std::vector<uint32_t> indices;
	struct Base {
		uint32_t vertex, index;
	};
	std::vector<Base> bases;
	for (const auto &m : p_snap.meshes) {
		bases.push_back({ uint32_t(positions.size() / 3), uint32_t(indices.size()) });
		positions.insert(positions.end(), m->positions.begin(), m->positions.end());
		normals.insert(normals.end(), m->normals.begin(), m->normals.end());
		uvs.insert(uvs.end(), m->uvs.begin(), m->uvs.end());
		indices.insert(indices.end(), m->indices.begin(), m->indices.end());
	}
	// Deformed instances: a private copy of their mesh normals (updated per frame) and uvs.
	for (const GIInstance &gi : p_snap.instances) {
		if (!gi.deformed || deformed_by_instance.count(gi.rs_instance.get_id())) {
			continue;
		}
		const GIMeshCPU &m = *p_snap.meshes[gi.mesh];
		DeformedGPU d;
		d.mesh = gi.mesh;
		d.rs_instance = gi.rs_instance;
		d.region_base = uint32_t(normals.size() / 4);
		normals.insert(normals.end(), m.normals.begin(), m.normals.end());
		uvs.insert(uvs.end(), m.uvs.begin(), m.uvs.end());
		d.sources.resize(m.surfaces.size());
		deformed_by_instance[gi.rs_instance.get_id()] = deformed.size();
		deformed.push_back(d);
	}
	if (indices.empty()) {
		return;
	}
	const auto build_input = RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT;
	PackedByteArray pos_bytes = bytes_of(positions);
	PackedByteArray idx_bytes = bytes_of(indices);
	if (build_rt) {
		vertex_buffer = rd->vertex_buffer_create(pos_bytes.size(), pos_bytes, build_input);
	}
	position_storage = _storage(pos_bytes);
	normal_storage = _storage(bytes_of(normals));
	index_storage = _storage(idx_bytes);
	uv_storage = _storage(bytes_of(uvs));

	for (size_t mi = 0; mi < p_snap.meshes.size(); mi++) {
		const GIMeshCPU &m = *p_snap.meshes[mi];
		if (!build_rt) {
			MeshGPU gpu;
			gpu.surface_count = uint32_t(m.surfaces.size());
			gpu.vertex_base = bases[mi].vertex;
			gpu.index_base = bases[mi].index;
			meshes.push_back(gpu);
			continue;
		}
		TypedArray<RDAccelerationStructureGeometry> geoms;
		std::vector<RID> index_buffers_of_mesh;
		for (const GIMeshCPU::Surface &s : m.surfaces) {
			Ref<RDAccelerationStructureGeometry> g;
			g.instantiate();
			g->set_flags(RD::ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT);
			g->set_vertex_buffer(vertex_buffer);
			g->set_vertex_offset((bases[mi].vertex + s.vertex_base) * 12);
			g->set_vertex_count(s.vertex_count);
			g->set_vertex_stride(12);
			g->set_vertex_format(RD::DATA_FORMAT_R32G32B32_SFLOAT);
			std::vector<uint32_t> sidx(m.indices.begin() + s.index_base, m.indices.begin() + s.index_base + s.index_count);
			RID ib = rd->index_buffer_create(s.index_count, RD::INDEX_BUFFER_FORMAT_UINT32, bytes_of(sidx), false, build_input);
			blas_index_buffers.push_back(ib);
			index_buffers_of_mesh.push_back(ib);
			g->set_index_buffer(ib);
			g->set_index_offset(0);
			g->set_index_count(s.index_count);
			geoms.push_back(g);
		}
		MeshGPU gpu;
		gpu.surface_count = uint32_t(m.surfaces.size());
		gpu.index_buffers = index_buffers_of_mesh;
		gpu.vertex_base = bases[mi].vertex;
		gpu.index_base = bases[mi].index;
		gpu.blas = rd->blas_create(geoms, RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT);
		rd->blas_build(gpu.blas);
		gpu.sbt_range = rd->hit_sbt_range_alloc(hit_sbt, gpu.surface_count);
		PackedInt32Array groups;
		groups.resize(gpu.surface_count);
		groups.fill(0);
		rd->hit_sbt_range_update(hit_sbt, gpu.sbt_range, 0, groups);
		meshes.push_back(gpu);
	}
	mesh_set_hash = p_snap.mesh_set_hash;
}

void GIWorldTracer::_update_deformed(const GISceneSnapshot &p_snap, bool p_rt) {
	if (deformed.empty()) {
		return;
	}
	// Tier A: deformed BLASes are rebuilt round robin within deformed_triangle_budget triangles per
	// frame (always at least one); the others keep their last build for a few frames (GI barely
	// notices). Normals are decoded for the rebuilt ones only (they match their BLAS).
	std::vector<RID> builds;
	uint32_t budget_used = 0;
	uint32_t built = 0;
	const size_t n = deformed.size();
	const size_t start = n > 0 ? deformed_cursor % n : 0;
	size_t last_built = start;
	const int64_t cl = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(cl, skin_normals_pipeline);
	for (size_t k = 0; k < n; k++) {
		const size_t di = (start + k) % n;
		DeformedGPU &d = deformed[di];
		const GIMeshCPU &m = *p_snap.meshes[d.mesh];
		std::vector<RID> src(m.surfaces.size());
		bool all_valid = true;
		for (size_t s = 0; s < m.surfaces.size(); s++) {
			src[s] = RenderingServer::get_singleton()->instance_surface_get_vertex_buffer_rd_rid(d.rs_instance, m.surfaces[s].source_surface);
			all_valid = all_valid && src[s].is_valid();
		}
		if (!all_valid) {
			continue; // not deformed on the GPU yet: the static BLAS is used this frame
		}
		d.triangles = uint32_t(m.indices.size() / 3);
		bool build = true;
		if (!p_rt) {
			d.sources = src;
		} else {
			int slot = d.slot_src[0] == src && d.slot_blas[0].is_valid() ? 0 : (d.slot_src[1] == src && d.slot_blas[1].is_valid() ? 1 : -1);
			if (!deformed_blas_cache && slot < 0) {
				slot = 0; // single BLAS: recreated whenever the buffers change (as before the cache)
				d.next_evict = 0;
			} else if (!deformed_blas_cache) {
				slot = 0;
			}
			if (!deformed_blas_cache && d.slot_src[0] != src) {
				slot = -1;
			}
			if (slot < 0) { // a buffer not seen yet: (re)create a BLAS over it, built this frame
				slot = !deformed_blas_cache ? 0 : (!d.slot_blas[0].is_valid() ? 0 : (!d.slot_blas[1].is_valid() ? 1 : int(d.next_evict)));
				d.next_evict = uint32_t(slot ^ 1);
				if (d.slot_blas[slot].is_valid()) {
					rd->free_rid(d.slot_blas[slot]);
				}
				TypedArray<RDAccelerationStructureGeometry> geoms;
				for (size_t s = 0; s < m.surfaces.size(); s++) {
					Ref<RDAccelerationStructureGeometry> g;
					g.instantiate();
					g->set_flags(RD::ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT);
					g->set_vertex_buffer(src[s]);
					g->set_vertex_offset(0);
					g->set_vertex_count(m.surfaces[s].vertex_count);
					g->set_vertex_stride(12);
					g->set_vertex_format(RD::DATA_FORMAT_R32G32B32_SFLOAT);
					g->set_index_buffer(meshes[d.mesh].index_buffers[s]);
					g->set_index_offset(0);
					g->set_index_count(m.surfaces[s].index_count);
					geoms.push_back(g);
				}
				d.slot_blas[slot] = rd->blas_create(geoms, RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT);
				d.slot_src[slot] = src;
			} else {
				build = !deformed_blas_cache || built == 0 || budget_used + d.triangles <= deformed_triangle_budget;
			}
			d.blas = d.slot_blas[slot];
			d.sources = src;
			if (build) {
				builds.push_back(d.blas);
				budget_used += d.triangles;
				built++;
				last_built = di;
			}
		}
		if (!build) {
			continue;
		}
		for (size_t s = 0; s < m.surfaces.size(); s++) {
			const GIMeshCPU::Surface &surf = m.surfaces[s];
			if (!surf.has_normal) {
				continue;
			}
			TypedArray<RDUniform> us;
			us.push_back(uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 0, src[s]));
			us.push_back(uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, normal_storage));
			rd->compute_list_bind_uniform_set(cl, UniformSetCacheRD::get_cache(skin_normals_shader, 0, us), 0);
			PackedInt32Array pc;
			pc.resize(4);
			pc[0] = int32_t(surf.vertex_count);
			pc[1] = int32_t(surf.vertex_count * 3);
			pc[2] = (surf.has_normal ? 1 : 0) + (surf.has_tangent ? 1 : 0);
			pc[3] = int32_t(d.region_base + surf.vertex_base);
			PackedByteArray pcb = pc.to_byte_array();
			rd->compute_list_set_push_constant(cl, pcb, pcb.size());
			rd->compute_list_dispatch(cl, (surf.vertex_count + 63) / 64, 1, 1);
		}
	}
	rd->compute_list_end();
	for (const RID &b : builds) {
		rd->blas_build(b); // full rebuild (no refit API)
	}
	deformed_cursor = uint32_t(last_built + 1);
	stats.deformed_triangles_built = budget_used;
}

bool GIWorldTracer::update_scene(const GISceneSnapshot &p_snap, bool p_deformed, Tier p_tier) {
	const uint64_t t0 = Time::get_singleton()->get_ticks_usec();
	const bool ok = _update_scene(p_snap, p_deformed, p_tier);
	stats.update_ms = double(Time::get_singleton()->get_ticks_usec() - t0) / 1000.0;
	stats.frames++;
	stats.meshes = uint32_t(p_snap.meshes.size());
	stats.instances = uint32_t(p_snap.instances.size());
	uint64_t tris = 0, dtris = 0;
	uint32_t surfaces = 0, deformed_count = 0;
	for (const GIInstance &gi : p_snap.instances) {
		const GIMeshCPU &m = *p_snap.meshes[gi.mesh];
		surfaces += uint32_t(m.surfaces.size());
		const uint64_t t = m.indices.size() / 3;
		tris += t;
		if (gi.deformed) {
			dtris += t;
			deformed_count++;
		}
	}
	stats.surfaces = surfaces;
	stats.triangles = tris;
	stats.deformed = deformed_count;
	stats.deformed_triangles = dtris;
	return ok;
}

bool GIWorldTracer::_update_scene(const GISceneSnapshot &p_snap, bool p_deformed, Tier p_tier) {
	if (!ready) {
		return false;
	}
	if (p_tier == TIER_A_RT && !rt_supported) {
		p_tier = TIER_B_VOXEL;
	}
	if (p_snap.mesh_set_hash != mesh_set_hash || p_tier != built_tier) {
		built_tier = p_tier;
		const uint64_t r0 = Time::get_singleton()->get_ticks_usec();
		_rebuild_geometry(p_snap);
		stats.rebuild_ms = double(Time::get_singleton()->get_ticks_usec() - r0) / 1000.0;
		stats.rebuilds++;
	}
	const bool rt = built_tier == TIER_A_RT;
	if (meshes.empty()) {
		instances_used = 0;
		return false;
	}
	if (p_deformed) {
		_update_deformed(p_snap, rt);
	}
	vox_draws.clear();

	// Geometry/material table: one row per instance surface; instance custom index = first row.
	struct GeoRow {
		uint32_t index_base, vertex_base, texture, alpha_scissor_bits;
		float albedo[4];
		float emission[4];
		uint32_t emission_texture, emission_multiply;
		float emission_energy, opacity;
	};
	std::vector<GeoRow> rows;
	texture_table.clear();
	std::unordered_map<uint64_t, uint32_t> tex_index;
	TypedArray<RDAccelerationStructureInstance> insts;
	std::vector<uint32_t> vertex_bases, index_bases;
	{
		uint32_t v = 0, i = 0;
		for (const auto &m : p_snap.meshes) {
			vertex_bases.push_back(v);
			index_bases.push_back(i);
			v += uint32_t(m->positions.size() / 3);
			i += uint32_t(m->indices.size());
		}
	}
	for (const GIInstance &gi : p_snap.instances) {
		const GIMeshCPU &m = *p_snap.meshes[gi.mesh];
		// Instance objects are pooled across frames (tens of thousands of allocations per frame otherwise).
		if (insts.size() >= int64_t(inst_pool.size())) {
			Ref<RDAccelerationStructureInstance> fresh;
			fresh.instantiate();
			inst_pool.push_back(fresh);
		}
		const Ref<RDAccelerationStructureInstance> &ri = inst_pool[insts.size()];
		uint32_t vertex_base = vertex_bases[gi.mesh];
		ri->set_blas(meshes[gi.mesh].blas);
		const DeformedGPU *def = nullptr;
		if (gi.deformed && p_deformed) {
			auto it = deformed_by_instance.find(gi.rs_instance.get_id());
			if (it != deformed_by_instance.end() && (rt ? deformed[it->second].blas.is_valid() : !deformed[it->second].sources.empty() && deformed[it->second].sources[0].is_valid())) {
				def = &deformed[it->second];
				if (rt) {
					ri->set_blas(def->blas);
				}
				vertex_base = def->region_base;
			}
		}
		ri->set_transform(gi.xform);
		ri->set_id(uint32_t(rows.size()));
		ri->set_hit_sbt_range(meshes[gi.mesh].sbt_range);
		bool alpha_tested = false;
		for (size_t s = 0; s < m.surfaces.size(); s++) {
			const GIMaterial &mat = (*gi.materials)[s];
			auto slot_of = [&](const RID &p_tex) {
				if (!p_tex.is_valid()) {
					return 0xFFFFFFFFu;
				}
				auto it = tex_index.find(p_tex.get_id());
				if (it != tex_index.end()) {
					return it->second;
				}
				if (uint32_t(texture_table.size()) >= MAX_TEXTURES) {
					return 0xFFFFFFFFu;
				}
				RID rdt = RenderingServer::get_singleton()->texture_get_rd_texture(p_tex, true); // sRGB view: samples linear
				if (!rdt.is_valid()) {
					return 0xFFFFFFFFu;
				}
				const uint32_t slot = uint32_t(texture_table.size());
				texture_table.push_back(rdt);
				tex_index[p_tex.get_id()] = slot;
				return slot;
			};
			const uint32_t ti = slot_of(mat.albedo_texture);
			const uint32_t ei = slot_of(mat.emission_texture);
			alpha_tested = alpha_tested || (mat.alpha_scissor >= 0.0f && ti != 0xFFFFFFFFu) || mat.opacity < 1.0f;
			float scissor = ti != 0xFFFFFFFFu ? mat.alpha_scissor : -1.0f;
			uint32_t scissor_bits;
			memcpy(&scissor_bits, &scissor, 4);
			rows.push_back({ index_bases[gi.mesh] + m.surfaces[s].index_base, vertex_base + m.surfaces[s].vertex_base, ti, scissor_bits,
					{ mat.albedo.r, mat.albedo.g, mat.albedo.b, mat.double_sided ? 1.0f : 0.0f }, { mat.emission.r, mat.emission.g, mat.emission.b, 1.0f },
					ei, mat.emission_multiply ? 1u : 0u, mat.emission_energy, mat.opacity });
			if (!rt) {
				VoxDraw vd;
				vd.xform = gi.xform;
				vd.positions = def ? def->sources[s] : position_storage;
				vd.vertex_base = def ? 0u : vertex_bases[gi.mesh] + m.surfaces[s].vertex_base;
				vd.index_base = index_bases[gi.mesh] + m.surfaces[s].index_base;
				vd.tri_count = m.surfaces[s].index_count / 3;
				Color va = mat.albedo * (ti != 0xFFFFFFFFu ? mat.texture_average : Color(1, 1, 1));
				const float cov = (ti != 0xFFFFFFFFu && mat.alpha_scissor >= 0.0f ? mat.coverage : 1.0f) * mat.opacity; // < 0.5: rays pass
				vd.albedo[0] = va.r;
				vd.albedo[1] = va.g;
				vd.albedo[2] = va.b;
				vd.albedo[3] = cov;
				// Emission texture: its average (like the albedo texture).
				Color ve = mat.emission;
				if (ei != 0xFFFFFFFFu) {
					ve = mat.emission_multiply ? mat.emission * mat.emission_texture_average : mat.emission + mat.emission_texture_average * mat.emission_energy;
				}
				vd.emission[0] = ve.r;
				vd.emission[1] = ve.g;
				vd.emission[2] = ve.b;
				vd.emission[3] = mat.double_sided ? 1.0f : 0.0f;
				vd.aabb = gi.aabb;
				vd.key = (gi.key << 8) ^ uint64_t(s);
				vd.deformed = def != nullptr;
				vox_draws.push_back(vd);
			}
		}
		// Alpha tested instances run the any-hit shader; everything else stays on the opaque fast path.
		ri->set_mask(gi.cast_shadow ? 0x03 : 0x01); // bit 1: shadow caster (shadow rays use mask 0x02)
		ri->set_flags(RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT |
				(alpha_tested ? RD::ACCELERATION_STRUCTURE_INSTANCE_FORCE_NO_OPAQUE_BIT : RD::ACCELERATION_STRUCTURE_INSTANCE_FORCE_OPAQUE_BIT));
		insts.push_back(ri);
	}
	instances_used = uint32_t(insts.size());
	if (instances_used == 0) {
		return false;
	}
	if (rt && instances_used > tlas_capacity) {
		if (tlas.is_valid()) {
			rd->free_rid(tlas);
		}
		tlas_capacity = MAX(instances_used * 2, 64u);
		tlas = rd->tlas_create(tlas_capacity, RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT);
	}
	if (rt) {
		rd->tlas_build(tlas, insts);
	}

	PackedByteArray geo_bytes = bytes_of(rows);
	if (uint32_t(rows.size()) > geo_capacity) {
		if (geo_buffer.is_valid()) {
			rd->free_rid(geo_buffer);
		}
		geo_capacity = MAX(uint32_t(rows.size()) * 2, 64u);
		geo_buffer = rd->storage_buffer_create(geo_capacity * sizeof(GeoRow));
	}
	rd->buffer_update(geo_buffer, 0, geo_bytes.size(), geo_bytes);

	struct LightRow {
		float vec_type[4];
		float color_range[4];
		float params[4];
		float spot[4];
		float area_a[4];
		float area_b[4];
	};
	std::vector<LightRow> lrows;
	for (const GILight &l : p_snap.lights) {
		lrows.push_back({ { l.vec.x, l.vec.y, l.vec.z, float(l.type) }, { l.color.r, l.color.g, l.color.b, l.range },
				{ l.attenuation, l.shadow ? 1.0f : 0.0f, l.spot_exp, 0.0f }, { l.spot_dir.x, l.spot_dir.y, l.spot_dir.z, l.spot_cos },
				{ l.area_a.x, l.area_a.y, l.area_a.z, 0.0f }, { l.area_b.x, l.area_b.y, l.area_b.z, 0.0f } });
	}
	lights_used = uint32_t(lrows.size());
	if (MAX(lights_used, 1u) > light_capacity) {
		if (light_buffer.is_valid()) {
			rd->free_rid(light_buffer);
		}
		light_capacity = MAX(lights_used * 2, 16u);
		light_buffer = rd->storage_buffer_create(light_capacity * sizeof(LightRow));
	}
	if (!lrows.empty()) {
		PackedByteArray lb = bytes_of(lrows);
		rd->buffer_update(light_buffer, 0, lb.size(), lb);
	}
	return true;
}

void GIWorldTracer::trace(const RID &p_frame_ubo, const RID &p_probes, const RID &p_rays, const RID &p_counter, const RID &p_fog_rays, const RID &p_refl, uint32_t p_ray_count) {
	TypedArray<RDUniform> us;
	us.push_back(uni(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, p_frame_ubo));
	us.push_back(uni(RD::UNIFORM_TYPE_ACCELERATION_STRUCTURE, 1, tlas));
	us.push_back(uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 2, p_probes));
	us.push_back(uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 3, p_rays));
	us.push_back(uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 4, p_counter));
	us.push_back(uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 5, geo_buffer));
	us.push_back(uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 6, normal_storage));
	us.push_back(uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 7, index_storage));
	us.push_back(uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 8, light_buffer));
	us.push_back(uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 9, cache));
	us.push_back(uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 10, p_fog_rays));
	us.push_back(uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 11, p_refl));
	us.push_back(uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 12, uv_storage));
	{
		Ref<RDUniform> u;
		u.instantiate();
		u->set_uniform_type(RD::UNIFORM_TYPE_TEXTURE);
		u->set_binding(13);
		for (uint32_t i = 0; i < MAX_TEXTURES; i++) {
			u->add_id(i < uint32_t(texture_table.size()) ? RID(texture_table[i]) : white_texture);
		}
		us.push_back(u);
	}
	us.push_back(uni(RD::UNIFORM_TYPE_SAMPLER, 14, texture_sampler));
	RID uset = UniformSetCacheRD::get_cache(rt_shader, 0, us);
	const int64_t list = rd->raytracing_list_begin();
	rd->raytracing_list_bind_raytracing_pipeline(list, rt_pipeline);
	rd->raytracing_list_bind_uniform_set(list, uset, 0);
	rd->raytracing_list_trace_rays(list, 0, hit_sbt, p_ray_count, 1, 1);
	rd->raytracing_list_end();
}

void GIWorldTracer::resolve_cache(const RID &p_frame_ubo) {
	TypedArray<RDUniform> us;
	us.push_back(uni(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, p_frame_ubo));
	us.push_back(uni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, cache));
	const int64_t cl = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(cl, cache_resolve_pipeline);
	rd->compute_list_bind_uniform_set(cl, UniformSetCacheRD::get_cache(cache_resolve_shader, 0, us), 0);
	rd->compute_list_dispatch(cl, CACHE_ENTRIES / 64, 1, 1);
	rd->compute_list_end();
}

} // namespace goghot
