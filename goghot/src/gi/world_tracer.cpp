#include "world_tracer.h"

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

bool GIWorldTracer::init(RenderingDevice *p_rd, const String &, const String &) {
	rd = p_rd;
	if (rd == nullptr || !rd->has_feature(RD::SUPPORTS_RAYTRACING_PIPELINE)) {
		return false;
	}
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
			return false;
		}
	}
	rt_shader = rd->shader_create_from_spirv(spirv);

	Ref<RDShaderSource> ssrc;
	ssrc.instantiate();
	ssrc->set_stage_source(RD::SHADER_STAGE_MISS, String(RT_HEADER) + String(GI_RT_SHADOW_MISS_GLSL));
	Ref<RDShaderSPIRV> sspirv = rd->shader_compile_spirv_from_source(ssrc);
	if (!sspirv->get_stage_compile_error(RD::SHADER_STAGE_MISS).is_empty()) {
		UtilityFunctions::push_error("Goghot shadow miss shader: ", sspirv->get_stage_compile_error(RD::SHADER_STAGE_MISS));
		return false;
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
		return false;
	}
	hit_sbt = rd->hit_sbt_create(rt_pipeline, 64);

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
	for (const RID &b : blas_index_buffers) {
		rd->free_rid(b);
	}
	blas_index_buffers.clear();
	for (RID *r : { &vertex_buffer, &normal_storage, &index_storage, &uv_storage }) {
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
	_free_geometry();
	for (RID *r : { &tlas, &geo_buffer, &light_buffer, &cache, &cache_resolve_shader, &rt_shader, &shadow_miss_shader, &white_texture, &texture_sampler }) {
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
	if (indices.empty()) {
		return;
	}
	const auto build_input = RD::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT;
	PackedByteArray pos_bytes = bytes_of(positions);
	PackedByteArray idx_bytes = bytes_of(indices);
	vertex_buffer = rd->vertex_buffer_create(pos_bytes.size(), pos_bytes, build_input);
	normal_storage = _storage(bytes_of(normals));
	index_storage = _storage(idx_bytes);
	uv_storage = _storage(bytes_of(uvs));

	for (size_t mi = 0; mi < p_snap.meshes.size(); mi++) {
		const GIMeshCPU &m = *p_snap.meshes[mi];
		TypedArray<RDAccelerationStructureGeometry> geoms;
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
			g->set_index_buffer(ib);
			g->set_index_offset(0);
			g->set_index_count(s.index_count);
			geoms.push_back(g);
		}
		MeshGPU gpu;
		gpu.surface_count = uint32_t(m.surfaces.size());
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

bool GIWorldTracer::update_scene(const GISceneSnapshot &p_snap) {
	if (!ready) {
		return false;
	}
	if (p_snap.mesh_set_hash != mesh_set_hash) {
		_rebuild_geometry(p_snap);
	}
	if (meshes.empty()) {
		instances_used = 0;
		return false;
	}

	// Geometry/material table: one row per instance surface; instance custom index = first row.
	struct GeoRow {
		uint32_t index_base, vertex_base, texture, alpha_scissor_bits;
		float albedo[4];
		float emission[4];
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
		Ref<RDAccelerationStructureInstance> ri;
		ri.instantiate();
		ri->set_blas(meshes[gi.mesh].blas);
		ri->set_transform(gi.xform);
		ri->set_id(uint32_t(rows.size()));
		ri->set_hit_sbt_range(meshes[gi.mesh].sbt_range);
		bool alpha_tested = false;
		for (size_t s = 0; s < m.surfaces.size(); s++) {
			const GIMaterial &mat = gi.materials[s];
			uint32_t ti = 0xFFFFFFFFu;
			if (mat.albedo_texture.is_valid()) {
				auto it = tex_index.find(mat.albedo_texture.get_id());
				if (it != tex_index.end()) {
					ti = it->second;
				} else if (uint32_t(texture_table.size()) < MAX_TEXTURES) {
					RID rdt = RenderingServer::get_singleton()->texture_get_rd_texture(mat.albedo_texture, true);
					if (rdt.is_valid()) {
						ti = uint32_t(texture_table.size());
						texture_table.push_back(rdt);
						tex_index[mat.albedo_texture.get_id()] = ti;
					}
				}
			}
			alpha_tested = alpha_tested || (mat.alpha_scissor >= 0.0f && ti != 0xFFFFFFFFu);
			float scissor = ti != 0xFFFFFFFFu ? mat.alpha_scissor : -1.0f;
			uint32_t scissor_bits;
			memcpy(&scissor_bits, &scissor, 4);
			rows.push_back({ index_bases[gi.mesh] + m.surfaces[s].index_base, vertex_bases[gi.mesh] + m.surfaces[s].vertex_base, ti, scissor_bits,
					{ mat.albedo.r, mat.albedo.g, mat.albedo.b, mat.double_sided ? 1.0f : 0.0f }, { mat.emission.r, mat.emission.g, mat.emission.b, 1.0f } });
		}
		// Alpha tested instances run the any-hit shader; everything else stays on the opaque fast path.
		ri->set_flags(RD::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT |
				(alpha_tested ? RD::ACCELERATION_STRUCTURE_INSTANCE_FORCE_NO_OPAQUE_BIT : RD::ACCELERATION_STRUCTURE_INSTANCE_FORCE_OPAQUE_BIT));
		insts.push_back(ri);
	}
	instances_used = uint32_t(insts.size());
	if (instances_used == 0) {
		return false;
	}
	if (instances_used > tlas_capacity) {
		if (tlas.is_valid()) {
			rd->free_rid(tlas);
		}
		tlas_capacity = MAX(instances_used * 2, 64u);
		tlas = rd->tlas_create(tlas_capacity, RD::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT);
	}
	rd->tlas_build(tlas, insts);

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
	};
	std::vector<LightRow> lrows;
	for (const GILight &l : p_snap.lights) {
		lrows.push_back({ { l.vec.x, l.vec.y, l.vec.z, float(l.type) }, { l.color.r, l.color.g, l.color.b, l.range },
				{ l.attenuation, l.shadow ? 1.0f : 0.0f, 0.0f, 0.0f } });
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
