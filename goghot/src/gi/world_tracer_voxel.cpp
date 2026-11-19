// Tier B of GIWorldTracer: voxel clipmap world tracer for GPUs without ray tracing.
// Every frame: incremental voxelization (gi_voxelize.glsl; the toroidal clipmap persists, only boxes
// that changed are cleared and re-voxelized: new slabs after a scroll, moved / deformed / added /
// removed draws) -> light injection with voxel-traced shadows and radiance cache bounces
// (gi_voxel_light.glsl) -> resolve the unresolved probe / fog / reflection rays by marching the
// clipmap (gi_voxel_rays.glsl).
#include "world_tracer.h"

#include <godot_cpp/classes/rd_sampler_state.hpp>
#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/uniform_set_cache_rd.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>

#include "gi/gi_cache.glsl.gen.h"
#include "gi/gi_common.glsl.gen.h"
#include "gi/gi_voxel_common.glsl.gen.h"
#include "gi/gi_voxel_light.glsl.gen.h"
#include "gi/gi_voxel_rays.glsl.gen.h"
#include "gi/gi_voxelize.glsl.gen.h"

using namespace godot;

namespace goghot {

using RD = RenderingDevice;

namespace {

Ref<RDUniform> vuni(RD::UniformType p_type, int p_binding, const RID &p_a, const RID &p_b = RID()) {
	Ref<RDUniform> u;
	u.instantiate();
	u->set_uniform_type(p_type);
	u->set_binding(p_binding);
	u->add_id(p_a);
	if (p_b.is_valid()) {
		u->add_id(p_b);
	}
	return u;
}

} // namespace

bool GIWorldTracer::_init_voxel() {
	const String head = String("#version 450\n") + String(GI_COMMON_GLSL);
	const String vox = String("#define VOX_OCC_BINDING 1\n#define VOX_NORMAL_BINDING 2\n") + String(GI_VOXEL_COMMON_GLSL); // + 3 = second face
	const String sources[5] = {
		head + String("#define VOXELIZE_PASS 0\n") + String(GI_VOXELIZE_GLSL),
		head + String("#define VOXELIZE_PASS 1\n") + String(GI_VOXELIZE_GLSL),
		head + vox + String("#define CACHE_BINDING 13\n") + String(GI_CACHE_GLSL) + String(GI_VOXEL_LIGHT_GLSL),
		head + vox + String(GI_VOXEL_RAYS_GLSL),
		head + String("#define VOXELIZE_PASS 2\n") + String(GI_VOXELIZE_GLSL),
	};
	RID *shaders[5] = { &vox_voxelize_shader, &vox_voxelize2_shader, &vox_light_shader, &vox_rays_shader, &vox_clear_shader };
	RID *pipes[5] = { &vox_voxelize_pipeline, &vox_voxelize2_pipeline, &vox_light_pipeline, &vox_rays_pipeline, &vox_clear_pipeline };
	for (int i = 0; i < 5; i++) {
		Ref<RDShaderSource> src;
		src.instantiate();
		src->set_stage_source(RD::SHADER_STAGE_COMPUTE, sources[i]);
		Ref<RDShaderSPIRV> spirv = rd->shader_compile_spirv_from_source(src);
		String err = spirv->get_stage_compile_error(RD::SHADER_STAGE_COMPUTE);
		if (!err.is_empty()) {
			UtilityFunctions::push_error("Goghot voxel shader ", i, ": ", err);
			return false;
		}
		*shaders[i] = rd->shader_create_from_spirv(spirv);
		*pipes[i] = rd->compute_pipeline_create(*shaders[i]);
	}
	Ref<RDSamplerState> ss;
	ss.instantiate();
	vox_sampler = rd->sampler_create(ss); // nearest, texelFetch only
	return true;
}

static void free_rids(RenderingDevice *rd, std::initializer_list<RID *> p_rids) {
	for (RID *r : p_rids) {
		if (r->is_valid()) {
			rd->free_rid(*r);
		}
		*r = RID();
	}
}

void GIWorldTracer::_free_voxel() {
	if (rd == nullptr) {
		return;
	}
	free_rids(rd, { &vox_occ, &vox_key[0], &vox_key[1], &vox_albedo[0], &vox_albedo[1], &vox_normal[0], &vox_normal[1], &vox_emission[0], &vox_emission[1],
						  &vox_lit[0][0], &vox_lit[0][1], &vox_lit[1][0], &vox_lit[1][1],
						  &vox_voxelize_shader, &vox_voxelize2_shader, &vox_light_shader, &vox_rays_shader, &vox_clear_shader, &vox_sampler });
	vox_voxelize_pipeline = vox_voxelize2_pipeline = vox_light_pipeline = vox_rays_pipeline = vox_clear_pipeline = RID();
	vox_res = 0;
	vox_valid = false;
	vox_prev.clear();
}

void GIWorldTracer::_ensure_voxel_textures(int p_res) {
	if (p_res == vox_res && vox_occ.is_valid()) {
		return;
	}
	free_rids(rd, { &vox_occ, &vox_key[0], &vox_key[1], &vox_albedo[0], &vox_albedo[1], &vox_normal[0], &vox_normal[1], &vox_emission[0], &vox_emission[1],
						  &vox_lit[0][0], &vox_lit[0][1], &vox_lit[1][0], &vox_lit[1][1] });
	vox_res = p_res;
	vox_valid = false; // new (cleared) textures: rebuild everything
	auto make = [&](RD::DataFormat p_format, bool p_atomic) {
		Ref<RDTextureFormat> tf;
		tf.instantiate();
		tf->set_texture_type(RD::TEXTURE_TYPE_3D);
		tf->set_format(p_format);
		tf->set_width(p_res);
		tf->set_height(p_res);
		tf->set_depth(p_res * VOXEL_CASCADES); // cascades stacked along Z
		uint32_t usage = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_UPDATE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
		if (p_atomic) {
			usage |= RD::TEXTURE_USAGE_STORAGE_ATOMIC_BIT;
		}
		tf->set_usage_bits(usage);
		Ref<RDTextureView> tv;
		tv.instantiate();
		RID t = rd->texture_create(tf, tv);
		rd->texture_clear(t, Color(0, 0, 0, 0), 0, 1, 0, 1);
		return t;
	};
	vox_occ = make(RD::DATA_FORMAT_R32_UINT, true);
	for (int f = 0; f < 2; f++) {
		vox_key[f] = make(RD::DATA_FORMAT_R32_UINT, true);
		vox_albedo[f] = make(RD::DATA_FORMAT_R8G8B8A8_UNORM, false);
		vox_normal[f] = make(RD::DATA_FORMAT_R8G8B8A8_UNORM, false);
		vox_emission[f] = make(RD::DATA_FORMAT_R16G16B16A16_SFLOAT, false);
		vox_lit[f][0] = make(RD::DATA_FORMAT_R16G16B16A16_SFLOAT, false);
		vox_lit[f][1] = make(RD::DATA_FORMAT_R16G16B16A16_SFLOAT, false);
	}
}

void GIWorldTracer::voxel_setup(const Vector3 &p_camera, int p_resolution, float p_finest_size, float r_origin[VOXEL_CASCADES][4]) {
	_ensure_voxel_textures(p_resolution);
	for (int c = 0; c < VOXEL_CASCADES; c++) {
		const float size = p_finest_size * float(1 << (2 * c)); // x4 per cascade
		const Vector3 cell = (p_camera / size).floor();
		const Vector3 origin = (cell - Vector3(p_resolution / 2, p_resolution / 2, p_resolution / 2)) * size;
		r_origin[c][0] = origin.x;
		r_origin[c][1] = origin.y;
		r_origin[c][2] = origin.z;
		r_origin[c][3] = size;
		for (int k = 0; k < 4; k++) {
			vox_origin_cur[c][k] = r_origin[c][k];
		}
	}
}

bool GIWorldTracer::VoxDraw::same_as(const VoxDraw &p_o) const {
	return !deformed && !p_o.deformed && xform == p_o.xform && positions == p_o.positions && vertex_base == p_o.vertex_base && index_base == p_o.index_base &&
			tri_count == p_o.tri_count && memcmp(albedo, p_o.albedo, sizeof(albedo)) == 0 && memcmp(emission, p_o.emission, sizeof(emission)) == 0;
}

void GIWorldTracer::_voxel_dispatch_pc(int64_t p_list, const VoxDraw *p_draw, uint32_t p_draw_id, const int p_lo[4], const int p_hi[4]) {
	struct {
		float x[4], y[4], z[4];
		float albedo[4], emission[4];
		uint32_t vertex_base, index_base, tri_count, draw_id;
		int32_t box_lo[4], box_hi[4];
	} pc = {};
	if (p_draw != nullptr) {
		for (int r = 0; r < 3; r++) {
			float *row = r == 0 ? pc.x : (r == 1 ? pc.y : pc.z);
			row[0] = p_draw->xform.basis[r][0];
			row[1] = p_draw->xform.basis[r][1];
			row[2] = p_draw->xform.basis[r][2];
			row[3] = p_draw->xform.origin[r];
		}
		memcpy(pc.albedo, p_draw->albedo, 16);
		memcpy(pc.emission, p_draw->emission, 16);
		pc.vertex_base = p_draw->vertex_base;
		pc.index_base = p_draw->index_base;
		pc.tri_count = p_draw->tri_count;
		pc.draw_id = p_draw_id;
	}
	memcpy(pc.box_lo, p_lo, 16);
	memcpy(pc.box_hi, p_hi, 16);
	PackedByteArray b;
	b.resize(sizeof(pc));
	memcpy(b.ptrw(), &pc, sizeof(pc));
	rd->compute_list_set_push_constant(p_list, b, b.size());
}

void GIWorldTracer::voxel_update(const RID &p_frame_ubo) {
	if (vox_res == 0) {
		return;
	}
	rd->capture_timestamp("goghot_gi.vox_build_begin");
	const int N = vox_res;
	struct Box {
		int lo[4]; // xyz first voxel (cascade local), w cascade
		int hi[4]; // xyz last voxel, w: 1 = clear the light history too
	};
	std::vector<Box> boxes;
	bool full[VOXEL_CASCADES];
	for (int c = 0; c < VOXEL_CASCADES; c++) {
		const float size = vox_origin_cur[c][3];
		full[c] = !vox_valid || size != vox_origin_prev[c][3];
		int shift[3] = {};
		for (int k = 0; k < 3 && !full[c]; k++) {
			shift[k] = int(std::lround((vox_origin_cur[c][k] - vox_origin_prev[c][k]) / size));
			full[c] = std::abs(shift[k]) >= N;
		}
		if (full[c]) {
			boxes.push_back({ { 0, 0, 0, c }, { N - 1, N - 1, N - 1, 1 } });
			continue;
		}
		// Scroll: the slabs that entered the clipmap (stale texels of the far side, light history too).
		for (int k = 0; k < 3; k++) {
			if (shift[k] == 0) {
				continue;
			}
			Box bx = { { 0, 0, 0, c }, { N - 1, N - 1, N - 1, 1 } };
			if (shift[k] > 0) {
				bx.lo[k] = N - shift[k];
			} else {
				bx.hi[k] = -shift[k] - 1;
			}
			boxes.push_back(bx);
		}
	}
	// Changed draws: dirty their old and new bounds (light history kept, so bounces near movers stay).
	std::vector<AABB> dirty;
	std::unordered_map<uint64_t, VoxDraw> now;
	now.reserve(vox_draws.size());
	for (const VoxDraw &d : vox_draws) {
		now[d.key] = d;
		auto it = vox_prev.find(d.key);
		if (it == vox_prev.end()) {
			dirty.push_back(d.aabb);
		} else if (!d.same_as(it->second)) {
			dirty.push_back(d.aabb);
			dirty.push_back(it->second.aabb);
		}
	}
	for (const auto &kv : vox_prev) {
		if (now.find(kv.first) == now.end()) {
			dirty.push_back(kv.second.aabb);
		}
	}
	vox_prev = std::move(now);
	for (int c = 0; c < VOXEL_CASCADES; c++) {
		if (full[c] || dirty.empty()) {
			continue;
		}
		const Vector3 org(vox_origin_cur[c][0], vox_origin_cur[c][1], vox_origin_cur[c][2]);
		const float size = vox_origin_cur[c][3];
		std::vector<Box> cb;
		for (const AABB &a : dirty) {
			Box bx = { { 0, 0, 0, c }, { 0, 0, 0, 0 } };
			bool empty = false;
			for (int k = 0; k < 3; k++) {
				// 2 voxels of margin: conservative voxelization touches up to 1 voxel past the bounds.
				bx.lo[k] = std::max(int(std::floor((a.position[k] - org[k]) / size)) - 2, 0);
				bx.hi[k] = std::min(int(std::floor((a.position[k] + a.size[k] - org[k]) / size)) + 2, N - 1);
				empty = empty || bx.lo[k] > bx.hi[k];
			}
			if (!empty) {
				cb.push_back(bx);
			}
		}
		if (cb.size() > 8) { // many movers: one union box per cascade
			Box u = cb[0];
			for (const Box &bx : cb) {
				for (int k = 0; k < 3; k++) {
					u.lo[k] = std::min(u.lo[k], bx.lo[k]);
					u.hi[k] = std::max(u.hi[k], bx.hi[k]);
				}
			}
			cb.assign(1, u);
		}
		boxes.insert(boxes.end(), cb.begin(), cb.end());
	}
	memcpy(vox_origin_prev, vox_origin_cur, sizeof(vox_origin_prev));
	vox_valid = true;
	vox_boxes_last = int(boxes.size());

	const Ref<RDUniform> ubo = vuni(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, p_frame_ubo);
	if (!boxes.empty()) {
		// Clear the boxes (all first: every cleared voxel is then rebuilt from all draws touching it).
		TypedArray<RDUniform> us;
		us.push_back(ubo);
		us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 3, vox_occ));
		us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 4, vox_key[0]));
		us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 5, vox_key[1]));
		us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 6, vox_albedo[0]));
		us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 7, vox_albedo[1]));
		us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 8, vox_normal[0]));
		us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 9, vox_normal[1]));
		us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 10, vox_emission[0]));
		us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 11, vox_emission[1]));
		us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 12, vox_lit[0][0]));
		us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 13, vox_lit[0][1]));
		us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 14, vox_lit[1][0]));
		us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 15, vox_lit[1][1]));
		int64_t cl = rd->compute_list_begin();
		rd->compute_list_bind_compute_pipeline(cl, vox_clear_pipeline);
		rd->compute_list_bind_uniform_set(cl, UniformSetCacheRD::get_cache(vox_clear_shader, 0, us), 0);
		for (const Box &bx : boxes) {
			const uint64_t count = uint64_t(bx.hi[0] - bx.lo[0] + 1) * uint64_t(bx.hi[1] - bx.lo[1] + 1) * uint64_t(bx.hi[2] - bx.lo[2] + 1);
			_voxel_dispatch_pc(cl, nullptr, 0, bx.lo, bx.hi);
			const uint32_t groups = uint32_t((count + 63) / 64);
			rd->compute_list_dispatch(cl, std::min(groups, 4096u), (groups + 4095) / 4096, 1);
		}
		rd->compute_list_end();
	}

	// Voxelize the boxes, two passes (bid for the face slots, then the winners write): one dispatch
	// per (box, draw overlapping it).
	for (int pass = 0; pass < 2 && !boxes.empty(); pass++) {
		const RID shader = pass == 0 ? vox_voxelize_shader : vox_voxelize2_shader;
		int64_t cl = rd->compute_list_begin();
		rd->compute_list_bind_compute_pipeline(cl, pass == 0 ? vox_voxelize_pipeline : vox_voxelize2_pipeline);
		for (const Box &bx : boxes) {
			const int c = bx.lo[3];
			const float size = vox_origin_cur[c][3];
			const Vector3 lo = Vector3(vox_origin_cur[c][0], vox_origin_cur[c][1], vox_origin_cur[c][2]) + Vector3(bx.lo[0], bx.lo[1], bx.lo[2]) * size;
			const AABB box_world(lo - Vector3(size, size, size), Vector3(bx.hi[0] - bx.lo[0] + 3, bx.hi[1] - bx.lo[1] + 3, bx.hi[2] - bx.lo[2] + 3) * size);
			uint32_t draw_id = 0;
			for (const VoxDraw &d : vox_draws) {
				draw_id++;
				if (!d.positions.is_valid() || d.tri_count == 0 || !d.aabb.intersects(box_world)) {
					continue;
				}
				TypedArray<RDUniform> us;
				us.push_back(ubo);
				us.push_back(vuni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 1, d.positions));
				us.push_back(vuni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 2, index_storage));
				us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 3, vox_occ));
				us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 4, vox_key[0]));
				us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 5, vox_key[1]));
				us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 6, vox_albedo[0]));
				us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 7, vox_albedo[1]));
				us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 8, vox_normal[0]));
				us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 9, vox_normal[1]));
				us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 10, vox_emission[0]));
				us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 11, vox_emission[1]));
				rd->compute_list_bind_uniform_set(cl, UniformSetCacheRD::get_cache(shader, 0, us), 0);
				_voxel_dispatch_pc(cl, &d, draw_id, bx.lo, bx.hi);
				rd->compute_list_dispatch(cl, d.tri_count, 1, 1); // one workgroup per triangle
			}
		}
		rd->compute_list_end();
	}
	rd->capture_timestamp("goghot_gi.vox_build_end");
	rd->capture_timestamp("goghot_gi.vox_light_begin");

	// Light injection per face (reads last frame's lit volumes for the bounce, writes this frame's).
	TypedArray<RDUniform> us;
	us.push_back(ubo);
	us.push_back(vuni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, vox_sampler, vox_occ));
	us.push_back(vuni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 2, vox_sampler, vox_normal[0]));
	us.push_back(vuni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 3, vox_sampler, vox_normal[1]));
	us.push_back(vuni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 4, vox_sampler, vox_albedo[0]));
	us.push_back(vuni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 5, vox_sampler, vox_albedo[1]));
	us.push_back(vuni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 6, vox_sampler, vox_emission[0]));
	us.push_back(vuni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 7, vox_sampler, vox_emission[1]));
	us.push_back(vuni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 8, vox_sampler, vox_lit[0][vox_ping ^ 1]));
	us.push_back(vuni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 9, vox_sampler, vox_lit[1][vox_ping ^ 1]));
	us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 10, vox_lit[0][vox_ping]));
	us.push_back(vuni(RD::UNIFORM_TYPE_IMAGE, 11, vox_lit[1][vox_ping]));
	us.push_back(vuni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 12, light_buffer));
	us.push_back(vuni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 13, cache));
	const int64_t cl = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(cl, vox_light_pipeline);
	rd->compute_list_bind_uniform_set(cl, UniformSetCacheRD::get_cache(vox_light_shader, 0, us), 0);
	rd->compute_list_dispatch(cl, (vox_res + 3) / 4, (vox_res + 3) / 4, (vox_res * VOXEL_CASCADES + 3) / 4);
	rd->compute_list_end();
	rd->capture_timestamp("goghot_gi.vox_light_end");
}

void GIWorldTracer::voxel_trace(const RID &p_frame_ubo, const RID &p_probes, const RID &p_rays, const RID &p_counter, const RID &p_fog_rays, const RID &p_refl, uint32_t p_ray_count) {
	if (vox_res == 0) {
		return;
	}
	TypedArray<RDUniform> us;
	us.push_back(vuni(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, p_frame_ubo));
	us.push_back(vuni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, vox_sampler, vox_occ));
	us.push_back(vuni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 2, vox_sampler, vox_normal[0]));
	us.push_back(vuni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 3, vox_sampler, vox_normal[1]));
	us.push_back(vuni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 5, vox_sampler, vox_lit[0][vox_ping]));
	us.push_back(vuni(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 6, vox_sampler, vox_lit[1][vox_ping]));
	us.push_back(vuni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 8, p_probes));
	us.push_back(vuni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 9, p_rays));
	us.push_back(vuni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 10, p_counter));
	us.push_back(vuni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 11, p_fog_rays));
	us.push_back(vuni(RD::UNIFORM_TYPE_STORAGE_BUFFER, 12, p_refl));
	rd->capture_timestamp("goghot_gi.vox_trace_begin");
	const int64_t cl = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(cl, vox_rays_pipeline);
	rd->compute_list_bind_uniform_set(cl, UniformSetCacheRD::get_cache(vox_rays_shader, 0, us), 0);
	rd->compute_list_dispatch(cl, (p_ray_count + 63) / 64, 1, 1);
	rd->compute_list_end();
	rd->capture_timestamp("goghot_gi.vox_trace_end");
	vox_ping ^= 1;
}

} // namespace goghot
