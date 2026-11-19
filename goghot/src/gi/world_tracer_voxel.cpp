// Tier B of GIWorldTracer: voxel clipmap world tracer for GPUs without ray tracing.
// Every frame: clear -> voxelize all instance surfaces (gi_voxelize.glsl) -> light injection with
// voxel-traced shadows and radiance cache bounces (gi_voxel_light.glsl) -> resolve the unresolved
// probe / fog / reflection rays by marching the clipmap (gi_voxel_rays.glsl).
#include "world_tracer.h"

#include <godot_cpp/classes/rd_sampler_state.hpp>
#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/uniform_set_cache_rd.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

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
	const String sources[4] = {
		head + String("#define VOXELIZE_PASS 0\n") + String(GI_VOXELIZE_GLSL),
		head + String("#define VOXELIZE_PASS 1\n") + String(GI_VOXELIZE_GLSL),
		head + vox + String("#define CACHE_BINDING 13\n") + String(GI_CACHE_GLSL) + String(GI_VOXEL_LIGHT_GLSL),
		head + vox + String(GI_VOXEL_RAYS_GLSL),
	};
	RID *shaders[4] = { &vox_voxelize_shader, &vox_voxelize2_shader, &vox_light_shader, &vox_rays_shader };
	RID *pipes[4] = { &vox_voxelize_pipeline, &vox_voxelize2_pipeline, &vox_light_pipeline, &vox_rays_pipeline };
	for (int i = 0; i < 4; i++) {
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
						  &vox_voxelize_shader, &vox_voxelize2_shader, &vox_light_shader, &vox_rays_shader, &vox_sampler });
	vox_voxelize_pipeline = vox_voxelize2_pipeline = vox_light_pipeline = vox_rays_pipeline = RID();
	vox_res = 0;
}

void GIWorldTracer::_ensure_voxel_textures(int p_res) {
	if (p_res == vox_res && vox_occ.is_valid()) {
		return;
	}
	free_rids(rd, { &vox_occ, &vox_key[0], &vox_key[1], &vox_albedo[0], &vox_albedo[1], &vox_normal[0], &vox_normal[1], &vox_emission[0], &vox_emission[1],
						  &vox_lit[0][0], &vox_lit[0][1], &vox_lit[1][0], &vox_lit[1][1] });
	vox_res = p_res;
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
	}
}

void GIWorldTracer::voxel_update(const RID &p_frame_ubo) {
	if (vox_res == 0) {
		return;
	}
	for (const RID &t : { vox_occ, vox_key[0], vox_key[1], vox_albedo[0], vox_albedo[1], vox_normal[0], vox_normal[1], vox_emission[0], vox_emission[1] }) {
		rd->texture_clear(t, Color(0, 0, 0, 0), 0, 1, 0, 1);
	}
	const Ref<RDUniform> ubo = vuni(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, p_frame_ubo);

	// Voxelize, two passes (bid for the face slots, then the winners write): one dispatch per draw.
	for (int pass = 0; pass < 2; pass++) {
		const RID shader = pass == 0 ? vox_voxelize_shader : vox_voxelize2_shader;
		int64_t cl = rd->compute_list_begin();
		rd->compute_list_bind_compute_pipeline(cl, pass == 0 ? vox_voxelize_pipeline : vox_voxelize2_pipeline);
		uint32_t draw_id = 0;
		for (const VoxDraw &d : vox_draws) {
			draw_id++;
			if (!d.positions.is_valid() || d.tri_count == 0) {
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
			struct {
				float x[4], y[4], z[4];
				float albedo[4], emission[4];
				uint32_t vertex_base, index_base, tri_count, draw_id;
			} pc;
			for (int r = 0; r < 3; r++) {
				float *row = r == 0 ? pc.x : (r == 1 ? pc.y : pc.z);
				row[0] = d.xform.basis[r][0];
				row[1] = d.xform.basis[r][1];
				row[2] = d.xform.basis[r][2];
				row[3] = d.xform.origin[r];
			}
			memcpy(pc.albedo, d.albedo, 16);
			memcpy(pc.emission, d.emission, 16);
			pc.vertex_base = d.vertex_base;
			pc.index_base = d.index_base;
			pc.tri_count = d.tri_count;
			pc.draw_id = draw_id;
			PackedByteArray b;
			b.resize(sizeof(pc));
			memcpy(b.ptrw(), &pc, sizeof(pc));
			rd->compute_list_set_push_constant(cl, b, b.size());
			rd->compute_list_dispatch(cl, d.tri_count, 1, 1); // one workgroup per triangle
		}
		rd->compute_list_end();
	}

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
	const int64_t cl = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(cl, vox_rays_pipeline);
	rd->compute_list_bind_uniform_set(cl, UniformSetCacheRD::get_cache(vox_rays_shader, 0, us), 0);
	rd->compute_list_dispatch(cl, (p_ray_count + 63) / 64, 1, 1);
	rd->compute_list_end();
	vox_ping ^= 1;
}

} // namespace goghot
