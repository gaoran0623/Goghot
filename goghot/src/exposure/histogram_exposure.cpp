#include "histogram_exposure.h"

#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/render_data.hpp>
#include <godot_cpp/classes/render_scene_buffers_rd.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/classes/uniform_set_cache_rd.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "exposure/exposure_apply.glsl.gen.h"
#include "exposure/exposure_common.glsl.gen.h"
#include "exposure/exposure_histogram.glsl.gen.h"
#include "exposure/exposure_reduce.glsl.gen.h"

using namespace godot;

namespace goghot {

enum { PASS_HIST, PASS_REDUCE, PASS_APPLY };

static Ref<RDUniform> make_uniform(RenderingDevice::UniformType p_type, int p_binding, const RID &p_rid) {
	Ref<RDUniform> u;
	u.instantiate();
	u->set_uniform_type(p_type);
	u->set_binding(p_binding);
	u->add_id(p_rid);
	return u;
}

void GoghotHistogramExposure::_create_resources() {
	rd = RenderingServer::get_singleton()->get_rendering_device();
	if (rd == nullptr) {
		return; // Compatibility renderer: effect does nothing.
	}
	const char *bodies[3] = { EXPOSURE_HISTOGRAM_GLSL, EXPOSURE_REDUCE_GLSL, EXPOSURE_APPLY_GLSL };
	for (int i = 0; i < 3; i++) {
		Ref<RDShaderSource> src;
		src.instantiate();
		src->set_stage_source(RenderingDevice::SHADER_STAGE_COMPUTE, String(EXPOSURE_COMMON_GLSL) + String(bodies[i]));
		Ref<RDShaderSPIRV> spirv = rd->shader_compile_spirv_from_source(src);
		String err = spirv->get_stage_compile_error(RenderingDevice::SHADER_STAGE_COMPUTE);
		if (!err.is_empty()) {
			UtilityFunctions::push_error("Goghot GoghotHistogramExposure shader ", i, ": ", err);
			return;
		}
		shaders[i] = rd->shader_create_from_spirv(spirv);
		pipelines[i] = rd->compute_pipeline_create(shaders[i]);
	}
	hist_buffer = rd->storage_buffer_create(BINS * 4);
	PackedFloat32Array state;
	state.resize(4);
	state.fill(0.0f);
	state_buffer = rd->storage_buffer_create(16, state.to_byte_array());
	ready = true;
}

void GoghotHistogramExposure::_free_resources() {
	if (rd == nullptr) {
		return;
	}
	for (int i = 0; i < 3; i++) {
		if (shaders[i].is_valid()) {
			rd->free_rid(shaders[i]); // Also frees the dependent pipeline.
		}
		shaders[i] = RID();
		pipelines[i] = RID();
	}
	for (RID *b : { &hist_buffer, &state_buffer }) {
		if (b->is_valid()) {
			rd->free_rid(*b);
		}
		*b = RID();
	}
	ready = false;
	rd = nullptr;
}

PackedByteArray GoghotHistogramExposure::_push_constants(const Vector2i &p_size, float p_dt) const {
	// Layout must match Params in exposure_common.glsl (80 bytes).
	struct {
		int32_t size[2];
		float min_log2, max_log2, low_pct, high_pct, key, comp_ev, min_ev, max_ev;
		float speed_up, speed_down, dt, center_weight, center_sigma, snap;
		int32_t stride, pad0, pad1, pad2;
	} pc = {
		{ p_size.x, p_size.y },
		MIN_LOG2, MAX_LOG2, low_percentile, high_percentile, key, compensation_ev, min_ev, max_ev,
		speed_up, speed_down, p_dt, center_weight, center_sigma, snap ? 1.0f : 0.0f,
		METER_STRIDE, 0, 0, 0
	};
	static_assert(sizeof(pc) == 80, "push constant layout");
	PackedByteArray bytes;
	bytes.resize(sizeof(pc));
	memcpy(bytes.ptrw(), &pc, sizeof(pc));
	return bytes;
}

void GoghotHistogramExposure::_render_callback(int32_t p_effect_callback_type, RenderData *p_render_data) {
	if (p_effect_callback_type != EFFECT_CALLBACK_TYPE_POST_TRANSPARENT || !ready || p_render_data == nullptr) {
		return;
	}
	RenderSceneBuffersRD *rb = Object::cast_to<RenderSceneBuffersRD>(p_render_data->get_render_scene_buffers().ptr());
	if (rb == nullptr) {
		return;
	}
	const Vector2i size = rb->get_internal_size();
	const uint64_t now = Time::get_singleton()->get_ticks_usec();
	float dt = fixed_dt;
	if (dt <= 0.0f) {
		dt = last_usec > 0 ? CLAMP(float(now - last_usec) / 1e6f, 0.0f, 0.25f) : 0.0f;
	}
	last_usec = now;

	PackedByteArray pc = _push_constants(size, dt);
	snap = false;

	Ref<RDUniform> u_hist = make_uniform(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER, 1, hist_buffer);
	Ref<RDUniform> u_state = make_uniform(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER, 2, state_buffer);
	auto uset = [&](int p_pass, const RID &p_color) {
		TypedArray<RDUniform> us;
		if (p_color.is_valid()) {
			us.push_back(make_uniform(RenderingDevice::UNIFORM_TYPE_IMAGE, 0, p_color));
		}
		us.push_back(u_hist);
		us.push_back(u_state);
		return UniformSetCacheRD::get_cache(shaders[p_pass], 0, us);
	};

	const Vector2i msize = (size + Vector2i(METER_STRIDE - 1, METER_STRIDE - 1)) / METER_STRIDE;
	const int64_t cl = rd->compute_list_begin();

	// Meter view 0 only; apply the same exposure to all views.
	rd->compute_list_bind_compute_pipeline(cl, pipelines[PASS_HIST]);
	rd->compute_list_bind_uniform_set(cl, uset(PASS_HIST, rb->get_color_layer(0)), 0);
	rd->compute_list_set_push_constant(cl, pc, pc.size());
	rd->compute_list_dispatch(cl, (msize.x + 15) / 16, (msize.y + 15) / 16, 1);
	rd->compute_list_add_barrier(cl);

	rd->compute_list_bind_compute_pipeline(cl, pipelines[PASS_REDUCE]);
	rd->compute_list_bind_uniform_set(cl, uset(PASS_REDUCE, RID()), 0);
	rd->compute_list_set_push_constant(cl, pc, pc.size());
	rd->compute_list_dispatch(cl, 1, 1, 1);
	rd->compute_list_add_barrier(cl);

	rd->compute_list_bind_compute_pipeline(cl, pipelines[PASS_APPLY]);
	rd->compute_list_set_push_constant(cl, pc, pc.size());
	for (uint32_t view = 0; view < rb->get_view_count(); view++) {
		rd->compute_list_bind_uniform_set(cl, uset(PASS_APPLY, rb->get_color_layer(view)), 0);
		rd->compute_list_dispatch(cl, (size.x + 7) / 8, (size.y + 7) / 8, 1);
	}
	rd->compute_list_end();
}

void GoghotHistogramExposure::_notification(int p_what) {
	if (p_what == NOTIFICATION_PREDELETE) {
		_free_resources();
	}
}

GoghotHistogramExposure::GoghotHistogramExposure() {
	set_effect_callback_type(EFFECT_CALLBACK_TYPE_POST_TRANSPARENT);
	_create_resources();
}

GoghotHistogramExposure::~GoghotHistogramExposure() {
	_free_resources();
}

void GoghotHistogramExposure::_bind_methods() {
#define GOGHOT_BIND(m_type, m_name, m_hint, m_hint_str)                                                   \
	ClassDB::bind_method(D_METHOD("set_" #m_name, "value"), &GoghotHistogramExposure::set_##m_name);           \
	ClassDB::bind_method(D_METHOD("get_" #m_name), &GoghotHistogramExposure::get_##m_name);                    \
	ADD_PROPERTY(PropertyInfo(m_type, #m_name, m_hint, m_hint_str), "set_" #m_name, "get_" #m_name);
	GOGHOT_BIND(Variant::FLOAT, low_percentile, PROPERTY_HINT_RANGE, "0,1,0.01")
	GOGHOT_BIND(Variant::FLOAT, high_percentile, PROPERTY_HINT_RANGE, "0,1,0.01")
	GOGHOT_BIND(Variant::FLOAT, key, PROPERTY_HINT_RANGE, "0.01,1,0.01")
	GOGHOT_BIND(Variant::FLOAT, compensation_ev, PROPERTY_HINT_RANGE, "-8,8,0.1,suffix:EV")
	GOGHOT_BIND(Variant::FLOAT, min_ev, PROPERTY_HINT_RANGE, "-16,16,0.1,suffix:EV")
	GOGHOT_BIND(Variant::FLOAT, max_ev, PROPERTY_HINT_RANGE, "-16,16,0.1,suffix:EV")
	GOGHOT_BIND(Variant::FLOAT, speed_up, PROPERTY_HINT_RANGE, "0,20,0.1,suffix:1/s")
	GOGHOT_BIND(Variant::FLOAT, speed_down, PROPERTY_HINT_RANGE, "0,20,0.1,suffix:1/s")
	GOGHOT_BIND(Variant::FLOAT, center_weight, PROPERTY_HINT_RANGE, "0,10,0.1")
	GOGHOT_BIND(Variant::FLOAT, center_sigma, PROPERTY_HINT_RANGE, "0.05,1,0.01")
	GOGHOT_BIND(Variant::FLOAT, fixed_dt, PROPERTY_HINT_NONE, "")
	GOGHOT_BIND(Variant::BOOL, snap, PROPERTY_HINT_NONE, "")
#undef GOGHOT_BIND
}

} // namespace goghot
