// Goghot histogram auto exposure: shared declarations (push constants must match HistogramExposure::_push_constants).
#version 450
#define BINS 128
layout(push_constant, std430) uniform Params {
	ivec2 size;
	float min_log2;
	float max_log2;
	float low_pct;
	float high_pct;
	float key;
	float comp_ev;
	float min_ev;
	float max_ev;
	float speed_up;
	float speed_down;
	float dt;
	float center_weight;
	float center_sigma;
	float snap;
	int stride;
	int pad0;
	int pad1;
	int pad2;
} params;
layout(set = 0, binding = 1, std430) restrict buffer Hist { uint hist[BINS]; };
layout(set = 0, binding = 2, std430) restrict buffer State { float ev; float initialized; };
float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
