// Goghot screen probe GI: shared declarations. Every GI pass is COMMON + its own body.
// Frame UBO layout must match GoghotScreenProbeGI::_update_frame_ubo (std140).
#version 450

layout(set = 0, binding = 0, std140) uniform Frame {
	mat4 proj; // current view -> clip (Vulkan corrected: reverse Z, flipped Y)
	mat4 inv_proj;
	mat4 cur_to_prev_view; // current view space -> previous view space
	mat4 prev_proj;
	ivec2 size; // internal resolution
	ivec2 probe_grid; // uniform probes (one per spacing^2 tile)
	ivec2 adaptive_grid; // sub-tiles (spacing/2) that may get an adaptive probe
	int spacing;
	int frame;
	vec4 sky; // rgb: linear sky radiance used for rays that leave the screen
	int rays; // rays per probe (<= 32)
	int max_adaptive;
	int hzb_levels;
	int debug_mode; // 0 normal, 1 no temporal, 2 no spatial filter
	float thickness; // relative (fraction of linear depth) surface thickness for screen traces
	float max_distance; // view-space ray length
	float plane_threshold; // relative plane distance for probe/pixel similarity
	float normal_power;
	float temporal_max_frames;
	float near;
	int has_history;
	int max_steps;
	float pad_clip;
	float response_sigma; // temporal-variance response threshold (0 = off)
	float response_min_frames; // history length floor when responding to change
	int jitter; // 1 = jitter probe positions per frame (0 for debugging)
}
fr;

struct Probe {
	vec4 pos; // view-space position, w = 1 valid / 0 invalid
	vec4 normal; // view-space normal, w = pixel index (x + y * width)
};

const float PI = 3.14159265359;

vec3 view_from_depth(vec2 uv, float depth) {
	vec4 p = fr.inv_proj * vec4(uv * 2.0 - 1.0, depth, 1.0);
	return p.xyz / p.w;
}

float linear_depth_from_ndc(vec2 uv, float depth) {
	return -view_from_depth(uv, depth).z;
}

vec3 decode_normal(vec4 nr) {
	return normalize(nr.xyz * 2.0 - 1.0);
}

uint hash_u(uint x) {
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

float hash_f(uint x) {
	return float(hash_u(x) >> 8) * (1.0 / 16777216.0);
}

// Similarity of a sample point to a reference surface (plane + normal).
float plane_weight(vec3 ref_pos, vec3 ref_n, vec3 p) {
	float d = abs(dot(ref_n, p - ref_pos));
	float scale = max(-ref_pos.z, fr.near) * fr.plane_threshold;
	float x = d / scale;
	return exp(-x * x);
}

float normal_weight(vec3 a, vec3 b) {
	return pow(max(dot(a, b), 0.0), fr.normal_power);
}
