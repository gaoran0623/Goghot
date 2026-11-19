// Goghot screen probe GI: shared declarations. Every GI pass is COMMON + its own body.
// Frame UBO layout must match FrameUBO in screen_probe_gi.cpp (std140).
// No #version here: the C++ side prepends it (and #extension lines for ray tracing stages).

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
	mat4 view_to_world; // camera transform
	int light_count;
	int cache_mask; // radiance cache entries - 1 (power of two)
	int rt_enabled; // world tracer available and scene present
	int cache_mature_frames; // cached radiance is returned instead of a fresh sample once this old
	float cache_min_cell; // radiance cache cell size at distance 0 (m)
	float cache_cell_scale; // cell size grows with distance * scale
	float cache_max_frames; // temporal accumulation cap of cache entries
	float pad3;
	ivec4 fog_grid; // x: voxels per axis, y: rays per voxel, z: enabled, w: probe ray count (fog rays follow)
	vec4 fog_origin; // xyz: world position of voxel (0,0,0) min corner, w: voxel size
	ivec4 fog_shift; // xyz: voxel offset of this volume in the previous one
	float fog_max_frames;
	float pad4;
	float pad5;
	float pad6;
	ivec4 refl_grid; // xy: half resolution size, z: enabled, w: launch index where reflection rays start
	vec4 refl_params; // x: max traced roughness, y/z: history frames at roughness 0 / max, w: normal power
	vec4 vox_origin[3]; // tier B clipmap cascades: xyz world min corner, w voxel size
	ivec4 vox_params; // x: voxels per axis, y: cascades, z: tier B active, w: bounce rays per voxel face
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

// Per-probe ray i (cosine distributed around n, Hammersley + per-probe/frame rotation).
// Shared by the screen trace and the world tracer so both see the same ray.
vec3 probe_ray_dir(uint probe_index, uint lane, vec3 n) {
	vec3 t = normalize(abs(n.y) < 0.99 ? cross(n, vec3(0, 1, 0)) : cross(n, vec3(1, 0, 0)));
	vec3 b = cross(n, t);
	uint seed = probe_index * 7177u + uint(fr.frame) * 31337u;
	uint bits = bitfieldReverse(lane);
	vec2 h = vec2((float(lane) + 0.5) / float(fr.rays), float(bits) * 2.3283064365386963e-10);
	vec2 u = fract(h + vec2(hash_f(seed), hash_f(seed + 1u)));
	float r = sqrt(u.x);
	float phi = 2.0 * PI * u.y;
	return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u.x)));
}

vec3 probe_ray_origin(Probe pr) {
	return pr.pos.xyz + pr.normal.xyz * max(-pr.pos.z, fr.near) * 0.002;
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
