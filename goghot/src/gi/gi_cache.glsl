// World-space radiance cache: spatial hash (SHaRC-like). Needs CACHE_BINDING defined.
// Cells: world position quantized by a size that grows with camera distance (power-of-two levels)
// plus the dominant normal axis/sign, so the two sides of a wall never share a cell.
// Each entry accumulates incident indirect radiance samples (fixed point atomics) during the frame;
// gi_cache_resolve.glsl folds them into `radiance` (temporal, capped) and evicts stale entries.
struct CacheEntry {
	uint key; // checksum, 0 = empty
	uint last_frame;
	uint acc_r;
	uint acc_g;
	uint acc_b;
	uint acc_n;
	uint pad0;
	uint pad1;
	vec4 radiance; // rgb incident indirect radiance (= indirect irradiance / PI), a = frames
};

layout(set = 0, binding = CACHE_BINDING, std430) restrict buffer Cache {
	CacheEntry cache[];
};

const float CACHE_FIXED_SCALE = 1024.0;
const uint CACHE_PROBE_STEPS = 8u;

void cache_key(vec3 p, vec3 n, out uint slot, out uint check) {
	vec3 cam = fr.view_to_world[3].xyz;
	float cell = max(fr.cache_min_cell, distance(p, cam) * fr.cache_cell_scale);
	uint level = uint(clamp(ceil(log2(cell / fr.cache_min_cell)), 0.0, 15.0));
	float size = fr.cache_min_cell * exp2(float(level));
	ivec3 c = ivec3(floor(p / size));
	vec3 a = abs(n);
	uint axis = (a.x > a.y && a.x > a.z) ? 0u : (a.y > a.z ? 1u : 2u);
	uint nbin = axis * 2u + (n[axis] < 0.0 ? 1u : 0u);
	uint h = hash_u(uint(c.x) * 73856093u ^ hash_u(uint(c.y) * 19349663u ^ hash_u(uint(c.z) * 83492791u ^ (level << 3u | nbin))));
	slot = h & uint(fr.cache_mask);
	check = hash_u(h ^ 0x9e3779b9u) | 1u;
}

bool cache_lookup(vec3 p, vec3 n, out vec4 radiance) {
	uint slot, check;
	cache_key(p, n, slot, check);
	for (uint i = 0u; i < CACHE_PROBE_STEPS; i++) {
		uint idx = (slot + i) & uint(fr.cache_mask);
		uint k = cache[idx].key;
		if (k == check) {
			radiance = cache[idx].radiance;
			return true;
		}
		if (k == 0u) {
			break;
		}
	}
	radiance = vec4(0.0);
	return false;
}

void cache_insert(vec3 p, vec3 n, vec3 radiance) {
	uint slot, check;
	cache_key(p, n, slot, check);
	uvec3 q = uvec3(clamp(radiance, vec3(0.0), vec3(4000.0)) * CACHE_FIXED_SCALE);
	for (uint i = 0u; i < CACHE_PROBE_STEPS; i++) {
		uint idx = (slot + i) & uint(fr.cache_mask);
		uint prev = atomicCompSwap(cache[idx].key, 0u, check);
		if (prev == 0u || prev == check) {
			atomicAdd(cache[idx].acc_r, q.r);
			atomicAdd(cache[idx].acc_g, q.g);
			atomicAdd(cache[idx].acc_b, q.b);
			atomicAdd(cache[idx].acc_n, 1u);
			atomicMax(cache[idx].last_frame, uint(fr.frame));
			return;
		}
	}
}
