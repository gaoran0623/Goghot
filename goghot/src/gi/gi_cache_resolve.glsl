// Radiance cache resolve: fold this frame's samples into each entry (temporal, capped at
// cache_max_frames), clear the accumulators, evict entries unused for CACHE_EVICT_FRAMES.
layout(local_size_x = 64) in;

const uint CACHE_EVICT_FRAMES = 240u;

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i > uint(fr.cache_mask)) {
		return;
	}
	if (cache[i].key == 0u) {
		return;
	}
	uint n = cache[i].acc_n;
	if (n > 0u) {
		vec3 fresh = vec3(cache[i].acc_r, cache[i].acc_g, cache[i].acc_b) / (CACHE_FIXED_SCALE * float(n));
		vec4 r = cache[i].radiance;
		float frames = min(r.w + 1.0, fr.cache_max_frames);
		cache[i].radiance = vec4(mix(r.rgb, fresh, 1.0 / frames), frames);
		cache[i].acc_r = 0u;
		cache[i].acc_g = 0u;
		cache[i].acc_b = 0u;
		cache[i].acc_n = 0u;
	} else if (uint(fr.frame) - cache[i].last_frame > CACHE_EVICT_FRAMES) {
		cache[i].key = 0u;
		cache[i].radiance = vec4(0.0);
	}
}
