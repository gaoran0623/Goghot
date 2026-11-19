// Gather: mean radiance of each probe's rays = irradiance / PI (rays are cosine distributed).
// Rays still unresolved here (no world tracer, or no scene) take the sky radiance.
layout(local_size_x = 64) in;
layout(set = 0, binding = 1, std430) restrict readonly buffer Counter { uint counter[4]; };
layout(set = 0, binding = 2, std430) restrict readonly buffer Rays { vec4 rays[]; };
layout(set = 0, binding = 3, std430) restrict writeonly buffer Radiance { vec4 radiance[]; };

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= uint(fr.probe_grid.x * fr.probe_grid.y) + min(counter[0], uint(fr.max_adaptive))) {
		return;
	}
	vec3 sum = vec3(0.0);
	float n = 0.0;
	for (uint k = 0u; k < 32u; k++) {
		vec4 r = rays[i * 32u + k];
		if (r.w < 0.0) {
			continue;
		}
		sum += r.w > 0.0 ? r.rgb : fr.sky.rgb;
		n += 1.0;
	}
	radiance[i] = n > 0.0 ? vec4(sum / n, 1.0) : vec4(0.0);
}
