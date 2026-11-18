// Uniform probes: one per spacing x spacing tile, at a per-frame jittered pixel.
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 1) uniform sampler2D depth_tex;
layout(set = 0, binding = 2) uniform sampler2D normal_tex;
layout(set = 0, binding = 3, std430) restrict writeonly buffer Probes { Probe probes[]; };
layout(set = 0, binding = 4, std430) restrict buffer Counter { uint counter[4]; };

void main() {
	ivec2 t = ivec2(gl_GlobalInvocationID.xy);
	if (t == ivec2(0)) {
		counter[0] = 0u; // adaptive probe count for this frame
	}
	if (any(greaterThanEqual(t, fr.probe_grid))) {
		return;
	}
	uint seed = uint(t.x + t.y * fr.probe_grid.x) * 9781u + uint(fr.frame * fr.jitter) * 6271u;
	ivec2 jitter = ivec2(hash_f(seed) * float(fr.spacing), hash_f(seed + 1u) * float(fr.spacing));
	ivec2 pix = min(t * fr.spacing + jitter, fr.size - 1);
	float depth = texelFetch(depth_tex, pix, 0).r;
	Probe pr;
	vec2 uv = (vec2(pix) + 0.5) / vec2(fr.size);
	pr.pos = vec4(view_from_depth(uv, depth), depth > 0.0 ? 1.0 : 0.0);
	pr.normal = vec4(decode_normal(texelFetch(normal_tex, pix, 0)), float(pix.x + pix.y * fr.size.x));
	probes[t.x + t.y * fr.probe_grid.x] = pr;
}
