// Adaptive probes: for every spacing/2 sub-tile whose (jittered) sample is not represented by the
// parent uniform probe (different plane or normal, or parent invalid), allocate an extra probe.
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 1) uniform sampler2D depth_tex;
layout(set = 0, binding = 2) uniform sampler2D normal_tex;
layout(set = 0, binding = 3, std430) restrict buffer Probes { Probe probes[]; };
layout(set = 0, binding = 4, std430) restrict buffer Counter { uint counter[4]; };
layout(set = 0, binding = 5, std430) restrict writeonly buffer AdaptiveMap { int adaptive_map[]; };

void main() {
	ivec2 st = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(st, fr.adaptive_grid))) {
		return;
	}
	int map_index = st.x + st.y * fr.adaptive_grid.x;
	adaptive_map[map_index] = -1;
	int half_spacing = fr.spacing / 2;
	uint seed = uint(map_index) * 7919u + uint(fr.frame * fr.jitter) * 104729u + 17u;
	ivec2 jitter = ivec2(hash_f(seed) * float(half_spacing), hash_f(seed + 1u) * float(half_spacing));
	ivec2 pix = st * half_spacing + jitter;
	if (any(greaterThanEqual(pix, fr.size))) {
		return;
	}
	float depth = texelFetch(depth_tex, pix, 0).r;
	if (depth <= 0.0) {
		return; // sky
	}
	vec2 uv = (vec2(pix) + 0.5) / vec2(fr.size);
	vec3 p = view_from_depth(uv, depth);
	vec3 n = decode_normal(texelFetch(normal_tex, pix, 0));

	ivec2 parent = min(pix / fr.spacing, fr.probe_grid - 1);
	Probe par = probes[parent.x + parent.y * fr.probe_grid.x];
	bool covered = par.pos.w > 0.5 && plane_weight(par.pos.xyz, par.normal.xyz, p) > 0.5 && dot(par.normal.xyz, n) > 0.9;
	if (covered) {
		return;
	}
	uint slot = atomicAdd(counter[0], 1u);
	if (slot >= uint(fr.max_adaptive)) {
		return;
	}
	int index = fr.probe_grid.x * fr.probe_grid.y + int(slot);
	Probe pr;
	pr.pos = vec4(p, 1.0);
	pr.normal = vec4(n, float(pix.x + pix.y * fr.size.x));
	probes[index] = pr;
	adaptive_map[map_index] = index;
}
