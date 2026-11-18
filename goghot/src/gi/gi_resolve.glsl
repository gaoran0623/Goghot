// Per-pixel resolve: interpolate the 2x2 surrounding uniform probes (+ the adaptive probe of the
// pixel sub-tile), weighted by bilinear position, plane distance and normal similarity. Falls back
// to a wider, plane-only search so every non-sky pixel gets a value when any nearby probe is valid.
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 1) uniform sampler2D depth_tex;
layout(set = 0, binding = 2) uniform sampler2D normal_tex;
layout(set = 0, binding = 3, std430) restrict readonly buffer Probes { Probe probes[]; };
layout(set = 0, binding = 4, std430) restrict readonly buffer Filtered { vec4 filtered[]; };
layout(set = 0, binding = 5, std430) restrict readonly buffer AdaptiveMap { int adaptive_map[]; };
layout(rgba16f, set = 0, binding = 6) uniform restrict writeonly image2D resolved;

void accumulate(uint j, vec3 p, vec3 n, float w_extra, bool use_normal, inout vec4 sum) {
	Probe q = probes[j];
	vec4 r = filtered[j];
	if (q.pos.w < 0.5 || r.w <= 0.0) {
		return;
	}
	float w = w_extra * plane_weight(p, n, q.pos.xyz);
	if (use_normal) {
		w *= normal_weight(n, q.normal.xyz);
	}
	sum += vec4(r.rgb, 1.0) * w;
}

void main() {
	ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pix, fr.size))) {
		return;
	}
	float depth = texelFetch(depth_tex, pix, 0).r;
	if (depth <= 0.0) {
		imageStore(resolved, pix, vec4(0.0)); // sky
		return;
	}
	vec2 uv = (vec2(pix) + 0.5) / vec2(fr.size);
	vec3 p = view_from_depth(uv, depth);
	vec3 n = decode_normal(texelFetch(normal_tex, pix, 0));

	vec4 sum = vec4(0.0);
	vec2 f = (vec2(pix) + 0.5) / float(fr.spacing) - 0.5;
	ivec2 base = ivec2(floor(f));
	vec2 fr2 = f - vec2(base);
	for (int y = 0; y <= 1; y++) {
		for (int x = 0; x <= 1; x++) {
			ivec2 tt = clamp(base + ivec2(x, y), ivec2(0), fr.probe_grid - 1);
			float wb = (x == 0 ? 1.0 - fr2.x : fr2.x) * (y == 0 ? 1.0 - fr2.y : fr2.y);
			accumulate(uint(tt.x + tt.y * fr.probe_grid.x), p, n, max(wb, 1e-3), true, sum);
		}
	}
	ivec2 st = min(pix / max(fr.spacing / 2, 1), fr.adaptive_grid - 1);
	int ai = adaptive_map[st.x + st.y * fr.adaptive_grid.x];
	if (ai >= 0) {
		accumulate(uint(ai), p, n, 1.0, true, sum);
	}
	if (sum.w < 1e-3) { // nothing similar nearby: 3x3 tiles, plane only
		sum = vec4(0.0);
		ivec2 tile = min(pix / fr.spacing, fr.probe_grid - 1);
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				ivec2 tt = clamp(tile + ivec2(x, y), ivec2(0), fr.probe_grid - 1);
				accumulate(uint(tt.x + tt.y * fr.probe_grid.x), p, n, 1.0, false, sum);
			}
		}
	}
	imageStore(resolved, pix, sum.w > 1e-6 ? vec4(sum.rgb / sum.w, 1.0) : vec4(0.0));
}
