// Screen trace: one workgroup per probe, one cosine-distributed ray per thread.
// HZB march; a hit takes its radiance from the previous frame lit color (reprojected), which also
// gives multi-bounce and emissive GI through frame feedback. Per-ray result in rays[]:
//   w = 1: resolved (screen hit), w = 0: unresolved (left the screen / no data) -> world tracer,
//   which falls back to the sky. Without the world tracer the gather pass uses the sky directly.
layout(local_size_x = 32) in;
layout(set = 0, binding = 1) uniform sampler2D hzb; // mip chain, closest depth
layout(set = 0, binding = 2) uniform sampler2D prev_color; // previous frame HDR, linear sampler
layout(set = 0, binding = 3, std430) restrict readonly buffer Probes { Probe probes[]; };
layout(set = 0, binding = 4, std430) restrict readonly buffer Counter { uint counter[4]; };
layout(set = 0, binding = 5, std430) restrict writeonly buffer Rays { vec4 rays[]; };

// Screen-space position: xy in pixels, z = NDC depth (reverse Z). Linear in screen space.
vec3 to_screen(vec3 view_pos) {
	vec4 c = fr.proj * vec4(view_pos, 1.0);
	c.xyz /= c.w;
	return vec3((c.xy * 0.5 + 0.5) * vec2(fr.size), c.z);
}

// Returns true on hit; hit_px = pixel of the hit.
bool trace_screen(vec3 origin, vec3 dir, out vec2 hit_px) {
	float len = fr.max_distance;
	if (dir.z > 0.0) { // towards the camera: stop before the near plane
		len = min(len, (-fr.near * 1.01 - origin.z) / dir.z);
	}
	if (len <= 0.0) {
		return false;
	}
	vec3 s0 = to_screen(origin);
	vec3 s1 = to_screen(origin + dir * len);
	vec3 d = s1 - s0;
	float d_len = length(d.xy);
	if (d_len < 1e-4) {
		return false;
	}
	// Parametric end where the ray leaves the screen.
	float t_end = 1.0;
	vec2 sz = vec2(fr.size);
	if (d.x > 0.0) t_end = min(t_end, (sz.x - 0.5 - s0.x) / d.x);
	if (d.x < 0.0) t_end = min(t_end, (0.5 - s0.x) / d.x);
	if (d.y > 0.0) t_end = min(t_end, (sz.y - 0.5 - s0.y) / d.y);
	if (d.y < 0.0) t_end = min(t_end, (0.5 - s0.y) / d.y);

	float t = 1.5 / d_len; // skip the starting pixel
	int level = 0;
	int max_level = fr.hzb_levels - 1;
	for (int i = 0; i < fr.max_steps && t < t_end; i++) {
		vec3 pos = s0 + d * t;
		float cell_size = float(1 << level);
		vec2 cell = floor(pos.xy / cell_size);
		// t at which the ray leaves this cell
		vec2 bound = (cell + step(0.0, d.xy)) * cell_size;
		vec2 tc = vec2(1e30);
		if (abs(d.x) > 1e-6) tc.x = (bound.x - s0.x) / d.x;
		if (abs(d.y) > 1e-6) tc.y = (bound.y - s0.y) / d.y;
		float t_exit = min(min(tc.x, tc.y), t_end) + 1e-4 / d_len;

		ivec2 msize = textureSize(hzb, level);
		float surf = texelFetch(hzb, clamp(ivec2(cell), ivec2(0), msize - 1), level).r;
		float z_far = min(pos.z, s0.z + d.z * t_exit); // farthest ray depth within the cell
		if (z_far > surf) {
			// Ray stays in front of the closest surface in this cell: skip it, go coarser.
			t = t_exit;
			level = min(level + 1, max_level);
			continue;
		}
		if (level > 0) {
			level--;
			continue;
		}
		// Level 0: the ray reaches the depth of this pixel. Thickness test in linear depth.
		vec2 uv = pos.xy / sz;
		float ray_lin = linear_depth_from_ndc(uv, max(min(pos.z, 1.0), 1e-7));
		float surf_lin = linear_depth_from_ndc(uv, max(surf, 1e-7));
		if (surf > 0.0 && ray_lin - surf_lin < surf_lin * fr.thickness) {
			hit_px = pos.xy;
			return true;
		}
		t = t_exit; // passed behind a thin surface: keep marching at full resolution
	}
	return false;
}

void main() {
	uint probe_index = gl_WorkGroupID.x;
	uint lane = gl_LocalInvocationID.x;
	uint total = uint(fr.probe_grid.x * fr.probe_grid.y) + min(counter[0], uint(fr.max_adaptive));
	if (probe_index >= total) {
		return;
	}
	Probe pr = probes[probe_index];
	vec4 result = vec4(0.0, 0.0, 0.0, -1.0); // w < 0: no ray (invalid probe / lane >= rays)
	if (pr.pos.w > 0.5 && lane < uint(fr.rays)) {
		vec3 dir = probe_ray_dir(probe_index, lane, pr.normal.xyz);
		vec2 hit_px;
		result = vec4(0.0); // unresolved
		if (trace_screen(probe_ray_origin(pr), dir, hit_px)) {
			vec2 huv = hit_px / vec2(fr.size);
			float hd = texelFetch(hzb, ivec2(hit_px), 0).r;
			vec3 hv = view_from_depth(huv, hd);
			vec4 pc4 = fr.prev_proj * (fr.cur_to_prev_view * vec4(hv, 1.0));
			vec2 puv = (pc4.xy / pc4.w) * 0.5 + 0.5;
			if (fr.has_history != 0 && pc4.w > 0.0 && all(greaterThanEqual(puv, vec2(0.0))) && all(lessThanEqual(puv, vec2(1.0)))) {
				result = vec4(textureLod(prev_color, puv, 0.0).rgb, 1.0);
			}
		}
	}
	rays[probe_index * 32u + lane] = result;
}
