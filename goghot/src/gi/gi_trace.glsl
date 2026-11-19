// Screen trace: one workgroup per probe, one cosine-distributed ray per thread.
// HZB march; a hit takes its radiance from the previous frame lit color (reprojected), which also
// gives multi-bounce and emissive GI through frame feedback. Per-ray result in rays[]:
//   w = 1: resolved (screen hit), w = 0: unresolved (left the screen / no data) -> world tracer,
//   which falls back to the sky. Without the world tracer the gather pass uses the sky directly.
layout(local_size_x = 32) in;
layout(set = 0, binding = 2) uniform sampler2D prev_color; // previous frame HDR, linear sampler
layout(set = 0, binding = 3, std430) restrict readonly buffer Probes { Probe probes[]; };
layout(set = 0, binding = 4, std430) restrict readonly buffer Counter { uint counter[4]; };
layout(set = 0, binding = 5, std430) restrict writeonly buffer Rays { vec4 rays[]; };

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
	if (result.w == 0.0 && fr.rt_enabled == 0) {
		result = vec4(sky_radiance(normalize(mat3(fr.view_to_world) * probe_ray_dir(probe_index, lane, pr.normal.xyz))), 1.0); // no world tracer: sky
	}
	rays[probe_index * 32u + lane] = result;
}
