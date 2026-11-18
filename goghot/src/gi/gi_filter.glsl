// Spatial filter on probes: bilateral average with the 3x3 uniform probes around the probe tile,
// weighted by plane distance and normal similarity.
layout(local_size_x = 64) in;
layout(set = 0, binding = 1, std430) restrict readonly buffer Probes { Probe probes[]; };
layout(set = 0, binding = 2, std430) restrict readonly buffer Counter { uint counter[4]; };
layout(set = 0, binding = 3, std430) restrict readonly buffer Radiance { vec4 radiance[]; };
layout(set = 0, binding = 4, std430) restrict writeonly buffer Filtered { vec4 filtered[]; };

void main() {
	uint i = gl_GlobalInvocationID.x;
	uint uniform_count = uint(fr.probe_grid.x * fr.probe_grid.y);
	if (i >= uniform_count + min(counter[0], uint(fr.max_adaptive))) {
		return;
	}
	Probe pr = probes[i];
	vec4 self = radiance[i];
	if (pr.pos.w < 0.5 || fr.debug_mode == 2) {
		filtered[i] = self;
		return;
	}
	int pix_index = int(pr.normal.w);
	ivec2 pix = ivec2(pix_index % fr.size.x, pix_index / fr.size.x);
	ivec2 tile = min(pix / fr.spacing, fr.probe_grid - 1);
	vec4 sum = vec4(self.rgb, 1.0) * self.w;
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			ivec2 tt = tile + ivec2(x, y);
			if (any(lessThan(tt, ivec2(0))) || any(greaterThanEqual(tt, fr.probe_grid))) {
				continue;
			}
			uint j = uint(tt.x + tt.y * fr.probe_grid.x);
			if (j == i) {
				continue;
			}
			Probe q = probes[j];
			vec4 r = radiance[j];
			if (q.pos.w < 0.5 || r.w <= 0.0) {
				continue;
			}
			float w = plane_weight(pr.pos.xyz, pr.normal.xyz, q.pos.xyz) * normal_weight(pr.normal.xyz, q.normal.xyz);
			sum += vec4(r.rgb, 1.0) * w;
		}
	}
	filtered[i] = sum.w > 0.0 ? vec4(sum.rgb / sum.w, 1.0) : self;
}
