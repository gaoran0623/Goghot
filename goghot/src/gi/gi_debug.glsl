// Debug views (POST_TRANSPARENT, overwrite the HDR color). Variants:
//   DEBUG_PASS 0: per pixel, mode 1 = GI (rbgi/ambient: indirect diffuse on a white surface),
//                 mode 2 = reflections (rbgi/reflection)
//   DEBUG_PASS 1: per probe, draws each probe as a 5x5 dot (green uniform, red adaptive)
//   DEBUG_PASS 2: per pixel, tier B voxels: march the clipmap along the view ray, show the lit face
layout(local_size_x = 64) in;
layout(rgba16f, set = 0, binding = 1) uniform restrict image2D color;

layout(push_constant, std430) uniform PC {
	int mode;
	int pad0;
	int pad1;
	int pad2;
}
pc;

#if DEBUG_PASS == 0
layout(rgba16f, set = 0, binding = 2) uniform restrict readonly image2D ambient_img;
layout(rgba16f, set = 0, binding = 3) uniform restrict readonly image2D reflection_img;
void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.x % uint(fr.size.x), gl_GlobalInvocationID.x / uint(fr.size.x));
	if (p.y >= fr.size.y) {
		return;
	}
	vec3 c = pc.mode == 1 ? imageLoad(ambient_img, p).rgb : imageLoad(reflection_img, p).rgb;
	imageStore(color, p, vec4(c, 1.0));
}
#elif DEBUG_PASS == 1
layout(set = 0, binding = 4, std430) restrict readonly buffer Probes { Probe probes[]; };
layout(set = 0, binding = 5, std430) restrict readonly buffer Counter { uint counter[4]; };
void main() {
	uint i = gl_GlobalInvocationID.x;
	uint uniform_count = uint(fr.probe_grid.x * fr.probe_grid.y);
	if (i >= uniform_count + min(counter[0], uint(fr.max_adaptive))) {
		return;
	}
	Probe pr = probes[i];
	if (pr.pos.w < 0.5) {
		return;
	}
	int pi = int(pr.normal.w);
	ivec2 c0 = ivec2(pi % fr.size.x, pi / fr.size.x);
	vec3 col = i < uniform_count ? vec3(0.1, 4.0, 0.1) : vec3(4.0, 0.1, 0.1);
	for (int y = -2; y <= 2; y++) {
		for (int x = -2; x <= 2; x++) {
			ivec2 q = c0 + ivec2(x, y);
			if (all(greaterThanEqual(q, ivec2(0))) && all(lessThan(q, fr.size))) {
				imageStore(color, q, vec4(col, 1.0));
			}
		}
	}
}
#else
layout(set = 0, binding = 9) uniform sampler3D lit0;
layout(set = 0, binding = 10) uniform sampler3D lit1;
void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.x % uint(fr.size.x), gl_GlobalInvocationID.x / uint(fr.size.x));
	if (p.y >= fr.size.y) {
		return;
	}
	vec2 uv = (vec2(p) + 0.5) / vec2(fr.size);
	vec3 vdir = normalize(view_from_depth(uv, 0.5));
	vec3 o = fr.view_to_world[3].xyz;
	vec3 d = normalize(mat3(fr.view_to_world) * vdir);
	ivec3 ht;
	int hc;
	float t = vox_trace(o, d, 1e4, 0.0, ht, hc);
	vec3 c = fr.sky.rgb;
	if (t >= 0.0) {
		vec3 hn;
		int face = vox_face_for_ray(ht, d, hn);
		c = face < 0 ? vec3(1.0, 0.0, 1.0) : (face == 0 ? texelFetch(lit0, ht, 0).rgb : texelFetch(lit1, ht, 0).rgb);
		c *= 0.6 + 0.4 * float((hc + 1) % 2); // darker tint every other cascade
	}
	imageStore(color, p, vec4(c, 1.0));
}
#endif
