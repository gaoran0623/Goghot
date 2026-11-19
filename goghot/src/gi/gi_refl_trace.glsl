// Reflection rays, half resolution: one GGX (VNDF) sampled ray per 2x2 block, from a pixel of the
// block chosen per frame. Screen trace first (previous frame color at the hit); rays that leave the
// screen are stored for the world tracer (gi_rt_raygen.glsl, reflection mode), which shades them
// or returns the sky. Pixels rougher than refl_params.x get no ray (they reuse the probe GI).
// Included after gi_screen_trace.glsl (binding 1 = HZB).
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 2) uniform sampler2D prev_color;
layout(set = 0, binding = 3) uniform sampler2D depth_tex;
layout(set = 0, binding = 4) uniform sampler2D normal_tex;

struct ReflRay {
	vec4 origin; // xyz world origin, w: -1 no ray, 0 needs world trace, 1 resolved
	vec4 dir; // xyz world direction, w: full resolution pixel index
	vec4 radiance;
};
layout(set = 0, binding = 5, std430) restrict writeonly buffer ReflRays { ReflRay refl[]; };

float decode_roughness(float w) {
	// Godot packs a dynamic-object flag into the roughness channel (see scene_forward_clustered).
	return (w > 0.5 ? 1.0 - w : w) * (255.0 / 127.0);
}

// Heitz 2018, "Sampling the GGX Distribution of Visible Normals". Ve in tangent space (z = normal).
vec3 sample_vndf(vec3 Ve, float alpha, vec2 u) {
	vec3 Vh = normalize(vec3(alpha * Ve.x, alpha * Ve.y, Ve.z));
	float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) * inversesqrt(lensq) : vec3(1.0, 0.0, 0.0);
	vec3 T2 = cross(Vh, T1);
	float r = sqrt(u.x);
	float phi = 2.0 * PI * u.y;
	float t1 = r * cos(phi);
	float t2 = r * sin(phi);
	float s = 0.5 * (1.0 + Vh.z);
	t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;
	vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
	return normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
}

void main() {
	ivec2 q = ivec2(gl_GlobalInvocationID.xy);
	ivec2 half_size = fr.refl_grid.xy;
	if (any(greaterThanEqual(q, half_size))) {
		return;
	}
	uint index = uint(q.x + q.y * half_size.x);
	uint seed = index * 3266489917u + uint(fr.frame) * 668265263u;
	ivec2 pix = min(q * 2 + ivec2(hash_u(seed) & 1u, (hash_u(seed + 1u) >> 1u) & 1u), fr.size - 1);
	ReflRay r;
	r.origin = vec4(0.0, 0.0, 0.0, -1.0);
	r.dir = vec4(0.0, 0.0, 0.0, float(pix.x + pix.y * fr.size.x));
	r.radiance = vec4(0.0);

	float depth = texelFetch(depth_tex, pix, 0).r;
	vec4 nr = texelFetch(normal_tex, pix, 0);
	float rough = decode_roughness(nr.w);
	if (depth <= 0.0 || rough > fr.refl_params.x) {
		refl[index] = r;
		return;
	}
	vec2 uv = (vec2(pix) + 0.5) / vec2(fr.size);
	vec3 p = view_from_depth(uv, depth);
	vec3 n = decode_normal(nr);
	vec3 v = -normalize(p);
	vec3 t = normalize(abs(n.y) < 0.99 ? cross(n, vec3(0, 1, 0)) : cross(n, vec3(1, 0, 0)));
	vec3 b = cross(n, t);
	vec3 ve = vec3(dot(v, t), dot(v, b), dot(v, n));
	float alpha = max(rough * rough, 0.001);
	vec3 h = sample_vndf(normalize(vec3(ve.xy, max(ve.z, 1e-3))), alpha, vec2(hash_f(seed + 2u), hash_f(seed + 3u)));
	h = t * h.x + b * h.y + n * h.z;
	vec3 l = reflect(-v, h);
	if (dot(l, n) <= 0.0) {
		l = reflect(-v, n);
	}
	vec3 origin = p + n * max(-p.z, fr.near) * 0.002;
	r.origin = vec4((fr.view_to_world * vec4(origin, 1.0)).xyz, 0.0);
	r.dir.xyz = normalize(mat3(fr.view_to_world) * l);
	r.radiance = vec4(normalize(mat3(fr.view_to_world) * n), 0.0); // surface normal for the world tracer, until resolved

	vec2 hit_px;
	if (trace_screen(origin, l, hit_px)) {
		vec2 huv = hit_px / vec2(fr.size);
		vec3 hv = view_from_depth(huv, texelFetch(hzb, ivec2(hit_px), 0).r);
		vec4 pc4 = fr.prev_proj * (fr.cur_to_prev_view * vec4(hv, 1.0));
		vec2 puv = (pc4.xy / pc4.w) * 0.5 + 0.5;
		if (fr.has_history != 0 && pc4.w > 0.0 && all(greaterThanEqual(puv, vec2(0.0))) && all(lessThanEqual(puv, vec2(1.0)))) {
			r.radiance = vec4(textureLod(prev_color, puv, 0.0).rgb, 1.0);
			r.origin.w = 1.0;
		}
	}
	if (r.origin.w == 0.0 && fr.rt_enabled == 0) {
		r.radiance = vec4(fr.sky.rgb, 1.0); // no world tracer: sky
		r.origin.w = 1.0;
	}
	refl[index] = r;
}
