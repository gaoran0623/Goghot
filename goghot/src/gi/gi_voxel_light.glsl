// Tier B light injection, one thread per voxel (all cascades): for occupied voxels
//   lit = emission + albedo * (direct + cached indirect)
// direct: lights with shadows traced through the voxels; indirect: radiance cache at the voxel. One
// cosine bounce ray per voxel per frame (radiance from last frame's lit voxels, or the sky) is
// inserted into the cache, so bounces accumulate over frames like in tier A.
// Included after gi_voxel_common.glsl (bindings 1-3) and gi_cache.glsl (CACHE_BINDING 13).
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
layout(set = 0, binding = 4) uniform sampler3D vox_albedo0;
layout(set = 0, binding = 5) uniform sampler3D vox_albedo1;
layout(set = 0, binding = 6) uniform sampler3D vox_emission0;
layout(set = 0, binding = 7) uniform sampler3D vox_emission1;
layout(set = 0, binding = 8) uniform sampler3D lit_prev; // face 0
layout(set = 0, binding = 9) uniform sampler3D lit_prev_neg; // face 1
layout(rgba16f, set = 0, binding = 10) uniform restrict writeonly image3D lit_out;
layout(rgba16f, set = 0, binding = 11) uniform restrict writeonly image3D lit_out_neg;
struct Light {
	vec4 vec_type;
	vec4 color_range;
	vec4 params;
};
layout(set = 0, binding = 12, std430) restrict readonly buffer Lights { Light lights[]; };

float omni_attenuation(float dist, float range, float decay) {
	float nd = dist / max(range, 1e-4);
	nd *= nd;
	nd *= nd;
	nd = max(1.0 - nd, 0.0);
	nd *= nd;
	return nd * pow(max(dist, 1e-4), -decay);
}

vec3 face_lit(ivec3 t, vec3 p, float size, vec3 n, vec3 albedo, vec3 emission) {
	vec3 direct = vec3(0.0);
	ivec3 ht;
	int hc;
	for (int i = 0; i < fr.light_count; i++) {
		Light l = lights[i];
		vec3 L;
		float dist;
		float atten = 1.0;
		if (l.vec_type.w < 0.5) {
			L = -l.vec_type.xyz;
			dist = 1e4;
		} else {
			vec3 d = l.vec_type.xyz - p;
			dist = length(d);
			L = d / max(dist, 1e-5);
			atten = omni_attenuation(dist, l.color_range.w, l.params.x);
		}
		float ndl = dot(n, L);
		if (ndl <= 0.0 || atten <= 0.0) {
			continue;
		}
		float vis = 1.0;
		if (l.params.y > 0.5) {
			// Start off the face so the voxel (and its grown neighbours) do not shadow it.
			vis = vox_trace(p + n * size * 1.5, L, dist - size * 2.0, 0.5, ht, hc) < 0.0 ? 1.0 : 0.0;
		}
		direct += l.color_range.rgb * (ndl * atten * vis);
	}
	vec4 cached;
	vec3 indirect = cache_lookup(p, n, cached) ? cached.rgb : vec3(0.0);
	// Bounce samples for the cache (last frame's lit voxels, or the sky): fr.vox_params.w per face.
	vec3 tt = normalize(abs(n.y) < 0.99 ? cross(n, vec3(0, 1, 0)) : cross(n, vec3(1, 0, 0)));
	vec3 bb = cross(n, tt);
	vec3 incoming = vec3(0.0);
	int bounces = max(fr.vox_params.w, 1);
	for (int k = 0; k < bounces; k++) {
		uint seed = uint(t.x + t.y * 131 + t.z * 17161) * 9781u + uint(fr.frame * bounces + k) * 6271u + (n.x + n.y + n.z > 0.0 ? 0u : 77u);
		float r1 = hash_f(seed), r2 = hash_f(seed + 1u);
		float rr = sqrt(r1), phi = 2.0 * PI * r2;
		vec3 dir = normalize(tt * (rr * cos(phi)) + bb * (rr * sin(phi)) + n * sqrt(max(0.0, 1.0 - r1)));
		float th = vox_trace(p + n * size * 1.5, dir, 1e4, 0.5, ht, hc);
		vec3 s = fr.sky.rgb;
		if (th >= 0.0) {
			vec3 hn;
			int face = vox_face_for_ray(ht, dir, hn);
			s = face < 0 ? vec3(0.0) : (face == 0 ? texelFetch(lit_prev, ht, 0).rgb : texelFetch(lit_prev_neg, ht, 0).rgb);
		}
		incoming += s;
	}
	incoming /= float(bounces);
	cache_insert(p, n, incoming);
	return emission + albedo * (direct + indirect);
}

void main() {
	ivec3 t = ivec3(gl_GlobalInvocationID);
	int N = fr.vox_params.x;
	if (any(greaterThanEqual(t, ivec3(N, N, N * fr.vox_params.y)))) {
		return;
	}
	if (vox_occupancy(t) <= 0.0) {
		imageStore(lit_out, t, vec4(0.0));
		imageStore(lit_out_neg, t, vec4(0.0));
		return;
	}
	int c;
	ivec3 v = vox_local(t, c);
	float size = vox_size(c);
	vec3 p = fr.vox_origin[c].xyz + (vec3(v) + 0.5) * size;
	vec4 na = texelFetch(vox_normal_tex, t, 0);
	vec4 nb = texelFetch(vox_normal_neg_tex, t, 0);
	imageStore(lit_out, t, na.a > 0.5 ? vec4(face_lit(t, p, size, normalize(na.xyz * 2.0 - 1.0), texelFetch(vox_albedo0, t, 0).rgb, texelFetch(vox_emission0, t, 0).rgb), 1.0) : vec4(0.0));
	imageStore(lit_out_neg, t, nb.a > 0.5 ? vec4(face_lit(t, p, size, normalize(nb.xyz * 2.0 - 1.0), texelFetch(vox_albedo1, t, 0).rgb, texelFetch(vox_emission1, t, 0).rgb), 1.0) : vec4(0.0));
}
