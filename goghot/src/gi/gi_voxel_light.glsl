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
	vec4 spot;
	vec4 area_a;
	vec4 area_b;
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

// Area light (type 3): one random point q on the rectangle (seed), its single sample estimate of
// the rectangle's irradiance: radiance * cos_light * area / d^2 (times Godot's distance window at the
// closest point). Returns false when p is behind the light. L / dist point to q (shadow ray target).
bool area_light_sample(Light l, vec3 p, uint seed, out vec3 L, out float dist, out float atten) {
	vec3 c = l.vec_type.xyz;
	vec3 e = l.spot.xyz;
	atten = 0.0;
	L = vec3(0.0, 1.0, 0.0);
	dist = 1.0;
	if (dot(p - c, e) <= 0.0) {
		return false;
	}
	float ha = length(l.area_a.xyz), hb = length(l.area_b.xyz);
	vec3 ua = l.area_a.xyz / max(ha, 1e-6), ub = l.area_b.xyz / max(hb, 1e-6);
	vec3 rel = p - c;
	vec3 closest = c + ua * clamp(dot(rel, ua), -ha, ha) + ub * clamp(dot(rel, ub), -hb, hb);
	float dc = max(distance(p, closest), 1e-4);
	float window = omni_attenuation(dc, l.color_range.w, l.params.x) * dc * dc;
	vec3 q = c + l.area_a.xyz * (hash_f(seed) * 2.0 - 1.0) + l.area_b.xyz * (hash_f(seed + 1u) * 2.0 - 1.0);
	vec3 d = q - p;
	dist = max(length(d), 1e-4);
	L = d / dist;
	float cos_l = dot(-L, e);
	if (cos_l <= 0.0) {
		return false;
	}
	atten = window * cos_l * (4.0 * ha * hb) / (dist * dist);
	return true;
}

// Omni / spot distance attenuation (Godot's), times Godot's spot cone falloff for spots.
float light_attenuation(Light l, vec3 L, float dist) {
	float a = omni_attenuation(dist, l.color_range.w, l.params.x);
	if (l.vec_type.w > 1.5) {
		float scos = max(dot(-L, l.spot.xyz), l.spot.w);
		float rim = max(1e-4, (1.0 - scos) / max(1.0 - l.spot.w, 1e-6));
		a *= 1.0 - pow(rim, l.params.z);
	}
	return a;
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
		} else if (l.vec_type.w > 2.5) {
			uint seed = uint(t.x + t.y * 131 + t.z * 17161) * 7919u + uint(fr.frame) * 2654435u + uint(i) * 104729u + (n.x + n.y + n.z > 0.0 ? 0u : 31u);
			if (!area_light_sample(l, p, seed, L, dist, atten)) {
				continue;
			}
		} else {
			vec3 d = l.vec_type.xyz - p;
			dist = length(d);
			L = d / max(dist, 1e-5);
			atten = light_attenuation(l, L, dist);
		}
		float ndl = dot(n, L);
		if (ndl <= 0.0 || atten <= 0.0) {
			continue;
		}
		float vis = 1.0;
		if (l.params.y > 0.5) {
			// Start off the face so the voxel (and its grown neighbours) do not shadow it.
			vec3 so = p + n * size * 1.5;
			float tmax = dist - size * 2.0;
			if (l.vec_type.w > 2.5) {
				// Area lights usually sit on a wall / ceiling, whose (conservatively thickened) voxels
				// would block grazing shadow rays before they reach the panel: stop the ray 1.5 voxels
				// in front of the panel's plane instead (occluders in the room still count).
				float le = dot(L, l.spot.xyz); // < 0: towards the panel
				if (le < -1e-4) {
					tmax = min(tmax, (size * 1.5 - dot(so - l.vec_type.xyz, l.spot.xyz)) / le);
				}
			}
			vis = tmax <= 0.0 || vox_trace(so, L, tmax, 0.5, ht, hc) < 0.0 ? 1.0 : 0.0;
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
		vec3 s = sky_radiance(dir);
		if (th >= 0.0) {
			vec3 hn;
			int face = vox_face_for_ray(ht, dir, hn);
			s = face < 0 ? max(texelFetch(lit_prev, ht, 0).rgb, texelFetch(lit_prev_neg, ht, 0).rgb) : (face == 0 ? texelFetch(lit_prev, ht, 0).rgb : texelFetch(lit_prev_neg, ht, 0).rgb);
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
