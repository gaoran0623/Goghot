// World tracer ray generation: one invocation per probe ray left unresolved by the screen trace.
// Hit radiance = emission + albedo * (direct + indirect). Direct is always traced fresh (shadow
// rays). Indirect comes from one cosine bounce ray whose hit is shaded the same way, using the
// radiance cache only for its own indirect term (infinite bounces build up over frames). The cache
// stores incident indirect radiance per cell; a mature cell replaces the noisy single sample.
// Back faces return black (the ray started inside geometry or passed through a wall gap).
layout(location = 0) rayPayloadEXT HitPayload hit;
layout(location = 1) rayPayloadEXT float shadow_visible;

float omni_attenuation(float dist, float range, float decay) {
	float nd = dist / max(range, 1e-4);
	nd *= nd;
	nd *= nd;
	nd = max(1.0 - nd, 0.0);
	nd *= nd;
	return nd * pow(max(dist, 1e-4), -decay);
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

float trace_shadow(vec3 o, vec3 d, float tmax) {
	shadow_visible = 0.0;
	traceRayEXT(tlas, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
			0xFF, 0, 1, 1, o, 0.0, d, tmax, 1);
	return shadow_visible;
}

vec3 direct_light(vec3 p, vec3 n) {
	vec3 sum = vec3(0.0);
	vec3 o = p + n * max(0.002, length(p - fr.view_to_world[3].xyz) * 1e-4);
	for (int i = 0; i < fr.light_count; i++) {
		Light l = lights[i];
		vec3 L;
		float dist;
		float atten = 1.0;
		if (l.vec_type.w < 0.5) {
			L = -l.vec_type.xyz;
			dist = 1e5;
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
		float vis = l.params.y > 0.5 ? trace_shadow(o, L, dist - 0.01) : 1.0;
		sum += l.color_range.rgb * (ndl * atten * vis);
	}
	return sum;
}

bool trace_hit(vec3 o, vec3 d) {
	hit.t = -1.0;
	traceRayEXT(tlas, gl_RayFlagsNoneEXT, 0xFF, 0, 1, 0, o, 0.0, d, 1e5, 0); // opacity comes from instance flags
	return hit.t >= 0.0;
}

vec3 cosine_dir(vec3 n, vec2 u) {
	vec3 t = normalize(abs(n.y) < 0.99 ? cross(n, vec3(0, 1, 0)) : cross(n, vec3(1, 0, 0)));
	vec3 b = cross(n, t);
	float r = sqrt(u.x);
	float phi = 2.0 * PI * u.y;
	return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u.x)));
}

// Radiance along a ray: miss -> sky, back face -> black, hit -> emission + albedo * (fresh direct +
// cached indirect). No extra bounce ray (used by fog and reflection rays).
vec3 shade_simple(vec3 origin, vec3 dir) {
	if (!trace_hit(origin, dir)) {
		return fr.sky.rgb;
	}
	if (hit.backface > 0.5) {
		return vec3(0.0);
	}
	vec3 p = origin + dir * hit.t;
	vec3 nrm = hit.normal;
	vec3 albedo = hit.albedo;
	vec3 emission = hit.emission;
	vec4 c;
	vec3 ind = cache_lookup(p, nrm, c) ? c.rgb : vec3(0.0);
	return emission + albedo * (direct_light(p, nrm) + ind);
}

// Fog GI volume ray: uniform sphere direction from a jittered point in the voxel. Hits are shaded
// with fresh direct light + cached indirect (no extra bounce ray); misses see the sky.
void fog_ray(uint i) {
	uint rpv = uint(max(fr.fog_grid.y, 1));
	uint voxel = i / rpv;
	uint n = uint(fr.fog_grid.x);
	if (voxel >= n * n * n) {
		return;
	}
	uvec3 v = uvec3(voxel % n, (voxel / n) % n, voxel / (n * n));
	uint seed = i * 2654435761u + uint(fr.frame) * 40503u;
	vec3 jitter = vec3(hash_f(seed), hash_f(seed + 1u), hash_f(seed + 2u));
	vec3 origin = fr.fog_origin.xyz + (vec3(v) + jitter) * fr.fog_origin.w;
	float z = hash_f(seed + 3u) * 2.0 - 1.0;
	float phi = 2.0 * PI * hash_f(seed + 4u);
	vec3 dir = vec3(sqrt(max(0.0, 1.0 - z * z)) * vec2(cos(phi), sin(phi)), z);
	fog_rays[i] = vec4(shade_simple(origin, dir), 1.0);
}

// Reflection ray left unresolved by the screen trace (gi_refl_trace.glsl).
void refl_ray(uint i) {
	if (i >= uint(fr.refl_grid.x * fr.refl_grid.y) || refl[i].origin.w != 0.0) {
		return;
	}
	refl[i].radiance = vec4(shade_simple(refl[i].origin.xyz, refl[i].dir.xyz), 1.0);
	refl[i].origin.w = 1.0;
}

void main() {
	uint ray_index = gl_LaunchIDEXT.x;
	if (fr.refl_grid.z != 0 && ray_index >= uint(fr.refl_grid.w)) {
		refl_ray(ray_index - uint(fr.refl_grid.w));
		return;
	}
	if (ray_index >= uint(fr.fog_grid.w)) {
		if (fr.fog_grid.z != 0) {
			fog_ray(ray_index - uint(fr.fog_grid.w));
		}
		return;
	}
	uint probe_index = ray_index / 32u;
	uint lane = ray_index % 32u;
	uint total = uint(fr.probe_grid.x * fr.probe_grid.y) + min(counter[0], uint(fr.max_adaptive));
	if (probe_index >= total || rays[ray_index].w != 0.0) {
		return; // no ray, or already resolved by the screen trace
	}
	Probe pr = probes[probe_index];
	mat3 v2w = mat3(fr.view_to_world);
	vec3 origin = (fr.view_to_world * vec4(probe_ray_origin(pr), 1.0)).xyz;
	vec3 dir = normalize(v2w * probe_ray_dir(probe_index, lane, pr.normal.xyz));

	vec3 radiance = fr.sky.rgb;
	if (trace_hit(origin, dir)) {
		if (hit.backface > 0.5) {
			radiance = vec3(0.0);
		} else {
			vec3 p = origin + dir * hit.t;
			vec3 n = hit.normal;
			vec3 albedo = hit.albedo;
			vec3 emission = hit.emission;
			vec3 direct = direct_light(p, n);

			// Indirect at p: one cosine bounce ray. Its hit radiance is computed with fresh direct light
			// (shadow rays) plus the cached indirect there, so light changes show up in the first two
			// bounces immediately; only higher bounces go through the cache's temporal accumulation.
			uint seed = ray_index * 9176u + uint(fr.frame) * 26699u;
			vec3 d2 = cosine_dir(n, vec2(hash_f(seed), hash_f(seed + 7u)));
			vec3 o2 = p + n * max(0.002, hit.t * 1e-4);
			vec3 incoming = fr.sky.rgb;
			if (trace_hit(o2, d2)) {
				if (hit.backface > 0.5) {
					incoming = vec3(0.0);
				} else {
					vec3 p2 = o2 + d2 * hit.t;
					vec3 n2 = hit.normal;
					vec3 a2 = hit.albedo;
					vec3 e2 = hit.emission;
					vec4 c2;
					vec3 ind2 = cache_lookup(p2, n2, c2) ? c2.rgb : vec3(0.0);
					incoming = e2 + a2 * (direct_light(p2, n2) + ind2);
				}
			}
			// Cache stores incident indirect radiance (cosine-weighted mean = indirect irradiance / PI).
			cache_insert(p, n, incoming);
			vec3 indirect = incoming;
			vec4 c;
			if (cache_lookup(p, n, c) && c.w >= float(fr.cache_mature_frames)) {
				indirect = c.rgb; // smoothed
			}
			radiance = emission + albedo * (direct + indirect);
		}
	}
	rays[ray_index] = vec4(radiance, 1.0);
}
