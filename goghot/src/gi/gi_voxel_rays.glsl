// Tier B: resolves the same rays as the hardware tracer (gi_rt_raygen.glsl) by marching the voxel
// clipmap: probe rays [0, fog_grid.w), fog rays [fog_grid.w, refl_grid.w), reflection rays after.
// Hit radiance = the hit voxel's lit value (emission + albedo * (direct + cached indirect)); back
// faces (no face towards the ray) are black; misses see the sky.
// Included after gi_voxel_common.glsl (bindings 1-3).
layout(local_size_x = 64) in;
layout(set = 0, binding = 5) uniform sampler3D lit_tex;
layout(set = 0, binding = 6) uniform sampler3D lit_neg_tex;
layout(set = 0, binding = 8, std430) restrict readonly buffer Probes { Probe probes[]; };
layout(set = 0, binding = 9, std430) restrict buffer Rays { vec4 rays[]; };
layout(set = 0, binding = 10, std430) restrict readonly buffer Counter { uint counter[4]; };
layout(set = 0, binding = 11, std430) restrict writeonly buffer FogRays { vec4 fog_rays[]; };
struct ReflRay {
	vec4 origin;
	vec4 dir;
	vec4 radiance;
};
layout(set = 0, binding = 12, std430) restrict buffer ReflRays { ReflRay refl[]; };

vec3 shade_voxel_ray(vec3 o, vec3 d, float offset) {
	ivec3 ht;
	int hc;
	vec3 start = o + d * offset;
	// A voxel holding only faces that look away from the ray is usually an edge voxel of a thin
	// surface whose facing side landed in the next voxel (conservative voxelization): step past it
	// a few times instead of returning black. Rays start off their surface, so this is not the
	// "started inside geometry" case.
	for (int k = 0; k < 4; k++) {
		float t = vox_trace(start, d, 1e4, 0.5, ht, hc);
		if (t < 0.0) {
			return sky_radiance(d);
		}
		vec3 hn;
		int face = vox_face_for_ray(ht, d, hn);
		if (face >= 0) {
			return face == 0 ? texelFetch(lit_tex, ht, 0).rgb : texelFetch(lit_neg_tex, ht, 0).rgb;
		}
		start += d * (t + vox_size(hc) * 1.01);
	}
	return vec3(0.0);
}

void main() {
	uint ray_index = gl_GlobalInvocationID.x;
	if (fr.refl_grid.z != 0 && ray_index >= uint(fr.refl_grid.w)) {
		uint i = ray_index - uint(fr.refl_grid.w);
		if (i < uint(fr.refl_grid.x * fr.refl_grid.y) && refl[i].origin.w == 0.0) {
			int c = vox_cascade_of(refl[i].origin.xyz);
			float size = c >= 0 ? vox_size(c) : 0.0;
			// Off the surface along its normal first (stored by gi_refl_trace.glsl): grazing reflections
			// on a floor would otherwise start inside the floor's own voxels.
			vec3 nw = refl[i].radiance.xyz;
			refl[i].radiance = vec4(shade_voxel_ray(refl[i].origin.xyz + nw * size, refl[i].dir.xyz, size * 0.5), 1.0);
			refl[i].origin.w = 1.0;
		}
		return;
	}
	if (ray_index >= uint(fr.fog_grid.w)) {
		if (fr.fog_grid.z == 0) {
			return;
		}
		uint i = ray_index - uint(fr.fog_grid.w);
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
		fog_rays[i] = vec4(shade_voxel_ray(origin, dir, 0.0), 1.0);
		return;
	}
	uint probe_index = ray_index / 32u;
	uint lane = ray_index % 32u;
	uint total = uint(fr.probe_grid.x * fr.probe_grid.y) + min(counter[0], uint(fr.max_adaptive));
	if (probe_index >= total || rays[ray_index].w != 0.0) {
		return;
	}
	Probe pr = probes[probe_index];
	vec3 origin = (fr.view_to_world * vec4(probe_ray_origin(pr), 1.0)).xyz;
	vec3 dir = normalize(mat3(fr.view_to_world) * probe_ray_dir(probe_index, lane, pr.normal.xyz));
	vec3 nw = normalize(mat3(fr.view_to_world) * pr.normal.xyz);
	int c = vox_cascade_of(origin);
	float size = c >= 0 ? vox_size(c) : 0.0;
	// Leave the surface's own (conservatively grown) voxels along the normal first: an offset along
	// the ray alone keeps grazing rays inside the floor they start on.
	rays[ray_index] = vec4(shade_voxel_ray(origin + nw * size, dir, size * 0.5), 1.0);
}
