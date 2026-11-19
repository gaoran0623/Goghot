// Fog GI volume: average this frame's rays of each voxel (traced by gi_rt_raygen.glsl, mode fog)
// and blend with the previous volume. The volume is world aligned and follows the camera in whole
// voxels; fr.fog_shift is how far it moved since last frame, history outside the old volume is
// dropped. Output rgb = mean incident radiance (all directions, without direct light, which the
// fog shader adds itself); read by Godot volumetric fog via engine patch 3.
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
layout(set = 0, binding = 1, std430) restrict readonly buffer FogRays { vec4 fog_rays[]; };
layout(set = 0, binding = 2) uniform sampler3D history; // previous volume (nearest)
layout(rgba16f, set = 0, binding = 3) uniform restrict writeonly image3D volume_out;

void main() {
	ivec3 v = ivec3(gl_GlobalInvocationID);
	int n = fr.fog_grid.x;
	if (any(greaterThanEqual(v, ivec3(n)))) {
		return;
	}
	uint voxel = uint(v.x + v.y * n + v.z * n * n);
	vec3 sum = vec3(0.0);
	for (int i = 0; i < fr.fog_grid.y; i++) {
		sum += fog_rays[voxel * uint(fr.fog_grid.y) + uint(i)].rgb;
	}
	vec3 fresh = sum / float(max(fr.fog_grid.y, 1));
	ivec3 hv = v + fr.fog_shift.xyz;
	vec4 h = vec4(0.0);
	if (fr.has_history != 0 && all(greaterThanEqual(hv, ivec3(0))) && all(lessThan(hv, ivec3(n)))) {
		h = texelFetch(history, hv, 0);
	}
	float count = min(h.a + 1.0, fr.fog_max_frames);
	imageStore(volume_out, v, vec4(mix(h.rgb, fresh, 1.0 / count), count));
}
