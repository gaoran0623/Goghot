// Tier B voxelization, deterministic two pass (VOXELIZE_PASS 0 / 1): one workgroup per triangle of
// one instance surface (push constants), its 64 threads splitting the triangle's voxel bounding box
// (big triangles such as ground planes cover thousands of voxels), into every cascade.
// A voxel is touched when the triangle's plane crosses the voxel box and the voxel center projects
// inside the triangle (grown by half a voxel), conservative enough that walls stay closed.
// Each voxel has two face slots (by the sign of the normal's dominant axis, see gi_voxel_common).
//   pass 0: every touching triangle bids for its face slot with imageAtomicMax(rank << 24 | id),
//           rank = how close its plane is to the voxel center; occupancy = max coverage.
//   pass 1: only the winning triangle writes the slot's albedo / normal / emission.
// So the result does not depend on GPU scheduling (last-write-wins would flicker every frame).
// Incremental: each dispatch covers one cascade and only the voxels inside the box pc.box_lo..box_hi
// (cascade local). The volume persists across frames; the CPU clears (pass 2) and re-voxelizes only
// boxes that changed (new clipmap slabs after a scroll, moved / deformed / added / removed draws),
// with every draw overlapping the box. Re-voxelizing the same triangles is idempotent (atomic max of
// the same keys, the same winners write the same values), so overlapping boxes are fine.
layout(local_size_x = 64) in;
#if VOXELIZE_PASS != 2
layout(set = 0, binding = 1, std430) restrict readonly buffer Positions { float positions[]; };
layout(set = 0, binding = 2, std430) restrict readonly buffer Indices { uint indices[]; };
#endif
layout(r32ui, set = 0, binding = 3) uniform restrict uimage3D vox_occ;
layout(r32ui, set = 0, binding = 4) uniform restrict uimage3D vox_key0;
layout(r32ui, set = 0, binding = 5) uniform restrict uimage3D vox_key1;
layout(rgba8, set = 0, binding = 6) uniform restrict writeonly image3D vox_albedo0;
layout(rgba8, set = 0, binding = 7) uniform restrict writeonly image3D vox_albedo1;
layout(rgba8, set = 0, binding = 8) uniform restrict writeonly image3D vox_normal0;
layout(rgba8, set = 0, binding = 9) uniform restrict writeonly image3D vox_normal1;
layout(rgba16f, set = 0, binding = 10) uniform restrict writeonly image3D vox_emission0;
layout(rgba16f, set = 0, binding = 11) uniform restrict writeonly image3D vox_emission1;
// Emitter coverage per face slot (16 bits each, 1.0 = 2048): sum over emissive triangles of the share
// of the voxel they really cover. Conservative voxelization grows a surface by about one voxel, so a
// small emitter (0.15 m cube, 0.125 m voxels) would cover 2-3 voxels per axis and emit several times
// its power; the winner's emission is scaled by min(coverage, 1) to keep the emitted power.
layout(r32ui, set = 0, binding = 16) uniform restrict uimage3D vox_ecov;

layout(push_constant, std430) uniform PC {
	vec4 xform_x; // rows of the instance transform (object -> world)
	vec4 xform_y;
	vec4 xform_z;
	vec4 albedo; // rgb voxel albedo, a coverage
	vec4 emission; // rgb emission, a: double sided
	uint vertex_base;
	uint index_base;
	uint tri_count;
	uint draw_id; // unique per draw this frame (tie-break of the bids)
	ivec4 box_lo; // xyz: first voxel (cascade local), w: cascade
	ivec4 box_hi; // xyz: last voxel, w (pass 2): 1 = also clear the light history
}
pc;

#if VOXELIZE_PASS == 2
layout(rgba16f, set = 0, binding = 12) uniform restrict writeonly image3D lit0a;
layout(rgba16f, set = 0, binding = 13) uniform restrict writeonly image3D lit0b;
layout(rgba16f, set = 0, binding = 14) uniform restrict writeonly image3D lit1a;
layout(rgba16f, set = 0, binding = 15) uniform restrict writeonly image3D lit1b;

void main() {
	ivec3 ext = pc.box_hi.xyz - pc.box_lo.xyz + 1;
	uint k = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * 64u;
	if (k >= uint(ext.x * ext.y * ext.z)) {
		return;
	}
	ivec3 v = pc.box_lo.xyz + ivec3(int(k % uint(ext.x)), int((k / uint(ext.x)) % uint(ext.y)), int(k / uint(ext.x * ext.y)));
	ivec3 t = vox_tex(pc.box_lo.w, v);
	imageStore(vox_occ, t, uvec4(0u));
	imageStore(vox_key0, t, uvec4(0u));
	imageStore(vox_key1, t, uvec4(0u));
	imageStore(vox_albedo0, t, vec4(0.0));
	imageStore(vox_albedo1, t, vec4(0.0));
	imageStore(vox_normal0, t, vec4(0.0));
	imageStore(vox_normal1, t, vec4(0.0));
	imageStore(vox_emission0, t, vec4(0.0));
	imageStore(vox_emission1, t, vec4(0.0));
	imageStore(vox_ecov, t, uvec4(0u));
	if (pc.box_hi.w == 1) {
		imageStore(lit0a, t, vec4(0.0));
		imageStore(lit0b, t, vec4(0.0));
		imageStore(lit1a, t, vec4(0.0));
		imageStore(lit1b, t, vec4(0.0));
	}
}
#else

vec3 world_pos(uint i) {
	uint b = (pc.vertex_base + i) * 3u;
	vec4 p = vec4(positions[b], positions[b + 1u], positions[b + 2u], 1.0);
	return vec3(dot(pc.xform_x, p), dot(pc.xform_y, p), dot(pc.xform_z, p));
}

void face_write(int slot, ivec3 t, uint key, vec3 face_n) {
#if VOXELIZE_PASS == 0
	if (slot == 0) {
		imageAtomicMax(vox_key0, t, key);
	} else {
		imageAtomicMax(vox_key1, t, key);
	}
#else
	uint owner = slot == 0 ? imageLoad(vox_key0, t).r : imageLoad(vox_key1, t).r;
	if (owner != key) {
		return;
	}
	vec4 nrm = vec4(face_n * 0.5 + 0.5, 1.0);
	uint ec = imageLoad(vox_ecov, t).r;
	vec3 emission = pc.emission.rgb * min(float(slot == 0 ? (ec & 0xFFFFu) : (ec >> 16u)) * (1.0 / 2048.0), 1.0);
	if (slot == 0) {
		imageStore(vox_albedo0, t, vec4(pc.albedo.rgb, 1.0));
		imageStore(vox_normal0, t, nrm);
		imageStore(vox_emission0, t, vec4(emission, 1.0));
	} else {
		imageStore(vox_albedo1, t, vec4(pc.albedo.rgb, 1.0));
		imageStore(vox_normal1, t, nrm);
		imageStore(vox_emission1, t, vec4(emission, 1.0));
	}
#endif
}

void main() {
	uint tri = gl_WorkGroupID.x;
	uint lane = gl_LocalInvocationID.x;
	if (tri >= pc.tri_count) {
		return;
	}
	uint ib = pc.index_base + tri * 3u;
	vec3 a = world_pos(indices[ib]);
	vec3 b = world_pos(indices[ib + 1u]);
	vec3 c = world_pos(indices[ib + 2u]);
	vec3 n = cross(b - a, c - a);
	float area2 = length(n);
	if (area2 < 1e-12) {
		return;
	}
	n /= area2;
	n = -n; // Godot front faces are clockwise: this winding's normal points inwards
	vec3 an = abs(n);
	int axis = (an.x > an.y && an.x > an.z) ? 0 : (an.y > an.z ? 1 : 2);
	bool positive = n[axis] >= 0.0;
	bool double_sided = pc.emission.a > 0.5;
	bool emissive = dot(pc.emission.rgb, vec3(1.0)) > 0.0;
	uint id = ((pc.draw_id & 0xFFu) << 16u) | (tri & 0xFFFFu);
	{
		int cas = pc.box_lo.w;
		float size = fr.vox_origin[cas].w;
		vec3 org = fr.vox_origin[cas].xyz;
		vec3 lo = (min(a, min(b, c)) - org) / size - 0.5;
		vec3 hi = (max(a, max(b, c)) - org) / size + 0.5;
		ivec3 vlo = max(ivec3(floor(lo)), pc.box_lo.xyz);
		ivec3 vhi = min(ivec3(floor(hi)), pc.box_hi.xyz);
		if (any(greaterThan(vlo, vhi))) {
			return;
		}
		float half_extent = 0.5 * size * (an.x + an.y + an.z); // voxel box projected on n
		float grow = 0.5 * size;
		ivec3 ext = vhi - vlo + 1;
		uint count = uint(ext.x * ext.y * ext.z);
		vec3 e0 = cross(n, b - a), e1 = cross(n, c - b), e2 = cross(n, a - c);
		float l0 = length(e0), l1 = length(e1), l2 = length(e2);
		for (uint k = lane; k < count; k += 64u) {
			int x = vlo.x + int(k % uint(ext.x));
			int y = vlo.y + int((k / uint(ext.x)) % uint(ext.y));
			int z = vlo.z + int(k / uint(ext.x * ext.y));
			vec3 p = org + (vec3(x, y, z) + 0.5) * size;
			float d = dot(n, p - a);
			if (abs(d) > half_extent) {
				continue;
			}
			vec3 q = p - d * n;
			bool inside = dot(q - a, e0) >= -grow * l0 && dot(q - b, e1) >= -grow * l1 && dot(q - c, e2) >= -grow * l2;
			bool inside_flipped = dot(q - a, e0) <= grow * l0 && dot(q - b, e1) <= grow * l1 && dot(q - c, e2) <= grow * l2;
			if (!inside && !inside_flipped) {
				continue;
			}
			ivec3 t = vox_tex(cas, ivec3(x, y, z));
			uint rank = 255u - uint(clamp(abs(d) / max(half_extent, 1e-6), 0.0, 1.0) * 254.0);
			uint key = (rank << 24u) | id;
#if VOXELIZE_PASS == 0
			imageAtomicMax(vox_occ, t, uint(clamp(pc.albedo.a, 0.0, 1.0) * 255.0));
			if (emissive) {
				// Share of the voxel covered: per edge, a half plane over a voxel wide square (summing
				// to 1 across a shared edge), capped by the triangle's own area.
				float sg = inside ? 1.0 : -1.0;
				float cov = clamp(0.5 + sg * dot(q - a, e0) / (l0 * size), 0.0, 1.0) * clamp(0.5 + sg * dot(q - b, e1) / (l1 * size), 0.0, 1.0) *
						clamp(0.5 + sg * dot(q - c, e2) / (l2 * size), 0.0, 1.0);
				uint cq = uint(min(cov, 0.5 * area2 / (size * size)) * 2048.0 + 0.5);
				if (positive || double_sided) {
					imageAtomicAdd(vox_ecov, t, cq);
				}
				if (!positive || double_sided) {
					imageAtomicAdd(vox_ecov, t, cq << 16u);
				}
			}
#endif
			if (positive || double_sided) {
				face_write(0, t, key, positive ? n : -n);
			}
			if (!positive || double_sided) {
				face_write(1, t, key, positive ? -n : n);
			}
		}
	}
}
#endif
