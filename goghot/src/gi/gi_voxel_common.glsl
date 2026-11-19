// Tier B world tracer: camera-centered voxel clipmap (fr.vox_params.y cascades of N^3 voxels,
// stacked along Z in one 3D texture per attribute) and a DDA ray march through it.
// Needs VOX_OCC_BINDING and VOX_NORMAL_BINDING (+1 for the second face) defined.
//   occ: occupancy (coverage * 255, max over triangles; alpha tested foliage < 255)
//   normal[0/1]: the voxel's two possible faces, split by the sign of the normal's dominant axis, so
//   a wall thinner than a voxel keeps its inside and outside apart (lit separately, no false back
//   faces, no light leaking through). xyz = n * 0.5 + 0.5, a: 1 = face present.

layout(set = 0, binding = VOX_OCC_BINDING) uniform usampler3D vox_occ_tex;
layout(set = 0, binding = VOX_NORMAL_BINDING) uniform sampler3D vox_normal_tex;
layout(set = 0, binding = VOX_NORMAL_BINDING + 1) uniform sampler3D vox_normal_neg_tex;

float vox_occupancy(ivec3 t) {
	return float(texelFetch(vox_occ_tex, t, 0).r) * (1.0 / 255.0);
}

// Face of voxel t seen by a ray with direction d: 0 / 1, or -1 when only back faces are present.
int vox_face_for_ray(ivec3 t, vec3 d, out vec3 n) {
	vec4 a = texelFetch(vox_normal_tex, t, 0);
	vec4 b = texelFetch(vox_normal_neg_tex, t, 0);
	vec3 na = normalize(a.xyz * 2.0 - 1.0);
	vec3 nb = normalize(b.xyz * 2.0 - 1.0);
	float da = a.a > 0.5 ? dot(na, d) : 1.0;
	float db = b.a > 0.5 ? dot(nb, d) : 1.0;
	if (da >= 0.0 && db >= 0.0) {
		n = a.a > 0.5 ? na : nb;
		return -1;
	}
	if (da <= db) {
		n = na;
		return 0;
	}
	n = nb;
	return 1;
}

int vox_n() {
	return fr.vox_params.x;
}

float vox_size(int c) {
	return fr.vox_origin[c].w;
}

// Voxel coordinate of world position p in cascade c (may be outside [0, N)).
vec3 vox_coord(int c, vec3 p) {
	return (p - fr.vox_origin[c].xyz) / fr.vox_origin[c].w;
}

bool vox_inside(int c, vec3 v, float margin) {
	return all(greaterThanEqual(v, vec3(margin))) && all(lessThan(v, vec3(float(vox_n()) - margin)));
}

int vox_cascade_of(vec3 p) {
	for (int c = 0; c < fr.vox_params.y; c++) {
		if (vox_inside(c, vox_coord(c, p), 1.0)) {
			return c;
		}
	}
	return -1;
}

// March from o along d (normalized) up to tmax. Finest cascade first, coarser once the ray leaves
// it. Hit = first voxel whose occupancy passes `threshold` (alpha-tested foliage uses its coverage).
// Returns t of the hit (< 0: miss) and the hit voxel (cascade + texel).
float vox_trace(vec3 o, vec3 d, float tmax, float threshold, out ivec3 hit_tex, out int hit_c) {
	hit_c = -1;
	hit_tex = ivec3(0);
	int c = vox_cascade_of(o);
	if (c < 0) {
		return -1.0;
	}
	float t = 0.0;
	vec3 inv = 1.0 / max(abs(d), vec3(1e-6)) * sign(d + 1e-12);
	const int MAX_STEPS = 384;
	int steps = 0;
	while (c < fr.vox_params.y && t < tmax && steps < MAX_STEPS) {
		float size = vox_size(c);
		vec3 pc = vox_coord(c, o + d * t);
		if (!vox_inside(c, pc, 0.0)) {
			c++;
			continue;
		}
		ivec3 v = ivec3(floor(pc));
		ivec3 stp = ivec3(sign(d));
		vec3 next = (vec3(v) + step(0.0, d) - pc) * inv; // t (in voxels) to the next boundary per axis
		vec3 delta = abs(inv);
		float tv = 0.0; // t in voxel units since pc
		// Toroidal texel of v (see vox_tex), stepped along with v instead of recomputed per step.
		const int N = vox_n();
		ivec3 w = vox_wrap(v + vox_cell(c));
		const int zoff = c * N;
		while (steps < MAX_STEPS) {
			steps++;
			if (any(lessThan(v, ivec3(0))) || any(greaterThanEqual(v, ivec3(vox_n())))) {
				break; // left this cascade: continue in the next coarser one
			}
			if (vox_occupancy(ivec3(w.xy, w.z + zoff)) > threshold) {
				hit_c = c;
				hit_tex = ivec3(w.xy, w.z + zoff);
				return t + tv * size;
			}
			if (next.x < next.y && next.x < next.z) {
				tv = next.x;
				next.x += delta.x;
				v.x += stp.x;
				w.x = w.x + stp.x == N ? 0 : (w.x + stp.x < 0 ? N - 1 : w.x + stp.x);
			} else if (next.y < next.z) {
				tv = next.y;
				next.y += delta.y;
				v.y += stp.y;
				w.y = w.y + stp.y == N ? 0 : (w.y + stp.y < 0 ? N - 1 : w.y + stp.y);
			} else {
				tv = next.z;
				next.z += delta.z;
				v.z += stp.z;
				w.z = w.z + stp.z == N ? 0 : (w.z + stp.z < 0 ? N - 1 : w.z + stp.z);
			}
			if (t + tv * size > tmax) {
				return -1.0;
			}
		}
		t += tv * size + 1e-3 * size;
		c++;
	}
	return -1.0;
}
