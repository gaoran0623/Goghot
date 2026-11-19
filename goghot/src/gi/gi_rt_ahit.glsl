// World tracer any hit: alpha test for alpha scissor materials and stochastic transparency for alpha
// blended ones (only those instances are traced non-opaque; opaque geometry never runs this).
hitAttributeEXT vec2 bary;

void main() {
	Geo g = geos[uint(gl_InstanceCustomIndexEXT) + uint(gl_GeometryIndexEXT)];
	float scissor = uintBitsToFloat(g.alpha_scissor);
	if (scissor >= 0.0 && sample_albedo(g, hit_uv(g, uint(gl_PrimitiveID), bary)).a < scissor) {
		ignoreIntersectionEXT;
	}
	if (g.opacity < 1.0) {
		float a = g.opacity * (g.texture != 0xFFFFFFFFu ? sample_albedo(g, hit_uv(g, uint(gl_PrimitiveID), bary)).a : 1.0);
		uint seed = (gl_LaunchIDEXT.x * 73856093u) ^ (uint(gl_PrimitiveID) * 19349663u) ^ (uint(fr.frame) * 83492791u) ^ uint(gl_HitTEXT * 4096.0);
		if (hash_f(seed) >= a) {
			ignoreIntersectionEXT;
		}
	}
}
