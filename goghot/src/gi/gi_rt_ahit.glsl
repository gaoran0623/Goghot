// World tracer any hit: alpha test for alpha scissor materials (only alpha tested instances are
// traced non-opaque, so this runs for foliage cards and the like, never for opaque geometry).
hitAttributeEXT vec2 bary;

void main() {
	Geo g = geos[uint(gl_InstanceCustomIndexEXT) + uint(gl_GeometryIndexEXT)];
	float scissor = uintBitsToFloat(g.alpha_scissor);
	if (scissor >= 0.0 && sample_albedo(g, hit_uv(g, uint(gl_PrimitiveID), bary)).a < scissor) {
		ignoreIntersectionEXT;
	}
}
