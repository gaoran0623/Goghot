// World tracer closest hit: interpolated world normal, per instance-surface material, facing.
// Facing is decided by the shading normal (Godot front faces are clockwise; instances are traced
// with culling disabled), so winding conventions do not matter.
layout(location = 0) rayPayloadInEXT HitPayload hit;
hitAttributeEXT vec2 bary;

void main() {
	Geo g = geos[uint(gl_InstanceCustomIndexEXT) + uint(gl_GeometryIndexEXT)];
	uint base = g.index_base + uint(gl_PrimitiveID) * 3u;
	uint i0 = indices[base] + g.vertex_base;
	uint i1 = indices[base + 1u] + g.vertex_base;
	uint i2 = indices[base + 2u] + g.vertex_base;
	vec3 n = normals[i0].xyz * (1.0 - bary.x - bary.y) + normals[i1].xyz * bary.x + normals[i2].xyz * bary.y;
	n = normalize(transpose(mat3(gl_WorldToObjectEXT)) * n);
	bool back = dot(n, gl_WorldRayDirectionEXT) > 0.0;
	if (back && g.albedo.w > 0.5) { // double sided: the back is a front
		n = -n;
		back = false;
	}
	hit.normal = n;
	hit.t = gl_HitTEXT;
	hit.albedo = g.albedo.rgb * sample_albedo(g, hit_uv(g, uint(gl_PrimitiveID), bary)).rgb;
	hit.emission = g.emission.rgb;
	if (g.emission_texture != 0xFFFFFFFFu) {
		vec3 et = textureLod(sampler2D(textures[nonuniformEXT(g.emission_texture)], texture_sampler), hit_uv(g, uint(gl_PrimitiveID), bary), 0.0).rgb;
		hit.emission = g.emission_multiply != 0u ? g.emission.rgb * et : g.emission.rgb + et * g.emission_energy;
	}
	hit.backface = back ? 1.0 : 0.0;
}
