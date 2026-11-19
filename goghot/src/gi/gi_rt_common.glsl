// World tracer (hardware ray tracing) shared declarations: set 0 bindings and payloads.
// Prepended after gi_common.glsl and gi_cache.glsl (CACHE_BINDING 9).
layout(set = 0, binding = 1) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 2, std430) restrict readonly buffer Probes { Probe probes[]; };
layout(set = 0, binding = 3, std430) restrict buffer Rays { vec4 rays[]; };
layout(set = 0, binding = 4, std430) restrict readonly buffer Counter { uint counter[4]; };

struct Geo {
	uint index_base; // into indices[]
	uint vertex_base; // into normals[] / uvs[]
	uint texture; // albedo texture slot, 0xFFFFFFFF = none
	uint alpha_scissor; // float bits, < 0 = opaque
	vec4 albedo; // rgb linear albedo (times the texture), w: 1 = double sided
	vec4 emission; // linear radiance
};
layout(set = 0, binding = 5, std430) restrict readonly buffer Geos { Geo geos[]; };
layout(set = 0, binding = 6, std430) restrict readonly buffer Normals { vec4 normals[]; };
layout(set = 0, binding = 7, std430) restrict readonly buffer Indices { uint indices[]; };

struct Light {
	vec4 vec_type; // xyz: direction light travels (directional) or position (omni); w: 0 dir, 1 omni
	vec4 color_range; // rgb: linear color * energy; w: range
	vec4 params; // x: attenuation exponent, y: casts shadow
};
layout(set = 0, binding = 8, std430) restrict readonly buffer Lights { Light lights[]; };
layout(set = 0, binding = 10, std430) restrict writeonly buffer FogRays { vec4 fog_rays[]; };
struct ReflRay {
	vec4 origin; // xyz world origin, w: -1 no ray, 0 needs world trace, 1 resolved
	vec4 dir; // xyz world direction, w: pixel index
	vec4 radiance;
};
layout(set = 0, binding = 11, std430) restrict buffer ReflRays { ReflRay refl[]; };
layout(set = 0, binding = 12, std430) restrict readonly buffer UVs { vec2 uvs[]; };
layout(set = 0, binding = 13) uniform texture2D textures[128];
layout(set = 0, binding = 14) uniform sampler texture_sampler;

// Texture coordinate and albedo texture lookup at a hit (closest and any hit).
vec2 hit_uv(Geo g, uint prim, vec2 bary) {
	uint base = g.index_base + prim * 3u;
	return uvs[indices[base] + g.vertex_base] * (1.0 - bary.x - bary.y) + uvs[indices[base + 1u] + g.vertex_base] * bary.x + uvs[indices[base + 2u] + g.vertex_base] * bary.y;
}

vec4 sample_albedo(Geo g, vec2 uv) {
	if (g.texture == 0xFFFFFFFFu) {
		return vec4(1.0);
	}
	return textureLod(sampler2D(textures[nonuniformEXT(g.texture)], texture_sampler), uv, 0.0);
}

struct HitPayload {
	vec3 normal; // world, shading normal
	float t; // < 0: miss
	vec3 albedo;
	float backface;
	vec3 emission;
	float pad;
};
