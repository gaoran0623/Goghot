// World tracer (hardware ray tracing) shared declarations: set 0 bindings and payloads.
// Prepended after gi_common.glsl and gi_cache.glsl (CACHE_BINDING 9).
layout(set = 0, binding = 1) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 2, std430) restrict readonly buffer Probes { Probe probes[]; };
layout(set = 0, binding = 3, std430) restrict buffer Rays { vec4 rays[]; };
layout(set = 0, binding = 4, std430) restrict readonly buffer Counter { uint counter[4]; };

struct Geo {
	uint index_base; // into indices[]
	uint vertex_base; // into normals[]
	uint pad0;
	uint pad1;
	vec4 albedo; // linear diffuse albedo
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

struct HitPayload {
	vec3 normal; // world, shading normal
	float t; // < 0: miss
	vec3 albedo;
	float backface;
	vec3 emission;
	float pad;
};
