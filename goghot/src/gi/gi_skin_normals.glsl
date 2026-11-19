// Deformed (skinned) surfaces: decode the instance's per-frame normals from Godot's deformed vertex
// buffer (positions as 3 floats per vertex, then per vertex `stride` packed uints: oct normal first)
// into the world tracer's normal buffer region of that instance surface.
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std430) restrict readonly buffer Src { uint src[]; };
layout(set = 0, binding = 1, std430) restrict buffer Dst { vec4 normals[]; };
layout(push_constant, std430) uniform PC {
	uint vertex_count;
	uint normal_offset; // in uints: vertex_count * 3
	uint stride; // packed uints per vertex (normal [+ tangent])
	uint dst_base;
}
pc;

vec3 oct_to_vec3(vec2 oct) {
	vec3 v = vec3(oct.xy, 1.0 - abs(oct.x) - abs(oct.y));
	float t = max(-v.z, 0.0);
	v.xy += t * -sign(v.xy);
	return normalize(v);
}

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= pc.vertex_count) {
		return;
	}
	uint packed_n = src[pc.normal_offset + i * pc.stride];
	vec2 oct = vec2((uvec2(packed_n) >> uvec2(0, 16)) & uvec2(0xFFFF)) / 65535.0 * 2.0 - 1.0;
	normals[pc.dst_base + i] = vec4(oct_to_vec3(oct), 0.0);
}
