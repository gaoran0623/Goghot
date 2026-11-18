// HZB level 0: copy the depth buffer (reverse Z: larger = closer).
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 1) uniform sampler2D depth_tex;
layout(r32f, set = 0, binding = 2) uniform restrict writeonly image2D dst;
void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(p, fr.size))) {
		return;
	}
	imageStore(dst, p, vec4(texelFetch(depth_tex, p, 0).r));
}
