// Goghot histogram auto exposure. Concatenated after exposure_common.glsl at shader creation.
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba16f, set = 0, binding = 0) uniform restrict image2D color_img;
void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(p, params.size))) { return; }
	vec4 c = imageLoad(color_img, p);
	imageStore(color_img, p, vec4(c.rgb * exp2(-ev), c.a));
}
