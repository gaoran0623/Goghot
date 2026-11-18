// Color capture (POST_SKY): keep this frame lit HDR color (before transparents and exposure)
// for next frame screen traces.
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba16f, set = 0, binding = 1) uniform restrict readonly image2D src;
layout(rgba16f, set = 0, binding = 2) uniform restrict writeonly image2D dst;
void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(p, fr.size))) {
		return;
	}
	imageStore(dst, p, imageLoad(src, p));
}
