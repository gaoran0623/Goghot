// HZB level N from N-1: closest depth (max, reverse Z) of the 2x2 footprint, plus the extra
// row/column when the source size is odd so no texel is dropped.
layout(local_size_x = 8, local_size_y = 8) in;
layout(r32f, set = 0, binding = 1) uniform restrict readonly image2D src;
layout(r32f, set = 0, binding = 2) uniform restrict writeonly image2D dst;
void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	ivec2 dsize = imageSize(dst);
	ivec2 ssize = imageSize(src);
	if (any(greaterThanEqual(p, dsize))) {
		return;
	}
	ivec2 s = p * 2;
	ivec2 ext = ivec2(1);
	if (p.x == dsize.x - 1 && (ssize.x & 1) == 1) {
		ext.x = 2;
	}
	if (p.y == dsize.y - 1 && (ssize.y & 1) == 1) {
		ext.y = 2;
	}
	float d = 0.0;
	for (int y = 0; y <= ext.y; y++) {
		for (int x = 0; x <= ext.x; x++) {
			d = max(d, imageLoad(src, min(s + ivec2(x, y), ssize - 1)).r);
		}
	}
	imageStore(dst, p, vec4(d));
}
