// Goghot histogram auto exposure. Concatenated after exposure_common.glsl at shader creation.
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, set = 0, binding = 0) uniform restrict readonly image2D color_img;
shared uint local_hist[BINS];
void main() {
	uint li = gl_LocalInvocationIndex;
	if (li < BINS) { local_hist[li] = 0u; }
	barrier();
	ivec2 p = ivec2(gl_GlobalInvocationID.xy) * params.stride + params.stride / 2;
	float l = all(lessThan(p, params.size)) ? lum(imageLoad(color_img, p).rgb) : 0.0;
	float lg = log2(max(l, 1e-12));
	// Pixels darker than the histogram range (e.g. unlit black) are ignored, not counted as darkest bin.
	if (all(lessThan(p, params.size)) && lg >= params.min_log2) {
		float t = clamp((lg - params.min_log2) / (params.max_log2 - params.min_log2), 0.0, 1.0);
		uint bin = min(uint(t * float(BINS)), uint(BINS - 1));
		vec2 uv = (vec2(p) + 0.5) / vec2(params.size) - 0.5;
		uv.x *= float(params.size.x) / float(params.size.y); // distance in units of screen height
		float w = 1.0 + params.center_weight * exp(-dot(uv, uv) / (2.0 * params.center_sigma * params.center_sigma));
		atomicAdd(local_hist[bin], uint(w * 16.0 + 0.5));
	}
	barrier();
	if (li < BINS && local_hist[li] > 0u) { atomicAdd(hist[li], local_hist[li]); }
}
