// Temporal accumulation in screen space. History is reprojected with depth and validated by
// expected linear depth and normal; accepted history is blended with 1/count (count capped).
//
// Change detection (dual EMA): next to the long history, each pixel keeps a fast luminance EMA
// and second moment over FAST_FRAMES. The noise estimate comes from that short window, so a slow
// lighting trend does not inflate it (a long-window variance around a lagging mean would include
// the trend and blind the detector). Under a lighting change, also a gradual one
// that is far below the noise per frame, the fast EMA follows while the long history lags, so
// their difference grows past k * (noise of the fast EMA) and the history is shortened. Pure noise
// rarely gets there, so static scenes keep the full history (no flicker from the detector).
//
// Writes the GI ambient buffer (rgb = irradiance / PI, i.e. the "ambient light color" the scene
// shader multiplies by albedo) and leaves reflections to the engine (alpha 0).
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 1) uniform sampler2D depth_tex;
layout(set = 0, binding = 2) uniform sampler2D normal_tex;
layout(rgba16f, set = 0, binding = 3) uniform restrict readonly image2D resolved;
layout(set = 0, binding = 4) uniform sampler2D history_prev; // rgb, a = count
layout(set = 0, binding = 5) uniform sampler2D aux_prev; // view normal, linear depth
layout(rgba16f, set = 0, binding = 6) uniform restrict writeonly image2D history_cur;
layout(rgba32f, set = 0, binding = 7) uniform restrict writeonly image2D aux_cur;
layout(rgba16f, set = 0, binding = 8) uniform restrict writeonly image2D ambient_out;
layout(rgba16f, set = 0, binding = 9) uniform restrict writeonly image2D reflection_out;
layout(set = 0, binding = 10) uniform sampler2D moments_prev; // fast EMA: rgb = color, a = luminance 2nd moment
layout(rgba32f, set = 0, binding = 11) uniform restrict writeonly image2D moments_cur;

const float FAST_FRAMES = 4.0;

float lum(vec3 c) {
	return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
	ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pix, fr.size))) {
		return;
	}
	imageStore(reflection_out, pix, vec4(0.0));
	float depth = texelFetch(depth_tex, pix, 0).r;
	if (depth <= 0.0) {
		imageStore(history_cur, pix, vec4(0.0));
		imageStore(aux_cur, pix, vec4(0.0));
		imageStore(moments_cur, pix, vec4(0.0));
		imageStore(ambient_out, pix, vec4(0.0, 0.0, 0.0, 1.0));
		return;
	}
	vec2 uv = (vec2(pix) + 0.5) / vec2(fr.size);
	vec3 p = view_from_depth(uv, depth);
	vec3 n = decode_normal(texelFetch(normal_tex, pix, 0));
	vec4 cur = imageLoad(resolved, pix);
	float cur_l = lum(cur.rgb);

	vec4 result = vec4(cur.rgb, cur.w > 0.0 ? 1.0 : 0.0);
	vec4 mom_out = vec4(cur.rgb, cur_l * cur_l);
	if (fr.has_history != 0 && fr.debug_mode != 1) {
		vec4 pv = fr.cur_to_prev_view * vec4(p, 1.0);
		vec4 pc4 = fr.prev_proj * pv;
		vec2 puv = (pc4.xy / pc4.w) * 0.5 + 0.5;
		if (pc4.w > 0.0 && all(greaterThanEqual(puv, vec2(0.0))) && all(lessThan(puv, vec2(1.0)))) {
			ivec2 ppix = ivec2(puv * vec2(fr.size));
			vec4 aux = texelFetch(aux_prev, ppix, 0);
			vec4 hist = texelFetch(history_prev, ppix, 0);
			vec4 mom = texelFetch(moments_prev, ppix, 0);
			vec3 n_prev_expected = normalize(mat3(fr.cur_to_prev_view) * n);
			float expected_lin = -pv.z;
			bool ok = aux.w > 0.0 && hist.a > 0.0 &&
					abs(aux.w - expected_lin) < expected_lin * 0.05 &&
					dot(n_prev_expected, aux.xyz) > 0.8;
			if (ok && cur.w > 0.0) {
				float count = min(hist.a + 1.0, fr.temporal_max_frames);
				float fa = 1.0 / min(count, FAST_FRAMES);
				vec3 fast = mix(mom.rgb, cur.rgb, fa);
				float fast_m2 = mix(mom.a, cur_l * cur_l, fa);
				float fast_l = lum(fast);
				float slow_l = lum(hist.rgb);
				if (fr.response_sigma > 0.0 && count > fr.response_min_frames) {
					float sigma = sqrt(max(fast_m2 - fast_l * fast_l, 0.0));
					// Std of an EMA with alpha = 1/F over iid samples: sigma * sqrt(alpha / (2 - alpha)).
					float a = 1.0 / FAST_FRAMES;
					float sigma_fast = sigma * sqrt(a / (2.0 - a));
					// Per channel (a hue shift at constant luminance, e.g. a sunset sky, must be caught
					// too), with the luminance noise scaled to each channel.
					// Relative floor 0.25% (was 1%): a 1% floor hid lags of a few % (multi-bounce settling after a
					// light change), which then took the full history to fade.
					vec3 sig_c = sigma_fast * hist.rgb / max(slow_l, 1e-5) + 0.0025 * hist.rgb + 1e-5;
					vec3 dev3 = abs(fast - hist.rgb) / sig_c;
					float dev = max(dev3.r, max(dev3.g, dev3.b));
					if (dev > fr.response_sigma) {
						float r = fr.response_sigma / dev;
						count = max(fr.response_min_frames, count * r * r);
					}
				}
				result = vec4(mix(hist.rgb, cur.rgb, 1.0 / count), count);
				mom_out = vec4(fast, fast_m2);
			} else if (ok) {
				result = hist; // no probe data this frame: keep history
				mom_out = mom;
			}
		}
	}
	imageStore(history_cur, pix, result);
	imageStore(aux_cur, pix, vec4(n, -p.z));
	imageStore(moments_cur, pix, mom_out);
	imageStore(ambient_out, pix, vec4(result.rgb, 1.0));
}
