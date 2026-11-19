// Reflection resolve, full resolution: weighted 3x3 gather of the half-resolution reflection samples
// (plane / normal / roughness similarity), temporal accumulation (history length grows with
// roughness), and the rough fallback: towards refl_params.x the result blends into the probe GI
// (diffuse irradiance / PI is the limit of a fully rough lobe), beyond it the probe GI is used.
// Writes rbgi/reflection (rgb prefiltered incident radiance, alpha 1 = replace the engine's).
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 1) uniform sampler2D depth_tex;
layout(set = 0, binding = 2) uniform sampler2D normal_tex;
struct ReflRay {
	vec4 origin;
	vec4 dir;
	vec4 radiance;
};
layout(set = 0, binding = 3, std430) restrict readonly buffer ReflRays { ReflRay refl[]; };
layout(set = 0, binding = 4) uniform sampler2D history_prev; // rgb, a = count
layout(set = 0, binding = 5) uniform sampler2D aux_prev; // GI aux of the previous frame: view normal, linear depth
layout(rgba16f, set = 0, binding = 6) uniform restrict writeonly image2D history_cur;
layout(rgba16f, set = 0, binding = 7) uniform restrict readonly image2D ambient_in; // this frame's GI
layout(rgba16f, set = 0, binding = 8) uniform restrict writeonly image2D reflection_out;

float decode_roughness(float w) {
	return (w > 0.5 ? 1.0 - w : w) * (255.0 / 127.0);
}

void main() {
	ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pix, fr.size))) {
		return;
	}
	float depth = texelFetch(depth_tex, pix, 0).r;
	if (depth <= 0.0) {
		imageStore(history_cur, pix, vec4(0.0));
		imageStore(reflection_out, pix, vec4(0.0));
		return;
	}
	vec4 nr = texelFetch(normal_tex, pix, 0);
	float rough = decode_roughness(nr.w);
	vec3 gi = imageLoad(ambient_in, pix).rgb;
	float rough_max = fr.refl_params.x;
	if (rough > rough_max) {
		imageStore(history_cur, pix, vec4(0.0));
		imageStore(reflection_out, pix, vec4(gi, 1.0));
		return;
	}
	vec2 uv = (vec2(pix) + 0.5) / vec2(fr.size);
	vec3 p = view_from_depth(uv, depth);
	vec3 n = decode_normal(nr);

	// Spatial gather over the 3x3 half resolution neighbourhood.
	ivec2 q0 = pix / 2;
	vec4 sum = vec4(0.0);
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			ivec2 q = q0 + ivec2(x, y);
			if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, fr.refl_grid.xy))) {
				continue;
			}
			ReflRay s = refl[q.x + q.y * fr.refl_grid.x];
			if (s.origin.w < 0.5) {
				continue;
			}
			int si = int(s.dir.w);
			ivec2 sp = ivec2(si % fr.size.x, si / fr.size.x);
			vec4 snr = texelFetch(normal_tex, sp, 0);
			vec3 spos = view_from_depth((vec2(sp) + 0.5) / vec2(fr.size), texelFetch(depth_tex, sp, 0).r);
			float w = plane_weight(p, n, spos) * pow(max(dot(n, decode_normal(snr)), 0.0), fr.refl_params.w);
			w *= exp(-abs(decode_roughness(snr.w) - rough) * 10.0);
			w *= exp(-dot(vec2(sp - pix), vec2(sp - pix)) / (2.0 * 4.0));
			sum += vec4(s.radiance.rgb, 1.0) * w;
		}
	}
	vec4 cur = sum.w > 1e-4 ? vec4(sum.rgb / sum.w, 1.0) : vec4(0.0);

	// Temporal.
	vec4 result = vec4(cur.rgb, cur.w);
	if (fr.has_history != 0) {
		vec4 pv = fr.cur_to_prev_view * vec4(p, 1.0);
		vec4 pc4 = fr.prev_proj * pv;
		vec2 puv = (pc4.xy / pc4.w) * 0.5 + 0.5;
		if (pc4.w > 0.0 && all(greaterThanEqual(puv, vec2(0.0))) && all(lessThan(puv, vec2(1.0)))) {
			ivec2 ppix = ivec2(puv * vec2(fr.size));
			vec4 aux = texelFetch(aux_prev, ppix, 0);
			vec4 hist = texelFetch(history_prev, ppix, 0);
			bool ok = aux.w > 0.0 && hist.a > 0.0 && abs(aux.w - (-pv.z)) < -pv.z * 0.05 &&
					dot(normalize(mat3(fr.cur_to_prev_view) * n), aux.xyz) > 0.9;
			if (ok) {
				float max_frames = mix(fr.refl_params.y, fr.refl_params.z, clamp(rough / max(rough_max, 1e-3), 0.0, 1.0));
				if (cur.w > 0.0) {
					float count = min(hist.a + 1.0, max_frames);
					result = vec4(mix(hist.rgb, cur.rgb, 1.0 / count), count);
				} else {
					result = hist;
				}
			}
		}
	}
	if (result.a <= 0.0) {
		result = vec4(gi, 0.0); // nothing yet: diffuse GI is the best guess
	}
	imageStore(history_cur, pix, result);
	float to_gi = smoothstep(rough_max - 0.2, rough_max, rough);
	imageStore(reflection_out, pix, vec4(mix(result.rgb, gi, to_gi), 1.0));
}
