// Goghot histogram auto exposure. Concatenated after exposure_common.glsl at shader creation.
layout(local_size_x = 1) in;
void main() {
	float total = 0.0;
	for (int i = 0; i < BINS; i++) { total += float(hist[i]); }
	float lo = total * params.low_pct;
	float hi = total * params.high_pct;
	float acc = 0.0, sum_w = 0.0, sum_log = 0.0;
	for (int i = 0; i < BINS; i++) {
		float c = float(hist[i]);
		// portion of this bin inside [lo, hi]
		float a = clamp(lo - acc, 0.0, c);
		float b = clamp(hi - acc, 0.0, c);
		float w = b - a;
		float center = params.min_log2 + (float(i) + 0.5) / float(BINS) * (params.max_log2 - params.min_log2);
		sum_w += w;
		sum_log += w * center;
		acc += c;
		hist[i] = 0u; // clear for next frame
	}
	float avg_log2 = sum_w > 0.0 ? sum_log / sum_w : log2(params.key);
	// EV here = -log2(scale); scale = key / avg luminance * 2^comp
	float target_ev = clamp(avg_log2 - log2(params.key) - params.comp_ev, params.min_ev, params.max_ev);
	if (initialized < 0.5 || params.snap > 0.5) {
		ev = target_ev;
		initialized = 1.0;
	} else {
		float speed = target_ev > ev ? params.speed_down : params.speed_up;
		ev += (target_ev - ev) * (1.0 - exp(-params.dt * speed));
	}
}
