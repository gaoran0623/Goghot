// World tracer miss: no geometry (the raygen uses the sky radiance).
layout(location = 0) rayPayloadInEXT HitPayload hit;
void main() {
	hit.t = -1.0;
}
