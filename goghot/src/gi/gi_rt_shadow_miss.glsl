// Shadow ray miss: the light is visible.
layout(location = 1) rayPayloadInEXT float shadow_visible;
void main() {
	shadow_visible = 1.0;
}
