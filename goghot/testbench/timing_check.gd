# Validates GPU timing: draws a full-screen quad with a tunable ALU loop and compares
# viewport_get_measured_render_time_gpu against GPU-bound wall-clock frame time.
#
#   godot --path goghot/testbench --rendering-driver vulkan --resolution 960x540 --script res://timing_check.gd
extends SceneTree

const FRAMES := 120
const ITERS := [0, 2000, 8000, 32000]

var mat: ShaderMaterial


func _initialize() -> void:
	var root := Node3D.new()
	get_root().add_child(root)
	var cam := Camera3D.new()
	root.add_child(cam)
	cam.current = true
	var sh := Shader.new()
	sh.code = """
shader_type spatial;
render_mode unshaded, cull_disabled;
uniform int iters = 0;
void vertex() { POSITION = vec4(VERTEX.xy * 2.0, 0.5, 1.0); }
void fragment() {
	vec2 p = FRAGCOORD.xy * 0.001;
	float a = 0.0;
	for (int i = 0; i < iters; i++) { a = fract(sin(a * 12.9898 + p.x + float(i)) * 43758.5453 + p.y); }
	ALBEDO = vec3(a);
}
"""
	mat = ShaderMaterial.new()
	mat.shader = sh
	var mi := MeshInstance3D.new()
	var q := QuadMesh.new()
	mi.mesh = q
	mi.material_override = mat
	mi.extra_cull_margin = 16384.0
	root.add_child(mi)
	_run.call_deferred()


func _run() -> void:
	var vp := get_root().get_viewport_rid()
	RenderingServer.viewport_set_measure_render_time(vp, true)
	print("TIMING: iters | viewport_gpu_ms (median) | wall_ms per frame (GPU-bound, median)")
	for it in ITERS:
		mat.set_shader_parameter("iters", it)
		for i in 30:
			await RenderingServer.frame_post_draw
		var gpu := []
		var wall := []
		var t := Time.get_ticks_usec()
		for i in FRAMES:
			await RenderingServer.frame_post_draw
			var now := Time.get_ticks_usec()
			wall.append((now - t) / 1000.0)
			t = now
			gpu.append(RenderingServer.viewport_get_measured_render_time_gpu(vp))
		gpu.sort()
		wall.sort()
		print("TIMING: %6d | %8.3f | %8.3f" % [it, gpu[FRAMES / 2], wall[FRAMES / 2]])
	quit(0)
