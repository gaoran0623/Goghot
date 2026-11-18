# Phase 0 hardware ray tracing smoke test.
# Builds a Cornell box BLAS/TLAS with the RenderingDevice ray tracing API,
# traces primary + shadow rays, and writes the result to a PNG.
#
# Usage:
#   godot --path goghot/demos/rt_smoke --rendering-driver vulkan --script res://rt_smoke.gd -- <out.png>
extends SceneTree

const W := 512
const H := 512

const RT_SHADER := """
#[raygen]
#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba8) uniform restrict writeonly image2D out_image;

layout(location = 0) rayPayloadEXT vec3 payload;

void main() {
	vec2 uv = (vec2(gl_LaunchIDEXT.xy) + 0.5) / vec2(gl_LaunchSizeEXT.xy);
	vec2 ndc = uv * 2.0 - 1.0;
	vec3 origin = vec3(0.0, 1.0, 3.4);
	vec3 dir = normalize(vec3(ndc.x * 0.42, -ndc.y * 0.42, -1.0));
	payload = vec3(0.0);
	traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, 0, 1, 0, origin, 0.001, dir, 100.0, 0);
	vec3 c = payload / (payload + 1.0); // Reinhard
	c = pow(c, vec3(1.0 / 2.2));
	imageStore(out_image, ivec2(gl_LaunchIDEXT.xy), vec4(c, 1.0));
}

#[miss]
#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
	payload = vec3(0.05);
}

#[closest_hit]
#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

struct Tri {
	vec4 normal;
	vec4 albedo; // .w = emission strength
};
layout(set = 0, binding = 2, std430) restrict readonly buffer Tris {
	Tri tris[];
};

layout(location = 0) rayPayloadInEXT vec3 payload;
layout(location = 1) rayPayloadEXT float shadow_visible;

void main() {
	Tri t = tris[gl_PrimitiveID];
	if (t.albedo.w > 0.0) {
		payload = t.albedo.rgb * t.albedo.w;
		return;
	}
	vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
	vec3 n = t.normal.xyz;
	vec3 light_pos = vec3(0.0, 1.97, 0.0);
	vec3 to_l = light_pos - p;
	float dist = length(to_l);
	vec3 l = to_l / dist;
	float ndl = max(dot(n, l), 0.0);

	shadow_visible = 0.0;
	if (ndl > 0.0) {
		traceRayEXT(tlas,
				gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
				0xFF, 0, 1, 1, p + n * 0.001, 0.001, l, dist - 0.05, 1);
	}
	vec3 direct = t.albedo.rgb * ndl * shadow_visible * 6.0 / (dist * dist);
	vec3 ambient = t.albedo.rgb * 0.03;
	payload = direct + ambient;
}
"""

const SHADOW_MISS := """
#[miss]
#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 1) rayPayloadInEXT float shadow_visible;

void main() {
	shadow_visible = 1.0;
}
"""

var verts := PackedVector3Array()
var tri_data := PackedFloat32Array()


func add_quad(a: Vector3, b: Vector3, c: Vector3, d: Vector3, albedo: Color, emission := 0.0) -> void:
	var n := (b - a).cross(c - a).normalized()
	for tri in [[a, b, c], [a, c, d]]:
		for v in tri:
			verts.push_back(v)
		tri_data.append_array([n.x, n.y, n.z, 0.0, albedo.r, albedo.g, albedo.b, emission])


func add_box(center: Vector3, half: Vector3, rot_y: float, albedo: Color) -> void:
	var basis := Basis(Vector3.UP, rot_y)
	var p := func(x: float, y: float, z: float) -> Vector3:
		return center + basis * Vector3(x * half.x, y * half.y, z * half.z)
	add_quad(p.call(-1, 1, 1), p.call(1, 1, 1), p.call(1, 1, -1), p.call(-1, 1, -1), albedo) # top
	add_quad(p.call(-1, -1, 1), p.call(1, -1, 1), p.call(1, 1, 1), p.call(-1, 1, 1), albedo) # front
	add_quad(p.call(1, -1, -1), p.call(-1, -1, -1), p.call(-1, 1, -1), p.call(1, 1, -1), albedo) # back
	add_quad(p.call(-1, -1, -1), p.call(-1, -1, 1), p.call(-1, 1, 1), p.call(-1, 1, -1), albedo) # left
	add_quad(p.call(1, -1, 1), p.call(1, -1, -1), p.call(1, 1, -1), p.call(1, 1, 1), albedo) # right


func build_cornell_box() -> void:
	var white := Color(0.73, 0.73, 0.73)
	# Floor, ceiling, back wall (normals facing inward).
	add_quad(Vector3(-1, 0, 1), Vector3(1, 0, 1), Vector3(1, 0, -1), Vector3(-1, 0, -1), white)
	add_quad(Vector3(-1, 2, -1), Vector3(1, 2, -1), Vector3(1, 2, 1), Vector3(-1, 2, 1), white)
	add_quad(Vector3(-1, 0, -1), Vector3(1, 0, -1), Vector3(1, 2, -1), Vector3(-1, 2, -1), white)
	# Left red, right green.
	add_quad(Vector3(-1, 0, 1), Vector3(-1, 0, -1), Vector3(-1, 2, -1), Vector3(-1, 2, 1), Color(0.65, 0.05, 0.05))
	add_quad(Vector3(1, 0, -1), Vector3(1, 0, 1), Vector3(1, 2, 1), Vector3(1, 2, -1), Color(0.12, 0.45, 0.15))
	# Ceiling light.
	add_quad(Vector3(-0.25, 1.99, -0.25), Vector3(0.25, 1.99, -0.25), Vector3(0.25, 1.99, 0.25), Vector3(-0.25, 1.99, 0.25), Color(1, 0.9, 0.75), 4.0)
	# Boxes.
	add_box(Vector3(0.35, 0.3, 0.35), Vector3(0.3, 0.3, 0.3), -0.3, white)
	add_box(Vector3(-0.35, 0.6, -0.3), Vector3(0.3, 0.6, 0.3), 0.3, white)


func fail(msg: String) -> void:
	printerr("RT_SMOKE: FAIL: ", msg)
	quit(1)


func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	var out_path: String = args[0] if args.size() > 0 else "user://rt_smoke.png"

	var rd := RenderingServer.create_local_rendering_device()
	if rd == null:
		fail("no local RenderingDevice (needs Vulkan, not --headless)")
		return
	print("RT_SMOKE: device = ", rd.get_device_name())
	if not rd.has_feature(RenderingDevice.SUPPORTS_RAYTRACING_PIPELINE):
		fail("SUPPORTS_RAYTRACING_PIPELINE is false")
		return

	build_cornell_box()
	var tri_count := verts.size() / 3
	print("RT_SMOKE: triangles = ", tri_count)

	# Shaders.
	var src := RDShaderSource.new()
	src.set_stage_source(RenderingDevice.SHADER_STAGE_RAYGEN, _stage(RT_SHADER, "raygen"))
	src.set_stage_source(RenderingDevice.SHADER_STAGE_MISS, _stage(RT_SHADER, "miss"))
	src.set_stage_source(RenderingDevice.SHADER_STAGE_CLOSEST_HIT, _stage(RT_SHADER, "closest_hit"))
	var spirv := rd.shader_compile_spirv_from_source(src)
	for stage in [RenderingDevice.SHADER_STAGE_RAYGEN, RenderingDevice.SHADER_STAGE_MISS, RenderingDevice.SHADER_STAGE_CLOSEST_HIT]:
		var err := spirv.get_stage_compile_error(stage)
		if not err.is_empty():
			fail("shader stage %d: %s" % [stage, err])
			return
	var shader := rd.shader_create_from_spirv(spirv)

	var shadow_src := RDShaderSource.new()
	shadow_src.set_stage_source(RenderingDevice.SHADER_STAGE_MISS, _stage(SHADOW_MISS, "miss"))
	var shadow_spirv := rd.shader_compile_spirv_from_source(shadow_src)
	if not shadow_spirv.compile_error_miss.is_empty():
		fail("shadow miss: " + shadow_spirv.compile_error_miss)
		return
	var shadow_shader := rd.shader_create_from_spirv(shadow_spirv)
	if not shader.is_valid() or not shadow_shader.is_valid():
		fail("shader_create_from_spirv failed")
		return

	var ps := func(s: RID) -> RDPipelineShader:
		var p := RDPipelineShader.new()
		p.shader = s
		return p
	var hit_group := RDHitGroup.new()
	hit_group.closest_hit_shader = ps.call(shader)
	var pipeline := rd.raytracing_pipeline_create([ps.call(shader)], [ps.call(shader), ps.call(shadow_shader)], [hit_group], 2)
	if not pipeline.is_valid():
		fail("raytracing_pipeline_create failed")
		return

	# Geometry.
	var vbytes := verts.to_byte_array()
	var vb := rd.vertex_buffer_create(vbytes.size(), vbytes, RenderingDevice.BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT)
	var geom := RDAccelerationStructureGeometry.new()
	geom.flags = RenderingDevice.ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT
	geom.vertex_buffer = vb
	geom.vertex_stride = 12
	geom.vertex_format = RenderingDevice.DATA_FORMAT_R32G32B32_SFLOAT
	geom.vertex_count = verts.size()
	var blas := rd.blas_create([geom], 0)
	var tlas := rd.tlas_create(1, 0)
	if not blas.is_valid() or not tlas.is_valid():
		fail("blas/tlas create failed")
		return
	if rd.blas_build(blas) != OK:
		fail("blas_build failed")
		return

	var hit_sbt := rd.hit_sbt_create(pipeline, 1)
	var inst := RDAccelerationStructureInstance.new()
	inst.blas = blas
	inst.hit_sbt_range = rd.hit_sbt_range_alloc(hit_sbt, 1)
	rd.hit_sbt_range_update(hit_sbt, inst.hit_sbt_range, 0, [0])
	if rd.tlas_build(tlas, [inst]) != OK:
		fail("tlas_build failed")
		return

	# Output image + triangle data.
	var fmt := RDTextureFormat.new()
	fmt.width = W
	fmt.height = H
	fmt.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
	fmt.usage_bits = RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	var tex := rd.texture_create(fmt, RDTextureView.new())
	var tbytes := tri_data.to_byte_array()
	var sb := rd.storage_buffer_create(tbytes.size(), tbytes)

	var u_tlas := RDUniform.new()
	u_tlas.uniform_type = RenderingDevice.UNIFORM_TYPE_ACCELERATION_STRUCTURE
	u_tlas.binding = 0
	u_tlas.add_id(tlas)
	var u_img := RDUniform.new()
	u_img.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_img.binding = 1
	u_img.add_id(tex)
	var u_sb := RDUniform.new()
	u_sb.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_sb.binding = 2
	u_sb.add_id(sb)
	var uset := rd.uniform_set_create([u_tlas, u_img, u_sb], shader, 0)
	if not uset.is_valid():
		fail("uniform_set_create failed")
		return

	var t0 := Time.get_ticks_usec()
	var list := rd.raytracing_list_begin()
	rd.raytracing_list_bind_raytracing_pipeline(list, pipeline)
	rd.raytracing_list_bind_uniform_set(list, uset, 0)
	rd.raytracing_list_trace_rays(list, 0, hit_sbt, W, H, 1)
	rd.raytracing_list_end()
	rd.submit()
	rd.sync()
	print("RT_SMOKE: trace+sync %.2f ms (wall clock, includes AS build)" % ((Time.get_ticks_usec() - t0) / 1000.0))

	var data := rd.texture_get_data(tex, 0)
	var img := Image.create_from_data(W, H, false, Image.FORMAT_RGBA8, data)
	if img.save_png(out_path) != OK:
		fail("could not save " + out_path)
		return

	# Sanity: the red wall should be red, the green wall green.
	var left := img.get_pixel(int(W * 0.04), H / 2)
	var right := img.get_pixel(int(W * 0.96), H / 2)
	print("RT_SMOKE: left wall = ", left, " right wall = ", right)
	var ok := left.r > left.g * 1.3 and right.g > right.r * 1.3
	print("RT_SMOKE: wrote ", ProjectSettings.globalize_path(out_path))
	print("RT_SMOKE: ", "PASS" if ok else "FAIL (unexpected wall colors)")

	for rid in [uset, sb, tex, tlas, blas, vb, pipeline, shadow_shader, shader]:
		rd.free_rid(rid)
	rd.free()
	quit(0 if ok else 1)


func _stage(source: String, name: String) -> String:
	# Extract one "#[name]" section from a multi-stage source string.
	var start := source.find("#[" + name + "]")
	assert(start >= 0)
	start = source.find("\n", start) + 1
	var end := source.find("\n#[", start)
	return source.substr(start, (end if end >= 0 else source.length()) - start)
