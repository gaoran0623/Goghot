# Renders one test-bench scene along its fixed camera path and writes PNG frames + timings.
#
# Usage (normally called by tools/run.py):
#   godot --path goghot/testbench --rendering-driver vulkan --resolution 960x540 \
#         --script res://capture.gd -- <scene.json> <out_dir>
#         [gi_mode] [exposure]
# gi_mode: baseline (flat ambient, default) | inject_identity | inject_red (engine patch 1 checks) | screen_probe (phase 2) | rt (phase 3) | node (phase 6: GoghotGI node)
#          | godot_ssil (SSIL + SSAO) | godot_sdfgi (SDFGI) | godot_voxelgi (VoxelGI baked at load; dynamic objects not baked)
#          native Godot GI, default settings, for comparison
# exposure: fixed (default, exposure 1, comparable to Cycles) | auto (histogram auto exposure, GDScript proto) | auto_cpp (GDExtension)
#
# Scene JSON is shared with tools/blender_ref.py so Godot and Cycles build the same scene.
# Light units (non-physical Godot mode):
#   sun.energy E   -> Godot DirectionalLight3D.light_energy = E   (Cycles sun strength = pi * E)
#   point.energy E -> Godot OmniLight3D energy E, inverse-square   (Cycles point power = 4 * pi^2 * E)
#   emission c * s -> radiance c * s in both
#   sky color L    -> background radiance L in both
extends SceneTree

const WARMUP_FRAMES := 90 # long enough for GI history + multi-bounce feedback to converge (flicker = steady state)

var spec: Dictionary
var camera: Camera3D
var env: Environment
var light_nodes := [] # [Light3D, spec dict]
var skinned := [] # [Skeleton3D, object spec, mesh spec]
var gi_mode := "baseline"
var exposure_mode := "fixed"
var inject_effect: CompositorEffect
var exposure_effect: CompositorEffect
var gi_effect: CompositorEffect
var gi_params := {} # property overrides for the GI effect (bench experiments only)


func _v3(a: Array) -> Vector3:
	return Vector3(a[0], a[1], a[2])


# Spec colors are linear (like Cycles). Godot Color properties are sRGB-encoded, so convert.
func _col(a: Array) -> Color:
	return Color(a[0], a[1], a[2]).linear_to_srgb()


# glTF object: {"type": "gltf", "path": <under assets/>, "center", "rot_y"}. Materials are made pure
# diffuse like the other scenes and blender_ref.py: Lambert, metallic 0, specular 0 (albedo texture,
# normal map, alpha mask and double sidedness kept).
func _load_gltf(o: Dictionary) -> Node3D:
	var path := ProjectSettings.globalize_path("res://assets/" + str(o.path))
	var doc := GLTFDocument.new()
	var state := GLTFState.new()
	var err := doc.append_from_file(path, state)
	if err != OK:
		push_error("glTF load failed: " + path)
		return Node3D.new()
	var scn: Node3D = doc.generate_scene(state)
	scn.position = _v3(o.get("center", [0, 0, 0]))
	scn.rotation_degrees.y = o.get("rot_y", 0.0)
	for mi: MeshInstance3D in scn.find_children("*", "MeshInstance3D", true, false):
		for i in mi.mesh.get_surface_count():
			var m := mi.mesh.surface_get_material(i) as BaseMaterial3D
			if m == null:
				continue
			m.diffuse_mode = BaseMaterial3D.DIFFUSE_LAMBERT
			m.metallic = 0.0
			m.metallic_texture = null
			m.metallic_specular = 0.0
	return scn


# multimesh: boxes of one size, "instances": [[x, y, z, rot_y], ...]
# csg: box "size" at "center" minus box "holes": [{"center", "size"}] (CSGBox3D subtraction)
# gridmap: box "item_size" in cells "cells": [[i, j, k], ...] of "cell_size" (centered), node at "center"
func _build_special(o: Dictionary, mat: Material) -> Node3D:
	match o.type:
		"multimesh":
			var bm := BoxMesh.new()
			bm.size = _v3(o.size)
			var mm := MultiMesh.new()
			mm.transform_format = MultiMesh.TRANSFORM_3D
			mm.mesh = bm
			mm.instance_count = o.instances.size()
			for i in o.instances.size():
				var t: Array = o.instances[i]
				mm.set_instance_transform(i, Transform3D(Basis(Vector3.UP, deg_to_rad(t[3])), Vector3(t[0], t[1], t[2])))
			var mmi := MultiMeshInstance3D.new()
			mmi.multimesh = mm
			mmi.material_override = mat
			return mmi
		"csg":
			var c := CSGBox3D.new()
			c.size = _v3(o.size)
			c.position = _v3(o.center)
			c.material = mat
			for hole: Dictionary in o.get("holes", []):
				var h := CSGBox3D.new()
				h.size = _v3(hole.size)
				h.position = _v3(hole.center) - _v3(o.center)
				h.operation = CSGShape3D.OPERATION_SUBTRACTION
				c.add_child(h)
			return c
		"gridmap":
			var lib := MeshLibrary.new()
			var bm2 := BoxMesh.new()
			bm2.size = _v3(o.item_size)
			bm2.material = mat
			lib.create_item(0)
			lib.set_item_mesh(0, bm2)
			var gm := GridMap.new()
			gm.mesh_library = lib
			gm.cell_size = _v3(o.cell_size)
			gm.position = _v3(o.get("center", [0, 0, 0]))
			for c2: Array in o.cells:
				gm.set_cell_item(Vector3i(int(c2[0]), int(c2[1]), int(c2[2])), 0)
			return gm
	return Node3D.new()


func _local_to_root(n: Node3D, root: Node3D) -> Transform3D:
	var t := n.transform
	var p := n.get_parent()
	while p != null and p != root:
		if p is Node3D:
			t = p.transform * t
		p = p.get_parent()
	return t


func _build(root: Node3D) -> void:
	var mats := {}
	for name in spec.materials:
		var m: Dictionary = spec.materials[name]
		var mat := StandardMaterial3D.new()
		mat.albedo_color = _col(m.get("albedo", [0.8, 0.8, 0.8]))
		mat.roughness = m.get("roughness", 1.0)
		mat.metallic = m.get("metallic", 0.0)
		mat.metallic_specular = 0.0 if m.get("specular", 0.0) == 0.0 else 0.5
		mat.diffuse_mode = BaseMaterial3D.DIFFUSE_LAMBERT # Cycles Principled diffuse is Lambertian (Godot default is Burley)
		if m.has("albedo_texture"):
			var img := Image.load_from_file(ProjectSettings.globalize_path("res://assets/" + str(m.albedo_texture)))
			mat.albedo_texture = ImageTexture.create_from_image(img)
			mat.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR # no mipmaps: Cycles does not mip either
		if m.has("alpha_scissor"):
			mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR
			mat.alpha_scissor_threshold = m.alpha_scissor
		if m.get("double_sided", false):
			mat.cull_mode = BaseMaterial3D.CULL_DISABLED
		if m.has("emission"):
			mat.emission_enabled = true
			mat.emission = _col(m.emission)
			mat.emission_energy_multiplier = m.get("emission_strength", 1.0)
		if m.has("emission_texture"): # emission = (emission + texture) * strength (Godot's add operator)
			mat.emission_enabled = true
			if not m.has("emission"):
				mat.emission = Color(0, 0, 0)
			mat.emission_energy_multiplier = m.get("emission_strength", 1.0)
			var eimg := Image.load_from_file(ProjectSettings.globalize_path("res://assets/" + str(m.emission_texture)))
			mat.emission_texture = ImageTexture.create_from_image(eimg)
			mat.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR
		mats[name] = mat

	for o: Dictionary in spec.objects:
		if o.type == "gltf":
			root.add_child(_load_gltf(o))
			continue
		if o.type == "multimesh" or o.type == "csg" or o.type == "gridmap":
			root.add_child(_build_special(o, mats[o.material]))
			continue
		var mi := MeshInstance3D.new()
		match o.type:
			"box":
				var bm := BoxMesh.new()
				bm.size = _v3(o.size)
				mi.mesh = bm
			"skinned":
				_build_skinned(mi, o)
			"quad":
				# QuadMesh faces local +Z; spec gives the world normal and up (right = up x normal).
				var qm := QuadMesh.new()
				qm.size = Vector2(o.size[0], o.size[1])
				mi.mesh = qm
			_:
				push_error("unknown object type " + str(o.type))
				continue
		mi.material_override = mats[o.material]
		mi.position = _v3(o.center)
		if o.type == "quad":
			var nrm := _v3(o.normal).normalized()
			var right := _v3(o.up).cross(nrm).normalized()
			mi.basis = Basis(right, nrm.cross(right), nrm)
		else:
			mi.rotation_degrees.y = o.get("rot_y", 0.0)
		root.add_child(mi)

	for l: Dictionary in spec.lights:
		var node: Light3D
		match l.type:
			"sun":
				var d := DirectionalLight3D.new()
				d.light_angular_distance = 0.5
				node = d
			"point":
				var p := OmniLight3D.new()
				p.omni_attenuation = 2.0 # inverse-square, to match Cycles
				p.omni_range = 100.0
				node = p
			"spot":
				# Cone half angle "angle"; edge falloff 1 - rim^(1 / spot_attenuation), nearly hard with the
				# default 0.05 (Cycles: spot blend "blend", default 0.05).
				var sp := SpotLight3D.new()
				sp.spot_attenuation = 2.0 # inverse-square distance falloff
				sp.spot_range = 100.0
				sp.spot_angle = l.angle
				sp.spot_angle_attenuation = l.get("cone_attenuation", 0.05)
				node = sp
			_:
				push_error("unknown light type " + str(l.type))
				continue
		node.shadow_enabled = true
		root.add_child(node)
		light_nodes.append([node, l])

	env = Environment.new()
	env.background_mode = Environment.BG_COLOR
	# Baseline "no GI": flat, unoccluded ambient equal to the sky radiance. GI work replaces this.
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_energy = 1.0
	env.ambient_light_sky_contribution = 0.0 # no Sky resource: otherwise fog ambient would read a black sky
	env.reflected_light_source = Environment.REFLECTION_SOURCE_DISABLED
	env.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	env.tonemap_exposure = 1.0
	if spec.has("fog"):
		# Godot: extinction = density, scattering = albedo * density (matches Cycles scatter+absorption).
		var fog: Dictionary = spec.fog
		env.volumetric_fog_enabled = true
		env.volumetric_fog_density = fog.density
		env.volumetric_fog_albedo = _col(fog.get("albedo", [1, 1, 1]))
		env.volumetric_fog_anisotropy = fog.get("anisotropy", 0.0)
		env.volumetric_fog_length = fog.get("length", 150.0)
		env.volumetric_fog_ambient_inject = 1.0
		env.volumetric_fog_sky_affect = 1.0
		env.volumetric_fog_temporal_reprojection_enabled = true
	var we := WorldEnvironment.new()
	we.environment = env
	var effects := []
	if gi_mode.begins_with("inject_"):
		# Engine patch 1 validation (see gi_inject_test.gd).
		inject_effect = load("res://gi_inject_test.gd").new()
		effects.append(inject_effect)
	elif gi_mode == "screen_probe" or gi_mode == "rt":
		# screen_probe: phase 2 (screen trace + sky). rt: phase 3 (+ hardware RT world tracer and
		# radiance cache, scene fed by a GoghotGIScene node).
		gi_effect = ClassDB.instantiate("GoghotScreenProbeGI")
		gi_effect.world_tracing = gi_mode == "rt"
		for k in gi_params:
			gi_effect.set(k, gi_params[k])
		effects.append(gi_effect)
		effects.append(gi_effect.create_color_capture())
		if gi_mode == "rt":
			var scene_node: Node = ClassDB.instantiate("GoghotGIScene")
			scene_node.gi = gi_effect
			root.add_child(scene_node)
	elif gi_mode == "node":
		# Phase 6: one GoghotGI node does all the wiring (compositor, scene collection, sky).
		var node: Node = ClassDB.instantiate("GoghotGI")
		for k in gi_params: # node properties (quality, tracer, debug_view, auto_exposure)
			node.set(k, gi_params[k])
		root.add_child(node)
	elif gi_mode == "godot_ssil":
		env.ssil_enabled = true
		env.ssao_enabled = true
	elif gi_mode == "godot_sdfgi":
		env.sdfgi_enabled = true
	elif gi_mode == "godot_voxelgi" or gi_mode == "godot_voxelgi_interior":
		var vgi := VoxelGI.new()
		var box := AABB()
		var first := true
		for c: VisualInstance3D in root.find_children("*", "VisualInstance3D", true, false):
			if c is GeometryInstance3D:
				var a: AABB = _local_to_root(c, root) * c.get_aabb()
				box = a if first else box.merge(a)
				first = false
		for gmap: GridMap in root.find_children("*", "GridMap", true, false):
			for cell: Vector3i in gmap.get_used_cells():
				var a2 := AABB(gmap.position + gmap.map_to_local(cell) - gmap.cell_size * 0.5, gmap.cell_size)
				box = a2 if first else box.merge(a2)
				first = false
		box = box.grow(0.5)
		vgi.position = box.get_center()
		vgi.size = box.size
		root.add_child(vgi)
		if gi_mode == "godot_voxelgi_interior": # VoxelGIData.interior: no sky light through the probe
			var data := VoxelGIData.new()
			data.interior = true
			vgi.data = data # bake() reuses existing data and keeps its settings
		vgi.bake.call_deferred() # needs the scene in the tree
	elif gi_mode != "baseline":
		push_error("unknown gi mode " + gi_mode)
	if exposure_mode == "auto" or exposure_mode == "auto_cpp":
		if exposure_mode == "auto_cpp":
			exposure_effect = ClassDB.instantiate("GoghotHistogramExposure") # GDExtension (addons/goghot)
		else:
			exposure_effect = load("res://proto/histogram_exposure.gd").new() # GDScript prototype
		exposure_effect.fixed_dt = 1.0 / 60.0 # deterministic adaptation per captured frame
		effects.append(exposure_effect)
	elif exposure_mode != "fixed":
		push_error("unknown exposure mode " + exposure_mode)
	if not effects.is_empty():
		var comp := Compositor.new()
		comp.compositor_effects = effects
		we.compositor = comp
	root.add_child(we)
	_apply_frame(0)

	camera = Camera3D.new()
	camera.fov = spec.camera.get("fov_y", 50.0)
	camera.near = 0.05
	camera.far = 500.0
	root.add_child(camera)
	camera.current = true


# Skinned prism ("skinned_meshes" arrays + bone chain along +Y, bone k rest at k * bone_step).
func _build_skinned(mi: MeshInstance3D, o: Dictionary) -> void:
	var m: Dictionary = spec.skinned_meshes[o.mesh]
	var arr := []
	arr.resize(Mesh.ARRAY_MAX)
	var v := PackedVector3Array()
	var n := PackedVector3Array()
	for p in m.vertices:
		v.push_back(Vector3(p[0], p[1], p[2]))
	for p in m.normals:
		n.push_back(Vector3(p[0], p[1], p[2]))
	var b := PackedInt32Array()
	var w := PackedFloat32Array()
	for i in m.bones.size():
		for k in 4:
			b.push_back(int(m.bones[i][k]))
			w.push_back(float(m.weights[i][k]))
	arr[Mesh.ARRAY_VERTEX] = v
	arr[Mesh.ARRAY_NORMAL] = n
	arr[Mesh.ARRAY_BONES] = b
	arr[Mesh.ARRAY_WEIGHTS] = w
	arr[Mesh.ARRAY_INDEX] = PackedInt32Array(m.indices)
	var am := ArrayMesh.new()
	am.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arr)
	mi.mesh = am
	var sk := Skeleton3D.new()
	var skin := Skin.new()
	for k in int(m.bone_count):
		sk.add_bone("b%d" % k)
		if k > 0:
			sk.set_bone_parent(k, k - 1)
			sk.set_bone_rest(k, Transform3D(Basis(), Vector3(0, m.bone_step, 0)))
		skin.add_bind(k, Transform3D(Basis(), Vector3(0, -k * m.bone_step, 0))) # inverse rest global
	sk.reset_bone_poses()
	mi.add_child(sk)
	mi.skin = skin
	mi.skeleton = NodePath(sk.name)
	skinned.append([sk, o, m])


func _apply_skinned(f: int) -> void:
	var hold: int = spec.camera.get("hold_frames", 0)
	for s in skinned:
		var sk: Skeleton3D = s[0]
		var a: Dictionary = s[1].anim
		for k in int(s[2].bone_count):
			var ang := deg_to_rad(a.amplitude) * sin(a.speed * max(f - hold, 0) + a.phase + 0.6 * k) if f >= hold else deg_to_rad(a.amplitude) * sin(a.phase + 0.6 * k)
			sk.set_bone_pose_rotation(k, Quaternion(Vector3(0, 0, 1), ang))


# Value of `field` at frame f. Objects may carry "keys": [{"frame": F, <field>: value}, ...]
# (linear interpolation, clamped). Must match tools/blender_ref.py keyed().
func _keyed(o: Dictionary, field: String, f: int) -> Variant:
	var keys: Array = o.get("keys", []).filter(func(k): return k.has(field))
	if keys.is_empty():
		return o[field]
	if f <= keys[0].frame:
		return keys[0][field]
	for i in range(1, keys.size()):
		if f <= keys[i].frame:
			var t := float(f - keys[i - 1].frame) / float(keys[i].frame - keys[i - 1].frame)
			var a = keys[i - 1][field]
			var b = keys[i][field]
			if a is Array:
				return range(a.size()).map(func(k): return lerpf(a[k], b[k], t))
			return lerpf(a, b, t)
	return keys[-1][field]


func _apply_frame(f: int) -> void:
	_apply_skinned(f)
	for pair in light_nodes:
		var node: Light3D = pair[0]
		var l: Dictionary = pair[1]
		node.light_color = _col(_keyed(l, "color", f))
		node.light_energy = _keyed(l, "energy", f)
		if node is DirectionalLight3D:
			var dir := _v3(_keyed(l, "direction", f)).normalized()
			node.look_at_from_position(Vector3.ZERO, dir, Vector3.UP if absf(dir.y) < 0.99 else Vector3.FORWARD)
		else:
			node.position = _v3(_keyed(l, "position", f))
			if node is SpotLight3D:
				var sdir := _v3(_keyed(l, "direction", f)).normalized()
				node.look_at_from_position(node.position, node.position + sdir, Vector3.UP if absf(sdir.y) < 0.99 else Vector3.FORWARD)
	var sky_col := _col(_keyed(spec.sky, "color", f))
	env.background_color = sky_col
	env.ambient_light_color = sky_col
	if gi_effect:
		var sky_lin: Array = _keyed(spec.sky, "color", f)
		gi_effect.sky_color = Color(sky_lin[0], sky_lin[1], sky_lin[2]) # linear radiance for missed rays
	if inject_effect:
		var lin: Array = _keyed(spec.sky, "color", f)
		inject_effect.ambient = Color(lin[0], lin[1], lin[2]) if gi_mode == "inject_identity" else Color(1.0, 0.0, 0.0)


# Returns [position, target] for frame index f.
func _camera_at(f: int) -> Array:
	var path: Array = spec.camera.path
	var hold: int = spec.camera.get("hold_frames", 0)
	if f < hold or path.size() == 1:
		return [_v3(path[0].position), _v3(path[0].target)]
	f -= hold
	for i in range(1, path.size()):
		var n: int = path[i].frames
		if f <= n:
			var t := float(f) / n
			t = t * t * (3.0 - 2.0 * t)
			return [_v3(path[i - 1].position).lerp(_v3(path[i].position), t), _v3(path[i - 1].target).lerp(_v3(path[i].target), t)]
		f -= n
	return [_v3(path[-1].position), _v3(path[-1].target)]


# GPU ms between "goghot_<pass>_begin" / "_end" timestamp pairs of the last measured frame.
func _goghot_timings() -> Dictionary:
	var rd := RenderingServer.get_rendering_device()
	var begins := {}
	var out := {}
	for i in rd.get_captured_timestamps_count():
		var name := rd.get_captured_timestamp_name(i)
		if not name.begins_with("goghot_"):
			continue
		if name.ends_with("_begin"):
			begins[name.trim_suffix("_begin")] = rd.get_captured_timestamp_gpu_time(i)
		elif name.ends_with("_end"):
			var key := name.trim_suffix("_end")
			if begins.has(key):
				out[key.trim_prefix("goghot_")] = (rd.get_captured_timestamp_gpu_time(i) - begins[key]) / 1e6
	return out


func _total_frames() -> int:
	var n: int = spec.camera.get("hold_frames", 0)
	for i in range(1, spec.camera.path.size()):
		n += int(spec.camera.path[i].frames)
	return max(n, 1)


func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() < 2:
		printerr("usage: -- <scene.json> <out_dir>")
		quit(2)
		return
	spec = JSON.parse_string(FileAccess.get_file_as_string(args[0]))
	if spec == null:
		printerr("cannot parse ", args[0])
		quit(2)
		return
	var out_dir: String = args[1]
	if args.size() > 2:
		gi_mode = args[2]
	if args.size() > 3:
		exposure_mode = args[3]
	if args.size() > 4 and not args[4].is_empty():
		gi_params = JSON.parse_string(args[4])
	DirAccess.make_dir_recursive_absolute(out_dir)

	var root := Node3D.new()
	get_root().add_child(root)
	_build(root)
	_run.call_deferred(out_dir)


func _run(out_dir: String) -> void:
	var vp_rid := get_root().get_viewport_rid()
	RenderingServer.viewport_set_measure_render_time(vp_rid, true)

	var n := _total_frames()
	var gpu_ms := []
	var pass_ms := {} # goghot pass name -> per-frame GPU ms (from RD timestamps)
	for f in range(-WARMUP_FRAMES, n):
		var pt := _camera_at(max(f, 0))
		camera.look_at_from_position(pt[0], pt[1], Vector3.UP)
		_apply_frame(max(f, 0))
		if f == -1 and exposure_effect:
			exposure_effect.snap = true # warmup done (fog/TAA converged): start the capture adapted
		await RenderingServer.frame_post_draw
		if f < 0:
			continue
		gpu_ms.append(RenderingServer.viewport_get_measured_render_time_gpu(vp_rid))
		var marks := _goghot_timings()
		for k in marks:
			if not pass_ms.has(k):
				pass_ms[k] = []
			pass_ms[k].append(marks[k])
		var img := get_root().get_texture().get_image()
		img.save_png(out_dir.path_join("frame_%04d.png" % f))

	var meta := {
		"frames": n,
		"hold_frames": spec.camera.get("hold_frames", 0),
		"resolution": [get_root().size.x, get_root().size.y],
		"gpu_ms": gpu_ms,
		"pass_ms": pass_ms,
		"device": RenderingServer.get_video_adapter_name(),
		"gi_mode": gi_mode,
		"exposure": exposure_mode,
		"gi_params": gi_params,
	}
	var fa := FileAccess.open(out_dir.path_join("capture.json"), FileAccess.WRITE)
	fa.store_string(JSON.stringify(meta, "\t"))
	fa.close()
	print("CAPTURE: wrote %d frames to %s" % [n, out_dir])
	quit(0)
