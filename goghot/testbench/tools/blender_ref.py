# Renders Cycles reference frames for a test-bench scene JSON.
#
#   blender -b --factory-startup --python blender_ref.py -- <scene.json> <out_dir> <width> <height> <frame> [<frame> ...]
#
# Unit conversions must stay in sync with capture.gd (see its header).
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
scene_path, out_dir, width, height = argv[0], argv[1], int(argv[2]), int(argv[3])
frames = [int(f) for f in argv[4:]]
spec = json.load(open(scene_path, encoding="utf-8"))
samples = int(os.environ.get("GOGHOT_REF_SAMPLES", "1024"))


def v3(a):  # Godot (x, y, z), Y-up -> Blender (x, -z, y), Z-up
    return Vector((a[0], -a[2], a[1]))


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def keyed(o, field, f):  # Must match capture.gd _keyed.
    keys = [k for k in o.get("keys", []) if field in k]
    if not keys:
        return o[field]
    if f <= keys[0]["frame"]:
        return keys[0][field]
    for i in range(1, len(keys)):
        if f <= keys[i]["frame"]:
            t = (f - keys[i - 1]["frame"]) / (keys[i]["frame"] - keys[i - 1]["frame"])
            a, b = keys[i - 1][field], keys[i][field]
            if isinstance(a, list):
                return [a[k] + (b[k] - a[k]) * t for k in range(len(a))]
            return a + (b - a) * t
    return keys[-1][field]


def camera_at(f):  # Must match capture.gd _camera_at.
    cam = spec["camera"]
    path, hold = cam["path"], cam.get("hold_frames", 0)
    if f < hold or len(path) == 1:
        return path[0]["position"], path[0]["target"]
    f -= hold
    for i in range(1, len(path)):
        n = path[i]["frames"]
        if f <= n:
            t = smoothstep(f / n)
            lerp = lambda a, b: [a[k] + (b[k] - a[k]) * t for k in range(3)]
            return lerp(path[i - 1]["position"], path[i]["position"]), lerp(path[i - 1]["target"], path[i]["target"])
        f -= n
    return path[-1]["position"], path[-1]["target"]


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# Render settings.
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
    try:
        prefs.compute_device_type = backend
        prefs.get_devices()
        if any(d.type == backend for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == backend
            scene.cycles.device = "GPU"
            break
    except TypeError:
        continue
scene.cycles.samples = samples
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 12
scene.cycles.diffuse_bounces = 8
scene.cycles.glossy_bounces = 4
scene.cycles.transmission_bounces = 12
scene.cycles.sample_clamp_indirect = 0.0
scene.render.resolution_x = width
scene.render.resolution_y = height
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

# World: uniform sky radiance.
world = bpy.data.worlds.new("Sky")
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Strength"].default_value = 1.0
scene.world = world

# Materials.
mats = {}
for name, m in spec["materials"].items():
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*m.get("albedo", [0.8, 0.8, 0.8]), 1.0)
    bsdf.inputs["Roughness"].default_value = m.get("roughness", 1.0)
    bsdf.inputs["Metallic"].default_value = m.get("metallic", 0.0)
    bsdf.inputs["Specular IOR Level"].default_value = 0.0 if m.get("specular", 0.0) == 0.0 else 0.5
    if "albedo_texture" in m:
        nt = mat.node_tree
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(os.path.join(os.path.dirname(scene_path), "..", "assets", m["albedo_texture"]))
        tex.interpolation = "Linear"
        mul = nt.nodes.new("ShaderNodeMix")
        mul.data_type = "RGBA"
        mul.blend_type = "MULTIPLY"
        mul.inputs["Factor"].default_value = 1.0
        mul.inputs[6].default_value = (*m.get("albedo", [1, 1, 1]), 1.0)
        nt.links.new(tex.outputs["Color"], mul.inputs[7])
        nt.links.new(mul.outputs[2], bsdf.inputs["Base Color"])
        if "alpha_scissor" in m:  # binary alpha test, like Godot's alpha scissor
            gt = nt.nodes.new("ShaderNodeMath")
            gt.operation = "GREATER_THAN"
            gt.inputs[1].default_value = m["alpha_scissor"]
            nt.links.new(tex.outputs["Alpha"], gt.inputs[0])
            nt.links.new(gt.outputs[0], bsdf.inputs["Alpha"])
    if "emission" in m:
        bsdf.inputs["Emission Color"].default_value = (*m["emission"], 1.0)
        bsdf.inputs["Emission Strength"].default_value = m.get("emission_strength", 1.0)
    mats[name] = mat

# Objects.
skinned_objs = []


def skin_pose(o, m, f):
    # Same as capture.gd _apply_skinned + Godot LBS: S_k = G_k * Rest_k^-1, chain along +Y, rotation about Z.
    hold = spec["camera"].get("hold_frames", 0)
    a = o["anim"]
    step = m["bone_step"]
    mats = []
    g = Matrix.Identity(4)
    for k in range(m["bone_count"]):
        t = max(f - hold, 0) if f >= hold else 0
        ang = math.radians(a["amplitude"]) * math.sin(a["speed"] * t + a["phase"] + 0.6 * k)
        local = (Matrix.Translation((0, step, 0)) if k > 0 else Matrix.Identity(4)) @ Matrix.Rotation(ang, 4, "Z")
        g = g @ local
        mats.append(g @ Matrix.Translation((0, -k * step, 0)))
    out = []
    for p, bs, ws in zip(m["vertices"], m["bones"], m["weights"]):
        v = Vector((p[0], p[1], p[2], 1.0))
        acc = Vector((0, 0, 0, 0))
        for bi, wi in zip(bs, ws):
            if wi > 0:
                acc += (mats[bi] @ v) * wi
        out.append(v3([acc.x, acc.y, acc.z]))  # Godot local (Y up) -> Blender local
    return out


for o in spec["objects"]:
    if o["type"] == "box":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=v3(o["center"]))
        ob = bpy.context.active_object
        s = o["size"]
        ob.scale = (s[0], s[2], s[1])
        ob.rotation_euler = (0.0, 0.0, math.radians(o.get("rot_y", 0.0)))
    elif o["type"] == "skinned":
        m = spec["skinned_meshes"][o["mesh"]]
        me = bpy.data.meshes.new("skinned")
        tris = [m["indices"][i:i + 3] for i in range(0, len(m["indices"]), 3)]
        me.from_pydata([v3(p) for p in m["vertices"]], [], [t[::-1] for t in tris])
        ob = bpy.data.objects.new("skinned", me)
        scene.collection.objects.link(ob)
        ob.location = v3(o["center"])
        ob.rotation_euler = (0.0, 0.0, math.radians(o.get("rot_y", 0.0)))
        skinned_objs.append((ob, o, m))
    elif o["type"] == "quad":
        # Plane in local XY facing +Z, same basis as capture.gd (right = up x normal).
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        ob = bpy.context.active_object
        n = Vector(o["normal"]).normalized()
        r = Vector(o["up"]).cross(n).normalized()
        u = n.cross(r)
        bx, by, bz = v3(r), v3(u), v3(n)
        s = o["size"]
        ob.matrix_world = Matrix(((bx.x * s[0], by.x * s[1], bz.x, 0), (bx.y * s[0], by.y * s[1], bz.y, 0), (bx.z * s[0], by.z * s[1], bz.z, 0), (0, 0, 0, 1)))
        ob.location = v3(o["center"])
    else:
        raise ValueError("unknown object type " + o["type"])
    ob.data.materials.append(mats[o["material"]])
    if o["type"] == "skinned":
        for poly in ob.data.polygons:
            poly.use_smooth = False

# Lights (values applied per frame in apply_frame).
light_objs = []
for i, l in enumerate(spec["lights"]):
    if l["type"] == "sun":
        ld = bpy.data.lights.new(f"sun{i}", "SUN")
        ld.angle = math.radians(0.5)
    elif l["type"] == "point":
        ld = bpy.data.lights.new(f"point{i}", "POINT")
        ld.shadow_soft_size = 0.0
        if hasattr(ld, "use_soft_falloff"):
            ld.use_soft_falloff = False
    else:
        raise ValueError("unknown light type " + l["type"])
    ob = bpy.data.objects.new(ld.name, ld)
    scene.collection.objects.link(ob)
    light_objs.append((ob, l))


def apply_frame(f):
    for ob, o, m in skinned_objs:
        for vert, co in zip(ob.data.vertices, skin_pose(o, m, f)):
            vert.co = co
        ob.data.update()
    for ob, l in light_objs:
        ob.data.color = keyed(l, "color", f)
        e = keyed(l, "energy", f)
        if l["type"] == "sun":
            ob.data.energy = math.pi * e
            ob.rotation_euler = v3(keyed(l, "direction", f)).normalized().to_track_quat("-Z", "Y").to_euler()
        else:
            ob.data.energy = 4.0 * math.pi * math.pi * e
            ob.location = v3(keyed(l, "position", f))
    bg.inputs["Color"].default_value = (*keyed(spec["sky"], "color", f), 1.0)


# Fog: a cube of side 2*length around the scene origin approximates Godot's froxel fog range.
# Scatter + absorption with the same color => extinction = density, scattering = albedo * density (as in Godot).
if "fog" in spec:
    fog = spec["fog"]
    L = fog.get("length", 150.0)
    fm = bpy.data.materials.new("fog")
    fm.use_nodes = True
    nt = fm.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    sc = nt.nodes.new("ShaderNodeVolumeScatter")
    ab = nt.nodes.new("ShaderNodeVolumeAbsorption")
    add = nt.nodes.new("ShaderNodeAddShader")
    col = (*fog.get("albedo", [1, 1, 1]), 1.0)
    sc.inputs["Color"].default_value = col
    sc.inputs["Density"].default_value = fog["density"]
    sc.inputs["Anisotropy"].default_value = fog.get("anisotropy", 0.0)
    ab.inputs["Color"].default_value = col
    ab.inputs["Density"].default_value = fog["density"]
    nt.links.new(sc.outputs[0], add.inputs[0])
    nt.links.new(ab.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], nt.nodes["Material Output"].inputs["Volume"])
    bpy.ops.mesh.primitive_cube_add(size=2.0 * L, location=(0, 0, 0))
    bpy.context.active_object.data.materials.append(fm)
    scene.cycles.volume_bounces = 2

# Camera (Godot fov is vertical).
cd = bpy.data.cameras.new("cam")
cd.sensor_fit = "VERTICAL"
cd.angle_y = math.radians(spec["camera"].get("fov_y", 50.0))
cd.clip_start = 0.05
cd.clip_end = 500.0
cam = bpy.data.objects.new("cam", cd)
scene.collection.objects.link(cam)
scene.camera = cam

os.makedirs(out_dir, exist_ok=True)
for f in frames:
    pos, target = camera_at(f)
    p, t = v3(pos), v3(target)
    cam.location = p
    cam.rotation_euler = (t - p).to_track_quat("-Z", "Y").to_euler()
    apply_frame(f)
    scene.render.filepath = os.path.join(out_dir, f"ref_{f:04d}.png")
    bpy.ops.render.render(write_still=True)
    print(f"BLENDER_REF: wrote {scene.render.filepath}")
