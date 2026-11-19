@tool
extends Node
# Editor preview check (tools/editor_shot.py): runs inside the editor when shot.tscn is opened, waits
# GOGHOT_SHOT_FRAMES frames, saves the editor's first 3D viewport to GOGHOT_SHOT_OUT and quits.
# Does nothing when the environment variables are not set (opening the scene by hand is harmless).

var frames_left := -1


func _ready() -> void:
	if not Engine.is_editor_hint() or not OS.has_environment("GOGHOT_SHOT_OUT"):
		return
	frames_left = int(OS.get_environment("GOGHOT_SHOT_FRAMES")) if OS.has_environment("GOGHOT_SHOT_FRAMES") else 600
	if OS.get_environment("GOGHOT_SHOT_PREVIEW") == "0": # comparison shot: GoghotGI preview off
		for n in get_parent().find_children("*", "GoghotGI", true, false):
			n.editor_preview = false
	print("EDITOR_SHOT: waiting ", frames_left, " frames")


func _process(_delta: float) -> void:
	if frames_left < 0:
		return
	frames_left -= 1
	if frames_left == 0:
		var ei = Engine.get_singleton("EditorInterface")
		ei.set_main_screen_editor("3D")
		var vp: SubViewport = ei.get_editor_viewport_3d(0)
		vp.get_texture().get_image().save_png(OS.get_environment("GOGHOT_SHOT_OUT"))
		print("EDITOR_SHOT: saved ", OS.get_environment("GOGHOT_SHOT_OUT"))
		get_tree().quit()
