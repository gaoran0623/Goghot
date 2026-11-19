"""Editor preview check: opens a scene in the Godot editor (with GoghotGI.editor_preview), waits, saves
the editor's 3D viewport to a PNG and quits.

    python goghot/testbench/tools/editor_shot.py --godot <editor binary> [--scene res://demo/gi_demo.tscn] --out shot.png

Enables addons/goghot_editor_shot through a temporary override.cfg (removed afterwards).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--godot", required=True)
    ap.add_argument("--scene", default="res://demo/gi_demo.tscn")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=600)
    args = ap.parse_args()
    override = BENCH / "override.cfg"
    override.write_text('[editor_plugins]\nenabled=PackedStringArray("res://addons/goghot_editor_shot/plugin.cfg")\n', encoding="utf-8")
    env = dict(os.environ, GOGHOT_SHOT_SCENE=args.scene, GOGHOT_SHOT_OUT=str(Path(args.out).resolve()), GOGHOT_SHOT_FRAMES=str(args.frames))
    try:
        rc = subprocess.run([args.godot, "--editor", "--path", str(BENCH), "--rendering-driver", "vulkan"], env=env, timeout=600).returncode
    finally:
        override.unlink(missing_ok=True)
    print("editor exited", rc, "->", args.out if Path(args.out).exists() else "no image")
    return 0 if Path(args.out).exists() else 1


if __name__ == "__main__":
    sys.exit(main())
