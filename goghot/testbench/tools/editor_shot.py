"""Editor preview check: opens addons/goghot_editor_shot/shot.tscn (the demo scene with GoghotGI.editor_preview
on, plus a @tool node) in the Godot editor; the node waits, saves the editor's 3D viewport to a PNG and quits.

    python goghot/testbench/tools/editor_shot.py --godot <editor binary> --out shot.png [--frames 600]

The editor log goes next to the PNG (<out>.log).
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--no-preview", action="store_true", help="comparison shot with GoghotGI.editor_preview off")
    args = ap.parse_args()
    godot = str(Path(args.godot).resolve())
    out = Path(args.out).resolve()
    out.unlink(missing_ok=True)
    env = dict(os.environ, GOGHOT_SHOT_OUT=str(out), GOGHOT_SHOT_FRAMES=str(args.frames), GOGHOT_SHOT_PREVIEW="0" if args.no_preview else "1")
    with open(out.with_suffix(".log"), "w", encoding="utf-8", errors="replace") as log:
        try:
            rc = subprocess.run([godot, "--editor", "--path", str(BENCH), "--rendering-driver", "vulkan", "res://addons/goghot_editor_shot/shot.tscn"],
                                env=env, timeout=args.timeout, stdout=log, stderr=subprocess.STDOUT).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    print("editor exited", rc, "->", out if out.exists() else "no image")
    return 0 if out.exists() else 1


if __name__ == "__main__":
    sys.exit(main())
