"""Fetches third-party test-bench assets that are not redistributed in this repository.

    python goghot/testbench/tools/fetch_assets.py

Sponza (Crytek, CRYENGINE Limited License Agreement), glTF version from KhronosGroup/glTF-Sample-Assets,
pinned commit, into goghot/testbench/assets/sponza/ (ignored by git; .gdignore keeps Godot from
importing it, capture.gd loads it at runtime with GLTFDocument).
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/KhronosGroup/glTF-Sample-Assets.git"
COMMIT = "f0920f5b32d172a30efe3f9257e965918831e6ea"
DEST = Path(__file__).resolve().parents[1] / "assets" / "sponza"


def main():
    if (DEST / "Sponza.gltf").exists():
        print(f"already present: {DEST}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "assets"
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse", "--no-checkout", REPO_URL, str(clone)], check=True)
        subprocess.run(["git", "-C", str(clone), "sparse-checkout", "set", "Models/Sponza"], check=True)
        subprocess.run(["git", "-C", str(clone), "checkout", "-q", COMMIT], check=True)
        src = clone / "Models" / "Sponza"
        DEST.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src / "glTF", DEST, dirs_exist_ok=True)
        shutil.copy2(src / "LICENSE.md", DEST / "LICENSE.md")
    (DEST / ".gdignore").touch()
    print(f"fetched Sponza into {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
