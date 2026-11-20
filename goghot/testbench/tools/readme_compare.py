"""Builds the README comparison images from test bench reports.

For each scene: No GI | Godot SDFGI | Goghot (tier A) | Cycles reference, one camera frame, with each
render's FLIP against the Cycles reference (lower is better).

    python readme_compare.py --frame 60 --out ../../docs/compare \
        --goghot <report dir> --sdfgi <report dir> --baseline <report dir> scene1 scene2 ...

The tiles are the images the bench scored (left tile of <scene>/compare_NNNN.png: Godot | Cycles | FLIP).
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TILE_W, TILE_H = 640, 360
BAR = 34


def font(size):
    for name in ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def tiles(report, scene, frame):
    sheet = Image.open(Path(report) / scene / f"compare_{frame:04d}.png").convert("RGB")
    w = sheet.width // 3
    return sheet.crop((0, 0, w, sheet.height)), sheet.crop((w, 0, 2 * w, sheet.height))


def flip_of(report, scene, frame):
    r = json.load(open(Path(report) / "results.json", encoding="utf-8"))
    pf = r.get(scene, {}).get("per_frame", {}).get(str(frame), {})
    return pf.get("flip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=60, help="preferred frame (else the middle scored frame of each scene)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--goghot", required=True)
    ap.add_argument("--sdfgi", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("scenes", nargs="+")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f_label, f_value = font(22), font(18)
    for scene in a.scenes:
        frames = sorted(int(x.stem.split("_")[1]) for x in (Path(a.goghot) / scene).glob("compare_*.png"))
        frame = a.frame if a.frame in frames else frames[len(frames) // 2]
        cols = []
        for title, rep in [("No GI", a.baseline), ("Godot SDFGI", a.sdfgi), ("Goghot (tier A)", a.goghot)]:
            img, _ = tiles(rep, scene, frame)
            cols.append((title, img, flip_of(rep, scene, frame)))
        _, ref = tiles(a.goghot, scene, frame)
        cols.append(("Cycles reference", ref, None))
        canvas = Image.new("RGB", (TILE_W * len(cols), TILE_H + BAR), (24, 24, 28))
        d = ImageDraw.Draw(canvas)
        for i, (title, img, flip) in enumerate(cols):
            canvas.paste(img.resize((TILE_W, TILE_H), Image.LANCZOS), (i * TILE_W, BAR))
            d.text((i * TILE_W + 12, 5), title, fill=(235, 235, 235), font=f_label)
            if flip is not None:
                txt = f"FLIP {flip:.3f}"
                tw = d.textlength(txt, font=f_value)
                d.text((i * TILE_W + TILE_W - tw - 12, 8), txt, fill=(170, 200, 255), font=f_value)
        canvas.save(out / f"{scene}.jpg", quality=88, optimize=True)
        print(scene, frame, [round(c[2], 3) if c[2] is not None else None for c in cols])


if __name__ == "__main__":
    main()
