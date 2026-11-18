"""Goghot test bench runner.

Renders every scene in goghot/testbench/scenes with Godot, renders (cached) Cycles
references, computes metrics and writes a report to goghot/reports/<timestamp>_<tag>/.

    python goghot/testbench/tools/run.py --godot bin/godot.windows.editor.dev.x86_64.exe [--tag name] [--scenes a,b]

Metrics:
  flip / ssim     vs Cycles reference, on reference frames (static frame 0, mid-move, last)
  flicker         mean abs diff between consecutive frames during the static hold (0..255 scale)
  gpu_ms          mean / p95 viewport GPU time over all captured frames (whole frame, not only GI)
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
GOGHOT = BENCH.parent
REPO = GOGHOT.parent
REF_CACHE = BENCH / ".refcache"  # gitignored; keyed by scene hash + resolution + samples


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def find_blender(explicit):
    if explicit:
        return explicit
    found = shutil.which("blender")
    if found:
        return found
    base = Path("C:/Program Files/Blender Foundation")
    if base.exists():
        cands = sorted(base.glob("Blender */blender.exe"), key=lambda p: [int(x) for x in p.parent.name.split()[-1].split(".")])
        if cands:
            return str(cands[-1])
    return None


def total_frames(spec):
    cam = spec["camera"]
    return max(cam.get("hold_frames", 0) + sum(p["frames"] for p in cam["path"][1:]), 1)


def ref_frames(spec):
    n = total_frames(spec)
    hold = spec["camera"].get("hold_frames", 0)
    return sorted({0, min(n - 1, hold + (n - hold) // 2), n - 1})


def run(cmd, log_path):
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
    return proc.returncode


def render_refs(blender, scene_path, spec, w, h, samples, logs_dir):
    key = hashlib.sha1(scene_path.read_bytes() + (HERE / "blender_ref.py").read_bytes()).hexdigest()[:12]
    cache = REF_CACHE / f"{scene_path.stem}_{key}_{w}x{h}_s{samples}"
    frames = ref_frames(spec)
    missing = [f for f in frames if not (cache / f"ref_{f:04d}.png").exists()]
    if missing:
        env = dict(os.environ, GOGHOT_REF_SAMPLES=str(samples))
        cmd = [blender, "-b", "--factory-startup", "--python", str(HERE / "blender_ref.py"), "--",
               str(scene_path), str(cache), str(w), str(h), *map(str, missing)]
        with open(logs_dir / f"{scene_path.stem}_blender.log", "w", encoding="utf-8", errors="replace") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        if rc != 0:
            return None, f"blender exited {rc}"
    return cache, None


def metrics(frames_dir, ref_dir, spec, cap):
    import flip_evaluator as flip
    from skimage.metrics import structural_similarity

    out = {"per_frame": {}}
    if ref_dir:
        flips, ssims = [], []
        for f in ref_frames(spec):
            test = load_rgb(frames_dir / f"frame_{f:04d}.png")
            ref = load_rgb(ref_dir / f"ref_{f:04d}.png")
            err_map, mean_flip, _ = flip.evaluate(ref, test, "LDR")
            ssim = structural_similarity(ref, test, channel_axis=2, data_range=1.0)
            Image.fromarray((np.clip(err_map, 0, 1) * 255).astype(np.uint8)).save(frames_dir.parent / f"flip_{f:04d}.png")
            out["per_frame"][f] = {"flip": float(mean_flip), "ssim": float(ssim)}
            flips.append(mean_flip)
            ssims.append(ssim)
        out["flip"] = float(np.mean(flips))
        out["ssim"] = float(np.mean(ssims))

    hold = spec["camera"].get("hold_frames", 0)
    if hold >= 2:
        diffs = []
        prev = load_rgb(frames_dir / "frame_0000.png")
        for f in range(1, hold):
            cur = load_rgb(frames_dir / f"frame_{f:04d}.png")
            diffs.append(np.abs(cur - prev).mean() * 255.0)
            prev = cur
        out["flicker"] = float(np.mean(diffs))
        out["flicker_max"] = float(np.max(diffs))

    # Exposure behaviour: "key" = geometric mean of linear luminance of the final image, per frame.
    keys = []
    for f in range(cap["frames"]):
        img = load_rgb(frames_dir / f"frame_{f:04d}.png")
        lin = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
        l = lin @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        keys.append(float(np.exp(np.log(np.maximum(l, 1e-4)).mean())))
    out["key_series"] = keys
    out["key_mean"] = float(np.mean(keys))
    out["key_min"] = float(np.min(keys))
    out["key_max"] = float(np.max(keys))
    out["clipped_pct"] = float(np.mean(load_rgb(frames_dir / "frame_0000.png").max(axis=2) >= 0.999) * 100.0)

    gpu = np.array(cap["gpu_ms"][2:] or cap["gpu_ms"], dtype=np.float64)
    out["gpu_ms_mean"] = float(gpu.mean())
    out["gpu_ms_p95"] = float(np.percentile(gpu, 95))
    return out


def fmt(v, spec="{:.4f}"):
    return "—" if v is None else spec.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--godot", required=True, help="path to the Godot editor binary")
    ap.add_argument("--blender", help="path to blender.exe (auto-detected)")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--scenes", help="comma-separated scene names (default: all)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--ref-samples", type=int, default=1024)
    ap.add_argument("--no-ref", action="store_true", help="skip Cycles references (no FLIP/SSIM)")
    ap.add_argument("--exposure", default="fixed", choices=["fixed", "auto", "auto_cpp"],
                    help="auto = histogram auto exposure; FLIP/SSIM vs Cycles are skipped (not comparable), key metrics reported")
    ap.add_argument("--gi", default="baseline", help="GI mode passed to capture.gd (baseline, inject_identity, inject_red, ...)")
    args = ap.parse_args()
    args.godot = str(Path(args.godot).resolve())

    scenes = sorted((BENCH / "scenes").glob("*.json"))
    if args.scenes:
        wanted = set(args.scenes.split(","))
        scenes = [s for s in scenes if s.stem in wanted]

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = GOGHOT / "reports" / f"{stamp}_{args.tag}"
    logs_dir = report_dir / "logs"
    logs_dir.mkdir(parents=True)
    blender = None if args.no_ref else find_blender(args.blender)
    if not args.no_ref and not blender:
        print("WARNING: Blender not found, FLIP/SSIM skipped")

    results = {}
    for scene_path in scenes:
        name = scene_path.stem
        print(f"[{name}] capturing")
        spec = json.loads(scene_path.read_text(encoding="utf-8"))
        frames_dir = report_dir / name / "frames"
        cmd = [args.godot, "--path", str(BENCH), "--rendering-driver", "vulkan",
               "--resolution", f"{args.width}x{args.height}", "--script", "res://capture.gd", "--",
               str(scene_path), str(frames_dir), args.gi, args.exposure]
        rc = run(cmd, logs_dir / f"{name}_godot.log")
        cap_path = frames_dir / "capture.json"
        if rc != 0 or not cap_path.exists():
            results[name] = {"error": f"godot exited {rc}, see logs/{name}_godot.log"}
            print(f"[{name}] FAILED: {results[name]['error']}")
            continue
        cap = json.loads(cap_path.read_text())

        ref_dir, ref_err = (None, "skipped")
        if blender and args.exposure == "fixed":
            print(f"[{name}] reference (Cycles)")
            ref_dir, ref_err = render_refs(blender, scene_path, spec, args.width, args.height, args.ref_samples, logs_dir)
        m = metrics(frames_dir, ref_dir, spec, cap)
        m["ref_error"] = ref_err
        m["device"] = cap["device"]
        m["gi_mode"] = cap["gi_mode"]
        results[name] = m

        # Side-by-side contact sheet for the report: test | reference | FLIP.
        for f in ref_frames(spec):
            tiles = [Image.open(frames_dir / f"frame_{f:04d}.png").convert("RGB")]
            if ref_dir:
                tiles.append(Image.open(ref_dir / f"ref_{f:04d}.png").convert("RGB"))
                tiles.append(Image.open(report_dir / name / f"flip_{f:04d}.png").convert("RGB"))
            sheet = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height))
            x = 0
            for t in tiles:
                sheet.paste(t, (x, 0))
                x += t.width
            sheet.save(report_dir / name / f"compare_{f:04d}.png")
        print(f"[{name}] flip={fmt(m.get('flip'))} ssim={fmt(m.get('ssim'))} flicker={fmt(m.get('flicker'), '{:.3f}')} gpu={m['gpu_ms_mean']:.2f}ms")

    (report_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [f"# Test bench: {args.tag} ({stamp}), GI `{args.gi}`, exposure `{args.exposure}`", "",
             f"Resolution {args.width}x{args.height}, Cycles {args.ref_samples} spp. "
             "Images: `<scene>/compare_NNNN.png` = Godot | Cycles | FLIP.", "",
             "| scene | FLIP ↓ | SSIM ↑ | flicker ↓ (static, /255) | key mean [min, max] | GPU ms mean / p95 (whole frame) |",
             "|---|---|---|---|---|---|"]
    for name, m in results.items():
        if "error" in m:
            lines.append(f"| {name} | error: {m['error']} | | | | |")
        else:
            lines.append(f"| {name} | {fmt(m.get('flip'))} | {fmt(m.get('ssim'))} | {fmt(m.get('flicker'), '{:.3f}')} | "
                         f"{m['key_mean']:.3f} [{m['key_min']:.3f}, {m['key_max']:.3f}] | "
                         f"{m['gpu_ms_mean']:.2f} / {m['gpu_ms_p95']:.2f} |")
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {report_dir}")
    return 0 if all("error" not in m for m in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
