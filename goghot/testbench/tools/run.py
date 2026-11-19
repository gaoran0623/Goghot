"""Goghot test bench runner.

Renders every scene in goghot/testbench/scenes with Godot, renders (cached) Cycles
references, computes metrics and writes a report to goghot/reports/<timestamp>_<tag>/.

    python goghot/testbench/tools/run.py --godot bin/godot.windows.editor.dev.x86_64.exe [--tag name] [--scenes a,b]

Metrics:
  flip / ssim     vs Cycles reference, on reference frames (static frame 0, mid-move, last)
  flip_norm       FLIP after scaling the image (linear) so its mean luminance matches the reference:
                  structure errors (leaks, occlusion, color bleeding) without the overall brightness bias
  lum_ratio       mean linear luminance, image / reference (overall brightness bias; 1 = none)
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


def srgb_to_linear(a):
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(a):
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(np.maximum(a, 0.0), 1.0 / 2.4) - 0.055)


LUM = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def brightness_normalized(test, ref):
    """(test scaled in linear space to the reference's mean luminance, luminance ratio test / ref)."""
    tl, rl = srgb_to_linear(test), srgb_to_linear(ref)
    t_mean, r_mean = float((tl @ LUM).mean()), float((rl @ LUM).mean())
    ratio = t_mean / max(r_mean, 1e-6)
    scaled = np.clip(linear_to_srgb(tl / max(ratio, 1e-6)), 0.0, 1.0).astype(np.float32)
    return scaled, ratio


def ref_scores(frames_dir, ref_dir, spec, out, save_maps=True):
    import flip_evaluator as flip
    from skimage.metrics import structural_similarity

    flips, norms, ratios, ssims = [], [], [], []
    for f in ref_frames(spec):
        test = load_rgb(frames_dir / f"frame_{f:04d}.png")
        ref = load_rgb(ref_dir / f"ref_{f:04d}.png")
        err_map, mean_flip, _ = flip.evaluate(ref, test, "LDR")
        scaled, ratio = brightness_normalized(test, ref)
        _, mean_norm, _ = flip.evaluate(ref, scaled, "LDR")
        ssim = structural_similarity(ref, test, channel_axis=2, data_range=1.0)
        if save_maps:
            Image.fromarray((np.clip(err_map, 0, 1) * 255).astype(np.uint8)).save(frames_dir.parent / f"flip_{f:04d}.png")
        out["per_frame"][f] = {"flip": float(mean_flip), "flip_norm": float(mean_norm), "lum_ratio": ratio, "ssim": float(ssim)}
        flips.append(mean_flip)
        norms.append(mean_norm)
        ratios.append(ratio)
        ssims.append(ssim)
    out["flip"] = float(np.mean(flips))
    out["flip_norm"] = float(np.mean(norms))
    out["lum_ratio"] = float(np.mean(ratios))
    out["ssim"] = float(np.mean(ssims))


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
    out = {"per_frame": {}}
    if ref_dir:
        ref_scores(frames_dir, ref_dir, spec, out)

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
        # Split the static hold into true flicker and convergence drift: per pixel linear fit over the
        # hold; residual std > 1/255 is visible flicker, the fitted change over the hold is drift.
        frames = np.stack([np.asarray(Image.open(frames_dir / f"frame_{f:04d}.png").convert("L"), dtype=np.float32) for f in range(hold)])
        t = np.arange(hold, dtype=np.float32) - (hold - 1) / 2.0
        slope = (frames * t[:, None, None]).sum(0) / (t * t).sum()
        resid = frames - frames.mean(0) - slope * t[:, None, None]
        out["flicker_px"] = float((resid.std(0) > 1.0).mean() * 100.0)
        out["drift"] = float(np.abs(slope).mean() * hold)

    # Exposure behaviour: "key" = geometric mean of linear luminance of the final image, per frame.
    keys = []
    for f in range(cap["frames"]):
        img = load_rgb(frames_dir / f"frame_{f:04d}.png")
        l = srgb_to_linear(img) @ LUM
        keys.append(float(np.exp(np.log(np.maximum(l, 1e-4)).mean())))
    out["key_series"] = keys
    out["key_mean"] = float(np.mean(keys))
    out["key_min"] = float(np.min(keys))
    out["key_max"] = float(np.max(keys))
    out["clipped_pct"] = float(np.mean(load_rgb(frames_dir / "frame_0000.png").max(axis=2) >= 0.999) * 100.0)

    gpu = np.array(cap["gpu_ms"][2:] or cap["gpu_ms"], dtype=np.float64)
    out["gpu_ms_mean"] = float(gpu.mean())
    out["gpu_ms_p95"] = float(np.percentile(gpu, 95))
    # Goghot passes (RD timestamps); "goghot_ms" = sum of all goghot passes, the GI budget number.
    passes = {k: np.array(v[2:] or v, dtype=np.float64) for k, v in cap.get("pass_ms", {}).items() if v}
    out["pass_ms"] = {k: float(v.mean()) for k, v in passes.items()}
    # Sub-pass markers ("gi.<name>") are nested inside a pass: reported, not added to the total.
    out["goghot_ms"] = float(sum(v.mean() for k, v in passes.items() if "." not in k)) if passes else 0.0
    return out


def fmt(v, spec="{:.4f}"):
    return "—" if v is None else spec.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--godot", help="path to the Godot editor binary")
    ap.add_argument("--blender", help="path to blender.exe (auto-detected)")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--scenes", help="comma-separated scene names (default: all)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--ref-samples", type=int, default=1024)
    ap.add_argument("--no-ref", action="store_true", help="skip Cycles references (no FLIP/SSIM)")
    ap.add_argument("--exposure", default="fixed", choices=["fixed", "auto", "auto_cpp"],
                    help="auto = histogram auto exposure; FLIP/SSIM vs Cycles are skipped (not comparable), key metrics reported")
    ap.add_argument("--gi-params", default="", help='JSON property overrides for the GI effect, e.g. {"probe_spacing": 16}')
    ap.add_argument("--gi", default="baseline", help="GI mode passed to capture.gd (baseline, inject_identity, inject_red, screen_probe, rt, node, godot_ssil, godot_sdfgi, godot_voxelgi, godot_voxelgi_interior)")
    ap.add_argument("--rescore", nargs="+", metavar="REPORT_DIR",
                    help="recompute the reference metrics of existing reports from their saved frames (no rendering); --godot not needed")
    args = ap.parse_args()
    if args.rescore:
        return rescore(args)
    if not args.godot:
        ap.error("--godot is required")
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
               str(scene_path), str(frames_dir), args.gi, args.exposure, args.gi_params]
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
        print(f"[{name}] flip={fmt(m.get('flip'))} flip_norm={fmt(m.get('flip_norm'))} lum={fmt(m.get('lum_ratio'), '{:.3f}')} ssim={fmt(m.get('ssim'))} flicker={fmt(m.get('flicker'), '{:.3f}')} flicker_px={fmt(m.get('flicker_px'), '{:.2f}')}% drift={fmt(m.get('drift'), '{:.2f}')} gpu={m['gpu_ms_mean']:.2f}ms goghot={m['goghot_ms']:.3f}ms")

    meta = {"tag": args.tag, "stamp": stamp, "gi": args.gi, "gi_params": args.gi_params, "exposure": args.exposure,
            "width": args.width, "height": args.height, "ref_samples": args.ref_samples}
    (report_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_report(report_dir, results, meta)
    print(f"report: {report_dir}")
    return 0 if all("error" not in m for m in results.values()) else 1


def write_report(report_dir, results, meta, title=None):
    (report_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    title = title or f"# Test bench: {meta['tag']} ({meta['stamp']}), GI `{meta['gi']}` {meta['gi_params']}, exposure `{meta['exposure']}`"
    lines = [title, "",
             f"Resolution {meta['width']}x{meta['height']}, Cycles {meta['ref_samples']} spp. "
             "Images: `<scene>/compare_NNNN.png` = Godot | Cycles | FLIP. "
             "FLIP norm = FLIP after matching the mean luminance to the reference; lum = mean luminance / reference.", "",
             "| scene | FLIP ↓ | FLIP norm ↓ | lum | SSIM ↑ | flicker ↓ (static, /255) | flicker px % ↓ (detrended std > 1/255) | drift /255 (over hold) | key mean [min, max] | GPU ms mean / p95 (whole frame) | goghot passes ms |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in results.items():
        if "error" in m:
            lines.append(f"| {name} | error: {m['error']} | | | | | | | | | |")
        else:
            lines.append(f"| {name} | {fmt(m.get('flip'))} | {fmt(m.get('flip_norm'))} | {fmt(m.get('lum_ratio'), '{:.3f}')} | {fmt(m.get('ssim'))} | "
                         f"{fmt(m.get('flicker'), '{:.3f}')} | {fmt(m.get('flicker_px'), '{:.2f}')} | {fmt(m.get('drift'), '{:.2f}')} | "
                         f"{m['key_mean']:.3f} [{m['key_min']:.3f}, {m['key_max']:.3f}] | "
                         f"{m['gpu_ms_mean']:.2f} / {m['gpu_ms_p95']:.2f} | "
                         + (", ".join(f"{k} {v:.3f}" for k, v in m["pass_ms"].items()) or "—") + " |")
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def rescore(args):
    """Recompute flip / flip_norm / lum_ratio / ssim of existing reports from their saved frames and
    the cached Cycles references (references must exist: same scene file and blender_ref.py)."""
    for d in args.rescore:
        report_dir = Path(d).resolve()
        results = json.loads((report_dir / "results.json").read_text(encoding="utf-8"))
        meta_path = report_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        w, h, samples = (meta["width"], meta["height"], meta["ref_samples"]) if meta else (args.width, args.height, args.ref_samples)
        for name, m in results.items():
            if "error" in m:
                continue
            scene_path = BENCH / "scenes" / f"{name}.json"
            spec = json.loads(scene_path.read_text(encoding="utf-8"))
            key = hashlib.sha1(scene_path.read_bytes() + (HERE / "blender_ref.py").read_bytes()).hexdigest()[:12]
            ref_dir = REF_CACHE / f"{name}_{key}_{w}x{h}_s{samples}"
            if not all((ref_dir / f"ref_{f:04d}.png").exists() for f in ref_frames(spec)):
                print(f"{report_dir.name} [{name}] no cached reference, skipped")
                continue
            m.setdefault("per_frame", {})
            m["per_frame"] = {}
            ref_scores(report_dir / name / "frames", ref_dir, spec, m, save_maps=False)
            print(f"{report_dir.name} [{name}] flip={m['flip']:.4f} flip_norm={m['flip_norm']:.4f} lum={m['lum_ratio']:.3f}")
        title = None
        if meta is None:  # report from before meta.json: keep its original title line
            readme = report_dir / "README.md"
            title = readme.read_text(encoding="utf-8").splitlines()[0] if readme.exists() else f"# Test bench: {report_dir.name}"
            meta = {"width": w, "height": h, "ref_samples": samples}
        write_report(report_dir, results, meta, title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
