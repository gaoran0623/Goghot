# goghot/

Goghot real-time GI. Engine-side changes stay minimal (see `../CLAUDE.md`); everything else lives here.

## Using it

Build the engine (with the Goghot patches) and the extension (below), copy `testbench/addons/goghot/`
into your project, then **add one `GoghotGI` node to the scene**. That is all:

- it adds the GI effects to the Compositor of the scene's `WorldEnvironment` (creating the Compositor
  if needed; falls back to the current `Camera3D` without a WorldEnvironment) and removes them again
  when the node leaves the tree;
- it collects meshes (static, alpha-tested, skinned), materials and lights for the world tracer;
- the sky radiance for rays that miss is derived from the active `Environment`.

| `GoghotGI` property | |
|---|---|
| `quality` | Low (screen only, 32 px probes, no world reflections / fog GI), Medium (32 px probes), **High** (default, 16 px probes), Ultra (finer voxels / fog) |
| `tracer` | Auto (hardware RT if supported, else voxels), A hardware RT, B voxels, C screen only |
| `auto_exposure` | adds the histogram auto exposure (`GoghotHistogramExposure`) |
| `debug_view` | Off, GI (indirect diffuse on white), Reflections, Probes (green uniform / red adaptive), Voxels (tier B) |
| `editor_preview` | run GI in the editor viewport too |
| `enabled`, `sky_from_environment` | |

- Exclude an object from GI (it neither occludes nor bounces light): set its `gi_mode` to *Disabled*.
- Every parameter of the effect stays reachable through `get_effect()` (`GoghotScreenProbeGI`).
- Example scene: `testbench/demo/gi_demo.tscn`.
- Requirements / limits: Forward+, Vulkan, single view (no XR), no MSAA; GI affects opaque geometry,
  reflections and volumetric fog, not transparent materials. See `reports/PHASE*.md` for measured
  quality and cost and their known limitations.

| path | what |
|---|---|
| `src/` | GDExtension source: `exposure/` histogram auto exposure, `gi/` screen probe GI (`GoghotScreenProbeGI` + `GoghotGIColorCapture`), tier A world tracer + radiance cache (`world_tracer.*`), scene collection node (`GoghotGIScene`). GLSL in `src/*/*.glsl`, embedded at build time |
| `api/` | `extension_api.json` + `gdextension_interface.json` dumped from **our patched engine**; re-dump when engine patches change |
| `thirdparty/godot-cpp/` | godot-cpp submodule |
| `testbench/addons/goghot/` | `.gdextension` + built DLL (bin/ is gitignored) |
| `demos/rt_smoke/` | Phase 0 hardware RT smoke test: Cornell box via RD `blas/tlas/raytracing_pipeline` API → PNG, self-checking |
| `testbench/` | Test bench Godot project. `scenes/*.json` are shared by Godot (`capture.gd`) and Cycles (`tools/blender_ref.py`) |
| `testbench/tools/run.py` | Renders all scenes, Cycles references (cached), FLIP/SSIM/flicker/GPU ms, writes `reports/<time>_<tag>/` |
| `testbench/timing_check.gd` | Sanity check that viewport GPU timing scales with a known ALU load (rerun after driver/engine updates) |
| `reports/` | Test bench reports and phase summaries |
| `KNOWN_ISSUES.md` | Engine/tooling issues we deliberately do not fix |

## Build and run (Windows, from repo root)

```
python -m SCons platform=windows target=editor dev_build=yes compiledb=yes d3d12=no accesskit=no -j11
bin/godot.windows.editor.dev.x86_64.console.exe --path goghot/demos/rt_smoke --rendering-driver vulkan --script res://rt_smoke.gd -- rt_smoke.png
python -m SCons -C goghot target=template_debug -j11
bin/godot.windows.editor.dev.x86_64.console.exe --path goghot/testbench --headless --import   # first time: registers the extension
python goghot/testbench/tools/run.py --godot bin/godot.windows.editor.dev.x86_64.console.exe --tag <name> [--gi baseline|inject_identity|inject_red|screen_probe|rt|node] [--gi-params '{json}'] [--exposure fixed|auto|auto_cpp]
```

Clone with `git submodule update --init` for godot-cpp.

Python deps: `scons zstandard numpy pillow scikit-image flip-evaluator`. Blender (any 4.x/5.x) is auto-detected for references.
