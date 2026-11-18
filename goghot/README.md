# goghot/

Goghot real-time GI. Engine-side changes stay minimal (see `../CLAUDE.md`); everything else lives here.

| path | what |
|---|---|
| `src/` | GDExtension source (`exposure/`: histogram auto exposure). GLSL in `src/*/*.glsl`, embedded at build time |
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
python goghot/testbench/tools/run.py --godot bin/godot.windows.editor.dev.x86_64.console.exe --tag <name> [--gi baseline|inject_identity|inject_red] [--exposure fixed|auto|auto_cpp]
```

Clone with `git submodule update --init` for godot-cpp.

Python deps: `scons zstandard numpy pillow scikit-image flip-evaluator`. Blender (any 4.x/5.x) is auto-detected for references.
