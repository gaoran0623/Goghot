# Known issues (engine / tooling, not fixed on purpose)

One line per issue: what, where, impact, workaround. See CLAUDE.md "范围纪律".

- `re-spirv` prints `OpTypeAccelerationStructureKHR is not supported yet.` when a ray tracing shader is created (`thirdparty/re-spirv/re-spirv.cpp:1861`). Harmless: the shader still builds and traces correctly. Ignore.
- Building on Windows needs `d3d12=no accesskit=no` unless their SDKs are installed (`misc/scripts/install_d3d12_sdk_windows.py`, `install_accesskit.py`). Goghot only targets Vulkan, so we disable them.
- SCons needs the Python `zstandard` module (`pip install zstandard`).
- Vulkan SDK is not installed on the dev machine, so `--gpu-validation` has no layers to load until it is.
- RD timestamps (`capture_timestamp`) around our compute passes are sometimes inflated (e.g. GI pass > whole viewport time), likely because the RD draw graph reorders unrelated work into the bracket. Most frames agree with the GI-on/off viewport delta. Use the viewport delta for budget decisions; the per-pass column in reports is indicative only.
