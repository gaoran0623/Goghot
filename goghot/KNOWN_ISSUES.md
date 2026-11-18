# Known issues (engine / tooling, not fixed on purpose)

One line per issue: what, where, impact, workaround. See CLAUDE.md "范围纪律".

- `re-spirv` prints `OpTypeAccelerationStructureKHR is not supported yet.` when a ray tracing shader is created (`thirdparty/re-spirv/re-spirv.cpp:1861`). Harmless: the shader still builds and traces correctly. Ignore.
- Building on Windows needs `d3d12=no accesskit=no` unless their SDKs are installed (`misc/scripts/install_d3d12_sdk_windows.py`, `install_accesskit.py`). Goghot only targets Vulkan, so we disable them.
- SCons needs the Python `zstandard` module (`pip install zstandard`).
- Vulkan SDK is not installed on the dev machine, so `--gpu-validation` has no layers to load until it is.
