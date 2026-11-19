# Goghot — Godot 通用实时 GI 分支

## 目标
在 Godot 里实现接近 Lumen 的全自动实时全局光照：不烘焙，摆好太阳/天空/灯光，
按物理单位打光，室内、室外、室内外过渡都正确。必须是通用方案，不针对某个游戏。

## 范围纪律（最高优先级）
**只做 GI 本身。不要修复与当前任务不直接相关的引擎 bug、警告、代码风格或重构。**
- 碰到引擎 bug 时：
  1. 判断它是否**直接阻塞**当前阶段任务。不阻塞 → 记一行到 `goghot/KNOWN_ISSUES.md`，继续原任务。
  2. 阻塞 → 优先在 GDExtension 侧绕过（换 API、换格式、加 workaround）。
  3. 绕不过 → 停下来，向人汇报（现象、复现步骤、推测原因、建议的最小补丁），**等确认再改引擎**。
- 不顺手修：编译警告、上游 TODO/FIXME、无关模块（2D、物理、编辑器 UI、导入器、其他平台等）、上游测试失败。
- 引擎补丁只允许下面"引擎补丁清单"里的条目；要新增条目先问。
- 一个问题排查约 30 分钟没有进展 → 停下汇报，不要越挖越深。

## 代码组织
- 仓库：`gaoran0623/Goghot`（fork 自 `godotengine/godot`，remote `upstream`），基于上游 `master`（当前 4.8-beta）。
- RenderingDevice 已有光追 API，直接用，不要自己加：
  `blas_create / blas_build / tlas_create / tlas_build / raytracing_pipeline_create / hit_sbt_create / raytracing_list_*`
  （见 `servers/rendering/rendering_device.h`，已绑定到脚本/GDExtension）。
- **引擎补丁清单**（总量目标 < 500 行，每项独立 commit，便于 rebase 上游）：
  1. scene shader 可读取外部间接光/环境光贴图，替换 ambient 项（StandardMaterial3D 直接可用）
  2. 暴露 mesh 与蒙皮结果的 GPU 顶点缓冲 RID（用于 BLAS）
  3. 体积雾可采样外部 GI 缓存
- **主体全部放在 GDExtension**（仓库根目录 `goghot/`）：探针、追踪、缓存、降噪、曝光。
- commit 前缀：引擎补丁用 `[engine]`，扩展用 `[goghot]`，二者不混在同一个 commit。
- 不向上游提交 PR（Godot 不接受 AI 生成代码）。

## 构建与调试（Windows）
- 引擎：`scons platform=windows target=editor dev_build=yes compiledb=yes`（scons 不在 PATH 上，需先 `pip install scons`）
- 调试 GPU：`--rendering-driver vulkan --gpu-validation`（系统卡死时加 `--gpu-abort`）
- 测试台出帧：`--fixed-fps 60 --write-movie <out.png>`，配合固定相机路径脚本
- GPU 崩溃（DEVICE_LOST 等）先开 validation layer 再排查，不要盲改。

## 架构
统一 Final Gather + 可替换追踪器：
- Screen Space Probes：默认 16×16 像素间距（`probe_spacing` 可配置；原规格 32，2026-11-19 按第 3 阶段数据改为 16）+ 深度/法线突变处自适应补点；每探针 16–32 条八面体方向光线，重要性采样；可分帧更新
- 追踪顺序：屏幕追踪(HZB, 上一帧颜色) → 世界追踪器 → 天空
- 世界追踪器档位：
  - A：硬件光追（Vulkan，BLAS/TLAS，any-hit 支持 alpha 植被，蒙皮用 compute 预蒙皮）
  - B：无 RT 显卡时用体素/HDDA 追踪（参考 HDDAGI, godotengine/godot#119869）
  - C：低端回退 SSIL + SSAO + 稀疏探针
- 世界空间辐射缓存：空间哈希缓存（SHaRC 思路），camera-centered clipmap；负责多次反弹、屏幕外自发光、远距离降噪；半透明与体积雾直接采样它
- 命中材质：自动从材质贴图生成低分辨率 albedo/emissive 表，按实例索引
- 反射：粗糙复用探针；光滑 SSR → 追踪回退
- 曝光：物理灯光单位默认开启；直方图自动曝光（剔除高低百分位 + 中心加权）+ 可选曝光区域

## 阶段（按顺序；不提前做后面阶段的功能）
0. 构建 + 跑通光追 demo + **测试台**（先于一切功能）
1. 引擎补丁 1（ambient 注入）+ 直方图曝光
2. 屏幕探针 + 屏幕追踪 + 时间/空间滤波
3. A 档硬件光追追踪器 + 哈希辐射缓存
4. 反射、植被 any-hit、蒙皮、体积雾
5. B 档体素追踪器、C 档回退
6. 易用性：一个节点启用、画质预设、编辑器预览、per-object 排除 GI、调试视图
7. 实用化（1.0，2026-11-19 定）：
   - 内容覆盖：SpotLight3D、MultiMeshInstance3D、CSG、GridMap、自发光贴图、Panorama 天空、物理灯光单位
   - 测试台补齐：玻璃/半透明、聚光灯 + MultiMesh 植被、动态场景（刚体移动/开关灯/增删物体）、大场景；运动拖影指标；30 分钟长跑显存不增长
   - 验收：所有场景 lum_ratio 在 0.85–1.15；A 档每个场景不输最好的原生 Godot GI；flicker_px < 0.5%；编辑器预览目视确认
   - 硬件：RTX 3060 级 1080p A 档 ≤ 4 ms、B 档在无 RT 显卡实测（需人工提供硬件，未测如实写）
   - 已知限制（不做）：ShaderMaterial 命中材质（需 surface cache）、自定义天空 shader（需新引擎补丁）

## 测试台（每次改动都跑）
- 场景：Cornell Box、室内、室外白天、室外黄昏+雾、室内→室外推门过渡、茂密植被、大量自发光、玻璃、蒙皮角色群、昼夜循环
- 每个场景固定相机路径；命令行渲染帧序列
- 参考图：Blender Cycles 渲染同一场景
- 指标：FLIP/SSIM vs 参考；静止画面相邻帧差（闪烁）；运动时拖影；GPU 时间戳（GI 总耗时）
- 每个场景 × 已实现的档位出报告，存 `goghot/reports/`
- **同一套默认参数必须在所有场景成立，禁止逐场景调参**
- 测试台没跑过的场景不要声称"已通过"，如实写未测。

## 验收（初版目标，可调）
- 1080p，RTX 3060 级，A 档 GI 总耗时 ≤ 4 ms
- 静止画面无可见闪烁；室内无天光漏光
- FLIP 指标逐阶段下降并记录

## 工作规则
- 每个阶段结束写简短总结 + 截图到 `goghot/reports/`，**等人工验收再进下一阶段**
- 改动前先确认属于当前阶段；不属于就记下来，不做
- 设计取舍不确定时，给出推荐方案和理由再问，不要罗列一堆选项
