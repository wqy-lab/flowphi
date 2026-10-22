# 在这台电脑上查看 FlowPhi 效果

## 已准备好的环境

项目路径：`E:\Project\phigros\flowphi`。

- 独立标准 CPython 3.13.16：`.local/python/`。
- 项目环境：`.venv/`，已经按 `uv.lock` 安装运行依赖。
- 项目本地 uv：`.local/uv-tool/bin/uv.exe`。
- 已生成可直接打开的示例工程：`.local/preview-demo.fingering.json`，23 个音符。
- 已验证图形窗口创建与绘制；20 项候选评估测试在此独立环境中全部通过。

本机 Anaconda 的 Qt DLL 与项目要求的 Qt 6.11.1 存在冲突，所以项目使用上述独立 CPython。请使用项目入口，不要改用 Anaconda 的 Python 启动。

## 先看预览

在资源管理器双击项目根目录的 `Start-FlowPhi.cmd`。它默认打开准备好的示例工程。

也可以在 PowerShell 中运行：

```powershell
Set-Location E:\Project\phigros\flowphi
.\.venv\Scripts\python.exe main.py --project .local\preview-demo.fingering.json
```

点击“播放”，或按空格。左上预览区域有“谱面与触控轨迹”和“手部姿态”两个页签；后者是简化关节预览，不是带皮肤的视频。

查看已下载谱面：点击“打开谱面”，选择 `E:\Project\phigros\chart` 中的 All Asset ZIP；若询问包内谱面，选择 `charts/IN.json`。点击“生成指法初稿”，首次先保留通用手型、左右食指与中指。生成完成后播放或拖动时间轴检查。

## 配置 AI 助手

点击“模型修改”，在助手的“连接”页填写：

| 字段 | 内容 |
|---|---|
| 协议 | 与服务商一致：Chat Completions、Responses 或 Messages |
| 网关地址 | 服务商提供的 API 根地址，程序自动补接口路径 |
| API 密钥 | 对应服务的密钥 |
| 模型名称 | 服务商要求的模型 ID，在对话页顶部填写 |
| 推理强度 | 初次使用“默认推理” |
| 补充提示词 | 可以留空 |

点击“保存连接与提示词”。所选模型与服务必须支持工具调用。程序原有协议适配已保留，本次未配置密钥或实际调用模型。

初次可以发送：

> 先读取当前这段动作，评估交换左右手是否可行，对比错误、警告和手指行程；只评估，不要采用。

看到“评估候选打法”的工具结果后，再按需要让助手采用某个候选。评估完整检查整首触控方案，不能因此宣称官方 AP 或皮肤无穿模。

## 导出带皮肤的 3D 视频

以下组件本次尚未安装或下载：

1. **Blender**：从 https://www.blender.org/download/ 获取 Windows 版本，在生成视频窗口填写 `blender.exe` 的绝对路径。
2. **FFmpeg / FFprobe**：https://ffmpeg.org/download.html 的 Windows 下载入口提供编译包。把其中两个程序放到项目的 `ffmpeg/bin/ffmpeg.exe` 和 `ffmpeg/bin/ffprobe.exe`。项目启动时会自动识别这一位置。
3. **手模**：原版支持 https://blendswap.com/blend/22269 ；本机也已准备好 MOD 的兼容模型 `.local/models/MOD-FlowPhi.blend`，初稿测量和视频导出都选择这一文件。不要直接选择原始 `HandRigs.blend`。
4. **Phira 兼容游戏资源包**：渲染器需要按键贴图、打击特效和音效；接受资源目录或包含 `info.yml` 与 `click.png` 的资源 ZIP。下载的 All Asset 谱面包包含曲目素材，不包含这一套界面资源。

准备好后，重新生成初稿时选定手模以测量手型；生成视频时使用同一手模。点击“生成片段视频”，填写手模、资源包、Blender 路径，勾选“生成前自动检查方案”，先做 5 秒、360p、15 FPS 的片段。

## 验证或维护环境

### 2026-10-20 视频环境检查更新

- 已下载并解压官方 Blender 4.5.3 LTS 便携版，路径为 `.local/tools/blender-4.5.3-windows-x64/blender.exe`，已验证启动。
- `ffmpeg/bin/` 已升级为 Gyan 的 FFmpeg / FFprobe 9.0.2 essentials 构建。原本机 4.4.5 缺少 PNG 解码器，不能完成最终合成；新版已验证 PNG、FFV1、H.264 和 AAC。
- 已从 `TeamFlos/phira` 官方仓库获取贴图和打击音，资源目录为 `.local/resources/phira/`。
- 已从 https://mightbemod.itch.io/simple-blender-hand-models 下载 MOD 的 `HandRigs.blend`，并生成 `.local/models/MOD-FlowPhi.blend`。作者允许修改、用于任何项目且无需署名；原文件保留，模型未纳入 Git。
- 已生成 `.local/video-smoke/screen-preview-NO-HANDS.mp4`：70 Minutes Fighters EZ 的音乐时间 11.5–16.5 秒，510×286、15 FPS、75 帧、5.000 秒，含音乐和打击音。这是谱面合成检查，不含手部。
- `.local/video-smoke.py --model .local/models/MOD-FlowPhi.blend --start 25` 可测量手型、生成指法、自动检查并导出 640×360、15 FPS、5 秒手部视频。再次导出时指定新的 `--output-name`，避免覆盖动作数据；`--reuse-project` 可复用该模型生成的工程。
- 已成功输出 `.local/video-smoke/hands-25s/render/handcam.mp4`，70 Minutes Fighters EZ 的 25–30 秒，640×360、15 FPS、75 帧、5.000 秒，含双手、音乐和打击音，完整解码通过。18 个音符的编译后内部判定检查通过。
- 当前为流程验证样片：Blender 报告最大指腹接触偏差约 9.97 mm、最大手指碰撞代理重叠约 0.65 mm；屏幕表面最小间隙约 0.50 mm。样片有手指超出画面上缘，仍需调整手型比例、镜头构图及运动拟合，不能视为最终质量或官方游戏 AP 验证。

### MOD 手模转换

转换器只使用 `Hand` 和 `Hand Rig`，排除兽爪模型，复制左右双手，转换坐标、骨骼名与蒙皮权重，补齐渲染器使用的拇指基部和前臂骨。原手模不变；转换文件可在现有 UI 里作为普通兼容手模使用。

```powershell
& .local/tools/blender-4.5.3-windows-x64/blender.exe --background --disable-autoexec .local/models/HandRigs.blend --python-exit-code 1 --python flowphi/rendering/convert_mod_hand.py -- .local/models/MOD-FlowPhi.blend
& .local/tools/blender-4.5.3-windows-x64/blender.exe --background --disable-autoexec .local/models/MOD-FlowPhi.blend --python-exit-code 1 --python tests/blender_mod_hand_contract.py
```

骨骼检查覆盖左右镜像、所需骨骼、手指权重和双手 30 个手指关节的实际网格变形。新手模没有内置相机时，渲染器会自动创建相机。

### c&s2 IN 整首预览

输入是 `projects/c&s2IN.fingering.json`（CROSS†SOUL IN，1305 个音符、878 段动作），不是重新生成的打法。导出副本在 `.local/cs2-full/cs2-measured-copy.fingering.json`，只换成实测 MOD 手型；原触点、编辑和姿态已逐项核对不变，原工程文件哈希也已核对。

原工程的整谱检查有 14 处轨迹判定错误和 5 条双手交叉警告，报告保存在 `.local/cs2-full/source-validation.json`。此次是保留现有指法的问题预览，不代表内部判定通过或官方游戏 AP。

整首预览使用 640×360、15 FPS，输出目录 `output/cs2-IN-full/render/`。为避免高精度皮肤求解耗时过长，任务使用 `preview_sampling=true`（输出帧、触控和姿态关键时刻）、`pose_avoidance=false`，仍执行接触求解和关节限速。轨迹数据不变，帧间皮肤接触不能据此验收。任务配置、诊断、渲染日志均保留在输出目录。

已输出 `output/cs2-IN-full/render/handcam.mp4`，时长 139.266667 秒、2089 帧、约 16.2 MB，H.264 视频和 AAC 音轨完整解码通过。已抽查 5、59、105、130 秒画面，双手和谱面可见；仍有明显穿插和接触偏差。预览诊断最大指腹偏差约 153.4 mm、屏幕最深穿入约 41.5 mm，不可当作接触或动作质量通过。验证摘要位于 `output/cs2-IN-full/verification.json`；原工程文件哈希未变。

批量写入对象动画可通过 `bake_bulk_transforms=true` 启用；默认仍为关闭。已对比 2 秒、30 帧的双手对象和骨骼矩阵，结果一致；Blender 4.5 动画槽绑定和触控采样的专项检查通过。

在项目目录执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.local\uv-tool\bin\uv.exe sync --locked --no-dev --cache-dir .local\uv-cache
```

应用日志位于 `.local/flowphi.log`；后台任务日志位于 `.local/editor-cache/jobs/`。一键入口启动失败时会保留控制台错误。
