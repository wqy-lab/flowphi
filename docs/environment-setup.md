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
3. **指定手模**：https://blendswap.com/blend/22269 。解压后选择其中的 `.blend` 文件；现有渲染器依赖这套模型的骨骼命名，其他模型不能假定直接兼容。
4. **Phira 兼容游戏资源包**：渲染器需要按键贴图、打击特效和音效；接受资源目录或包含 `info.yml` 与 `click.png` 的资源 ZIP。下载的 All Asset 谱面包包含曲目素材，不包含这一套界面资源。

准备好后，重新生成初稿时选定手模以测量手型；生成视频时使用同一手模。点击“生成片段视频”，填写手模、资源包、Blender 路径，勾选“生成前自动检查方案”，先做 5 秒、360p、15 FPS 的片段。

## 验证或维护环境

在项目目录执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.local\uv-tool\bin\uv.exe sync --locked --no-dev --cache-dir .local\uv-cache
```

应用日志位于 `.local/flowphi.log`；后台任务日志位于 `.local/editor-cache/jobs/`。一键入口启动失败时会保留控制台错误。
