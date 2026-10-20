# FlowPhi

## 生成视频

生成手部视频还需要：
  - **Blender**
  - **FFmpeg 和 FFprobe**
  - **手部骨骼模型**
    - 可到[BlendSwap](https://blendswap.com/blend/22269)下载
  - **游戏资源包**

## 从源码运行

```sh
uv sync --locked
uv run main.py
```

## 许可

MIT

## 候选打法评估

AI 助手支持先在副本中评估多个打法，再采用通过内部检查的候选；评估会对比错误、警告与平面手指行程，采用后可撤销。

详见[候选打法评估说明](docs/candidate-evaluation.md)。内部检查不代表官方游戏 AP 或 Blender 皮肤无穿模。

生成器也支持[黄键接续](docs/drag-continuation.md)：复用已有触点或延长短动作，减少不必要的单独按下。

密集黄键还支持[黄键长条整体规划](docs/drag-ribbons.md)，按完整窗口比较连划、分段接力和多指驻点；逐颗算法只在完全无解时兜底。
