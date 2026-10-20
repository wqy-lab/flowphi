# 候选打法评估

AI 助手新增 `evaluate_candidate` 和 `commit_candidate`。前者只在副本中试算，后者采用已通过内部检查的候选，自动保存原方案版本并保留撤销能力。

在现有 AI 助手中可以说：

> 比较 24–29 秒这段保持原分工和交换左右手的打法，先评估，不要采用。告诉我各自的问题、警告和手指行程。

也可以在授权修改时说：

> 试算这段的两种手序，选择通过内部检查、接续行程较小的方案采用；有移动过快警告时先说明。

需要按原有方式配置模型连接。本功能不附带模型或 API 密钥，不会自行启动视频渲染。

## 工具接口

`evaluate_candidate` 接受 `operations` 数组，操作语义与原有 `edit_editor` 一致：分配、交替、交换、轨迹、按下/接续、释放、Flick 与姿态。编号从 1 开始，操作中的时间为音乐秒数；未写范围时沿用消息发送时的选择。每个候选最多 200 项操作，不接受偏好设置修改。

示例：

```json
{
  "operations": [
    {"op": "swap", "scope": {"time_range": [24, 29]}, "values": {}}
  ]
}
```

返回 `candidate_id`、`eligible`、原方案/候选报告、错误和警告数量差、平面手指行程差、修改的动作和姿态标记。原方案存在错误也会单独报告，不会掩盖。最多返回 50 条问题详情，并给出省略数量及完整问题类型计数。

`commit_candidate` 接受：

```json
{"candidate_id": "评估返回的编号"}
```

最近 8 个候选仅保留在当前编辑助手会话的内存中；退出应用后不保留。未通过检查、已过期、切换工程、工程发生编辑（即使随后撤销）或音乐偏移改变时，拒绝采用旧候选。采用候选不需要额外弹窗；是否采用由用户对 AI 的任务授权决定。

## 检查范围和含义

所有候选都完整检查整首谱面的触控，不只检查被改片段。未修改的动作、之前已经开始的 Hold 和之后的衔接也参与检查。流程为：工程副本 → 操作 → 连续触点解析 → PSAP 编译/回读 → 占用与判定区域等验证。

`eligible=true` 表示完整内部检查通过，警告可以存在；不表示优于原方案，也不表示所有运动合理。比较时还应看用户指定风格、警告和行程。

行程指标来自编译后的平面触控，单位为米：接触路径、同指触点间转移、两者合计，以及转移平均速度的最大值。它们不含抬指高度、腕部运动或皮肤变化，不是自然度分数。

`native_ap_validated=false` 和 `mesh_checked=false` 始终明确保留。姿态操作虽然可试算和采用，仍需后续运动采样和 Blender 皮肤接触/碰撞检查。候选功能不增加音符，不修改原谱面，也未新增动作重组操作。

## 无界面复用

核心服务位于 `flowphi/services/candidates.py` 和 `operations.py`，不依赖编辑器窗口或模型网络请求。

```python
evaluation = evaluate_candidate(document, chart, operations, offset=music_offset)
store = CandidateStore()
result = store.remember(evaluation, document, revision=document.revision,
                        offset=music_offset)
candidate, report = store.prepare_commit(result['candidate_id'], document,
                                         offset=music_offset)
# 应用层负责保存版本，再以 replace_plan 提交并刷新界面。
```

独立调用时请显式给出操作范围；需要默认选择时传入 `selected_ids`（内部从 0 开始）或音乐秒数的 `fallback_range`。完整评估通过后台 worker 运行，期间可取消；计算时工程改变，结果不会被登记为可提交候选。

## 验证

在已安装项目依赖的环境中运行：

```text
python -m unittest discover -s tests -v
```

用项目自带原创 demo 检查：评估不改原工程、无效同指多押、越界轨迹、提前释放 Hold、音乐偏移与编号、候选淘汰、工程变更拒绝提交、采用后撤销、姿态检查边界，以及真正启动后台 worker 的往返。无需模型密钥、Blender 或下载的官方素材。
