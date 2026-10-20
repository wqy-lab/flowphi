# Agent 交接字条

给后续接手这个仓库的 agent。按日期追加，最新在上。

## 2026-10-20 · 双手交叉：手序代价 + `hand_crossing` 警告

**改了什么**

- `planner.hand_crossing()`：对同时按下的左右手采样腕部 anchor 的横向顺序，超过 12 mm 容差（`CROSS_TOLERANCE`）的部分按深度 × 时长计费（`CROSS_WEIGHT` 每 20 mm，持续超过 0.25 s 后线性放大）。
- `scoring.py`：新增 `crossing` 代价，并把深度、时长、时刻记进 `decision['crossing']`；`fingering_objective == 'load'` 时和其它几何代价一起清零。
- `validation.hand_crossing_warnings()`：检查方案时给出 `hand_crossing` 警告（深度/时长/时刻），不改 `valid`。此前只有指尖距离和腕距 < 55 mm 的软惩罚，深交叉时两腕相距很远，距离判据看不见。

**为什么是软约束而不是硬拒**

Phigros 反手很少但存在，硬拒会把谱面推成无解或临时指法。现在是“有替代打法就避开，没有就交付并提示”。容差就是给换手瞬间两手掠过中线用的。

**实测**

- CROSS†SOUL IN 40–56.5 s（左右食指+中指、beam 6、从空手位起算）：修改前后方案完全一致，总分 72.7682、0 降级、0 交叉；耗时 5.2 s → 6.0 s。
- 合成用例（只启用左右食指，右手中指不可用；长条 + 更远处的单击）：交叉 37 mm / 73 ms 被记录并在验证中警告——双指谱的真实情况。
- 全套测试 67 项通过、74.3 s（交叉检查带腕部 x 范围预筛和锚点缓存，基本没有额外开销）。

**已知缺口（下一步）**

长条中途换手仍未实现。判定层已经接受一个 Hold 由多段接触并集覆盖（`ui/validation.py` 的 union 覆盖检查），所以只要规划器能产出“A 手松开、B 手零间隙接住”的候选即可；目前没有这个动作，强交叉代价在少数谱段只能靠换指或重新排序回避。窗口级前瞻（长条 + 其后 1–2 秒整体比较）也还没做，那才是“谱面设计本来就要求换手”的谱段的正解。

## 2026-10-20 · 黄键长条：去掉与逐颗 baseline 的对比

**改了什么**

开启“黄键长条整体规划”时，`plan_window`（`flowphi/planning/ribbons.py`）不再把逐颗算法当成第四个对比策略：

- 删掉了窗口开头那次 `planner._run_plain(...)` 预跑，以及 `pool` 里的 `baseline` 候选、`reports['baseline']` 的常规报告。
- 现在窗口内只比较三类整体策略：`continuous`（连续划动）、`relay`（分段接力）、`stationary`（多指驻点），有合格方案就按评分排序选择。
- 逐颗算法仍在，但只作为兜底：三类策略在该窗口都没有合格候选时，才按需调用一次 `_run_plain` 保住这些音符。此时 `reports['baseline']` 带 `fallback=True`，`selected='baseline'`，进度文案显示“逐颗兜底”。
- 关闭“黄键长条整体规划”开关时的路径没动，仍是原来的 `planner._run_plain`。

**为什么**

逐颗算法逐键独立处理，无法把整条黄键长条当作一个整体来规划，在长条窗口里不会优于整体策略，把它放进比较池只是重复搜索，还让诊断报告显示一个几乎总是更差的伪基线。所以“与逐颗 baseline 对比”这个设计前提已作废：**不要**再把 baseline 加回候选池、不要恢复窗口开头那次预跑。

**怎么判断**

- 正常窗口：`diagnostics.ribbons.windows[*].candidates` 只有 `continuous` / `relay` / `stationary` 三项（含它们的 `rejections`）。
- 若出现 `candidates['baseline'].fallback == True`，说明这个窗口三类策略都无解，去看各策略的 `rejections` 排查，而不是拿 baseline 当成绩对比。
- 旧工程里保存的 `diagnostics` 可能还带 baseline 项，读取时按可选字段处理。

**相关文件**

- `flowphi/planning/ribbons.py`（`plan_window`）
- `flowphi/ui/worker.py`（进度文案 `baseline='逐颗兜底'`）
- `docs/drag-ribbons.md`、`README.md`（说明改为三类策略 + 兜底）
- `tests/test_ribbons.py`（`test_simple_chain_has_valid_continuous_candidate` 断言不再有 baseline 对比项；进度测试改为期望首个策略 `continuous`；新增 `test_baseline_is_only_a_rescue_when_every_strategy_fails` 覆盖兜底路径）

**验证**

在仓库根目录用项目自带 `.venv` 跑：`python -m unittest discover -s tests`，60 个测试全部通过。兜底路径由新增测试覆盖（把 `_family_candidates` 强制返回空后，窗口仍能退回逐颗算法，`checked_report` 通过，全部音符保留）。
