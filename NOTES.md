# Agent 交接字条

给后续接手这个仓库的 agent。按日期追加，最新在上。

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
