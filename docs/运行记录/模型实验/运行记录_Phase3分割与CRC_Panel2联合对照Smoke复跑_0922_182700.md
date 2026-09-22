# Phase 3 分割与 CRC Panel 2 联合对照 Smoke 复跑

- 时间戳：`0922_182700`。
- 问题：前次修正运行 `0921_175712` 仍以 `EXIT_CODE=1` 结束；日志中的旧调用 `cell(*item[:5])` 将 CRC 样本元组按错误位置传给 `CellToRegionAttention`，使 `cells` 被当作 `assignment_at_cells`。
- 修正：联合训练脚本改为显式传入 `region_tokens, assignment_at_cells, cells, cell_valid, total_cell_count`。本次同时加入针对该顺序的合成回归测试。
- 输入：冻结 Phase 2 checkpoint、既有 GDPH Phase 3 train/val loader、CRC panel_2 空间隔离 manifest `0921_173523`，以及医生审核派生 H&E/XCell cache；所有源数据与 checkpoint 只读。
- 输出：`/nfs-medical3/zyh/fewclick_v2/semantic_phase3/phase3_seg_panel2_joint_smoke_0922_182700/`；日志为同根 `phase3_seg_panel2_joint_smoke_0922_182700.log`。
- 计算：GPU 1，通过命名 tmux session 运行；完成后写入实际 worker PID 与 `EXIT_CODE`。
- 对照：`seg_only` 与 `seg_plus_panel2` 同一 Phase 3 checkpoint、随机种子、GDPH patch 顺序和步数（12）起始；后者额外接受权重 0.1 的 panel_2 observed-composition CE。
- 验收：两个 arm 的真实 GDPH loader → Phase 2 → segmentation loss，以及 CRC loader → Phase 2 → cell-to-region fusion → panel_2 loss → backward 均有限；记录 GDPH validation region accuracy 和 CRC validation panel_2 CE。若 joint arm 的主指标低于 `seg_only` 超过 0.01，则拒绝该辅助权重。
- 完成：tmux pane PID `4114412`，任务以 `EXIT_CODE=0` 结束。`seg_only` / `seg_plus_panel2` 的 GDPH validation region accuracy 均为 `0.896484375`；联合 arm 训练期 panel_2 CE 为 `1.6979376`。本次未计算独立 CRC validation CE，故其结果仅用于确认联合 forward/backward 链路可运行；完整验收转入 `0922_184000`。
