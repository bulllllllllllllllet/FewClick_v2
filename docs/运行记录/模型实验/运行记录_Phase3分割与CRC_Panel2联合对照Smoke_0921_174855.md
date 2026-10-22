# Phase 3 分割与 CRC Panel 2 联合对照 Smoke

- 主任务：使用冻结 Phase 2、既有真实 GDPH H&E 组织分割训练/验证 patch，训练 v4 `CellToRegionAttention` 与 12 类 region head，并以验证 region accuracy 作为唯一主指标。
- 辅助任务：使用 CRC panel_2 空间隔离 manifest 的小规模 train/val patch；真实 H&E 与 H&E 检测/XCell reg64 作为输入，五类免疫细胞区域观测组成仅用于 auxiliary CE target。
- 对照：`seg_only` 与 `seg_plus_panel2` 从同一个 v4 Phase 3 checkpoint、相同随机种子、相同 GDPH patch 顺序和相同步数开始；后者仅额外接收固定权重的 panel_2 梯度。不会使用 shuffled label 伪对照来替代主任务对照。
- 边界：CRC 与 GDPH 是不同队列；本实验仅测试该辅助项在短程联合优化中是否明显损害/改善 GDPH 主任务 region accuracy，不能当作 CRC 分割效果、跨癌种泛化或最终 FewClick click 指标。
- 输出：`/nfs-medical3/zyh/fewclick_v2/semantic_phase3/phase3_seg_panel2_joint_smoke_0921_174855/`；GPU 1，tmux 及 `EXIT_CODE` 日志。
- 命令：`PYTHONPATH=. conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/run_phase3_seg_panel2_joint_smoke.py --timestamp 0921_174855 --device cuda:1 --steps 12 --panel2-weight 0.1 --limit-train 8 --limit-val 8 --panel2-train-per-case 4 --panel2-val-per-case 2`。
- 验收：两个 arm 的真实 GDPH loader→Phase2→seg loss 与真实 CRC loader→Phase2→panel_2 target→aux loss→backward 均有限；报告 GDPH 验证 region accuracy 及 CRC panel_2 验证 CE。若联合 arm 主指标低于 `seg_only` 超过 0.01，直接拒绝该权重；若无明显退化，仅视为可进一步扩大验证的候选。
- 首次结果：失败，`EXIT_CODE=1`；未产生任何训练指标。原因是 v4 `CellToRegionAttention` 默认 cell_dim 为 68，而既有 checkpoint 与实际 H&E cell input 均为 74 维。已将构造参数显式固定为 `(256,74)`，以新时间戳重跑；此前目录不可使用。
