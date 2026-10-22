# Phase 3 与 CRC Panel 2 联合扩展 Gate

- 时间戳：`0922_182926`。
- 前置证据：`0922_184000` 的 12-step 真实 smoke 使独立 CRC panel_2 CE 从 `1.7737813` 降至 `1.5642970`，同时 GDPH validation region accuracy 未下降。
- 目的：在更大的、仍受限的真实输入上确认该现象不完全依赖单一 seed 或 12-step 短程波动。
- 输入：冻结 Phase 2；GDPH train/val 各 32 patch；CRC panel_2 空间隔离 train 每例 8 patch、val 每例 4 patch；每个 arm 复用同一个初始化、GDPH 顺序和 seed。CRC 数据按既有 manifest `0921_173523` 的 split，不按病例划分。
- 对照：`seg_only` 与 `seg_plus_panel2`，panel_2 权重 `0.1`，100 steps，seed `17/29/43`；每个独立运行写入时间戳子目录。
- 输出：`/nfs-medical3/zyh/fewclick_v2/semantic_phase3/phase3_seg_panel2_joint_smoke_0922_182926_{17,29,43}/` 与同根日志。
- 计算和监控：GPU 1，tmux session `phase3_panel2_gate_0922_182926`，每个子运行追加 `EXIT_CODE`；该任务只读输入，输出为新目录。
- 验收：所有步骤有限；三个 seed 的联合 arm 均不使 GDPH validation region accuracy 低于同 seed `seg_only` 超过 `0.01`；报告 CRC validation CE 的均值和逐 seed 值。若失败，不扩大到全量训练。
- 状态：待运行。
