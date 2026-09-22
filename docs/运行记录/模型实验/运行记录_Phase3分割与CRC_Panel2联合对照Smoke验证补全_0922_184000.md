# Phase 3 分割与 CRC Panel 2 联合对照 Smoke 验证补全

- 时间戳：`0922_184000`。
- 依据：`0922_182700` 已完成真实 12-step joint smoke，但其结果仅记录训练期 panel_2 loss，缺少原验收计划要求的独立 CRC validation CE。
- 修正：runner 在固定、空间隔离的 CRC `val` patch 上，以每个有效 token 加权汇总 `val_panel2_loss` 和 `val_panel2_tokens`；同时将写入训练日志的 loss 显式 detach，去除无关 autograd 警告。
- 输入和对照：与 `0922_182700` 完全相同，包括冻结 Phase 2、GDPH 主任务 train/val、CRC panel_2 train/val、12 steps、weight 0.1、相同 seed 和两 arm。
- 输出：`/nfs-medical3/zyh/fewclick_v2/semantic_phase3/phase3_seg_panel2_joint_smoke_0922_184000/`，日志为同根 `phase3_seg_panel2_joint_smoke_0922_184000.log`。
- 计算与监控：GPU 1、tmux session `phase3_panel2_joint_0922_184000`、日志追加 `EXIT_CODE`；实际 worker PID 于完成后填写。
- 验收：所有前向、两个损失、backward、GDPH validation region accuracy 和独立 CRC `val_panel2_loss` 均有限；联合 arm 的 GDPH accuracy 不比 `seg_only` 低超过 0.01。
- 完成：tmux pane PID `4129937`，`EXIT_CODE=0`。CRC validation 共保留 `366` 个有效 panel_2 token。`seg_only` 的 `val_panel2_loss=1.7737813`，`seg_plus_panel2` 的 `val_panel2_loss=1.5642970`；联合监督使该局部观测组成 CE 降低 `0.2094843`。两 arm 的 GDPH validation region accuracy 均为 `0.896484375`，差值为 `0`，未触发“低于 0.01 即拒绝”的门槛。此处仅为冻结 Phase 2 下的 12-step 诊断：CRC 与 GDPH 不同队列，不能作为 CRC 分割增益、跨癌种泛化或最终 FewClick 点击效果结论。
