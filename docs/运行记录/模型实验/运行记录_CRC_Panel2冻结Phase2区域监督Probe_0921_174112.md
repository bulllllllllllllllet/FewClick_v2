# CRC Panel 2 冻结 Phase 2 区域监督 Probe

- 问题：在不使用 mIF/protein 作为输入的前提下，冻结的 H&E Phase 2 region token 是否能预测 panel_2 五类免疫细胞的区域观测组成。
- 输入：空间隔离 manifest `crc_panel2_spatial_manifest_0921_173523`；每 split 按病例均衡抽取小规模 patch（train 96、val 30），但病例不作为 split 单位；真实注册 H&E、冻结 v4 Phase 2 checkpoint。
- 模型：H&E → 冻结 Phase 2 64 个 token → 可训练 `ObservedRegionEvidenceHeads.immune_head`。panel_2 细胞计数、coverage 和映射仅用于 loss target；模型输入不包含其数值、marker 或 mIF 图像。
- 坐标/监督：沿用已验证的软 footprint 投影和 `min_token_grid_fraction=0.8`；只对有效 token 计算 coverage 加权 composition CE。target 仍是 `candidate_only` 的区域观测，不当作完整组织组成或组织分割真值。
- 对照：报告验证集的 train-only 组成均值 predictor CE 作为无图像信息参考；不将随机未训练 head 称为有效基线，也不把此 probe 说成 FewClick 分割改进。
- 输出：`/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_panel2_observed_probe_0921_174112/`，GPU 1、tmux、退出码日志。
- 命令：`PYTHONPATH=. conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/train_crc_panel2_observed_probe.py --timestamp 0921_174112 --device cuda:1 --train-per-case 32 --val-per-case 10 --epochs 10`。
- 验收：真实 loader、冻结 Phase 2、投影 target、loss/backward、训练/验证均完成；所有张量有限；报告 train-only mean baseline 与最佳 val loss。只有最佳 val loss 严格低于该 baseline 时，才认为此 H&E 表示对该区域观测存在初步可学习信号。
- 结果：完成，GPU 1，`EXIT_CODE=0`。train/val 各从三病例均衡抽取 96/30 个 patch，验证有效 token 数 1,819。train-only 组成均值 predictor 的验证 CE 为 0.770354；最佳 H&E token head 验证 CE 为 0.754778（epoch 9），相对降低约 2.02%。
- 解释：冻结的 H&E Phase 2 token 对 panel_2 区域观测组成存在初步、幅度较弱的可学习信号。结果严格优于无图像信息的 train-only mean baseline，但没有组织分割标签或主任务对照，不能据此声称 FewClick 分割效果已经改善；也不能将其解释为逐细胞身份识别成功。
- 下一步：仅在相同空间 manifest 与固定分割训练协议下，比较 Phase 3 共享融合表示的 `seg-only` 和 `seg + panel_2`；同时报告原分割指标与 panel_2 辅助指标，若主指标无改善或退化则拒绝融合。
- 状态：完成；`best_head.pth` 仅为诊断 probe checkpoint，不进入部署或正式 FewClick 推理链。
