# Phase 3 全切片 RNA 联合分割 Gate

问题：全切片 RNA gate 已证明同切片内 H&E→RNA 信号略优于打乱对照，但尚未检验该监督加入 FewClick 的 Phase 3 cell branch 后是否会伤害原始 v4 分割训练。

方案：扩展现有 `run_phase3_seg_rna_joint_smoke.py`，使其可读取 64 份 CosMx contract 并在每个 GDPH segmentation 优化步骤中随机抽取 36 个 RNA patch。对照 `seg_only`、`paired_rna`、`shuffled_rna`；Phase 2 与原始 Phase 3 checkpoint 固定，更新相同的 Phase 3 `CellToRegionAttention` 和 region classification head。先以 20 steps 的真实 loader/model/loss 路径 smoke 验证，随后才运行三个 seed、200 steps 的中等规模 gate。

输入：GDPH 原始 v4 的训练/验证 routing、Phase 2 checkpoint、Phase 3 checkpoint，以及 `cosmx_region_rna_target_smoke_0908_062000_x*_y*/` 的 64 份 H&E-only/RNA target contract。RNA 只参与训练 auxiliary MSE，不进入推理。

验收：所有 arm 的 loss/metric 必须 finite；paired RNA 不应相对 `seg_only` 产生实质性 GDPH validation region-accuracy 下降，且应与 shuffled RNA 区分。该 gate 仍不是最终点击分割 Dice/IoU；只有通过后才值得实现 Phase 3 cache → Phase 4–6 的完整 downstream 消融。

20-step smoke：`0908_213100` 已通过真实 GDPH loader、冻结 Phase 2、原始 Phase 3 checkpoint、64 个 CosMx contract 和联合 loss 路径。三 arm 均为 finite：`seg_only/paired_rna/shuffled_rna` 的 validation region accuracy 为 `0.870588/0.874510/0.874510`，mean segmentation loss 为 `0.231522/0.233257/0.233193`。结论是扩大后的 RNA batch 没有出现立即的分割退化；但 paired 与 shuffled 在 20 steps 持平，尚无下游提升证据。

下一步：三个 seed 各 200 steps，固定每 seed 的 GDPH 64 train/64 val patch，RNA batch 为 36 patch。仅在 paired RNA 相对 `seg_only` 不退化、且相对 shuffled 有一致方向差异时，才进入真正的 Phase 3 cache 到 Phase 4–6 Dice/IoU 消融。
