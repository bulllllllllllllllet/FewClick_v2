# Phase 3 分割与 RNA 交替训练 Smoke

GDPH 有组织分割 GT 而 COAD 有 RNA，二者不能在同一 patch 上算联合 loss。本实验冻结
Phase 2，从历史正式 Phase 3 cell/head checkpoint 开始；每一步以 GDPH batch 计算 region
segmentation CE，并以 COAD 的 cached token/cell batch 计算 RNA auxiliary MSE，更新同一
cell attention。比较 `seg_only`、正确 RNA 和训练 target 打乱 RNA；GDPH 固定 8 train/8 val
patch、每臂 20 steps，仅验证训练链路及是否发生立即的 segmentation 退化，不是正式性能结论。

命令：`CUDA_VISIBLE_DEVICES=1 conda run -n aligner python ...run_phase3_seg_rna_joint_smoke.py --contract-dir ...region_rna_target_smoke_0908_032000 --timestamp 0908_034000`。
输出：`/nfs-medical3/zyh/fewclick_v2/phase3_seg_rna_joint_smoke/phase3_seg_rna_joint_smoke_0908_034000/`。

## 完成结果

第一次 `0908_034000` 因误用早期 SLIC index 而报 patch coverage mismatch；未训练。已按
历史正式 v4 log 改用与 routing 同源的 multiscale patch index 及
`slic_mmap_trainval_20260717_151426`，重跑目录为 `0908_034200`，`EXIT_CODE=0`。

| Arm | GDPH mean segmentation loss | COAD mean RNA loss | GDPH val region accuracy |
| --- | ---: | ---: | ---: |
| seg only | 0.26044 | 35.27482（未加入总 loss） | 0.85938 |
| paired RNA | 0.27773 | 22.66322 | 0.85742 |
| shuffled RNA | 0.27773 | 22.46709 | 0.85547 |

解读：20 个交替优化步之后，paired RNA 相对 seg-only 的 region accuracy 差为 `-0.00195`，
相对 shuffled 为 `+0.00195`。这说明小权重 RNA auxiliary loss 没有立即导致明显的 GDPH
分割遗忘；但 8 个 train/8 个 val GDPH patch、单 seed 和 20 steps 的差异远不足以证明
downstream gain。下一步的正式 gate 必须增加固定 GDPH train/val 样本、至少 3 seeds，
并以 Dice/mIoU/Boundary-F1 而非 region accuracy 评估。

artifact：`/nfs-medical3/zyh/fewclick_v2/phase3_seg_rna_joint_smoke/phase3_seg_rna_joint_smoke_0908_034200/`。

## 中等规模重复计划

以三个 seed 各自随机但固定抽取 64 个 GDPH train patch 与 64 个独立 val patch，运行 200
交替步骤，仍比较 seg-only、paired RNA、shuffled RNA。Phase 2 和初始正式 Phase 3 checkpoint
固定；仅加 RNA head/auxiliary loss。该运行继续只报告 Phase 3 region accuracy 的稳定性，
不是 Dice/mIoU/Boundary-F1 下游结论；若三 seed 都不出现实质性退化，才有资格开始重建
token cache 并进入 Phase 4–6 的真正下游消融。
