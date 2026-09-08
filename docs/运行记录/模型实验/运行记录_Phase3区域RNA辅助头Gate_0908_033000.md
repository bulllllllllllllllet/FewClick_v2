# Phase 3 区域 RNA 辅助头 Gate

## 目的

在已通过的 Phase 2 soft-region RNA target cache 上，真实调用 v4 的
`CellToRegionAttention`，冻结 Phase 2 token、训练 cell-to-region attention 与 8 维
RNA head。验证实现的是“Phase 3 融合 token 后接 RNA prediction head”，而不是把 RNA
拼进输入。

## 对照与边界

- 配对臂：每个 fused region token 回归其正确 soft-assignment RNA program target。
- 打乱臂：仅在 train patch/token 行内置换 target；H&E token、cell input、split 不变，
  test 始终对正确 RNA target 评分。
- 4 个空间组 GroupKFold，3 seeds；Phase 2 checkpoint、cell/RNA cache 均冻结。
- 当前仅 9 个 COAD patch，且 nuclei-class one-hot 不可用而置零；此实验只验证
  `CellToRegionAttention + RNA head` 的机制，绝不作为 segmentation 或泛化结论。

## 命令与验收

首先用 `--epochs 1 --seeds 20260908` 经过真实模型/loss/evaluation smoke；通过后运行：

`CUDA_VISIBLE_DEVICES=1 conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/train_phase3_region_rna_aux_gate.py --device cuda:0 --contract-dir /nfs-medical3/zyh/fewclick_v2/region_rna_target_smoke/region_rna_target_smoke_0908_032000 --timestamp 0908_033000 --epochs 100`

输出：`/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/phase3_region_rna_aux_gate_0908_033000/`。
验收：所有 loss/metric finite，且 paired RNA 的 held-out R2 高于 shuffled RNA。通过仅表示
可继续设计包含 segmentation loss 的多任务 Phase 3 gate。

## 当前状态

真实 1-epoch smoke 已通过（`0908_032900`）：8 个 fold（2 arms × 4 spatial folds）
均完成，loss/metric 有限并写出 metadata；因仅一个 epoch，paired/shuffled R2 均为大幅
负值，不能作质量判读。该 smoke 的唯一结论是 v4-contract `CellToRegionAttention`、RNA
head、train-only shuffle、loss 与 held-out evaluation 链路真实可运行。

下一步将以新目录运行 3 seeds × 2 arms × 4 folds × 100 epochs；完成后在此追加最终指标。
