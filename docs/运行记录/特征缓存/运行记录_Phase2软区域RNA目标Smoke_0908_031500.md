# Phase 2 软区域 RNA target Smoke

## 目的

验证 RNA region-supervision 的关键空间契约，而非训练或替换 FewClick：冻结 v4
Phase 2 在配对 COAD H&E patch 上输出 64 个 soft region token；每个 token 的 RNA
target 必须由同一 `assignment_at_cells` 对 cell-level RNA program 加权汇聚。

## 输入与范围

- H&E/RNA：`/nfs-medical3/fyx/PKU_DATA/Xenium/COAD` 的既有 `1000,2000,1500×1500` 窗口。
- H&E cell feature：`/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_xcell_baseline_0831_111806/` 的逐 ID `reg-64`。
- Phase 2：冻结 checkpoint `.../phase_2_region_encoder_train_20260717_161240/checkpoint_epoch_028.pth`，64 soft tokens。
- 分块：窗口中 3×3 个 512px H&E patch。COAD H&E 的物理 mpp 尚未审计，本 smoke 只验证像素/ID/软 assignment 契约，不解释为跨数据集的 10× 结果。

RNA program 用 transcript qv≥20、每细胞≥30 transcripts 后的独立哈希 gene view 的 8 维 SVD
构造；它仅作 target，不输入任何模型。COAD pilot 没有可用 H&E nuclei-class 字段，74 维
cell feature 的 7 个 nuclei one-hot 暂置零，明确不以 RNA-derived label 伪造该字段。

## 命令、资源与验收

`CUDA_VISIBLE_DEVICES=1 conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/build_region_rna_soft_target_smoke.py --device cuda:0 --timestamp 0908_031500`

计算为 GPU 1；输出根：
`/nfs-medical3/zyh/fewclick_v2/region_rna_target_smoke/region_rna_target_smoke_0908_031500/`。

验收条件：所有 patch 的 cell ID 与 RNA/XCell 对齐；每个有效 cell 的 64-way assignment
和为 1；token、cell feature、assignment、RNA target 均 finite；并写出可审计的 patch
manifest。通过只代表训练输入契约可用，不代表 RNA 能提升 FewClick。

## 完成结果

首次 `0908_031500` 使用全窗口 SVD program 完成了 token/ID/assignment contract；为避免
SVD 对 held-out cells 的拟合泄漏，已将 target 改为固定 hash gene-bucket mean，并以新目录
`0908_032000` 重跑。重跑通过：9 个 patch、`cells=[9,167,74]`、
`assignment=[9,64,167]`、`region_tokens=[9,64,256]`、`rna_targets=[9,64,8]` 均 finite；
每个 valid cell 的 64-way assignment 和为 1。8 个 program 的 gene 数为
`[182,166,143,190,163,159,156,169]`。

最终 artifact：
`/nfs-medical3/zyh/fewclick_v2/region_rna_target_smoke/region_rna_target_smoke_0908_032000/`。
该 contract 通过，进入但不等同于后续 Phase 3 auxiliary-head training gate。
