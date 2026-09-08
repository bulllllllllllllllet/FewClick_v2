# CosMx 语义区域 RNA 目标 Smoke

目的：把已经通过 ID/坐标/marker 契约的 COAD CosMx，接入 FewClick v5 的同一 region RNA target 定义。冻结 v4 Phase 2 在一个 COAD H&E 视野的 3×3 patch 上产生 64-way soft assignment；每个 region 的 RNA target 由该 assignment 对**全部** RNA 合格细胞的固定、可解释组织/TLS program score 加权平均得到。输入 Phase 3 的 cell `reg-64` 只由 H&E→CTransPath→冻结 XCellFormer 得到，不含 RNA。

输入：`/nfs-medical3/fyx/PKU_DATA/CosMx/COAD` 的 H&E、polygon、transcript parquet；已通过的 global-to-H&E affine（`cosmx_he_transform_0907_153100`）；冻结 v4 Phase 2 checkpoint 和原始 XCellFormer 权重。

输出：`/nfs-medical3/zyh/fewclick_v2/cosmx_region_rna_target_smoke/cosmx_region_rna_target_smoke_0908_052000/`。

计算与监控：GPU 1；预计数分钟，使用 tmux `fewclick_cosmx_region_0908_052000`，日志位于上述 output root 的 `logs/`。验收条件：每个 patch 均有 RNA cells；输入 cell feature 均明确为 H&E-only、74 维 finite；有效 cell 的 Phase 2 assignment 和为 1；每个 `[64, program]` target finite，并保留 RNA coverage 与 patch manifest。通过仅代表 CosMx cache contract，不能代表 segmentation 收益。

完成结果：`tmux fewclick_cosmx_region_0908_052000` 正常结束，真实 worker PID `330635`，`EXIT_CODE=0`。

- 选择 H&E window `(x=25500, y=34750, size=1536)`，共 9 个 512×512 patch。
- 881 个有 RNA 的 CosMx cell 参与同一 Phase 2 assignment 的 target 聚合；847 个输入细胞的 `reg-64` 均由冻结 H&E→CTransPath→XCellFormer 生成，未读取 RNA。
- 目标为 6 个固定语义 program：epithelial、stromal_ecm、t_cell、b_plasma_tls、myeloid、muscle。除 epithelial 的 CEACAM5 外，其余所列 marker 在此 window 的转录本中均覆盖；每个 target 使用全部 RNA cell 的 assignment 加权均值。
- cache shape：`cells=[9,153,74]`、`assignment=[9,64,153]`、`region_tokens=[9,64,256]`、`rna_targets=[9,64,6]`。所有数组 finite；每个有效输入 cell 的 64-way assignment 和为 1。

产物：`/nfs-medical3/zyh/fewclick_v2/cosmx_region_rna_target_smoke/cosmx_region_rna_target_smoke_0908_052000/`。

解释边界：本 cache 仅通过数据与 tensor contract；下一步 paired-vs-shuffled Phase 3 auxiliary gate 才能检验其配对 RNA 信号是否可由 H&E-only token 学到，二者均不等于 FewClick segmentation 提升。

## 多视野扩展计划

单一 1536px view 的 9 patch 不足以判断 paired signal 是否稳定。下一轮在同一 COAD section 中选择三个与首个 view 不重叠、且每个四象限均至少有 20 个细胞的 H&E window；每个 view 仍独立生成 immutable cache，之后合并为 36 patch，并以 `window × quadrant` 的 16 个空间组做 GroupKFold。该设置仍不是病例级泛化，但能排除单一局部视野的偶然性。

扩展结果：三个新 view 均 `EXIT_CODE=0`。`0908_054001`、`0908_054002`、`0908_054003` 分别有 1,071/894/684 个 RNA target cell 和 1,006/894/684 个 H&E-only input cell；每份均为 9 patch，所有 tensor finite、assignment sum=1。连同首个 view 共 36 patch、3,530 个 RNA target cell。它们随后用于多视野 gate。
