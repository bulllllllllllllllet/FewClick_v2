# CosMx COAD RNA 区域监督契约审计

目的：在既有 CosMx global-to-H&E affine 与 crop smoke 已通过的前提下，确认 CosMx COAD 的转录本 `fov + cell_ID` 能否严格映射至 polygon/annotation 的 `cellID_fov`，并核验可用于组织/TLS region RNA target 的 marker 覆盖。RNA 仅作为训练期 target；本次不训练 FewClick，也不修改 v4。

输入（只读）：`/nfs-medical3/fyx/PKU_DATA/CosMx/COAD/` 的 `transcripts/tx_file.parquet`、`segmentation_mask/polygons.csv`、`HE.tif`，以及已通过 affine 审计的 56 个 COAD cell：`/nfs-medical3/zyh/fewclick_v2/cosmx_coordinate_audit/cosmx_he_transform_0907_153100/CosMx_COAD_converted_sampled_cells.csv`。

输出：`/nfs-medical3/zyh/fewclick_v2/cosmx_rna_contract_audit/cosmx_rna_contract_audit_0908_051000/`。

实现：新增 `benchmarks/v5/phase_3_cell_region/tools/audit_cosmx_rna_contract.py`，只读取选中 cell 对应的 parquet 行，输出 cell ID overlap、转录本数、marker program 覆盖以及 global 坐标经已审计 affine 后是否落入 H&E raster。

计算：CPU；预计为分钟级读取，不占 GPU。监测标准输出和 NFS `metrics.json`；成功条件为选中 cell 的 transcript ID overlap ≥95%、经 affine 后坐标均在 H&E 范围、且至少一个组织或 TLS program 覆盖不少于两个 marker 并有非零细胞变化。若失败，仅记录数据契约问题，不启动 cache 或训练。

结果：命令正常结束，`EXIT_CODE=0`。

- 56/56 个已 affine 对齐的 cell 均能以 `cellID_fov` 对应到转录本表的 `cell_ID + fov`；共读到 62,081 条转录本，cell ID overlap 为 `1.0`。
- 这些转录本按已审计 global-to-H&E affine 映射后，`100%` 落在 H&E raster（32,812 × 46,378）内。此结果与既有 raster-level coordinate audit 一致；它不额外声称 polygon 边界级精度。
- marker 覆盖显著优于当前 COAD Xenium：上皮 6/7、间质 ECM 7/7、T 6/6、B/plasma/TLS 9/9、髓系 6/6、肌成纤维 4/4、脂肪 3/4；黏液为 2/4，血液为 1/4。
- 由此通过该数据契约，可将 CosMx COAD 用作 Phase 2 soft assignment 聚合的、组织/TLS 语义 RNA region target 的下一步 cache smoke。仍不得将 RNA 放入在线推理输入。

产物：`/nfs-medical3/zyh/fewclick_v2/cosmx_rna_contract_audit/cosmx_rna_contract_audit_0908_051000/`。
