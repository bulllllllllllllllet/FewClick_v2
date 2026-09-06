# FewClick_v2 implementation status

## 08-31 02:20:13

- Purpose: 明确 RNA 特征小实验的证据边界与验收标准。
- Changed: `docs/建议_RNA特征小实验_0831_022013.md`
- Validation: NFS 数据路径、H&E、细胞边界和带 `cell_id` 的转录本字段已完成有界抽查 — passed
- Artifacts: 无
- Docs record: `docs/建议_RNA特征小实验_0831_022013.md`（recommendation）
- Commit: included in `Run RNA feature pilot`

## 08-31 02:26:52

- Purpose: 实现并运行 COAD Xenium RNA 特征证据型小实验。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_xenium_rna_pilot.py`; `docs/运行记录_RNA特征小实验_0831_022417.md`
- Validation: `conda run -n aligner python -m py_compile benchmarks/v5/phase_3_cell_region/tools/run_xenium_rna_pilot.py`；`conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/run_xenium_rna_pilot.py --timestamp 0831_022417` — passed；融合特征 macro-F1/ balanced accuracy 增益分别为 `+0.3987` / `+0.3833`
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_rna_feature_pilot_0831_022417/`
- Docs record: `docs/运行记录_RNA特征小实验_0831_022417.md`（run record）
- Commit: included in `Run RNA feature pilot`

## 08-31 02:32:09

- Purpose: 增加 RNA 特征行置换负对照，检验此前增益是否依赖正确的细胞—RNA 配对。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_xenium_rna_pilot.py`; `docs/建议_RNA打乱对照_0831_023209.md`; `docs/运行记录_RNA打乱对照_0831_023421.md`
- Validation: `conda run -n aligner python -m py_compile benchmarks/v5/phase_3_cell_region/tools/run_xenium_rna_pilot.py`；`conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/run_xenium_rna_pilot.py --timestamp 0831_023421` — passed；真实配对相对打乱 RNA 的 macro-F1 / balanced accuracy 下降为 `0.4031` / `0.3868`
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_rna_feature_pilot_0831_023421/`
- Docs record: `docs/建议_RNA打乱对照_0831_023209.md`（recommendation）；`docs/运行记录_RNA打乱对照_0831_023421.md`（run record）
- Commit: included in `Add RNA shuffle control`

## 08-31 11:18:06

- Purpose: 用冻结 v4 的真实 `CTransPath -> XCellFormer reg-64` 视觉编码器复核 RNA 特征小实验。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_xenium_xcell_baseline.py`; `docs/建议_XCell视觉基线_0831_111806.md`; `docs/运行记录_XCell视觉基线_0831_111806.md`
- Validation: `conda run -n aligner python -m py_compile benchmarks/v5/phase_3_cell_region/tools/run_xenium_xcell_baseline.py`；`CUDA_VISIBLE_DEVICES=1 conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/run_xenium_xcell_baseline.py --timestamp 0831_111806 --device cuda:0`（tmux）— passed；1,020 个细胞的 XCell `reg-64` 均为有限值，XCell+RNA macro-F1 / balanced accuracy 为 `0.9007` / `0.9088`，正确 RNA 相对打乱 RNA 的差为 `0.5263` / `0.4751`。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_xcell_baseline_0831_111806/`
- Docs record: `docs/建议_XCell视觉基线_0831_111806.md`（recommendation）；`docs/运行记录_XCell视觉基线_0831_111806.md`（run record）
- Commit: included in `Evaluate XCell visual baseline`

## 08-31 12:14:08

- Purpose: 使用 COAD Xenium 数据提供方的 `annotation` 验证 RNA 对真实细胞类型注释分类的增益。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/export_xenium_annotations.py`; `benchmarks/v5/phase_3_cell_region/tools/run_xenium_celltype_pilot.py`; `docs/建议_真实细胞类型分类验证_0831_121408.md`; `docs/运行记录_真实细胞类型分类验证_0831_121408.md`
- Validation: `conda run -n muf-clust python benchmarks/v5/phase_3_cell_region/tools/export_xenium_annotations.py --output …/annotations.csv`；`conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/run_xenium_celltype_pilot.py --annotations …/annotations.csv --output-dir …` — passed；941 个细胞、四类注释均跨四空间块存在，正确 RNA 相对打乱 RNA 的 macro-F1 / balanced accuracy 增益为 `0.7026` / `0.6160`。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_celltype_pilot_0831_121408/`
- Docs record: `docs/建议_真实细胞类型分类验证_0831_121408.md`（recommendation）；`docs/运行记录_真实细胞类型分类验证_0831_121408.md`（run record）
- Commit: included in `Evaluate annotated cell types`

## 09-07 01:05:49

- Purpose: 梳理冻结 v4 的真实训练链路，并形成 FewClick_v2 大规模重训的分阶段方案与前置门禁。
- Changed: `docs/建议_大规模重训总体方案_0907_010549.md`
- Validation: 核查 `benchmarks/v4` Phase 1–6 训练入口、配置、cache 构建器、联调台账及现有 v4 checkpoint 目录 — passed
- Artifacts: 无
- Docs record: `docs/建议_大规模重训总体方案_0907_010549.md`（recommendation）
- Commit: included in `Plan large-scale retraining`

## 09-07 01:21:42

- Purpose: 区分单癌种/多癌种训练要求，并审计当前本地多癌种数据对细胞预训练与组织分割主训练的就绪性。
- Changed: `docs/建议_多癌种数据就绪性审计_0907_012142.md`
- Validation: 对数据目录 CSV 的候选条目、COAD/HCC/OV 的 Xenium/CosMx 实际文件、肺/乳腺/肾/前列腺 Xenium archive 合约及存储规模完成只读抽查 — passed
- Artifacts: 无
- Docs record: `docs/建议_多癌种数据就绪性审计_0907_012142.md`（recommendation）
- Commit: included in `Audit multicancer data readiness`

## 09-07 01:28:47

- Purpose: 对数据集目录 CSV 的全部服务器路径做有界、只读存在性和模态线索审计，筛选多癌种细胞学习候选。
- Changed: `benchmarks/v5/tools/audit_dataset_catalog.py`; `docs/运行记录_数据集目录全量审计_0907_012847.md`
- Validation: `conda run -n aligner python benchmarks/v5/tools/audit_dataset_catalog.py --timestamp 0907_013108` — passed；`audit.csv` 含 44 条记录，32 条存在、12 条缺失、0 条错误；平台粒度区分后得到 6 个已解压的 Xenium/CosMx RNA+边界候选。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/dataset_audit/dataset_catalog_audit_0907_012847/`; `/nfs-medical3/zyh/fewclick_v2/dataset_audit/dataset_catalog_audit_0907_013108/`
- Docs record: `docs/运行记录_数据集目录全量审计_0907_012847.md`（run record）
- Commit: included in `Audit dataset catalog paths`

## 09-07 01:32:58

- Purpose: 深度审计 COAD/HCC/OV 的 Xenium/CosMx 六个候选 cohort，验证 H&E、边界、H5AD annotation 和转录本的实际数据合约。
- Changed: `benchmarks/v5/tools/audit_cell_supervision_cohorts.py`; `docs/运行记录_细胞监督候选深度审计_0907_013258.md`
- Validation: `conda run -n muf-clust python benchmarks/v5/tools/audit_cell_supervision_cohorts.py --timestamp 0907_014551`（tmux）— passed；5/6 cohort 通过，H5AD—边界 ID overlap `0.9996–1.0000`、annotation 覆盖约 `90%`；CosMx-OV 因转录本 Parquet footer 异常降级为 review。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/dataset_audit/cell_supervision_audit_0907_014551/`
- Docs record: `docs/运行记录_细胞监督候选深度审计_0907_013258.md`（run record）
- Commit: included in `Audit cell supervision cohorts`

## 09-07 01:52:33

- Purpose: 统计可用 cohort 的可证明 section/donor 证据，并将跨癌种 H5AD `annotation` 映射为第一版 broad cell ontology。
- Changed: `benchmarks/v5/tools/audit_cell_ontology.py`; `docs/运行记录_细胞队列样本与本体审计_0907_015233.md`
- Validation: `conda run -n muf-clust python benchmarks/v5/tools/audit_cell_ontology.py --timestamp 0907_015339` — passed；5 个 contract-passing cohort 各发现 1 个 H&E section，所有已注释原始标签映射至 8 个 broad classes，未映射类别为 0；donor 数被正确保留为 unknown。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/dataset_audit/cell_ontology_audit_0907_015339/`
- Docs record: `docs/运行记录_细胞队列样本与本体审计_0907_015233.md`（run record）
- Commit: included in `Audit cell ontology`
