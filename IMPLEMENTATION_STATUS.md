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
