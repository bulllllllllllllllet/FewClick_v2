# COAD 组织/TLS RNA Marker 审计

审计 COAD Xenium 配对窗口中预定义的上皮、间质、T/B/TLS、髓系、肌肉、黏液、血液和脂肪
marker 是否实际出现在 panel 且具有细胞间变化。只有覆盖至少两个 marker、并有非零变异的
program 才能作为 region RNA target；不再使用 hash gene bucket。

命令：`conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/audit_coad_tissue_rna_markers.py --output-dir /nfs-medical3/zyh/fewclick_v2/tissue_rna_marker_audit/tissue_marker_audit_0908_041000`。

输入：COAD Xenium 的 `transcripts.parquet`，以及既有 `xenium_xcell_baseline_0831_111806/cell_summary.csv` 中的 1,020 个细胞；仅统计给定窗口（1000--2000, 1500--2500）的转录本。输出：`/nfs-medical3/zyh/fewclick_v2/tissue_rna_marker_audit/tissue_marker_audit_0908_041000/`。

验收条件：一个 program 至少覆盖 2 个 marker，且细胞间存在非零变化，才可成为后续区域 RNA target。

结果：命令正常结束（`EXIT_CODE=0`）。

- 上皮 program 覆盖 `EPCAM`、`MUC1`（2/7），75.59% 细胞为非零，均值/标准差为 1.531/0.917，可作为一个较弱但可用的上皮 target。
- B/plasma/TLS program 覆盖 `MS4A1`、`CD79A`、`MZB1`、`CXCL13`（4/9），仅 2.35% 细胞非零；可用于稀有 TLS/免疫区域的候选 target，但该窗口样本太少，不能单独训练或评价。
- T cell 只有 `CD3E`（1/6），脂肪只有 `LPL`（1/4）；其余间质、髓系、肌肉、黏液、血液 program 均为 0 个 marker，不满足验收条件。

结论：现有 COAD Xenium panel 不适合承载全面的组织语义 RNA supervision；应优先寻找全转录组或更宽 panel 的同域数据。此审计不改变已完成的 Phase 3 结论。
