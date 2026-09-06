# Phikon RNA 教师 Adapter 探索训练

## 目的

在已缓存的 COAD Phikon-v2 H&E 特征上训练小型 adapter，判断正确配对 RNA program 作为训练目标能否改善**测试时仅使用 H&E**的细胞类别预测。由于冻结 backbone gate 未通过，本运行是探索性诊断，不授权替换 v4 backbone 或进入 FewClick Phase 3。

## 三臂与判定

- `h_and_e_label_only`：Phikon CLS -> adapter -> cell-type head。
- `h_and_e_plus_paired_rna_teacher`：相同 H&E 输入，另加 RNA program regression loss；RNA 不进入输入。
- `h_and_e_plus_shuffled_rna_teacher`：RNA target 在训练细胞行间打乱。

使用同一 COAD section 的 4 空间块 split、3 个随机种子。paired RNA 必须同时超过 label-only 与 shuffled RNA，才可称为 RNA teacher 有效。H&E TIFF 未携带可读 mpp；Xenium morphology OME metadata 仅证明其原始 morphology 为 `0.2125 µm/px`，故本次不解释为最终物理尺度验证。

## 命令与产物

`CUDA_VISIBLE_DEVICES=1 conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/train_phikon_rna_teacher_probe.py --device cuda:0 --output-dir /nfs-medical3/zyh/fewclick_v2/rna_teacher_probe/phikon_rna_teacher_0907_033007`

## 完成结果

首次 `0907_033007` 已完成训练但因 Pandas aggregate 的 tuple key 不能 JSON 序列化而在写最终 metadata 时退出；`seed_metrics.csv` 已产生。修正序列化后用相同种子、相同输入完整重跑为 `0907_033211`，exit code 为 0。

| Arm | macro-F1（mean ± std） | balanced accuracy（mean ± std） |
| --- | ---: | ---: |
| H&E + label only | 0.2352 ± 0.0024 | 0.2406 ± 0.0020 |
| H&E + paired RNA teacher | 0.2388 ± 0.0041 | 0.2440 ± 0.0031 |
| H&E + shuffled RNA teacher | 0.2390 ± 0.0040 | 0.2420 ± 0.0029 |

- paired RNA 相对 label-only 的 macro-F1 为 `+0.0036`，但相对 shuffled RNA 为 `-0.0002`；未满足 paired 必须稳定超过 shuffled 的验收条件。
- 所有模型在测试时只输入 H&E Phikon feature；RNA 从未作为测试输入。
- 最终结果：`/nfs-medical3/zyh/fewclick_v2/rna_teacher_probe/phikon_rna_teacher_0907_033211/`；日志：`/nfs-medical3/zyh/fewclick_v2/rna_teacher_probe/logs/phikon_rna_teacher_0907_033211/run.log`。

**结论：不通过 RNA-teacher gain gate。** 这不否定 RNA 本身含有细胞身份信息，而是表明在单 COAD section、当前 H&E crop 与冻结 Phikon feature 条件下，RNA supervision 尚未被蒸馏为可检验的 H&E-only 增益。不得据此扩大训练或替换 FewClick 默认 backbone；先补齐 mpp/多视野 contract 和多 section spatial split。
