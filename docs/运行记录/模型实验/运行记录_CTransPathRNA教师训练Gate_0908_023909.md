# CTransPath RNA 教师训练 Gate

## 问题与边界

在不改变 v4、且测试/下游缓存均只输入 H&E 的前提下，检验“仅微调
CTransPath 最后一 stage，再训练新的 `768 -> 64` adapter”能否把正确配对的
RNA 程序蒸馏为更好的细胞类型表征。RNA 绝不作为推理输入。本记录是 COAD
单 section 的空间块训练 gate，不是跨 cohort 结论，也不授权替换 v4。

## 已查阅的依据

- `docs/建议/模型与训练/建议_PhikonV2替代CTransPath与验证设计_0907_021303.md`：RNA 必须比较正确配对、打乱与仅标签三臂，并在 H&E-only 测试下判定。
- `docs/建议/数据与评估/建议_冻结Gate失败诊断_0907_174000.md`：五 cohort 冻结特征没有稳定 paired 优势，不能直接扩大为全量 LoRA/重训。
- `docs/运行记录/模型实验/运行记录_PhikonRNA教师Adapter探索训练_0907_033007.md`：冻结 Phikon 的 paired RNA 对 shuffled 无优势；本次换回与 v4 同源的 CTransPath，只检验训练路径可行性。

## 训练设计

输入为同一 COAD 1500×1500 H&E 窗口中按细胞边界 soft-attention 的 crop；
标注与 RNA transcript 按 `cell_id` 对齐。以四个空间块做 GroupKFold：

- `label_only`：H&E -> CTransPath -> 新 adapter -> 细胞类别。
- `paired_rna`：相同 H&E 输入，额外回归正确配对 RNA SVD target。
- `shuffled_rna`：仅把训练行的 RNA target 确定性置换。

骨干只解冻 `layers.3` 与 `norm`；新的 64 维 adapter 不复用旧
XCellFormer，因而不会冒充与 v4 checkpoint 兼容。主指标是 held-out spatial
block 的 macro-F1 和 balanced accuracy。

## 前置真实 smoke

`CUDA_VISIBLE_DEVICES=1 conda run -n aligner python ...train_ctranspath_rna_teacher_gate.py --device cuda:0 --timestamp 0908_154500 --epochs 1 --folds 2 --arms label_only --batch-size 48 --output-root /nfs-medical3/zyh/fewclick_v2/ctranspath_rna_teacher_smoke`

- CTransPath 历史权重 `/home/zhaoyh/FewClick/module/checkpoint/ctranspath.pth` 已真实加载，单 crop 输出为 finite `[1,768]`。
- 两个真实 fold 都完成：训练/测试细胞为 `479/462`、`462/479`；final loss 分别为 `1.4842`、`1.3557`，且最后 stage 有 31 个 parameter tensors 参与训练。
- 前台执行时限在最终 `metrics.json` 写入前中断；已保留 `fold_metrics.csv`，不把该一 epoch 数值当作质量结论。

Smoke artifact：`/nfs-medical3/zyh/fewclick_v2/ctranspath_rna_teacher_smoke/ctranspath_rna_teacher_gate_0908_154500/`。

## 待启动的长作业

计划命令：

`CUDA_VISIBLE_DEVICES=1 conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/train_ctranspath_rna_teacher_gate.py --device cuda:0 --timestamp 0908_024000 --epochs 20 --folds 4 --batch-size 24 --output-root /nfs-medical3/zyh/fewclick_v2/ctranspath_rna_teacher_gate`

计算：GPU 1（进程内 `cuda:0`）；以 tmux `fewclick_ctrans_rna_0908_024000` 运行，日志为
`/nfs-medical3/zyh/fewclick_v2/ctranspath_rna_teacher_gate/logs/ctranspath_rna_teacher_0908_024000.log`，结束时写入 `EXIT_CODE`。

验收条件：所有 12 个 fold 的 cell/RNA/边界合约、有限 loss、有限指标均成立；且 `paired_rna` 的 macro-F1 必须同时高于 `label_only` 和 `shuffled_rna`，否则仅报告负结果，不接入 FewClick 下游。

## 完成结果

作业的 12 个 fold 已全部完成；tmux 记录的 `EXIT_CODE=1` 来自最后序列化 metadata
时将 NumPy array 错用为 pandas `Index`（`labels.unique()`）的异常。该异常发生在
`fold_metrics.csv` 写入之后，不影响任何训练、预测或 fold 指标。修复入口并从原始
fold 文件生成了独立汇总，不覆盖原目录。

| Arm | macro-F1（mean ± std） | balanced accuracy（mean ± std） |
| --- | ---: | ---: |
| `label_only` | 0.185373 ± 0.070619 | 0.268239 ± 0.049921 |
| `paired_rna` | 0.184954 ± 0.074460 | 0.265556 ± 0.035741 |
| `shuffled_rna` | 0.185191 ± 0.073989 | 0.265556 ± 0.035741 |

- paired − label-only macro-F1：`-0.000419`。
- paired − shuffled macro-F1：`-0.000237`。

**结论：不通过 RNA-teacher gate。** 正确配对 RNA 没有超过仅标签监督，且没有超过
打乱 RNA 对照；不能据此声称 RNA 监督提升了 H&E-only CTransPath 表征，也不能将它接入
FewClick 下游。这个结论与此前冻结 Phikon RNA teacher 的阴性结果方向一致。

原始 fold 产物：
`/nfs-medical3/zyh/fewclick_v2/ctranspath_rna_teacher_gate/ctranspath_rna_teacher_gate_0908_024000/fold_metrics.csv`。
正式汇总：
`/nfs-medical3/zyh/fewclick_v2/ctranspath_rna_teacher_gate_summary/ctranspath_rna_teacher_summary_0908_030100/`。
