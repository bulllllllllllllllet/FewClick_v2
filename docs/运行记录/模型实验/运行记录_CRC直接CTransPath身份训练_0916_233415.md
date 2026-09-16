# CRC 直接 CTransPath 五类细胞身份训练

- 时间戳：`0916_233415`
- 目的：以冻结 CTransPath 的 768 维 H&E 细胞及其局部上下文特征，替换已拒绝的 `reg64` 身份特征，训练 CRC 五类细胞身份分类候选。
- 输入：`/home/zhaoyh/FewClick_v2/.cache/semantic_phase3/crc_ctranspath_cell_features_nfs6_0916_222359/`（1,553 图块；逐文件验证 768 维、有限值和 cell-id 对齐）；医生审核标签和固定 pooled manifest `crc_pooled_identity_manifest_0916_110200/manifest.json`。原始标签、图像均只读。
- 划分：固定同病例内图块 pooled train/val；不得解释为患者独立泛化。
- 模型：冻结 CTransPath 特征 + `LayerNorm(768) -> Linear(256) -> GELU -> Dropout(0.15) -> Linear(5)`；五类为 `CD4_T`、`CD8_T`、`Treg`、`B_cell`、`CD163_Macrophage`。
- 训练：GPU1，15 epoch，batch 512，CPU 16 线程；前 3 epoch 普通交叉熵，随后使用 train-only 平方根逆频率的 deferred re-weighting（DRW，`alpha=0.5`）。
- 指标：每个病例分别计算五类 macro-F1，再平均为 `patient_macro_f1`；以该值选最优 checkpoint。同时保留全局 macro-F1、逐病例及逐类 support/F1。
- 监控：tmux 日志和输出 `metrics.json`；训练前以真实 loader/缓存跑 1 epoch smoke。
- 验收：缓存和标签对齐、loss 有限、smoke 正常完成、正式训练写出 best checkpoint 和完整 metrics。结果完成后补充本文件。

## 修正说明

首次 smoke 将“某个图块没有属于五类本体的医生审核细胞”错误视为缓存不合法。该情况是正常的弱监督覆盖差异；现改为保留图块完整性检查，但不把无有效标签图块并入监督样本，也不因此中断训练。

## 完成结果

- 1 epoch 真实 GPU smoke：`EXIT_CODE=0`，加载、反向传播、按病例验证、checkpoint 和指标文件均成功。
- 正式训练：`EXIT_CODE=0`，完成15/15 epoch；输出为`/home/zhaoyh/FewClick_v2/.cache/semantic_phase3/ctranspath_identity_0916_234100/`。
- 最优模型：epoch 11，`patient_macro_f1=0.200883`，全局`macro_f1=0.246435`。验证集逐类F1为 CD4_T 0.457083、CD8_T 0.024186、Treg 0、B_cell 0.041916、CD163_Macrophage 0.708990。
- 判定：低于既有`reg64` DRW内部验证基准0.221140，且Treg仍未识别；本候选拒绝，不进入后续组织分割融合。该结论仅针对当前固定同病例内部图块协议。
