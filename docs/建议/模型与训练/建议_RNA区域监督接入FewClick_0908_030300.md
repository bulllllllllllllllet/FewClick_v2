# RNA 以区域监督接入 FewClick 的建议

## 问题与当前证据

RNA 在 COAD 单 section 的 H&E-only CTransPath 细胞类别蒸馏中未通过：paired
RNA 的 spatial-block macro-F1 不超过 label-only 或 shuffled RNA。此前冻结 Phikon
也得到同方向的阴性结果。因此不建议继续扩大“RNA -> 单细胞类型分类器”的训练。

这不表示 RNA 对 FewClick 无用：正确配对 RNA 本身在局部表征任务中有强信息，而
v4 TLS smoke 证明 cell-to-region 分支会影响最终 Dice/recall。关键问题是把 RNA
监督放在与最终 `region token -> prompt -> mask` 更一致的层级。

## 推荐：区域 RNA program 蒸馏（首选）

在训练期，把一个 Region 内细胞的 RNA 汇聚成若干**预先固定、可解释的 gene-program
score**（例如免疫浸润、B/T cell、髓系、增殖、上皮/基质 program）。让 Phase 3 的
H&E-only `region token` 预测这些连续 target；推理时仍只由 H&E cell feature 和
region feature 生成 token，RNA 不输入也不缓存。

```text
训练：H&E cell/region -> Phase 3 region token -> RNA program heads
                                      └-> FewClick segmentation heads
推理：H&E cell/region -> Phase 3 region token -> Phase 5/6
```

该方案不要求“单个细胞外观足以分辨全部 RNA 类型”，而检验组织邻域形态是否能反映
区域分子生态；这更贴合 FewClick 的聚合结构。

### 与现有 cell feature 的范围区别（实现约束）

这里的 Region 不能简单理解为一个固定的方格，也不能复用之前 cell-level pilot 的
“一个细胞一条 RNA SVD target”。v4 的实际链路是：

- 每个 cell feature 是一个细胞 soft-attention H&E crop 的 `reg-64`，外加细胞的坐标、面积、核类别；其感受野是局部细胞邻域。
- Phase 2 先从整个输入 patch 的特征图产生 `K` 个**软** region token；Phase 3 对每个 cell 取该 `K` 个 token 的 assignment，再以 attention 聚合 cell feature。

所以 RNA teacher target 必须用同一个 soft assignment 构造。对 region token `k`，应由所有
坐标在该 patch 内、且拥有合格 transcript 的细胞计算：

`program_target[k] = sum_i assignment[k,i] * program_score[i] / sum_i assignment[k,i]`。

其中 `assignment[k,i]` 直接取 Phase 2 在 cell centroid 的双线性采样值；不能用四象限、
任意 crop 边界或另一个 RNA spot 网格代替。即使 v4 为内存限制只向 attention 提供了
确定性抽样的 cells，RNA target 仍应先在全部可用 RNA cells 上算出，再与 token 对齐；
并另存 coverage/有效细胞数，防止低 coverage token 的伪标签污染训练。

首个实现将使用独立的 gene-program set，而非此前用于单细胞 RNA SVD 的哈希 gene view；
同时固定 transcript qv、cell-id、Phase 2 checkpoint、patch/坐标变换及 train-only
normalization。这样它才是“用区域 RNA 监督现有 region token”，而不是换了一个范围的
RNA 分类实验。

## 必须的 gate

1. 同一 region 划分下比较：视觉/分割 loss only、加真实 RNA program、加 region 行打乱 RNA program。
2. RNA target 的 gene set 与用于构建细胞 annotation 的 marker 分离；normalization、program 权重和 split 只用 train 拟合。
3. 先做 spatial-block，随后至少做 donor/section 留出；不得只报随机细胞拆分。
4. 只有真实 RNA 臂同时超过视觉-only 与 shuffle，并在 GDPH/TLS 的 Phase 3 替换消融中提升 Dice/mIoU/Boundary-F1，才保留该 auxiliary loss。

## 其他可用方式

- **RNA 引导的区域伪标签/采样**：用 RNA program 定义免疫热点、肿瘤-基质边界或 TLS 候选区，训练 Phase 2/3 的 region proposal 与难例采样；它不要求部署期 RNA。
- **细胞组成而非细胞身份**：从 RNA 构造每个 region 的 broad cell-composition target，监督聚合器预测比例/不确定性；适合单细胞类别不稳定但区域统计相对稳定的情况。
- **配对多模态部署分支**：若未来某些病例确实同时有空间 RNA，可建立显式的 `H&E-only` 与 `H&E+RNA` 两个模式，并分别评测。不能将这个研究模式伪装成普适 H&E-only FewClick。

## 不推荐的下一步

- 不再仅因 RNA 自身分类高分而把原始 RNA feature 塞入推理输入；新 WSI 通常没有该模态，且会造成部署不一致。
- 不在当前单 COAD section 阴性 cell-type gate 后直接做全量 LoRA 或端到端六模块重训。

## 验收与风险

首个小实验应采用一个与 TLS/免疫相关的 region RNA program，报告 paired-vs-shuffled
区域 prediction 指标及下游 cell/no-cell/RNA-region-supervised 三臂消融。主要风险是
现有每 cohort 仅少量 section、跨平台 RNA 量纲不同；因此必须先统一 program score 和
donor/section split，负结果也应保留而非扩大训练。
