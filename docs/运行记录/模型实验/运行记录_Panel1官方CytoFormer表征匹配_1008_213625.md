# Panel1 官方 CytoFormer 表征匹配

- 启动时间：2026-10-08 21:36:25 CST。
- 目的：在固定 3,177 个已标注细胞、717 个 512px 图块和原有空间划分上，比较官方 CytoFormer 56µm 视野特征与冻结 CTransPath 多视野特征的细胞检索、三级/五类匹配及聚类表现。此轮不训练分类头。
- 输入：`panel1_small_cell_crop_full_1008_143551/cells.csv`、原始配准 H&E/mask、`panel1_metric_views_1008_152914`、官方源码 `cytoformer_source_raw_1008_203300`。用户提供只读权重 `/nfs-medical3/zyh/cytoformer/checkpoint.pth`，大小 2,726,343,169 字节，实测 SHA256 `e93362f4e894cad5d02cf72d944702f7fd9b95b5fc25eedf01406adf1c7b7ead`，与 Hugging Face 官方文件元数据一致；不复制或修改原文件。
- 代码版本：`f433632`；`benchmarks/v5/tools/extract_crc_panel1_cytoformer.py` 和 `evaluate_crc_panel1_metric_matching.py`。通过 FewClick 产物目录下的符号链接和校验审计将用户权重交给现有提取器。
- 输出：权重引用 `semantic_phase3/cytoformer_official_weight_ref_1008_213625`；单图块 smoke `semantic_phase3/cytoformer_official_smoke_1008_213625`；完整特征 `semantic_phase3/cytoformer_official_features_1008_213625`；匹配评估 `semantic_phase3/panel1_cytoformer_matching_1008_213625`。路径共同前缀为 `/nfs-medical3/zyh/fewclick_v2/`。
- 计算：`aligner` 环境，GPU1，CPU4线程，批量32；先 `--only-patch x21504_y11264` 通过真实权重、图像、mask、strict加载与有限特征检查，再全量；全量和评估用命名 tmux、时间戳日志、真实 worker PID 与 `EXIT_CODE` 记录。
- 验收：smoke 和全量提取分别成功；全量为 3,177 行/1,536 维、源清单/权重/输出哈希一致且特征有限；评估与原基线同一 train-only 检索库和空间划分，报告补充测试的三级和五类 macro-F1、accuracy 与无监督 ARI，并对比原 CTransPath。该清单来自单个 2019 病例，不能推断跨病人或跨癌种表现。图块组成只有选中细胞标签，不是完整核组成真值。
- 监控：全量特征 tmux `p1_cytoformer_1008_213625`、真实 Python worker PID `3974422`、日志 `semantic_phase3/run_cytoformer_official_1008_213625.log`；匹配 tmux `p1_cyto_eval_1008_213625`、worker PID `3981600`、日志 `semantic_phase3/run_panel1_cytoformer_matching_1008_213625.log`。两日志均有 `EXIT_CODE=0`。
- 状态：真实权重单图块 smoke 通过（1 个目标，严格加载、真实 H&E/mask 与有限输出）；717 图块/3,177 细胞全量完成，特征为 float32 `(3177,1536)` 且全有限，审计 `source_cells_sha256=00186523e69575c42aea9f5af8c2714b8d47f493bae00a832ab78b966f2befcf`、`features_sha256=e60cb10fbfba0b598eb022a97fd9dabc88167119445153872a6ce5145745bd29`。源代码官方 revision `36b667e40a440d0a8faea8296709aadfcffe6616`，权重 strict 加载通过。

## 同清单结果

所有表征冻结；只用 train 1,359 个细胞拟合标准化、检索库、每类 3 个原型和 PCA/KMeans。下面主表采用 train-only z-score 后 L2 归一，补充 test 含每细类 50 个细胞，共 250 个、92 图块。三级为上皮/内皮/间质；五类为上皮/血管内皮/淋巴内皮/CAF/SMA 间质。完整原 val/test、单位归一化、混淆矩阵和逐类 F1 在 `panel1_cytoformer_matching_1008_213625/metrics.json`。

| 表征 | 三级 5-NN macro-F1 / accuracy | 三级 3 原型 macro-F1 / accuracy | 五类 5-NN macro-F1 / accuracy | K=3 对三级 ARI |
|---|---:|---:|---:|---:|
| CTransPath 128px soft | 0.527 / 0.524 | 0.478 / 0.476 | 0.328 / 0.336 | −0.008 |
| CTransPath 112px raw | 0.524 / 0.536 | 0.483 / 0.488 | 0.330 / 0.340 | 0.007 |
| 官方 CytoFormer 112px raw | **0.646 / 0.660** | **0.654 / 0.652** | **0.478 / 0.480** | **0.174** |

- CytoFormer 相对 CTransPath 128px soft 的三级 5-NN macro-F1 差为 `+0.119`、accuracy 差为 `+0.136`。按 92 个图块配对重抽样 2,000 次，macro-F1 差 95% bootstrap 区间 `[+0.030,+0.215]`，accuracy 差区间 `[+0.044,+0.233]`。三级 3 原型 macro-F1 差 `+0.176`、区间 `[+0.087,+0.262]`。固定协议与种子 20261008 的明细见 `panel1_cytoformer_matching_1008_213625/paired_patch_bootstrap.json`。
- 原 val/test 上官方 CytoFormer 三级 5-NN macro-F1 分别 0.666/0.667，对应 CTransPath 128px soft 为 0.497/0.556；补充 test 的五类 5-NN逐类 F1 为上皮 0.581、血管内皮 0.458、淋巴内皮 0.326、CAF 0.388、SMA 间质 0.636。细类尤其淋巴与 CAF 尚未达到可靠识别。
- 官方 CytoFormer 的 K=3 对三级 ARI 为 0.174，虽好于该基线的 −0.008，仍远不足以将无监督簇直接解释为真实细胞类别。补充 test 中每图块至少 10 个已选标签细胞的组成诊断只覆盖 3 个图块，原型预测平均总变差 0.192；类别均衡抽样组成不可替代整图所有核的真实组成。
- 结论范围：同一 2019 病例内、空间分组隔离的 H&E 已选细胞任务，官方 CytoFormer 冻结特征匹配明显优于本次冻结 CTransPath 对照；这尚不是跨病人/癌种识别，也不是 Phase3 组织分割或点击 Dice 增益。若接入全图组成，需要对全部核的 unknown/other 与类别来源做独立核验，再进行分割配对实验。
