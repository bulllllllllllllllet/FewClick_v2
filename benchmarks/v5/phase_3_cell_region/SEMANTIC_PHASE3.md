# 第一阶段实现接口

实现：身份 reg64→128→类别概率；软区域完整检测清单组成；物理面积密度；冻结旧 Phase3 加零初始化残差。输出仍为256维。空间关系、子类头、器官条件、自有编码器、最终交互训练暂不在此版本。

`semantic_cells.py` 不读取核类别或训练标签。`semantic_contract.py` 拒绝真实核类别输入和未验证配准。没有有效标注时不更新参数。第一版只接收全量已检测细胞清单，不实现子样本比例估计；现有最多255细胞的截断缓存不能直接宣称完整。

## Manifest

JSON 包含 `schema_version="semantic_phase3_v1"`、显式 `identity_classes`、`tissue_classes` 字符串列表、`provenance` 和 `samples`。

provenance 必须给出 `feature_kind="he_only_xcell_reg64"`、`sampling="complete_detected_inventory"`、`nuclei_class_source="he_prediction"` 或 `"unavailable"`、`detector_version`、`xcell_checkpoint_sha256`、`phase2_checkpoint_sha256`、`ontology_version`、`coordinate_version`、`label_source`、`registration_qc`。后者为 `cell_correspondence_verified` / `region_correspondence_verified` / `same_he_annotation`，仅区域配准不允许启用细胞标签。声明来自数据构建及QC记录，读取器本身不能证明配准正确。

sample 字段：`sample_id`、`case_id`（患者级规范分组）、`organ`、`split`（train/val/test）、`path`（相对manifest或绝对NPZ路径）、`sha256`。不同切片/扫描须使用同一个患者group；全manifest检查病例跨split和重复路径，NPZ逐次校验hash。当前训练入口要求独立病例train/val，不能把三例内部空间验证伪装为患者留出。

每个NPZ保存一张patch：

| 字段 | 形状 | 说明 |
|---|---|---|
| region_tokens | K×256 float32 | 冻结Phase2输出 |
| cells | N×74 float32 | xy、log面积、7维核类、reg64；无核类时使用unknown one-hot |
| cell_ids / cell_valid | N / N bool | 唯一cell ID；padding不计入统计 |
| assignment_at_cells | K×N float32 | 细胞中心取样，K维和为1 |
| region_area_um2 | K float32 | 全有效patch域的软区域物理面积，明确网格尺度；不是原始像素数 |
| identity_allowed / identity_valid | N×C bool / N bool | 允许标签集合、已审核的可用标记；未标全部false |
| tissue_target / tissue_valid | K×T float32 / K bool | 独立组织ROI的软类别分布；未知不当背景 |
| composition_target / composition_valid | K×C float32 / K bool | 完整、相同分母的独立组成标签；部分marker不填完整组成 |

含GT核类的旧数据只允许诊断复现，不允许用此manifest作为部署主结果。CRC现有规则细胞类型不能自动映射为该词表：需明确QC与坐标对应，并实际提取H&E reg64。

## 训练

`conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/train_semantic_phase3.py --manifest <verified.json> --baseline-checkpoint <v4_phase3.pth> --stage identity --timestamp MMDD_HHMMSS`

融合阶段额外 `--stage fusion --identity-checkpoint <best_identity.pth>`；该checkpoint必须来自同一manifest。首版每步一patch，病例均衡抽样；尚未实现设计中的有效batch16累积，不作为同预算正式对比。输出只能写入FewClick NFS产物根的新目录，保存epoch候选而不覆盖旧checkpoint。

评价：患者宏平均supported-class F1、每类precision/recall/support/confusion、Brier、15箱ECE、独立组成MAE。部分身份集合不算确定类别准确率；区域F1使用soft组织分布argmax，仅作区域probe。温度暂为1，未实现校准拟合；没有将此结果表述为J5 Dice或端到端检测分类精度。

独立评价/缓存导出：`conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/evaluate_semantic_phase3.py --manifest <verified.json> --checkpoint <best.pth> --split val --mode evaluate --timestamp MMDD_HHMMSS`。`--mode export` 只允许融合checkpoint，输出每patch的256维region tokens及源contract索引/指纹；不导出教师标签。该缓存是patch级接口，尚未接通whole-slide的现有routing消费者，不应直接覆盖旧缓存。

## 当前验证结果与下一步

2026-09-11：16项测试通过；真实GDPH smoke 使用patch `1001258-20-HE-DX1__x2048_y1536` 的96个细胞、真实组织mask和冻结Phase2/Phase3 checkpoint。读入像素尺度0.25200000394 µm/px，组织loss三步为0.917275/0.874323/0.871028，初始化和保存重载最大误差均为0，输出 `[1,64,256]`。身份头在该smoke中为随机冻结头，仅验证形状与融合计算，不能据此评价身份或分割增益。

首轮正式CRC训练尚缺经过核验的H&E细胞特征—身份标签配对manifest及组织监督范围。直接检查 `crc_label/data/{1,2,3}/summary.json` 均只记录一个Tumor annotation，其他类仅在classifiers中有名称；这只说明该QuPath入口未提供结构组织多类标注，不排除更深的派生目录另有候选ROI。后续先核对这些ROI与配准，再构建manifest和正式训练，不能将现有mIF规则标签自动视为H&E逐细胞真值。

## 验证

`conda run -n aligner python -m unittest discover -s benchmarks/v5/phase_3_cell_region/tests -p 'test_semantic*.py' -v`

单元测试使用显式合成fixture验证计算，不用于科学指标。真实数据smoke另登记输入、标签来源和允许的结论。

## 迁移记录

2026-09-11 15:10:06；源revision `82a516b31c1dea52006e50c82e9941cb537bc5cb`。从冻结 `benchmarks/v4/phase_3_cell_region/src/model.py` 复制 `CellToRegionAttention` 的参数命名和计算至 `src/frozen_cell_region.py`；只复制源码，未迁移权重、输出、数据或缓存。校验/统计放在新包装层；checkpoint采用严格load。v4模型只在测试中用于对照。新实现通过v5路径导入。
