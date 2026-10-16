# Phase 6：掩膜解码与边界实验

这里包含三条模型路线：交互式修正、稠密像素解码、稠密预测之上的点细化。按研究用途分目录，模型权重、缓存和实验结果均在 NFS，不放在源码目录。

## 目录与阅读入口

| 目录 | 内容 | 先看哪个文件 |
|---|---|---|
| [dense/](dense/) | 带来主要边界提升的稠密分支；初训、复现、扩量及有限解冻 | [dense_prompt_decoder.py](dense/dense_prompt_decoder.py) |
| [point/](point/) | 在冻结稠密预测上增加像素细化；训练目标对照 | [point_boundary_detail.py](point/point_boundary_detail.py)、[point_objective.py](point/point_objective.py) |
| [iterative/](iterative/) | 根据后续正负点击迭代修正；图块及整张切片状态 | [iterative_decoder.py](iterative/iterative_decoder.py)、[原分支说明](iterative/README.md) |
| [boundary/](boundary/) | 早期RGB修边实验；共享的冻结J5加载与空间数据读取 | [boundary_data.py](boundary/boundary_data.py)、[boundary_refinement.py](boundary/boundary_refinement.py) |
| [diagnostics/](diagnostics/) | 容量/错误诊断、配对报告、可视化 | [report_dense_boundary.py](diagnostics/report_dense_boundary.py) |
| [external/](external/) | 独立交互模型对照；官方SegNext最小源码、输入桥接与适配训练 | [segnext_adapter.py](external/segnext_adapter.py)、[run_segnext.py](external/run_segnext.py) |
| [tests/](tests/) | 上述模块的测试 | `python -m unittest discover -s benchmarks/v5/phase_6_mask_decoder/tests` |

阅读稠密分支建议按：`dense_prompt_decoder.py`（模型）→ `train_dense_boundary.py`（初训/特征缓存）→ `continue_dense_boundary.py`（复现、扩量、有限解冻）。`high_resolution=True`使用128×128浅层特征融合；输出是加到原预测logit上的像素残差。

新增多尺度对照阅读入口：[multiscale_dense_decoder.py](dense/multiscale_dense_decoder.py)（原生浅层/中层/stem逐级融合）→ [prepare_multiscale_cache.py](dense/prepare_multiscale_cache.py)（冻结encoder缓存）→ [train_multiscale_dense.py](dense/train_multiscale_dense.py)（control/native/middle/full256四臂暖启动对照）。这条路线保留完整原生特征，经可学习投影融合；当前处于实验验证阶段。

逐点视觉交互入口：[pixel_prompt_decoder.py](dense/pixel_prompt_decoder.py) → [train_pixel_prompt.py](dense/train_pixel_prompt.py)。保留原稠密预测，读取每个点击附近的缓存图像特征，通过正负分开的attention修正像素；control/pooled/per_click三臂检验继续训练、汇总提示和逐点提示。该路线与多尺度加层实验独立，固定提示，不包含动态纠错或编码器解冻。

SegNext是独立系统对照：相同512图块与点击，内部按预训练配置插值到1024，使用官方ViT-B、密集提示融合与FPN。先编码FP32特征，再冻结backbone适配原生交互/分割头。上游MIT许可与修改清单在[segnext_vendor/](external/segnext_vendor/)。依赖使用独立target目录和[固定版本清单](external/segnext_requirements.txt)，不修改aligner现有torch。公共入口为`python -m benchmarks.v5.phase_6_mask_decoder.run_segnext --stage prepare/train ...`，细节见[运行记录](../../../docs/运行记录/模型实验/运行记录_SegNext同提示组织适配_1015_200654.md)。

SegNext续训用`--resume <checkpoint> --optimizer-policy reset/restore --epochs <总轮数>`，轮数为累计目标；保留绝对epoch样本顺序。旧8轮checkpoint缺少optimizer/RNG，需要明确reset；新checkpoint完整保存训练状态，可显式restore，状态缺失或cache不一致时报错。正式曲线继承旧轮次并标明新训练起点，16/24/32轮记录milestones。[续训记录](../../../docs/运行记录/模型实验/运行记录_SegNext冻结编码器续训32轮_1016_004100.md)。

点细化建议按：`point_boundary_detail.py`（模型与采样）→ `train_point_boundary.py`（原采样点损失对照）→ `point_objective.py`（恢复边界监督/整图Dice）→ `train_point_objective.py`（三臂训练）。点细化是实验路线，不能视为已经替换稠密分支。

交互分支另有7项pytest风格的函数测试；完整验证方式见本次[整理运行记录](../../../docs/运行记录/模型实验/运行记录_Phase6代码目录整理_1015_132345.md)。安装pytest的环境也可使用 `python -m pytest benchmarks/v5/phase_6_mask_decoder/tests` 统一运行。

## 命令和导入保持原样

**分类目录只改变源码位置；公共Python模块名仍保持原样。** 根包通过标准的包搜索路径定位这些文件，避免增加一套子包API或大量兼容文件。不要使用 `phase_6_mask_decoder.dense.*` / `point.*` 等第二套导入路径，也不要直接执行分类目录中的脚本。

```python
from benchmarks.v5.phase_6_mask_decoder.dense_prompt_decoder import DensePromptDecoder
from benchmarks.v5.phase_6_mask_decoder.point_boundary_detail import PointBoundaryDetail
```

在仓库根目录使用 aligner 环境运行：

```bash
# 稠密初训；继续训练/扩量；点细化目标对照
python -m benchmarks.v5.phase_6_mask_decoder.train_dense_boundary --help
python -m benchmarks.v5.phase_6_mask_decoder.continue_dense_boundary --help
python -m benchmarks.v5.phase_6_mask_decoder.train_point_objective --help

# 交互修正；配对报告
python -m benchmarks.v5.phase_6_mask_decoder.train_iterative --help
python -m benchmarks.v5.phase_6_mask_decoder.report_dense_boundary --help
```

每个入口使用独立输出目录，所有产物写入 `/nfs-medical3/zyh/fewclick_v2/` 的带时间戳目录。`module_source("模块名")`用于获取整理后的源码位置，跨目录的源码哈希记录不再假定所有文件互为相邻文件。新增模块需登记在根 `__init__.py` 的 `_MODULE_GROUPS`。

固定测试可视化入口：[visualize_test_boundary.py](diagnostics/visualize_test_boundary.py)。按预先固定的test图块和原提示生成GT/预测/放大图，模型检查点预先指定；`--prepared`复用同一批样本追加已完成多尺度模型。测试展示不参与训练或选模，逐例记录旧encoder历史暴露，不能把少量图当完整test成绩。

## 数据与结果记录

使用既有空间分块数据，不重新按病例分组；同病例可跨split。旧J5初始化对部分验证区域存在历史暴露，当前验证仍属开发评价。GT可用于训练损失和声明过的模拟点击，不进入实际推理选点或模型输入。

- [边界探索续跑清单](../../../docs/建议/模型与训练/组织边界探索续跑清单_1014_200033.md)：实验进度、结果与产物路径。
- [本次目录迁移清单](LAYOUT_MIGRATION_1015_132345.md)：旧文件位置与新位置对照。

本次整理不改变模型计算、损失、超参数、权重键、数据或实验结果。根目录保留包入口和说明，完整源码位于上面的分类目录。

## 2026-10-16路线状态补充

用户已停止SegNext路线：冻结encoder适配总32轮仍低于dense基准，保存代码和历史产物，不继续延长或解冻。当前dense的固定图块提示成绩尚未代表整片远处提示成绩；旧v4整片入口和v5迭代接口均未集成此dense头。后续工作以[整体流程与同片泛化审计](../../../docs/建议/模型与训练/建议_整体流程审计与同片跨癌种泛化_1016_114959.md)为准。
