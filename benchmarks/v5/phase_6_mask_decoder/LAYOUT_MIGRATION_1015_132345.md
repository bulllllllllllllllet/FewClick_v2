# 源码目录迁移：1015_132345

源版本：15ea5ae；用户授权整理目录。模型与训练入口的公共模块名保持 `benchmarks.v5.phase_6_mask_decoder.<原文件名>`，类的公开模块名及checkpoint state_dict键不变。文件只保留一份，不生成顶层兼容脚本。

| 原位置（本目录内） | 新位置 |
|---|---|
| `episodes.py` | [iterative/episodes.py](iterative/episodes.py) |
| `iterative_decoder.py` | [iterative/iterative_decoder.py](iterative/iterative_decoder.py) |
| `interactive.py` | [iterative/interactive.py](iterative/interactive.py) |
| `whole_slide_interactive.py` | [iterative/whole_slide_interactive.py](iterative/whole_slide_interactive.py) |
| `prepare_spatial_routing.py` | [iterative/prepare_spatial_routing.py](iterative/prepare_spatial_routing.py) |
| `train_iterative.py` | [iterative/train_iterative.py](iterative/train_iterative.py) |
| `evaluate_iterative.py` | [iterative/evaluate_iterative.py](iterative/evaluate_iterative.py) |
| `boundary_data.py` | [boundary/boundary_data.py](boundary/boundary_data.py) |
| `boundary_refinement.py` | [boundary/boundary_refinement.py](boundary/boundary_refinement.py) |
| `probe_boundary_refinement.py` | [boundary/probe_boundary_refinement.py](boundary/probe_boundary_refinement.py) |
| `dense_prompt_decoder.py` | [dense/dense_prompt_decoder.py](dense/dense_prompt_decoder.py) |
| `train_dense_boundary.py` | [dense/train_dense_boundary.py](dense/train_dense_boundary.py) |
| `continue_dense_boundary.py` | [dense/continue_dense_boundary.py](dense/continue_dense_boundary.py) |
| `point_boundary_detail.py` | [point/point_boundary_detail.py](point/point_boundary_detail.py) |
| `point_objective.py` | [point/point_objective.py](point/point_objective.py) |
| `train_point_boundary.py` | [point/train_point_boundary.py](point/train_point_boundary.py) |
| `train_point_objective.py` | [point/train_point_objective.py](point/train_point_objective.py) |
| `audit_boundary_capacity.py` | [diagnostics/audit_boundary_capacity.py](diagnostics/audit_boundary_capacity.py) |
| `report_dense_boundary.py` | [diagnostics/report_dense_boundary.py](diagnostics/report_dense_boundary.py) |
| `visualize_boundary_refinement.py` | [diagnostics/visualize_boundary_refinement.py](diagnostics/visualize_boundary_refinement.py) |
| 原 `README.md` 内容 | [iterative/README.md](iterative/README.md) |

保留内容字节不变的模块包括全部交互代码、两种点模型/损失和原点训练入口。仅 `train_dense_boundary`、`audit_boundary_capacity`、`train_point_objective` 修改跨目录源码哈希定位；模型/loss和超参数没有修改。新目标训练对历史point源代码仍严格校验原SHA256，只按模块登记位置查找迁移后的同一份内容。

整理前后SHA256及验证产物保存在 `/nfs-medical3/zyh/fewclick_v2/code_organization/phase6_1015_132345/`。历史run.json里的绝对源码路径是当时位置，历史文件不回写；需要手动核查历史源码时，按本表或 `module_source` 定位。实验权重、缓存、图片和历史结果不移动、不覆盖。
