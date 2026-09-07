# 五 cohort 冻结 cell-encoder gate

## 目的

在 CosMx global-to-H&E 仿射变换通过 raster-level 审计及真实 crop smoke 后，使用 5 个 cohort 的 288 个 H&E 双视野特征重跑 leave-one-cohort-out cell adapter gate。该运行检验新增 CosMx COAD/HCC 是否改善冻结视觉表征的跨 cohort 信号。

## 设计与边界

- Manifest：`/nfs-medical3/zyh/fewclick_v2/multisection_cell_manifest/multisection_cell_manifest_0907_154620/`
- Feature：`/nfs-medical3/zyh/fewclick_v2/multisection_cell_smoke/multisection_cell_smoke_0907_154620/phikon_dualview_features.npz`
- 每次留出一个 cohort，其他四个训练；正确 H&E 配对与训练期 H&E feature—label 打乱对照各运行 3 seeds。
- 模型是冻结 Phikon 特征上的 `2048 -> 256 -> 8` adapter/classifier；RNA 不读取、不输入模型。
- 仍没有独立 donor/patient ID，因此这只是跨 cohort/平台 gate，不是病例级泛化，也不能据此宣称精确边界配准。

## 运行与验收

- 输出：`/nfs-medical3/zyh/fewclick_v2/multisection_cell_gate/multisection_frozen_gate_0907_155700/`
- GPU：`CUDA_VISIBLE_DEVICES=1`，tmux 会话 `fewclick_fivecohort_gate_0907_155700`，日志位于同级 `logs/`。
- 通过条件：fold 的 held-out cohort 不在训练集；feature/loss finite；正确配对全体 held-out macro-F1 和 balanced accuracy 高于 shuffled control。结果将在运行完成后追加。

首次运行使用 `0907_155700` 输出目录时，发现 `smoked_cells.csv` 保留 manifest 顺序而 feature 按 `cohort_id` 分组顺序写出，触发 feature-to-cell identity check，`EXIT_CODE=1`。不复用错序 metadata；已将 gate 显式按该已记录的 cohort 分组顺序稳定重排并再次逐 ID 检验，重跑使用新输出 `0907_155300`。

## 最终状态

- 重跑 `EXIT_CODE=0`。5 个 held-out cohort × 正确/打乱 H&E 两臂 × 3 seeds 均完成；训练/test cohort 无交集，输入 feature 为 finite `[288,2048]`，训练 loss finite。
- 正确 H&E 配对：macro-F1 `0.1180`，balanced accuracy `0.1309`；打乱 H&E 配对：`0.1273`、`0.1260`。虽 BA 有 `+0.0049`，macro-F1 低 `-0.0093`，未同时超过 shuffled control，故不通过 gate。
- 结论：将候选从 3 个扩大到 5 个 cohort 修复了 data contract，却没有在每 cohort 约 56–64 个均衡 cell 的小样本条件下证明冻结视觉表征跨 cohort 泛化。不要据此进入 LoRA 或全量训练；应先扩展每类 cell 数和独立 section/donor，再进行同一 gate。

结果：`/nfs-medical3/zyh/fewclick_v2/multisection_cell_gate/multisection_frozen_gate_0907_155300/`。
