# 多 Section 冻结 cell-encoder gate

## 目的

在不向模型输入 RNA 的条件下，验证已经通过坐标 smoke 的三套 Xenium H&E 双视野 Phikon 特征，是否存在跨癌种 section 的 broad cell ontology 信号。此 gate 只训练轻量 `2048 -> 256 -> 8` cell adapter/classifier；Phikon backbone 冻结。

## 设计

- 输入 manifest：`/nfs-medical3/zyh/fewclick_v2/multisection_cell_manifest/multisection_cell_manifest_0907_145854/`
- 输入特征：`/nfs-medical3/zyh/fewclick_v2/multisection_cell_smoke/multisection_cell_smoke_0907_150803/phikon_dualview_features.npz`
- 细胞：Xenium COAD 56、HCC 56、OV 64；CosMx 仍因未验证坐标变换排除。
- split：leave-one-cohort-out。每个 test cohort 从不出现在相应训练 fold；注意 cohort 不是独立 donor 证据，因此这是跨 section/cancer gate，**不是病例级泛化**。
- 对照：`paired_he` 使用正确 H&E feature—label 配对；`shuffled_he` 仅在训练集随机打乱 feature 与标签的配对。二者测试集都用真实 H&E feature。
- 训练期与推理期均不读取 RNA、转录本或蛋白。

## 运行

- 输出：`/nfs-medical3/zyh/fewclick_v2/multisection_cell_gate/multisection_frozen_gate_0907_151300/`
- 计算：单 GPU `CUDA_VISIBLE_DEVICES=1`，`aligner`；tmux `fewclick_frozen_gate_0907_151300`，日志写入同级 `logs/`。
- 接受准则：每个 fold 的 train/test cohort 无交集、输入 feature `[N,2048]` finite、loss finite；全体 held-out prediction 的 `paired_he` macro-F1 和 balanced accuracy 必须高于 `shuffled_he`。即使通过，也只证明该小型 section 集的视觉信号，不能替代病例级或大规模训练。

## 最终状态

- 运行完成，`EXIT_CODE=0`，3 个 held-out cohort × 2 个 arm × 3 seeds；每个 fold 的 train/test cohort 均无重叠，输入为 finite `[176,2048]` H&E feature，训练 loss 也均 finite。
- 正确 H&E 配对的汇总 macro-F1 为 `0.1159`、balanced accuracy 为 `0.1224`；打乱训练 H&E 配对对照为 `0.1458/0.1510`。因此不通过预注册的 paired-over-shuffled 接受准则。
- 三个 held-out fold 的正确配对平均 macro-F1 为 `0.0940`、平均 balanced accuracy 为 `0.1255`。训练 loss 接近 0 而 held-out 指标接近 8 类机会水平，说明小样本 adapter 能记住训练 section，不能跨 section/cancer 泛化。
- 结论：当前 176-cell 均衡 smoke 只能证明数据读取契约，不足以支撑正式多癌种训练或决定 LoRA/RNA teacher 有效。下一步应扩大每类抽样量、修复 CosMx 坐标契约并取得多个独立 section/donor；在此之前不启动全量长训练。

结果：`/nfs-medical3/zyh/fewclick_v2/multisection_cell_gate/multisection_frozen_gate_0907_151300/`。
