# CRC Panel 1/2 共享候选数据整理

- 时间戳：`0922_191000`。
- 目的：将已通过局部 QC 的 CRC H&E patch 与公共 panel_1 H&E 物理网格相交后，同时写出 panel_1 结构相关和 panel_2 免疫相关的观测组成字段，作为两条辅助监督支线共用的候选层。
- 输入：医生审核派生 H&E/XCell cache、既有局部配准矩阵、`crc_region_supervision_inventory_0921_004107`；输入均只读。
- 输出：`/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_patch_region_alignment_0922_191000/`。
- 边界：这是 patch-footprint 与 1024 µm 网格的候选相交，不是 Phase 2 token 的精确 receptive-field 监督；`candidate_only=true`。不会以当前 panel_1 coverage 预设训练阈值，先报告其分布、有效映射细胞数与每类非零覆盖。
- 验收：每条候选都有完整 panel_1/panel_2 观测字段；与既有 inventory 的网格计数一致；不改变原始数据、旧候选或 split。
- 完成：`EXIT_CODE=0`，生成 2,473 条候选。所有记录均写入 panel_1 五类结构计数、观测/映射细胞数、coverage，以及既有 panel_2 字段。panel_1 coverage 的 10/25/50/75/90% 分位数为 `0.118/0.190/0.277/0.391/0.538`；2,469 条有至少一个映射结构细胞，359 条 coverage 不低于 `0.5`。五类在候选中均有非零计数，但这不构成 token-level 真值或训练阈值选择依据。
