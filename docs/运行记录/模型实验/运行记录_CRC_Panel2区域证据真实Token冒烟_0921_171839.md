# CRC Panel 2 区域证据真实 Token 冒烟

- 输入：`crc_patch_region_alignment_0921_170938` 的候选空间对应、三例各一个注册 H&E patch、冻结的 v4 Phase 2 checkpoint `checkpoint_epoch_028.pth`。
- 模块：`smoke_crc_panel2_region_evidence.py`。输入图像只含 H&E；panel_2 五类细胞观测仅为训练期 target，绝不作为模型输入。
- 输出：`/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_panel2_region_evidence_smoke_0921_171839/`。
- 方法：将 Phase 2 的 `assignment_low[K,H,W]` 以显式物理坐标投影至公共 H&E 1024 µm 网格，再仅在 token 对某网格的质量比例不低于 `0.80` 时，附加该网格的 panel_2 观测组成 target。
- 坐标边界：局部注册 H&E patch 使用其 panel_2 DAPI patch 原点和已有全局组合矩阵；局部 H&E/DAPI 残差没有被宣称为精确全切片校准。此路径是可审计的软空间监督候选，不建立逐 token 或逐细胞的精确真值。
- 命令：`PYTHONPATH=. conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/smoke_crc_panel2_region_evidence.py --timestamp 0921_171839 --device cpu`。
- 验收：三个病例各一真实 patch 完成 H&E → 冻结 Phase 2 → 软空间投影 → panel_2 head → loss/backward；每例至少一个有效 token target，所有张量及损失有限。该冒烟不报告模型效果，不保存可部署训练模型。
- 结果：失败，`EXIT_CODE=1`；未产生训练结果。排查显示低分辨率 `assignment_low` 的像素坐标在映射时错误地同时放大了原点坐标和像素间距，导致其物理网格为 `[348,267]` 等，而同一 patch 的候选网格为 `[21,16]` 等。已补充 `source_pixel_stride`，将 assignment 像素中心转换回原始 512 px patch 坐标后再应用 mpp；该修正先经合成几何测试验证，使用新时间戳重跑。
- 状态：失败归档；不得使用该次失败产物。
