# CRC 局部 Patch 到 Panel 2 区域证据候选对齐

- 输入：医生审核通过的 1,553 个局部 H&E patch 缓存、三例 CRC 的 panel_2 到 panel_1 H&E 组合配准矩阵，以及 `crc_region_supervision_inventory_0921_004107`。
- 模块：`benchmarks/v5/phase_3_cell_region/tools/build_crc_patch_region_alignment.py`；仅 CPU，输入数据均只读。
- 输出：`/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_patch_region_alignment_0921_170938/`。
- 目的：把局部 patch 的四角通过已存配准矩阵映射到公共 H&E 坐标（`mpp=0.2457`），检出与其相交的 1024 µm 网格，并携带 panel_2 五类免疫细胞观测计数与覆盖率，作为区域监督候选表。
- 空间与泄露边界：表中保留 `spatial_group_id`；后续 split 必须按该字段及上下文足迹去重/缓冲。病例 ID 只作溯源及分层报告，不作划分依据。
- 非目标：空间相交不证明 FewClick phase-2 region token 与该网格一一对应；故全部记录标为 `candidate_only=true`，不得直接作为训练 target，直到 token-footprint 契约完成验证。
- 实际命令：`PYTHONPATH=. conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/build_crc_patch_region_alignment.py --timestamp 0921_170938`。
- 验收标准：输出必须只包含本地 QC 成功且 `he_tissue_fraction >= 0.15` 的 patch；每行具备 panel_2 五类计数、观测/映射计数、覆盖率和空间组；单元测试验证矩阵角点与网格相交逻辑。
- 结果：完成，`EXIT_CODE=0`。共 2,473 条 patch--网格候选记录，来自 1,553 个唯一 patch；按病例为 2017：275/180、2019：1,272/809、2022：926/564（记录数/patch 数）。对应 panel_2 覆盖率中位数为 0.941752、0.862186、0.643057；五类计数、观测/映射计数、覆盖率、空间组字段均存在，数值字段无非有限值。
- 验证：`PYTHONPATH=. conda run -n aligner python -m unittest benchmarks.v5.phase_3_cell_region.tests.test_crc_patch_region_alignment benchmarks.v5.phase_3_cell_region.tests.test_crc_region_supervision -v`，3 tests passed；`py_compile` 和 `git diff --check` passed。
- 状态：完成；仅作为 panel_2 区域辅助监督候选，尚未连接 Phase 2 token footprint，未启动 GPU 训练。
