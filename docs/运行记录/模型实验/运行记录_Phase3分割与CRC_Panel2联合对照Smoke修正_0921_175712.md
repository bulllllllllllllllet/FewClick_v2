# Phase 3 分割与 CRC Panel 2 联合对照 Smoke（接口修正）

- 前序 `0921_174855` 在 checkpoint 加载阶段失败，无训练指标；原因及不可用产物已记录于前序运行记录。
- 修正：显式以 `CellToRegionAttention(256, 74)` 加载既有 v4 74 维 H&E cell checkpoint，避免默认 68 维配置。
- 其余输入、空间隔离 panel_2 target、主任务对照、GPU 1、12 steps、验收标准与前序记录一致。
- 输出：`/nfs-medical3/zyh/fewclick_v2/semantic_phase3/phase3_seg_panel2_joint_smoke_0921_175712/`。
- 状态：运行中。
