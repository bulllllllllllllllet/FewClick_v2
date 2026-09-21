# CRC Panel 2 区域证据真实 Token 冒烟（几何修正）

- 前序失败：`运行记录_CRC_Panel2区域证据真实Token冒烟_0921_171839.md` 发现 Phase 2 低分辨率 assignment 的 source 坐标步长错误；此前产物不可用。
- 修正：`patch_pixel_grid_indices` 新增显式 `source_pixel_stride`。对于 512 px 输入和 32×32 `assignment_low`，每个 assignment 像素中心先按 16 个 level-0 px 还原，再经 panel_2 DAPI→公共 H&E 矩阵和 `mpp=0.2457` 映射至 1024 µm 网格。原点不再被错误倍乘。
- 输入/范围/坐标边界与前序记录相同；仍只做三病例各一个真实 H&E patch 的 H&E→冻结 Phase 2→panel_2 loss/backward 冒烟，不训练正式模型、不报告分割效果。
- 输出：`/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_panel2_region_evidence_smoke_0921_172704/`。
- 命令：`PYTHONPATH=. conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/smoke_crc_panel2_region_evidence.py --timestamp 0921_172704 --device cuda:1`。
- 验收：每例至少一个有效 panel_2 token target；有限 assignment、target、loss 和梯度；结果文件注明该监督仍为 `candidate_only`。
- 结果：完成，GPU 1，`EXIT_CODE=0`。三例的有效 token 数依次为 58、64、61；第 1/2/3 例的物理网格分别为 `{(21,16),(22,16)}`、`{(25,13)}`、`{(18,13),(18,14)}`，均与候选表的网格一致。三步（每例一步）诊断 head loss 为 1.666897、1.565095、1.449894，均有限；assignment shape 均为 `64×64×64`。
- 结论：已验证真实 CRC H&E 可产生冻结 Phase 2 token，并可在明确的软空间投影和高重叠门控下接受 panel_2 区域观测组成监督。由于每个 512 px patch 仅覆盖一个或两个 1024 µm 网格，绝大部分 token 得到相同的 patch 级区域上下文 target；这不是细粒度 token/组织真值，也不构成分割效果或蛋白监督收益证据。
- 状态：完成；产物仅供下一阶段小规模、空间组隔离的辅助监督对照使用。
