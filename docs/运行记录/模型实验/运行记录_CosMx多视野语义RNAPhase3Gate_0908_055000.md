# CosMx 多视野语义 RNA Phase 3 Gate

目的：将四个 COAD CosMx H&E window 的独立 region-RNA contract 合并为 36 个 patch，复验真实语义 RNA target 是否稳定优于打乱 target。输入 cell `reg-64` 均为冻结 H&E-only XCellFormer；Phase 2 冻结；RNA 不进入输入或推理 cache。

输入 cache：`0908_052000`、`0908_054001`、`0908_054002`、`0908_054003`。四个 view 的 H&E 起点分别为 `(25500,34750)`、`(18000,27000)`、`(22000,26000)`、`(15000,21000)`，每个为 1536px、9 patch。

分割：每个 window 的 3×3 patch 按相对四象限分组，合成 16 个 `window × quadrant` 空间组；GroupKFold=4。该设置避免随机 cell 拆分，但所有数据仍来自同一 COAD section，不能称为病例级泛化。

命令：`CUDA_VISIBLE_DEVICES=1 conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/train_phase3_region_rna_aux_gate.py --contract-dirs <four dirs> --timestamp 0908_055000 --device cuda:0 --epochs 100 --seeds 20260908 20260909 20260910`。

计算：GPU 1、tmux `fewclick_cosmx_multi_0908_055000`、日志 `/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/logs/cosmx_multi_0908_055000.log`。验收条件：24/24 folds finite，且 paired mean R² > shuffled mean R²；paired R² 还应为正，才算通过稳定可预测性门禁。

完成结果：`tmux fewclick_cosmx_multi_0908_055000` 正常结束，worker PID `441334`，`EXIT_CODE=0`；24/24 fold 均 finite。

| arm | mean normalized MSE | mean R² vs train mean |
| --- | ---: | ---: |
| paired RNA target | 1.070393 | -0.070393 |
| shuffled RNA target | 1.043057 | -0.043057 |

paired minus shuffled R² 为 `-0.027336`，不满足主要验收条件。四个 fold 中，paired 的 mean R² 为 `-0.0474/-0.0207/-0.1973/-0.0162`；fold 2 是明显的跨空间分布失配，其他 fold 也未稳定超过 shuffled。故单 window 中的 `+0.0607` 是不稳定局部现象，而非可复现的 H&E→RNA supervision signal。

结论：停止将当前这 6 个 CosMx program 作为 H&E-only FewClick Phase 3 auxiliary loss 接入 GDPH、TLS 或 Phase 4--6 分割。数据/坐标/cache 的 contract 均通过，失败原因更符合**语义 RNA program 在跨空间 window 的 H&E 可预测性不足，或与冻结 XCell/v4 region token 的表征不匹配**，而非 ID 或坐标错误。下一条线若要重启，需换成与下游标注同域同任务的 TLS/immune hotspot target，且至少有多个 section/donor；不得仅增加训练步数或模型容量。

产物：`/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/phase3_region_rna_aux_gate_0908_055000/`。
