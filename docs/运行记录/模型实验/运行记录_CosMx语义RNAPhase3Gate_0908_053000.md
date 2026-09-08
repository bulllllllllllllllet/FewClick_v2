# CosMx 语义 RNA Phase 3 Gate

目的：使用 COAD CosMx 的 H&E-only `reg-64` cell feature 与 6 个固定组织/TLS RNA program target，训练 v4-contract `CellToRegionAttention + RNA head`。比较真实配对 target 与按 region 行打乱的 target；RNA 始终只作监督 target。

输入 cache：`/nfs-medical3/zyh/fewclick_v2/cosmx_region_rna_target_smoke/cosmx_region_rna_target_smoke_0908_052000/`。Phase 2 token 和 assignment 均冻结；每个 spatial fold 只以训练 patch 的 target 均值/标准差做规范化。

命令：`CUDA_VISIBLE_DEVICES=1 conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/train_phase3_region_rna_aux_gate.py --contract-dir ...cosmx_region_rna_target_smoke_0908_052000 --timestamp 0908_053000 --device cuda:0 --epochs 100 --seeds 20260908 20260909 20260910`。

计算：GPU 1，tmux `fewclick_cosmx_semantic_0908_053000`，日志 `/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/logs/cosmx_semantic_0908_053000.log`。验收条件：24 个 fold 均 finite，且 paired 的 mean R2 高于 shuffled；9 patch 单 window 仅为机制 gate，不报告为分割收益。

完成结果：`tmux fewclick_cosmx_semantic_0908_053000` 正常结束，worker PID `338143`，`EXIT_CODE=0`，24/24 fold 均有限。

| arm | mean normalized MSE | mean R² vs train mean |
| --- | ---: | ---: |
| paired RNA target | 1.146505 | -0.146505 |
| shuffled RNA target | 1.207221 | -0.207221 |

真实配对相对打乱的 mean R² 差为 `+0.060716`，方向符合 RNA-H&E 配对信息被保留的预期；但 paired R² 仍为负，表示在这个单 1536px window、仅 9 patch 的 spatial CV 中，它尚未超过训练折均值 baseline。结论是**弱的正向机制信号，未通过稳定可预测性门禁**。

不进入 GDPH 或 Phase 4--6 segmentation。合理下一步是扩展到 CosMx COAD 的多个空间 window，并保持真实/shuffle/no-RNA 的同一 split；只有 paired 结果稳定为正且再出现下游 TLS/免疫收益，才接入分割训练。

产物：`/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/phase3_region_rna_aux_gate_0908_053000/`。
