# CosMx 全切片 RNA 规模 Gate

问题：此前同一张 COAD CosMx H&E 切片仅使用 4 个局部 view（36 patch、3,530 个 RNA target cell），paired-vs-shuffled RNA 预测比较不稳定。当前按用户方向暂不把跨病人泛化作为门槛，直接以全切片的 64 个空间不重叠 view 检验扩大样本后，H&E cell/region 特征对 RNA 组织 program 是否仍优于随机对应。

输入：`/nfs-medical3/zyh/fewclick_v2/cosmx_fullslide_rna_cache/cosmx_region_rna_target_smoke_0908_062000_x*_y*/region_rna_contract.npz` 的 64 个缓存。它们合计 576 个 512×512 patch、41,781 个 RNA target cell；输入为冻结 v4 Phase 2 `region_tokens`、H&E-only XCell cell features、soft assignment，目标为 6 个语义 RNA program（epithelial、stromal_ecm、t_cell、b_plasma_tls、myeloid、muscle）。RNA 不作为模型输入，也不写入在线缓存。

方法：运行 `train_phase3_region_rna_aux_gate.py`，训练 v4-contract `CellToRegionAttention` 和线性 RNA head；每个 fold 分别训练真实配对 target 与在训练集内打乱 cell target 的对照。采用 3 个随机种子、4-fold `window × quadrant` 空间组划分，得到各 12 个 paired / shuffled fold。GPU 1；输出根为 `/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/phase3_region_rna_aux_gate_0908_212344/`，tmux 会话 `fewclick_cosmx_fullgate_0908_212344`，日志将包含真实 worker PID 与 `EXIT_CODE`。

验收标准：64 个 contract 均可加载、训练损失及指标 finite；主要判据为 paired 平均 `R²_vs_train_mean` 高于 shuffled 平均值。即使通过，这也只证明本切片内 H&E→RNA 的辅助监督信号，不代表跨病人泛化或下游分割改进；若不通过，将否定“此前仅因样本量少”这一解释。

启动状态：2026-09-08 21:25:10（UTC+08:00）已在 GPU 1 启动。tmux `fewclick_cosmx_fullgate_0908_212344`；launcher PID `2236180`，真实 Python worker PID `2236493`。日志为 `/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/fullgate_0908_212344.log`；启动后 GPU 利用率 90%、显存 1,442 MiB，64 个 contract 已被脚本接受。完成后将在本记录补充 24 fold 指标和结论。
