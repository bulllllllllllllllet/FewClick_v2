# Phase 3 与 CRC Panel1 联合分割 Gate

- 时间戳：`0922_222700`。
- 前置证据：真实 12-step smoke `0922_222500` 未使 GDPH val region accuracy 下降，同时 CRC Panel1 CE 有限。
- 对照：`seg_only` 与 `seg_plus_panel1`；每个 seed 完全复用 v4 Phase3 初始化、GDPH train/val patch 选择和训练顺序。联合 arm 只增加权重 `0.1` 的 Panel1 observed-composition CE。
- 输入：冻结 Phase2；GDPH train/val 各 32 patch；空间隔离 CRC token contract 的 train/val 各 24/12 patch；Panel1 仅辅助区域组成，不作为 GDPH 像素真值。
- 运行：GPU `cuda:1`，tmux `phase3_panel1_seg_gate_0922_222700`，seed `17/29/43`，各 100 steps；输出 `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/phase3_seg_panel1_joint_smoke_0922_222700_{17,29,43}/` 与日志，日志追加 `EXIT_CODE`。
- 验收：全部有限；逐 seed 联合 arm 的 GDPH val region accuracy 不低于 `seg_only` 超过 0.01；报告辅助 CE。仅当平均主指标提高且不出现 seed 退化时，才将赢家接入固定 FewClick 点击评测。
- 完成：三 seed 均 `EXIT_CODE=0`。seed `17/29/43` 的 GDPH val region accuracy 为 `seg_only=0.8747544/0.9115914/0.8975874`，`seg_plus_panel1=0.8737721/0.9111002/0.8980798`，逐 seed 差=`-0.0009823/-0.0004912/+0.0004924`，平均差=`-0.0003270`。Panel1 validation CE 从 `1.6558214/1.6861524/1.6630736` 降至 `1.6058983/1.6553262/1.6232042`，平均降 `0.0402068`，每 seed 均改善。
- 结论：当前权重和 100-step 预算下，辅助任务本身有效，但没有提升 GDPH 分割主指标；不将该 checkpoint 接入点击评测，也不把辅助 CE 改善宣称为 FewClick 效果提升。下一轮应在不调 test 的前提下，先用 validation 消融 `panel_weight`、交替频率或可训练图像表示，再决定是否进行点击评测。
- 状态：完成。
