# 迭代 Decoder 收紧证据与保持损失 Probe

- 时间：2026-09-23 16:12:01 CST。状态：完成，`EXIT_CODE=0`；两轮均通过开发性门禁。
- 问题：符号单调/对向提示版本两轮均未通过；负点错误区与正确正区视觉差分别约 0.142/0.071，region 更新仍伤害正确正区约 0.022。
- 改动：相似证据 `max(new_sim−opposite_sim−0.05,0)`，使弱区分区域不传播；正确区域保持损失权重 0.5→2.0。正负单调结构和像素点击锚点保持。
- 输入：冻结 J5、空间划分和 cell 路由同 `运行记录_迭代Decoder符号单调与对向保护Probe_0923_160724.md`。数据 seed 与 64 train/24 val episode 固定，便于配对。
- 输出：`/nfs-medical3/zyh/fewclick_v2/iterative_decoder/probe_0923_161201/`；日志 `probe_0923_161201.log`。
- 计算：GPU 1，64 train/24 val、2 epochs、每例最多 2 轮、batch 1；tmux `iterative_decoder_0923_161201`。
- 监控：日志、逐轮 JSONL、quality、checkpoint、`EXIT_CODE`。
- 验收：远端 FP/FN 修正方向正确，正负点击冲突 0，正负 Dice 均至少 −0.001，正确正区 ≥−0.01、正确负区 ≤+0.01。仅满足全部条件才保存合格 checkpoint。
- 评价边界：该证据阈值参考了探索性 val 诊断；结果不能作为独立验证。旧 J5 初始化暴露仍在。
- 结果：epoch 0/1 的 val 正点 Dice +0.000799/+0.002269，负点 Dice +0.000107/+0.000256；正点远端 FN 概率 +0.01659/+0.02674，负点远端 FP −0.01868/−0.00918；新增负点后正确正区 −0.00732/−0.00331，新增正点后正确负区 +0.00165/+0.00271。像素点击冲突均 0，所有 quality checks 均通过；按合格轮次中最低 val loss 保存 epoch 1 checkpoint。
- 追加重载评估（2026-09-23 16:13:53）：以 epoch 1 checkpoint 在 GPU 4 重载并评估 64 个空间 val episode，每例 2 轮；输出 `/nfs-medical3/zyh/fewclick_v2/iterative_decoder/eval64_0923_161353/`。验收为双侧远端方向、正确区保持和 Dice 同门槛；仍为暴露初始化下的开发性复核。
- 重载评估结果：64 个 val episode 共 128 步，点击冲突 0；正点 Dice +0.005847、远端 FN +0.028804、正确负区 +0.003621；负点 Dice −0.000162、远端 FP −0.009287、正确正区 −0.002946。均通过预设开发性门槛，但负点 Dice 增益尚未稳定为正。
- 较低暴露子集复核（2026-09-23 16:16:38）：同一 checkpoint 在新空间 val 且 `historical_split=val` 的 64 个 episode 上重载评估，GPU 4，输出 `/nfs-medical3/zyh/fewclick_v2/iterative_decoder/eval_historical_val_0923_161638/`。该子集未参与旧 v4 checkpoint 的梯度训练，但曾用于旧模型选择，仍不是独立测试。记录双侧方向、Dice 和正确区变化，不据此改训练超参数。
- 子集结果：128 步、像素点击冲突 0；正点 Dice +0.011251、远端 FN +0.033616、正确负区 +0.003054；负点 Dice +0.000446、远端 FP −0.010739、正确正区 −0.002786。双侧均在开发性门槛内；仍受旧模型选择与当前方案调参影响。
