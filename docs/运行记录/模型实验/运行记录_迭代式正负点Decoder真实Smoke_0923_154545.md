# 迭代式正负点 Decoder 真实路径 Smoke

- 时间：2026-09-23 15:45:45 CST。
- 状态：完成；两次真实路径 smoke 均 `EXIT_CODE=0`。
- 目的：检验 v5 逐点提示、上一轮 logits、正负纠错、soft assignment 点击投影、真实图块 loss/梯度、checkpoint 和逐轮记录。
- 模块：`benchmarks.v5.phase_6_mask_decoder.prepare_spatial_routing` 与 `benchmarks.v5.phase_6_mask_decoder.train_iterative`。
- 输入：冻结 v4 J5 checkpoint `/nfs-medical3/zyh/v4/phase6/stabilization_runs/phase6_joint_pixel_20260722_114809/checkpoint_epoch_001.pth`；对应 `run_metadata.json`；v5 空间划分 `/nfs-medical3/zyh/fewclick_v2/extensible_prompt/tissue_spatial_split_0922_001235/`；既有 v4 cell feature routing 和源 patch index。
- 输出：`/nfs-medical3/zyh/fewclick_v2/iterative_decoder/spatial_routing_0923_154545/` 与 `/nfs-medical3/zyh/fewclick_v2/iterative_decoder/smoke_0923_154545/`。已有源数据、v4 checkpoint 和 v4 目录只读。
- 计算：准备路由用 CPU；真实模型用 GPU 1，初始 4 个 train、2 个 val 图块，每图最多 2 轮，1 epoch，batch 1。通过后再定更大预算。
- 监控：命名 tmux 会话，时间戳日志，进程 PID 与 `EXIT_CODE`；保存逐轮 JSONL、epoch summary、checkpoint。
- 验收：路由无缺失/重复，真实图块完整加载；至少同时出现正负纠错步骤；loss 与梯度有限；非冲突点击按符号满足阈值；checkpoint 能读取；不得把旧 checkpoint 对新空间 val 的潜在暴露当成未见区域泛化证据。
- 评价边界：旧 v4 checkpoint 曾使用历史训练划分，新空间 val 可能含其见过的图块。此次仅为开发性路径验证；后续正式评价需审计初始化暴露，必要时按新空间 train 重新训练基线。
- 首次结果：`smoke_0923_154545/`，4 train/2 val 图块、各最多 2 轮；train/val loss 0.9588/0.8183，正负各 4/2 步；region 硬投影造成 train 5/8、val 1/4 的最终点击冲突。梯度和 checkpoint 正常，但该结构未通过点击门禁。
- 修正后结果：加局部像素残差后在新时间戳目录 `smoke_0923_154843/` 复跑，同样 4 train/2 val；train/val loss 0.7962/0.8065，正负各 4/2 步，最终像素点击冲突 0。仍需大样本检查 region 硬投影的全局副作用，见后续配对训练记录。
