# 迭代 Decoder 符号单调与对向提示保护 Probe

- 时间：2026-09-23 16:07:24 CST。状态：完成，`EXIT_CODE=0`；`no_eligible_checkpoint`。
- 问题：`train_0923_155945` 的 256/64 开发性训练 `EXIT_CODE=0`，但全部 epoch 负点后正确正区平均下降约 0.021–0.022、Dice 下降约 0.0023–0.0029，`status=no_eligible_checkpoint`。原有自由符号的图 delta 可越过对向提示保护。
- 改动：region logits 仅按新增点击符号单调更新；正点只能增加、负点只能减少。修正幅度由图上下文 head 预测，并乘以“新增点击视觉相似度超过对向提示相似度”的非负证据及上一轮概率门控。视觉相似度基于原始视觉 token，不含 sign/坐标 embedding。像素层保留局部点击锚点。
- 输入：同 `运行记录_迭代Decoder空间划分扩大训练_0923_155945.md` 的冻结 J5、v5 空间划分和路由。
- 输出：`/nfs-medical3/zyh/fewclick_v2/iterative_decoder/probe_0923_160724/`；日志 `probe_0923_160724.log`。
- 计算：GPU 1，64 train/24 val、2 epochs、每例最多 2 轮、batch 1；tmux `iterative_decoder_0923_160724`。
- 监控：逐 10 图块日志、epoch 指标、质量门禁、checkpoint、`EXIT_CODE`。
- 验收：两符号的远端变化方向正确，像素冲突为零；尤其负点后已正确正区下降不超过 0.01，Dice 下降不超过 0.001。通过后才扩大预算。
- 评价边界：冻结 v4 初始化对空间 val 有历史暴露；仅用于开发性架构筛选。
- 追加诊断（2026-09-23 16:09:19）：用 `evaluate_iterative` 重载 epoch 0 checkpoint，在相同 24 个 val episode 上把“原正确区域变化”拆成 region 更新与像素点击锚点贡献。GPU 4；输出 `/nfs-medical3/zyh/fewclick_v2/iterative_decoder/anchor_audit_0923_160919/`；验收是两项贡献之和与总变化一致，明确定位损伤来源。
- 诊断结果：负点后原正确正区平均变化 −0.0224415 = region 更新 −0.0223358 + 像素锚点 −0.0001057；正点后原正确负区 +0.0031857 = region 更新 +0.0032174 + 锚点 −0.0000317。checkpoint 重载评估完整通过。损伤几乎全部来自 region 更新，不能归因于 2px 点击锚点。
- 追加证据分布诊断（2026-09-23 16:10:33）：同一 epoch 0 checkpoint 与 24 val episode，输出 `/nfs-medical3/zyh/fewclick_v2/iterative_decoder/evidence_audit_0923_161033/`；记录错误区域与原正确区域的新增点击相对对向提示相似度差，判断是否存在可用的阈值分离。GPU 4，输出逐轮 JSONL 和 summary。
- Probe 结果：epoch 0/1 的 val 正点 Dice +0.001988/+0.001934，负点 Dice −0.002570/−0.002575；负点远端 FP 变化 −0.05244/−0.05245，但原正确正区仍 −0.02244/−0.02245。像素点击冲突均 0，质量门禁均不通过。符号单调本身不能保护与负点视觉相似的真实目标。
- 证据分布：新增负点相对对向提示的非负视觉差，在错误区域平均 0.14192、原正确正区平均 0.07137；正点为 0.08137/0.03153。存在一定分离但大量重叠，简单线性门控过宽。下一步试平方证据门控并提高正确区域保持损失，若仍无合格点则不把开发性模型用于交互推理。
