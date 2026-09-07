# Phikon LoRA 双视野 RNA Gate

在 256px 与 384px H&E 双视野上，冻结 Phikon 前 22 个 block，仅对最后 2 个 block 的 attention `query`、`value` 线性层插入 rank-8 LoRA；同时训练双视野 adapter 和 cell/RNA heads。三臂仍为 label-only、paired RNA teacher、shuffled RNA teacher。测试时严格只给两路 H&E crop。

这是单 section 的真实 image end-to-end LoRA 小 gate，默认 8 epoch、单 seed，仅用于确认 encoder 微调是否值得扩大；不会替换 v4 或作为正式结论。

输出：`/nfs-medical3/zyh/fewclick_v2/lora_gate/phikon_lora_dualview_0907_132238/`。

## 完成结果

前三次启动分别暴露了 `src` 相对路径、LoRA 参数设备/FP16 dtype、以及 FP16 优化后非有限 CLS 的工程问题；未产生可用训练指标。最终稳定运行使用 BF16 backbone、rank-8 Q/V LoRA、`5e-5` 学习率和相同 8 epoch，完成四个空间 fold 与三臂评估，exit code 为 0。

| Arm | macro-F1 | balanced accuracy |
| --- | ---: | ---: |
| dual-view H&E + LoRA label-only | **0.2485** | **0.2643** |
| dual-view H&E + LoRA paired RNA teacher | 0.2453 | 0.2419 |
| dual-view H&E + LoRA shuffled RNA teacher | 0.2249 | 0.2276 |

LoRA 端到端路径已通过：真实 H&E crop、最后两层 Q/V LoRA、双视野 adapter 和 H&E-only test 均正常运行。paired RNA 明显优于 shuffled RNA，但未超过 label-only，因此在这一单 section gate 中不能声称 RNA teacher 带来额外 H&E-only 收益。下一步应保留双视野 LoRA 的 label-only 配置作为多 section 候选，并在更多 section/seed 上复核后再考虑 RNA loss。

最终 artifact：`/nfs-medical3/zyh/fewclick_v2/lora_gate/phikon_lora_dualview_0907_133329/`；日志：`/nfs-medical3/zyh/fewclick_v2/lora_gate/logs/phikon_lora_dualview_0907_133329/run.log`。
