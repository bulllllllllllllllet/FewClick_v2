# 远端迁移验收与统计

本模块只读取已结束的固定G4输出，不改推理、模型或采样源码。

```bash
CUDA_VISIBLE_DEVICES= /home/zhaoyh/miniconda3/envs/aligner/bin/python \
  -m benchmarks.v5.transfer_reporting.report --root TRANSFER_ROOT --output NEW_TIMESTAMPED_OUTPUT
CUDA_VISIBLE_DEVICES= /home/zhaoyh/miniconda3/envs/aligner/bin/python \
  -m benchmarks.v5.transfer_reporting.acceptance --root TRANSFER_ROOT \
  --report NEW_TIMESTAMPED_OUTPUT --output NEW_TIMESTAMPED_ACCEPTANCE
```

`report`逐行核对九组18700行稀疏结果、一组837对完整源标注参考，实际标签SHA/分母/边界、全体支持点与L合同、最大视野隔离、6040个源预测mask缓存身份及SAM候选选择规则。完整源标注参考不复制点击曲线，仅在同837对上比较。原始全量预测未保存，不能据计数审计声称重算全部预测mask。

五组点先对pair平均；按域、same/cross、有/无目标、同片固定距离区间分层。共享2000次按域分层的query WSI bootstrap只作固定支持任务/权重下的探索性区间，不覆盖支持共享相关性、病例总体或训练seed。支持查询WSI依赖图只有两个连通分量；不声称935独立样本。FPA仅限valid像素，unknown不视作认证阴性。

`acceptance`绑定G3原始40k实际消费/开发选模checkpoint，回放20张原生MultiverSeg门控mask，并验证160张清洁迁移门控mask已经重算。来源更正的派生汇总保留原报告SHA，XCell采用用户确认的内部数据来源。历史M3同时具有J5和混合local/transfer增量训练，不可与清洁本图适配行作等R训练架构比较。

实际记录：`docs/运行记录/模型实验/运行记录_迁移全量验收与分标注轨道统计_1025_224400.md`。本阶段完成不替代G5重复、G6机制、G7概念/TLS、G8整片效率或G9审稿闭环。
