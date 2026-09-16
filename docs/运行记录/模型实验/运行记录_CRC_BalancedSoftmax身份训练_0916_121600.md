# CRC Balanced Softmax 身份训练

## 目的

在严格逆频率、DRW与强制等量细胞采样均无法稳定识别Treg后，独立检验 Balanced Softmax 是否能通过训练期的类别先验修正改善长尾决策边界，而不复制少类细胞或改变原始样本频率。

## 设计

- 对每个训练细胞的 identity logits 加入 `log(train_class_count)` 后计算交叉熵；五类 counts 仅从 train split 精确医生审核标签统计。
- **仅训练 loss**使用该项；checkpoint推理、完整图块验证与概率输出均使用普通 logits，不输入val标签或病例先验。
- 与类别权重、DRW、patch sampling和细胞级均衡采样互斥；固定相同 `reg64`、数据、seed、pool内部划分、GPU1、30 epoch/patience5。
- 输出：`identity_0916_121700/`；日志：`crc_balanced_softmax_0916_121700.log`。

## 验收

先完成1 epoch真实smoke，再训练正式候选并以独立val报告五类F1、precision/recall、每病例macro-F1。超过DRW的0.221140只是必要条件；若Treg仍为零，或仅以大量假阳性换取recall，则候选拒绝。该指标不是患者独立泛化或最终组织分割性能。

## 结果

真实1 epoch smoke `identity_0916_121700/`通过；正式训练`identity_0916_121800/`完成12个epoch，最佳为epoch 6。独立val重评估`evaluate_val_0916_122200/`完成，patient macro-F1=`0.174523`，远低于DRW的`0.221140`。

Balanced Softmax 仍发生少类过补偿：2019 Treg recall=0.6585但precision=0.0114（F1=0.0223）；2017 Treg仍为0，且2017 Macrophage F1仅0.0547。该候选拒绝，不进入fusion。结合已完成的权重、DRW、强制均衡采样和Balanced Softmax，对当前纯`reg64`分类头继续调整类别平衡预计价值很低；下一优先项应是把先前已验证可行的细胞空间邻域/组织上下文扩展到当前1,553图块，而非继续堆叠长尾loss。
