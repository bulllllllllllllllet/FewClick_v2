# Panel1轻量MLP与门控多尺度Probe

- 问题：验证非线性分类头和局部主导的上下文融合是否超过当前线性probe。
- 输入：原Panel1固定训练/val/test清单与已缓存128/256 CTransPath特征；补充均衡评估清单的双视野特征。图像编码器冻结，不使用补充评估集训练。
- 结构：`128_mlp`为768→256→5的LayerNorm/GELU/Dropout MLP；`gated_128_256_mlp`将128分支映射256维、256分支映射128维，并加入局部分支前128维与上下文分支的乘积，再经256→5分类头。StandardScaler只在原train拟合，类别平衡交叉熵，3个随机种子，按原val macro-F1选择。
- 验收：原val macro-F1与均衡补充test macro-F1均报告；比较当前128 Logistic baseline（原val macro-F1 0.389、补充test macro-F1 0.393），并检查训练/评估清单hash。
- 输出：`/nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_mlp_probe_0923_164800/`。首次运行因未限制PyTorch CPU线程主动停止；第二次训练权重已生成，但评估路径误指向scores目录而退出；现用实际补充清单目录重跑。
- 状态：完成，最终重跑`EXIT_CODE=0`。原val上128-MLP准确率0.736、macro-F1 0.454；门控多尺度准确率0.733、macro-F1 0.457。均衡补充test上128-MLP准确率0.364、macro-F1 0.318；门控多尺度准确率0.356、macro-F1 0.292。当前线性128基线补充test为0.408/0.393，故MLP和门控结构均未通过替换门槛，表现出单病例验证集过拟合或分布敏感。
