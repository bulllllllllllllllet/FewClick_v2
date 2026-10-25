# 细胞、层级与容量的重训控制

本模块实现goal.md的G6，保持G3/G4运行中的源码与任务不变。

用户已确认：XCell使用内部数据集训练，当前CRC/TIGER未参与XCell训练。上游CTransPath的公开TCGA/PAIP图像预训练单独记账，不与XCell训练数据混同。旧“未核定”是归档证据缺口，按本次确认更新，不作为发现重叠的证据。无细胞/形态控制检验特征贡献，不能直接分离分子监督因果效应。

| 重训行 | 信息/结构 | 主要对照 |
|---|---|---|
| m3_none | 三视野、区域解码，无细胞分支 | 无细胞控制 |
| m3_morphology | 相同检测位置/面积/unknown身份/总计数，64维XCell输入置零 | 区分几何计数与XCell表征 |
| m3_xcell | 三视野、区域解码、完整74维细胞输入 | 与前两组同预算重训 |
| m3_shared_encoder | 三个真实视野共享同一encoder参数 | 对照参数容量；计算仍多于单视野 |
| m1_xcell | 仅fine视野与相同细胞合同 | 单视野容量/信息基线 |
| m3_no_hierarchy | 移除pre-graph上下行层级注意力，保留物理cross-scale图边 | 只检验该层级模块，不等于移除全部跨尺度信息 |
| m3d_morphology | Dense三视野，形态/计数输入 | 与G3两组Dense none/xcell形成确认 |

全部从generic COCO backbone和新头开始，不载入J5。固定相同40k实际点击JSONL、10k更新、前1k头部预热、每1k在相同302对/3+2点组0选模，再评1510点组×4预算。损失/LR/优化器/BN合同复用已验证trainer；XCell输入变化后必须完整重训，不以推理清零结果替代训练消融。morphology仅在输入张量副本上将列10:74置零，不覆盖任何缓存。

shared_encoder确实将三个ModuleList位置绑定同一个encoder对象，optimizer按parameter identity去重。参数总量与实际可训练量分开记录；保留模型内未使用的semantic模块不应解释为有效容量收益。区域解码不使用semantic输出且冻结该分支；Dense确认组按G3合同将fine semantic通道作为二值监督latent特征训练，因此区域/Dense差异不应单独归因为小型像素残差头。

CPU门控对每行执行4个真实CRC/TIGER训练任务、前2步预热后2步联合；磁盘保存后销毁模型/optimizer、新建后恢复再继续。验证head/backbone实际参数变化、共享optimizer无重复、输入过滤保持几何，以及none/morphology对被排除表征扰动的预测不变性。CPU门控不是正式GPU收敛结论。

`train.py`和`evaluate.py`复用冻结G3训练/评分代码，仅在当前子进程绑定此处显式builder。builder的来源SHA包含在checkpoint contract。正式队列必须再通过4次原生GPU更新和fresh恢复门控；随后重新从generic权重初始化正式训练，禁止从短门控模型接着跑。

`benchmarks.v5.experiment_orchestration.mechanism_queue`等待G3各GPU的训练及评分均完成，且实际GPU无compute进程后才开始。GPU1排m1/no_hierarchy，GPU3排Dense morphology，GPU4排none/morphology，GPU5排xcell/shared；不使用GPU0/2。每行依次native gate→10k训练→6040评分，异常保留、不重试或替代算法。最终并入G3两条Dense对照。

G6完整结论还需所有正式训练/评分通过。G5独立seed、G7留出/TLS、G8 WSI/效率和G9审稿证据仍单独验收，不因消融路径可运行而完成。
