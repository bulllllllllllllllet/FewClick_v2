# 三seed重复的任务顺序合同

原seed20261025已有G3完整结果；新增20261026/20261027，四主方法SAM-FT、WSI-SAM-FT、M3D无细胞、M3D XCell。

`prepare.py` 对原40k recipe和已实现点击JSONL同步作双射排列，不修改任何任务或点击。每seed单独tasks/realized SHA；验证文件、catalog、空间split、预算/阈值/开发选模策略逐字保留。同seed四方法共用同一顺序；跨seed改变顺序以及可变化的新头/dropout RNG。公共权重相同，Dense内部固定初始化仍按冻结实现，不能把这些当全部参数独立随机初始化。

`gate_cpu.py` 已在两个seed实际CRC/TIGER任务通过数据、模型、loss、非零梯度与head更新、optimizer/model checkpoint重载和评分。该门控不是新增模型的精度或原生GPU门控。

`queue.py` 在原GPU队列之后，逐方法/seed执行原训练器4更新原生GPU门控（fresh model/optimizer/RNG resume），再从新初始化完整10k×4训练并评分6040行。GPU1/3/5等对应G6全部结束；GPU4还等外部细胞与外部适配评分完成，且实际空闲。两个新增seed共产生8组正式训练，不能把coordinator排队状态称为这些训练已完成。

每seed任务数与坐标多重集相同，顺序SHA不同；不能继续把原fe8311f9...作为新顺序SHA。基协议SHA、原坐标SHA、新排列、任务多重集SHA、gate坐标SHA与loader验收均归档。主统计须报告全部三个seed及均值/标准差，不选最佳seed。来源/空间bootstrap仅表示任务抽样变化，不能替代训练重复。

与原G3约定一样，各自预训练、视野/细胞条件、参数量不同；这是新增目标监督与预算匹配，不能称等总预训练的架构证明。当前是开发集评价，不是独立病例盲测。源文件已由运行队列contract锁定，新增独立报告可放新文件，不能改已锁定训练/模型/采样文件。

`report.py --gate`真实核验原始四个完整训练/选模/6040评分路径，并重算两个新seed的40k双射排列与点击多重集。`--wait-for-coordinator`观察原coordinator真实PID，等待完整24阶段及日志EXIT0后检查所有12次正式训练；不会由观察超时重启训练。最终报告三个seed的全部Dice/BF1、均值及样本标准差（ddof1），以及同seed配对差。它核验实际GT分母、计数代数与10次302对开发选模，不声称重新推理所有未保存的完整mask。新脚本与门控源码SHA一致后才能排全量报告，运行期间保持冻结。
