# 留出概念支持查询迁移

独立于local表。`evaluate.py`绑定同fold masked训练协议、实际40k消费、seen-best权重、原GT local终评禁选模合同及冻结remote任务。推理对象禁用GT API并移除标签/计数，模型仅接收RGB/可用特征与支持点；query人为点0、class ID不输入、远端热图0。scorer在预测后才访问query GT，absence/FPA限已标valid区。

FewClick输出graphDense和同特征prototype诊断；SAM输出encoder-feature prototype；WSI-SAM输出fine+真实2x中央对齐、各视野归一化拼接的prototype。后两者是自定义冻结特征检索包装，不是官方native remote decoder。温度.1、margin0、阈值.5固定，不增加R监督更新。不同公共先验/视野/容量仍披露。

N/A/L实际115/65/62 pairs，5组点×4预算，每输出2300/1300/1240行；合同里继承父任务的pair_coverage/unique_fields不能作本子集分母，run重新计数。N只有7个域/类型strata，无TIGER same_present；native gate28行/输出，A/L16行。正查询Dice与阴性FPA分列。

实际NoCell两更新CPU checkpoint仅门控接口：7 native/cache errors0，两输出各28保存概率/GT计数独立复算一致，GT API禁用/query maps0。SAM/WSI helper用真实G3权重验证CPU编码/cache概率和2x对齐；这些权重训练见过留出类别，不算未见类结果。`--checkpoint-gate`只允许与`--gate`合用，全量正式评分不接受它。

`queue.py`先真实prepare/独立核对9个条件，正式CPU coordinator433592逐条件等local producer349631完成train，再用该正式权重gate→全量remote score，最多2worker、CUDA隐藏，不等所有local完成。现为真实等待，无正式训练或R精度。源contract捕获的当前文件冻结，新工作放snapshot外。source/weight/task gate不一致即停，不重试；独立条件保留。local/G7/overall不能仅因队列启动称完成，最终仍需完整独立统计/像素/选模验收。记录1026_013700。
