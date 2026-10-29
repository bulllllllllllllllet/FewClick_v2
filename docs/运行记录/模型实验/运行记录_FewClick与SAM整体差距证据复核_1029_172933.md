# FewClick 与 SAM 整体差距证据复核

时间：2026-10-29T17:29:33.625536+08:00。状态：complete_accepted_readonly_diagnosis_qualified。

用户要求：整体研究为什么 FewClick 比 SAM 差并给出结论。本次仅分析原始已接受计数、实际训练任务、学习曲线和实现；不训练、不变更权重/阈值/任务、不占 GPU。

输入：G3 原 6040 行/方法、10×302 开发选模行、40000 实际训练任务和 catalog；G5 三 seed、G6 消融、G7 外部、G4 迁移及已接受 Dense 失败小试。原局部诊断与公平性合同一并引用。

命令：aligner Python 执行 benchmarks.v5.fair_comparison.overall_gap_review；CPU 单进程，数值线程上限2。输出：`/nfs-medical3/zyh/fewclick_v2/fair_comparison/overall_gap_review_1029_172933`。

接受条件：重新校验输入 SHA、逐行 TP/FP/FN/GT/点击一致性、完整302对/5组/4预算，train/val标签面积分层与独立复算一致；不将事后分层、损失曲线或结构推断表述成因果证明。记录进程PID/命令/日志/返回码，复核报告后更新连续性文件。

分析范围：预训练与优化自由度；区域汇聚与 Dense 条件化；小目标及类/域混杂；点击预算响应；训练面积分布、学习曲线、全局梯度裁剪；细胞/多尺度增益；已经排查的错误和仍不能确定的原因。

## 完成与结论

本次整体分析完成且独立复算接受，没有训练或换权重。四方法各6040行/四预算、10次选模302行、实际40000任务/消费SHA复核；统计4189886 EXIT0，缓存阶段6018 EXIT0，独立16055 EXIT0，64个阶段mask计数与聚合误差≤1.11e-16。

新结论：小目标<5%在实际训练占32.85%，开发占39.74%；存在小目标的14个域-类分组SAM均领先，不是单一脂肪问题。主首seed小目标SAM/FewClick Dice51.21/29.09%，FewClick小目标precision25.00/recall74.00%；贡献总差距68.17%是算术描述而非因果比例。5+1 SAM/FewClick71.78/57.30%，多点未缩小整体差距。注册参数Few126.03M/SAM93.74M，不能仅归因于SAM参数更多；注册参数不等于可训练/有效容量。

复核固定32极端面积诊断案例：同一Dense训练模型base→Dense总体46.64→55.34%，24/32改善；14小目标9.26→15.88%，11/14改善。base是中间输出而非另训region-only。因此Dense多数在补救上游，不能由此前脂肪单例认定主要是Dense搞坏。既有完整消融Dense收益保留，困难负Dense小试无效不采用。

最符合证据的机制假设是软区域汇聚/点击表示及图预测对小目标与相似非目标区分不足，加上SAM完整提示分割预训练/更广可训练层的适配差异；尚无单因素因果分解。实际Dense保留64×64深特征/128×128浅层/点击热图，64region不是硬性能上限。已有负类梯度、训练小目标、非零backbone梯度均确认；无证据以漏写loss/脂肪未训练/独有裁剪解释，train/val代表性差距与延长训练仍未知。报告区分局部、外部与支持查询迁移，不预告主15WSI反超。

主15目标与原后台工作不变；本次不更新未经现场核验的细胞计数/PID活性，不把只读分析当WSI实验接受。native usageLimited未改。后续质量修复应先查区域/点击表示与代表性train/val误差，不继续盲扫负loss或验证阈值。

完整报告：`/nfs-medical3/zyh/fewclick_v2/fair_comparison/overall_gap_review_1029_172933/report_zh.md`；独立接受：`/nfs-medical3/zyh/fewclick_v2/fair_comparison/overall_gap_review_1029_172933/independent_acceptance.json`。执行命令与真实PID/EXIT_CODE见三个execution.json及日志；统计源与模型源SHA绑定于接受文件。
