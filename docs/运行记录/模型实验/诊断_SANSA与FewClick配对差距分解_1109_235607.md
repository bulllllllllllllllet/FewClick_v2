# SANSA/FewClick配对差距分解

用户要求拆分SANSA优势来源。只读现有同片11000预测，三方法FewClick/适配SANSA/SAM2encoder+FewClick，完整身份与rowsSHA核验。不新增训练，不用GT选阈值或模型。按目标面积、类别拆Dice/precision/recall/BF1/空预测/预测面积比与absentFPA，1000次按queryWSI配对聚类bootstrap。诊断观察不是模块因果消融，跨方法encoder/adapter/memory/decoder同时不同；依据预测面积选择的诊断子集不作主表。目标不存在时另表，空预测precision记0。后续有必要时才能设计单变量干预，不因系统排名推断具体模块强弱。

代码benchmarks/v5/sansa_gap_diagnostic/analyze.py，aligner CPU，输出/nfs-medical3/zyh/fewclick_v2/cvpr_research/sansa_gap_diagnostic_1109_235607。验收读取三方法2分片共33000实际行、SHA、共享GT计数和完整身份，保存report+acceptance。当前preparing。

2026-11-09T23:57:09.367106+08:00 只读诊断complete_accepted：三方法33000row完整身份和6rowsSHA通过，共享GT正像素/有效计数一致。SANSA-Few类别宏Dice差4.9451pp，75片配对cluster bootstrap95%CI[3.9395,5.9958]pp，仅开发评估不作独立确认。原Few在7334present中1569空预测，大面积2130中254空预测；按原classmacro分母加法分解，空预测子集贡献3.2798pp/总4.9451pp(66.3%)，大面积贡献4.6957pp/总6.3516pp(73.9%)。此为结果误差分解，不能把空预测原因直接归于某模块。大面积Recall Few58.4888/SANSA66.4990%、precision70.5566/74.9196%、BF1物理2µm26.1412/39.8533%。预测面积/GT比.7439/.9117，说明原模型平均欠覆盖。两者预测面积均处于GT的.5–2倍诊断子集1269例：FewDice82.5010/SANSA80.6441%，BF135.4120/45.7671%；该子集按预测/GT选择，不能代替主结果或宣称整体反超。原Few误报2.3912%仍优SANSA9.5687%。支持阶段/跨图匹配/像素投影原因尚未因果分离；推荐下一步固定病例的支持分割及query区域表示oracle审计，不自动训练。证据/nfs-medical3/zyh/fewclick_v2/cvpr_research/sansa_gap_diagnostic_1109_235607/report.json与delta_decomposition.json。
