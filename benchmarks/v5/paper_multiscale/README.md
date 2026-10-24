# 三物理视野 FewClick 的重新对比

本目录实现用户2026-10-25要求的新实验，使用当前数据重新设计，不复现原文旧分数。源起点为`2d82a3d75fa8a5f8993fc056be2647c0c5490a6d`；v4模块按冻结依赖导入，没有修改v4。

模型对应论文的核心计算：256/512/1024µm同中心视野（512²输入约0.5/1/2MPP）；各尺度独立encoder初始化自同一J5；每尺度图像推断的细胞attention和count；物理重叠top4确定稀疏边；自下而上attention；零门控fine到parent及ancestor的attention；三个尺度节点在同尺度KNN及跨尺度边上共同图解码；fine assignment将概率投影回像素。提示从像素处的soft assignment读取，完整正负提示参与cross attention。最终图节点使用直接二元head。

实现选择：同尺度空间KNN=6、top4软重叠父节点、同一cell fusion参数跨尺度共享、带符号embedding的共享Set Encoder、两个图Transformer block。这些是公式未完全规定处的显式选择，不声称精确重放原始投稿实现。原始J5只作为所有对照共同初始化；旧编码器对部分CRC开发区域的历史暴露仍须披露。

四臂为`single_region`、`multi_region`、`single_dense`、`multi_dense`。所有臂共享数据清单、支持查询任务、点数和更新预算；Dense以零像素残差初始化，不加载旧Dense权重。正式验证在新配对清单上，不能与旧0.74/0.82直接相减。当前任务是patch支持—查询，不称完整WSI验收。

大视野取真实源图，不将同图缩放伪装成大视野。不完整视野排除；训练/验证最大视野不重叠。该筛选使IGNITE暂不满足主验证条件，正式主对照只含CRC和TIGER。模型forward不接收组织类别、查询GT或数据域ID；模拟支持点和损失可使用GT。

入口：`data`建立新清单；`prepare`从native上下文准备RGB及细胞并逐tile落盘；`train`训练、验证、存盘、重载与恢复；`pipeline`管理准备分片及四臂任务。

运行记录：`docs/运行记录/模型实验/运行记录_论文三视野模型重新对比_1025_033600.md`。
