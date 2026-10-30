# P1I 共享提示decoder及真实门控

时间：2026-10-31T01:48:32.514165+08:00。状态：三臂工程门控complete且独立接受；P1I整体preparing，正式1000更新对照未执行。当前无活跃worker。P1C单seed细胞结果保留，LEMON仅暂定候选。

## 固定设计与真实输入

代码`benchmarks/v5/encoder_research/`，新v5实现，v4只读。冻结COCO-ResNet、Phikon-v2、public SAM image encoder；三者fine512同物理范围，无额外组织尺度或细胞输入。公共归一化/空间编码分辨率按模型记账：ResNet512、Phikon512不resize/不center crop、SAM512→1024（额外编码分辨率披露）；分别得到2048/1024/256通道，统一32grid后learned1x1投影至128。

公共head相同300449参数：signed正负支持原型、query空间特征/原型交互/余弦相似度、本图点heatmap、公共RGB细边界支路。远端query heat为零，支持原型仅来自support图；forward没有GT/类ID/目标present标志。encoder始终eval/frozen，不加载原SAM decoder或旧J5 target heads。adapter容量不同必须报告，不称等容量。公共层先初始化、adapter后初始化，seed固定20261031，独立重算三臂公共初始化SHA完全一致。

AdamW lr.001/WD.0001，valid-pixel BCE+softDice；正式pilot固定1000更新×4个相同冻结任务，固定末步，local302与remote另表；质量评估门槛沿路线。完整contract及源码SHA：`/nfs-medical3/zyh/fewclick_v2/cvpr_research/image_decoder_gate_bilinear_1031_014441/contract.json`。原权重身份链接P0 tissue contract的image_encoder_weight_identities.json与encoder_inventory_1030_182424/local_weight_identities.json，受控臂不继承旧目标数据训练。

## 工程故障与等价修复

013254初次ResNet worker1008107真实4次更新和模型重载成功，但继续一步逐位状态不同，exit1；队列停止，未运行后两模型。014114严格deterministic复核worker1021398明确报grid_sampler_2d_backward_cuda没有确定性实现，exit1。两次故障产物/log/checkpoint完整保留，不作为质量试验，不降低接受阈值。

修复：使用固定双线性权重/GEMM实现等价point采样均值和bilinear align_corners=False上采样；保留数学定义、模型参数、数据/损失/阈值。与原PyTorch算子双精度前向最大误差≤4.45e-16、反向≤2.85e-14，见layer_equivalence.json。严格deterministic、cuDNN deterministic、禁TF32、CUBLAS_WORKSPACE_CONFIG=:4096:8全部启用；仍要求逐位继续一步相等。

## 已接受的真实门控

命令：aligner/python -m benchmarks.v5.encoder_research.gate_shared_decoder --arm {resnet,phikon,sam} --output /nfs-medical3/zyh/fewclick_v2/cvpr_research/image_decoder_gate_bilinear_1031_014441/{arm}，实际解释器/home/zhaoyh/miniconda3/envs/aligner/bin/python；按launch.sh单GPU物理3串行，Phikon已有v5 overlay，CPU2，全部exit0。

|encoder|总head参数|其中adapter|实际worker|秒|
|---|---:|---:|---:|---:|
|resnet|562977|262528|1038968|2.44|
|phikon|431905|131456|1039658|5.27|
|sam|333601|33152|1040357|2.97|

同两CRC/两TIGER实际train任务、4更新；finite loss/全部梯度/22个state tensor变化、encoder无梯度、真实512 mask；保存模型/optimizer并新建重载输出完全一致，继续一步所有参数逐位相同。source-only remote输入也实际通过。独立CPU重算公共初始化SHA、磁盘head结构/参数量/finite mask以及每tensor继续误差0；接受文件independent_acceptance.json。

这里的4步loss仅证明可训练性，不是validation Dice或encoder优劣。门控耗时不是完整训练或WSI效率结论。

## 下一步

按固定任务需要提取/缓存三臂冻结空间特征，保持图像/权重/清单/源码SHA；先实际小批后全范围，并保存各缓存接受。然后相同seed、相同公共head初始化和1000x4任务训练，梯度/实际消费/重载/逐任务评分独立接受后才比较local/remote。未运行正式实验，不把门控标成P1I完成。选定image底座后，验证LEMON连续细胞下游mask增益，再决定P2。


## 2026-10-31T02:05:43.215261+08:00 正式pilot接续

三臂正式缓存/训练/评分已启动，接续[docs/运行记录/模型实验/运行记录_CVPR_P1I正式缓存训练与评分_1031_015107.md](/home/zhaoyh/FewClick_v2/docs/运行记录/模型实验/运行记录_CVPR_P1I正式缓存训练与评分_1031_015107.md)。当前ResNet cache已执行完成，独立接受与正式训练依赖真实阶段；尚未产生正式质量结论。
