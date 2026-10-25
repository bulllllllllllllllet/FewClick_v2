# 普通H&E细胞特征控制

`gate.py`保存冻结公共CTransPath的768维输出，捕获位置在XCell前。XCell仍运行一次作为已有缓存回放参考，其输出不进入raw768特征，也不作为控制分支训练信号。

```bash
CUDA_VISIBLE_DEVICES=3 /home/zhaoyh/miniconda3/envs/aligner/bin/python \
  -m benchmarks.v5.generic_cell_control.gate --catalog FIXED_CATALOG \
  --output NEW_TIMESTAMPED_OUTPUT
```

实际两训练样本CRC/TIGER×fine/2x/4x六项输入已通过：fine复用原核实例，context重做原InstanSeg；细胞位置/面积/未知类型、数量和顺序一致，原XCell74维参考逐值误差0。保存raw768、geometry10及实例，独立磁盘重放和核计数也通过。组织GT路径禁用，输入特征提取不读GT。

这是原生输入门控，不是完整1737视野缓存、64维变换拟合、受控训练或精度实验。后续转换若采用PCA/标准化，只能在训练细胞拟合，不改变模型74维细胞接口，且不得按验证分数挑变换。普通H&E外观控制不等于同数据同网络预训练目标的因果消融。

记录：`docs/运行记录/特征缓存/运行记录_通用HE细胞特征原生输入门控_1025_231000.md`。
