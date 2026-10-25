# 整片物理网格与缓存路径

`plan.py`对既定TLS30清单读取真实WSI header，构造fine256µm与精确2x/4x整数源框的全有效网格，不用GT/预测/tissue过滤选择query。为获得真实4x上下文，外缘fine像素显式不预测；末行/列补最后有效起点，允许重叠，不填白/复制上下文。annotation_json内容不打开，原清单的标注数量/历史提示索引仅原样保存在private provenance，不参与网格生成。TLS30与主队列30不同。

`audit_plan.py`独立核对全部网格文件SHA、每tile三视野边界与中心、Cartesian乘积完整性、矩形并集面积和未覆盖边缘。实际30片/139934 tiles通过，覆盖各片矩形面积88.916%–93.622%，不是组织覆盖率或全片模型完成。

`cache_gate.py`绑定已验收G3无细胞模型的完整40k训练与seen-selected权重，固定CRC支持任务，实际读取预定WSI小区域；将三视野encode结果写盘/读回，切换1+1/3+2 task，检查无query编码及query提示图0，再与8次完整原生forward逐值比较。NoCell忽略的占位cell张量不表示已执行细胞检测。图仍是每个三视野窗口192节点，不是跨整片全局图。

首个边角gate几乎全白，只验收背景路径；旧源码保存在background_source_1026_011800。随后固定中央grid位置（不读GT/分数）组织gate通过，fine白底约1.4%–4.4%，8个native errors均0。两预算概率先平均再>=.5阈值，错权重cache拒绝。`audit_cache.py`fresh模型/磁盘cache重放，以独立NaN层聚合复算全部2399401像素/预算，并检查保存mask与错input cache拒绝。

这些是CPU4-query机械门控，支持任务不是TLS标签。不是整片精度/交互延迟、完整细胞系统效率、TLS概念泛化或全30执行。TLS穷尽标注/确认阴性未解决，不将未标注组织当负类。后续仍需整片流式执行/落盘/覆盖、细胞准备、native GPU门控与冷启动/编码/缓存IO/首点/后续点/整片写盘的分项效率。所有原始产物保留，状态与SHA/PID见goal/registry及模型实验记录1026_010800。
