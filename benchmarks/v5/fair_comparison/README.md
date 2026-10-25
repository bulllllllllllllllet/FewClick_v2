# 补充实验的公平性合同

连续任务入口是根目录 `goal.md`；阶段证据写入其链接的 JSON 进度表。禁止用启动成功代替实验完成。

## 当前比较可以支持什么

已完成四臂共享历史 J5 和相同新增任务、更新预算及验证序列，是增量训练匹配的系统消融。三编码器与额外视野同时变化，不能独立归因于层级设计。旧 J5 的历史目标数据监督不会因换一个训练清单而消失。当前是同队列空间验证，保留病例来源但不要求病例互斥。

| 行 | 初始化/额外信息 | 新增目标训练 | 可支持的结论 |
|---|---|---|---|
| 历史四臂 | J5 目标监督、XCell；M3 额外 2x/4x 视野 | 原 40k 混合 local/transfer 任务 | 同起点增量系统差异 |
| SAM-public | 公共 SAM ViT-B；单视野 | 0 | 相同本图点击下的公共权重参考 |
| WSI-SAM-public | 公共 WSI-SAM/MobileSAM；fine+2x | 0 | 同上，保留预训练/视野差异 |
| SAM-FT / WSI-SAM-FT | 各自公共初始化；来源需记账 | 新固定 local recipe，待执行 | 等新增标签池与适配预算，不能声称全部预训练相同 |
| 受控 FewClick | 通用 COCO backbone，新头；含 image-only 控制 | 同一新 local recipe，待执行 | 排除历史 J5 监督后的比较，仍披露公共先验和容量 |
| 闭集监督参考 | 通用 backbone，CRC12/TIGER7 类头，待门控 | 同标签池全类别监督 | 注释资源一致的闭集参考；多类别信号比单任务二值监督更多 |

XCell 训练来源重叠尚未核定，不能把有 XCell 的行称为无目标域先验。image-only 控制是排除此支路历史监督的控制实验，不改变正式主模型定义。预训练病例/空间重叠尚未确认时写“未核定”，不写“无泄漏”。

## 数据、提示与指标

- 使用已完成缓存的 CRC1200/300、TIGER156/81 train/val。再次审计最大 4x 物理视野跨 split 隔离；同病例残余相关性保留，统计时按来源/空间聚类。
- `protocol.py` 固定本图任务。每个 dataset/target 的 val 按预先固定的 SHA 顺序最多取20个视野，稀有类别全保留，5组独立点击；不看模型结果。背景作正目标排除，背景像素仍作有效负类。少于381个视野被选时明确报告覆盖数。
- local 只使用同一张图的点和单张图监督。transfer 的 support/query 分离，另设任务与表，不能向查询图照搬支持坐标。GT 模拟点必须明确披露。
- 保存最大5正2负，3+2选模；1+1/3+1/5+1共享正点前缀和第一个负点。第0点组用于选模，全部5组是同一开发任务的点击稳健性分析，不是独立测试。
- 新训练 recipe 固定40k图像/目标/正负点数量/随机种子；每步4任务、10k步、每1k步验证。各 trainer 用共享 resolver 从支持 GT 生成点，必须保存实际坐标并核对任务 SHA。仅有 recipe 不能宣称已完成实际训练坐标一致性验证。
- 冻结 `>=0.5` 阈值。SAM 选择预测 IoU 最高的 mask，禁止用 GT 选候选；WSI-SAM 使用单个融合 mask。公开权重不在 val 选择 checkpoint。
- 新 BF1 用双侧4邻域边界、2µm欧氏距离（分别使用 x/y MPP），GT 有边界就纳入，预测无边界计0。GT 无边界统一 NA 并报告数量；Dice 与缺失目标误报分别报告。旧方形膨胀容差的分母修正表独立保存，不能冒充欧氏 BF1。
- SAM 输入 fine512 后原生编码器 resize1024；WSI-SAM 使用与 FewClick 相同的 fine512 和真实2x512缓存，原生桥接各 resize1024。FewClick-M3 为3幅512。信息/编码分辨率不同必须披露，不称等视野或等计算。
- 公开基线脚本当前记录的是含重新编码的整次推理时间，不是缓存后的在线点击延迟。正式效率实验后续分别测准备、编码、点更新与 WSI。

## 执行和验收

`audit.py` 审计四臂任务和预算 SHA、真实训练完成/重载、最大视野重叠，并输出旧 BF1 独立修正版。`test_metrics.py` 验证漏边界计0、各模型相同可评性、欧氏/方形差异、各向异性 MPP、忽略标签边缘。

`public_baselines.py --gate` 在真实 CRC/TIGER 输入、全部四个点击预算上门控。完整评分必须经过门控，失败原样保留，禁止空 mask 静默回退。输出逐任务 rows、SHA、权重/配置/源码、PID、进度、分组分数。

后续 FT 还需真实梯度、多步 loss、参数变化、checkpoint 重载及 resume 门控；正式等监督结论需审计目标域全部历史训练暴露。所有新结论以进度表中的实际证据为准。

## G3已实现的训练合同

`realize_training.py`已把40k recipe一次性转成实际坐标，所有方法只读同一个JSONL；训练完成必须使消费记录SHA与该文件完全一致。每步4任务、10k步，前1k步只训练头，随后联合更新；每1k步在固定302对、第0点组、3+2上选最佳Dice。头LR1e-4、backbone LR1e-5，AdamW/weight_decay1e-4、相同余弦衰减、梯度裁剪1。此为固定预算协议，未做方法专属超参搜索，不能称每种方法的最佳可能效果。

`adapted_models.py`通过原生SAM/WSI-SAM模块构造可微路径，禁止调用其带no_grad的顶层训练。SAM的mask候选始终按预测IoU选择，GT只进入损失及IoU校准；WSI-SAM使用两视野单融合mask。两者都允许联合更新各自image encoder，BN统计固定。FewClick联合更新各encoder layer3/4和新头，BN固定；额外图结构assignment正则独立记账，模型容量和可训练参数量不匹配。

通用FewClick只加载COCO DeepLab公开权重，不调用旧J5加载函数；组织区域/细胞融合/prompt/graph新建，Dense零残差初始化。12个semantic输入通道是generic ASPP＋新投影，作为latent特征受二值mask损失训练，不赋予历史组织语义。image-only通过NoCell忽略全部细胞证据，另有扰动全部细胞输入时预测完全一致的实测门控。此为受控训练变体，不冒充旧论文checkpoint。

闭集参考使用共享generic backbone和CRC12/TIGER7分类头，全类别CE；CRC ID0–11映射0–11，TIGER ID1–7映射0–6，255保持ignore。其预测为类别argmax后抽取指定目标，额外提供dataset和target ID，单独标为闭集参考。当前catalog没有TIGER healthy_glands（ID4）实例，不能声称测了完整七类目标覆盖。

`verify_adapted_gates.py`对真实checkpoint逐张量审计：两种FewClick预热后的三个backbone共954张量必须等于generic COCO，联合更新后每个backbone确实改变；所有方法必须有头部与backbone实际参数变化、磁盘/optimizer/RNG恢复和重新构建模型后的继续训练。门控仅4次更新，不能用门控精度宣称收敛。

`adaptation_pipeline.py`在四张GPU上排队五组训练和最佳checkpoint的全点组/全预算评分。源码/门控/坐标SHA不符则停止；异常写入状态，不重试、不替代算法。`evaluate_checkpoint.py`还支持保留J5系统的同任务重评分，仍标retained-J5。`report_comparison.py`核对完整逐任务身份、点击与GT计数一致，按图像目标对平均点组，并作来源空间聚类配对bootstrap；它不替代独立训练seed重复或盲测。
