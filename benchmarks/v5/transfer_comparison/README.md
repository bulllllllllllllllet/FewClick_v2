# 支持—查询迁移补充实验

本模块推进根目录goal.md的G4；G3正在运行，相关源码保持冻结。迁移结果单独成R表，不与本图交互L表混排。

## 固定任务

复用L表302个支持图像—目标对及全部5组最大5正2负的序列；查询图分别按same/cross WSI与present/absent取固定SHA排序第一项。GT只用于预先定义任务/模拟支持点及最终评分，不用于推理。所有支持/查询来自同一固定空间开发catalog；同WSI的最大4x视野必须互不相交，跨WSI必须已验证且space不同。不具备条件的任务写出excluded，禁止替换协议。

当前935个远端任务对×5点组=4675；有目标2350点组、无目标2325点组。212个支持视野、330个查询视野，605个查询重复任务对；不能当935独立WSI。背景正目标仍排除，GT未知255仍不评分。数据没有TIGER healthy_glands ID4，不能宣称七目标覆盖。

3+2为主，1+1/3+1/5+1取相同支持点前缀。query人工点数0；支持点只读取支持特征，严禁在query端重画热图。阈值0.5不在R集调节。present报告Dice/precision/recall/BF1，absent单列平均FPA及≥1%/≥5%比例；same/cross与数据域分层。

`TransferData.materialize_inputs`不访问任何标签路径、class ID、GT有效性或GT计数；模型字典仅含图像/细胞和支持点。`supervision`是评分端独立API。全体任务另行核验GT有/无目标、点符号、点与L表相等及最大视野隔离。真实测试把支持及查询GT路径设为不可用后，仍可完成图像缓存加载及真实FewClick/SAM特征预测。

## 比较条件

| 方法 | 支持信息 | 权重/可支持结论 |
|---|---|---|
| retained M3 graph | 支持正负点 | 原四臂M3 checkpoint；历史J5监督；系统参考 |
| retained M3 same-feature prototype | 同样的点、同一个encoder/cell/hierarchy特征 | 固定特征检索诊断；不能冒充重训架构消融 |
| SAM-public feature prototype | 同样的点、SAM公共embedding | 自建迁移基线；不是官方SAM原生远端推理 |
| 干净初始化/同域适配后迁移 | 同样支持点 | 待G3选定权重；local适配后的迁移泛化，另记训练经历 |
| SAM-public → MultiverSeg predicted context | 同样支持点先生成完整预测mask | 同人工点预算的两模型系统；生成器/权重/预测maskSHA与成本全部记账 |
| MultiverSeg full-GT reference | 完整支持mask、query无点 | 额外密集标注的参考轨道；含255未知像素的支持图不可进入此轨道 |

原型对支持正/负点读取特征，分别L2归一化并取均值，再与查询特征作余弦相似度差。固定温度0.1、margin0；不使用query GT，不生成GT oracle点。区域原型先按soft assignment采样支持token，再在同一个查询assignment上投影。Dense SAM原型对query embedding分类后双线性上采样512。合成左右互换测试验证按query外观而不是照搬源坐标分类。

M3原型和学习解码器共享完全相同的已训练编码特征。替换解码器是固定权重诊断；不消除历史J5、不同参数量/先验，也不能把M3D相对区域原型差异仅归因于Set Encoder。当前先运行保留M3和SAM公共特征；G3新权重完成后才加入受控迁移行。

## 官方近期上下文方法

MultiverSeg官方源码与[项目](https://multiverseg.csail.mit.edu/)说明其上下文为image–segmentation对。固定[MultiverSeg](https://github.com/halleewong/MultiverSeg) commit `f351a0cdb3d6222b4d9b6055a06aba84f8e8c2b6`、UniverSeg `833a0c34c65e38d675e21bd48ddec6797cc03259`及ScribblePrompt `182c44975f77749b559974ce8db558c8bde57788`。只导入原生非空context分支，检查原生forward输出逐值相同；不构造/调用无context的ScribblePrompt替代分支，也不加载其未用权重。

ICCV2025 v1来自官方download.sh链接，75,643,477B，SHA `fd7ee1b7cbe913794cb9c3e9c4f8bc8fc649325133366c1f723e285f81f43cdd`。其预训练与本队列重叠尚未核定，不能称严格未见域。模型按官方WBC例子先PIL双线性128再灰度/255，支持mask最近邻128，预测logits双线性回512；色彩、输入分辨率和额外先验均与FewClick不同，须披露。

预测支持mask来自支持点驱动的SAM-public，以预测IoU选候选、0.5阈值生成完整预测mask；未标像素不是人工负标注。完整GT轨道要求所有源像素已标注，不能把255填成背景。记录5个点位置与262144个已标注像素是不同信息量的账本，不意味着完整mask等于262144次人工点击。

## 门控与执行

`protocol.py`生成/验证任务。`test_transfer.py`检查代数、真实图像加载及GT路径隔离。`model_gate.py`用真实CPU权重在CRC/TIGER有/无目标四种任务上验证prototype、FewClick学习解码与query热图为零。`gate_context.py`真实运行官方v1在预测/完整GT上下文共8种情况，核对原生forward逐值一致并拒绝不完整GT。

`evaluate_sparse.py --gate`先经过同一缓存/推理/评分/逐行输出路径，每方法16条。完整每方法18700评分行，特征缓存64项，CPU4线程可与GPU训练并行。每次运行输出真实worker PID、源码/模型/协议SHA、逐任务结果及状态。CPU/缓存命中影响时间，当前联合graph+prototype计时不能当作两个方法各自独立在线时延；正式效率留G8。

G4完成仍需全量稀疏行、受控新权重迁移、近期方法全量及分轨迹统计。门控成功不等于已完成全量比较；G5独立训练seed、G6细胞/容量机制、G7泛化/TLS、G8 WSI/效率和G9证据闭环仍保留。

`evaluate_context.py` 是两个不同标注预算的全量入口。输入专用类禁止labels/supervision API，移除GT计数且将source/query标签路径设为不可读；另一个scorer对象只在预测后读query GT。SAM生成context只看source RGB/点，缓存键含图像、点和权重SHA；实际门控与原SAM重新编码比较。完整GT context只能由独立benchmark端提供完整source mask，不能按稀疏负点补成全负背景。

实际覆盖：212个支持视野中187个完整、25个含ignore。稀疏预测context全935对×5点组×4预算=18700行；完整GT参考仅837对、每对1行，98对按标注覆盖排除。参考预算记full_mask，不复制成5点组或点击曲线；与点方法比较时必须限制在同837对。CPU4线程、CUDA隐藏，模型/依赖/源码/官方权重manifest均与真实gate核对。正式入口须`--verified-gate`通过16条稀疏+4条完整GT门控。保存的row latency只含MultiverSeg预测，不含SAM生成/IO；总wall time单列，不作G8独立效率证据。
