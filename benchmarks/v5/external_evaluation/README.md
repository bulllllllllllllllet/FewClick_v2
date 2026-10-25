# 外部域的真实物理视野

`prepare.py` 在现有源目录中审计 PUMA/OCELOT/IGNITE 的原生物理覆盖。每张可用 ROI 只取几何中心，沿用训练时256/512/1024µm三视野和原生RGB裁剪，统一512像素输出。不按GT移动中心，不填充不足的上下文。

2026-10-25的现有目录中，只有10张 IGNITE 满足视野条件；PUMA205张和OCELOT663张均不足。这10张原有split为train7/val1/test2；当前CRC/TIGER训练和选模未使用IGNITE，因此全部只作当前版本外部域评估，原split保留作来源信息。不声称其完全未被历史模型/通用上游预训练见过，也不声称10个独立病例。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/home/zhaoyh/miniconda3/envs/aligner/bin/python -m benchmarks.v5.external_evaluation.prepare \
  --output /nfs-medical3/zyh/fewclick_v2/fair_comparison/external_MMDD_HHMMSS
```

输出availability.parquet（全部1276源的覆盖审计）、fields.parquet（10个真实三视野）、local_maximum.json（34个图像目标对×5组点击）、contract.json。每组最大5正2负，预算3+2/1+1/3+1/5+1嵌套；沿用2µm内部点策略。实际覆盖9类，不能声称覆盖完整16类。图像/标签/点击SHA和原生输入SHA保留。细胞缓存明确为not_prepared，不能给主模型静默补零。

准备验收包括真实图像尺寸、源级GT schema、原生vips裁剪逐像素回放、磁盘reload、真实物理尺寸以及1190个点击的符号检查。任务构建读取GT，推理输入必须另行隔离GT。已准备输入不等于模型精度或泛化结论；后续需细胞缓存、真实模型门控、全部匹配对手评分及来源聚类统计。权重、阈值、超参数只能来自CRC/TIGER开发协议，禁止在此重新选模。

`evaluate_public.py` 提供单独的输入和GT评分API；真实门控将GT路径改成不可读，随后原生推理仍须成功。SAM缓存图像embedding，按原生predicted-IoU选mask，门控与SAMAdapter完整重新编码逐像素比较。WSI-SAM调用原生双视野接口。CPU4线程、CUDA隐藏。门控8行后正式680行；正式入口必须引用已完成gate，核对源码与协议SHA。门控概率落盘，直接重算TP/FP/FN。输入GT模拟点与GT评分均披露，不声称使用人工点击。CPU缓存耗时不能冒充GPU在线效率。

`wsi_reader.py` 是后续整片流程的输入入口：通过同一SDK底层KFB ROI函数直接传播读取异常，避免高层接口把异常替换成白图。KFB只有vendor `CapRes`，SDK将它同时写作`mpp-x`和`mpp-t`（不是独立测得的y轴MPP）；显式记录等向来源。使用原生fine整数框及2x/4x框，不继承旧10x网格；边界不足直接报错，尚未实施整片边缘覆盖协议。当前门控仅代表片真实三视野读取与坏读传播，不含模型、细胞或全片指标。

`cell_data.py` 只读图像和真实cell cache；缓存不完整直接拒绝，没有自动零特征替代。CPU门控 `gate_cell_data.py` 用真实CRC/TIGER已有细胞与实际训练gate checkpoint，验证新装配与LocalData的15个张量、模型输出逐值一致，真实loss有限；GT路径不可读不影响新装配。它不能替代新的外部核检测门控。

`cells.py` 沿用InstanSeg/CTransPath/XCell与训练同一prepare_variant：fine从准备好的512图提取，middle/coarse从原生context提取，unknown核类和74维/255上限不变。原生GPU门控先提取1个field的全部3尺度并回放缓存；通过后`--reuse`做其余9个。`cell_queue.py` 等G6的GPU4无细胞/形态两组完整评分完成且GPU4实际空闲，再依次启动原生gate/full。队列锁定源码/协议，失败保留，不自动重跑或换算法。队列正在等待时不要改其contract列出的源文件。新模型外部评分为后续独立验收，排队成功不代表缓存完成。
