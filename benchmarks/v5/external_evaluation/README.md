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
