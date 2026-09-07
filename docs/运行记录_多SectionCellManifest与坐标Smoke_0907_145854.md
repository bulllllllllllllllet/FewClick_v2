# 多 Section cell manifest 与坐标 smoke

## 目的

将已经通过 H&E、边界和 H5AD annotation 数据合约审计的五个 cohort 转换为可重放的 v1 cell-encoder manifest，并以真实 H&E 双视野 crop + Phikon-v2 编码对每个可直接使用的 section 做小规模 smoke。RNA 仅用于离线训练监督；本次 smoke 不向模型输入 RNA。

## 输入与边界

- cohort contract：`/nfs-medical3/zyh/fewclick_v2/dataset_audit/cell_supervision_audit_0907_014551/cohorts.json`
- ontology：`v1_0907`，来自 `benchmarks/v5/tools/audit_cell_ontology.py`
- 读取数据根：`/nfs-medical3/fyx/PKU_DATA/{Xenium,CosMx}/{COAD,HCC,OV}`（只读）
- 输出根：`/nfs-medical3/zyh/fewclick_v2/multisection_cell_manifest/multisection_cell_manifest_0907_145854/`
- 计算：manifest 使用 CPU；H&E encoder smoke 使用单张 GPU（`CUDA_VISIBLE_DEVICES=1`）。

## 当前已知风险与处理

- Xenium 三个 cohort 的边界坐标可直接落入对应 H&E 的图像范围，可进入本次 crop smoke。
- CosMx COAD/HCC 的 `x_global_px/y_global_px` 与 H&E TIFF 像素尺寸不在同一坐标范围；目录中尚未发现显式 affine/alignment 文件。因此 manifest 将它们标记为 `requires_coordinate_transform`，不提取 H&E crop、更不进入训练。直接缩放或猜测变换会损害 H&E-RNA 配对，不采用。
- 每个癌种目前只确认到一张 H&E section，donor/patient ID 未知；这不是病例级泛化验证，也不能作为最终多癌种结果。

## 验收标准

1. manifest 的每条 sampled cell 均有稳定 cell ID、raw annotation、broad ontology、H&E/RNA/boundary 路径和中心坐标来源；未映射 annotation 为 0。
2. 三个 Xenium section 的所有 sampled center 均在 H&E 边界内；256px、384px crop 均可进入 H&E-only Phikon-v2，输出 `[N, 1024]` 且 finite。
3. CosMx 在得到可验证坐标变换前保持排除，不因 manifest 完成而被视为可训练。

## 命令与监控

```bash
conda run -n muf-clust python benchmarks/v5/tools/build_multisection_cell_manifest.py \
  --timestamp 0907_145854 --samples-per-class 8

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/nfs-medical3/zyh/fewclick_v2/vendor/phikon-v2:$PYTHONPATH \
conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/smoke_multisection_cell_crops.py \
  --manifest-dir /nfs-medical3/zyh/fewclick_v2/multisection_cell_manifest/multisection_cell_manifest_0907_145854 \
  --timestamp 0907_145854 --device cuda:0
```

## 最终状态

- 首次运行（`0907_145854`）在第三个大型 TIFF 的 libvips 临时缓冲写入时因本机 `/tmp` 仅余约 5.3 GB 而退出，`EXIT_CODE=1`；空输出目录与失败日志均保留，未修改源数据。
- 重跑（`0907_150803`）将 `TMPDIR`、`VIPS_TMPDIR` 显式设为 `/nfs-medical3/zyh/fewclick_v2/multisection_cell_smoke/vips_tmp_0907_150803`，`EXIT_CODE=0`。
- Xenium COAD 56、HCC 56、OV 64 个抽样 cell 的所有中心均位于各自 H&E raster 内；256px 与 384px 两种 crop 均无需边缘平移，并各产生 finite 的 `[N,1024]` Phikon CLS 特征。
- 因此三套 Xenium 可作为**首轮多 section cell-encoder 数据加载/坐标候选**。这只是数据管线门禁通过，不是训练准确率，也不是病例级泛化证据。
- CosMx COAD/HCC 的 112 个 manifest cell 仍标记为 `requires_coordinate_transform`，未读入 H&E encoder、未纳入训练。下一步必须找到或标定并可视化核验其 H&E 到 CosMx global-pixel 的转换。

结果：`/nfs-medical3/zyh/fewclick_v2/multisection_cell_smoke/multisection_cell_smoke_0907_150803/metrics.json`；特征：同目录 `phikon_dualview_features.npz`。
