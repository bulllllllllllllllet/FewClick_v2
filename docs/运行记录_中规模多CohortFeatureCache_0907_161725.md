# 中规模五 cohort H&E cell feature cache

## 目的

将当前每 cohort、每 broad class 8 个 cell 的坐标 smoke 扩展到 64 个，构建可恢复的 H&E-only 双视野 Phikon 特征 cache，为下一轮跨 cohort gate 提供足够样本。预期约 2,304 个 cell，约为前一轮的 8 倍。

## 输入与范围

- cohort contract：`/nfs-medical3/zyh/fewclick_v2/dataset_audit/cell_supervision_audit_0907_014551/cohorts.json`
- ontology：`v1_0907`；采样按 cohort × broad class 平衡，最多 64 cell/class。
- CosMx COAD/HCC 使用已审计 global-to-H&E affine：`/nfs-medical3/zyh/fewclick_v2/cosmx_coordinate_audit/cosmx_he_transform_0907_153100/`。
- 训练和 cache 构建都只读取 H&E；RNA、蛋白、transcript 不输入 Phikon。

## 实现与执行计划

1. 生成新的 64-per-class raw manifest，再应用已审计 CosMx transform，产生独立 immutable manifest。
2. `cache_multisection_cell_features.py` 按 cohort 分片写入 `cells.csv`、双视野 feature `.npz`、shard completion record 和全局 index。已完成 shard 保留，即使后续 cohort 失败也可诊断。
3. 先用每 cohort 4 个真实 cell 运行 cache smoke，验证 image loader、feature shape、有限值和 shard identity；通过后用单 GPU 运行完整 cache。

## 运行约束与验收

- output root：`/nfs-medical3/zyh/fewclick_v2/multisection_cell_feature_cache/`
- 计算：单 GPU `CUDA_VISIBLE_DEVICES=1`；完整 cache 在 tmux 中运行，libvips `TMPDIR/VIPS_TMPDIR` 指向 NFS。
- 通过条件：每个 shard 的 `cell_id` 顺序与双视野 feature 对应，所有 feature `[N,1024]` finite，中心坐标在 H&E raster 内，index 中五 cohort 均有 `complete` record。
- 该 cache 仍只能用于 H&E cell encoder 预训练/gate；没有组织像素 GT，不能混入 FewClick 主分割 loss。

最终状态和运行时间会在 cache 完成后追加。

## 分片 cache smoke

- `0907_161900` 使用每 cohort 4 个真实 cell 运行完成，`EXIT_CODE=0`。
- 五个 shard 各自写出 `cells.csv`、`phikon_dualview_features.npz` 与 `COMPLETED.json`；每个 view 的 feature shape 为 `[4,1024]` 且所有值 finite；root `index.json` 标记 `all_complete=true`。
- smoke artifact：`/nfs-medical3/zyh/fewclick_v2/multisection_cell_feature_cache/multisection_cell_feature_cache_0907_161900/`。

完整 2,266-cell cache 将在 code commit 后以同一 manifest 启动；tmux session、GPU PID 和每个 cohort shard 状态会在其运行时记录。

## 完整 cache 启动

- Timestamp/output：`0907_162801`，`/nfs-medical3/zyh/fewclick_v2/multisection_cell_feature_cache/multisection_cell_feature_cache_0907_162801/`
- Code revision：`cfd12ba`。
- Command：`CUDA_VISIBLE_DEVICES=1 ... cache_multisection_cell_features.py --manifest-dir ...multisection_cell_manifest_0907_161850 --timestamp 0907_162801 --device cuda:0`。
- tmux：`fewclick_featurecache_full_0907_162801`；日志 `/nfs-medical3/zyh/fewclick_v2/multisection_cell_feature_cache/logs/feature_cache_full_0907_162801.log`；`TMPDIR` 与 `VIPS_TMPDIR` 均使用 NFS 临时目录。
- 预计：单 GPU、约 1 小时量级；以 root `index.json` 中的 cohort completion record 监控，所有五 shard `complete` 且 `all_complete=true` 才结束。
