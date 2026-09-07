# Phikon-v2 权重下载与真实 H&E Smoke 运行记录

## 目的

将已选定的 Phikon-v2 权重下载至 FewClick_v2 专用 NFS 路径，并通过 COAD Xenium 的真实 224×224 H&E crop 运行 `Phikon-v2 CLS-1024 -> v5 CellFeatureAdapter-256`，验证模型、依赖、GPU、数据读取和输出有限性。

## 输入与环境

- 权重来源：`owkin/phikon-v2`（官方 Hugging Face repository）。
- 权重目录：`/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2/`。
- H&E 输入：`/nfs-medical3/fyx/PKU_DATA/Xenium/COAD/HE.tif`，像素窗口 `(x=1000, y=2000, size=224)`。
- 代码：`benchmarks/v5/phase_3_cell_region/tools/download_phikon_v2.py`、`smoke_phikon_v2.py`；新模块 `src/phikon_v2.py`。
- 环境：`aligner` 的 PyTorch/CUDA；现有 `transformers==4.25.1` 不满足 Phikon-v2 的 DINOv2 接口，使用 `/nfs-medical3/zyh/fewclick_v2/vendor/phikon-v2/` 的 v5 专用依赖 overlay，不修改冻结 v4 环境。
- 计算：GPU 1，单 GPU；下载和 smoke 使用 tmux，日志写入本次 artifact。

## 命令

```bash
conda run -n aligner python -m pip install --no-deps --target /nfs-medical3/zyh/fewclick_v2/vendor/phikon-v2 -r benchmarks/v5/phase_3_cell_region/requirements-phikon-v2.txt
conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/download_phikon_v2.py
PYTHONPATH=/nfs-medical3/zyh/fewclick_v2/vendor/phikon-v2:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/smoke_phikon_v2.py --device cuda:0 --timestamp 0907_021752
```

## 验收条件

1. 权重目录同时包含 config、预处理配置和至少一个模型权重文件，且不覆盖已有文件。
2. 真实 H&E crop 输出 shape 为 `[1, 1024]`，adapter 输出为 `[1, 256]`，所有数值有限。
3. 输出保存到 `/nfs-medical3/zyh/fewclick_v2/backbone_smoke/runs/phikon_v2_smoke_0907_021752/`；下载日志保留在 `/nfs-medical3/zyh/fewclick_v2/backbone_smoke/phikon_v2_smoke_0907_021752/`。

## 最终状态

第一次依赖尝试被主动停止：未写入 v5 overlay 或模型目录。原因是未使用 `--no-deps` 时，pip 会尝试把已有的 `torch==2.9.1+cu128` 解析为 `torch==2.14.0`，会产生不必要的多 GB 下载并破坏“仅替换 v5 Transformer 层”的环境边界。已将依赖文件改为 `transformers==4.49.0` 与 `tokenizers==0.21.4`，后续用 `--no-deps` 安装到 v5 专用 overlay，复用 aligner 已验证的 PyTorch、CUDA、`huggingface_hub` 和 `safetensors`。

下载阶段的 Python Hugging Face client 在 tmux 中没有继承代理变量，出现 `Network is unreachable`；权重文件实际通过同一官方 repository 的 IPv4 CDN 直连、带 Mozilla User-Agent 且支持断点续传的命令下载。下载前已用官方 API 枚举 repository 文件；未从第三方镜像获取任何权重。

真实 smoke 的首次自动调用被输出目录存在性保护拦截（下载日志已经占用了原先的 artifact root）；没有执行任何模型前向。随后以独立的 `runs/` output root 重跑，成功完成。

### 完成结果（09-07 02:31:33）

- 权重目录：`/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2/`，包含 `config.json`、`preprocessor_config.json`、`model.safetensors`、`README.md`、`LICENSE.pdf`。
- `model.safetensors` SHA-256：`261ae680fa699b3b951597fd57aa19c02ef735805acb104b93af69b36d928569`；与官方 ETag 一致。
- v5 dependency overlay：`transformers==4.49.0`、`tokenizers==0.21.4`，路径为 `/nfs-medical3/zyh/fewclick_v2/vendor/phikon-v2/`；复用 aligner 的 `torch==2.9.1+cu128`，未修改 v4 依赖。
- 真实 COAD H&E crop 在 GPU 1 上成功完成前向：Phikon CLS shape `[1, 1024]`，adapter shape `[1, 256]`，两者均为 finite。
- 最终 smoke artifact：`/nfs-medical3/zyh/fewclick_v2/backbone_smoke/runs/phikon_v2_smoke_0907_021752/`，其中 `metrics.json` 记录了模型、H&E、坐标与 tensor 合约。
