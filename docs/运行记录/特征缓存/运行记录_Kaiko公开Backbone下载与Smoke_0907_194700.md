# Kaiko 公开病理 backbone 下载与 smoke

## 目的

因 UNI2-h 需要 Hugging Face gated 账户而当前服务器不能认证，改用公开可下载的 Kaiko `vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms` 作为独立病理 feature backbone 候选。目标是先验证真实 H&E crop 前向，再决定是否加入 CTransPath / Phikon 的公平 frozen-feature 对照；不替换 v4。

## 已核对输入与依赖

- Hugging Face 仓库：`1aurent/vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms`；元数据为 `gated=false`，`model.safetensors` 大小约 1.22GB，模型卡为 Kaiko non-commercial license。
- 模型配置：`vit_large_patch14_reg4_dinov2`，`num_features=1024`，ImageNet mean/std，配置中的标准输入为 518px；实验中不得沿用 Phikon 的归一化或假称两者的 pixel crop/mpp 相同。
- 当前 `aligner` 的 `timm==0.9.8` 没有该 `reg4` 架构。新版 timm 将作为 `/nfs-medical3/zyh/fewclick_v2/vendor/kaiko_timm_0907_194700/` overlay 安装，不修改冻结 v4 环境。

## 运行与验收

- 权重目标：`/nfs-medical3/zyh/fewclick_v2/model_weights/kaiko_vit_large_patch14_reg4/kaiko_0907_194700/`；使用 `huggingface_hub.snapshot_download` 直接写入该带时间戳目录。
- 下载 tmux：`fewclick_kaiko_download_0907_194700`，日志：`/nfs-medical3/zyh/fewclick_v2/model_weights/logs/kaiko_download_0907_194700.log`。
- smoke 将使用已审计的 COAD Xenium H&E 真实 crop，并检查权重 identity、输入 `[1,3,518,518]`、1024 维有限输出与 GPU memory；其输出目录和命令在下载完成后补充。
- 接受条件：下载的 `config.json` / `model.safetensors` 完整，overlay 能构造匹配架构，真实 H&E 前向结果 finite。失败时保留 CTransPath/Phikon，不用替代 checkpoint。

## 最终状态

- 初始 `huggingface_hub.snapshot_download` 在约 576MB 时停止增长；未删除该 partial，而是使用服务器确认支持 HTTP Range 的官方 Hugging Face CDN 地址继续下载。`curl -C -` 完成后对模型卡给出的 SHA-256 `149c8b21c60a7115a17572073b09ef142411c91b5194c84a99ebba8b83f20e69` 校验通过，`EXIT_CODE=0`。
- Kaiko smoke 完成：真实 Xenium COAD H&E 的 `[1000,2000,518]` RGB crop，以模型卡的 ImageNet normalization 输入本地权重，GPU 输出 finite `[1,1024]` feature。`timm==1.0.24` overlay 成功构建 `vit_large_patch14_reg4_dinov2`；v4 及其内置环境没有改动。
- smoke artifact：`/nfs-medical3/zyh/fewclick_v2/kaiko_smoke/kaiko_pathology_smoke_0907_221000/`。这只证明权重、依赖与真实 H&E 前向路径有效，不表示 Kaiko 已优于 CTransPath 或 Phikon。
- 下一步：为当前相同的细胞 crop 建立 Kaiko feature cache，并在与 CTransPath/Phikon 相同的 spatial split 下进行 frozen linear probe；只有胜出后才考虑接入 FewClick Phase 3。
