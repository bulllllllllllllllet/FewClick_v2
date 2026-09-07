# Phikon-v2 冻结 Backbone A/B/C Gate 运行记录

## 目的

在完全相同的、带数据提供方真实细胞注释的 COAD Xenium H&E crop 上，比较冻结 CTransPath 与冻结 Phikon-v2 的可线性分性，并增加 H&E—细胞配对打乱的负对照。该运行验证第一层 backbone gate 的代码和证据链，不作为多癌种或 FewClick 分割最终结果。

## 输入和分割

- 样本：此前 XCell pilot 的同一 COAD Xenium 窗口 `(1000, 2000, 1500)`；从 1,020 个候选细胞中选出具有注释且四个空间块均覆盖的类别，预期为 941 个细胞、`CD4T`、`Epithelial`、`Fibroblast`、`Macrophage`。
- 图像：同一 128×128 soft-attended H&E crop，两个 backbone 各自 resize/preprocess 到 224×224；因此比较的是 backbone 表征，而不是实例选择或视觉输入的差异。
- A：冻结 CTransPath CLS/pooled 768 维；B：冻结 Phikon-v2 CLS 1024 维；C：B 的 feature row 按确定性 permutation 重新分配给细胞。
- 评估：四象限空间块 `GroupKFold`，每折训练相同的 `StandardScaler + class_weight=balanced LogisticRegression`；报告 macro-F1 和 balanced accuracy。
- GPU：`CUDA_VISIBLE_DEVICES=1` 下的 `cuda:0`；输出 `/nfs-medical3/zyh/fewclick_v2/backbone_probe/phikon_backbone_probe_0907_024652/`。

## 命令

```bash
PYTHONPATH=/nfs-medical3/zyh/fewclick_v2/vendor/phikon-v2:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1 conda run -n aligner python \
  benchmarks/v5/phase_3_cell_region/tools/run_phikon_backbone_probe.py \
  --device cuda:0 --timestamp 0907_024652
```

## 验收条件

所有 A/B/C 特征 shape 和数值有限；C 的 macro-F1 必须低于 paired Phikon-v2，防止分类器只利用细胞排序或其他泄漏。B 与 A 的差值仅作为本 section 的候选 backbone 证据，后续仍须在多 cohort spatial-block split 上复核。

## 最终状态

初始时间戳 `0907_024652` 的 tmux 预启动留下空输出目录，第二次调用被“不覆盖已有结果”保护拒绝；没有发生模型前向。重试 `0907_024859` 后，发现 image 已裁成局部窗口而 centroid 仍为全局坐标，导致 crop 为空；已在 `build_attended_crops` 中显式执行 `(center_x - 1000, center_y - 2000)` 的局部坐标转换。最终有效运行使用 `0907_024957`，结果如下。

| Arm | macro-F1 | balanced accuracy | 相对 CTransPath macro-F1 |
| --- | ---: | ---: | ---: |
| A: frozen CTransPath-768 | 0.2515 | 0.2850 | 0.0000 |
| B: frozen Phikon-v2-1024 | 0.2340 | 0.2332 | -0.0175 |
| C: B 的 H&E—cell 配对打乱 | 0.2310 | 0.2305 | -0.0204 |

- 有效样本：941 个细胞、4 个真实数据提供方注释类别、4 个空间象限折。
- B 相对 C 的 macro-F1 只高 `0.0030`，未满足“配对 H&E 明显优于打乱对照”的实现 gate；B 也未超过 A。
- 特征文件：`ctranspath_768.npy` SHA-256 `74f71656…ba35a23f`，`phikon_v2_1024.npy` SHA-256 `fadd7b5d…6d8220f43f`。
- 最终 artifact：`/nfs-medical3/zyh/fewclick_v2/backbone_probe/phikon_backbone_probe_0907_024957/`，日志：`/nfs-medical3/zyh/fewclick_v2/backbone_probe/logs/phikon_backbone_probe_0907_024957/run.log`。

**结论：不通过，停止进入 adapter/RNA 训练。** 这个单 section、128×128 像素 soft-attended crop 的结果不能证明 Phikon-v2 无效，更不能证明其优于 CTransPath。首要排查是 H&E 的实际 mpp 与 Phikon 的 20×/224 输入尺度是否匹配，以及将单窗口四象限 split 扩展到已经审计的多个 section；完成这些前不更新 v5 默认 backbone。
