# Kaiko 冻结 backbone gate

## 目的

在先前 CTransPath/Phikon 的同一 COAD Xenium 局部窗口与空间四折上，新增公开 Kaiko ViT-L/14-reg4 冻结特征臂。只判断其是否值得进入 FewClick Phase 3 候选，不宣称细胞类型分类或最终分割结论。

## 设计

- 保留 941 个、四个数据提供方 annotation 类别及相同 128px soft-attention H&E 原始 crop。
- CTransPath、Phikon、Kaiko 均从同一 crop 得到输入，但按各自模型卡 resize/normalization；Kaiko resize 至 518px、ImageNet mean/std、输出 1024 维。
- 评估沿用四空间块 GroupKFold、相同线性分类器；新增 Kaiko feature 行打乱对照。
- GPU：`CUDA_VISIBLE_DEVICES=1`；输出 `/nfs-medical3/zyh/fewclick_v2/backbone_probe/kaiko_backbone_probe_0907_221500/`；验收要求 feature finite、对应 ID 不变、Kaiko paired 同时高于其 shuffle 且高于 CTransPath，才进入后续 Phase 3 候选讨论。

## 最终状态

- 初始 `0907_221500` 运行在 Phikon 初始化前停止，`EXIT_CODE=1`：新脚本未将已有 v5 Phikon `transformers` overlay 加入解释器路径，错误加载冻结环境的 `transformers==4.25.1`。没有生成 feature、分类指标或可解释的模型比较结果。
- `0907_222500` 重跑修复了 Phikon overlay，但 CTransPath 辅助模块已在 Kaiko overlay 前导入旧 `timm==0.9.8`，故 Kaiko 架构未找到，`EXIT_CODE=1`。仍没有输出任何 feature 或有效指标。
- `0907_222700` 将 Kaiko overlay 前置后，旧 CTransPath helper 与 `timm==1.0.24` 发生 API 冲突，`EXIT_CODE=1`。这说明三个模型不能安全地在同一解释器重建；仍没有输出任何 feature 或有效指标。
- 采用可审计修正：加载先前 `phikon_backbone_probe_0907_024957` 已保存的 CTransPath `[941,768]`，逐 cell ID 验证与当前 deterministic cells 相同；本次解释器只提取 Phikon/Kaiko，消除旧/新 timm 冲突。以 `0907_223000` 重跑；最终结果待补充。

## 有效运行结果

- `0907_223000` 完成，`EXIT_CODE=0`。941 个 cell ID 与历史 CTransPath feature 逐项一致；Kaiko `[941,1024]` feature 全部 finite。

| Frozen feature | Macro-F1 | Balanced accuracy |
| --- | ---: | ---: |
| CTransPath-768 | 0.2515 | 0.2850 |
| Phikon-v2-1024 | 0.2344 | 0.2338 |
| Kaiko ViT-L/14-reg4-1024 | 0.2265 | 0.2430 |
| Kaiko H&E—cell feature 打乱 | 0.2425 | 0.2437 |

- Kaiko 相对 CTransPath 的 Macro-F1 为 `-0.0250`；相对其自身 shuffle 为 `-0.0160`，balanced accuracy 也略低于 shuffle。因此不通过 paired-over-shuffled 与超过 v4 CTransPath 的双重 gate。
- 结论：在这个完全相同的单 section、128px soft-attention crop 路线上，Kaiko 不能替代 CTransPath；不要把它接入 FewClick Phase 3 或启动下游训练。该结论不排除模型在经 mpp 审计的原始 518px 物理视野、边界感知 crop 或独立多 section 数据上的表现，但当前证据不支持继续投入。
- artifact：`/nfs-medical3/zyh/fewclick_v2/backbone_probe/kaiko_backbone_probe_0907_223000/`。
