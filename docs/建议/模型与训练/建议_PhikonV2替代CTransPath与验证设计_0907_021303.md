# 以 Phikon-v2 替代 CTransPath 的建议与训练验证设计

## 问题

为 v5 的 H&E 细胞表征分支选择一个较 CTransPath 更新、可复现且与当前数据规模匹配的病理 foundation backbone，并设计能够判断“训练是否真正有效”的实验，而不是只看到某个混合模型的数值变高。

## 已查证的事实

- 冻结 v4 的细胞视觉链是 `soft-attention H&E crop -> CTransPath-768 -> XCellFormer reg-64`。CTransPath 的输出维度为 768，已有 XCellFormer checkpoint 也是按该输入分布训练的；因此新 backbone 不能直接加载旧 `he_model_best.pth` 后声称是公平替换。
- 当前可严格审计的配对空间队列有 COAD、HCC、OV 的 Xenium/CosMx，其中 5 个 cohort 已通过 H&E—边界—H5AD annotation 合约；当前可证明约为 5 张对齐 H&E section、约 161.7 万个含 annotation 覆盖证据的细胞。扩展 archive 可提高细胞 crop 数量，但尚未获得直接可用的细胞类型 annotation。
- 因此这些数据足以做预训练 foundation model 的 **frozen feature + 小型适配器/LoRA** 训练，不足以从随机初始化训练一个新的大视觉 backbone，也不足以仅用大量相邻细胞 crop 宣称百万级独立样本的泛化能力。
- `environment.yml` 固定 `transformers==4.25.1`；Phikon-v2 的当前官方加载接口使用 `AutoImageProcessor` 和 `AutoModel`，正式实现前须用独立 v5 环境或经验证的依赖升级做一次真实加载 smoke，不能修改冻结 v4 环境作为替代。

## 候选比较

| 候选 | 发布时间/规模 | 训练数据与接口 | 对本项目的判断 |
| --- | --- | --- | --- |
| CTransPath（现基线） | 2022；768 维 | 旧版病理 Transformer | 保留为冻结公平基线，不再作为新训练主干。 |
| **Phikon-v2（推荐）** | 2024；ViT-L/16，约 0.3B；1024 维 | 4.56 亿张 224×224、20× H&E tile，来自约 6 万张 WSI；官方 `transformers` 接口；权重公开但为非商业许可 | 表征规模、年代和泛癌种覆盖均显著高于 CTransPath；比 1B 级模型更适合先做细胞 crop 缓存和多组消融。 |
| H-optimus-0 | 2024；约 1.1B；1536 维 | >50 万张 H&E WSI；需同意 gated 模型访问条款 | 可在获得权限和显存预算后作为上限对照；不作为第一落地，因为权重大、特征缓存/训练成本高且目前有访问门槛。 |
| UNI2-h | 2025；ViT-H/14 | >2 亿张 H&E/IHC 图像；模型下载受访问条款约束 | 很新且有吸引力，但同样应作为获权后的第二候选，而非阻塞 v5 的首个可复现实验。 |

Phikon-v2 模型卡说明其 CLS 特征为 1024 维；正式 v5 应新增 `1024 -> d_cell` 的可训练 adapter，再用新训练的 cell/region transformer，不复用依赖 CTransPath-768 的旧 XCellFormer 权重。建议先令 `d_cell=256` 或 512，并把维度、归一化和 crop/mpp 规范写入 checkpoint manifest。

## 推荐

选择 **Phikon-v2 作为 v5 的主 backbone**，采用以下边界：

```text
H&E cell crop -> frozen Phikon-v2 (CLS-1024) -> trainable cell adapter -> cell embedding
                                                      |                         |
                                            cell/RNA program heads       Phase 3 region fusion
```

RNA、蛋白和 annotation 只在训练时作为监督目标；新 WSI 的离线缓存以及线上点击始终只使用 H&E 推出的 cell embedding 和 region token。该设计不改变现有“Phase 2/3 离线、Phase 5/6 在线”的交互路径。

首轮只冻结 Phikon-v2、训练 adapter 与任务头；只有在跨空间块验证稳定正增益后，再尝试 LoRA（最后 2–4 个 Transformer block）。不进行全参数微调。

## 实验设计：先判定训练是否有效，再接入 FewClick

### 0. 固定的前置合约

1. 为 5 个 contract-passing cohort 生成版本化 manifest：H&E 路径、坐标变换、细胞边界、原始 annotation、8 类 ontology、RNA program target、crop 的实际 mpp。
2. 使用空间块而非随机细胞划分：每个 section 切为互不重叠的大块，按 70/15/15 生成 train/validation/test。这样仍是较宽松的 section 内验证，但不会让相邻细胞 crop 同时进入训练和测试。
3. 主开发结果汇总所有 section 的 spatial-block test；另报一个“整 section 留出”的补充结果（例如留出 OV Xenium），但不把它作为当前版本的唯一门槛或病例级泛化结论。
4. 同一 split、同一 crop、同一 adapter/head、同一训练步数、同一类别重加权和三个随机种子必须用于所有 backbone 对照。

### 1. Backbone 选择 gate（低成本，先做）

在固定空间划分上，仅训练线性分类头，backbone 全冻结：

| Arm | H&E feature source | 训练内容 | 用途 |
| --- | --- | --- | --- |
| A | 冻结 CTransPath-768 | 线性细胞类别头 | 历史视觉基线 |
| B | 冻结 Phikon-v2-1024 | 相同线性细胞类别头 | 判断新 backbone 的原始表征是否更好 |
| C | 冻结 Phikon-v2-1024 | 相同头，但输入 H&E crop 被确定性打乱 | 检查是否只利用类别/空间泄漏 |

主指标为 8 类 ontology 的 macro-F1 与 balanced accuracy，附每类 F1、混淆矩阵及按 cohort 的宏平均。C 应接近随机/显著低于 B；否则先排查 split 或 metadata leakage。B 相对 A 的提升是“换 backbone 的收益”，不是 RNA 收益。

### 2. H&E-only cell encoder 训练 gate（判断我们的训练是否有效）

以 B 的 frozen Phikon 特征为共同起点，只改变适配器和训练目标：

| Arm | Phikon | adapter/task heads | 训练目标 | 要回答的问题 |
| --- | --- | --- | --- | --- |
| B | frozen | 线性 cell-type head | annotation | 新 backbone 的无训练基线 |
| D | frozen | MLP adapter + cell-type head | annotation | 单纯细胞监督是否有效 |
| E | frozen | 同 D + RNA program head | annotation + 正确配对 RNA program | RNA 辅助训练能否提升 H&E-only embedding |
| F | frozen | 同 E | annotation + 行置换后的 RNA program | RNA 增益是否依赖正确配对 |
| G（仅通过后） | LoRA last blocks | 同 E | annotation + 正确 RNA | 轻量微调是否值得其额外成本 |

注意：E/F 的推理输入都只有 H&E；RNA 不能作为模型输入或测试时 feature。此前局部试验中的“XCell + RNA”是证明 RNA 信息本身有用，不能作为本表 E 的结果。

判定训练有效的预注册门槛：

- D 相对 B、E 相对 D，分别在三个随机种子上均有正向变化；汇总的 spatial-block test macro-F1 至少提高 **0.03 absolute**，且按空间块 bootstrap 的 95% CI 不跨 0。
- E 必须优于 F；若差值不足 0.02 macro-F1，不能声称 RNA 配对监督带来 H&E 细胞表征收益。
- 任一少数类 F1 不得比 D 下降超过 0.05；同时报告类频率，避免仅由上皮细胞主导的 accuracy 假提升。
- 记录 feature extraction 的 cells/s、峰值显存和每张 WSI 的离线 cache 时间；在线单击延迟不作为本阶段的指标，因为该分支已离线缓存。

### 3. FewClick 下游 gate（仅在阶段 2 通过后）

固定 GDPH 的 Phase 1/2、数据划分、prompt episode 和 Phase 5/6 训练配置，只替换 Phase 3 cell feature：

| Arm | Phase 3 cell feature | 要验证的结论 |
| --- | --- | --- |
| V4 | CTransPath + 已有 XCellFormer reg-64 | 冻结历史基线 |
| P0 | Phikon-v2 frozen + 新 adapter（未做 E 的 RNA 辅助训练） | 单独的新 backbone 收益 |
| P1 | E 的 H&E-only cell embedding | 我们的细胞/RNA 训练收益 |
| P2 | P1 但 cell token 置零 | 下游确实使用了 cell branch，而非偶然差异 |

在同一 test split 的 1、3、5、10 click 预算下报告 Dice、mIoU、Boundary-F1、每类分数和首屏 cache 时间。P1 必须相对 P0 在平均 Dice 或 mIoU 上取得预先设定的正增益，并在至少多数类上不退化，才推进大规模 cache 重建；否则停在 cell encoder，检查 ontology、配准、crop mpp 或 domain shift。

## 推荐执行顺序与停止规则

1. 先做 Phikon-v2 真实加载、一个 H&E crop 的 mpp/shape/有限特征 smoke；失败时只解决依赖与数据契约，不启动训练。
2. 完成阶段 1 的 A/B/C。若 B 不优于 A，则保留 CTransPath，停止后续 Phikon 训练。
3. 完成阶段 2 的 B/D/E/F（3 seeds）。若 D、E 不达门槛，不接入 FewClick；先修正标签本体、空间划分或目标定义。
4. 仅对通过的 E 尝试 G；若 G 的质量/吞吐折中不值，选 E 的 frozen backbone 版本。
5. 通过后才训练新的 v5 Phase 3、重建 token cache、开始阶段 3 的 GDPH FewClick 消融。

## 风险与备选

- Phikon-v2 是非商业许可；任何商业部署前须重新核查许可。科研内部实验可先按模型卡条款执行。
- 其推荐 224×224/20× 域与本地 H&E 实际 mpp 可能不一致；未经 mpp 审计不能把像素 crop 尺寸直接当作 20×。
- `transformers` 依赖升级可能影响旧环境；在独立 v5 环境验证，不改变 v4 基线。
- 若 Phikon-v2 的吞吐无法接受，先缓存 CLS-1024，再训练 adapter；若仍不可接受，可评测规模较小的 H-optimus-0-mini（需取得权重与重新核对许可），但它不替代本推荐的主对照。

## 证据来源

- Phikon-v2 官方模型卡：<https://huggingface.co/owkin/phikon-v2>（模型、预训练规模、1024 维 CLS、加载接口与许可）。
- Phikon-v2 论文：<https://arxiv.org/abs/2409.09173>。
- UNI 官方仓库：<https://github.com/mahmoodlab/UNI>（UNI2-h 于 2025 年发布、ViT-H/14、训练图像规模及下载入口）。
- H-optimus-0 官方模型卡：<https://huggingface.co/bioptimus/H-optimus-0>（当前访问受 gating 约束）。
- 本项目资料：`docs/建议/模型与训练/建议_大规模重训总体方案_0907_010549.md`、`docs/建议/数据与评估/建议_多癌种数据就绪性审计_0907_012142.md`、`docs/运行记录/数据审计/运行记录_细胞队列样本与本体审计_0907_015233.md`。
