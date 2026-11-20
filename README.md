# FewClick_v2

**少量点击驱动的病理组织检索与全切片分割。**

FewClick_v2 研究如何在一个支持图像块上提供少量正、负点击，将用户指定的组织目标传播到同一张全切片图像（Whole-Slide Image，WSI）的其他位置。远端查询区域无需追加人工提示；切换切片或目标时，重新选择支持图像块并提供点击。

本仓库包含模型实现、训练与评估流程、公平对照、消融实验，以及中英文论文材料。当前固定研究方法为 **HG-mix**，核心代码位于 [`benchmarks/v5/`](benchmarks/v5/)。

[当前任务与进度](goal.md) · [代码导航](benchmarks/v5/README.md) · [实验记录](docs/运行记录/模型实验/) · [论文源码](论文/CVPR_HG_mix/CVPR2027_English_1115/) · [仓库维护](docs/仓库维护.md)

## 任务流程

1. **定义目标**：在当前 WSI 的一个支持图像块上标记正、负点击，指定本次要分割的组织。
2. **提取多尺度特征**：读取支持与查询位置的精细、2×、4×物理视野，提取组织形态与上下文特征。
3. **支持到查询传播**：结合提示条件下的区域解码、点击相似度修正和稠密概率迁移，预测远端查询区域。
4. **整片输出与评价**：遍历切片网格、融合重叠窗口，并按唯一像素累计整片指标。

本图交互、同片远端图块迁移和整片 WSI 评价分别记录。整片结果同时报告完整评分域与排除支持输入覆盖范围的 **strict-remote** 指标；远端图块评价要求支持与查询的完整 4× 输入范围互不重叠。

## 当前方法：HG-mix

- **冻结的 Phikon-v2 图像编码器**：共享编码器处理三个物理尺度；精细视野的整幅特征与四子块拼接特征等权融合。
- **多尺度区域上下文**：软分配汇聚区域 token，通过跨尺度交互组织空间证据。
- **正负提示条件化**：编码支持点击，执行查询区域解码和像素投影，并加入点击相似度修正。
- **稠密概率迁移**：将点击锚定后的支持预测通过粗到细对应关系传递给查询区域，以有界残差细化输出。
- **特征缓存与整片执行**：支持编码、查询特征和解码路径的复用，并保留输入、权重、覆盖范围与指标复算记录。

当前 HG-mix 的有效预测分支使用图像特征；仓库也保留早期细胞特征与 RNA/空间组学研究模块。完整方法见[论文方法源码](论文/CVPR_HG_mix/CVPR2027_English_1115/sec/3_method.tex)，实现入口见 [`hg_combination_study/model.py`](benchmarks/v5/hg_combination_study/model.py)。

## 代码与文档导航

| 路径 | 内容 |
| --- | --- |
| [`benchmarks/v5/hg_combination_study/`](benchmarks/v5/hg_combination_study/) | HG / HG-mix 模型、训练和图块评价 |
| [`benchmarks/v5/whole_slide_once/`](benchmarks/v5/whole_slide_once/) | HG-mix / SANSA 同片整片推理与唯一像素评分 |
| [`benchmarks/v5/whole_slide_sam/`](benchmarks/v5/whole_slide_sam/) | SAM 系列整片对照 |
| [`benchmarks/v5/hg_paper_ablation/`](benchmarks/v5/hg_paper_ablation/) | 固定方法的核心结构消融 |
| [`benchmarks/v5/reviewer_sample/`](benchmarks/v5/reviewer_sample/) | 归因对照及 15 / 30 WSI 扩展评价 |
| [`benchmarks/v5/paper_figures/`](benchmarks/v5/paper_figures/) | 论文图表生成与排版脚本 |
| [`benchmarks/v4/`](benchmarks/v4/) | 冻结的历史基线与复用模块 |
| [`docs/`](docs/) | 数据说明、实验方案、公平合同与运行记录 |
| [`论文/`](论文/) | 当前论文、历史稿件与可复现性材料 |
| [`module/`](module/)、[`XCellFormer.py`](XCellFormer.py) | 从早期项目继承的编码器及基础组件 |

## 环境与运行入口

项目使用 `aligner` Conda 环境。已有环境时，在仓库根目录运行：

```bash
conda activate aligner
```

新机器可从仓库中的基础环境快照开始：

```bash
conda env create -f environment.yml
conda activate aligner
```

[`environment.yml`](environment.yml) 是 Linux 环境快照。具体实验还需要对应的图像编码器依赖、外部模型源码、预训练权重、数据与任务清单；它们的位置和版本由实验合同记录。原始数据、模型权重及特征缓存保存在外部存储中，不随 Git 仓库分发。

训练与推理采用 `python -m benchmarks.v5.<模块>` 形式的实验入口。运行前按以下顺序确定任务及参数：

1. 阅读 [`goal.md`](goal.md)，确认当前方法、任务范围与续接入口。
2. 查阅[实验 registry](docs/运行记录/模型实验/CVPR研究进度_1030_174832.json) 和[当前运行记录](docs/运行记录/模型实验/运行记录_SAM当前9500公平适配与WSI并行_1112_171010.md)，取得实际命令、合同、清单与权重路径。
3. 按对应记录执行输入检查和功能门控，再运行训练或评价。

实验脚本依赖各自的 `contract.json`、任务清单和已验收权重；当前仓库以研究复现流程为入口，尚未提供面向任意 WSI 的统一安装包与一键推理命令。

## 实验与评价约定

- 数据按空间块划分，保留患者/切片来源，并审计多尺度完整输入范围。
- 查询标注用于监督或评分；远端推理、候选提示选择和阈值选择不读取查询 GT。
- 比较方法时记录预训练来源、目标数据监督、输入视野、提示预算、模型选择与指标规则，见[公平对比合同](benchmarks/v5/fair_comparison/README.md)。
- 以运行记录中的验收证据判断实验完成状态。正式 15 WSI 六方法结果见[结果表](docs/运行记录/模型实验/结果表_WSI六方法正式15片最终结果_1113_224245.md)；后续控制实验见[15 WSI 记录](docs/运行记录/模型实验/运行记录_ProtoMS核心消融完整15WSI_1116_214757.md)与[30 WSI 记录](docs/运行记录/模型实验/运行记录_四方法消融完整30WSI_1117_173543.md)。

## 论文材料

当前稿件位于 [`论文/CVPR_HG_mix/CVPR2027_English_1115/`](论文/CVPR_HG_mix/CVPR2027_English_1115/)：

| 材料 | 源码 | PDF |
| --- | --- | --- |
| 英文正文 | [main.tex](论文/CVPR_HG_mix/CVPR2027_English_1115/main.tex) | [main.pdf](论文/CVPR_HG_mix/CVPR2027_English_1115/main.pdf) |
| 中文阅读版 | [main_ch.tex](论文/CVPR_HG_mix/CVPR2027_English_1115/main_ch.tex) | [main_ch.pdf](论文/CVPR_HG_mix/CVPR2027_English_1115/main_ch.pdf) |
| 英文补充材料 | [supplementary.tex](论文/CVPR_HG_mix/CVPR2027_English_1115/supplementary.tex) | [supplementary.pdf](论文/CVPR_HG_mix/CVPR2027_English_1115/supplementary.pdf) |

安装所需 TeX 工具后，可在稿件目录运行：

```bash
cd 论文/CVPR_HG_mix/CVPR2027_English_1115
bash build.sh      # 英文正文与补充材料：pdfLaTeX + BibTeX
bash build_ch.sh   # 中文阅读版：XeLaTeX + BibTeX，需要 xeCJK / Noto CJK 字体
```

实验记录与稿件分别维护；最新实验是否已写入论文，以对应收尾记录为准。

## 开发与文件管理

新方法和实验在 `benchmarks/v5/` 中开发；`benchmarks/v4/` 保持只读。协作约定见 [`AGENTS.md`](AGENTS.md)，变更记录见 [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md)。

数据、权重、缓存和运行产物存入 `/nfs-medical3/zyh/fewclick_v2/` 下的时间戳目录。Git 保存代码、配置、实验记录和正式论文材料；可再生编译文件与本地历史归档由忽略规则管理。归档位置与恢复方式见[仓库维护说明](docs/仓库维护.md)。

本仓库从早期 FewClick 工程迁移而来，其中部分基础组件源于 XCellAligner；迁移范围见[迁移清单](MIGRATION_MANIFEST_20260831_020020.md)。
