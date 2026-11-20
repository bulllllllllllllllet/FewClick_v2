# FewClick_v2：v5 研究代码

当前主线是支持图像块少量正负点击驱动的同片组织检索与整片分割，固定研究方法为 **HG-mix**。项目说明、环境与论文入口见[根目录 README](../../README.md)。

| 模块 | 用途 |
| --- | --- |
| [hg_combination_study](hg_combination_study/) | HG / HG-mix 模型、训练与远端图块评价 |
| [same_slide_training](same_slide_training/) | 同片训练任务清单与空间范围审计 |
| [whole_slide_once](whole_slide_once/) | HG-mix / SANSA 整片推理和唯一像素指标 |
| [sam_current_fair](sam_current_fair/)、[sam_med2d_current](sam_med2d_current/)、[whole_slide_sam](whole_slide_sam/) | SAM 系列匹配适配与整片对照 |
| [hg_paper_ablation](hg_paper_ablation/)、[reviewer_sample](reviewer_sample/) | 核心消融、归因控制与扩展 WSI 评价 |
| [paper_inference_supplement](paper_inference_supplement/)、[paper_figures](paper_figures/) | 补充推理、效率统计与论文图表 |
| [fair_comparison](fair_comparison/README.md) | 公平比较与监督来源约定 |

其他目录保留历史阶段、候选模型和专项诊断的实现。各自的脚本与 README 对应当时的实验合同；当前任务以 [`goal.md`](../../goal.md) 链接的 registry 和运行记录为准。

新实现使用 `benchmarks.v5` 命名空间，保留 `benchmarks/v4/` 冻结基线。数据、权重、缓存及运行产物写入 `/nfs-medical3/zyh/fewclick_v2/` 下的时间戳目录，记录代码版本、输入/权重身份和验收证据。
