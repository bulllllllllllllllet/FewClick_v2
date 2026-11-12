# FewClick 当前论文入口

2026-11-12：用户选定 **HG-mix** 为后续固定研究方法。当前主结果为同片不同patch迁移：整体present类别宏Dice42.1118%、大目标78.4937%、absent误报面积7.7494%。这些数字不代表原稿本图或完整WSI拼接协议。

- [新版写作工作区](CVPR_HG_mix/README.md)：当前为准备阶段，正文待按新方法重写。
- [AAAI旧主稿档案](AAAI27_FewClick_paper/FewClick_AAAI27.tex)：从旧仓库只读复制，保留原文和历史图表，不代表当前HG-mix论文。
- [AAAI旧补充材料](可复现性材料/FewClick_Supplementary_EN.tex)：同样为历史材料。

三条执行方案：

1. [SAM系列公平对照](../docs/建议/数据与评估/方案_HGmix固定基线SAM公平对照_1112_163751.md)
2. [收益归因与最小消融](../docs/建议/模型与训练/方案_HGmix收益归因与最小消融_1112_163751.md)
3. [论文迁移与重写](../docs/建议/模型与训练/方案_HGmix论文迁移与重写_1112_163751.md)

追加必做实验：[一次正负支持点击后的整片WSI Dice](../docs/建议/数据与评估/方案_HGmix整片WSI一次点击分割与评价_1112_165407.md)，纳入对照、消融与论文三线；当前是准备/预审阶段，尚无新模型整片Dice。

固定方法及逐文件复制SHA：`/nfs-medical3/zyh/fewclick_v2/cvpr_research/hg_mix_frozen_1112_163751/{method_freeze.json,paper_migration_manifest.json}`。

原目录 `/home/zhaoyh/FewClick/论文` 保持只读。迁入主稿和补充材料共41文件，约42.26MiB；编译日志/cache、旧代码包和无关文档未迁入。后续新版在独立工作区修改，避免旧结果与新证据混淆。
