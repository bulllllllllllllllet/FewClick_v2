# FewClick CVPR 2027 英文正式稿（实验图表占位版）

入口：`main.pdf`；独立补充材料：`supplementary.pdf`。源码入口为 `main.tex` 和 `supplementary.tex`。

- 根据 `../中文初稿_1114.md` 翻译并按CVPR结构整理，沿用原AAAI稿的“问题—动机—设计—验证”叙述方式。保留当前HG-mix的方法与结果，未迁入旧AAAI稿的细胞学习、旧数据或旧结论。
- 使用已下载的官方CVPR2027-v1模板，10pt、letterpaper、双栏、匿名review模式。`cvpr.sty`和`ieeenat_fullname.bst`保持原版。
- 当前正文及图表占位共8页；第9页为参考文献和留白的LLM声明。补充材料独立4页。
- 已嵌入用户指定的 `../figures/最终方法图.pptx` 导出的PDF；仅在LaTeX中裁去上下白边。实验表1–5和图2–3只保留空框、编号、标题与引用。正文中的既有数值与分析保留。
- 补充材料表S1–S9同样保留占位，完整参数、逐类结果、效率、提示敏感性、目标大小、边界、bootstrap与选例细则已在文字中说明。
- `experimental_table_data.json`保存中文稿14张原始表格的所有数值，后续填表无需重新提取结果。
- 数据来源及CRC/TIGER正式引用、标注审批/使用许可、LLM声明继续留白。未虚构数据来源、作者身份、审批编号或许可。
- 未启动训练或推理。中文原稿、AAAI原稿与官方模板备份均未修改。

## 编译

本机运行 `bash build.sh`。脚本编译正文、BibTeX和补充材料；`texmf/`随包提供此次本机缺失的标准TeX依赖，避免依赖临时目录。Overleaf中上传源码包，以 `main.tex` 为主文件即可；切换为 `supplementary.tex` 可生成独立补充材料。

## 文件组织

- `sec/0_abstract.tex`：摘要。
- `sec/1_introduction.tex`、`sec/2_related.tex`：引言、相关工作。
- `sec/3_method.tex`：完整方法，包括融合、层级区域、提示编码、点修正、概率迁移、训练与WSI推理。
- `sec/4_experiments.tex`：实验协议与分析；图表占位集中在此。
- `sec/5_conclusion.tex`：讨论与结论。
- `sec/supplement.tex`：独立补充材料。
- `figures/method.pdf`：已确认方法图的可打印版本。
- `references.bib`：当前引用的BibTeX。
- `translation_audit.json`：来源哈希、章节对应、页数、编译检查与已知留白。
- `preview/overview.png`：全文页面缩略预览。

## 结果表位置映射

中文4.4→正文表1（远端）；4.5→表2（本图）；4.6→表3、图2（WSI）；4.7→表4（表征与结构消融）；4.8计时→表5；4.8案例→图3。其余明细及4.9逐类结果迁入独立补充材料。文内保留任务规模与协议的限定，不把远端开发队列称为盲测。

## 引文核对

原AAAI BibTeX复用HookNet、SAM、SAM-Med2D、WSI-SAM及Set Transformer条目，新增并核对Phikon-v2、SAM2、SANSA、HSNet。核对来源：

- https://arxiv.org/abs/2409.09173
- https://arxiv.org/abs/2408.00714
- https://arxiv.org/abs/2308.16184
- https://proceedings.mlr.press/v254/liu24a.html
- https://proceedings.neurips.cc/paper_files/paper/2025/hash/ac6ab84809cef04dbe11e9b80d63c281-Abstract-Conference.html
- https://openaccess.thecvf.com/content/ICCV2021/html/Min_Hypercorrelation_Squeeze_for_Few-Shot_Segmentation_ICCV_2021_paper.html

所有实验图表填入后需重新编译确认页数；当前文件是完整英文审阅稿，审批、数据引用与声明尚未定稿。


2026-11-15：实验图表第一批已填入；当前状态见 图表完成清单_1115.md。原“全部实验图表留白”描述为初始版本状态。


最新：补充材料8表已全部完成，正文新增6方法逐组织WSI Dice表；状态见图表完成清单_1115.md。


## 2026-11-16T11:08:59.065467+08:00 抽样归因结论入稿

已将完成的同Phikon-v2/同视野/七编码Proto-MS基线、fixed-grid重训、三片完整WSI三读出及核心消融写入正文和补充材料。正文为8页，参考文献等2页；补充材料6页。新增表S3（1087任务）、S4（三片WSI五方法）和S11（原15片逐片配对差值）；原11000任务及15片主表保留。原15片整片/strict-remote增益95%区间为[7.62,13.56]/[8.02,13.87]pp。训练病例隔离声明保留。已完成事实与效果直接陈述，不添加未做实验的防御性段落。

为保持正文8页，将训练清单、完整跨尺度消融配置及原anchoring/coarse-prior诊断集中到补充材料，精简重复计时、分层与案例描述。数字由接受JSON生成，数据快照`sampled_attribution_data.json`，核对记录`sampled_attribution_update.json`。本次仅编辑及编译，无新训练/推理。


## 2026-11-16T12:02:17.601078+08:00 scientific-manuscript-editor 写作技能润色

已安装并应用用户指定的 `scientific-manuscript-editor` v1.9.0（https://github.com/deathcats4/scientific-manuscript-editor）。范围为新增同特征对照、固定区域和WSI归因段落及其补充内容；按当前英文稿语体修改，不将技能内容作为论文科学证据。

- Proto-MS先介绍预测方式与控制条件，再给出Dice及FPA对比，删去重复控制条件的结尾。
- WSI段落明确窗口均值与整片唯一像素累计的区别，保留“helps explain”的解释强度。
- fixed-grid维持小幅增益定位；模块消融与large-target结果的联系表达更简洁。
- 表S3补足目标面积阈值、FPA和指标方向；表S4补足分层平均定义。
- 新增`MANUSCRIPT-CONTEXT.md`，区分作者决定、实测证据和当前解释。

两份正文源文件的全部数字token、标签、引用及交叉引用保持一致，表格数据体逐字一致。正文仍8页（PDF总10页），补充6页；编译无警告，表S3/S4版面检查通过。旧版本备份`history/scientific_editor_1116_115751`，核对记录`scientific_editor_update.json`；本次没有新训练或推理。


## 2026-11-16T12:14:53.606139+08:00 完整中文阅读版

`main_ch.pdf`为当前英文`main.tex`的完整中文译稿，共10页（正文8页，参考文献与声明2页）。包含摘要、引言、相关工作、方法、实验、讨论与结论，以及三幅图的中文图注和五张中文表格。模型／方法专名、数学符号、原始图内标记和参考文献条目保留。英语正文及PDF未修改；本次未翻译独立的英文补充材料。

源码入口`main_ch.tex`，中文章节`sec_ch/`，中文表格`tables_ch/`。使用XeLaTeX、xeCJK与系统Noto CJK字体，运行`PATH=/home/zhaoyh/TinyTeX/bin/x86_64-linux:$PATH bash build_ch.sh`生成。103个章节／段落／图注块已逐一对应，行间公式逐字一致，行内公式、数值、引用、标签和交叉引用核验通过，所有表格数字行保持一致。无缺字、未定义引用或溢出警告；第1、3、7、8页视觉检查通过。核对记录`chinese_translation_audit.json`。


## 2026-11-16T12:47:55.601629+08:00 中英文摘要同步突出WSI

英文与中文摘要均以单个支持图像块的三正两负点击完成同片整片分割为主要能力，先报告15WSI/152任务的整片43.75%、严格远端40.98%及相对SANSA的+10.52/+10.86pp。11000远端开发任务仅报告目标缺失FPA从14.79%降至7.75%（-7.04pp），摘要不再展开该队列的Dice及消融数字；正文实验内容保留。中英文数值一致，编译均无警告，均为8页正文+2页参考文献等。备份`history/abstract_wsi_1116_124636`，核对`abstract_wsi_update.json`。
