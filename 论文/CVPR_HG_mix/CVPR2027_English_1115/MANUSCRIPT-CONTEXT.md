# Manuscript context

## Identity and scope
- FewClick CVPR2027 English manuscript; fixed HG-mix method. Main text: `main.tex`; supplement: `supplementary.tex`.
- Current deliverable: `main_ch.tex` / `main_ch.pdf`, a complete Chinese reading edition of the English main article, including captions and tables. English `main.tex` remains canonical; Chinese text is in `sec_ch/` and `tables_ch/`.
- Experiment continuity remains in repository `goal.md` and its linked registry; this file records writing decisions only.

## Canonical terminology
- FewClick = the frozen HG-mix method in this manuscript.
- Proto-MS = signed click prototypes/similarities with a lightweight CNN, sharing frozen Phikon-v2 and seven encodings.
- Full-slide Dice and strict-remote Dice are different scoring domains. Present-window Dice is not whole-slide Dice.
- Owner = choose the covering window with the nearest physical center for each pixel; fused = average overlapping probabilities.
- FPA = false-positive area; percentage points describe metric differences.

## Current author state
- [accepted] State the comparative advantage in Dice and target-absent false-positive control in the Introduction contributions. Origin: user supplied `sec_ch/1_introduction.tex` and requested this addition; scoped to the six evaluated methods in Tables 1–2 and mirrored in English.
- [accepted] End both abstracts by stating that target-present segmentation accuracy remains comparable to SANSA alongside the FPA reduction. Origin: user, 2026-11-16, “最后再补一句类似：同时保持了一致的准确率”; expressed as comparable segmentation accuracy rather than a new pixel-accuracy metric.
- [accepted] Lead both abstracts with WSI capability and full-slide/strict-remote gains; emphasize only absent-target FPA reduction for the remote development queue. Origin: user, 2026-11-16, “突出wsi的优势…主要突出误报率下降即可” and “中英文需要同时调整”.
- [accepted] Translate the complete English main article into Chinese, retaining proper names, and compile `main_ch.pdf`. Origin: user, 2026-11-16, “现在把整篇英文稿翻译成中文（专有名词不用） 编译一个 main_ch.pdf出来”.
- [accepted] Write completed findings directly, without defensive endings. Origin: user, 2026-11-16, “当前已有的结论可以补充进论文 不需要后面的防御性措辞”.
- [accepted] Use scientific-manuscript-editor. Origin: user supplied its GitHub repository and requested use, then “继续”.
- [working] Dense transfer has the larger effect among the two tested WSI component removals. Origin: current §4.5; measured comparisons in S4.
- [working] Different spatial aggregation and lower absent-window FPA help explain the larger WSI gap. Origin: current §4.4 and supplement; no quantitative causal decomposition is claimed.
- [working] Learned assignments yield a modest present-Dice gain; fixed-grid large-target Dice is slightly higher. Origin: current §4.5 and S3.

## Evidence roles
- `sampled_attribution_data.json` contains accepted metrics and source hashes; sources remain under `/nfs-medical3/zyh/fewclick_v2/cvpr_research/reviewer_sample_1116_025000/`.
- Main patch results: original 11,000 tasks. S3 controls: shared 1,087-task subset with original within-tissue size weighting.
- Main WSI results and paired intervals: original 15 slides. S4 attribution: first three complete formal WSIs, 32 slide–target pairs.
- Same-feature control fixes backbone, views, encoding budget, and adaptation budget; decoder architecture/capacity differ.
- Patient-test exclusion sentence is author-supplied and retained; it does not redefine the spatial-block training/development split.

## Claim boundaries
- Keep observations and their interpretation distinct; no new mechanism, convergence claim, or module-necessity claim is introduced by prose editing.
- Preserve the lower Proto-MS FPA and the fixed grid's slightly higher large-target Dice.
- Keep three-slide diagnostic scores separate from 15-slide primary scores and intervals.
- The skill repository supplies writing guidance, not scientific evidence or manuscript references.


### 2026-11-16 方法公式与冻结实现核验
中英文方法同步补全坐标范围、可学习尺度嵌入与固定索引、停止梯度及32网格支持概率、粗细对应双端展开与1e-8截断、矩阵维度、确定性与分配损失平均范围；修正指示函数字体。18个相关源码哈希与HG-mix冻结记录一致。密集迁移CPU代数核对通过；没有修改模型或实验结果。细节见 `/nfs-medical3/zyh/fewclick_v2/cvpr_research/paper_method_audit_1116_131014/method_review.json`。


### 图2布局与配色（2026-11-16用户指定）
图2采用两行四列，移除评分域面板，真实标注与支持框合并。真实标注及预测使用浅绿色覆盖，支持框青色。图3维持原状。病例和六方法分数未变；证据 `/nfs-medical3/zyh/fewclick_v2/cvpr_research/paper_figure2_green_1116_140702/acceptance.json`。


### 图2配色加深（2026-11-16最新用户要求）
图2正点击深绿#006400、负点击深蓝#003C9D；覆盖绿色RGB(64,180,96)、alpha0.65，替代上一版浅绿色。符号稍加粗放大，青色支持框及2×4布局保持。


### 图2最新配色（2026-11-16）
按最新用户要求，绿色取前两版中间值RGB(104,209,120)，alpha0.65；正点击正红#FF0000，负点击天蓝#00BFFF。取代此前深绿／深蓝点击配色。


### 图2最新布局与选例（2026-11-16）
图2改为4行8列，组织依次肿瘤上皮、坏死、正常腺体、肌层。保留原肿瘤示例，新增三类各选FewClick整片Dice最高案例，按用户要求展示较好效果；中英文图注明确这一选例规则，替代旧的全部首例规则。列为支持点击、GT+支持框、FewClick、SANSA、SAM、SAM2、WSI-SAM、SAM-Med2D。配色保持中间绿覆盖、正红＋、天蓝×。


### 图2图注最新决定
用户要求删除主图图注中的选例说明句，已中英文同步删除；补充材料与选例元数据保留具体选择规则。


### 表3对齐（2026-11-16）
中英文表3的n与六方法列均居中，表头和数值共享同一列中心；组织名称左对齐。


### 表3最新列宽
按用户要求扩展至正文宽度，组织名称列20%且左对齐，n及六方法列等宽居中；表头单行。


### 表1／表2方法顺序
表2与表1统一为SAM、SAM2、WSI-SAM、SAM-Med2D、SANSA、FewClick，中英文一致。


### 实验正文与3.3标题（2026-11-16）
用户要求减少重复补充转引，用末页空间概括现有证据并突出优势。实验正文仅保留配置的统一补充转引，关键逐片／逐类WSI增益、误报、缓存、同特征控制、提示及大小分层直接报告。统计与消融的实质性边界保留。3.3标题为Prompt-Conditioned Region Hierarchy／提示条件下的区域层级。


### 贡献叙述最新优先级（2026-11-16T22:14:48.355416+08:00，用户明确指定）
依用户最新要求，中英文摘要、引言贡献、方法开头与讨论结论聚焦一个支持图像块的少量正负点击驱动同片整片传播，以及目标覆盖–误检权衡。区域层级定位为组织多尺度上下文的实现手段；方法段落名由可学习区域改为空间上下文池化，保留学习分配与全部公式。引言保留整片Dice与目标缺失FPA优势，结论明确密集迁移提高大目标Dice同时增加FPA，未宣称该模块单独降低误报。实验数据、表格、图片均保持；完整15WSI控制扩展继续运行，本文仍使用原三张控制结果。两版正文8页、总10页，补充6页，无编译警告。
证据：`/nfs-medical3/zyh/fewclick_v2/cvpr_research/paper_propagation_framing_1116_221217/acceptance.json`。此优先级取代此前以可学习区域本身为主的贡献组织。


### 2026-11-16T23:46:14.205023+08:00 LLM声明
依用户要求查询CVPR2027官方LLM Policy，并补全中英文LLM Usage Statement。明确工具ChatGPT/Codex、方法/实验规划、代码实现调试运行、结果分析、文献核查、图表制作及写作翻译的迭代辅助范围，保留作者责任，不虚构逐项人工验收。纠正此前审查遗漏的位置要求：声明须在正文后、参考文献前，已移至第9页。补充对应空白改为主文声明的位置说明。正文前8页提取文本完全不变；两版总10页、补充6页，无编译警告，25引用目标有效。其它数据来源/权限占位与字体/书目问题未在本次LLM声明任务中修改。
依据：https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines （LLM Policy > A. Authors）。用户本轮要求处理声明，取代此前留空待确认的安排。


### 2026-11-17T11:00:43.197339+08:00 用户最终PPTX替换方法图
按用户指定，将figures/最终方法图.pptx通过LibreOffice Impress直接导出为单页矢量PDF并替换figures/method.pdf。源PPTX不变，原图与稿件已备份；页面尺寸与旧图相同，保留上下72bp空白裁切，无内容裁断。中英文重新编译，正文8页、总10页，无编译警告，英文第3页视觉检查通过。同步完整源码ZIP与纯英文Overleaf ZIP。未改动实验或方法文字。
证据：`/nfs-medical3/zyh/fewclick_v2/cvpr_research/paper_method_pptx_1117_105922/acceptance.json`。


### 2026-11-17 讨论结尾补充未来工作
用户明确同意补充小目标缺陷及未来工作，并要求写入TeX。中英文结尾新增高分辨率细化、目标尺寸感知训练、目标缺失误报联合评估，保留同片传播与查询端不追加人工提示。仅为未来方向，无新实验或已实现声明。英文为适应版面压缩为四行；两版编译通过，正文仍8页、总10页，无LaTeX警告或overfull。备份：`history/conclusion_future_1117_155723`。


### 2026-11-17 数据来源与授权说明
[accepted] 用户明确确认“CRC是知情且授权的私有数据”。补充材料第1节新增Data sources and permissions：CRC为私有结直肠癌病理数据，已取得知情同意并获数据提供方授权用于本研究。未推定伦理委员会批准、审批编号、去标识化或数据公开承诺。IRB批准/监督或无需审查的理由仍待事实确认；已在TeX注释保留该未决事项。当前CRC来源为广东省人民医院原200WSI队列，不是SRRSH三病例mIF队列。TIGER来自官方AWS的公开WSIROIS训练ROI及组织标注，补入已核实的2022 Zenodo正式数据集引用与CC BY-NC 4.0说明。来源：https://zenodo.org/records/6014422 及 https://registry.opendata.aws/tiger/ 。删除原第9节占位结构，主文LLM声明保持。补充材料加入参考文献与BibTeX构建步骤。备份：`history/data_permissions_1117_160739`。


### 2026-11-17T16:40:09.547613+08:00 参考文献与图3字体修复
补齐SAM-Med2D可见arXiv来源、PFENet/PANet/SANSA/CyCTR正式页码，PFENet换正式DOI链接、CyCTR换官方论文集链接，TIGER补充文献显示DOI。图3从原CPU绘图脚本以pdf.fonttype=42重绘；4例选择及全部指标、36内嵌图像和PDF文字完全一致，PNG逐字节一致。最终图3字体为CID TrueType，带Unicode映射；三版PDF均无Type3、无编译警告或未定义引用。中英文正文8页/总10页，补充6页，图3及文献页视觉检查通过。无新训练或推理，15WSI新消融结果仍待同步。
备份与验证：`history/references_figure3_1117_163643`。
