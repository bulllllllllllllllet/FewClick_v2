# SAM2原生解码FewClick F

用户2026-11-11明确优先赶超SANSA，授权尝试原生SAM2 decoder路线。F单臂GPU3，>=12288MiB可共享；不启动C独立对照或额外seed。准备中，不等于功能学习/质量接受。沿用FewClick-v2、training-quality技能和现有空间公平合同，替代旧当前接续指针。

## 假设和结构

现有Few大目标73.8085/SANSA77.2201、FPA6.7115/14.7866；B/C/D未晋升，覆盖和边界仍弱。尝试SAM2.0 Hiera Large公开encoder+原生高分辨率mask decoder，FewClick三视野区域/层级上下文/正负点SignedMatcher生成coarse logits。单共享encoder分别处理fine/2x/4x，原2048→256头接口用256通道重复8次/sqrt8确定性升维；cell分支保持image-only。SAM原生fine高分辨率两层及deep+no_mem_embed与官方predictor核对。

native decoder输入：coarse logits双线性256x256送入公开SAM prompt encoder；remote query没有稀疏点、框或GT，support/local额外使用原任务实际正负点（归一化坐标×1024），单mask，不选GT候选。native输出双线性512与coarse logits固定相加，维持组织缺失证据通路；融合系数固定1，不按val调整。最终阈值0.5。不使用SAM2时序memory、不跨query更新状态。此为新FewClick衍生系统，不称旧架构单模块消融；同时更换encoder与增加native解码，旧SAM2.1 encoder-only阴性不是本轮匹配控制。

所有SAM2公开权重冻结，包括image encoder、prompt encoder和mask decoder；其forward必须对dense条件可微，梯度训练FewClick原region/层级/提示/matcher参数。公开SAM2.0权重与SANSA底座来源相同，但不加载SANSA Universal额外权重/adapter；不是相同全部预训练或容量。构造器暂沿用原Parent载入Phikon后完全替换；Phikon不参与任何forward、训练或输出state，构造依赖仍记SHA。冻结encoder节省反向，不意味免费编码；WSI时延/缓存成本是后续验收缺口。

## 不变合同与门控

固定seed20261103；train同9500消费：4000local+5000CRC同片present+500absent；同2375updates，每步4任务，AdamW lr1e-4/wd1e-4，原episode_loss（query BCE/Dice/boundary+remote支持mask .25+支持click .1+assignment正则）。本图和同片迁移分开，source/query full4x disjoint，保留spatial-block而非patient-disjoint。

工程：四组真实train输入CRC local/present/absent及TIGER local；official predictor feature/native mask精确；cached/full更新前后精确；原生decoder对dense条件非零梯度、SAM参数无梯度且真实更新后全state不变；matcher真实学习梯度、正负交换依赖。架构改变，不要求最终输出与Phikon旧模型一致。

工程通过→固定32train、256更新功能gate（沿用present Dice>=.2、std>=.02、swap中位>=.001；absent FPA<=.1）+fresh模型/optimizer继续精确及最终32预测重放→4步smoke→独立fresh2375/9500→6例同片evalgate→两个5500分片及独立mask重评分。门控成功仅证明基本可学习，不是击败SANSA。固定最后checkpoint，完整11类/全部大小与absent均保留。质量晋升须与original/SANSA完整比较并保留原误报守门；若Dice提高但误报恶化则报告tradeoff而非自动晋升。

失败原样保留并停止后继，不自动变更门槛/损失/融合权重或扫描seed。原已暴露开发集上结果不当独立确认集。

## 运行

模块`benchmarks.v5.native_sam2_study.run --root ROOT --arm F`；root=/nfs-medical3/zyh/fewclick_v2/cvpr_research/native_sam2_1111_213350；tmux=native_sam2_F_1111_213350；aligner环境。root contract记录sources/weights/消费SHA，source_snapshot保留启动源码；F/{controller_stage,stage,evaluation_stage}.json记录真实PID、命令、log/EXIT。通过门控自动串联后继，无需再次确认。

2026-11-11T21:39:04.098978+08:00 初工程四组complete_accepted/EXIT0：与官方predictor全部native特征及原生logit精确，cached/full更新前后精确，原生dense梯度非零，SAM全state不变。发现复用trainer收据decoder_freeze_applicable仍false；实际检查已执行且通过，仅修正文档化字段为true。旧controller1002926及其自有workflow1007228/functional1007524显式停止，未进入正式训练；旧root完整保留metadata_revision_stop.json。新root=/nfs-medical3/zyh/fewclick_v2/cvpr_research/native_sam2_1111_213802，新tmux=native_sam2_F_1111_213802，重新工程→原冻结学习/正式/评价流水线，不改变模型/配方/门槛。

2026-11-11T21:41:28.781135+08:00 F新root工程四组accepted/EXIT0，官方predictor特征/native logit精确，native条件梯度和SAM全冻结验证；fresh optimizer继续精确检查accepted。GPU3功能学习256 gate真实PID1016418现场存活，尚非学习质量通过/正式训练完成；通过自动fresh4→fresh2375/9500→11000同片独立评分，失败停止。源码隔离commit=60156bf59e2c7f766c34d79c337bb89772c77a2a，launch_verified/quality_targets已保存；root=/nfs-medical3/zyh/fewclick_v2/cvpr_research/native_sam2_1111_213802。

2026-11-12T00:46:10.326874+08:00 F功能256门控accepted，fresh正式2375/9500训练complete_accepted/EXIT0；现场消费SHA与公共队列一致、checkpoint SHA核对。SAM全冻结与fresh4重放通过，6例同片evalgate接受。GPU3实际推理PID1602057现场存活，进度1550/11000；完整质量尚无，不用部分预测/8例train指标判定追平SANSA。预计余推理+独立评分约3.7h（含磁盘评分另留余量）。保留当前流水线，不重启/新增训练。

2026-11-12T02:03:15.544032+08:00 用户询问F失效备选：完成条件路线建议，先完整F及coarse/native/fused诊断，停用则原Few+支持锚定query自适应优先；层级稠密匹配/真实高分辨率按错误证据后置。区分A/B/D已试阴性与新机制，SSP/HDMNet原文核对，不承诺反超、不追加训练。record=docs/建议/模型与训练/方案_原生SAM2候选失效后的备选顺序_1112_020315.md。

2026-11-12T11:41:04.215587+08:00 F/E/H/G全部2375/9500及11000同片评价complete_accepted/EXIT0；现场四组消费/权重、八rows/八mask/report/task SHA核验，三指标类宏独立复算一致。{"F": {"Dice": 25.2429, "large_Dice": 53.0666, "FPA": 27.3183}, "E": {"Dice": 39.7551, "large_Dice": 73.6234, "FPA": 6.6113}, "H": {"Dice": 39.969, "large_Dice": 76.5134, "FPA": 8.0987}, "G": {"Dice": 40.1631, "large_Dice": 74.6312, "FPA": 5.7414}}（单位%）。对照original39.1935/73.8085/FPA6.7115，SANSA41.5429/77.2201/FPA14.7866。F当前组合不晋升；E收益小且large未改善；H large76.5134最接近SANSA但FPA8.0987升于原；G整体40.1631/large74.6312/FPA5.7414较均衡。均未Dice反超SANSA，G/H保留候选，待逐类/配对不确定性及成本审查，不自动晋升或新训练。verified=/nfs-medical3/zyh/fewclick_v2/cvpr_research/FEHG_closeout_1112_114004/verified_comparison.json。
