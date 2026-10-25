# MultiverSeg双标注轨道全量迁移

2026-10-25 15:14:29，G4 preparing/running，已有官方v1原生context门控，不修改已运行的稀疏评分源码。

实际审计：原935远端对×5点组，212支持视野。187支持完整标注，25含ignore；完整支持mask参考仅837对可用，98对须排除。点→SAM预测mask→MultiverSeg轨道覆盖全部935对、4预算、18700行。完整GT参考每对一次，837行，不虚构点击曲线、不将同一预测重复20次作为新信息。比较完整GT参考时必须将其他方法限制在同837对，再对其5点组平均；不能直接把参考837对与稀疏全935对排名。

源/权重：`transfer_1025_131856/official_sources_1025_132600.json`、`multiverseg_weight_1025_132506.json`、独立dependencies_1025_132830，沿用已核验3个官方commit与v1 SHA；生成器为同SAM-public ViT-B。MultiverSeg灰度128与真实source mask，query人工点0。不用query类别ID或GT。完整GT支持262144已标像素与少量点明确分账。

先审计并保存每个支持图的有效像素数/标签SHA、每个pair full-GT资格；不按预测成绩删样本。推理使用只读RGB的独立TransferData子类，labels/supervision方法显式禁止，支持与query标签路径都不可用；完整GTmask在独立benchmark端先取得，再作为允许的source输入。预测mask只来自source点，无source/query组织GT。

门控：真实CPU4线程，CRC/TIGER×present/absent各4预算共16稀疏行，另4条完整GT参考；原生MultiverSeg wrapper逐值比较，SAM缓存mask与原生重新编码比较，已保存概率重算TP/FP/FN。成功后全量CPU命名tmux，CUDA隐藏、源码/输入/权重SHA、真实PID/进度/EXIT_CODE记录。先生成mask并缓存，用图像/点/权重SHA身份复用；不在query挑最优mask。运行模块 `benchmarks.v5.transfer_comparison.evaluate_context`。

验收：18700+837完整行、native gate20行、GT隔离、完整mask资格覆盖，方法分轨统计与来源依赖；近期公共先验/128分辨率差异披露。准备过程不是已完成精度；固定4个CPU线程的缓存次序时延不作GPU/WSI效率结论。G3受控迁移权重、独立seed、TLS/WSI和审稿证据仍待。

## 实际门控与正式运行

初始 `gate_context_stream_1025_151846` 真实20行exit0；随后加入官方source/weight manifest、依赖与生成器身份核对及生成概率有限性检查，重新在新目录 `gate_context_locked_1025_152208` 完成同20行exit0。不是失败重试，也没有更换算法；最终代码SHA68856bf025eaa40bea03f0a4de54330d6ad1e0ff8d6eba1dd5e33465f3556475。原生MVS forward逐值一致、SAM缓存/原生最大误差0、GT禁用输入对象、保存概率直接计数通过。

正式输出 `transfer_1025_131856/context_full_1025_152401`，tmux `fewclick_multiverseg_context_1025_152401`、真实workerPID1444317。4CPU线程、GPU隐藏。外层脚本/log同名，退出追加EXIT_CODE；ps真实存活、stage predicted_sparse已500/18700（约15:31）。完整GT837行会在稀疏轨结束后执行。source_contexts与逐行结果持续保存，支持覆盖与98排除pair已落盘，不仅保存在内存。

## 同期完成结果与边界

- Retained-M3两条全量各18700已完成，主present Dice graph .469993、same-feature prototype .343504；absent FPA .045580/.369965。相同已训练feature上的诊断，含历史J5监督，不是重训架构归因。独立18700×2行身份/点/GT/Dice/FPA回放通过，见 `retained_and_SAM_acceptance_1025_152840.json`。
- 上述acceptance同时核对SAM-FT真实10k/40k坐标SHA与最佳checkpoint SHA；四个交互适配模型均训练完成，SAM完整评分仍运行，闭集参考仍训练。
- 受控Dense image-only与XCell完整6040评分配对报告位于 `protocol_1025_124208/controlled_cells_pair_1025_152300`。75来源space bootstrap2000，主Dice.564044/.572893，差.008849、95%CI[-.003059,.022545]含0。不能据此声称稳定细胞增益，更不能隔离分子监督贡献；形态/计数及独立seed仍待。
- MultiverSeg全量尚未完成，不引用20行gate精度作方法排名。row latency只含MVS预测、未含SAM生成/IO，完整wall time另记；不是GPU/WSI效率结论。
