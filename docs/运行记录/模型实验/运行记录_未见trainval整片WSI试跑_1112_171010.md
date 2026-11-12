# 未见train/val的整片WSI一次支持试跑

2026-11-12T17:32:22.017496+08:00 用户授权SAM优先，余卡WSI试跑与估时，快速时扩原数据WSI；本轮允许实施不再只规划。root=/nfs-medical3/zyh/fewclick_v2/cvpr_research/whole_wsi_current_1112_171010；module benchmarks.v5.whole_slide_once.run，GPU3/2/5与sam0/1/4共享总6worker资源池。HGmix/SANSA当前9500已训练权重严格冻结，支持3正2负同一patch，其他patch无人类点；换WSI重置支持。

## 输入与接受

unseen_audit.json核对原cohort200(140train30val30test)、原完整patchindex、高分辨率catalog、当前1737catalog、实际9500及11000开发，30test均无train/val/当前开发成员交集；历史AAAI测试暴露仍明报，不加患者互斥。每片原索引64窗的GT→native坐标取整回放通过。来源患者字段取原cohort，修正旧confirmation模板残留的无效patient/spatial字段（不变HE/提示/模型）。cases.json沿用既有冻结support方案按SHA选择，无queryGT/预测挑提示。15片152个可用组织任务，前三片每片固定一个0/1/10优先合格组织做完整工程试跑。

真实支持RGB/GT、local/cache-full精确；内区/边缘/角落的padding与remote/cache-full精确；重叠概率融合与packbits回放；全网格覆盖权重>0；整片先累加唯一像素TPFPFN、落盘mask独立条带重评分；remote排除所有完整4x触及支持的tile输出，整片与remote分别统计。只报告合法标注域，并保留未知像素/覆盖。每case读取原始HE/GT完整SHA，mask/报告/坐标/权重绑定；支持GT仅工程定位回放，queryGT只在已固定strip预测后评分。BF1未实现，不能标整篇系统实验全部complete。

## 调度与估时

先3个WSI worker分别完成HGmix/SANSA工程及整片试跑，实际PID/log/EXIT在workers.json及pilot日志；最多3WSI与3SAM并行，不杀他人。神经encoder输出HGmix每querytile跨组织复用，流式处理整片，避免每目标重复编码；SANSA旧encode仅帧预处理，native编码成本仍计decode，不暗示其具备该神经缓存。

3片全部接受后给出实测15片完整152任务及剩余15片的ETA。为保证SAM优先并避免6槽资源抢占，正式15片等待SAM三臂终态后启动；若额外15成对全组织估计<=12h且磁盘满足1.5倍packed mask余量，自动扩原始30test，不扩到train/val。失败停本批，不改配方、不自动重试，旧产物保留。当前preparing，尚无整片质量。

2026-11-12T17:51:03.908438+08:00 SAM三组四实际train工程/256功能gate/fresh4均accepted，正式2375运行且真实PID存活：{"sam": {"status": "formal_running", "gpu": 0, "worker_pid": 1299712, "engineering": "accepted", "functional": "accepted", "fresh4_smoke": "accepted", "formal_step": 576, "total": 2375, "recent_seconds_per_update": 2.233260967284441, "remaining_train_hours": 1.1160101333735304, "cmd": ["/home/zhaoyh/miniconda3/envs/aligner/bin/python", "-m", "benchmarks.v5.sam_current_fair.train", "--root", "/nfs-medical3/zyh/fewclick_v2/cvpr_research/sam_current_fair_1112_171010", "--arm", "sam"]}, "sam2": {"status": "formal_running", "gpu": 1, "worker_pid": 1312291, "engineering": "accepted", "functional": "accepted", "fresh4_smoke": "accepted", "formal_step": 376, "total": 2375, "recent_seconds_per_update": 2.7900233083963393, "remaining_train_hours": 1.5492379426345229, "cmd": ["/home/zhaoyh/miniconda3/envs/aligner/bin/python", "-m", "benchmarks.v5.sam_current_fair.train", "--root", "/nfs-medical3/zyh/fewclick_v2/cvpr_research/sam_current_fair_1112_171010", "--arm", "sam2"]}, "wsi": {"status": "formal_running", "gpu": 4, "worker_pid": 1277638, "engineering": "accepted", "functional": "accepted", "fresh4_smoke": "accepted", "formal_step": 1526, "total": 2375, "recent_seconds_per_update": 1.028030605018139, "remaining_train_hours": 0.2424438843501111, "cmd": ["/home/zhaoyh/miniconda3/envs/aligner/bin/python", "-m", "benchmarks.v5.sam_current_fair.train", "--root", "/nfs-medical3/zyh/fewclick_v2/cvpr_research/sam_current_fair_1112_171010", "--arm", "wsi"]}}。HGmix三WSI内区/边缘/角落/cache-full/历史支持RGB-GT/融合pack工程全部accepted，GPU3/2/5真实PID存活，当前HG整片推理仍running；实测约[1.051, 0.871, 1.106]秒/窗，完整配对先估2–4h，尚无整片Dice。SAM-Med2D官方native CPU/真实backward接受并已排队首三SAM终态后，未GPU训练。整片15片全152目标需等SANSA实测估时，可能1–3天，不将窗口数当任务数；快才扩原test30。snapshot=/nfs-medical3/zyh/fewclick_v2/cvpr_research/sam_current_fair_1112_171010/progress_verified_1112_175103.json。
