# 冻结公共Decoder：唯一连续细胞修订

当前（2026-10-31T16:11:49.317015+08:00）：三臂真实冻结门控及3727 mask独立重放complete；正式三臂唯一修订Pilot running，尚无正式质量结果。

时间：2026-10-31T16:02:52.235680+08:00。阶段P1C，preparing。前一goal turn为progress：两seed首配方22266mask及暴露审计接受，但质量未过，不再晋升。按155213合同，只改变公共参数是否更新，原配方源码/报告保留。

新入口train_cell_only_pilot/accept_cell_only_pilot与cell_only_checks；同508705注册参数，仅cell76800可训练，公参requires_grad=False且grad必须None。原共有optimizer全状态逐字节保留、计数2000不变；新增cell组从0→2门控/1000正式。不能整体no_grad，B/C梯度需穿过固定decoder；A零cell不学习，明确不是充分优化的强基线。初始所有logits与父模型精确相同。

真实门控：三臂各相同8个CRC/TIGER local/present/absent任务，2更新×4，磁盘与optimizer真实继续一步；A额外评价全部3711mask，必须逐位与已接受2000步父模型local906/remote2805相同并GT复算；B/C各8mask。所以门控总3727mask，不把A的评价数当额外训练。门控全部接受才登记正式3×1000×4。

目录`/nfs-medical3/zyh/fewclick_v2/cvpr_research/cell_only_gate_1031_160252`，tmux `fewclick_cell_only_gate_1031_160252`，物理GPU3/CPU2串行；参数/源码/输入/父权重/参考mask与报告SHA冻结见contract.json/source_snapshot，真实worker/命令/日志/EXIT_CODE见stage.json/history.jsonl和各日志。命令train_cell_only_pilot --arm ... --parent-transfer-reference ... --gate ...，后accept_cell_only_pilot --pilot ...及--gate-root。失败停止不重试，不覆盖旧产物。

正式Pilot尚未启动；不改4000任务、学习率.001、norm/PCA、encoder、crop、融合、阈值或六项质量门槛。A为固定父模型的细胞适配控制，公共decoder不学习须披露；本修订不替代P1I有限联合适配/P1C染色敏感性，P2–P6仍未完成。

## 2026-10-31T16:08:23.792716+08:00 三臂冻结门控接受与正式唯一修订登记

门控三臂均独立接受，汇总`/nfs-medical3/zyh/fewclick_v2/cvpr_research/cell_only_gate_1031_160252/independent_gate_acceptance.json`。同508705注册/76800可训练；公共权重和完整optimizer状态逐字节不变，共享计数2000；新cell两步学习（A0梯度/B-C非零），磁盘恢复/真实继续一步完全一致。A全部3711mask与父模型逐位一致且GT指标重放，B/C各8mask重放，共3727门控mask；这不属于额外训练样本。

正式目录`/nfs-medical3/zyh/fewclick_v2/cvpr_research/cell_only_pilot_1031_160823`，tmux `fewclick_cell_only_pilot_1031_160823`，GPU物理3/CPU2串行，三臂各1000×4，公共计数仍2000，新组1000，全局optimizer调用计数3000。gate回执/全部源输入权重SHA已冻结；真实worker/命令/log/EXIT_CODE见stage/history。每臂3711mask独立验收，A仍须全部复现父mask；相同阈值/任务/输入/六项守门不改。完成后做三组配对比较，只有过原门槛才独立第二seed，不自动添加第三配方。本次刚启动，尚无正式质量结果。

## 2026-10-31T16:09:40.642696+08:00 正式合同评价数量元数据澄清

formal contract继承的expected_masks表是gate的A3711/B8/C8；正式模式源码与先登记记录一直要求每臂3711。正式质量结果出现前新增contract_mask_count_addendum.json明确正式三臂均3711，原合同保留；未改变任何模型/数据/训练/门槛/实际执行，不重启。后续验收/比较以正式3711全身份为准。

## 2026-10-31T16:11:49.317015+08:00 接续观察

门控coordinator973206已退出0，真实worker/命令历史见gate/runtime_observation.json与history.jsonl。A3711mask复现父模型、B/C各8mask均接受；公共权重/完整optimizer状态精确保持、new-cell更新/梯度、磁盘与真实继续一步门控全通过。

正式coordinator1003384、worker1003386确认存活；当前image_only/train，观察1000/1000。产物`/nfs-medical3/zyh/fewclick_v2/cvpr_research/cell_only_pilot_1031_160823`、tmux fewclick_cell_only_pilot_1031_160823；stage/history/log/EXIT_CODE是接续入口，不重复启动或修改冻结源码。正式每臂3711，gate旧表的元数据澄清见contract_mask_count_addendum.json（质量出现前登记，无运行改变）。

当前实质配方预算已用2次：初次共同更新、这次冻结公共参数；seed重复不是新配方。正式全三臂accepted_cell_only_pilot后实现冻结政策专用配对比较，验证A3711预测恒等parent，且公共state计数2000/new组1000；沿用六门槛和query-space条件区间。通过才独立第二seed；不通过停止此残差配方，禁止第三次LR/数据/表示混改盲扫。

P1I有限联合适配/P1C染色敏感性继续待办；当前只接受工程门控，未接受新细胞组织收益，P1整体仍未complete。

## 2026-10-31T16:22:55.228585+08:00 冻结政策专用配对比较登记

新入口compare_cell_only_pilots --run ... --parent-transfer-reference ... --output ...，CPU2无GPU，目录`/nfs-medical3/zyh/fewclick_v2/cvpr_research/cell_only_analysis_1031_162255`，tmux `fewclick_cell_only_analysis_1031_162255`。等待原coordinator1003384实际存活/退出0且全三臂accepted_cell_only_pilot后才执行；不重启训练。

比较核验共有参数/optimizer冻结2000、新cell1000、相同输入任务/源码/身份/实际消费、全部mask与接受SHA；重新对照A3711父mask及父指标恒等，读取正式mask数量addendum。沿用原六门槛/3预算/逐类NA及query-space2000次条件区间，不把A标为充分训练的强基线。无通过则停止此残差配方而非第三次修订。源码SHA在contract/source_snapshot，真实比较worker/命令/log/退出在stage/worker.log/exit_code。

## 2026-10-31T16:25:58.191870+08:00 P1I单一候选联合适配容量预检查

CPU2只读既定Phikon-v2权重header/config并重新核对P0记录SHA，不加载GPU、不推理、不训练、不新增候选。入口audit_phikon_joint_capacity --output `/nfs-medical3/zyh/fewclick_v2/cvpr_research/phikon_joint_capacity_1031_162558`，记录439权重tensor和候选末block23+最终norm的形状/参数量。FP32参数+梯度+两AdamW矩只作理论存储下界，不包括完整模型/激活/工作区，不能当实际显存已通过；梯度/重载/质量均尚未接受，正式联合适配预算/控制仍待登记。源码SHA=72118d206ff905863641b0fd5a5d9f13e53e644858dae5218303265c0101c697。当前细胞修订队列不变。

## 2026-10-31T22:34:36.179087+08:00 正式结论与接续

三臂完整接受各3711mask，comparison cell_only_analysis_1031_162255/result/comparison.json；macroDice A/XCell/LEMON=.37006975/.33852153/.36242340。A3711mask逐位等父模型，共享optimizer2000不变；两细胞臂未过主门槛，eligible_for_second_seed=[]。停止当前残差家族，第2配方预算耗尽。不得称全细胞encoder否定或WSI晋升。接续原P1I有限联合适配，当前记录 `docs/运行记录/模型实验/运行记录_CVPR_Phikon有限联合适配准备_1031_222610.md`。
