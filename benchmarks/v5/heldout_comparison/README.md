# 概念留出监督协议

`audit.py`预先定义necrosis、adipose、lymphoid-related三类本体fold，实际重读1737个标签SHA，并检查原40k任务的正目标、正负点和loss区域触及。按TIGER官方定义保守排除辅助标签：原位肿瘤可能包括坏死，rest包括脂肪等复合组织，相关stroma与淋巴形态不等同于独立新标签。官方主站直接抓取403，定义依据主站搜索索引与本地固定标签映射，不声称已存全页。

实际覆盖审计与12个masked-GT代表点probe通过；`supervision.py`将排除标签在训练/seen验证scorer中变255，保留RGB/cells和原GT文件不变。`gate_supervision.py`用fresh COCO/noJ5/image-only FewClick实际forward和原mask/point/assignment loss验证12例：改变ignored truth不改变loss，ignored输出梯度0。不能据此称RGB内的该形态不可见或输入像素梯度0；上下文/自监督assignment仍接触图像。

本轮只完成监督可行性与适配器门控。新fold任务、正负点、训练与seen-val选模必须重新版本化生成，旧40k不能仅去掉正目标后复用。训练/seen选择都不能把heldout像素当负类；heldout GT只在最终评分端读取。

full4x形态完全排除尚未认证。fine-clean数量只是上界；TIGER复合标签和原未知255不能证明组织不存在。主要FewClick用不含XCell语义先验的fresh image-only控制；公共SAM/WSI-SAM先验另记，不宣称所有预训练都未见过该类。闭集已知class-ID参考不作为新类失败对手。

记录：`docs/运行记录/数据审计/运行记录_三概念留出本体与监督排除覆盖_1026_003000.md`。全量训练和未见类精度未完成。

`prepare.py`生成独立masked catalog/seen选模/40k recipes，以及原GT和固定旧点击的heldout local/remote终评合同。不能把终评合同直接交给原evaluate_checkpoint绕过训练协议SHA；需要单独绑定训练合同的评估适配器。

`point_pipeline.py`验证原builder真实PID、summary和EXIT0后，依次运行三fold各64任务pilot与`audit_prepared.py`，全部通过才生成各40000实际训练点并再次审计。点审计逐像素核对1737标签的精确屏蔽、原GT保留、RGB/cells/空间字段不变、recipe成员、seen/训练点符号和原测试点序列。源码在队列contract中冻结，失败保留且不重试。当前坏死fold实际64点与完整协议审计通过；全fold/完整实际点仍待队列验收，不代表模型训练完成。

最新状态：首版三fold64点全部通过，但完整采样均失败；屏蔽后像素数足够不保证原2µm内部可采样。`geometric_candidates.py`按原picker最大5+5预算审计全候选池，N/A/L分别剔除2/3/1个训练pair，验证池不减少，129例独立原resolver回放一致。新geometry版协议/点击队列运行中，旧部分点击禁止作为完整训练输入。

`training.py`仅为fresh image-only FewClick及SAM/WSI-SAM包装原共享trainer和FoldData，把fold/exclusion/source身份写checkpoint合同。`evaluate.py`独立绑定seen训练合同、完成训练run及原GT终评合同，只读取seen-best权重并检查history/path/SHA。`gate_checkpoint.py`实际CPU head/joint两更新、三视野主干变化、fresh权重/optimizer/RNG及两域heldout预测精确重放通过；另拒绝错误fold、训练SHA、masked测试GT、测试选模和gate冒充正式结果。CPU harness显式保存CPU RNG，第一次误调用CUDA-only save失败原样保留。完整GPU训练/终评CLI和最终精度仍待原生门控与运行；CPU门控不替代它们。

最新geometry版本全部三fold40000实际点与完整GT/seen/测试点审计已通过EXIT0；旧失败版本保持禁止训练。`pipeline.py`真实生成每fold原实际点字节不变的8任务门控集（4更新，每次两域），列出3fold×3method的9条件计划；独立清单核对通过。正式coordinator349631等待raw02/03完整结束+EXIT0和GPU4/5真实空闲，按条件固定GPU执行gate→gate-checkpoint终评门控→全新初始化10k/40k→seen-best完整heldout local终评；阶段间再查空闲，源码锁定，失败不重试。GPU1/3留G6，0/2禁用。当前只是队列等待，原生GPU/训练未开始；remote另需适配/执行与独立统计。记录1026_012700与formal_queue_live_audit_1026_013200.json。
