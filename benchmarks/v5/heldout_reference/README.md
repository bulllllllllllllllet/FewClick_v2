# 相同测试任务的全标签开发参考

这里的all-label指G3当前可用标签池，TIGER缺失ID4不在其中；不是完整穷尽标注或所有生物概念。全标签模型训练和原302-pair开发集选模均接触被评价类别。冻结heldout测试图像/点击/阈值/指标完全相同，但新留出模型训练任务分布和seen-only选模标签不同，后续差值仅作描述，不能单独归因为类别监督删除。

`extract.py`绑定accepted G3/G4真实训练/40k消费/选中权重/parent rows SHA，逐字段核对原任务/点、实际GT分母和Dice代数，逐字节提取原行。本图3方法×3fold及R中SAM-feature/NoCell graphDense/region-prototype×3fold共18表19320行，独立math.fsum重算全部预算pair宏Dice和R absentFPA通过。

`wsi_remote.py`通过进程内入口包装复用冻结heldout_transfer相同WSI两视野feature/prototype/GT禁用推理代码，仅改为明确all-label G3权重绑定和参考role元数据。不是官方native远端decoder，不新增R训练。WSI-SAM选中step6000，10000/40000是完整训练trajectory预算。首版role元数据污染query coverage map，已拒绝并保留source/输出；修正后重新gate，coverage保持query-ID→valid/total格式。旧失败gate不得作通过证据。

`queue_wsi.py`单CPU串行各fold native gate→完整R评分，三fold2300/1300/1240行已完成。`audit_wsi.py`独立全部任务/点/GT/count/Dice/FPA/聚合、source/weight SHA、coverage map及60保存gate mask重算通过。新3表4840行加原提取，共21参考表24160行，不是独立病例数。新留出训练结果仍未完成。

合格汇总：/nfs-medical3/zyh/fewclick_v2/fair_comparison/heldout_1026_003000/all_label_reference_report_1026_030900/report.md。本图和R分表，R有目标Dice/无目标valid区FPA分列；未知255不是阴性。FewClick同encoder原型与graphDense仍有细像素feature访问差异，不当作严格等输入decoder因果消融。记录1026_023000。
