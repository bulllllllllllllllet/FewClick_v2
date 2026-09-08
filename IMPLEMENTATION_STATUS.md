# FewClick_v2 implementation status

## 08-31 02:20:13

- Purpose: 明确 RNA 特征小实验的证据边界与验收标准。
- Changed: `docs/建议/模型与训练/建议_RNA特征小实验_0831_022013.md`
- Validation: NFS 数据路径、H&E、细胞边界和带 `cell_id` 的转录本字段已完成有界抽查 — passed
- Artifacts: 无
- Docs record: `docs/建议/模型与训练/建议_RNA特征小实验_0831_022013.md`（recommendation）
- Commit: included in `Run RNA feature pilot`

## 08-31 02:26:52

- Purpose: 实现并运行 COAD Xenium RNA 特征证据型小实验。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_xenium_rna_pilot.py`; `docs/运行记录/模型实验/运行记录_RNA特征小实验_0831_022417.md`
- Validation: `conda run -n aligner python -m py_compile benchmarks/v5/phase_3_cell_region/tools/run_xenium_rna_pilot.py`；`conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/run_xenium_rna_pilot.py --timestamp 0831_022417` — passed；融合特征 macro-F1/ balanced accuracy 增益分别为 `+0.3987` / `+0.3833`
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_rna_feature_pilot_0831_022417/`
- Docs record: `docs/运行记录/模型实验/运行记录_RNA特征小实验_0831_022417.md`（run record）
- Commit: included in `Run RNA feature pilot`

## 08-31 02:32:09

- Purpose: 增加 RNA 特征行置换负对照，检验此前增益是否依赖正确的细胞—RNA 配对。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_xenium_rna_pilot.py`; `docs/建议/模型与训练/建议_RNA打乱对照_0831_023209.md`; `docs/运行记录/模型实验/运行记录_RNA打乱对照_0831_023421.md`
- Validation: `conda run -n aligner python -m py_compile benchmarks/v5/phase_3_cell_region/tools/run_xenium_rna_pilot.py`；`conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/run_xenium_rna_pilot.py --timestamp 0831_023421` — passed；真实配对相对打乱 RNA 的 macro-F1 / balanced accuracy 下降为 `0.4031` / `0.3868`
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_rna_feature_pilot_0831_023421/`
- Docs record: `docs/建议/模型与训练/建议_RNA打乱对照_0831_023209.md`（recommendation）；`docs/运行记录/模型实验/运行记录_RNA打乱对照_0831_023421.md`（run record）
- Commit: included in `Add RNA shuffle control`

## 08-31 11:18:06

- Purpose: 用冻结 v4 的真实 `CTransPath -> XCellFormer reg-64` 视觉编码器复核 RNA 特征小实验。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_xenium_xcell_baseline.py`; `docs/建议/模型与训练/建议_XCell视觉基线_0831_111806.md`; `docs/运行记录/模型实验/运行记录_XCell视觉基线_0831_111806.md`
- Validation: `conda run -n aligner python -m py_compile benchmarks/v5/phase_3_cell_region/tools/run_xenium_xcell_baseline.py`；`CUDA_VISIBLE_DEVICES=1 conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/run_xenium_xcell_baseline.py --timestamp 0831_111806 --device cuda:0`（tmux）— passed；1,020 个细胞的 XCell `reg-64` 均为有限值，XCell+RNA macro-F1 / balanced accuracy 为 `0.9007` / `0.9088`，正确 RNA 相对打乱 RNA 的差为 `0.5263` / `0.4751`。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_xcell_baseline_0831_111806/`
- Docs record: `docs/建议/模型与训练/建议_XCell视觉基线_0831_111806.md`（recommendation）；`docs/运行记录/模型实验/运行记录_XCell视觉基线_0831_111806.md`（run record）
- Commit: included in `Evaluate XCell visual baseline`

## 08-31 12:14:08

- Purpose: 使用 COAD Xenium 数据提供方的 `annotation` 验证 RNA 对真实细胞类型注释分类的增益。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/export_xenium_annotations.py`; `benchmarks/v5/phase_3_cell_region/tools/run_xenium_celltype_pilot.py`; `docs/建议/模型与训练/建议_真实细胞类型分类验证_0831_121408.md`; `docs/运行记录/模型实验/运行记录_真实细胞类型分类验证_0831_121408.md`
- Validation: `conda run -n muf-clust python benchmarks/v5/phase_3_cell_region/tools/export_xenium_annotations.py --output …/annotations.csv`；`conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/run_xenium_celltype_pilot.py --annotations …/annotations.csv --output-dir …` — passed；941 个细胞、四类注释均跨四空间块存在，正确 RNA 相对打乱 RNA 的 macro-F1 / balanced accuracy 增益为 `0.7026` / `0.6160`。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_celltype_pilot_0831_121408/`
- Docs record: `docs/建议/模型与训练/建议_真实细胞类型分类验证_0831_121408.md`（recommendation）；`docs/运行记录/模型实验/运行记录_真实细胞类型分类验证_0831_121408.md`（run record）
- Commit: included in `Evaluate annotated cell types`

## 09-07 01:05:49

- Purpose: 梳理冻结 v4 的真实训练链路，并形成 FewClick_v2 大规模重训的分阶段方案与前置门禁。
- Changed: `docs/建议/模型与训练/建议_大规模重训总体方案_0907_010549.md`
- Validation: 核查 `benchmarks/v4` Phase 1–6 训练入口、配置、cache 构建器、联调台账及现有 v4 checkpoint 目录 — passed
- Artifacts: 无
- Docs record: `docs/建议/模型与训练/建议_大规模重训总体方案_0907_010549.md`（recommendation）
- Commit: included in `Plan large-scale retraining`

## 09-07 01:21:42

- Purpose: 区分单癌种/多癌种训练要求，并审计当前本地多癌种数据对细胞预训练与组织分割主训练的就绪性。
- Changed: `docs/建议/数据与评估/建议_多癌种数据就绪性审计_0907_012142.md`
- Validation: 对数据目录 CSV 的候选条目、COAD/HCC/OV 的 Xenium/CosMx 实际文件、肺/乳腺/肾/前列腺 Xenium archive 合约及存储规模完成只读抽查 — passed
- Artifacts: 无
- Docs record: `docs/建议/数据与评估/建议_多癌种数据就绪性审计_0907_012142.md`（recommendation）
- Commit: included in `Audit multicancer data readiness`

## 09-07 01:28:47

- Purpose: 对数据集目录 CSV 的全部服务器路径做有界、只读存在性和模态线索审计，筛选多癌种细胞学习候选。
- Changed: `benchmarks/v5/tools/audit_dataset_catalog.py`; `docs/运行记录/数据审计/运行记录_数据集目录全量审计_0907_012847.md`
- Validation: `conda run -n aligner python benchmarks/v5/tools/audit_dataset_catalog.py --timestamp 0907_013108` — passed；`audit.csv` 含 44 条记录，32 条存在、12 条缺失、0 条错误；平台粒度区分后得到 6 个已解压的 Xenium/CosMx RNA+边界候选。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/dataset_audit/dataset_catalog_audit_0907_012847/`; `/nfs-medical3/zyh/fewclick_v2/dataset_audit/dataset_catalog_audit_0907_013108/`
- Docs record: `docs/运行记录/数据审计/运行记录_数据集目录全量审计_0907_012847.md`（run record）
- Commit: included in `Audit dataset catalog paths`

## 09-07 01:32:58

- Purpose: 深度审计 COAD/HCC/OV 的 Xenium/CosMx 六个候选 cohort，验证 H&E、边界、H5AD annotation 和转录本的实际数据合约。
- Changed: `benchmarks/v5/tools/audit_cell_supervision_cohorts.py`; `docs/运行记录/数据审计/运行记录_细胞监督候选深度审计_0907_013258.md`
- Validation: `conda run -n muf-clust python benchmarks/v5/tools/audit_cell_supervision_cohorts.py --timestamp 0907_014551`（tmux）— passed；5/6 cohort 通过，H5AD—边界 ID overlap `0.9996–1.0000`、annotation 覆盖约 `90%`；CosMx-OV 因转录本 Parquet footer 异常降级为 review。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/dataset_audit/cell_supervision_audit_0907_014551/`
- Docs record: `docs/运行记录/数据审计/运行记录_细胞监督候选深度审计_0907_013258.md`（run record）
- Commit: included in `Audit cell supervision cohorts`

## 09-07 01:52:33

- Purpose: 统计可用 cohort 的可证明 section/donor 证据，并将跨癌种 H5AD `annotation` 映射为第一版 broad cell ontology。
- Changed: `benchmarks/v5/tools/audit_cell_ontology.py`; `docs/运行记录/数据审计/运行记录_细胞队列样本与本体审计_0907_015233.md`
- Validation: `conda run -n muf-clust python benchmarks/v5/tools/audit_cell_ontology.py --timestamp 0907_015339` — passed；5 个 contract-passing cohort 各发现 1 个 H&E section，所有已注释原始标签映射至 8 个 broad classes，未映射类别为 0；donor 数被正确保留为 unknown。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/dataset_audit/cell_ontology_audit_0907_015339/`
- Docs record: `docs/运行记录/数据审计/运行记录_细胞队列样本与本体审计_0907_015233.md`（run record）
- Commit: included in `Audit cell ontology`

## 09-07 02:02:23

- Purpose: 审计肺、乳腺、肾、前列腺和皮肤黑色素瘤等其余 Xenium cancer archive 的 H&E 与 cell metadata 合约。
- Changed: `benchmarks/v5/tools/audit_xenium_archives.py`; `docs/运行记录/数据审计/运行记录_其他癌种Xenium归档审计_0907_020223.md`
- Validation: `conda run -n muf-clust python benchmarks/v5/tools/audit_xenium_archives.py --timestamp 0907_020223` — passed；11/11 root 存在，除 Prime lung 的 archive 内部成员读取异常外，其他 archive 的 `cells.parquet` metadata 均可读；累计可读 cells metadata `2,603,858` 行。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/dataset_audit/xenium_archive_audit_0907_020223/`
- Docs record: `docs/运行记录/数据审计/运行记录_其他癌种Xenium归档审计_0907_020223.md`（run record）
- Commit: included in `Audit Xenium cancer archives`

## 09-07 02:13:03

- Purpose: 选择替代 CTransPath 的 v5 病理 backbone，并预注册区分 backbone、细胞训练与 RNA 辅助训练收益的验证设计。
- Changed: `docs/建议/模型与训练/建议_PhikonV2替代CTransPath与验证设计_0907_021303.md`; `IMPLEMENTATION_STATUS.md`
- Validation: 核查冻结 v4 的 `CTransPath-768 -> XCellFormer` 输入/checkpoint 合约、v5 已审计的细胞监督 cohort、Phikon-v2 官方模型卡与 UNI/H-optimus 官方发布信息 — passed；未下载权重、未修改 v4 或启动训练。
- Artifacts: 无
- Docs record: `docs/建议/模型与训练/建议_PhikonV2替代CTransPath与验证设计_0907_021303.md`（recommendation）
- Commit: included in `Plan Phikon v2 evaluation`

## 09-07 02:31:33

- Purpose: 下载 Phikon-v2 到 v5 专用 NFS cache，并实现独立的 Phikon CLS-1024 与 H&E-only cell adapter 接口及真实 H&E GPU smoke。
- Changed: `benchmarks/v5/phase_3_cell_region/src/phikon_v2.py`; `benchmarks/v5/phase_3_cell_region/tools/download_phikon_v2.py`; `benchmarks/v5/phase_3_cell_region/tools/smoke_phikon_v2.py`; `benchmarks/v5/phase_3_cell_region/requirements-phikon-v2.txt`; `docs/运行记录/特征缓存/运行记录_PhikonV2权重下载与Smoke_0907_021752.md`; `IMPLEMENTATION_STATUS.md`
- Validation: `python -m py_compile ...`；v5 overlay `transformers==4.49.0` 下 adapter `[3,1024] -> [3,256]` 有限；`CUDA_VISIBLE_DEVICES=1 ... smoke_phikon_v2.py --device cuda:0 --timestamp 0907_021752 --output-root .../backbone_smoke/runs` — passed，真实 COAD H&E 输出 Phikon `[1,1024]` 与 adapter `[1,256]` 均 finite。
- Artifacts: 权重 `/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2/`（SHA-256 `261ae680…d928569`）；`/nfs-medical3/zyh/fewclick_v2/backbone_smoke/runs/phikon_v2_smoke_0907_021752/`
- Docs record: `docs/运行记录/特征缓存/运行记录_PhikonV2权重下载与Smoke_0907_021752.md`（run record）
- Commit: included in `Add Phikon v2 cell encoder`

## 09-07 02:49:57

- Purpose: 实现并运行 Phikon-v2 与 CTransPath 的冻结 A/B/C backbone gate，验证相同 H&E crop 下的真实细胞类型可线性分性和 H&E 配对打乱负对照。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_phikon_backbone_probe.py`; `docs/运行记录/模型实验/运行记录_PhikonV2冻结BackboneGate_0907_024652.md`; `IMPLEMENTATION_STATUS.md`
- Validation: `PYTHONPATH=.../vendor/phikon-v2 conda run -n aligner python -m py_compile ...run_phikon_backbone_probe.py`；`CUDA_VISIBLE_DEVICES=1 ... run_phikon_backbone_probe.py --device cuda:0 --timestamp 0907_024957` — passed，941 个真实注释细胞输出 CTransPath `[941,768]` 与 Phikon `[941,1024]` 均 finite；Phikon macro-F1 `0.2340`，CTransPath `0.2515`，Phikon H&E 打乱 `0.2310`，不通过候选 backbone gate。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/backbone_probe/phikon_backbone_probe_0907_024957/`; 日志 `/nfs-medical3/zyh/fewclick_v2/backbone_probe/logs/phikon_backbone_probe_0907_024957/run.log`
- Docs record: `docs/运行记录/模型实验/运行记录_PhikonV2冻结BackboneGate_0907_024652.md`（run record）
- Commit: included in `Evaluate Phikon backbone gate`

## 09-07 03:32:11

- Purpose: 实现探索性 Phikon H&E-only adapter 的 label-only、正确 RNA teacher 与打乱 RNA teacher 三臂空间块训练，检验 RNA 是否可蒸馏为 H&E-only 收益。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/train_phikon_rna_teacher_probe.py`; `docs/运行记录/模型实验/运行记录_PhikonRNA教师Adapter探索训练_0907_033007.md`; `IMPLEMENTATION_STATUS.md`
- Validation: `conda run -n aligner python -m py_compile ...train_phikon_rna_teacher_probe.py`；`CUDA_VISIBLE_DEVICES=1 ... train_phikon_rna_teacher_probe.py --device cuda:0 --output-dir ...phikon_rna_teacher_0907_033211` — passed，941 cells、4 spatial folds、3 seeds；paired RNA macro-F1 `0.2388±0.0041`，label-only `0.2352±0.0024`，shuffled RNA `0.2390±0.0040`，未通过 paired-over-shuffle gate。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/rna_teacher_probe/phikon_rna_teacher_0907_033211/`
- Docs record: `docs/运行记录/模型实验/运行记录_PhikonRNA教师Adapter探索训练_0907_033007.md`（run record）
- Commit: included in `Probe Phikon RNA teacher`

## 09-07 12:31:05

- Purpose: 在 H&E TIFF 缺少可读 mpp 的前提下，经验校准冻结 Phikon-v2 的细胞周围上下文尺度，避免固定 128px crop 造成不合适的 backbone 输入视野。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/calibrate_phikon_context_probe.py`; `docs/运行记录/模型实验/运行记录_Phikon上下文尺度校准_0907_123105.md`; `IMPLEMENTATION_STATUS.md`
- Validation: `PYTHONPATH=.../vendor/phikon-v2 conda run -n aligner python -m py_compile ...calibrate_phikon_context_probe.py`；GPU 运行 64/96/128/192/256/384 px 六尺度 — passed，941 cells 的特征均为 `[941,1024]` finite；256px paired macro-F1 `0.2691`，384px paired-minus-shuffled macro-F1 `+0.0226`，旧 128px 为 `0.2340/+0.0030`。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/context_calibration/phikon_context_0907_123105/`
- Docs record: `docs/运行记录/模型实验/运行记录_Phikon上下文尺度校准_0907_123105.md`（run record）
- Commit: included in `Calibrate Phikon context scale`

## 09-07 12:46:30

- Purpose: 用经验校准后的 256px 与 384px 双视野 Phikon 特征复测 H&E-only RNA teacher，检验扩大上下文后 RNA 蒸馏是否可通过。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/train_phikon_multiview_rna_teacher_probe.py`; `docs/运行记录/模型实验/运行记录_Phikon双视野RNA教师训练_0907_124630.md`; `IMPLEMENTATION_STATUS.md`
- Validation: `conda run -n aligner python -m py_compile ...train_phikon_multiview_rna_teacher_probe.py`；GPU 三臂、4 spatial folds、3 seeds — passed，dual-view label-only macro-F1 `0.2481±0.0061`，paired RNA `0.2477±0.0129`，shuffled RNA `0.2458±0.0069`；未通过 paired-over-label-only gate。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/rna_teacher_probe/phikon_multiview_rna_teacher_0907_124630/`
- Docs record: `docs/运行记录/模型实验/运行记录_Phikon双视野RNA教师训练_0907_124630.md`（run record）
- Commit: included in `Probe dual view RNA teacher`

## 09-07 13:33:29

- Purpose: 实现并执行真实 H&E 双视野 Phikon 最后两层 Q/V LoRA 的 RNA teacher gate，验证 encoder 可微调路径及 RNA 蒸馏增益。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_phikon_lora_rna_gate.py`; `docs/运行记录/模型实验/运行记录_PhikonLoRA双视野RNAGate_0907_132238.md`; `IMPLEMENTATION_STATUS.md`
- Validation: `PYTHONPATH=.../vendor/phikon-v2 conda run -n aligner python -m py_compile ...run_phikon_lora_rna_gate.py`；BF16、rank-8、8 epochs 的真实 H&E GPU run — passed，941 cells、4 spatial folds；label-only macro-F1 `0.2485`，paired RNA `0.2453`，shuffled RNA `0.2249`，不通过 paired-over-label-only gate。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/lora_gate/phikon_lora_dualview_0907_133329/`
- Docs record: `docs/运行记录/模型实验/运行记录_PhikonLoRA双视野RNAGate_0907_132238.md`（run record）
- Commit: included in `Evaluate Phikon LoRA gate`

## 09-07 15:08:03

- Purpose: 构建可重放的五 cohort 多 section cell manifest，并用真实 H&E 双视野 Phikon-v2 crop 验证首批可用 Xenium 的坐标和 encoder 数据管线。
- Changed: `benchmarks/v5/tools/build_multisection_cell_manifest.py`; `benchmarks/v5/phase_3_cell_region/tools/smoke_multisection_cell_crops.py`; `docs/运行记录/特征缓存/运行记录_多SectionCellManifest与坐标Smoke_0907_145854.md`
- Validation: `conda run -n muf-clust python ...build_multisection_cell_manifest.py --timestamp 0907_145854 --samples-per-class 8` — passed，5 cohort/288 cells；`CUDA_VISIBLE_DEVICES=1 ... smoke_multisection_cell_crops.py --timestamp 0907_150803 --device cuda:0` — passed，Xenium COAD/HCC/OV 共 176 cells 的 256/384px 特征均为 finite `[N,1024]`；首次运行因本机 `/tmp` 空间不足失败，重跑改用 NFS 临时目录后 `EXIT_CODE=0`。
- Artifacts: manifest `/nfs-medical3/zyh/fewclick_v2/multisection_cell_manifest/multisection_cell_manifest_0907_145854/`; smoke `/nfs-medical3/zyh/fewclick_v2/multisection_cell_smoke/multisection_cell_smoke_0907_150803/`
- Docs record: `docs/运行记录/特征缓存/运行记录_多SectionCellManifest与坐标Smoke_0907_145854.md`（run record）
- Commit: included in `Build multisection cell manifest`

## 09-07 15:13:00

- Purpose: 实现并运行冻结双视野 H&E cell adapter 的 leave-one-cohort-out 多 section gate，以检验小型跨癌种候选的视觉泛化信号。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_multisection_frozen_cell_gate.py`; `docs/运行记录/模型实验/运行记录_多Section冻结CellEncoderGate_0907_151300.md`
- Validation: `CUDA_VISIBLE_DEVICES=1 conda run -n aligner python ...run_multisection_frozen_cell_gate.py --timestamp 0907_151300 --device cuda:0` — executed with `EXIT_CODE=0`；176 个真实 H&E dual-view features、3 个 held-out cohort、3 seeds 的配对 macro-F1/BA 为 `0.1159/0.1224`，低于 shuffled `0.1458/0.1510`，不通过 gate（有效的负结果）。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/multisection_cell_gate/multisection_frozen_gate_0907_151300/`
- Docs record: `docs/运行记录/模型实验/运行记录_多Section冻结CellEncoderGate_0907_151300.md`（run record）
- Commit: included in `Gate frozen multisection encoder`

## 09-07 15:53:00

- Purpose: 审计并应用 CosMx global-to-H&E 仿射坐标契约，验证两套 CosMx 的真实 H&E crop/Phikon 特征，并以五 cohort 重跑冻结跨 cohort gate。
- Changed: `benchmarks/v5/tools/audit_cosmx_he_transform.py`; `benchmarks/v5/tools/apply_cosmx_he_transform_manifest.py`; `benchmarks/v5/phase_3_cell_region/tools/smoke_multisection_cell_crops.py`; `benchmarks/v5/phase_3_cell_region/tools/run_multisection_frozen_cell_gate.py`; `docs/运行记录/数据审计/运行记录_CosMx坐标变换审计_0907_153100.md`; `docs/运行记录/模型实验/运行记录_五Cohort冻结CellEncoderGate_0907_155700.md`
- Validation: transform audit + visual tissue overlays — passed（CosMx COAD/HCC global span 对 DAPI raster 误差约 `0.01%`，112/112 sampled center 转 H&E 后图内）；五 cohort GPU crop smoke — passed，288 cells 两视野 `[N,1024]` 均 finite；five-cohort frozen gate — executed `EXIT_CODE=0`，paired macro-F1/BA `0.1180/0.1309` 对 shuffled `0.1273/0.1260`，不通过 gate（有效负结果）。首次 five-cohort gate 因 smoke metadata 排序不一致以 `EXIT_CODE=1` 停止；修正 stable cohort-order identity check 后重跑成功。
- Artifacts: audit `/nfs-medical3/zyh/fewclick_v2/cosmx_coordinate_audit/cosmx_he_transform_0907_153100/`; transformed manifest `/nfs-medical3/zyh/fewclick_v2/multisection_cell_manifest/multisection_cell_manifest_0907_154620/`; smoke `/nfs-medical3/zyh/fewclick_v2/multisection_cell_smoke/multisection_cell_smoke_0907_154620/`; gate `/nfs-medical3/zyh/fewclick_v2/multisection_cell_gate/multisection_frozen_gate_0907_155300/`
- Docs record: `docs/运行记录/数据审计/运行记录_CosMx坐标变换审计_0907_153100.md`、`docs/运行记录/模型实验/运行记录_五Cohort冻结CellEncoderGate_0907_155700.md`（run records）
- Commit: included in `Audit CosMx H&E transform`

## 09-07 16:19:00

- Purpose: 实现 cohort 分片、可审计的 H&E-only 双视野 Phikon feature cache，并为 64-cell/class 中规模扩展完成真实五 cohort cache smoke。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/cache_multisection_cell_features.py`; `docs/运行记录/特征缓存/运行记录_中规模多CohortFeatureCache_0907_161725.md`
- Validation: `CUDA_VISIBLE_DEVICES=1 conda run -n aligner python ...cache_multisection_cell_features.py --timestamp 0907_161900 --max-cells-per-cohort 4` — passed，`EXIT_CODE=0`；五个真实 H&E shard 各输出 256px/384px `[4,1024]` finite feature、cell identity 和 `COMPLETED.json`，root index `all_complete=true`。
- Artifacts: 64-per-class raw manifest `/nfs-medical3/zyh/fewclick_v2/multisection_cell_manifest/multisection_cell_manifest_0907_161800/`; affine-applied manifest `/nfs-medical3/zyh/fewclick_v2/multisection_cell_manifest/multisection_cell_manifest_0907_161850/`; smoke `/nfs-medical3/zyh/fewclick_v2/multisection_cell_feature_cache/multisection_cell_feature_cache_0907_161900/`
- Docs record: `docs/运行记录/特征缓存/运行记录_中规模多CohortFeatureCache_0907_161725.md`（run record）
- Commit: included in `Cache multisection cell features`

## 09-07 16:28:01

- Purpose: 启动 64 cell/class 的五 cohort 中规模 H&E-only dual-view feature cache，扩大下一轮跨 cohort gate 的真实样本量。
- Changed: `docs/运行记录/特征缓存/运行记录_中规模多CohortFeatureCache_0907_161725.md`; `IMPLEMENTATION_STATUS.md`
- Validation: 前置真实五 shard cache smoke `0907_161900` — passed；完整任务 `tmux fewclick_featurecache_full_0907_162801` 已启动，worker PID `3127975`，初始日志显示 Phikon 已加载；结果待各 shard `COMPLETED.json` 与 root `index.json` 验收。
- Artifacts: 正在写入 `/nfs-medical3/zyh/fewclick_v2/multisection_cell_feature_cache/multisection_cell_feature_cache_0907_162801/`; 日志 `/nfs-medical3/zyh/fewclick_v2/multisection_cell_feature_cache/logs/feature_cache_full_0907_162801.log`
- Docs record: `docs/运行记录/特征缓存/运行记录_中规模多CohortFeatureCache_0907_161725.md`（run record）
- Commit: included in `Launch medium feature cache`

## 09-07 16:28:01 (completed)

- Purpose: 完成并验收 64 cell/class 的五 cohort 中规模 H&E-only dual-view feature cache。
- Changed: `docs/运行记录/特征缓存/运行记录_中规模多CohortFeatureCache_0907_161725.md`; `IMPLEMENTATION_STATUS.md`
- Validation: `tmux fewclick_featurecache_full_0907_162801` 日志 — passed，`EXIT_CODE=0`；root `index.json` 为 `all_complete=true`，五 shard 共 2,266 cells 的 256px/384px `[N,1024]` feature 全部 finite。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/multisection_cell_feature_cache/multisection_cell_feature_cache_0907_162801/`
- Docs record: `docs/运行记录/特征缓存/运行记录_中规模多CohortFeatureCache_0907_161725.md`（run record updated）
- Commit: included in `Record feature cache completion`

## 09-07 16:33:00

- Purpose: 使冻结多 section gate 支持 cohort-sharded feature cache，并运行 2,266-cell 五 cohort leave-one-cohort-out 对照实验。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_multisection_frozen_cell_gate.py`; `docs/运行记录/模型实验/运行记录_中规模五Cohort冻结Gate_0907_163300.md`
- Validation: `CUDA_VISIBLE_DEVICES=1 conda run -n aligner python ...run_multisection_frozen_cell_gate.py --cache-dir ...multisection_cell_feature_cache_0907_162801 --timestamp 0907_163300` — passed with `EXIT_CODE=0`；五 shard identity、`[2266,2048]` finite features 和 split isolation 均通过；paired macro-F1/BA `0.1247/0.1305`，shuffled `0.1264/0.1273`，不通过预注册双指标 gate（有效负结果）。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/multisection_cell_gate/multisection_frozen_gate_0907_163300/`
- Docs record: `docs/运行记录/模型实验/运行记录_中规模五Cohort冻结Gate_0907_163300.md`（run record）
- Commit: included in `Gate cached multisection features`

## 09-07 17:45:00

- Purpose: 归档 docs 下的建议与运行记录，并重放中规模 gate checkpoint 以定位 paired-vs-shuffled 未通过的原因。
- Changed: `docs/建议/{模型与训练,数据与评估}/`; `docs/运行记录/{数据审计,模型实验,特征缓存}/`; `benchmarks/v5/phase_3_cell_region/tools/analyze_cached_multisection_gate.py`; `docs/建议/数据与评估/建议_冻结Gate失败诊断_0907_174000.md`; `IMPLEMENTATION_STATUS.md`
- Validation: `quick_validate.py /home/zhaoyh/.codex/skills/fewclick-v2` — passed；仓库内旧 docs 根目录记录引用已重写；checkpoint-only diagnosis 重放 30 checkpoints — passed，cache identity、`[2266,2048]` finite feature 与 split isolation 均通过，paired/shuffled cohort、class 和 confusion artifacts 已输出。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/gate_diagnosis/multisection_frozen_gate_diagnosis_0907_174500/`
- Docs record: `docs/建议/数据与评估/建议_冻结Gate失败诊断_0907_174000.md`（recommendation）
- Commit: included in `Organize docs and diagnose gate`

## 09-07 18:03:29

- Purpose: 将冻结 H&E 跨 cohort 验证改为低容量、共同标签口径的正则线性 probe，排除高容量 MLP 记忆随机配对与测试缺类的干扰。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_regularized_multisection_linear_probe.py`; `docs/运行记录/模型实验/运行记录_正则线性跨域Probe_0907_175159.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `conda run --no-capture-output -n aligner python ...run_regularized_multisection_linear_probe.py --protocols same_platform_cross_cancer --seeds 20260907` — passed；真实 2,266-cell cache 的 3 个 Xenium held-out split、cell identity、共同类别过滤、train-only C 选择和 paired/shuffled 输出均通过。完整三 protocol 任务在 `tmux fewclick_linear_probe_0907_180000`（worker PID `3500681`）运行中；完成后需检查 `EXIT_CODE`、汇总指标并更新本条。
- Artifacts: smoke `/nfs-medical3/zyh/fewclick_v2/regularized_linear_cell_probe/regularized_linear_cell_probe_0907_175700_smoke/`; full output pending `/nfs-medical3/zyh/fewclick_v2/regularized_linear_cell_probe/regularized_linear_cell_probe_0907_180000/`; log `/nfs-medical3/zyh/fewclick_v2/regularized_linear_cell_probe/logs/regularized_linear_probe_0907_180000.log`。
- Docs record: `docs/运行记录/模型实验/运行记录_正则线性跨域Probe_0907_175159.md`（run record）。
- Commit: included in `Add regularized linear probe`.

## 09-07 18:09:00

- Purpose: 完成正则线性跨域 probe，并记录全 cohort、同平台跨癌种及同癌种跨平台的 paired-vs-shuffled 结果。
- Changed: `docs/运行记录/模型实验/运行记录_正则线性跨域Probe_0907_175159.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `tmux fewclick_linear_probe_0907_180000` — passed，`EXIT_CODE=0`；读取 `protocol_summary.csv` 与 `paired_deltas.csv` 确认 `loco_all` 不通过、同平台跨癌种仅有弱正向迹象、跨平台无稳定 macro-F1 增益。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/regularized_linear_cell_probe/regularized_linear_cell_probe_0907_180000/`; log `/nfs-medical3/zyh/fewclick_v2/regularized_linear_cell_probe/logs/regularized_linear_probe_0907_180000.log`。
- Docs record: `docs/运行记录/模型实验/运行记录_正则线性跨域Probe_0907_175159.md`（run record updated）。
- Commit: included in `Record linear probe results`.

## 09-07 19:39:28

- Purpose: 按用户要求检查 UNI2-h 是否可作为更新的病理 backbone 对照。
- Changed: `docs/运行记录/特征缓存/运行记录_UNI2h访问检查_0907_193928.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `HfApi.list_repo_files('MahmoodLab/UNI2-h')` — passed（仓库元数据可见）；`HfApi.whoami()` — blocked（`LocalTokenNotFoundError`，本机没有 Hugging Face 登录 token，权重受访问条款控制）。未下载权重、未启动模型或训练。
- Artifacts: 无；计划的 smoke 输出根为 `/nfs-medical3/zyh/fewclick_v2/uni2h_smoke/`，待获得访问权限后创建。
- Docs record: `docs/运行记录/特征缓存/运行记录_UNI2h访问检查_0907_193928.md`（access-check record）。
- Commit: included in `Record UNI2h access check`.

## 09-07 19:43:00

- Purpose: 按用户要求复核 UNI2-h 是否有可用的国内镜像。
- Changed: `docs/运行记录/特征缓存/运行记录_UNI2h访问检查_0907_193928.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: 只读 HTTP 检查 — `hf-mirror.com/MahmoodLab/UNI2-h` 最终重定向至官方 Hugging Face 页面；ModelScope 的 `MahmoodLab/UNI2-h/resolve/master/{config.json,pytorch_model.bin}` 均为 HTTP `404`。未发现可下载的合规国内镜像。
- Artifacts: 无下载或缓存产物。
- Docs record: `docs/运行记录/特征缓存/运行记录_UNI2h访问检查_0907_193928.md`（access-check record updated）。
- Commit: included in `Record UNI2h mirror check`.

## 09-07 19:47:00

- Purpose: 因 UNI2-h 无可用合规镜像，改用公开的 Kaiko ViT-L/14-reg4 病理 backbone 作为可复现对照候选，并启动权重下载。
- Changed: `docs/运行记录/特征缓存/运行记录_Kaiko公开Backbone下载与Smoke_0907_194700.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: Hugging Face metadata/README — passed（`gated=false`、约 1.22GB `model.safetensors`、1024-d output contract）；v5-only `timm==1.0.24` overlay — passed（可枚举 `vit_large_patch14_reg4_dinov2`，未改动 `aligner` 内置 `timm==0.9.8` 或 v4）。下载任务已在 `tmux fewclick_kaiko_download_0907_194700` 启动，worker PID `172372`；等待完整权重后真实 H&E smoke。
- Artifacts: 权重目标 `/nfs-medical3/zyh/fewclick_v2/model_weights/kaiko_vit_large_patch14_reg4/kaiko_0907_194700/`; overlay `/nfs-medical3/zyh/fewclick_v2/vendor/kaiko_timm_0907_194700/`; log `/nfs-medical3/zyh/fewclick_v2/model_weights/logs/kaiko_download_0907_194700.log`。
- Docs record: `docs/运行记录/特征缓存/运行记录_Kaiko公开Backbone下载与Smoke_0907_194700.md`（run record）。
- Commit: included in `Stage Kaiko backbone smoke`.

## 09-07 22:10:00

- Purpose: 完成公开 Kaiko 病理 backbone 的断点续传、权重校验及真实 H&E 前向 smoke。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/smoke_kaiko_pathology.py`; `docs/运行记录/特征缓存/运行记录_Kaiko公开Backbone下载与Smoke_0907_194700.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: 官方 CDN `curl -C -` 断点续传 + SHA-256 — passed，权重 `model.safetensors` 校验为 `149c8b21…b83f20e69`，`EXIT_CODE=0`；`CUDA_VISIBLE_DEVICES=1 conda run -n aligner python ...smoke_kaiko_pathology.py --device cuda:0 --timestamp 0907_221000` — passed，真实 518px H&E crop 输出 finite `[1,1024]`。
- Artifacts: 权重 `/nfs-medical3/zyh/fewclick_v2/model_weights/kaiko_vit_large_patch14_reg4/kaiko_0907_194700/`; overlay `/nfs-medical3/zyh/fewclick_v2/vendor/kaiko_timm_0907_194700/`; smoke `/nfs-medical3/zyh/fewclick_v2/kaiko_smoke/kaiko_pathology_smoke_0907_221000/`; logs `/nfs-medical3/zyh/fewclick_v2/model_weights/logs/kaiko_{download_0907_194700,resume_0907_202000,curl_0907_202500}.log`。
- Docs record: `docs/运行记录/特征缓存/运行记录_Kaiko公开Backbone下载与Smoke_0907_194700.md`（run record updated）。
- Commit: included in `Smoke Kaiko pathology backbone`.

## 09-07 22:30:00

- Purpose: 完成 Kaiko 与 CTransPath/Phikon 的同 crop 冻结 backbone 三方 gate，并记录所有失败重跑与有效负结果。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_kaiko_backbone_probe.py`; `docs/运行记录/模型实验/运行记录_Kaiko冻结BackboneGate_0907_221500.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `0907_221500` 与 `0907_222500` 因缺失 Phikon overlay、旧/新 timm 导入顺序停止；`0907_222700` 识别出 CTransPath helper 与 timm 1.0 API 冲突后停止，均无结果。修正为复用已审计的历史 CTransPath `[941,768]` 并逐 ID 验证后，`CUDA_VISIBLE_DEVICES=1 ...run_kaiko_backbone_probe.py --timestamp 0907_223000` — passed，`EXIT_CODE=0`，Kaiko `[941,1024]` finite；Kaiko Macro-F1/BA `0.2265/0.2430` 低于 CTransPath `0.2515/0.2850`，且 Macro-F1 低于 shuffle `0.2425`，不通过 gate。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/backbone_probe/kaiko_backbone_probe_0907_223000/`; logs `/nfs-medical3/zyh/fewclick_v2/backbone_probe/logs/kaiko_backbone_probe_{0907_221500,0907_222500,0907_222700,0907_223000}.log`。
- Docs record: `docs/运行记录/模型实验/运行记录_Kaiko冻结BackboneGate_0907_221500.md`（run record updated）。
- Commit: included in `Record Kaiko backbone result`.

## 09-07 23:41:05

- Purpose: 新增冻结 v4 J5 下游 TLS 的严格 cell-token 置零消融入口，不修改 v4。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_v4_cell_zero_inference.py`; `docs/运行记录/模型实验/运行记录_v4下游TLS细胞置零消融_0907_234105.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `conda run -n aligner python -m py_compile benchmarks/v5/phase_3_cell_region/tools/run_v4_cell_zero_inference.py` — passed；真实 rank-4 TLS WSI 推理 — blocked，冻结入口缺少 `vipa_slide_reader`（日志 `.../v4_tls_cell_zero_smoke_0907_234217/logs/run.log`，`EXIT_CODE=1`）。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/v4_tls_cell_ablation/v4_tls_cell_zero_smoke_0907_234217/`。
- Docs record: `docs/运行记录/模型实验/运行记录_v4下游TLS细胞置零消融_0907_234105.md`（run record）。
- Commit: included in `Add downstream cell ablation adapter`.

## 09-08 00:05:40

- Purpose: 恢复严格 TLS 下游消融所需的 KFB reader，并修正 v5 adapter 的独立 overlap scratch 路径。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_v4_cell_zero_inference.py`; `docs/运行记录/模型实验/运行记录_v4下游TLS细胞置零消融_0907_234105.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `PYTHONPATH=/home/zhaoyh/vipa-slide-reader/src conda run -n aligner python` 导入 `pyvips` + 冻结 `SlideReader` 并读取真实 KFB RGB tile — passed（`70040×71841`）；真实 rank-4 TLS J5 cell-zero run `0908_000540` 已通过模型载入与首批 tile，`4/2162` completed、`failures.jsonl` 为空，仍在运行。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/v4_tls_cell_ablation/v4_tls_cell_zero_smoke_0908_000540/`; worker PID `846921`; tmux `v5_tls_cellzero_0908_000540`。
- Docs record: `docs/运行记录/模型实验/运行记录_v4下游TLS细胞置零消融_0907_234105.md`（run record updated）。
- Commit: included in `Fix cell ablation scratch`.

## 09-08 01:12:59

- Purpose: 完成 v4 J5 TLS cell-zero 下游 smoke，修复已完成输出后 NFS scratch 清理导致的 v5 adapter 假失败。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_v4_cell_zero_inference.py`; `docs/运行记录/模型实验/运行记录_v4下游TLS细胞置零消融_0907_234105.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `v5_tls_cellzero_0908_000540` — 2,162/2,162 tile、空 failures、complete metadata 和三份 TIFF 均 passed；TLS evaluator 单独重放 — `EXIT_CODE=0`。full vs zero 的 held-out censored Dice 为 `0.7612` vs `0.6992`，仅为单 WSI smoke。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/v4_tls_cell_ablation/v4_tls_cell_zero_smoke_0908_000540/wsi_inference_0908_000540/`。
- Docs record: `docs/运行记录/模型实验/运行记录_v4下游TLS细胞置零消融_0907_234105.md`（run record updated）。
- Commit: included in `Record downstream cell smoke`.

## 09-08 02:39:09

- Purpose: 新增 CTransPath 最后一 stage 的 H&E-only RNA 教师空间块训练 gate，作为 v4 同源骨干的受控训练可行性检验。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/train_ctranspath_rna_teacher_gate.py`; `docs/运行记录/模型实验/运行记录_CTransPathRNA教师训练Gate_0908_023909.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `CUDA_VISIBLE_DEVICES=1 conda run -n aligner python ...train_ctranspath_rna_teacher_gate.py --epochs 1 --folds 2 --arms label_only` — passed training path: two real folds completed with finite loss and 31 final-stage parameter tensors tuned; front-end time limit interrupted only final aggregation.
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/ctranspath_rna_teacher_smoke/ctranspath_rna_teacher_gate_0908_154500/`。
- Docs record: `docs/运行记录/模型实验/运行记录_CTransPathRNA教师训练Gate_0908_023909.md`（run record）。
- Commit: `a601b1a` (`Add CTransPath RNA teacher gate`).

## 09-08 02:40:00

- Purpose: 启动 CTransPath RNA 教师三臂空间块训练 gate；RNA 仅作训练监督，所有评估为 H&E-only。
- Changed: `docs/运行记录/模型实验/运行记录_CTransPathRNA教师训练Gate_0908_023909.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `tmux fewclick_ctrans_rna_0908_024000` — running，真实 worker PID `1482398`，GPU 1；已确认 tmux、conda worker 和 Python 训练进程存在。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/ctranspath_rna_teacher_gate/ctranspath_rna_teacher_gate_0908_024000/`; log `/nfs-medical3/zyh/fewclick_v2/ctranspath_rna_teacher_gate/logs/ctranspath_rna_teacher_0908_024000.log`。
- Docs record: `docs/运行记录/模型实验/运行记录_CTransPathRNA教师训练Gate_0908_023909.md`（run record updated）。
- Commit: included in `Launch CTransPath RNA teacher gate`.

## 09-08 03:01:00

- Purpose: 修复 CTransPath RNA 教师训练结束时的 metadata 序列化错误，并从已完成的 12 个 fold 生成不可变正式汇总，不重训也不覆盖原产物。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/train_ctranspath_rna_teacher_gate.py`; `benchmarks/v5/phase_3_cell_region/tools/summarize_ctranspath_rna_teacher_gate.py`; `docs/运行记录/模型实验/运行记录_CTransPathRNA教师训练Gate_0908_023909.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `conda run -n aligner python ...summarize_ctranspath_rna_teacher_gate.py --fold-metrics ...0908_024000/fold_metrics.csv --output-dir ...0908_030100 --run-exit-code 1` — passed；12/12 fold 指标有限并生成 `summary.csv`、`metrics.json`。paired macro-F1 相对 label-only/shuffled 为 `-0.000419/-0.000237`，RNA gate 不通过。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/ctranspath_rna_teacher_gate_summary/ctranspath_rna_teacher_summary_0908_030100/`。
- Docs record: `docs/运行记录/模型实验/运行记录_CTransPathRNA教师训练Gate_0908_023909.md`（run record updated）。
- Commit: included in `Summarize CTransPath RNA gate`.

## 09-08 03:03:00

- Purpose: 基于 CTransPath/Phikon RNA-teacher 阴性 gate 与 v4 TLS cell 分支下游证据，记录 RNA 改为 region-program 训练监督的建议及其严格 paired/shuffle/downstream 验收条件。
- Changed: `docs/建议/模型与训练/建议_RNA区域监督接入FewClick_0908_030300.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: 逐项核对 CTransPath 12-fold 正式汇总、既有 RNA 打乱对照及 v4 TLS cell-zero 消融记录 — passed；建议不改变任何模型、数据或 NFS artifact。
- Artifacts: 无。
- Docs record: `docs/建议/模型与训练/建议_RNA区域监督接入FewClick_0908_030300.md`（recommendation record）。
- Commit: included in `Recommend RNA region supervision`.

## 09-08 03:10:00

- Purpose: 澄清并锁定 RNA region-supervision 的空间范围：必须沿用 v4 Phase 2 的软 region assignment 在 cell centroid 聚合 RNA program，不能误用单细胞 RNA target、四象限或独立 spot 网格。
- Changed: `docs/建议/模型与训练/建议_RNA区域监督接入FewClick_0908_030300.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: 核对 `benchmarks/v4/phase_2_region_encoder/src/model.py` 的 `[B,K,D]` soft token 与 `benchmarks/v4/phase_3_cell_region/src/model.py` 的 `[B,K,N]` cell assignment/attention contract — passed。
- Artifacts: 无。
- Docs record: `docs/建议/模型与训练/建议_RNA区域监督接入FewClick_0908_030300.md`（recommendation record updated）。
- Commit: included in `Clarify RNA region target`.

## 09-08 03:25:00

- Purpose: 实现并验证 v4 Phase 2 soft assignment 对齐的区域 RNA target cache 与 paired-vs-shuffled 区域 RNA 预测 gate；RNA 只作 target。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/build_region_rna_soft_target_smoke.py`; `benchmarks/v5/phase_3_cell_region/tools/run_region_rna_prediction_gate.py`; `docs/运行记录/特征缓存/运行记录_Phase2软区域RNA目标Smoke_0908_031500.md`; `docs/运行记录/模型实验/运行记录_软区域RNA预测Gate_0908_032500.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: Phase 2 soft target smoke `0908_032000` — passed（9 patches、64-way assignment per cell sums to one、全部 tensor finite）；spatial group prediction gate `0908_032500` — passed，paired/shuffled R2 为 `0.371491/-0.014517`。仅 9 patch mechanism gate，非 segmentation 结论。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/region_rna_target_smoke/region_rna_target_smoke_0908_032000/`; `/nfs-medical3/zyh/fewclick_v2/region_rna_prediction_gate/region_rna_prediction_gate_0908_032500/`。
- Docs record: `docs/运行记录/特征缓存/运行记录_Phase2软区域RNA目标Smoke_0908_031500.md`; `docs/运行记录/模型实验/运行记录_软区域RNA预测Gate_0908_032500.md`。
- Commit: included in `Build region RNA gate`.

## 09-08 03:30:00

- Purpose: 新增并验证冻结 Phase 2、可训练 v4-contract `CellToRegionAttention + RNA head` 的区域 RNA auxiliary gate；RNA 始终为 target。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/train_phase3_region_rna_aux_gate.py`; `docs/运行记录/模型实验/运行记录_Phase3区域RNA辅助头Gate_0908_033000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `CUDA_VISIBLE_DEVICES=1 conda run -n aligner python ...train_phase3_region_rna_aux_gate.py --timestamp 0908_032900 --epochs 1 --seeds 20260908` — passed；2 arms × 4 spatial folds 的真实 model/loss/evaluation 均有限。1 epoch 不用于质量结论。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/phase3_region_rna_aux_gate_0908_032900/`。
- Docs record: `docs/运行记录/模型实验/运行记录_Phase3区域RNA辅助头Gate_0908_033000.md`（run record）。
- Commit: included in `Add Phase3 RNA auxiliary gate`.

## 09-08 03:31:00

- Purpose: 启动 Phase 3 region RNA auxiliary head 的 paired/shuffled 三 seed 空间折训练 gate。
- Changed: `docs/运行记录/模型实验/运行记录_Phase3区域RNA辅助头Gate_0908_033000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `tmux fewclick_phase3_rna_0908_033000` — running；确认 tmux、conda 与真实 worker PID `3902382` 存在，分配 GPU 1。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/phase3_region_rna_aux_gate_0908_033000/`; log `/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/logs/phase3_region_rna_aux_0908_033000.log`。
- Docs record: `docs/运行记录/模型实验/运行记录_Phase3区域RNA辅助头Gate_0908_033000.md`（run record updated）。
- Commit: included in `Launch Phase3 RNA auxiliary gate`.

## 09-08 03:33:00

- Purpose: 完成并记录 Phase 3 region RNA auxiliary head 的 paired/shuffled 三 seed 空间折机制 gate。
- Changed: `docs/运行记录/模型实验/运行记录_Phase3区域RNA辅助头Gate_0908_033000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `fewclick_phase3_rna_0908_033000` — passed，`EXIT_CODE=0`；24/24 fold finite。paired/shuffled mean R2 为 `0.343834/-0.200843`，差 `+0.544677`；限于 9 patch mechanism gate。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/phase3_region_rna_aux_gate_0908_033000/`; log `/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/logs/phase3_region_rna_aux_0908_033000.log`。
- Docs record: `docs/运行记录/模型实验/运行记录_Phase3区域RNA辅助头Gate_0908_033000.md`（run record updated）。
- Commit: included in `Record Phase3 RNA auxiliary result`.

## 09-08 03:42:00

- Purpose: 实现 GDPH segmentation 与 COAD region-RNA auxiliary loss 的交替 Phase 3 smoke，并用历史正式匹配的 multiscale/SLIC/routing 契约完成重跑。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_phase3_seg_rna_joint_smoke.py`; `docs/运行记录/模型实验/运行记录_Phase3分割RNA交替训练Smoke_0908_034000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: 首次早期 SLIC index coverage mismatch 后，替换为 v4 正式 `multiscale_index_20260716_150251` + `slic_mmap_trainval_20260717_151426`；`CUDA_VISIBLE_DEVICES=1 ...run_phase3_seg_rna_joint_smoke.py --timestamp 0908_034200` — passed，三臂 20 alternating steps 和 GDPH val 均 finite。paired/seg-only val region accuracy 为 `0.85742/0.85938`，不可作正式收益结论。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/phase3_seg_rna_joint_smoke/phase3_seg_rna_joint_smoke_0908_034200/`。
- Docs record: `docs/运行记录/模型实验/运行记录_Phase3分割RNA交替训练Smoke_0908_034000.md`（run record updated）。
- Commit: included in `Run joint RNA segmentation smoke`.

## 09-08 03:50:00

- Purpose: 将交替 GDPH segmentation/COAD RNA 辅助训练扩展为三个 seed、随机固定 64/64 GDPH patch、200 steps 的中等规模 Phase 3 稳定性 gate。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_phase3_seg_rna_joint_smoke.py`; `docs/运行记录/模型实验/运行记录_Phase3分割RNA交替训练Smoke_0908_034000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `py_compile` — passed；tmux `fewclick_joint_rna_0908_035000` 已启动，第一个真实 worker PID `3938106` 在 GPU 1 运行。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/phase3_seg_rna_joint_smoke/phase3_seg_rna_joint_smoke_0908_03500{1,2,3}/`; logs `/nfs-medical3/zyh/fewclick_v2/phase3_seg_rna_joint_smoke/logs/joint_rna_0908_03500{1,2,3}.log`。
- Docs record: `docs/运行记录/模型实验/运行记录_Phase3分割RNA交替训练Smoke_0908_034000.md`（run record updated）。
- Commit: included in `Launch joint RNA stability gate`.

## 09-08 04:02:00

- Purpose: 独立审计 COAD transcript、cell boundary 与 H&E 的坐标一致性。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/audit_xenium_he_rna_geometry.py`; `docs/运行记录/数据审计/运行记录_COADRNAHE坐标配准审计_0908_040000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: 50 cells/43,614 transcripts geometry audit `0908_040200` — passed with 2px tolerance；strict inside 0.9289、within 2px 1.0、median signed distance +1.27px。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/rna_geometry_audit/rna_geometry_audit_0908_040200/`。
- Docs record: `docs/运行记录/数据审计/运行记录_COADRNAHE坐标配准审计_0908_040000.md`。
- Commit: included in `Audit COAD RNA geometry`.

## 09-08 04:10:00

- Purpose: 根据三 seed GDPH/COAD 交替训练中 paired 与 shuffled 无下游差异的结果，更新 RNA 接入策略为任务对齐、同域配对优先。
- Changed: `docs/建议/模型与训练/建议_RNA区域监督接入FewClick_0908_030300.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: 汇总三个 `0908_03500{1,2,3}` 完整 run（均 `EXIT_CODE=0`）— paired 平均 val region accuracy `0.91767`，shuffled `0.91792`，不支持泛 program 扩大训练。
- Artifacts: 已有 `/nfs-medical3/zyh/fewclick_v2/phase3_seg_rna_joint_smoke/phase3_seg_rna_joint_smoke_0908_03500{1,2,3}/`。
- Docs record: `docs/建议/模型与训练/建议_RNA区域监督接入FewClick_0908_030300.md`（recommendation updated）。
- Commit: included in `Refine RNA supervision strategy`.

## 09-08 04:10:00

- Purpose: 审计现有 COAD Xenium panel 的组织/TLS marker 覆盖，筛除不能表达组织语义的泛 hash RNA target。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/audit_coad_tissue_rna_markers.py`; `docs/运行记录/数据审计/运行记录_COAD组织TLSMarker审计_0908_041000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/audit_coad_tissue_rna_markers.py --output-dir /nfs-medical3/zyh/fewclick_v2/tissue_rna_marker_audit/tissue_marker_audit_0908_041000` — passed（1,020 cells；上皮 2/7、TLS 4/9，其余大部分组织 program 未覆盖）。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/tissue_rna_marker_audit/tissue_marker_audit_0908_041000/`。
- Docs record: `docs/运行记录/数据审计/运行记录_COAD组织TLSMarker审计_0908_041000.md`（run record）。
- Commit: included in `Audit RNA data candidates`.

## 09-08 04:50:00

- Purpose: 从本地数据目录核验可替代的空间病理 RNA 候选，区分细胞级和 Visium HD 区域级数据。
- Changed: `docs/运行记录/数据审计/运行记录_空间病理RNA候选数据可用性核验_0908_045000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: 只读检查 CSV 索引、CosMx/Visium/Xenium 的图像、表达、边界和 archive 内容 — passed；尚未将任何数据纳入训练。
- Artifacts: 无。
- Docs record: `docs/运行记录/数据审计/运行记录_空间病理RNA候选数据可用性核验_0908_045000.md`（data audit record）。
- Commit: included in `Audit RNA data candidates`.

## 09-08 04:55:00

- Purpose: 补充核验乳腺/肺 Xenium archive 内 gene panel 的组织与 TLS marker 覆盖。
- Changed: `docs/运行记录/数据审计/运行记录_空间病理RNA候选数据可用性核验_0908_045000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `bsdtar -tf` 读取 archive 清单、`rg` 读取 `gene_panel.json` — passed；乳腺 IDC 具上皮、间质、免疫、血管/肌成纤维和脂肪 marker，肺具上皮与 TLS marker。
- Artifacts: 无。
- Docs record: `docs/运行记录/数据审计/运行记录_空间病理RNA候选数据可用性核验_0908_045000.md`（data audit record updated）。
- Commit: included in `Verify Xenium marker panels`.

## 09-08 05:10:00

- Purpose: 验证 COAD CosMx 的转录本 cell ID、已审计 H&E affine 与组织/TLS marker 是否满足区域 RNA 监督的数据契约。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/audit_cosmx_rna_contract.py`; `docs/运行记录/数据审计/运行记录_CosMxCOADRNA区域监督契约审计_0908_051000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/audit_cosmx_rna_contract.py --output-dir /nfs-medical3/zyh/fewclick_v2/cosmx_rna_contract_audit/cosmx_rna_contract_audit_0908_051000` — passed（56/56 cell ID match、62,081 transcripts、H&E in-bounds 1.0；上皮/间质/T/TLS/髓系/肌成纤维 program 均有完整 marker 覆盖）。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/cosmx_rna_contract_audit/cosmx_rna_contract_audit_0908_051000/`。
- Docs record: `docs/运行记录/数据审计/运行记录_CosMxCOADRNA区域监督契约审计_0908_051000.md`（run record）。
- Commit: included in `Audit CosMx RNA contract`.

## 09-08 05:20:00

- Purpose: 用 COAD CosMx 构建首个由 H&E-only cell `reg-64` 输入和语义组织/TLS RNA target 组成的 Phase 2 soft-region cache smoke。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/build_cosmx_region_rna_target_smoke.py`; `docs/运行记录/特征缓存/运行记录_CosMx语义区域RNA目标Smoke_0908_052000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `tmux fewclick_cosmx_region_0908_052000` — passed，`EXIT_CODE=0`；9 patches、881 RNA target cells、847 H&E-only input cells，`cells=[9,153,74]`、`assignment=[9,64,153]`、`rna_targets=[9,64,6]` 全 finite，assignment sum=1。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/cosmx_region_rna_target_smoke/cosmx_region_rna_target_smoke_0908_052000/`。
- Docs record: `docs/运行记录/特征缓存/运行记录_CosMx语义区域RNA目标Smoke_0908_052000.md`（run record）。
- Commit: included in `Build CosMx RNA cache`.

## 09-08 05:30:00

- Purpose: 用 CosMx 语义组织/TLS region RNA target 运行 v4-contract Phase 3 paired-vs-shuffled 机制 gate。
- Changed: `docs/运行记录/模型实验/运行记录_CosMx语义RNAPhase3Gate_0908_053000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `tmux fewclick_cosmx_semantic_0908_053000` — passed，`EXIT_CODE=0`，24/24 fold finite；paired/shuffled mean R² `-0.146505/-0.207221`，差 `+0.060716`。单 window 结果未通过稳定可预测性门禁。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/phase3_region_rna_aux_gate_0908_053000/`。
- Docs record: `docs/运行记录/模型实验/运行记录_CosMx语义RNAPhase3Gate_0908_053000.md`（run record）。
- Commit: included in `Record CosMx RNA gate`.

## 09-08 05:50:00

- Purpose: 将 CosMx 语义 RNA cache 从单视野扩展至四个空间分离 view，并在 36 patch、16 个 `window × quadrant` 空间组上复验 paired-vs-shuffled Phase 3 gate。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/train_phase3_region_rna_aux_gate.py`; `docs/运行记录/特征缓存/运行记录_CosMx语义区域RNA目标Smoke_0908_052000.md`; `docs/运行记录/模型实验/运行记录_CosMx多视野语义RNAPhase3Gate_0908_055000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: 三个 `tmux fewclick_cosmx_windows_0908_054000` cache run 均 `EXIT_CODE=0`；`tmux fewclick_cosmx_multi_0908_055000` — passed（24/24 fold finite），但 paired/shuffled mean R² 为 `-0.070393/-0.043057`，差 `-0.027336`，未通过稳定可预测性门禁。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/cosmx_region_rna_target_smoke/cosmx_region_rna_target_smoke_0908_05400{1,2,3}/`; `/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/phase3_region_rna_aux_gate_0908_055000/`。
- Docs record: `docs/运行记录/特征缓存/运行记录_CosMx语义区域RNA目标Smoke_0908_052000.md`; `docs/运行记录/模型实验/运行记录_CosMx多视野语义RNAPhase3Gate_0908_055000.md`。
- Commit: included in `Evaluate multiwindow RNA gate`.

## 09-08 06:15:00

- Purpose: 为同一 COAD CosMx 全切片规模实验建立逐 cell RNA/几何索引、64 tile 空间清单和 index-backed H&E cache 路径。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/build_cosmx_fullslide_cell_index.py`; `benchmarks/v5/phase_3_cell_region/tools/select_cosmx_fullslide_tiles.py`; `benchmarks/v5/phase_3_cell_region/tools/build_cosmx_region_rna_target_smoke.py`; `docs/运行记录/特征缓存/运行记录_CosMx全切片规模扩展_0908_060000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: 全切片 index `0908_060000` — passed（292,447 transcript-matched cells）；64 tile manifest `0908_061000` — passed（576 patch、41,781 RNA target cells）；真实 index-backed edge-tile cache `0908_061200` — passed（`EXIT_CODE=0`，H&E-only cell feature、assignment、RNA target 全 finite）。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/cosmx_fullslide_rna_cache/cosmx_fullslide_cell_index_0908_060000/`; `/nfs-medical3/zyh/fewclick_v2/cosmx_fullslide_rna_cache/cosmx_fullslide_tile_manifest_0908_061000/`; `/nfs-medical3/zyh/fewclick_v2/cosmx_fullslide_rna_cache/cosmx_region_rna_target_smoke_0908_061200/`。
- Docs record: `docs/运行记录/特征缓存/运行记录_CosMx全切片规模扩展_0908_060000.md`（run record）。
- Commit: included in `Prepare fullslide RNA cache`.

## 09-08 06:20:00

- Purpose: 启动 COAD CosMx 全切片 64 个不重叠 tile 的 H&E-only cell feature / semantic RNA target 离线 cache，扩大 Phase 3 gate 的同切片样本量。
- Changed: `docs/运行记录/特征缓存/运行记录_CosMx全切片规模扩展_0908_060000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `tmux fewclick_cosmx_fullcache_0908_062000` 已启动；首个真实 worker PID `1653635` 在 GPU 1 运行，按 tile 独立日志与 `EXIT_CODE` 持久化。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/cosmx_fullslide_rna_cache/cosmx_region_rna_target_smoke_0908_062000_x*_y*/`; logs `/nfs-medical3/zyh/fewclick_v2/cosmx_fullslide_rna_cache/logs/fullcache_0908_062000_*.log`。
- Docs record: `docs/运行记录/特征缓存/运行记录_CosMx全切片规模扩展_0908_060000.md`（run record updated）。
- Commit: included in `Launch fullslide RNA cache`.

## 09-08 21:23:44

- Purpose: 收尾全切片 cache 并登记 576 patch 的同切片 RNA paired-vs-shuffled 规模 gate。
- Changed: `docs/运行记录/特征缓存/运行记录_CosMx全切片规模扩展_0908_060000.md`; `docs/运行记录/模型实验/运行记录_CosMx全切片RNA规模Gate_0908_212344.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: 只读核验 64 个 `region_rna_contract.npz`、64 个 `metadata.json` 和 64 个 `EXIT_CODE=0` — passed；将由真实训练路径继续核验 finite loss/metrics。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/cosmx_fullslide_rna_cache/cosmx_region_rna_target_smoke_0908_062000_x*_y*/`。
- Docs record: 上述特征缓存 record（updated）与模型实验 record（run record）。
- Commit: `4ab6f0d` (`Record fullslide RNA gate`)；随后仅回填本条 commit 字段并 amend。

## 09-08 21:25:10

- Purpose: 启动 CosMx 全切片 RNA paired-vs-shuffled 规模 gate。
- Changed: `docs/运行记录/模型实验/运行记录_CosMx全切片RNA规模Gate_0908_212344.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: 64 个 contract 已由真实训练 worker 加载，GPU 1 利用率 90%、显存 1,442 MiB — running；完成后检查 24/24 fold 的 finite 指标和 `EXIT_CODE`。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/phase3_region_rna_aux_gate_0908_212344/`（运行中）；日志 `/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/fullgate_0908_212344.log`。
- Docs record: `docs/运行记录/模型实验/运行记录_CosMx全切片RNA规模Gate_0908_212344.md`（run record updated）。
- Commit: pending.
