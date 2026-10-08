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

## 09-22 20:00:00

- Purpose: 在 CRC 双 panel 共享候选层上建立空间隔离、双 target 绑定的 manifest，作为 token 缓存和后续 panel_2/panel_1 训练的共同输入。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/build_crc_panel2_spatial_manifest.py`、`docs/运行记录/数据审计/运行记录_CRC多Panel空间隔离清单_0922_195841.md` 及本文件。
- Validation: `conda run -n aligner python -m unittest benchmarks.v5.phase_3_cell_region.tests.test_panel2_spatial_manifest -v` — 2 passed；真实构建 `--timestamp 0922_195841` — `EXIT_CODE=0`，861/188/89 patch 的跨 split buffer 审计通过。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_multpanel_spatial_manifest_0922_195841/`。
- Docs record: `docs/运行记录/数据审计/运行记录_CRC多Panel空间隔离清单_0922_195841.md`（运行记录）。
- Commit: pending.

## 09-22 19:10:00

- Purpose: 整理 CRC panel_1 结构证据和 panel_2 免疫证据的共享 patch×公共网格候选层，为后续可复用 token contract 准备输入。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/build_crc_patch_region_alignment.py`、`docs/运行记录/数据审计/运行记录_CRC_Panel1_Panel2共享候选数据整理_0922_191000.md` 及本文件。
- Validation: `conda run -n aligner python -m unittest benchmarks.v5.phase_3_cell_region.tests.test_crc_patch_region_alignment -v` — 2 passed；真实构建 `--timestamp 0922_191000` — `EXIT_CODE=0`，2,473 条候选字段与 inventory 一致。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_patch_region_alignment_0922_191000/`。
- Docs record: `docs/运行记录/数据审计/运行记录_CRC_Panel1_Panel2共享候选数据整理_0922_191000.md`（运行记录）。
- Commit: pending.

## 09-18 15:46:15

- Purpose: 忽略 Python 缓存及常见本地开发、测试和打包生成文件，避免其进入 Git 工作区。
- Changed: `.gitignore` 及本文件。
- Validation: `git check-ignore -v` 覆盖 `__pycache__`、`.pyc`、`.cache`、`.pytest_cache` 和 `.venv` — passed；`git diff --check` — passed。
- Artifacts: 无。
- Docs record: 无。
- Commit: `0328cd4` — `Ignore Python generated files`。

## 09-16 11:37:00

- Purpose: 针对扩量CRC五类身份训练中严格逆频率导致Treg过预测的问题，新增可控幂次逆频率加权，并完成平方根权重的真实训练与独立验证。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/train_semantic_phase3.py`、`benchmarks/v5/phase_3_cell_region/tests/test_identity_balancing.py`、`docs/建议/模型与训练/建议_CRC长尾细胞身份训练调整_0916_113507.md`、`docs/运行记录/模型实验/运行记录_CRC平方根逆频率身份训练_0916_113507.md`及本文件。
- Validation: `conda run -n aligner python -m unittest benchmarks.v5.phase_3_cell_region.tests.test_identity_balancing -v` — 6 passed；`py_compile`与`git diff --check`通过；GPU1真实训练15 epoch完成，最佳checkpoint独立val评价`EXIT_CODE=0`。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/{identity_0916_113507,evaluate_val_0916_113700}/`。
- Docs record: `建议_CRC长尾细胞身份训练调整_0916_113507.md`（建议）与`运行记录_CRC平方根逆频率身份训练_0916_113507.md`（运行记录）。
- Result: patient macro-F1=0.215664；严格逆频率的Treg泛滥预测被抑制，但Treg仍F1=0，候选不进入fusion。
- Commit: pending.

## 09-16 11:54:00

- Purpose: 实现并验证CRC身份分类的延迟重加权（DRW）：前三个epoch使用均匀损失，之后启用train-only平方根逆频率权重。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/train_semantic_phase3.py`、`benchmarks/v5/phase_3_cell_region/tests/test_identity_balancing.py`、`docs/运行记录/模型实验/运行记录_CRC延迟重加权身份训练_0916_115000.md`及本文件。
- Validation: `conda run -n aligner python -m unittest benchmarks.v5.phase_3_cell_region.tests.test_identity_balancing -v` — 7 passed；真实GPU1训练15 epoch完成，metrics前3 epoch `identity_reweighting_applied=false`、之后为true；最佳checkpoint独立val评价完成。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/{identity_0916_115000,evaluate_val_0916_115400}/`。
- Docs record: `docs/运行记录/模型实验/运行记录_CRC延迟重加权身份训练_0916_115000.md`（运行记录）。
- Result: patient macro-F1=0.221140，超过直接平方根权重0.215664；Treg在三个病例中仍均为F1=0，候选不进入fusion。
- Commit: pending.

## 09-16 12:09:00

- Purpose: 实现并检验CRC身份分类的细胞级类别感知均衡采样，单独排查少类有效训练曝光不足。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/train_semantic_phase3.py`、`benchmarks/v5/phase_3_cell_region/tests/test_identity_balancing.py`、`docs/运行记录/模型实验/运行记录_CRC细胞级类别感知采样身份训练_0916_120300.md`及本文件。
- Validation: 9项平衡采样单元测试通过；GPU1真实1 epoch smoke与正式训练均完成；最佳checkpoint独立val评价完成。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/{identity_0916_120400,identity_0916_120500,evaluate_val_0916_120900}/`。
- Docs record: `docs/运行记录/模型实验/运行记录_CRC细胞级类别感知采样身份训练_0916_120300.md`（运行记录）。
- Result: patient macro-F1=0.172248；2019 Treg recall=0.9024但precision=0.0101，且2017 Macrophage F1=0.00726，强制等量采样候选拒绝，不进入fusion。
- Commit: pending.

## 09-16 12:22:00

- Purpose: 实现并检验CRC身份分类的Balanced Softmax训练期决策边界校正，作为不重采样、不加类别权重的独立长尾对照。
- Changed: `semantic_losses.py`、`semantic_training.py`、`tools/train_semantic_phase3.py`、`tests/test_semantic_phase3.py`、`docs/运行记录/模型实验/运行记录_CRC_BalancedSoftmax身份训练_0916_121600.md`及本文件。
- Validation: 21项相关单元测试通过；GPU1真实smoke、正式训练和最佳checkpoint独立val均完成。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/{identity_0916_121700,identity_0916_121800,evaluate_val_0916_122200}/`。
- Docs record: `docs/运行记录/模型实验/运行记录_CRC_BalancedSoftmax身份训练_0916_121600.md`（运行记录）。
- Result: patient macro-F1=0.174523；2019 Treg recall=0.6585但precision=0.0114，候选拒绝，不进入fusion。
- Commit: pending.

## 09-16 13:45:00

- Purpose: 在扩量CRC身份任务上验证H&E-only细胞空间上下文（reg64、xy、面积、KNN及冻结区域token），并修复真实padding/无监督图块训练路径。
- Changed: `spatial_identity.py`、`train_crc_spatial_identity.py`、`test_spatial_identity.py`、`docs/运行记录/模型实验/运行记录_CRC扩量空间上下文身份训练_0916_122500.md`及本文件。
- Validation: 新增padding无有效邻居有限性测试通过；真实1 epoch smoke通过；GPU1正式训练与最佳checkpoint独立val评价完成。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/{spatial_identity_0916_133300,spatial_identity_0916_133500,evaluate_spatial_val_0916_134500}/`。
- Docs record: `docs/运行记录/模型实验/运行记录_CRC扩量空间上下文身份训练_0916_122500.md`（运行记录）。
- Result: patient macro-F1=0.203774，低于DRW 0.221140；CD8_T/Treg均为0，候选拒绝，不进入fusion。
- Commit: pending.

## 09-16 11:12:00

- Purpose: 对扩量`reg64`身份基线的异常零F1进行训练/标签/划分诊断，不修改医生审核标签或启动新训练。
- Changed: 更新`docs/运行记录/模型实验/运行记录_CRC扩量身份训练_0916_105533.md`，记录训练侧独立评价、权重饱和与逐细胞标签一致性检查。
- Validation: `evaluate_train_0916_111000`为`EXIT_CODE=0`；train+val五类计数精确恢复统一缓存总数；13个跨病例缓存-契约样本逐细胞标签完全一致。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/evaluate_train_0916_111000/`。
- Docs record: `docs/运行记录/模型实验/运行记录_CRC扩量身份训练_0916_105533.md`（运行记录）。
- Result: `effective_num_beta=0.999`使四个数万级类别权重饱和为同一值，当前训练并未真正平衡CD8_T/B_cell与Macrophage；应先修正平衡诊断，再判断reg64信息量。
- Commit: pending.

## 09-08 21:25:39

- Purpose: 收尾 CosMx 全切片 RNA paired-vs-shuffled 规模 gate，并记录其可解释范围。
- Changed: `docs/运行记录/模型实验/运行记录_CosMx全切片RNA规模Gate_0908_212344.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `fullgate_0908_212344.log` 的 `EXIT_CODE=0`；读取 `fold_metrics.csv` 确认 24/24 行 finite — passed；paired/shuffled 平均 R² `0.035376/-0.000828`，差 `+0.036204`。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/phase3_region_rna_aux_gate/phase3_region_rna_aux_gate_0908_212344/`。
- Docs record: `docs/运行记录/模型实验/运行记录_CosMx全切片RNA规模Gate_0908_212344.md`（run record updated）。
- Commit: pending.

## 09-08 21:31:00

- Purpose: 扩展 Phase 3 联合分割/RNA smoke，使其使用全切片 64 个 CosMx contract 的随机 RNA batch。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_phase3_seg_rna_joint_smoke.py`; `docs/运行记录/模型实验/运行记录_Phase3全切片RNA联合分割Gate_0908_213000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: `conda run -n aligner python -m py_compile ...run_phase3_seg_rna_joint_smoke.py`，以及真实 20-step `0908_213100` — passed（3 arm 全 finite；region accuracy `0.870588/0.874510/0.874510`）。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/phase3_seg_rna_joint_smoke/phase3_seg_rna_joint_smoke_0908_213100/`。
- Docs record: `docs/运行记录/模型实验/运行记录_Phase3全切片RNA联合分割Gate_0908_213000.md`（run record）。
- Commit: pending.

## 09-08 22:14:05

- Purpose: 启动三 seed 的全切片 RNA 联合 Phase 3 分割 gate。
- Changed: `docs/运行记录/模型实验/运行记录_Phase3全切片RNA联合分割Gate_0908_213000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: 3 个真实 Python worker 均已加载 64 个 contract，GPU 1/3/4 各占用约 1,654 MiB — running；最终验收为每个 seed 的三 arm 指标 finite 和 `EXIT_CODE=0`。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/phase3_seg_rna_joint_smoke/phase3_seg_rna_joint_smoke_0908_221400_s<seed>/`（运行中）。
- Docs record: `docs/运行记录/模型实验/运行记录_Phase3全切片RNA联合分割Gate_0908_213000.md`（run record updated）。
- Commit: pending.

## 09-08 22:16:38

- Purpose: 收尾三 seed 全切片 RNA 联合 Phase 3 分割 gate 并判定是否进入完整下游消融。
- Changed: `docs/运行记录/模型实验/运行记录_Phase3全切片RNA联合分割Gate_0908_213000.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: 三个 `joint_fullslide_0908_221400_s<seed>.log` 均 `EXIT_CODE=0`；读取三份 `metrics.json` 确认全部 finite — passed；平均 validation region accuracy 为 seg-only/paired/shuffled `0.917837/0.915706/0.915869`，paired 未优于任一对照。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/phase3_seg_rna_joint_smoke/phase3_seg_rna_joint_smoke_0908_221400_s{20260908,20260909,20260910}/`。
- Docs record: `docs/运行记录/模型实验/运行记录_Phase3全切片RNA联合分割Gate_0908_213000.md`（run record updated）。
- Commit: pending.

## 09-10 23:30:31

- Purpose: 根据第二次组会要求制定面向组织分割的 Phase 3 细胞身份、组成、空间关系与区域残差融合详细方案，明确阶段训练、病例划分、消融和三级指标门槛。
- Changed: `docs/建议/模型与训练/建议_Phase3细胞身份与组织结构改造方案_0910_233031.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: 对照冻结 Phase 3 模型、配置、训练与测试源码及已有实验记录核验接口；`git diff --check` 检查文档格式。本文为设计建议，数值门槛尚未实测，未修改模型代码、未启动训练。
- Artifacts: 无新增模型或数据产物。
- Docs record: `docs/建议/模型与训练/建议_Phase3细胞身份与组织结构改造方案_0910_233031.md`（recommendation）。
- Commit: 本次仅新增设计文档并追加记录，未创建代码提交。

## 09-10 23:48:57

- Purpose: 按用户确认将预训练初始化的自有细胞编码器列为第二项目阶段，第一阶段保留冻结XCell基线。
- Changed: `docs/建议/模型与训练/建议_Phase3细胞身份与组织结构改造方案_0910_233031.md`; `IMPLEMENTATION_STATUS.md`。
- Validation: 核对第一/第二项目阶段与A/B/C训练步骤的一致性，补充核类别标签泄漏约束、编码器微调与缓存更新、独立细胞任务和组织分割双重验收；`git diff --check`。
- Artifacts: 无；未启动训练。
- Docs record: `docs/建议/模型与训练/建议_Phase3细胞身份与组织结构改造方案_0910_233031.md`（recommendation updated）。
- Commit: 仅文档更新，未创建代码提交。

## 09-11 16:03:30

- Purpose: 实现第一项目阶段的细胞身份、组成与Phase3残差融合，并验证真实H&E/组织GT训练计算路径。
- Changed: `benchmarks/v5/phase_3_cell_region/src/{semantic_cells,composition,frozen_cell_region,semantic_cell_region,semantic_losses,semantic_contract,semantic_training}.py`，训练/评价/smoke工具、两份测试、`SEMANTIC_PHASE3.md`；建议文档及运行记录一并归档，v4不变。
- Validation: aligner `unittest discover -s benchmarks/v5/phase_3_cell_region/tests -p 'test_semantic*.py' -v` — 16 passed；真实GDPH 96-cell patch forward/backward/checkpoint reload/cache通过，初始和reload误差0；`git diff --check`。未执行CRC正式训练或最终点击分割评价。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/real_smoke_0911_151006/`。
- Docs record: `docs/运行记录/模型实验/运行记录_Phase3身份组成首版实现_0911_151006.md`；`docs/建议/模型与训练/建议_Phase3细胞身份与组织结构改造方案_0910_233031.md`。
- Commit: pending; implementation commit titled `Add semantic cell-to-region baseline`。
- Commit result: `680c94a` — `Add semantic cell-to-region baseline`；实现、测试、建议及运行记录已提交，未包含原始数据或模型产物。

## 09-11 19:53:38

- Purpose: 深入CRC公共坐标细胞表和niche数据，纠正CRC_1病例关联并生成正式训练前的复核候选。
- Changed: `benchmarks/v5/tools/audit_crc_training_candidates.py`、`render_crc_review_candidates.py`；CRC原审计报告更正、数据核验运行记录及本文件。
- Validation: aligner实际执行审计脚本（六表共120,000行抽读、三份完整offset0 niche表）与图像导出（36 crops、3 contact sheets），均完成；三例矩阵组合误差0、抽样坐标最大误差<6e-11 px，尺寸/MPP一致；人工查看一张contact sheet确认图像可读，未进行病理或配准验收；`git diff --check`。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_candidate_audit_0911_194537/`。
- Docs record: `docs/运行记录/数据审计/运行记录_CRC训练候选坐标与标签核验_0911_194537.md`。
- Findings: CRC_1对应2019-00538F；12,541个eligible primary niches；五类fraction分母不是全部细胞；panel2局部原路径不可访问。候选清单不具备verified训练状态，CRC正式训练尚未开始。
- Commit: pending; `Audit CRC training candidates`。
- Commit result: `059f735` — `Audit CRC training candidates`；审计与图像导出真实运行通过，原始NFS数据未修改。

## 09-11 21:38:42

- Purpose: 接入用户提供的panel2迁移路径，在真实CRC配准H&E上重新检测并试提取冻结XCell特征，避免将DAPI mask作为视觉输入。
- Changed: `benchmarks/v5/tools/prepare_crc_he_feature_probe.py`、`render_crc_cell_probe.py`、`benchmarks/v5/phase_3_cell_region/tests/test_crc_feature_probe.py`及相关运行记录。
- Validation: GPU1/PID3558468试提取结束，EXIT_CODE=0但metadata如实记录partial（4/6计划patch）；335条reg64 finite、191个待核验候选身份、源DAPI坐标误差<9e-12 px；4项单元测试和4幅可视化生成通过；未启动正式分类训练。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_he_feature_probe_0911_212850/`；日志同级 `crc_he_probe_0911_212850.log`。
- Docs record: `docs/运行记录/特征缓存/运行记录_CRC局部图块HE细胞特征试提取_0911_212850.md`。
- Commit: pending; `Prepare CRC H&E cell feature probe`。
- Commit result: `d6f3f22` — `Prepare CRC H&E cell feature probe`；提交前全套20项测试通过。

## 09-15 11:56:07

- Purpose: CRC候选身份由三类细分至五类，并从全表按稀有类别和空间分散采样；扩量目标三例各12块。
- Changed: `benchmarks/v5/phase_3_cell_region/src/crc_sampling.py`、`benchmarks/v5/tools/prepare_crc_he_feature_probe.py`、`tests/test_crc_feature_probe.py`及运行记录。
- Validation: aligner `python -m unittest discover -s benchmarks/v5/phase_3_cell_region/tests -v` — 22项通过；真实单块五类提取EXIT_CODE=0（207个HE检测、191个弱候选）；36块扩量运行中，PID887831/GPU1。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_he_feature_probe_0915_115607/`；单块验证为同级`crc_he_feature_probe_0915_115608/`。
- Docs record: `docs/运行记录/特征缓存/运行记录_CRC五类均衡扩量_0915_115607.md`（运行记录）。
- Commit: pending; 尚未将弱候选提升为verified真值，未启动正式分类训练。
- Final validation: 36/36图块完成、EXIT_CODE=0；20,210条有限reg64，16,762个五类弱候选（5,021/1,786/66/2,401/7,488）；26块实际覆盖多组XCell且完整保留检测。新增`validate_crc_he_feature_probe.py`真实检查通过，输出`integrity_audit.json`与`class_summary.csv`；最终23项unittest通过，接口文档同步更新。
- Commit result: `79ea4f7` — `Expand CRC candidate sampling to five classes`；另生成36幅匹配复核图，原始数据和冻结v4未修改。

## 09-15 12:10:53

- Purpose: 按用户要求核验CRC五类弱候选监督，从原始mIF重算强度、复核规则与HE/DAPI几何，统计歧义和阈值敏感性。
- Changed: `benchmarks/v5/tools/audit_crc_candidate_supervision.py`、`benchmarks/v5/phase_3_cell_region/tests/test_crc_supervision_audit.py`及审计运行记录。
- Validation: 26项unittest通过；36份原特征缓存只读完整性检查通过；原始八通道核验运行中，CPU/PID937702。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_candidate_qc_0915_121053/`及同级日志。
- Docs record: `docs/运行记录/数据审计/运行记录_CRC五类候选监督核验_0915_121053.md`（运行记录）。
- Commit: pending；不修改原始标签、不放行verified、不启动训练。
- Final validation: EXIT_CODE=0，36块/32,412个DAPI实例八通道原始强度重算通过（最大绝对误差0.000476563），阳性与完整类别规则不一致均0；16,762个候选有12,665个多marker竞争，6,544个通过几何诊断，去上界对照改变6,576个，联合诊断通过1,543个（Treg 0）。导出全部66个Treg的11页复核图，实际查看三例18个；不将诊断筛选等同身份真值，verified提升0。26项测试通过。
- Commit result: `35e14ac` — `Audit CRC candidate supervision against raw mIF`；核验工具、测试与结论已提交，原始数据及上轮特征缓存未修改。

## 09-15 12:29:44

- Purpose: 将CRC五类候选核验落地为盲法人工复核包，覆盖三例、五类和三种风险层，支持离线审核与CSV导出。
- Changed: `benchmarks/v5/tools/build_crc_manual_review_package.py`、`tests/test_crc_manual_review_package.py`、Phase3接口说明及本运行记录。
- Validation: 真实2022病例12项smoke EXIT_CODE=0；正式354项任务EXIT_CODE=0，354个ID/源细胞/图片均唯一，全部图片解码为1040×154，CSV/HTML引用一致；Chrome实际加载正式页面并查看三例各一项；全套29项unittest通过。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_manual_review_0915_122944/`；日志为同级`crc_manual_review_0915_122944.log`。
- Docs record: `docs/运行记录/数据审计/运行记录_CRC人工复核包构建_0915_122944.md`（运行记录）。
- Commit: pending；不修改源标签、不自动提升verified，等待两名审核者结果。
- Commit result: `2ee3dca` — `Build blinded CRC cell review package`；构建工具、测试、接口说明和运行记录已提交，原始数据未修改。

## 09-15 13:11:34

- Purpose: 因Windows单HTML在`file:`安全源与CSV下载上失败，改为单文件Excel人工复核交付，嵌入全部证据图并保留盲法。
- Changed: `benchmarks/v5/tools/build_crc_review_workbook.py`、Phase3接口说明及Excel交付运行记录。
- Validation: 3项真实smoke通过；正式任务EXIT_CODE=0，xlsx重新由LibreOffice打开，354行/媒体/绘图对象、R0001–R0354和判定/置信度LIST验证均通过；OOXML检查无每行自动标签、病例或风险层泄漏。提交前待复跑现有测试与脚本编译检查。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_manual_review_excel_0915_131134/CRC_manual_review_blinded.xlsx`。
- Docs record: `docs/运行记录/数据审计/运行记录_CRC人工复核Excel交付_0915_131134.md`（运行记录）。
- Commit: pending；不修改源标签、不自动verified，等待两位审核者完成xlsx。
- Commit result: `01b078d` — `Deliver CRC review workbook`；单文件Excel交付、运行记录和接口说明已提交，原始数据未修改。

## 09-15 19:48:54

- Purpose: 根据用户反馈，补充显示原自动类别的辅助复核Excel，使审核者能核验“确认/改类/不确定”并给出结论置信度。
- Changed: `benchmarks/v5/tools/build_crc_review_workbook.py`、Phase3接口说明及辅助复核运行记录。
- Validation: 3项真实smoke和正式354项任务均通过；354个媒体/绘图对象、R0001–R0354和三组LIST下拉验证通过；xlsx A/B列与私有分配表354/354一致；系统Python脚本编译通过。提交前待复跑现有测试。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_manual_review_assisted_excel_0915_194854/CRC_manual_review_assisted.xlsx`。
- Docs record: `docs/运行记录/数据审计/运行记录_CRC辅助复核Excel交付_0915_194854.md`（运行记录）。
- Commit: pending；显示原标签仅为辅助核验，不自动verified或启动训练。
- Commit result: `3953947` — `Add assisted CRC review workbook`；辅助模式、接口说明与运行记录已提交，原始标签和特征缓存未修改。

## 09-15 20:14:09

- Purpose: 按项目负责人确认直接采用医生审核的CRC panel2 mIF身份标签，生成不改写原始缓存的可追溯训练监督派生产物。
- Changed: `benchmarks/v5/tools/adopt_crc_expert_reviewed_labels.py`、`benchmarks/v5/phase_3_cell_region/tests/test_crc_label_adoption.py`、`SEMANTIC_PHASE3.md`、`docs/运行记录/数据审计/运行记录_CRC专家审核标签采用_0915_200000.md`及本文件。
- Validation: aligner `python -m unittest discover -s benchmarks/v5/phase_3_cell_region/tests -v` — 31 passed；真实36图块采用运行完成，逐文件SHA256、`reg64`有限性、接受掩码及五类计数复核通过（16,762 = 5,021/1,786/66/2,401/7,488）；`git diff --check`通过。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_expert_reviewed_mif_labels_0915_200001/`。
- Docs record: `docs/运行记录/数据审计/运行记录_CRC专家审核标签采用_0915_200000.md`（运行记录）。
- Commit result: `47d7ecb` — `Adopt expert reviewed CRC labels`；新缓存仅派生写入，原始mIF、标签表、probe与既有复核交付物均未修改。

## 09-15 20:21:10

- Purpose: 从医生审核CRC身份标签构建真实Phase 3 identity训练契约，并完成病例隔离的身份训练与封存test评价。
- Changed: `benchmarks/v5/tools/build_crc_identity_phase3_manifest.py`、`benchmarks/v5/phase_3_cell_region/tests/test_crc_identity_phase3_manifest.py`、`docs/运行记录/特征缓存/运行记录_CRC身份训练契约与Smoke_0915_201819.md`及本文件。
- Validation: aligner `python -m unittest discover -s benchmarks/v5/phase_3_cell_region/tests -v` — 32 passed；GPU1实际构建3图块smoke和36图块全量契约均为EXIT_CODE=0；真实identity训练EXIT_CODE=0；checkpoint/manifest指纹匹配的2022封存评价完成。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/{crc_identity_phase3_manifest_0915_201819,crc_identity_phase3_manifest_0915_202108,identity_0915_202109,evaluate_test_0915_202110}/`。
- Docs record: `docs/运行记录/特征缓存/运行记录_CRC身份训练契约与Smoke_0915_201819.md`（运行记录）。
- Result: val/test病例macro-F1为0.274683/0.243927；罕见CD8/Treg/B_cell未识别，当前非平衡基线不进入fusion。
- Commit result: `8938beb` — `Build CRC identity training contract`；真实训练产物仅写入FewClick NFS，原始医生审核标签和源图像未改写。

## 09-15 20:25:00

- Purpose: 处理CRC五类身份训练的类别不均衡，实施并比较train-only有效样本数加权和稀有类图块重采样。
- Changed: `semantic_losses.py`、`semantic_training.py`、`tools/train_semantic_phase3.py`、`tests/test_semantic_phase3.py`、`tests/test_identity_balancing.py`、`docs/运行记录/模型实验/运行记录_CRC身份类别平衡训练_0915_202500.md`及本文件。
- Validation: aligner `python -m unittest discover -s benchmarks/v5/phase_3_cell_region/tests -v` — 36 passed；GPU1一图块smoke与全量训练均EXIT_CODE=0；两份checkpoint均通过manifest指纹封存test评价。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/{identity_0915_202502,evaluate_test_0915_202503,identity_0915_202505,evaluate_test_0915_202506}/`。
- Result: loss-only与loss+patch重采样test macro-F1为0.223774/0.208525，均低于非平衡基线0.243927，且CD8/Treg/B_cell F1仍为0；候选拒绝，未进入fusion。
- Docs record: `docs/运行记录/模型实验/运行记录_CRC身份类别平衡训练_0915_202500.md`（运行记录）。
- Commit result: `446cefc` — `Add train-only identity balancing`；平衡统计、重采样和两次候选评价均未读取封存test标签参与训练或模型选择。

## 09-15 20:36:38

- Purpose: 按项目负责人要求建立三个CRC病例混合训练的图块级内部验证协议，并测试直接平均训练与平衡训练。
- Changed: `semantic_contract.py`、`semantic_training.py`、`tools/build_crc_pooled_identity_manifest.py`、`tests/test_semantic_contract.py`、`tests/test_crc_pooled_manifest.py`、`docs/运行记录/模型实验/运行记录_CRC混合病例身份训练_0915_203500.md`及本文件。
- Validation: aligner `python -m unittest discover -s benchmarks/v5/phase_3_cell_region/tests -v` — 38 passed；真实manifest SHA检查通过；GPU1两次混合训练均EXIT_CODE=0；内部val checkpoint/manifest指纹评价完成。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/{crc_pooled_identity_manifest_0915_203638,identity_0915_203639,identity_0915_203640,evaluate_val_0915_203641}/`。
- Result: 最佳直接平均内部val macro-F1为0.253330；CD8/Treg/B_cell F1均0。结果为同病例内部图块验证，不是患者独立泛化，候选不进入fusion。
- Docs record: `docs/运行记录/模型实验/运行记录_CRC混合病例身份训练_0915_203500.md`（运行记录）。
- Commit result: `5f2e55f` — `Add pooled CRC identity protocol`；协议显式标注同病例内部验证，未将指标表述为患者独立泛化。

## 09-15 21:25:01

- Purpose: 使用现有reg64、xy近邻和冻结Phase2区域token验证显式空间上下文能否改善CRC五类身份预测。
- Changed: `spatial_identity.py`、`tools/train_crc_spatial_identity.py`、`tools/evaluate_crc_spatial_identity.py`、`tests/test_spatial_identity.py`、`docs/运行记录/模型实验/运行记录_CRC空间上下文身份探针_0915_212501.md`及本文件。
- Validation: 空间模型单元forward/backward通过；真实GPU1训练EXIT_CODE=0；checkpoint/manifest一致的内部val评价完成；待提交前全套测试与`git diff --check`复跑。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/{spatial_identity_0915_212502,evaluate_spatial_val_0915_212503}/`。
- Result: 内部macro-F1 0.258831略高于直接MLP 0.253330，但CD8/Treg F1仍0，B_cell仅2019病例0.007692；候选拒绝，不进入fusion。
- Docs record: `docs/运行记录/模型实验/运行记录_CRC空间上下文身份探针_0915_212501.md`（运行记录）。
- Commit result: `109e1e7` — `Probe spatial CRC identity context`；H&E-only空间上下文真实运行完成，未使用mIF marker或标签作为模型输入。

## 09-16 11:07:00

- Purpose: 扩量CRC缓存完成后，采用1,553个成功图块的医生审核mIF转移标签，训练并独立验证五类H&E `reg64` 身份基线。
- Changed: `crc_sampling.py`、`prepare_crc_he_feature_probe.py`、`adopt_crc_expert_reviewed_labels.py`、`build_crc_quota_manifest.py`、`build_crc_cached_identity_manifest.py`、`semantic_cell_region.py`、`semantic_training.py`及对应测试/运行记录。新增多分片partial缓存的显式合并、缓存直连身份契约，以及身份阶段绕过冻结区域分支的接口；后者不改变fusion路径。
- Validation: `python -m unittest`相关19项通过，`py_compile`与`git diff --check`通过；1,553图块标签采用、3图块契约冒烟、1,553图块全量契约、1 epoch身份训练冒烟、15 epoch正式训练与best checkpoint独立val重评估均为`EXIT_CODE=0`；checkpoint/manifest SHA256一致。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/{crc_expert_reviewed_mif_labels_0916_105600,crc_cached_identity_manifest_0916_110100,crc_pooled_identity_manifest_0916_110200,identity_0916_110500,evaluate_val_0916_110700}/`。
- Docs record: `docs/运行记录/特征缓存/运行记录_CRC五类5万级扩量准备_0915_214116.md`与`docs/运行记录/模型实验/运行记录_CRC扩量身份训练_0916_105533.md`（运行记录）。
- Result: patient macro-F1 0.196451；CD8_T/Treg F1均为0，B_cell F1为0.004007。扩量纯`reg64`候选拒绝，不进入fusion；后续仅可在明确标为实验候选的空间/组织上下文设置下继续。
- Commit: pending.

## 09-16 23:41:00

- Purpose: 以冻结的直接 CTransPath-768 H&E细胞及局部上下文特征替换`reg64`，在固定CRC pooled内部协议下训练五类身份分类候选。
- Changed: `benchmarks/v5/tools/cache_crc_ctranspath_cell_features.py`、`benchmarks/v5/tools/train_crc_ctranspath_identity.py`、`docs/运行记录/特征缓存/运行记录_CRC直接CTransPath细胞特征替换_0916_135000.md`、`docs/运行记录/模型实验/运行记录_CRC直接CTransPath身份训练_0916_233415.md`及本文件。
- Validation: aligner `python -m py_compile benchmarks/v5/tools/train_crc_ctranspath_identity.py`通过；3图块真实特征smoke、1,553图块全量缓存、1 epoch训练smoke与15 epoch GPU1正式训练均为`EXIT_CODE=0`；全量特征复核为1,553个文件、0个无效、768维有限值且cell-id逐行对齐。
- Artifacts: 本机NFS容错缓存`/home/zhaoyh/FewClick_v2/.cache/semantic_phase3/{crc_ctranspath_cell_features_nfs6_0916_222359,ctranspath_identity_smoke_0916_233800,ctranspath_identity_0916_234100}/`；源H&E只读镜像为`/nfs-medical6/zyh/SRRSH_CRC_HE_selected_0916_215326/`。
- Docs record: `docs/运行记录/特征缓存/运行记录_CRC直接CTransPath细胞特征替换_0916_135000.md`与`docs/运行记录/模型实验/运行记录_CRC直接CTransPath身份训练_0916_233415.md`（运行记录）。
- Result: 最优内部`patient_macro_f1=0.200883`（epoch 11），低于`reg64` DRW 0.221140；Treg F1仍为0，候选拒绝，不进入组织分割fusion。
- Commit: `fd0f358` — `Train direct CTransPath CRC identity probe`。

## 09-17 20:52:58

- Purpose: 固化项目负责人确认的 SRRSH CRC 临床医生人工校验数据标准口径。
- Changed: `docs/数据集情况/SRRSH_CRC_数据目录审计报告_20260909_165250.md` 及本文件。
- Validation: `git diff --check` — passed；已复核新增说明与原有标签说明的一致性。
- Artifacts: 无。
- Docs record: `docs/数据集情况/SRRSH_CRC_数据目录审计报告_20260909_165250.md`（数据集情况更新）。
- Commit: pending.

## 09-20 21:45:02

- Purpose: 记录 FewClick 面向可扩展组织/病理任务的多原型与蛋白/组学监督方案，并将“不按病例划分且尽量避免数据泄露”固化为 FewClick skill 规则。
- Changed: `docs/建议/模型与训练/方案_可扩展原型与组学监督_0920_210632.md`、`/home/zhaoyh/.codex/skills/fewclick-v2/SKILL.md` 及本文件。
- Validation: `python /home/zhaoyh/.codex/skills/.system/skill-creator/scripts/quick_validate.py /home/zhaoyh/.codex/skills/fewclick-v2` — passed；`git diff --check` — passed。
- Artifacts: 无。
- Docs record: `docs/建议/模型与训练/方案_可扩展原型与组学监督_0920_210632.md`（建议记录）。
- Commit: pending.

## 09-20 21:56:52

- Purpose: 在 v5 建立可扩展交互任务的空间划分契约和正负多原型匹配首批接口，为 CRC 真实任务 manifest 与后续提示训练提供可验证基础。
- Changed: `benchmarks/v5/phase_5_prompt_encoder/{__init__.py,src/__init__.py,src/extensible_task_contract.py,src/multi_prototype.py,tests/test_extensible_task_contract.py,tests/test_multi_prototype.py}`、`docs/建议/模型与训练/方案_可扩展原型与组学监督_0920_210632.md` 及本文件。
- Validation: `conda run -n aligner python -m unittest discover -s benchmarks/v5/phase_5_prompt_encoder/tests -v` — 7 tests passed；`conda run -n aligner python -m unittest discover -s benchmarks/v5/phase_3_cell_region/tests -p 'test_semantic_contract.py' -v` — 8 tests passed；`conda run -n aligner python -m py_compile benchmarks/v5/phase_5_prompt_encoder/src/extensible_task_contract.py benchmarks/v5/phase_5_prompt_encoder/src/multi_prototype.py` 与 `git diff --check` — passed。
- Artifacts: 无；未启动训练或生成数据缓存。
- Docs record: `docs/建议/模型与训练/方案_可扩展原型与组学监督_0920_210632.md`（建议记录更新）。
- Commit: pending.

## 09-20 22:11:01

- Purpose: 构建并审计 SRRSH CRC 三例 Tumor ROI 的首个空间任务 manifest；ROI 外不推断为非肿瘤。
- Changed: `benchmarks/v5/phase_5_prompt_encoder/tools/build_crc_tumor_roi_manifest.py`、`benchmarks/v5/phase_5_prompt_encoder/tests/test_crc_tumor_roi_manifest.py`、`docs/运行记录/数据审计/运行记录_CRC肿瘤ROI空间任务清单_0920_221101.md` 及本文件。
- Validation: `PYTHONPATH=. conda run -n aligner python benchmarks/v5/phase_5_prompt_encoder/tools/build_crc_tumor_roi_manifest.py --timestamp 0920_221101` — `EXIT_CODE=0`；54 个空间组通过 manifest 校验；单元测试 8 passed。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/extensible_prompt/crc_tumor_roi_manifest_0920_221101/`。
- Docs record: `docs/运行记录/数据审计/运行记录_CRC肿瘤ROI空间任务清单_0920_221101.md`（运行记录）。
- Commit: pending.

## 09-21 00:41:07

- Purpose: 将 SRRSH CRC 双 panel 公共坐标细胞表流式聚合为 H&E 物理网格的结构/免疫区域证据 inventory，保留 panel、观测量与映射覆盖率边界。
- Changed: `benchmarks/v5/phase_3_cell_region/src/crc_region_supervision.py`、`tools/build_crc_region_supervision_inventory.py`、`tests/test_crc_region_supervision.py`、`docs/运行记录/数据审计/运行记录_CRC区域细胞蛋白证据清单_0921_000100.md` 及本文件。
- Validation: `PYTHONPATH=. conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/build_crc_region_supervision_inventory.py --timestamp 0921_004107` — completed；三例 panel1/panel2 总观测计数与公共坐标 manifest 一致；单元测试 passed；`py_compile` 与 `git diff --check` — passed。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_region_supervision_inventory_0921_004107/`。
- Docs record: `docs/运行记录/数据审计/运行记录_CRC区域细胞蛋白证据清单_0921_000100.md`（运行记录）。
- Commit: pending.

## 09-21 01:35:00

- Purpose: 为 CRC 区域证据新增 panel 分开的结构/免疫观测组成头和覆盖率加权损失，避免将局部 panel 观察值混为完整组织组成。
- Changed: `benchmarks/v5/phase_3_cell_region/src/{observed_region_evidence.py,semantic_losses.py}`、`tests/test_observed_region_evidence.py`、`docs/建议/模型与训练/方案_可扩展原型与组学监督_0920_210632.md` 及本文件。
- Validation: `conda run -n aligner python -m unittest benchmarks.v5.phase_3_cell_region.tests.test_observed_region_evidence benchmarks.v5.phase_3_cell_region.tests.test_crc_region_supervision -v` — 2 tests passed；`py_compile` 与 `git diff --check` — passed。
- Artifacts: 无；尚未启动训练。
- Docs record: `docs/建议/模型与训练/方案_可扩展原型与组学监督_0920_210632.md`（建议记录更新）。
- Commit: pending.

## 09-21 17:09:38

- Purpose: 优先建立 CRC panel_2 五类免疫细胞区域证据与医生审核 H&E 局部 patch 的空间候选对应，为真实 token-footprint 契约前的可审计输入准备。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/build_crc_patch_region_alignment.py`、`tests/test_crc_patch_region_alignment.py`、`docs/运行记录/数据审计/运行记录_CRC局部Patch_Panel2区域证据候选对齐_0921_170938.md`、`docs/建议/模型与训练/方案_可扩展原型与组学监督_0920_210632.md` 及本文件。
- Validation: `PYTHONPATH=. conda run -n aligner python benchmarks/v5/phase_3_cell_region/tools/build_crc_patch_region_alignment.py --timestamp 0921_170938` — `EXIT_CODE=0`，生成 2,473 条/1,553 patch 记录；相关单元测试 3 passed，`py_compile` 与 `git diff --check` passed。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_patch_region_alignment_0921_170938/`。
- Docs record: `docs/运行记录/数据审计/运行记录_CRC局部Patch_Panel2区域证据候选对齐_0921_170938.md`（运行记录）及方案更新。
- Commit: pending.

## 09-21 17:27:04

- Purpose: 建立 Phase 2 软 assignment 到 CRC panel_2 公共物理网格的显式 footprint 契约，并用真实 H&E token 运行 panel_2 区域观测组成 loss 的 forward/backward 冒烟。
- Changed: `benchmarks/v5/phase_3_cell_region/src/region_footprint_contract.py`、`tools/smoke_crc_panel2_region_evidence.py`、`tests/test_region_footprint_contract.py`、两份 CRC panel_2 token 运行记录、方案文档及本文件。
- Validation: 几何/目标门控单元测试 4 passed；首次 `0921_171839` 因低分辨率坐标步长错误失败且已归档，修正后 `PYTHONPATH=. conda run -n aligner python .../smoke_crc_panel2_region_evidence.py --timestamp 0921_172704 --device cuda:1` — `EXIT_CODE=0`，三病例各一真实 patch 的有效 token 数 58/64/61，loss 均有限。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_panel2_region_evidence_smoke_0921_172704/`；失败目录 `..._0921_171839/` 不可使用。
- Docs record: `docs/运行记录/模型实验/运行记录_CRC_Panel2区域证据真实Token冒烟_{0921_171839,修正_0921_172704}.md`（运行记录）及方案更新。
- Commit: pending.

## 09-21 17:35:23

- Purpose: 为 CRC panel_2 观测组成监督建立不按病例切分、但以物理 block 与 256 µm 足迹缓冲隔离的 train/val/test manifest。
- Changed: `benchmarks/v5/phase_3_cell_region/src/panel2_spatial_manifest.py`、`tools/build_crc_panel2_spatial_manifest.py`、`tests/test_panel2_spatial_manifest.py`、`docs/运行记录/数据审计/运行记录_CRC_Panel2空间隔离训练清单_0921_173523.md` 及本文件。
- Validation: 相关空间清单及 ExtensibleTaskContract 单元测试 6 passed；真实构建 `--timestamp 0921_173523` — `EXIT_CODE=0`，1,553 个候选中剔除 415 个跨 split 缓冲重叠 patch，保留 train/val/test=861/188/89；manifest overlap 审计通过。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_panel2_spatial_manifest_0921_173523/`。
- Docs record: `docs/运行记录/数据审计/运行记录_CRC_Panel2空间隔离训练清单_0921_173523.md`（运行记录）。
- Commit: pending.

## 09-21 17:41:12

- Purpose: 在空间隔离 CRC panel_2 清单上运行冻结 H&E Phase 2 token 的区域观测组成小规模 probe，先判断辅助监督是否具有可学习信号。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/train_crc_panel2_observed_probe.py`、`docs/运行记录/模型实验/运行记录_CRC_Panel2冻结Phase2区域监督Probe_0921_174112.md`、方案文档及本文件。
- Validation: 真实 GPU 1 训练 `--train-per-case 32 --val-per-case 10 --epochs 10` — `EXIT_CODE=0`；96/30 个三病例均衡 patch，1,819 个 val token，best val CE=0.754778 小于 train-only mean baseline=0.770354；产物 checkpoint 和 metrics 已复核，`py_compile` 与 `git diff --check` passed。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_panel2_observed_probe_0921_174112/`。
- Docs record: `docs/运行记录/模型实验/运行记录_CRC_Panel2冻结Phase2区域监督Probe_0921_174112.md`（运行记录）及方案更新。
- Commit: pending.

## 09-21 21:56:49

- Purpose: 为可扩展组织任务建立显式正负标签契约和统一像素评测，完成原Phase5/单原型/多原型的真实配对诊断。
- Changed: v5 phase_5_prompt_encoder 的 `src/tissue_tasks.py`、`src/tissue_evaluation.py`、两个对应测试、`tools/evaluate_tissue_prototypes.py`、`tools/render_tissue_diagnostic.py`及运行记录。
- Validation: phase5 v5测试13项通过；GPU1真实单episode及分层36episode执行完成，EXIT_CODE=0；20张WSI、36个patch，所有方法相同support/GT；缓存Phase2来源核对为epoch028；可视化已检查。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/extensible_prompt/tissue_prototype_eval_0921_210659/`（最终分层诊断）；先前顺序取样目录保留。
- Docs record: `docs/运行记录/模型实验/运行记录_组织原型统一像素评测_0921_210136.md`（运行记录）。
- Result: 未提示区域task-macro Dice为Phase5 0.600216、单原型0.479809、多原型0.479165；仅已知组织诊断，不是新类别或等预算训练比较。
- Commit: pending.

## 09-22 00:15:10

- Purpose: 固定组织原型任务头对照的空间划分，不按病例分组，保守隔离完整三尺度上下文。
- Changed: v5 Phase5 `src/tissue_spatial_split.py`、`tools/build_tissue_spatial_split.py`、对应单测及审计记录。
- Validation: aligner `python -m unittest discover -s benchmarks/v5/phase_5_prompt_encoder/tests -v` — 16项通过；真实全量索引构建完成；train/val/test各读取一项真实缓存，有限token及正负support检查通过；git diff --check通过。pytest未安装，未改环境。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/extensible_prompt/tissue_spatial_split_0922_001235/`。
- Docs record: `docs/运行记录/数据审计/运行记录_组织原型空间划分_0922_001235.md`（运行记录）。
- Result: 53534个patch，三个split覆盖全部组织和提示尺寸；保留历史暴露警示，仅支持内部冻结编码器任务头比较，尚未启动新训练。
- Commit: pending.

## 09-22 00:36:00

- Purpose: 实现可学习单/多原型，并完成固定空间任务上的原Phase5等更新预算对照。
- Changed: v5 Phase5 `src/learnable_tissue_prototype.py`、对应测试、`tools/train_tissue_prototype_comparison.py`及运行记录；v4未修改。
- Validation: aligner unittest 18项通过；三方法真实2步冒烟及3seed×100步对照均EXIT_CODE=0；所有checkpoint重载预测一致；保存mask重算216+648条指标完全一致；git diff --check通过。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/extensible_prompt/tissue_head_comparison_0922_003020/`及`tissue_head_comparison_0922_003340/`；同根tmux日志记录EXIT_CODE。
- Docs record: `docs/运行记录/模型实验/运行记录_组织可学习原型等步数对照_0922_003020.md`（运行记录）。
- Result: 未提示像素task-macro Dice三seed均值Phase5 0.532423，单原型0.649928，多原型0.637241。仅小预算已见组织内部验证，test未使用，不宣称收敛后的架构优劣或新组织泛化。
- Commit: pending.

## 09-22 00:50:45

- Purpose: 扩大固定任务集，比较0/100/300/1000步学习曲线及恒等token对照，检查训练预算影响。
- Changed: v5 Phase5训练runner的多快照/零训练对照/逐任务指标持久化；新增学习曲线汇总与mask复核工具、对应单测及运行记录；v4不变。
- Validation: aligner unittest 19项通过；真实多快照冒烟EXIT_CODE=0，与旧无中间快照第2步权重逐tensor严格一致；扩大实验9条1000步轨迹及36次checkpoint重载通过、EXIT_CODE=0；冒烟792条和扩大实验8208条像素指标由mask重算完全一致；本次diff检查通过。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/extensible_prompt/tissue_head_comparison_0922_004328/`及`tissue_head_comparison_0922_004524/`，同根tmux日志。
- Docs record: `docs/运行记录/模型实验/运行记录_组织原型学习曲线_0922_004328.md`（运行记录）。
- Result: 1000步未提示Dice三seed均值Phase5 0.489811、单原型0.563333、多原型0.562369；Phase5仍改善，原型接近平台；与前轮样本不同不可直接比较。test未使用。
- Commit: pending.

## 09-22 00:57:00

- Purpose: 验证现有1000步模型的同片跨patch提示迁移，不重新训练，不使用test。
- Changed: v5 Phase5 `src/cross_patch_support.py`、对应测试、`tools/evaluate_cross_patch_tissue.py`及运行记录；保持support/query局部坐标独立，隔离完整上下文。
- Validation: aligner unittest 21项通过；真实单query冒烟与完整评测EXIT_CODE=0；2112条指标由保存mask重算一致；所有local指标精确复现上轮对应query；本次diff检查通过。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/extensible_prompt/tissue_cross_patch_0922_005255/`及`tissue_cross_patch_0922_005516/`。
- Docs record: `docs/运行记录/模型实验/运行记录_组织跨patch提示迁移_0922_005255.md`（运行记录）。
- Result: 96/108对合格，全部12类覆盖；Phase5/单原型/多原型cross Dice为0.453415/0.533374/0.541090。精确标注数匹配70对子集趋势一致，26对数量差异已披露。仅同片内部迁移，不是新类别或跨癌种泛化。
- Commit: pending.

## 09-22 01:02:00

- Purpose: 增加跨patch support/query episodic训练入口，support与query使用完整上下文空间隔离。
- Changed: v5 `tools/train_cross_patch_tissue.py`、跨训练单测及运行记录；v4不变。
- Validation: aligner unittest 22项通过；真实GPU1两步冒烟36个空间配对、三方法checkpoint和loss均完成，EXIT_CODE=0；修正边界单测后重跑通过；未读取test。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/extensible_prompt/cross_patch_train_0922_005700/`及日志。
- Docs record: `docs/运行记录/模型实验/运行记录_组织跨patch提示训练_0922_005700.md`（运行记录）。
- Result: 训练链路可运行，尚未形成效果结论；下一步扩大任务和seed并在固定96对迁移验证集比较。
- Commit: pending.

## 09-22 01:12:00

- Purpose: 完成跨patch support/query训练，准备固定迁移验证。
- Changed: 修正跨patch runner按每个组织×提示尺寸收集`per_stratum`配对；更新训练记录。
- Validation: 训练前CPU 1步72对检查通过；GPU1三seed完整运行各288对、1000步、三模型，均EXIT_CODE=0；产物及3000条/seed loss记录存在。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/extensible_prompt/cross_patch_train_0922_011700/`、`011701/`、`011702/`及日志。
- Docs record: `docs/运行记录/模型实验/运行记录_组织跨patch提示训练_0922_005700.md`（运行记录）。
- Result: 仅报告训练loss；尚未运行固定96对Dice/IoU评测。
- Commit: pending.

## 09-22 01:30:00

- Purpose: 用固定96对跨patch迁移集评估完整训练的三seed checkpoint。
- Changed: evaluator增加`--checkpoint-root/--seeds`以读取独立跨patch训练产物；更新训练运行记录。
- Validation: 三seed评测均EXIT_CODE=0；固定配对、mask与指标路径完成；git diff --check通过。
- Artifacts: `tissue_cross_patch_0922_012000/012001/012002`及评测日志；训练产物`cross_patch_train_0922_011700/011701/011702`。
- Docs record: `docs/运行记录/模型实验/运行记录_组织跨patch提示训练_0922_005700.md`（运行记录）。
- Result: cross Dice均值Phase5 0.1714、单原型0.4829、多原型0.4977，低于未跨patch训练模型0.4534/0.5334/0.5411；本版跨patch训练不采用，先审计任务构造与坐标/数量分布。
- Commit: pending.

## 09-22 12:35:00

- Purpose: 以相同任务数量、顺序、实际support数量和初始化严格比较local与cross-patch训练。
- Changed: 新增`paired_training.py`、严格配对训练runner及单测；新增运行记录，未修改v4。
- Validation: aligner unittest 25项通过；真实2步冒烟EXIT_CODE=0；完整18条1000步训练轨迹与checkpoint重载通过、EXIT_CODE=0；70个固定验证对的2520条指标均由保存mask重算；九组初始化hash及训练/验证patch隔离核对通过；git diff --check通过。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/extensible_prompt/paired_tissue_training_0922_120811/`和`paired_tissue_training_0922_122000/`及同根日志。
- Docs record: `docs/运行记录/模型实验/运行记录_本地与跨patch严格配对对照_0922_120811.md`（运行记录）。
- Result: 冻结70对cross Dice：Phase5 cross训练0.4858，相对local+0.0696；单/多原型cross训练0.4943/0.4989，相对local-0.0472/-0.0329。仅内部已见组织评测，test未使用。
- Commit: pending.

## 09-22 13:20:00

- Purpose: 按组织错误模式测试75% local＋25% cross混合episode，缓解原型全cross训练退化。
- Changed: 新增组织TP/FP/FN错误分析模块、测试和分析工具；严格配对runner支持methods/arms及固定混合掩膜；更新运行记录。
- Validation: aligner unittest 28项通过；错误分析输入2520条记录完成；混合真实2步冒烟EXIT_CODE=0；完整18条1000步轨迹及70对验证EXIT_CODE=0；2520条mask指标即时重算；6组混合臂初始化hash与local/cross严格相同；git diff --check通过。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/extensible_prompt/paired_tissue_training_0922_130100/`、`paired_tissue_training_0922_130300/`及`paired_tissue_training_0922_122000/error_analysis.json`。
- Docs record: `docs/运行记录/模型实验/运行记录_跨patch组织错误分析_0922_124500.md`（运行记录）。
- Result: 远处提示Dice单原型local/mixed/cross=0.5415/0.5391/0.4943；多原型=0.5318/0.5365/0.4989。75/25混合多原型为当前兼顾local/远处提示的内部基线，test未使用。
- Commit: pending.

## 09-22 13:40:00

- Purpose: 在固定0.5阈值下评测当前mixed多原型对缺失目标query的拒识能力。
- Changed: `tissue_evaluation.py`增加空目标FP指标，新增缺失目标评测工具和运行记录；v4未修改。
- Validation: aligner unittest 29项通过；修正工具报告字典语法后py_compile通过；真实1对冒烟和64对完整评测均EXIT_CODE=0；192条指标从保存mask重算一致，缺失标签与空间隔离检查通过。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/extensible_prompt/absent_tissue_target_0922_132600/`及`absent_tissue_target_0922_133000/`和日志。
- Docs record: `docs/运行记录/模型实验/运行记录_组织缺失目标拒识评测_0922_132442.md`（运行记录）。
- Result: mixed多原型空目标误触发率0.7448，平均错误阳性面积0.2909；当前0.5阈值下拒识不足，需增加缺失目标训练任务后再评估。
- Commit: pending.

## 09-22 14:10:00

- Purpose: 将组织缺失目标作为显式负监督episode，并在固定存在/缺失目标集上与local、mixed严格对照。
- Changed: `paired_training.py`新增可复现的reject日程和全负query目标；paired runner新增同WSI空间隔离缺失query构造及`reject`臂；缺失目标evaluator支持冻结pair、多臂/seed/checkpoint步数；扩展单测与运行记录，未修改v4。
- Validation: aligner单测10项、`py_compile`和`git diff --check`通过；真实2步三臂训练冒烟、checkpoint重载、mask复算及1对缺失目标评测均EXIT_CODE=0；完整9 checkpoint×1000步、70对存在目标（1260指标）和64对缺失目标×3臂评测均EXIT_CODE=0。
- Artifacts: `paired_tissue_training_0922_135357/`、`paired_tissue_training_0922_135508/`、`absent_tissue_target_0922_140800/`、`0922_140801/`、`0922_140802/`及同根日志；首次约束审计失败记录`reject_training_full_0922_134910.log`。
- Docs record: `docs/运行记录/模型实验/运行记录_缺失目标拒识训练_0922_134102.md`（运行记录）。
- Result: reject相对同轮mixed将空目标activation 0.7188降至0.6510、FPA 0.2672降至0.2366，cross Dice 0.5440升至0.5517；仅内部空间区域验证。
- Commit: c03a382.

## 09-22 18:27:38

- Purpose: 修复 Phase 3 与 CRC panel_2 联合 smoke 的 cell-to-region 参数顺序，并补全独立 CRC validation observed-composition CE。
- Changed: `benchmarks/v5/phase_3_cell_region/src/panel2_joint_contract.py`、`tools/run_phase3_seg_panel2_joint_smoke.py`、`tests/test_panel2_joint_contract.py`、两份 `docs/运行记录/模型实验/运行记录_Phase3分割与CRC_Panel2联合对照Smoke*0922_*.md` 及本文件。
- Validation: `conda run -n aligner python -m unittest discover -s benchmarks/v5/phase_3_cell_region/tests -v` — 60 passed；真实 GPU1 12-step `seg_only`/`seg_plus_panel2` 对照两次均 `EXIT_CODE=0`，所有 loss 有限，GDPH validation region accuracy 均为 `0.896484375`，CRC validation panel_2 CE 为 `1.7737813/1.5642970`。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/phase3_seg_panel2_joint_smoke_{0922_182700,0922_184000}/`。
- Docs record: `docs/运行记录/模型实验/运行记录_Phase3分割与CRC_Panel2联合对照Smoke复跑_0922_182700.md` 与 `docs/运行记录/模型实验/运行记录_Phase3分割与CRC_Panel2联合对照Smoke验证补全_0922_184000.md`（运行记录）。
- Commit: pending.

## 09-22 22:02:00

- Purpose: 建立双 panel 冻结 token 合同，并用可比常数先验验证 Panel1/Panel2 区域组成训练。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/cache_crc_multpanel_token_contract.py`、`benchmarks/v5/phase_3_cell_region/tools/train_crc_panel2_cached.py`；缓存与 Panel1/Panel2 训练运行记录及本文件。训练器新增 `--panel`、step-0 validation、按训练采样口径的常数先验和封存评估对比。
- Validation: `conda run -n aligner python -m py_compile ...` — passed；真实 GPU `cuda:1` 两步 smoke — passed；Panel1 gate 三 seed×200 step 与长程三 seed×1000 step 均 `EXIT_CODE=0`、指标有限且 test token 均为 5,432。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_multpanel_token_contract_0922_201100/`；`crc_panel1_cached_train_0922_215417_{17,29,43}/`；`crc_panel1_cached_train_0922_215800_{17,29,43}/`；Panel2 `crc_panel2_cached_train_0922_210000_{17,29,43}/`。
- Docs record: `docs/运行记录/特征缓存/运行记录_CRC多Panel冻结Token合同缓存_0922_202200.md`、`docs/运行记录/模型实验/运行记录_CRC_Panel2缓存长程训练_0922_210000.md`、`docs/运行记录/模型实验/运行记录_CRC_Panel1缓存训练Gate_0922_215417.md`、`docs/运行记录/模型实验/运行记录_CRC_Panel1缓存长程训练_0922_215800.md`（运行记录）。
- Result: Panel1 确认性 CE `1.4406±0.0158` 优于常数先验 `1.5207`；Panel2 CE 未超过其常数先验，停止扩量。均未证明 FewClick 分割收益。
- Commit: `16d5ab3`.

## 09-22 22:30:00

- Purpose: 比较 Panel1 组成辅助监督对冻结 Phase2 下 GDPH 区域分割主指标的影响。
- Changed: `benchmarks/v5/phase_3_cell_region/tools/run_phase3_seg_panel1_joint_smoke.py`；Panel1 联合 smoke 与三 seed gate 运行记录；本文件。
- Validation: `py_compile` 通过；真实 GPU `cuda:1` 12-step smoke 与三 seed×100-step gate 均 `EXIT_CODE=0`。固定初始化、GDPH patch 选择和顺序，CRC contract 空间划分一致。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/phase3_seg_panel1_joint_smoke_0922_222500/`及`phase3_seg_panel1_joint_smoke_0922_222700_{17,29,43}/`。
- Docs record: `docs/运行记录/模型实验/运行记录_Phase3_CRC_Panel1联合分割对照Smoke_0922_221900.md`、`docs/运行记录/模型实验/运行记录_Phase3_CRC_Panel1联合分割Gate_0922_222700.md`（运行记录）。
- Result: Panel1 CE 平均降低0.0402，但GDPH val region accuracy平均变化-0.000327；未发现分割提升，未启动点击评测。
- Commit: `663f6e1`.

## 09-23 00:26:00

- Purpose: 修复Panel1对照输入和取样，补实际像素分割及交互点击评测。
- Changed: v5 `src/panel1_controlled.py`、`tools/run_panel1_controlled.py`、`tests/test_panel1_controlled.py`；运行记录及本文件。
- Validation: helper 6项与已有observed loss/输入契约3项测试通过；真实GPU1五臂两步smoke `panel1_controlled_0923_002800` 完成、EXIT_CODE=0；checkpoint重载及保存mask指标复算通过。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_controlled_0923_002800/`，worker PID 1454829。
- Docs record: `docs/运行记录/模型实验/运行记录_Panel1输入统一与像素点击对照_0923_001800.md`（运行记录）。
- Result: 统一屏蔽缺失面积/类别；覆盖三例CRC和空间组；独立固定数据seed；五臂同初始化、共同主任务日程；保存权重、像素mask、逐类Dice/IoU、真实2/4/6点击和NoC；历史val诊断，不使用test。
- Commit: 本次实现提交，后续结果条目记录版本。

## 09-23 00:34:00

- Purpose: 完成修正后的Panel1三seed五臂像素/点击对照并归档全部结果。
- Changed: `tools/summarize_panel1_controlled.py`、Panel1输入统一运行记录和本文件；训练与评测实现版本`0ae68cb`。
- Validation: 相关unit tests 9项通过；真实两步smoke通过；正式15臂全部完成、3日志EXIT_CODE=0；21个训练checkpoint保存；mask即时复算通过；跨seed固定manifest、每seed初始化/日程/首轮158个点击任务相同的汇总审计通过；报告工具在smoke和正式产物上均通过；示例图已查看，视觉质量最终由用户判断。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_controlled_0923_003000_{17,29,43}/`；汇总`panel1_controlled_report_0923_003300/`，launch/logs `panel1_launch_0923_003000/`。
- Docs record: `docs/运行记录/模型实验/运行记录_Panel1输入统一与像素点击对照_0923_001800.md`（运行记录）。
- Result: seg_only像素/6点击Dice=0.695062/0.509877；joint_003=0.695265/0.511074，2/4点击及NoC未改善；joint_010=0.695472/0.506055；预训练=0.693765/0.506298；额外分割预算=0.693790/0.502943。CRC辅助臂CE未超过同口径常数先验。历史val fine-only冻结点击头诊断，尚无可靠部署收益。
- Commit: 实现`0ae68cb`；本条与汇总工具/结果记录一起提交。

## 09-23 12:14:00

- Purpose: 验证Panel1已定位新H&E细胞的身份识别，重点评价上皮与血管内皮。
- Changed: `benchmarks/v5/tools/run_crc_panel1_cell_identity.py`、`refit_crc_panel1_cell_identity.py`、`run_crc_panel1_binary.py`、`benchmarks/v5/phase_3_cell_region/tests/test_crc_panel1_cell_identity.py`、本文件及Panel1单细胞试验记录。
- Validation: Python编译通过；2项空间/标签/逐类指标契约测试通过；真实40图块smoke和160图块正式特征提取/训练完成；五类三seed、缩放修正三seed、二类模型真实运行；二类模型从真实H&E图像及核mask独立加载推理成功；测试预测表与混淆矩阵复核。`aligner`环境未装pytest，因此契约函数通过直接调用执行。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_cell_identity_smoke_0923_115500/`、`panel1_cell_identity_0923_120304/`、`panel1_cell_identity_refit_0923_121156/`、`panel1_cell_binary_0923_121320/`。
- Docs record: `docs/运行记录/模型实验/运行记录_Panel1单细胞分类试验_0923_115000.md`（运行记录）。
- Result: 五类分类头macro-F1=0.16099，全判上皮；特征缩放后macro-F1均值约0.251，血管F1仍低。上皮/血管内皮二选一在同一病例空间划分、已成功匹配的测试细胞上血管precision/recall/F1=0.864/0.911/0.887，test macro-F1=0.936；同一test经前序实验使用，属于探索性结果。其他细胞强制二选一，新病例未验证。
- Commit: 本次提交；完整hash见Git历史。

## 09-23 12:32:00

- Purpose: 查明Panel1五类分类失败及二类高分的条件。
- Changed: `benchmarks/v5/tools/diagnose_crc_panel1_five_class.py`、`docs/运行记录/模型实验/运行记录_Panel1五类失败原因诊断_0923_122824.md`、前轮Panel1单细胞记录的后续诊断引用、本文件。
- Validation: 同一固定manifest的3,855个真实细胞特征完成train-only标准化、五类线性/RBF验证集选择、既有test逐类评估及10组类别两两验证探针，status=complete；二类模型在全部五类test细胞上的误判数量复核；Python编译通过。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_cell_five_diagnostic_0923_123026/`；首轮原样保留`panel1_cell_five_diagnostic_0923_122824/`。
- Docs record: `docs/运行记录/模型实验/运行记录_Panel1五类失败原因诊断_0923_122824.md`（运行记录）。
- Result: 五类标准化RBF的test macro-F1=0.3504，上皮/血管/淋巴/CAF/SMA的F1=0.879/0.291/0/0.072/0.510；二类模型对其他148个细胞中的114个误判为血管。test曾被前序实验使用，仅供诊断。
- Commit: 本次提交；完整hash见Git历史。

## 09-23 13:10:00

- Purpose: 用额外独立Panel1匹配细胞扩充五类训练集，并检验淋巴内皮/CAF训练图像增强的增益。
- Changed: `benchmarks/v5/tools/expand_crc_panel1_cell_identity.py`、`docs/运行记录/模型实验/运行记录_Panel1训练细胞扩量与稀有类增强_0923_124600.md`、本文件。
- Validation: 原val/test细胞逐行一致、train/eval空间组不交叉、旧特征校验复用；合成增强mask几何检查通过，真实120图块smoke通过；正式5,049个新增真实细胞特征和760个稀有类训练增强视图均有限，日志EXIT_CODE=0；旧/扩量/增强三臂模型及固定线性对照完整运行，指标产物status=complete；Python编译和定向diff检查通过。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_cell_expand_0923_124600/`、`panel1_cell_expand_features_0923_130008/`、`panel1_cell_expand_compare_0923_130424/`、`panel1_cell_expand_compare_0923_130633/`；真实smoke另见运行记录。
- Docs record: `docs/运行记录/模型实验/运行记录_Panel1训练细胞扩量与稀有类增强_0923_124600.md`（运行记录）。
- Result: 固定平衡线性分类器的旧/扩量/增强test五类macro-F1=0.4235/0.4413/0.4457；增强使CAF test F1=0.220，但血管和SMA相对仅扩量下降，淋巴test仍只有2个、F1=0。旧test多次使用，均为探索性结果。
- Commit: 本次提交；完整hash见Git历史。

## 09-23 13:46:00

- Purpose: 在Panel1现有可用训练空间组内，从未使用的匹配patch补充五类真实细胞，并复核扩量和稀有类增强的影响。
- Changed: `benchmarks/v5/tools/supplement_crc_panel1_spatial.py`、本运行记录、本文件。
- Validation: 新增1,178个细胞、61个patch，覆盖22个训练空间组；五类都增加，原val/test逐行一致且train/eval空间组隔离；新增特征1,178个、旧特征8,904个复用，896个稀有类增强视图；对照三臂完整、EXIT_CODE=0；Python编译检查通过。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_spatial_add_0923_131606/`、`panel1_spatial_features_0923_134100/`、`panel1_spatial_compare_0923_134500/`。
- Docs record: `docs/运行记录/模型实验/运行记录_Panel1空间分散五类训练补充_0923_131606.md`。
- Result: 旧训练/新增真实patch/再加增强的验证macro-F1=0.3726/0.3872/0.4146，既有test macro-F1=0.4413/0.4309/0.4499。真实扩量单独未改善test总分；增强臂较旧训练高0.0086，但淋巴test仅2个仍F1=0，血管/SMA未同时改善。既有test多次使用，结果仅探索性。
- Limitation: 此前43个未覆盖train空间组没有通过当前匹配表/配准筛选的可用细胞；新增样本来自原22个可用训练组内的新patch。未做跨病例验证。
- Commit: 本次实现提交；本条与实验记录一起提交。

## 09-23 14:12:06

- Purpose: 为Panel1五类细胞制作H&E局部/空间上下文并排抽查图，供人工判断当前128px视野是否缺少上下文。
- Changed: `benchmarks/v5/tools/visualize_crc_panel1_context_audit.py`、本数据审计记录、本文件。
- Validation: `python -m py_compile ...`通过；实际生成20例，目标HE核均存在、图像与mask尺寸一致、核质心误差≤0.1px；mIF来源标签及test预测逐行归档。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_cell_context_audit_0923_141206/`；包含总览、五张逐类图、`sample_audit.csv`及`audit.json`。
- Docs record: `docs/运行记录/数据审计/运行记录_Panel1五类细胞HE上下文可视抽查_0923_140603.md`。
- Commit: 本次可视化实现提交；本条与记录一并提交。

## 09-23 15:15:31

- Purpose: 对比Panel1五类细胞的128、256及双视野H&E特征，审计淋巴内皮在源匹配数据与固定清单中的覆盖。
- Changed: `benchmarks/v5/tools/probe_crc_panel1_multiscale_context.py`、`docs/运行记录/模型实验/运行记录_Panel1多尺度细胞上下文与淋巴标签审计_0923_144220.md`、本文件。
- Validation: Python编译通过；真实2图块12细胞smoke通过；全量425图块10,082细胞特征提取完成、源hash/维度/有限性检查通过、EXIT_CODE=0；三臂及0.25/0.5/0.75/1上下文权重对照完成、EXIT_CODE=0；128 test逐细胞预测复现既有线性模型；源审计3,968个空间候选patch，所选细胞与源候选对账无缺失。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_multiscale_0923_144220_{audit,smoke,features,compare}/`及`panel1_multiscale_weighted_0923_151043/`。
- Docs record: `docs/运行记录/模型实验/运行记录_Panel1多尺度细胞上下文与淋巴标签审计_0923_144220.md`。
- Result: val macro-F1为128=0.3890、256=0.3476、直接拼接=0.3333、低权重0.25融合=0.3848；扩大上下文无整体增益。已通过基本匹配/空间筛选的独立淋巴内皮候选train/val/test=159/99/54，固定清单仅107/18/2；该类既稀少，评估清单又明显漏抽。未选候选尚未逐个复核核mask。
- Commit: 本次实现提交；完整hash见Git历史。
## 09-23 16:13:29

- Purpose: 逐个复核Panel1未选淋巴内皮匹配，构建五类均衡补充评估清单并复测冻结多尺度模型。
- Changed: `benchmarks/v5/tools/audit_crc_panel1_eval_candidates.py`、`benchmarks/v5/tools/evaluate_crc_panel1_supplement.py`、两份Panel1补充评估运行记录、本文件。
- Validation: 两工具编译通过；真实源数据扫描与500细胞双视野提取/冻结分类器复测均`EXIT_CODE=0`；133个未选淋巴候选逐一核mask复核通过，500个补充细胞通过源hash与空间分组检查。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_eval_supplement_0923_155834/`和`panel1_eval_supplement_scores_0923_155610/`。
- Docs record: `docs/运行记录/数据审计/运行记录_Panel1未选淋巴细胞复核与补充评估_0923_155448.md`、`docs/运行记录/模型实验/运行记录_Panel1补充评估清单多尺度复测_0923_155610.md`。
- Result: 补充test五类各50个；128准确率0.408、macro-F1 0.393，256为0.344/0.332，直接融合为0.368/0.347；淋巴内皮128识别11/50。
- Commit: 本次实现提交；完整hash见Git历史。
## 09-23 16:51:00

- Purpose: 试验Panel1轻量非线性分类头与局部主导门控多尺度融合。
- Changed: `benchmarks/v5/tools/train_crc_panel1_mlp.py`、`docs/运行记录/模型实验/运行记录_Panel1轻量MLP与门控多尺度Probe_0923_162500.md`。
- Validation: 工具编译通过；3个随机种子、原train训练、原val选择、均衡补充test复测完成，最终`EXIT_CODE=0`；两次早期运行分别因线程限制和补充特征路径问题停止，未用于结果。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_mlp_probe_0923_165000/`。
- Docs record: `docs/运行记录/模型实验/运行记录_Panel1轻量MLP与门控多尺度Probe_0923_162500.md`。
- Result: 128-MLP原val macro-F1 0.454、补充test 0.318；门控128+256原val 0.457、补充test 0.292；均低于线性128补充test macro-F1 0.393，暂不替换基线。
- Commit: 本次实现提交；完整hash见Git历史。

## 09-23 15:31:58

- Purpose: 设计面向连续正负点纠错的 Phase 5/6 Decoder 改造方案，补全正点漏检扩张与原负区域保持。
- Changed: `docs/建议/模型与训练/建议_迭代式正负点纠错Decoder_0923_153007.md`、本文件。
- Validation: 对照当前 Phase 5/6 模型、在线 prompt 映射及损失接口审阅；文档结构和 `git diff --check` 检查。
- Artifacts: 无；仅设计文档。
- Docs record: `docs/建议/模型与训练/建议_迭代式正负点纠错Decoder_0923_153007.md`（建议）。
- Commit: 无；未改模型代码。

## 09-23 16:20:14

- Purpose: 实现并验证 v5 正负点迭代式纠错 Decoder，接入上一轮 logits、逐点 prompt、符号单调更新、soft assignment 后像素点击锚点和交互状态。
- Changed: `benchmarks/v5/phase_6_mask_decoder/`（decoder、逐轮 episode、训练/评估入口、单 patch/WSI 状态接口、测试）、`docs/建议/模型与训练/建议_迭代式正负点纠错Decoder_0923_153007.md`、相关 `docs/运行记录/模型实验/运行记录_迭代Decoder*.md`、本文件。
- Validation: 7 项定向模型/状态测试通过；Python compileall 和 `git diff --check` 通过；真实 v5 空间图块 smoke 4/2 通过；64/24 配对 probe 通过；256/64/3 epoch 开发训练 `EXIT_CODE=0`，epoch 2 质量门禁通过并生成 checkpoint。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/iterative_decoder/spatial_routing_0923_154545/`、`smoke_0923_154545/`、`smoke_0923_154843/`、`probe_0923_161201/`、`eval64_0923_161353/`、`eval_historical_val_0923_161638/`、`final_dev_0923_161452/`。
- Docs record: `docs/运行记录/模型实验/运行记录_迭代Decoder最终结构扩大训练_0923_161452.md` 及同目录配对/诊断记录（运行记录）；`docs/建议/模型与训练/建议_迭代式正负点纠错Decoder_0923_153007.md`（建议）。
- Result: 开发性最终 checkpoint 为 `/nfs-medical3/zyh/fewclick_v2/iterative_decoder/final_dev_0923_161452/checkpoint_epoch_002.pth`；正点后 Dice +0.002575，负点后 Dice −0.000709，负点远端 FP −0.018911，正确正区变化 −0.006800，像素/region 冲突均为0。旧 v4 初始化对部分空间 val 有历史暴露，不能当独立测试或部署结论。
- Commit: 未提交；工作树含用户已有的无关未跟踪文件，未安全分界，需用户确认后再提交。

## 09-23 16:13:29

- Purpose: 逐个复核Panel1未选淋巴内皮匹配，构建五类均衡补充评估清单并复测冻结多尺度模型。
- Changed: `benchmarks/v5/tools/audit_crc_panel1_eval_candidates.py`、`benchmarks/v5/tools/evaluate_crc_panel1_supplement.py`、两份Panel1补充评估运行记录、本文件。
- Validation: 两工具编译通过；真实源数据扫描与500细胞双视野提取/冻结分类器复测均`EXIT_CODE=0`；133个未选淋巴候选逐一核mask复核通过，500个补充细胞通过源hash与空间分组检查。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_eval_supplement_0923_155834/`和`panel1_eval_supplement_scores_0923_155610/`。
- Docs record: `docs/运行记录/数据审计/运行记录_Panel1未选淋巴细胞复核与补充评估_0923_155448.md`、`docs/运行记录/模型实验/运行记录_Panel1补充评估清单多尺度复测_0923_155610.md`。
- Result: 补充test五类各50个；128准确率0.408、macro-F1 0.393，256为0.344/0.332，直接融合为0.368/0.347；淋巴内皮128识别11/50。
- Commit: 本次实现提交；完整hash见Git历史。
## 09-23 19:15:00

- Purpose: 将训练空间内全部通过几何/核mask复核的候选加入训练并重训Panel1分类头。
- Changed: `benchmarks/v5/tools/extract_crc_panel1_expanded_features.py`、`benchmarks/v5/tools/train_crc_panel1_expanded_heads.py`、本全量扩充运行记录、本文件。
- Validation: 全量训练manifest生成`EXIT_CODE=0`，新增4,902个细胞；128/256特征提取`EXIT_CODE=0`，复用10,082并提取4,902；Logistic与MLP重训/评估`EXIT_CODE=0`，val、原test、均衡补充test均完成。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_train_all_0923_170500/`、`panel1_train_all_features_0923_172500/`、`panel1_expanded_heads_0923_173500/`。
- Docs record: `docs/运行记录/模型实验/运行记录_Panel1训练候选全量扩充Probe_0923_170500.md`。
- Result: 128 Logistic补充test macro-F1 0.398（原0.393），128 MLP 0.331，门控MLP 0.324；全量扩充对线性基线只有小幅提升，MLP仍未超过线性模型。
- Commit: 本次实现提交；完整hash见Git历史。

## 09-23 16:51:00

- Purpose: 试验Panel1轻量非线性分类头与局部主导门控多尺度融合。
- Changed: `benchmarks/v5/tools/train_crc_panel1_mlp.py`、`docs/运行记录/模型实验/运行记录_Panel1轻量MLP与门控多尺度Probe_0923_162500.md`。
- Validation: 工具编译通过；3个随机种子、原train训练、原val选择、均衡补充test复测完成，最终`EXIT_CODE=0`；两次早期运行分别因线程限制和补充特征路径问题停止，未用于结果。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_mlp_probe_0923_165000/`。
- Docs record: `docs/运行记录/模型实验/运行记录_Panel1轻量MLP与门控多尺度Probe_0923_162500.md`。
- Result: 128-MLP原val macro-F1 0.454、补充test 0.318；门控128+256原val 0.457、补充test 0.292；均低于线性128补充test macro-F1 0.393，暂不替换基线。
- Commit: 本次实现提交；完整hash见Git历史。

## 09-23 15:31:58

- Purpose: 设计面向连续正负点纠错的 Phase 5/6 Decoder 改造方案，补全正点漏检扩张与原负区域保持。
- Changed: `docs/建议/模型与训练/建议_迭代式正负点纠错Decoder_0923_153007.md`、本文件。
- Validation: 对照当前 Phase 5/6 模型、在线 prompt 映射及损失接口审阅；文档结构和 `git diff --check` 检查。
- Artifacts: 无；仅设计文档。
- Docs record: `docs/建议/模型与训练/建议_迭代式正负点纠错Decoder_0923_153007.md`（建议）。
- Commit: 无；未改模型代码。

## 09-23 16:20:14

- Purpose: 实现并验证 v5 正负点迭代式纠错 Decoder，接入上一轮 logits、逐点 prompt、符号单调更新、soft assignment 后像素点击锚点和交互状态。
- Changed: `benchmarks/v5/phase_6_mask_decoder/`（decoder、逐轮 episode、训练/评估入口、单 patch/WSI 状态接口、测试）、`docs/建议/模型与训练/建议_迭代式正负点纠错Decoder_0923_153007.md`、相关 `docs/运行记录/模型实验/运行记录_迭代Decoder*.md`、本文件。
- Validation: 7 项定向模型/状态测试通过；Python compileall 和 `git diff --check` 通过；真实 v5 空间图块 smoke 4/2 通过；64/24 配对 probe 通过；256/64/3 epoch 开发训练 `EXIT_CODE=0`，epoch 2 质量门禁通过并生成 checkpoint。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/iterative_decoder/spatial_routing_0923_154545/`、`smoke_0923_154545/`、`smoke_0923_154843/`、`probe_0923_161201/`、`eval64_0923_161353/`、`eval_historical_val_0923_161638/`、`final_dev_0923_161452/`。
- Docs record: `docs/运行记录/模型实验/运行记录_迭代Decoder最终结构扩大训练_0923_161452.md` 及同目录配对/诊断记录（运行记录）；`docs/建议/模型与训练/建议_迭代式正负点纠错Decoder_0923_153007.md`（建议）。
- Result: 开发性最终 checkpoint 为 `/nfs-medical3/zyh/fewclick_v2/iterative_decoder/final_dev_0923_161452/checkpoint_epoch_002.pth`；正点后 Dice +0.002575，负点后 Dice −0.000709，负点远端 FP −0.018911，正确正区变化 −0.006800，像素/region 冲突均为0。旧 v4 初始化对部分空间 val 有历史暴露，不能当独立测试或部署结论。
- Commit: 未提交；工作树含用户已有的无关未跟踪文件，未安全分界，需用户确认后再提交。

## 09-23 16:13:29

- Purpose: 逐个复核Panel1未选淋巴内皮匹配，构建五类均衡补充评估清单并复测冻结多尺度模型。
- Changed: `benchmarks/v5/tools/audit_crc_panel1_eval_candidates.py`、`benchmarks/v5/tools/evaluate_crc_panel1_supplement.py`、两份Panel1补充评估运行记录、本文件。
- Validation: 两工具编译通过；真实源数据扫描与500细胞双视野提取/冻结分类器复测均`EXIT_CODE=0`；133个未选淋巴候选逐一核mask复核通过，500个补充细胞通过源hash与空间分组检查。
- Artifacts: `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_eval_supplement_0923_155834/`和`panel1_eval_supplement_scores_0923_155610/`。
- Docs record: `docs/运行记录/数据审计/运行记录_Panel1未选淋巴细胞复核与补充评估_0923_155448.md`、`docs/运行记录/模型实验/运行记录_Panel1补充评估清单多尺度复测_0923_155610.md`。
- Result: 补充test五类各50个；128准确率0.408、macro-F1 0.393，256为0.344/0.332，直接融合为0.368/0.347；淋巴内皮128识别11/50。
- Commit: 本次实现提交；完整hash见Git历史。

## 09-24 13:43:24

- Purpose: 三例 Panel1 全量候选特征准备，修正全类型最近邻匹配并记录逐图块进度。
- Changed: benchmarks/v5/tools/extract_crc_panel1_three_case_cells.py。
- Validation: 前版两例真实 smoke 完成；修正版待真实 smoke。
- Artifacts: /nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_three_case_smoke_0924_134324_CASE。
- Docs record: docs/运行记录/特征缓存/运行记录_Panel1三例候选特征全量提取_0924_134324.md（运行记录）；补记此前 运行记录_Panel1三例全切片细胞特征可行性_0923_201237.md。
- Commit: pending。

## 09-24 三例特征 smoke 验收

- Validation: 三例各32真实图块全部EXIT_CODE=0；逐NPZ有限768维、匹配距离、ID唯一性和计数检查通过；候选特征469/205/602。
- Artifacts: panel1_three_case_smoke_0924_134324_{2017-23800D,2019-00538F,2022-39941E}，位于semantic_phase3。
- Docs record: docs/运行记录/特征缓存/运行记录_Panel1三例候选特征全量提取_0924_134324.md。
- Status: 启动三例全量候选缓存；不视为标签已复核或训练已完成。

## 09-24 14:00:00

- Purpose: 设计 Panel1 图块内细胞交叉注意力及同条件对照。
- Changed: docs/建议/模型与训练/建议_Panel1图块内细胞交叉注意力_0924_140000.md。
- Validation: 核对扩量缓存 14,984 个细胞具有 patch/HE 坐标/空间 split；补充集 500 个细胞具备相同字段。
- Artifacts: 待运行。
- Docs record: docs/建议/模型与训练/建议_Panel1图块内细胞交叉注意力_0924_140000.md（建议）。
- Commit: pending。

## 09-24 14:12:00

- Purpose: 实现 Panel1 目标细胞与同图块其他细胞的交叉注意力，核验128/128+256双视野效果。
- Changed: benchmarks/v5/tools/train_crc_panel1_patch_attention.py；建议及实验记录各一份。
- Validation: py_compile通过；真实两轮smoke通过；三种子×两输入视野×两模型全部完成EXIT_CODE=0，空间组核对、特征有限值及无邻居通路检查通过。
- Artifacts: /nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_patch_attention_{full_0924_140700,seed23_0924_141500,seed47_0924_141500,dual_seed11_0924_141000,dual_seed23_0924_141000,dual_seed47_0924_141000}。
- Docs record: docs/建议/模型与训练/建议_Panel1图块内细胞交叉注意力_0924_140000.md；docs/运行记录/模型实验/运行记录_Panel1图块细胞注意力对照_0924_141000.md。
- Result: 补充test小幅改善，原test无稳定提升；保留可选结构，暂不替换默认头。
- Commit: pending。

## 09-24 16:03:26

- Purpose: 检索并对照 2024–2026 H&E 单细胞识别与分子监督研究。
- Changed: docs/建议/模型与训练/建议_Panel1细胞识别近期文献对照_0924_160326.md。
- Validation: 核对 CytoFormer、Li 等、MIPHEI、GHIST、CGT、Hist2Cell 原文定义及评价指标；标注不可直接比较的口径。
- Artifacts: 无。
- Docs record: docs/建议/模型与训练/建议_Panel1细胞识别近期文献对照_0924_160326.md（建议）。
- Commit: pending。

## 10-08 13:34:00

- Purpose: 评估Panel1细胞128像素与周边256像素特征的自然聚类及融合权重。
- Changed: benchmarks/v5/tools/evaluate_crc_panel1_dualview_clustering.py；docs/运行记录/模型实验/运行记录_Panel1双视野细胞特征聚类_1008_132510.md。
- Validation: `python -m py_compile`、真实双臂权重smoke及五种表示×K=5/10/20完整评估均通过，正式日志均`EXIT_CODE=0`；空间组与特征有限值检查通过。
- Artifacts: /nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_dualview_cluster_{full_1008_132510,weighted_1008_133128}。
- Docs record: docs/运行记录/模型实验/运行记录_Panel1双视野细胞特征聚类_1008_132510.md（运行记录）。
- Result: 原始双视野等权或3:1/9:1融合未稳定优于单用细胞128视野；暂不接入Phase3。
- Commit: pending。

## 10-08 14:05:52

- Purpose: 复核Panel1三例候选细胞特征、源mIF标签与训练可用性。
- Changed: benchmarks/v5/tools/audit_crc_panel1_candidate_integrity.py；docs/运行记录/数据审计/运行记录_Panel1三例候选细胞标签与特征复核_1008_134723.md。
- Validation: 三例379,756条缓存一致性全量通过；2019/2022共177,977条源表SHA256、ID/坐标、六marker数值阈值和标签规则全量通过，日志均EXIT_CODE=0；真实图块smoke、py_compile通过。2017源表NFS5超时，未宣称复核完成。
- Artifacts: /nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_candidate_integrity_cache_full_1008_135230；panel1_candidate_marker_numeric_full_1008_140200。
- Docs record: docs/运行记录/数据审计/运行记录_Panel1三例候选细胞标签与特征复核_1008_134723.md（运行记录）。
- Result: 缓存内部无误；可作弱标签探索训练，但缺独立细胞配准QC，三例全量不能作已验证训练/测试真值。
- Commit: pending。

## 10-08 14:18:51

- Purpose: 生成三例Panel1 H&E与mIF DAPI配准的一张人工抽查总览图，并澄清2017原始标注的独立NFS挂载。
- Changed: benchmarks/v5/tools/visualize_crc_panel1_three_case_registration.py；docs/运行记录/数据审计/运行记录_Panel1三例配准一图抽查_1008_141000.md。
- Validation: 真实图块smoke及三例六图块正式绘图通过；六组512×512原图、84个展示细胞坐标和≤4px匹配距离核对通过；py_compile与git diff --check通过。
- Artifacts: /nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_registration_contact_sheet_1008_141725/three_case_registration_contact_sheet.png；同目录displayed_cells.csv。
- Docs record: docs/运行记录/数据审计/运行记录_Panel1三例配准一图抽查_1008_141000.md（运行记录）。
- Result: 三例各强/低对照图块均在单张总览图中展示，供人工确认局部配准；不以抽样图计算标签准确率。
- Commit: pending。

## 10-08 14:48:27

- Purpose: 比较Panel1 16/32/64/128像素单细胞视野并制作同细胞尺寸可视化。
- Changed: benchmarks/v5/tools/probe_crc_panel1_small_cell_crops.py；benchmarks/v5/tools/visualize_crc_panel1_small_crops.py；docs/运行记录/模型实验/运行记录_Panel1小视野细胞特征对照_1008_143551.md。
- Validation: 真实50细胞smoke和3,177细胞正式提取/同名单分类评估通过，正式日志EXIT_CODE=0；核掩膜保留、源图/mask哈希、空间组隔离及有限768维特征核对通过；可视化六行真实图完成，py_compile与git diff --check通过。
- Artifacts: /nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_small_cell_crop_full_1008_143551；panel1_small_crop_visual_1008_144519。
- Docs record: docs/运行记录/模型实验/运行记录_Panel1小视野细胞特征对照_1008_143551.md（运行记录）。
- Result: 16px有734/3,177核掩膜保留不足90%；32px原val macro-F1较128高0.024，但均衡补充test几乎持平，暂不替换默认输入。
- Commit: pending。
