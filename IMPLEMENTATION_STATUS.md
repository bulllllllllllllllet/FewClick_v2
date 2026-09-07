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
