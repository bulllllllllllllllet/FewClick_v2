# 数据路径文档

# 一、新染片子

## 1. 原始数据根目录

```text
/nfs-medical3/data/mIF_IHC_HE_ER_PR_202608/
```

| 批次目录                                              | 内容                                       | 扫描模式                                                       | 格式 |
| ----------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------- | ---- |
| `20260817-孙嘉炜（王振阳）-20张-IF三-GTP`           | mIF：**KI67** + PANCK + P63          | 荧光（滤片 DAPI/CY5/SpGreen/SpGold；DAPI 信号已像素级验证 ✓） | mrxs |
| `20260820-孙嘉炜（王振阳）-20张-IF三-GTP`           | mIF：**ER** + PANCK + P63            | 荧光                                                           | mrxs |
| `20260901-孙嘉炜（王振阳）-20张-IF三-GTP`           | mIF：**PR** + PANCK + P63            | 荧光                                                           | mrxs |
| `20260904-孙嘉炜（王振阳）-20张-HE-CTT-2606925`     | 明场扫描（批次名叫 HE）                    | 明场 RGB（无荧光滤片，已核实）                                 | mrxs |
| `20260909-孙嘉伟-22张-IHC-CTT-2506925/{ER,PR,Ki67}` | 显色 IHC（每片单 marker）                  | 明场 RGB                                                       | mrxs |
| `20260919-孙嘉炜（王振阳）-10张-HE-CTT-2506925`     | **HE 补扫（③ 批次，10/10 例）**     | 明场 RGB（2026-09-24 到位）                                    | mrxs |
| `20260922-孙嘉伟(王振阳)-10张-IHC-CTT-2506925`      | **IHC-PR 补扫（④ 批次，10/10 例）** | 明场 RGB（2026-09-24 到位）                                    | mrxs |

mIF 荧光通道与滤片对应（Slidedat.ini + 像素级双重核实）：SPGOLD=金通道（第一标记物 KI67/ER/PR）、SPGREEN=绿通道（PANCK）、CY5=远红（P63）、DAPI=蓝（核复染）。`phase0_mif_stats.csv`）。注意 **CY5/P63 信号普遍极弱**（p99 仅 2-12，多数片近空通道），若阶段一需要肌上皮/P63 信息需提前评估。荧光片 mpp=0.2031（明场 0.2499），跨模态配准必须显式换算。

## 2. 每个病例的切片-染色对应关系

### 2.1 染色设计（已确认部分 + 待确认部分）

一个病例（D26-XXXXX）的蜡块切出多张连续切片，按切片号分配染色。**mIF 部分已确认**：

```text
切片 1, 2 → mIF ER panel   (ER + PANCK + P63)      ✓ 20260820 批次，文件夹名 "D26-XXXXX 1 5 ER..." / "D26-XXXXX 2 5 ER..."
切片 3, 4 → mIF PR panel   (PR + PANCK + P63)      ✓ 20260901 批次，文件夹名 "D26-XXXXX 3 PR..." / "D26-XXXXX 4 PR..."
切片 5, 6 → mIF KI67 panel (KI67 + PANCK + P63)    ✓ 20260817 批次，文件夹名 "D26-XXXXX 5 KI67..." / "D26-XXXXX 6 KI67..."
```

注：ER 批次的 "1 5"/"2 5" 双数字中，前一个数字(1/2)为切片号，尾部的 "5" 为位点/蜡块号（与明场名结尾的 ⑤ 对应）。

**明场部分（HE 与 IHC）的切片归属（用户已确认同片复染）**：圆圈数字 = 切片号（①=1、②=2、③=3、④=4、⑤=5、⑥=6；③④ 两批为 2026-09-24 到位的补扫，同片关系由命名体系推断）。每病例 6 张切片的完整染色经过（奇数片复染 HE、偶数片复染 IHC，完全对称）：

| 切片 | 染色经过                        | 扫描件                                    | 阶段一用途                                |
| ---- | ------------------------------- | ----------------------------------------- | ----------------------------------------- |
| 切1  | mIF-ER → 复染 HE-B（①ER名）   | mIF-ER "1 5" + HE "①…⑤ER…"            | 无                                        |
| 切2  | mIF-ER → 复染 IHC-ER（②）     | mIF-ER "2 5" + IHC "②…⑤ER"             | **核心**（模型输入+PanCK监督+mask） |
| 切3  | mIF-PR → 复染 HE（③补扫）     | mIF-PR "3" + HE "③"（20260919 批次）     | 无                                        |
| 切4  | mIF-PR → 复染 IHC-PR（④补扫） | mIF-PR "4" + IHC "④ PR"（20260922 批次） | 阶段二（PR 阳性判定）                     |
| 切5  | mIF-KI67 → 复染 HE-A（ki67名） | mIF-KI67 "5" + HE "⑤ki67…"              | **肿瘤标签载体**（专家标注）        |
| 切6  | mIF-KI67 → 复染 IHC-Ki67（⑥） | mIF-KI67 "6" + IHC "⑥Ki67"               | 阶段二                                    |

**HE 与 IHC 不同片**（切5 vs 切2，隔 3 张片），配准方案见 `STAGE1_DETAILED_DESIGN.md`（同片配准 A 类 + 跨片配准 B 类，已废除 HE 中介）。

### 2.3 病例号/文件名笔误重映射（已定案，权威表 = 服务器 manifest）

**权威重映射表**：`/nfs-medical3/sxz/ihc_tumor_stage1/stage1_manifest.csv`（列 `canonical_id, case, batch, modality, panel, slide_no, original_path, rescan, notes`——每行原始路径到规范 ID 的映射与笔误备注）。**代码一律使用 canonical_id，不要解析原始文件名猜病例号**。

| 原始文件名中的编号                                | 所在批次          | 重映射为（已定案）    | 备注                                                                         |
| ------------------------------------------------- | ----------------- | --------------------- | ---------------------------------------------------------------------------- |
| `D26-17705 6`（双空格）                         | 20260817 mIF-KI67 | D26-17005             | 病例号笔误                                                                   |
| `D26-26636 5` / `5 6` 两张                    | 20260817 mIF-KI67 | D26-22636（s5/s6）    | 病例号笔误                                                                   |
| `D26-22623 1 5` / `2 5`                       | 20260820 mIF-ER   | D26-22636（s1/s2）    | 病例号笔误；**zarr 资产目录名保留原名 `D26-22623_2.zarr`（见 §7）** |
| `D26-22027 ⑥ Ki67`                             | 20260909 IHC-Ki67 | D26-20027（s6）       | 病例号笔误                                                                   |
| `D26-XXXXX⑤ ki57 …`（HE 批次全 10 张）        | 20260904 HE       | ki57 →**ki67** | 染色名笔误（文件名级别，病例号正确）                                         |
| `D26-17005 -14 NG PR` / `D26-19780 -14 NG PR` | 20260909 IHC-PR   | （无重映射）          | 命名规则不明，病例号正确                                                     |

## 3. HE 金字塔 tif 转换资产（标注平台的输入）

专家标注平台（imageInform，project 504）读不了 mrxs，因此 `/nfs-medical3/zyh/20260904_HE_CTT_2606925_pyramidal_tiff_20260914/` 里放了 20260904 批次 20 张明场 mrxs 的**金字塔 tif 转换版**（多分辨率分层大图，平台可缩放浏览），专家实际在这批 tif 上画标注：

```text
├── ①D26-16989⑤ ER570NG ....tif        # ① 家族 10 张（ER-panel 名）
├── D26-16989⑤ ki67 570NG ....tif      # ki67 家族 10 张（标注平台上传的是这批）
├── manifest.json / conversion_status.txt / CONVERSION_COMPLETE.txt
├── previews/ reports/ validation_deepzoom/ interrupted_batch/ work_tmp/
└── 9.12_tif.txt / 9.12(2)_tif.txt     # 两批转换清单（① 家族 / ki67 家族）
```

**重要**：标注 JSON 的坐标 = ki67 家族 tif 的 level-0 像素坐标。标签构建时以这批 tif（或其对应 mrxs）为 HE 坐标基准。

## 5. 运行环境与 mrxs 读取路线（8x5090，2026-09-18 实证）

| 环境                            | 用途                     | 说明                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/nfs-medical3/sxz/miniconda` | conda 根                 | conda 4.10.3                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `aligner` / `aligner_2`     | register 算法既有环境    | openslide 1.4.2，明场可读                                                                                                                                                                                                                                                                                                                                                                                                               |
| `ihc_ERPR_recognize`          | **本项目主力环境** | py3.11 + openslide-python(**libopenslide 4.0.1**) + tifffile/numpy/pillow(jpg2000✓)/matplotlib/scikit-image/scipy/simpleitk/opencv + imagecodecs                                                                                                                                                                                                                                                                                 |
| `biyesheji`                   | phase2 GPU 精配准        | py3.9 +**torch 2.8.0+cu128（RTX 5090 sm_120 可用）** + zarr 2.18.2 + scikit-image + simpleitk；注意：自带 libopenslide 3.4.1 拒读缺 `OBJECTIVE_MAGNIFICATION` 键的 mrxs（19049/20027/21412），需 `LD_LIBRARY_PATH=/nfs-medical3/sxz/miniconda/envs/ihc_ERPR_recognize/lib` 借 4.0.1；numcodecs 0.12 不认 zarr 资产的 Zstd(checksum=) 元数据，读取脚本需运行时丢弃该键（`phase2_extract_patches.py` 顶部有现成 monkeypatch） |

## 7. 阶段一加工产物路径速查（服务器 8x5090，2026-10-10 核实更新）

以下路径均已逐一核实存在。`$S = /nfs-medical3/sxz/ihc_tumor_stage1`。

### 7.1 mIF 解码结果（mrxs → zarr）

```text
$S/data/mif_zoom0_zarr/D26-XXXXX_2.zarr     # 10 例 ER 切2（A 类配准/训练主力）
```

- 内容：zarr v2，每例内含 `DAPI/ P63/ PANCK/ ER/` 四通道 array（uint8，chunks 2048²，zstd-3）+ `meta.json`（tiles/skipped）+ `preview/`。
- 覆盖：16989/17005/18564/19049/19057/19780/20027/21412/22407 共 9 例标准名 + **`D26-22623_2.zarr`（= D26-22636，目录名保留原始笔误名，勿当成独立病例）**；另有两份 19057 归档：`D26-19057_2.zarr.wrong_slice_20260924`（错片）与 `D26-19057_1.zarr.slice1_unused`（切1，未用）。
- 转换器与通道映射：`phase14_convert.py`（jpype + MiraxReader）；通道映射按 Slidedat.ini `STORING_CHANNEL_NUMBER` 解析——DAPI=FL0 plane0、P63(CY5)=FL0 plane1、PANCK(SpGreen)=FL0 plane2、ER(SpGold)=FL1 plane0。

### 7.2 医生标注

| 版本                                   | 路径                                                                    | 说明                                                                                                                                                                           |
| -------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **服务器固化版（训练用金标准）** | `$S/stage4_train_v1/annotations/D26-XXXXX__ER.json`（9 例，21412 无） | **chain 固化坐标**（已迁移到 IHC 切2 lvl0 坐标系）；结构 `shapes[].{metadata.label=肿瘤区域/非肿瘤区域, geometry.paths}`；评估口径 = 正类肿瘤区域内 / 负类非肿瘤区域内 |
| 原始平台导出（本地）                   | `data_info/20260904-孙嘉炜（王振阳）-20张-HE-CTT-2606925.json`        | 分组2，9 张已标；坐标 = HE ki67 家族金字塔 tif level-0 像素（画在 HE 切5 上，**未迁移**）                                                                                |
| 服务器副本                             | `$S/data_info_labels.json`                                            | 上行本地 JSON 的服务器副本                                                                                                                                                     |
| 迁移 QC 图                             | `$S/stage2_labels_v1/D26-XXXXX__label_transfer.png`（9 例）           | 标注迁移到 IHC 画布的叠加目检图                                                                                                                                                |

### 7.3 配准参数（变换 pkl）

**A 类：IHC-ER 切2 ↔ mIF-ER 切2（DAPI 主导）**——10 例全：

```text
$S/phase1_regA_v3/D26-XXXXX/D26-XXXXX__regA_dapi_transform.pkl
```

- 归档：`$S/phase1_regA_v3/D26-19057.wrong_slice_20260924/`（19057 错片期产物，勿用）；更早的 `phase1_regA/`（3 例）与 v2 为历史版本。
- pkl 内含：T_A（IHC thumb → mIF DAPI thumb 变换）、canvas/pad 信息、IHC 侧坐标链字段。

**B 类：IHC-ER ↔ HE（标注迁移用）**——两种口径：

```text
# 直配（HE↔IHC 一步）：8 例有效
$S/stage2_regB_v1/D26-XXXXX/D26-XXXXX__regB_transform.pkl

# 二阶段链式（HE→mIF→IHC 两步变换）：仅 2 例——D26-16989、D26-18564（直配质量不足改走链式）
$S/stage2_regB_v1_chain/D26-16989/D26-16989__regB_chain_transform.pkl
$S/stage2_regB_v1_chain/D26-18564/D26-18564__regB_chain_transform.pkl
```

- **二阶段（chain）pkl 结构**：存 T_B（HE → mIF DAPI thumb）+ `rega_pkl` 字段引用对应 A 类 pkl 路径；标注迁移时**两步 TransformPoint**（HE → mIF → IHC），脚本 `$S/stage2_scripts/phase1b_register_he_chain.py`。
- ⚠️ 16989/18564 在 `stage2_regB_v1/` 下也有直配版 pkl，但**以 chain 版为准**（其直配版是被淘汰的旧产物）。
- **这两张二阶段配准后落到 IHC 画布的医生标注框**：已直接固化为 §7.2 的训练金标准——`$S/stage4_train_v1/annotations/D26-16989__ER.json` 与 `$S/stage4_train_v1/annotations/D26-18564__ER.json`（无单独的中间标注框文件；chain 迁移 QC 见 `$S/stage2_labels_v1/` 对应 png 与 `$S/stage2_regB_v1_chain/*/…_regB_chain_overview.png`）。

# 二、模型之前训练用的片子（configs 目录，旧队列 BNK-26-1258-4，服务器 /nfs-medical6）

configs 目录共 7 个文件：4 个 local 实配 + 3 个 example 模板。核心是 **11 张 IHC 片**（`current_dataset_11slide.local.json`，版本 `tumor_epithelial_6train_5test_v1`，策略 "two slides per marker"），服务器权威版在 `/nfs-medical6/test1/align10/tumor_epithelial_6train_pipeline_20260726/configs/cases_11slide.json`。

### 1. 11 张划分总表

| Marker | 角色  | Case           | 患者      |
| ------ | ----- | -------------- | --------- |
| ER     | train | 4A_25-003745_1 | 4A-003745 |
| ER     | train | 4A_25-007613_1 | 4A-007613 |
| ER     | test  | 4C_25-001323_6 | 4C-001323 |
| ER     | test  | 4C_25-003270_6 | 4C-003270 |
| PR     | train | 4B_25-003297_6 | 4B-003297 |
| PR     | train | 4C_25-001323_1 | 4C-001323 |
| PR     | test  | 4A_25-003745_6 | 4A-003745 |
| PR     | test  | 4B_25-001333_1 | 4B-001333 |
| KI67   | train | 4B_25-003297_1 | 4B-003297 |
| KI67   | train | 4C_25-003270_1 | 4C-003270 |
| KI67   | test  | 4B_25-001333_6 | 4B-001333 |

划分按"每 marker 2 训 2/1 测"的片子级策略；实际只来自 **6 个患者**，其中 4A-003745、4C-001323、4C-003270 三个患者**横跨 train/test**（不同 marker 片）——此为旧队列已知口径，评估结论按片子读即可，但做患者级泛化外推时需记住这一点。

### 2. 每张片子的数据构成（11slide config 逐 case 字段）

- **ihc_slide**：`/nfs-medical6/data/BNK-26-1258-4/{ER,PR,ki67}/*.svs`
- **annotation_json**：`/nfs-medical6/gkw_data1/json/{ER,PR,Ki67}/`（dr228–dr239 共 11 份专家标注）
- **cell_table**：11 张统一用 `epithelial_multiclass_v1_20260724/features/.../cell_features_mpp_corrected.csv`（InstanSeg 分割 + MPP 修正特征，同一版本）
- **epithelial_table**：⚠️ 版本不统一——6 张训练片用 `training_six_slide_stable_inference` / `training_six_slide_marker_aware_inference` 的上皮预测；5 张 test 片分别用 `marker_conditional_acinar_exact_final_unseen_v19`（ER 2 张）、`cell_graph_guarded_inference_v34_minimal_isolate_pr`（PR 2 张）、`marker_conditional_acinar_exact_fourth_unseen_v24`（KI67 1 张）——test 片上皮表来自各自 marker 当时的最终推理版本

### 3. 其余 3 个配置的作用

| 文件                                              | 内容                                                                                                                    |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `current_alignment_sources_original.local.json` | 首批 3 张训练片（003745_1__ER、003297_6__PR、003297_1__KI67）的 mIF↔IHC 配准来源（pair_dir + mapping_csv + clean_ini） |
| `current_alignment_sources_expanded.local.json` | 扩增 3 张训练片（007613_1__ER、001323_1__PR、003270_1__KI67）的配准来源（来自 epithelial_training_expansion_20260725）  |
| `current_panck_sources.local.json`              | 同 6 张训练片的 PanCK 上皮监督表（ratio100 口径，首批 3 + 扩增 3）                                                      |

即：**旧队列真正的"训练片"是 6 张**（11 张里 role=train 的），配准与 PanCK 监督资产也只覆盖这 6 张；另 5 张是冻结测试片。


## 旧队列 6 张训练片的配准路径明细

**首批 3 张**（`current_alignment_sources_original.local.json`）：

| Case                 | pair_dir（全局配准产物）                                                                                         | mapping_csv（微 patch 对齐映射）                                                                                                                                     | clean_ini（清洗版 mIF ini）                                                                            |
| -------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 4A_25-003745_1__ER   | `/nfs-medical6/test1/align10/ihc_mif_v5logic_outputs/global/4A_25-003745_1__ER`                                | `.../ihc_mif_fast_neighbor_alignment_20260721/outputs/sparse_epithelial_source_refresh_v17_neighbor_growth_20260722/4A_25-003745_1__ER/accepted_micro_patches.csv` | `.../ihc_mif_fast_neighbor_alignment_20260721/configs/cleaned_ini/4A_25-003745_1_Slidedat.clean.ini` |
| 4B_25-003297_6__PR   | `.../ihc_mif_global_alignment_iteration_20260711/rerun_v5_global_pr_4b0032976_cleanini/4B_25-003297_6__PR`     | `.../sparse_epithelial_source_refresh_v18_neighbor_growth_allmarkers_20260722/4B_25-003297_6__PR/accepted_micro_patches_gold.csv`                                  | `.../ihc_mif_global_alignment_iteration_20260711/cleaned_ini/4B_25-003297_6_Slidedat.clean.ini`      |
| 4B_25-003297_1__KI67 | `.../ihc_mif_global_alignment_iteration_20260711/rerun_v5_global_ki67_4b0032971_cleanini/4B_25-003297_1__KI67` | `.../sparse_epithelial_source_refresh_v18_neighbor_growth_allmarkers_20260722/4B_25-003297_1__KI67/accepted_micro_patches_gold.csv`                                | `.../ihc_mif_global_alignment_iteration_20260711/cleaned_ini/4B_25-003297_1_Slidedat.clean.ini`      |

（`.../` 均为 `/nfs-medical6/test1/align10/` 前缀）

**扩增 3 张**（`current_alignment_sources_expanded.local.json`，统一根 `/nfs-medical6/test1/align10/epithelial_training_expansion_20260725/`）：

| Case                 | pair_dir                                                  | mapping_csv                                                                                     | clean_ini                                                 |
| -------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 4A_25-007613_1__ER   | `outputs/global/4A_25-007613_1__ER`                     | `outputs/sparse_epithelial_expansion_v1/4A_25-007613_1__ER/accepted_micro_patches_gold.csv`   | `configs/cleaned_ini/4A_25-007613_1_Slidedat.clean.ini` |
| 4C_25-001323_1__PR   | `outputs/global_pr_1323_constrained/4C_25-001323_1__PR` | `outputs/sparse_epithelial_expansion_v1/4C_25-001323_1__PR/accepted_micro_patches_gold.csv`   | `configs/cleaned_ini/4C_25-001323_1_Slidedat.clean.ini` |
| 4C_25-003270_1__KI67 | `outputs/global_ki67_3270_fresh/4C_25-003270_1__KI67`   | `outputs/sparse_epithelial_expansion_v1/4C_25-003270_1__KI67/accepted_micro_patches_gold.csv` | `configs/cleaned_ini/4C_25-003270_1_Slidedat.clean.ini` |

细节注意：① mapping 文件名不完全一致——003745_1 是 `accepted_micro_patches.csv`（无 `_gold` 后缀），其余 5 张均带 `_gold`；② 首批 3 张的产物散落在三个不同批次目录（v5logic / global_alignment_iteration / fast_neighbor_alignment），扩增 3 张则统一在 expansion 一个目录下。

顺带补上同 6 张训练片的 PanCK 监督表路径（`current_panck_sources.local.json`）：首批 3 张在 `.../epithelial_multiclass_v1_20260724/panck_stratified_qc_v4_ratio100/<case>/tables/panck_epithelial_supervision.csv`，扩增 3 张在 `.../epithelial_training_expansion_20260725/outputs/panck_supervision_expansion_v1_ratio100/<case>/tables/panck_epithelial_supervision.csv`。
