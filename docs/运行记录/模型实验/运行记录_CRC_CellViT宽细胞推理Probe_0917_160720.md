# CRC CellViT++ 宽细胞推理 Probe

- 时间戳：`0917_160720`
- 目的：不再将 H&E 单细胞任务设为 CD4_T/CD8_T/Treg 等精细免疫身份；使用官方 CellViT++ 权重，评估其在当前 SRRSH CRC H&E 上的核实例分割、通用宽细胞类别和 cell embedding 是否可作为 FewClick Phase 3 的候选细胞证据。
- 官方模型：PyPI `cellvit==1.0.9`；官方自动缓存 `CellViT-SAM-H-x40-AMP.pth`、`CellViT-256-x40-AMP.pth` 和 taxonomy classifier，来源为官方 Zenodo record `15094831`。第一轮采用 SAM 与 `pannuke` taxonomy。
- 默认宽类别：`Neoplastic`、`Inflammatory`、`Connective`、`Dead`、`Epithelial`。这些是预训练分类头的通用预测，不是 SRRSH 医生审核的精细 mIF 身份标签。
- 输入：只读 `medical6` 镜像中三个真实 CRC H&E 图块；源 WSI 的审计 MPP 为`0.2457 µm/px`，与 CellViT 0.25 µm/px 输入设定匹配。注册图块的尺度仍须用 Cellpose核直径/几何一致性复核。
- 输出：本机 `.cache/semantic_phase3/cellvit_crc_smoke_0917_160720/`，包括每图块实例数、实例几何、类别概率/标签、embedding（若官方图输出支持）及可视化叠加。
- 算力：独立 `cellvit` conda 环境；GPU1；最多16 CPU线程；权重缓存显式设为项目本机 cache，不使用不稳定 NFS5。
- 核验：必须完成真实模型加载与三个图块推理；核实例数须为正、坐标在图像边界内、类别均属于官方 taxonomy，并与现有 Cellpose 检测数量/尺度作只读比较。此步骤不训练新分类器、不进入 fusion。
- 接受标准：若模型运行稳定且核实例几何与现有检测在合理范围一致，则输出可进入“宽细胞证据”数据契约设计；是否能改善组织分割仍须以后续病例隔离消融决定。
