# CRC Panel1 缓存训练 Gate

- 时间戳：`0922_215417`。
- 问题：Panel1 的结构细胞区域组成能否从冻结 Phase 2 token 与细胞特征中学到超越训练集常数先验的信号。
- 输入：完整冻结 token contract `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_multpanel_token_contract_0922_201100/`；空间隔离 manifest `0922_195841`，train/val/test=`861/188/89`，封存 test 不用于选 checkpoint。
- 目标：`epithelial`、`blood_endothelial`、`lymphatic_endothelial`、`sma_stromal`、`fap_caf` 的已映射区域组成。仅采用与 `observed_composition_loss` 一致的 token：`observed_cells>=1`、类别计数和大于零、`coverage>0`。
- 先验与口径：训练时均匀抽 patch，patch 内按 coverage 加权；训练集常数先验在 test 的 CE 为 `1.520706`，均匀先验为 `1.609438`，按相同口径的 test 目标熵为 `1.219718`。此前 2-step 冒烟 test CE=`1.660381`，不能视为有效结果。
- 代码：`benchmarks/v5/phase_3_cell_region/tools/train_crc_panel2_cached.py --panel panel1`。本次补充在 step 0 保存初始验证 CE、按训练采样口径拟合常数先验，并在最后一次封存 test 同时输出模型与先验 CE；不读取 test 选择模型。
- 运行：GPU `cuda:1`，tmux `crc_panel1_gate_0922_215417`；seed `17/29/43`，各 `200` steps、每 `50` steps 验证。输出 `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_panel1_cached_train_0922_215417_{17,29,43}/`，日志同根并追加 `EXIT_CODE`。
- 验收：三个 seed 都产生有限损失和 checkpoint；至少两个 seed 的封存 test CE 优于训练集常数先验 `1.520706`，且平均 CE 优于它，才扩大到 1,000 steps。否则停止该 frozen-token 路线，审计监督对应或改为可训练图像表示。
- 冒烟：`cuda:1` 上真实 contract 两步完成，输出 `crc_panel1_cached_train_0922_215600`；`initial_val_ce=1.6605476`、`best_val_ce=1.6136451`、`train_prior_test_ce=1.5207058`、`test_ce=1.6603809`，新增指标字段与独立复算一致。
- 完成：seed `17/29/43` 的 test CE=`1.4522115/1.4738188/1.5077332`，均优于常数先验 `1.5207058`；best val CE=`1.5331774/1.5142035/1.4809156`，所有日志 `EXIT_CODE=0`。达到预注册升级条件。
- 状态：完成；后续 1,000-step 确认训练见 `运行记录_CRC_Panel1缓存长程训练_0922_215800.md`。
