# Phase 3 与 CRC Panel1 联合分割对照 Smoke

- 时间戳：`0922_221900`。
- 问题：在同一 GDPH 分割训练条件下，加入已通过 Panel1 缓存 gate 的结构细胞组成辅助损失，是否改变验证区域分割准确率。
- 输入：冻结 Phase2、v4 Phase3 checkpoint、GDPH train/val 各 8 patch；CRC 空间隔离 token contract `0922_201100` 的 train/val 各 12/6 patch。CRC 只作为辅助组成观测，不当作 GDPH 像素标签。
- 对照：`seg_only` 与 `seg_plus_panel1` 完全复用 seed、初始化、GDPH patch 顺序和步数；仅联合 arm 添加 Panel1 observed-composition CE，权重 `0.1`。
- 运行：GPU `cuda:1`，12 steps，seed `17`；输出 `/nfs-medical3/zyh/fewclick_v2/semantic_phase3/phase3_seg_panel1_joint_smoke_0922_221900/`。验收为两个 arm 有限，GDPH val region accuracy、CRC val Panel1 CE 均可计算；此 smoke 不解释点击效果。
- 首次尝试：输出目录 `phase3_seg_panel1_joint_smoke_0922_221900/` 已创建但在 loader 导入前因 aligner 的 `libtiff.so.6` 与 `libjpeg` ABI 冲突失败，未执行模型、未生成 metrics；保留该目录，不覆盖。
- 修正重试：仅在运行时设置 `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`，确认 `pyvips.version(0)=8` 后，使用新输出时间戳 `0922_222500` 重跑；代码、数据和训练配置不变。
- 完成：在 `LD_LIBRARY_PATH` 同时保留 aligner CUDA 库和系统图像库的真实 loader 环境下，`0922_222500` 的 12-step smoke 完成。`seg_only`/`seg_plus_panel1` GDPH val region accuracy 均为 `0.91321499`；CRC val Panel1 CE 为 `1.6666356/1.6609640`，358 个有效 token。无主任务退化，但该长度不足以下结论；进入三 seed gate。
- 状态：完成。
