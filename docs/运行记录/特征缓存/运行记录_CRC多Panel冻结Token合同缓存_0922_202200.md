# CRC 多 Panel 冻结 Token 合同缓存

- 时间标识：smoke `0922_201000`；全量 `0922_201100`。
- 输入：空间隔离 manifest `0922_195841`；冻结 Phase 2 checkpoint、已审核 H&E/XCell cache、双 panel 网格 target。
- 输出：`/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_multpanel_token_contract_{0922_201000,0922_201100}/`。
- 计算：先以一条真实 train sample 在 GPU1 smoke，检查 token、cells、assignment、双 panel valid target 和写后重读；通过后再按 split 全量生成。
- 验收：每个缓存条目保存 Phase 2 token、cell feature、assignment、panel_1/panel_2 counts/coverage/valid mask；数值有限，且 split/空间组只来自 manifest。
- Smoke 完成：`EXIT_CODE=0`；一条真实 train patch 成功写入并重读，包含 `[64,256]` token、`[294,74]` cell feature、`[64,294]` assignment 和双 panel target。metadata 记录空间 manifest SHA256 `566b487b19962411d93a91262e51d4d1643eecbbf4dd98981e91082b1a6c453b`。
- 全量完成：`/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_multpanel_token_contract_0922_201100/` 含 1,138 个 patch `.npz` 和 index；`metadata.json` 为 `status=complete`，日志 `crc_multpanel_token_contract_0922_201100.log` 为 `EXIT_CODE=0`。split 为 train/val/test=`861/188/89`。每个 patch 独立 `.npz`，中断不会覆盖既有文件。
