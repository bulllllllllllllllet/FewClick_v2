# FewClick supplementary experiment goal

Updated: 2026-10-25T16:02:04.456733+08:00. Native thread goal: **active**. G3 is complete; the overall objective is incomplete.

## Objective

以当前三物理视野FewClick完成版本化补充实验：公平性审计与固定本图/支持查询协议，SAM/WSI-SAM公共及同域适配、闭集参考和受控FewClick；随后完成统计重复、细胞/机制消融、严格概念与外部泛化、TLS、整片与效率，以及逐审稿challenge证据闭环。不要求FewClick获胜，不将历史J5比较称等总监督架构实验。

## Resume here

1. Read [registry](docs/运行记录/模型实验/公平对比进度_1025_124208.json) and [current record](docs/运行记录/模型实验/运行记录_公平对比协议固定与首批基线_1025_124208.md).
2. Inspect actual status/logs and ps for active PIDs. Completed jobs moved to registry.completed_jobs, never restart them.
3. Next independent work: **G4 clean G3 weights/SAM-FT feature transfer**. Retained diagnostics and ongoing MultiverSeg do not replace it.
4. New source may live outside locked snapshots. Do not edit running/queued trainer, model, sampler or inference files listed in contracts.

Full prior goal/history snapshot preserved: `/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/continuity_1025_160204/goal_before_G3_closeout_1025_160204.md` (SHA256 b166a50ea75b0c9c81aed0525b9c48b847b3391e525af7a61daf9afa17bd28dd). Detailed immutable results and run records remain unchanged.

## Route

| Phase | Current status | Required closure |
|---|---|---|
| G0 | complete | Historical four arms, matched incremental tasks; retained J5, mixed120 tasks |
| G1 | complete, qualified | Frozen inputs/points/selection/metrics and prior ledger; no equal-total-pretraining claim |
| G2 | complete | Local public SAM/WSI-SAM6040 each |
| G3 | complete, single seed | Five10k/40k trains and6040 scores each; independent full parity audit |
| G4 | running, incomplete | Clean adapted transfer still needed; native recent context and full statistics pending |
| G5 | queued, CPU gates passed | Two additional seeds/four methods; native gates, full trains, scores and mean/std pending |
| G6 | running | Seven mechanism trains/full scores; four slots currently active |
| G7 | partial preparation/queued | Strict heldout concepts/TLS and external adapted full scores pending |
| G8 | input gate only | Real WSI cells/model/coverage/stitch/cache/efficiency pending |
| G9 | pending | Original reviewer challenge-to-evidence closure and factual corrections |

## G3 result and acceptance

Fixed302 image-target pairs, five click groups, primary3+2; each adaptation10k updates/40k identical actual tasks. Actual-coordinate SHA `fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe`. Independently read all five training/selected-weight/6040-row outputs; point/GT/count parity and Dice replay passed in `/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/G3_complete_audit_1025_155140.json`.

| Interactive adapted method | Primary Dice | BF1@2µm |
|---|---:|---:|
| SAM-FT | .7019 | .3905 |
| WSI-SAM-FT | .7000 | .3693 |
| FewClick M3D XCell | .5729 | .2388 |
| FewClick M3D no cells | .5640 | .2269 |

Known-class/full-CE reference Dice.5153 is privileged and ignores clicks, separate from the interactive rows. Public0-adaptation SAM/WSI-SAM Dice.4505/.4967. Current evidence does **not** support FewClick superiority on local segmentation. Do not substitute old J5/mixed120-task results or only compare public baselines.

Primary corrected report: `/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/first_local_complete_1025_155355/report_user_provenance_1025_155355.md`. Corrected summary links raw-summary SHA; original reports/runtime snapshots preserved. All75-source-space bootstrap intervals are development-task uncertainty, not independent-patient or training-seed uncertainty. Dense cell delta.008849, CI[-.003059,.022545] includes0; no stable cell/molecular-supervision contribution claim yet.

## Active queues and frozen sources

- G6 coordinator1107196, `mechanisms_1025_134753/mechanism_formal_1025_141300`; currently GPU1 m1_xcell1483487, GPU3 Dense morphology1524245, GPU4 m3_none1337519, GPU5 m3_xcell1363539. Then remaining variants/gate/train/evaluate. Verify real state, not just this snapshot.
- G4 native MultiverSeg CPU4 worker1444317, `transfer_1025_131856/context_full_1025_152401`; predicted sparse18700 rows, full-GT837 single-pair rows.98 pairs excluded from dense reference due incomplete source GT; compare other methods on same837 subset. Native20-row gate passed, input object forbids all GT APIs; query human clicks0. Full score incomplete.
- External cells coordinator1295759, waits G6 GPU4 none+morphology complete and GPU idle, then native one-field gate→ten fields. Output `external_1025_143257/cell_formal_1025_145136`.
- External adapted coordinator1338557, waits real cells/full G3 trains/GPU4 idle, then same-final-checkpoint native8→full680 per method. Output `external_1025_143257/adapted_formal_1025_150010`.
- Repetition coordinator1557332, `replication_1025_153837/formal_1025_154653`, waiting per-GPU G6 completion; GPU4 also waits external adapted completion. Two extra seeds20261026/20261027 preserve exact40k multiset but change order; within seed all four methods share sequence. Dense internal initializer remains fixed. Preparation, real loader/model/loss/update/checkpoint CPU gates passed; **no extra seed GPU train yet**.

Use no GPU0/2. Future GPU jobs must chain dependencies to avoid racing these coordinators. Source contracts are authoritative: fair_comparison/model/mechanism files, active external inference/cell files, context transfer files, replication prepare/gate/queue all frozen. New independent modules/reports are allowed; native gates precede scaling.

## G4/G7/G8 facts

Retained-M3 graph/prototype18700 each and SAM-public feature prototype18700 complete. First two independently audited: presentDice.469993/.343504, absentFPA.045580/.369965; fixed-feature historicalJ5 diagnostic only. SAM-feature independent audit and all transfer cluster statistics still required. Clean local-adapted transfer is not yet implemented.

External IGNITE true256/512/1024µm centre fields10,34 pairs×5 point groups,9 classes. Original splits7train/1val/2test retained as provenance; all external-only to current CRC/TIGER adaptation, never externally fit/select weights. Public SAM/WSI-SAM680 each complete: Dice.457234/.510330, paired ROI CI includes0. PUMA205/OCELOT663 ROI cannot provide1024µm context; no fake padding/repeated-image context. Native cell extraction and adapted/FewClick scores still pending.

WSI strict reader gate passed actual first TLS case54039×65660, vendor CapRes.247925µm (SDK mpp-x/mpp-t aliases of one isotropic value); native1033/2066/4132 boxes and PNG replay/error propagation passed. High-level KFB exception-to-white behaviour is bypassed. No model/full-slide coverage/stitch or efficiency result yet. TLS annotation completeness/hard negatives **unconfirmed**; unannotated pixels are not certified negatives. CRC lymphocyte aggregate is not TLS. Legacy ~.70 TLS result was recall, not Dice; paper correction remains required.

## Frozen fairness decisions

- User2026-10-25: use redesigned current multiscale/data; old140/30/30 and old scores not constraints. Current primary M3/M3D; M1/M1D and shared/hierarchy controls remain.
- Current trainCRC1200/val300, TIGER156/81. Same cases may cross spatial splits; largest4x footprints train/val disjoint. Preserve case/slide metadata and residual dependence; do not impose patient-disjoint splitting.
- G3 generic COCO/new heads, no J5; Dense12 channels are binary-supervised latent features. Historical four-arm J5 exposure is separate. Closed reference gets class ID/full-class labels, not equal information. TIGER ID4 absent, do not claim seven-target coverage.
- XCell trained on **internal data, not current CRC/TIGER**, explicitly confirmed by user. Supersedes prior unknown-source language; upstream public CTransPath pretraining separate. Apply provenance overlay to frozen older snapshots, do not allege overlap from missing archived files.
- Different priors, field counts (SAM fine; WSI-SAM fine+2x; M3 fine+2x+4x), parameters/trainable layers and fixed hyperparameters remain disclosed; same incremental task pool is not equal total training or equal compute.
- Local positive-click segmentation and remote support→query/presence-absence are separate tables. Never copy source coordinates to a different query. No query GT/class ID in interactive inference or oracle candidate/threshold selection.
- Metric fixed>=.5; BF1 Euclidean2µm/anisotropic MPP, GT-boundary eligibility, no predicted boundary scores0. Legacy BF1 denominator correction separate; old Dice unchanged. All five point groups are development robustness, not blind test.
- Native goal remains active until statistics/seeds, all mechanism/generic/heldout/TLS/WSI/efficiency and reviewer evidence requirements are actually verified. Preserve output/version/SHA/PID/log/EXIT_CODE evidence at each milestone.
