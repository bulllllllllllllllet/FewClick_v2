# FewClick supplementary experiment goal

Updated: 2026-10-25T22:49:24.046733+08:00. Native thread goal: **active**. G3 is complete; the overall objective is incomplete.

## Objective

以当前三物理视野FewClick完成版本化补充实验：公平性审计与固定本图/支持查询协议，SAM/WSI-SAM公共及同域适配、闭集参考和受控FewClick；随后完成统计重复、细胞/机制消融、严格概念与外部泛化、TLS、整片与效率，以及逐审稿challenge证据闭环。不要求FewClick获胜，不将历史J5比较称等总监督架构实验。

## Resume here

1. Read [registry](docs/运行记录/模型实验/公平对比进度_1025_124208.json) and [current record](docs/运行记录/模型实验/运行记录_公平对比协议固定与首批基线_1025_124208.md).
2. Inspect actual status/logs and ps for active PIDs. Completed jobs moved to registry.completed_jobs, never restart them.
3. G4 fixed transfer comparison is independently accepted, qualified. Next: G6 complete mechanism acceptance/contrasts, G5 three-seed aggregation when last workers end, then strict heldout ontology and WSI pipeline. Reviewer mapping closure remains incomplete.
4. New source may live outside locked snapshots. Do not edit running/queued trainer, model, sampler or inference files listed in contracts.

Full prior goal/history snapshot preserved: `/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/continuity_1025_160204/goal_before_G3_closeout_1025_160204.md` (SHA256 b166a50ea75b0c9c81aed0525b9c48b847b3391e525af7a61daf9afa17bd28dd). Detailed immutable results and run records remain unchanged.

## Route

| Phase | Current status | Required closure |
|---|---|---|
| G0 | complete | Historical four arms, matched incremental tasks; retained J5, mixed120 tasks |
| G1 | complete, qualified | Frozen inputs/points/selection/metrics and prior ledger; no equal-total-pretraining claim |
| G2 | complete | Local public SAM/WSI-SAM6040 each |
| G3 | complete, single seed | Five10k/40k trains and6040 scores each; independent full parity audit |
| G4 | complete, qualified | Nine18700 sparse +837 dense-reference rows audited;180 gates replayed; exploratory cluster statistics |
| G5 | running | Extra seed26 complete execution; seed27 WSI/XCell complete, SAM scores/image train live; final audit/mean/std pending |
| G6 | scoring complete, acceptance pending | Seven10k trains/6040 scores and comparison ended; independent acceptance/contrasts pending |
| G7 | external subset complete, remainder pending | Six680-row external comparison audited; strict heldout concepts/TLS pending |
| G8 | input gate only | Real WSI cells/model/coverage/stitch/cache/efficiency pending |
| G9 | preparing | 11 human challenges + 14 AI checks mapped; experiments/factual corrections and submitted-version binding remain |

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

Replication coordinator1557332 remains live. At latest observation seed20261026 four runs complete execution; seed20261027 WSI/XCell complete, SAM evaluation3583783 and imageFewClick training3190652 live. Verify actual status first. Its train/eval/model sources remain frozen. No GPU0/2. No training reruns from observation timeouts.

G6, external cell/adapted and all G4 scoring jobs ended with summary + EXIT_CODE0; moved to registry.completed_jobs. Completion of execution does not replace independent acceptance.

## G4/G7/G8 facts

G4 all nine sparse outputs18700 each plus MultiverSeg dense-source837 pairs accepted. Actual GT/points/footprint separation/count algebra and6040 source mask cache SHA checks passed; three clean selected G3 weights/40k histories bound and160 clean+20 native context saved gate masks replayed. Main3+2: SAM-FT feature prototype presentDice.353329 / absentFPA.336394; clean FewClick noCell graphDense.318947/.162209; XCell.301492/.141253; MultiverSeg predicted-context.2023/.5093. No stable quality superiority claim. Historical M3.469993/.045580 has J5 AND prior mixed local/transfer adaptation, not an equal-R-training control. Dense-source reference.1369/.2093 only837 pairs; compare others on same subset. QueryWSI80, source-query dependency components only2: shared-query bootstrap intervals conditional/exploratory, not population/seed/patient significance. Accepted provenance report: transfer_1025_131856/acceptance_1025_224800/report_user_provenance.md.

External IGNITE true256/512/1024µm centre fields10,34 pairs×5 point groups,9 classes. Original splits7train/1val/2test retained as provenance; all external-only to current CRC/TIGER adaptation, never externally fit/select weights. Public SAM/WSI-SAM680 each complete: Dice.457234/.510330, paired ROI CI includes0. PUMA205/OCELOT663 ROI cannot provide1024µm context; no fake padding/repeated-image context. Native cells and all four adapted680 scores are complete and independently audited; full six-method ROI statistics saved in complete_report_1025_221700. Primary SAM-FT.651444 / WSI-SAM-FT.688043 / FewClick noCell.527190 / XCell.531922. FewClick remains lower than adapted SAMs; XCell minus noCell.004732 CI[-.025674,.027609] includes0. Only10 centres, not broad external/heldout proof.

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

## Clean adapted G4 completed execution

Real eight-case32-row gates passed for SAM-FT feature prototype and both G3 FewClick Dense conditions (each learned graph+Dense and same-feature prototype), native/cache errors0 in both domains, query maps0, input GT API forbidden, saved direct counts passed. CPU4/GPUhidden workers1736643(SAM),1736647(no cells),1736650(XCell), output `transfer_1025_131856/clean_<method>_1025_162450`,18700 rows/method; five outputs total, now all ended EXIT0. Independent full audits and statistics now accepted in acceptance_1025_224800; no R-supervised retraining was added. No remote supervised training added; this is local-adapted transfer generalization, not optimized R training. Record `docs/运行记录/模型实验/运行记录_清洁适配权重支持查询迁移_1025_161448.md`. Freeze its source hashes.

Query coverage audit:43/470 present pairs and32/465 absent pairs contain some ignore pixels in query. All metrics remain on annotated valid pixels; absent means no target annotated there, never certified whole-image or whole-slide negative. Keep full fixed task set and qualify absence/FPA. Auditquery_coverage_audit_1025_162400.json.

## Reviewer evidence tracking

docs/运行记录/模型实验/运行记录_审稿challenge逐条证据与修改边界_1025_163705.md maps all 11 human challenge groups and 14 AI self-checks to current evidence and missing work. Raw four reviews and SHA replay are in /nfs-medical3/zyh/fewclick_v2/fair_comparison/reviewer_evidence_1025_163705. XCell internal-data provenance is accepted. Unprompted-component recall/distance, complete WSI timing, original 5/30 selection and submitted-version binding remain separate gaps. Local TeX explanations are not evidence of actual upload changes. No paper edits applied or reviewer concerns declared all resolved.

## Latest accepted milestone

External six-method comparison actual10-field GT/point/680 rows, RGB/cell hashes, original development weights/40k consumption and32 saved same-checkpoint gate mask replays passed. Evidence external_1025_143257/audit_1025_221700.json and complete_report_1025_221700/summary.json; record docs/运行记录/模型实验/运行记录_外部适配完整验收与六方法统计_1025_221700.md. G6 independent acceptance and G5 full repeats still required. G4 accepted next in acceptance_1025_224800.
