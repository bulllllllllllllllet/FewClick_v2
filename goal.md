# FewClick supplementary experiment goal

Updated: 2026-10-26T01:04:33.804651+08:00. Native thread goal: **active**. G3/G4/G5 fixed comparisons accepted; the overall objective is incomplete.

## Objective

以当前三物理视野FewClick完成版本化补充实验：公平性审计与固定本图/支持查询协议，SAM/WSI-SAM公共及同域适配、闭集参考和受控FewClick；随后完成统计重复、细胞/机制消融、严格概念与外部泛化、TLS、整片与效率，以及逐审稿challenge证据闭环。不要求FewClick获胜，不将历史J5比较称等总监督架构实验。

## Resume here

1. Read [registry](docs/运行记录/模型实验/公平对比进度_1025_124208.json) and [current record](docs/运行记录/模型实验/运行记录_公平对比协议固定与首批基线_1025_124208.md).
2. Inspect actual status/logs and ps for active PIDs. Completed jobs moved to registry.completed_jobs, never restart them.
3. G4 fixed transfer comparison is independently accepted, qualified. Next: G5 three-seed aggregation is complete; next generic HE full-cell cache/controlled training, then strict heldout ontology and WSI pipeline. Reviewer mapping closure remains incomplete.
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
| G5 | complete, qualified | All12 trains/scores/selection/weights accepted; three-seed Dice/BF1 and ddof1 std independently recounted |
| G6 | partial; formal controls queued | Four raw groups/PCA dependency active; two-arm actual CPU model gate passed; full pipeline4089939 waits PCA, formal native gates/trains/scores pending |
| G7 | external accepted; heldout supervision prepared | Threefold coverage/12 model-loss gates passed; fullfold recipes/training/accuracy and TLS pending |
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

GenericHE corrected full history extraction now live: GPU1 PID3914369 (oldgroup00,434 fields), GPU3 PID3914376 (01,433), GPU4 PID3914383 (02,433), GPU5 PID3914488 (03,433). CPU PCA PID3924161 waits same four workers plus reused complete original-gate4 fields. Root generic_cells_1025_231000; outputs raw_history00..03_1025_234700 and pca_full_train_1025_234700. Inspect actual ps/status/logs. Freeze history_extract.py, fit.py and helpers, plus all generic/fair_comparison Python files captured by formal_controls_1026_001500/contract.json, paper_multiscale/model.py and losses.py while queues wait/run. No GPU0/2.

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

External six-method comparison actual10-field GT/point/680 rows, RGB/cell hashes, original development weights/40k consumption and32 saved same-checkpoint gate mask replays passed. Evidence external_1025_143257/audit_1025_221700.json and complete_report_1025_221700/summary.json; record docs/运行记录/模型实验/运行记录_外部适配完整验收与六方法统计_1025_221700.md. G6 generic HE control and G5 full repeats still required; seven mechanism runs already accepted. G4 accepted next in acceptance_1025_224800.

## G6 accepted controls

Seven mechanism variants plus two G3 Dense conditions independently accepted:9x40k actual tasks /6040 score rows and each10x302 development selection rows, actualGT/points/count algebra and selected-weight SHA verified. Native gates and all current source hashes bound. Acceptance: mechanisms_1025_134753/acceptance_1025_225200/acceptance.json and report.md.

Primary region none.544515 / morphology.533127 / XCell.537667; Dense none.564044 / morphology.557673 / XCell.572893. Dense−region XCell.035226 CI[.020060,.052833], none.019529 CI[.006328,.034669], morphology.024545 CI[.009671,.041209]. Dense also adds fine semantic latent training, not isolated projection change. M3−M1−.002279, independent−shared−.001389, hierarchy−noHierarchy+.001613: all source-space intervals include0. Dense XCell−none.008849 includes0, XCell−morphology.015220 CI[.002724,.028735] descriptive single-seed result; does not prove mIF-objective effect.

Morphology keeps position/area/unknown identity/count/detection prior; generic HE appearance control and same-network pretraining-objective control remain missing. G6 is partial, not complete; no winning-module requirement.

## G5 three-seed accepted

All12 formal10k/40k/6040 runs accepted, actual seed permutations/click multisets and8 extra native GPU gates checked; all10x302 development selections/selected weights verified. Independent math.fsum/statistics reread all72480 rows and recounted mean/sample std. report replication_1025_153837/complete_report_1025_230100/report.md; independent_aggregate_recount_1025_231100.json. Primary Dice SAM-FT 0.702063±0.001325; WSI-SAM-FT 0.701155±0.001211; FewClick noCell 0.568464±0.004273; XCell 0.572902±0.002516. XCell−none by seed+.008849/+.001616/+.002850, mean+.004438±.003869. Three seeds are not a population significance test; Dense internal initialization remains fixed.

## GenericHE input gate

Current input gate six CRC/TIGER train×three-view fields passed on GPU3, worker3713373 ended EXIT0. Raw public CTransPath768 captured before XCell, original74 reference max error0, geometry/order/total identical, native context re-detection and saved NPZ/nucleus count replay passed. Fine original masks reused; labels unavailable. Source/artifacts generic_cells_1025_231000/gate_1025_231000 and gate_independent_1025_231300.json. No full1737 cache, train-only64 transform or controlled model training completed. Next extend actual extraction path with native gate/shards; fit no transforms on validation cells.

## SAM score interpretation audit

User asked why SAM Dice is high: explain local GT-simulated3+2 present-target development tasks,10k/40k SAM adaptation and validation checkpoint selection. Three-seed mean.702063; seed25 CRC.663117/TIGER.805801,1+1.654306 vs3+2.701859. Report uses pair macro (micro would.847757), no background-TN reward and predicted-IoU candidate selection. Counts rechecked; no inference bug or causal explanation proven by score alone. Record docs/运行记录/模型实验/运行记录_SAM高Dice的指标与任务条件核对_1025_231900.md. Do not weaken SAM or alter protocol based on ranking; next action remains genericHE full extraction and heldout/WSI.

## GenericHE full extraction history correction

Old mod3 prototype has terminal EXIT1 in groups1/2 and owned group0 stopped EXIT143; preserve them, never treat as formal cache. Failure cases were originally the first item of old context groups02/03. Fresh direct replay error0; replay after new prefix changed several nucleus areas/XCell output. Specific underlying operator not isolated; this is measured call-history dependence, not benign last-bit noise. No tolerance relaxed.

New history_extract restores original completed.jsonl process groups/order, constructor order and per-call inference context. Same-source native gates: original4-field group (12 scales) + old02 first2 (6) + old03 first2 (6), all geometry and74 reference errors0. Full four groups plus reused verified original4 cover1737 fields/5211 scales. Full raw/transform/training still incomplete. PCA1530-train-cell native gate + independent direct covariance passed, val raw opened0. Both plain PCA64 and whitened PCA64 sensitivity are fixed before model results, both must be reported; source scales motivated sensitivity, no validation model selection.

Next independent work while extraction runs: actual genericHE projected inputs/model/loss/checkpoint gate and strict-heldout/WSI preparation; no conflicting GPU allocation. Record docs/运行记录/特征缓存/运行记录_通用HE完整分片提取与训练区变换合同_1025_232200.md; source/PID/evidence snapshot generic_cells_1025_231000/continuity_1025_235100.json.

## GenericHE actual model gate and formal queue

Two projected-cache/catalog variants accepted through real LocalData, four updates each (two head warmup/two joint), all3 backbones changed, fresh model+optimizer+RNG disk restore and exact two-domain predictions, resumed joint update. Both126031668 registered parameters and identical initial heads; same gate actual coordinates. Independent4 fields×2 arms compare original/new RGB, points, GT, geometry/count tensors exactly. Actual gate CPU4031341 ended EXIT0; evidence model_gate_1026_000800/summary.json and assembly_model_independent_1026_001500.json. Gate-fitted1530-cell PCA is NEVER formal training input.

Formal coordinator4089939 tmux fewclick_generic_controls_1026_001500 is verified live waiting full train PCA3924161. Then full1737-field assembly, GPU1 plain/GPU3 whitened actual native gates, NEW10k/40k train and selected-checkpoint6040 per arm. No retry; no accuracy or fullG6 closure claimed. GPU1/3 reserved; future GPU4/5 work must wait respective raw completion and actual idle. All source files in its contract frozen; add separate heldout/WSI modules.

Flag interpretation: model target_cohort_initialization=False means noJ5/target-supervised model weight init. GenericHE adds training-image unsupervised PCA; do not claim equal total pretraining or no target-image fit. Full comparison protocol SHAs legitimately change for new cell catalogs; independently check actualsame RGB/GT/points/40k, never falsify the original SHA. Next independent work while dependencies run is strict-heldout ontology/WSI pipeline, not repeated waiting-only turns. Record docs/运行记录/模型实验/运行记录_通用HE双投影装配与真实模型门控_1025_235500.md.

## Heldout ontology and supervision preparation

Predeclared N excludesCRC3/TIGER3,5 (testCRC3/TIGER5); A excludesCRC10/TIGER7 (testCRC10); L excludesCRC8/TIGER2,6,7 (testCRC8). Auxiliary broad labels are conservative exclusions, not synonymous test targets. Official TIGER primary search-index definitions + local mapping recorded; direct page403, no fabricated full-page retrieval. No accuracy-driven fold choice.

All1737 GT SHAs audited. Remaining train candidate pairs CRC/TIGER: N3869/326, A4194/401, L4393/66 (only33 TIGER fields). Old40k seen-target tasks with excluded negative points N4791/A2983/L9315. Must regenerate newfold tasks/negatives and mask seen-validation selection. Fine-clean is only full4x clean-pool upper bound.

12 real two-domain/two-split/fold point probes and freshCOCO/noJ5 image-only forward/loss passed: ignored-truth perturbation invariant and ignored output gradient0. RGB morphology still visible; context/assignment self-supervision remain image-based. Not input-gradient or all-pretraining novelty proof. OriginalGT unchanged and mask/points independently replayed. Artifacts heldout_1026_003000/audit_1026_003000, supervision_gate_1026_004200, independent_mask_gate_1026_004500.json. No formal heldout train/accuracy. Next independent task: fullfold manifests/actual points and context annotation audit while six genericHE jobs run.

## Heldout versioned protocol and actual-point queue

Builder88191/tmux fewclick_heldout_protocols_1026_004829 is live generating threefold catalogs/seen tasks and separate originalGT local/remote terminal tests. Point coordinator98745/tmux fewclick_heldout_points_1026_005200 waits same owned builder/summary/EXIT0, then all3x64 pilots+independent audits before 3x40000 full coordinates and final audits; no GPU and no retry. Source files captured in points_1026_005200/contract.json are frozen while waiting/running.

Actual necrosis native gate107528 passed64 tasks (CRC39/TIGER25) and direct resolver replay. Independent audit passed all1737 exact mask/originalGT, unchanged RGB/cells/spatial fields, all40000 recipe membership,1300 seen point groups and64 training point signs; original heldout tests40 local/115 remote pairs retain exact frozen points. This is NOT full coordinates or heldout model accuracy. Artifacts heldout_1026_003000/native_point_gate_1026_005300, protocol_continuity_1026_005400.json; record docs/运行记录/数据审计/运行记录_三fold完整任务生成与真实点击_1026_004500.md. Next observe same processes, accept full points, then actual checkpoint/train/eval adapter gates while G6 six workers continue. Overall goal active.

## Latest heldout correction and checkpoint gate

First builder88191 ended complete_protocols/EXIT0; point coordinator98745 ended FAILED after all3 pilots/audits passed but all3 full resolutions failed no source interior. Preserve full prefixes1208/680/176, actual first errors recipes1208/684/179 independently reproduced. No training on old partial manifests.

Full geometry audit149722 EXIT0 applied unchanged2um interiors/representative picker at max5+5 to all candidate pools. N5420→5418, A5929→5926, L5684→5683; removed only2/3/1 train pairs, no validation candidate removal. All6 rejections plus123 accepted direct original-resolver replays passed. New geometry protocol builder179911 and point coordinator181064 are verified live, root protocols_geometry_1026_010300 / points_geometry_1026_010300; same3x64+audit then3x40000+audit queue. Freeze captured source contract while running.

Checkpoint CPU gate first134778 failed CUDA-only shared RNG call; corrected explicitCPU harness149733 EXIT0 passed real2 updates, all3backbones changed, fresh model/optimizer/RNG and originalGT heldout CRC/TIGER predictions exactly replayed. Reject test selection, gate-as-formal, wrongfold/trainSHA/masked-testGT. training.py/evaluate.py implement distinct seen-training/checkpoint/terminal-test bindings; full native GPU trainer/evaluator remains ungated/unrun. Sources/probability/selected-weight SHAs independently verified. Record docs/运行记录/模型实验/运行记录_概念留出训练与独立终评衔接门控_1026_005500.md. Evidence checkpoint_gate_cpu_1026_005800/gate_acceptance.json, geometry_independent_1026_010300.json. No heldout accuracy yet; G6 six jobs remain live. Overall goal active.
