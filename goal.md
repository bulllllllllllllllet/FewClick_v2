# FewClick supplementary experiment goal

Updated: 2026-10-27T14:03:07.869265+08:00. Native thread goal: **active**. Heldout local/R accepted qualified; overall objective incomplete.

## Objective

以当前三物理视野FewClick完成版本化补充实验：公平性审计与固定本图/支持查询协议，SAM/WSI-SAM公共及同域适配、闭集参考和受控FewClick；随后完成统计重复、细胞/机制消融、严格概念与外部泛化、TLS、整片与效率，以及逐审稿challenge证据闭环。不要求FewClick获胜，不将历史J5比较称等总监督架构实验。

## Resume here

1. Read [registry](docs/运行记录/模型实验/公平对比进度_1025_124208.json) and [current record](docs/运行记录/模型实验/运行记录_公平对比协议固定与首批基线_1025_124208.md).
2. Inspect actual status/logs and ps for active PIDs. Completed jobs moved to registry.completed_jobs, never restart them.
3. G3/G4/G5 and heldout local/R counts/bindings are accepted qualified. Generic HE two-projection local training/scores also accepted; next raw/PCA/projection acceptance, paired statistics, full WSI independent acceptance/cells/GPU efficiency and reviewer closure.
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
| G6 | partial; both generic local controls accepted qualified | Full raw/PCA/projection independent audit and paired statistics pending |
| G7 | external and heldout local/R accepted qualified | 9train/4800local/19360R/240gate masks accepted; paired statistics, all-label descriptive comparisons and TLS pending |
| G8 | partial; first full3050tile CPU execution complete | Independent full cache/stitch acceptance, cells/GPU efficiency and main30 final cohort audit pending |
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

## Latest WSI coverage/cache milestone

TLS30 actual headers/139934 full-context-valid tiles complete and independently audited everybox/Cartesian grid/union coverage. Prediction domain covers88.916%-93.622% rectangular slide area; excluded outer margin lacks true4x, not full tissue coverage. No annotation content/model score used to select query grid; inherited annotation-count/index metadata only passes through private provenance. This is TLS30, not main-test30.

First corner cache gate nearlyallwhite, retained as background-only with exact original source snapshot. Fixed central grid probe has1.4%-4.4% white512 fine images and completed CPU worker274860 EXIT0: four disk caches, two task budgets,8 cached/native errors0, online encode0/query maps0, probability-then-threshold stitching. Independent fresh-model replay all2399401 pixels/budget exactly matches; wrongweight/input caches rejected. CRC support task30000000 is mechanical input, NOT TLS prompts or TLS accuracy. Image-only NoCell placeholders do not perform cell detection. Per-window192-node graph, not global cross-slide graph. Source/weights/cache/probability SHA independently bound.

Root wsi_global_1026_010800; evidence plan_audit_1026_011300, cache_central_gate_1026_011900, cache_central_independent_1026_012100. Record docs/运行记录/模型实验/运行记录_整片全网格与真实缓存拼接门控_1026_010800.md. Full streaming slide cache/inference/output, native GPU timing, complete cell preparation, main30 coverage and TLS annotation/negative confirmation remain. No full-slide performance or efficiency accepted. Corrected heldout builder179911 nowcomplete; parent181064 and actual full workers229034/229035/229036 remain live, no final40000 acceptance yet. G6 six workers live. Overall goal active.

## Latest full-point acceptance and heldout formal queue

Corrected point parent181064 and all3 full workers ended EXIT0. All3x40000 exact real points, full1737 masked GT/seen points/recipe memberships/original heldout point sequences accepted in points_geometry_1026_010300/full_audit. Actual SHAs: N0aa097bc325652480dded3e67dabe450070d1dcde028556a0679e3be4b2e2992, A895cfe4a7fe48691e76b395986a8d665978552d5794b037e87f97fe564994096, Lc1506898624ac24e36d1d121f2f693a311e92421c71563422c1ec2d2d06aa09c. Withinfold allmethods same file; crossfold different excluded concepts/tasks. Do not restart old/failed point jobs.

New pipeline.py real CPU preparation + independent original-byte/4two-domain-updates/9job/GPU4,5-only audit passed. Formal coordinator349631 tmux fewclick_heldout_formal_1026_012700 is verified live waiting owned raw02 PID3914383/GPU4 and raw03 PID3914488/GPU5, then actual GPUidle. First assigned Nimage/Nsam;9conditions x4stages nativegate/gate terminal scorer/NEW full10k40k/seen-selected originalGT local test. No GPU model has started. Nativegate never initializes formal model. Captured allheldout/shared Python sources frozen whilewaiting/running; add remote/WSI modules separately without editing snapshots. GPU1/3 remain G6, never0/2. Failure preserved/no retry, independent condition continues; separate remote evaluation and full independent final training/GT/selection/statistics still pending.

Record docs/运行记录/模型实验/运行记录_三fold三方法原生门控与正式训练队列_1026_012700.md; evidence queue_gate_independent_1026_013000.json and formal_queue_live_audit_1026_013200.json, root heldout_1026_003000. Next independent progress is heldout remote adapter/queue and full WSI streaming/nativecell/GPU efficiency. Overall goal active.

## Latest heldout remote interface and queue

New independent heldout_transfer implements selected seen-training/actual-coordinate/local-test/remote-test checkpoint binding and GT-forbidden inference objects. No targetID/querypoints; graph query maps0. FewClick graphDense+samefeature prototype; SAM feature proto and custom WSI fine+aligned2x normalized feature concat proto, not official native remote SAM decoders. Fixedtemp.1/threshold.5/noRupdates. ActualN/A/L115/65/62pairs ->2300/1300/1240 rows/output; parent935 count fields are ancestry only, actual coverage recomputed. N has7 strata/no TIGER same_present.

Real NoCell small CPUcheckpoint gate396749 EXIT0:7 cached/native errors0,28rows x2 outputs all saved counts/Dice independently recounted, GT API forbidden/maps0; NOT formal accuracy. Feature helper gate400579 used G3 seen-all-label SAM/WSI real weights, CPUcache output exact and 2x coordinate-ramp alignment passed; explicitly not unseen-label result.

Remote queue realprepare9jobs and independent plan audit passed. Formal CPU coordinator433592 tmux fewclick_heldout_remote_queue_1026_014400 is verifiedlive waiting localproducer349631;2CPUworker cap/CUDAhidden. Each completed full training must get its OWN remote native gate before fullR, same source/weight/testSHA; gates-only checkpoints never fullresults. No remote formal scores started. Freeze existing heldout_transfer Python and all captured imported sources; new work outside snapshots. Evidence remote_independent_1026_014300.json, remote_queue_live_1026_014700.json; record docs/运行记录/模型实验/运行记录_留出类支持查询迁移适配与真实门控_1026_013700.md.

Next full WSI streaming/cells/GPU efficiency and matched all-label reference subset extraction; final heldout fulltrain/local/R acceptance/statistics still pending. G6 sixworkers and local349631 remain live. Overall goal active.

## Latest full-WSI streaming execution

New StripeWriter holds only fine-native-height x canvas-width float sum/uint16 count, range-msync then MADV_DONTNEED releases mapped output pages. Fresh real central tissue decode two2399401-pixel budgets equals prior independent in-memory probabilities/masks exactly, edge-overlap/hole/order checks passed, including bounded-page version. Complete stream.py4tile actual read/encode/write-cache/two online/mmap gate614733 EXIT0 and independent every RGBhash/geometry/probability/mask/weighted count/source-output SHA passed. These small gates do not prove full WSI memory/latency.

Actual first complete3050tile CPU NoCell stream worker646382 tmux fewclick_wsi_full_cpu_1026_021756 is verifiedlive CACHE PREPARING, output wsi_global_1026_010800/full_cpu_slide0_1026_021756. Alloffline caches then1+1/3+2 complete online passes; no queryencode/maps, native0.247925um output is resized512 probability, no extra spatial detail. Validbbox excludes missing4x edges as audited. Estimate69GB vs actual6.787TB decimal free checked beforestart. Modelselected G3 NoCell after all-label adaptation, fixedCRC support, NOT TLS semantic prompts, full-cell runtime or GPU efficiency. OS/NFS cache state uncontrolled. Full cache/passes/coverage/independent sample acceptance still pending.

Freeze source files in full CPU contract (stream/stripe/cachehelper/reader/model) while running; new work outside snapshots. Source/SDKCPP/real PID/job-command record full_cpu_live_audit_1026_022000.json and stream_continuity_1026_022000.json. Record docs/运行记录/模型实验/运行记录_整片逐行流式拼接与首片完整执行_1026_015000.md. SixG6 + local349631 + R433592 remainlive. Next independent main30/GT audit, fullcell preparation/nativeGPU efficiency and same-subset all-label references; overall goal active.

## Latest all-label references and native heldout training

All-label SAME test-image/click/metric references accepted qualified:18 exact oldG3/G4 tables19320 rows with independent fsum stats, plus WSISAM custom two-view remote3fold4840 new rows/fullGT/point/weight/source/count/metrics and60 gate mask replays. Total21tables24160rows, not independent cases. Report heldout_1026_003000/all_label_reference_report_1026_030900/report.md. All-label means current available targets (TIGER4 absent); training AND original302pair development checkpoint selection expose test target. Newexcluded model sees different training-task distribution and seen-only selector; future difference is descriptive, not isolated supervision-deletion causality. WSISAM beststep6000 vs fulltrajectory10k/40k. Reference wrapper first gate758499 metadata polluted coverage map, explicitly rejected/preserved/source archived; fixed777292 reran, full coordinator849996 EXIT0 and independent audit accepted. No newheldout performance yet.

Raw02 PID3914383 and03 PID3914488 bothended full433fields/1299scales EXIT0; preserve as complete_execution, all4raw/PCA jointG6 acceptance stillpending. Allowed GPU4/5 then started9condition queue. Necrosis SAM native4 updates/fresh optimizerRNG restore/publicbridge maxerror1.192e-7 binaryexact and8-row terminal gate passed; fresh fulltrain GPU5 worker796373 running. Necrosis NoCell native4 updates/freshrestore/8terminalrows passed, independent warm/joint probe shows all3encoder backbones changed; fresh fulltrain GPU4 worker969093 running, resumeNone/newNactual SHA. Parent349631 and R433592 remainlive; R awaits completed fulltraining. No gateweights initialize formal models.

Evidence native_first_live_1026_024600.json/native_second_live_1026_031500.json, all_label_independent_1026_024500.json and all_label_wsi_acceptance_1026_030600/summary.json. Record docs/运行记录/模型实验/运行记录_全标签同任务参考提取与WSI迁移补齐_1026_023000.md; snapshot reference_continuity_1026_032000.json. FullCPU WSI646382 stillcachepreparing; fullcell/main30/GT/nativeGPU efficiency and allnew9train/local/R audits/statistics/reviewer closure remain. Overall goal active.

## Latest heldout acceptance and resumed work

Updated 2026-10-26T18:32:25.074145+08:00. User requested continuation; overall goal active. All previously active experiment workers ended EXIT0 and moved to completed_jobs; do not restart completed runs. New audits3773203/3780561 ended EXIT0: nine formal10k/40k trains,90seen-only selection tables,4800local rows and19360remote rows accepted qualified;240native saved gate masks pixel-recounted. Record docs/运行记录/模型实验/运行记录_留出概念九条件训练与本图独立验收_1026_130200.md. Single seed; paired statistics and same-task all-label descriptive comparisons pending. FewClick is lower on all three local folds; no favorable-result requirement.

GenericHE two full projections and first3050tile CPU WSI have complete execution only, independent preparation/full-WSI acceptance pending. GenericHE local verifier1165842 launched in tmux fewclick_generic_accept_1026_183200; inspect actual status. MainGDPH30 source/header/exact-declared-path exploration found0 overlap with current catalog/master; clinical patient identity/content/rescan/upstream prior and full GT census not established. Continue full-cell/nativeGPU WSI/efficiency and reviewer closure. New IHC/MIHIC records remain separate; no model/data integration inferred.

GenericHE verifier1165842已结束EXIT0，两个预设投影共12080行、20份302行开发选模及实际40k消费/选中权重/GT/源码绑定通过；主3+2 PCA64 Dice.562162、白化64 Dice.568247。单seed，两者均报告；完整raw/PCA/projection独立复核、配对统计仍待，G6不关闭。

## 1027_140300 脂肪留出异常诊断完成（限定结论）

2026-10-27T14:02:19.778340+08:00：按用户要求优先排查。GPU5 worker1196763完整重放两selected模型200份3+2预测，原TP/FP/FN200/200一致；独立原RGB/GT/点/SHA和Dense代数复核通过。相同20pair×5点组，all-label Dice.611023/base.404669，adipose-heldout .084588/base.112011；300正点落背景3 vs162。实际两域selected模型/真实训练任务loss探针1227032通过，CRC155861脂肪与TIGER17287rest像素valid=0/output-logit梯度0，未见误当负类。诊断20pair group0十个checkpoint：seen开发.445192→.601296，fat .273368→.084654；不能据此重选checkpoint/阈值。当前没有查到指定标签/点/权重/算分故障，确认真实低召回/点击响应失败；单一训练因素尚未因果隔离，不认定整个架构错误。完整原预测图/概率和轨迹保留，正式结果不改。记录 docs/运行记录/模型实验/运行记录_脂肪留出异常逐像素与训练轨迹排查_1027_135100.md；输出/nfs-medical3/zyh/fewclick_v2/fair_comparison/heldout_1026_003000/adipose_diagnostic_acceptance_1027_135800。下一继续普通HE raw/PCA/投影审计、配对统计及WSI/GPU效率，整体goal active。
