# FewClick supplementary experiment goal

Updated: 2026-10-25 13:38 CST. Native thread goal: active.

## Objective

Complete the versioned supplementary experiment route for the current physical-multiscale FewClick system. First establish auditable comparison conditions and complete local interaction baselines (SAM/WSI-SAM public and fine-tuned, closed-set supervised reference, controlled FewClick). Then proceed through transfer, statistical replication, cell ablations, held-out concepts/domains, TLS, WSI and efficiency. Success means reproducible evidence and accurately bounded conclusions; it does not require FewClick to win.

## Resume here

1. Read [registry](docs/运行记录/模型实验/公平对比进度_1025_124208.json).
2. Read [current run record](docs/运行记录/模型实验/运行记录_公平对比协议固定与首批基线_1025_124208.md).
3. Inspect the registry's active workers and actual status/log files before launching or repeating a job.
4. Continue the first incomplete phase whose prerequisites pass. Update the records when evidence changes.

Current phase: **G3: full adapted training (running)**. G2 public scores complete: SAM3+2 Dice .4505; WSI-SAM3+2 Dice .4967, on the same302 pairs ×5 point groups. Paired source-cluster report is `public_report_1025_130820`. All five G3 real gradient/reload/fresh-resume gates and independent generic-initialization/parameter-change checks passed. Full40k actual coordinates are frozen at `realized_train_1025_125928` (SHA `fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe`). Full10k-update queues launched at adaptation_1025_131451 (coordinator825263, tmux fewclick_adaptation_1025_131451); gate accuracy is not a performance result. Check registry and actual worker state before acting.

Evidence root: `/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208`. Local validation: 302 image-target pairs × 5 point groups = 1510 tasks, four nested budgets. Training: 40000 realized actual-click plans; each full trainer must reproduce the complete file SHA in its consumed log. Read the [fairness contract](benchmarks/v5/fair_comparison/README.md) before reporting a comparison.

## Frozen decisions

- User2026-10-25: use the newly implemented multiscale model and redesigned data; old 140/30/30 and old paper scores are not requirements for the new experiment.
- Primary model candidates: M3 (`multi_region`) and M3D (`multi_dense`); M1/M1D remain internal controls.
- Existing run: `/nfs-medical3/zyh/fewclick_v2/paper_multiscale/rerun_1025_033600/` (implementation `e41c060`). Four arms each completed10k updates/40k identical tasks; current results are single-seed development evidence.
- New data: CRC train1200/val300, TIGER train156/val81; full context train/val footprints disjoint. IGNITE has no eligible full-context val; PUMA/test excluded from current development.
- Current models have historical J5 target-data supervision. They are retained-system comparisons, not clean-initialization architecture evidence.
- Keep local interaction and remote support-query transfer separate. Do not copy support point coordinates into a different query image.
- New metrics use model-independent eligibility; a GT boundary missed entirely by prediction scores BF1=0. Legacy reports remain available. Do not silently replace old metrics or claim different BF1 definitions are interchangeable.
- For strict target-supervision controls, initialize the backbone from audited generic public weights and new heads; add an image-only FewClick control. User confirmed XCell trained on the internal dataset, not current CRC/TIGER; keep the upstream CTransPath image prior separately recorded.
- Same cases may appear across spatial splits; maintain full-footprint isolation and report residual dependence. No patient-disjoint split requirement.

## Route and completion evidence

| Phase | Status | Acceptance evidence |
|---|---|---|
| G0 four-arm screen | complete | Original run complete, matched task/val SHA, checkpoint reload, report |
| G1 fairness and manifests | complete (qualified) | Machine audit, input/exposure ledger, versioned metrics, frozen tasks and selection rules |
| G2 local public baselines | complete | SAM/WSI-SAM loaders, exact image/point geometry, full local report |
| G3 local adapted controls | running; all five real gates passed | Real gradient/checkpoint/resume gates; SAM-FT/WSI-SAM-FT/supervised reference and controlled FewClick |
| G4 transfer comparisons | running initial sparse diagnostics; full gates passed | Matching source-query inputs, prototype/control systems; no query GT |
| G5 replication/statistics | pending | Source/space cluster intervals, independent training seeds and cost |
| G6 cells and mechanisms | queued; CPU gates passed | Retrained no-cell/morphology/XCell and appropriate hierarchy/capacity controls |
| G7 unseen concepts/domains and TLS | pending | Exposure audit, valid context/GT, matched baselines, strict held-out initialization |
| G8 WSI and efficiency | pending | New-model real WSI path, complete coverage/coordinates, runtime/cache decomposition |
| G9 paper evidence closure | pending | Challenge-to-evidence map, corrections and limitations; submission actions require separate user instruction |

## Fairness verdict so far

The four internal arms share incremental training tasks, validation tasks, update counts, seed and shared pretrained components. Extra views and parameter counts differ by design. They do not establish superiority over equally trained external methods, independent generalization or an isolated hierarchy benefit. Legacy BF1 excludes some missed-boundary cases and needs a separate corrected report; Dice and FPA are unaffected by that denominator issue.

## Latest audit evidence

All four arms have identical incremental training/validation SHA and budget; largest-context cross-split overlap count is 0. Legacy square-tolerance BF1 corrected to common n=72: M1 .3043, M3 .2899, M1D .3274, M3D .3583. Dice is unchanged. Checkpoint selection used Dice−.25×FPA, so BF1 correction does not alter selected checkpoints. Old reports are preserved.

G3 details: native SAM/WSI-SAM bridges match public inference to max1.2e-7/0 probability error; five methods consumed the same gate-coordinate SHA. Both FewClick controls start from generic COCO, fresh task heads, no J5. Dense semantic channels are newly trained latent features. TIGER closed reference maps IDs1–7 to0–6 (failed initial gate retained); CPU image normalization reuses original he_to_tensor exactly. Current TIGER data do not contain class4. Full adaptation remains one seed/fixed hyperparameters and different public priors/views.

## Active formal workers

Implementation `d142fcb`; coordinator PID825263. GPU1 SAM-FT PID825669; GPU3 WSI-SAM-FT PID825670, followed by queued closed reference; GPU4 image-only FewClick PID825671; GPU5 XCell FewClick PID825672. Actual finite head updates observed in all four; first1000 updates are head warmup, so backbone gradient0 is expected now. Do not restart these jobs without inspecting actual process/status/log state. After each training, pipeline scores the selected checkpoint on6040 budget tasks; G3 is incomplete until all five full runs and reports are audited.

## G4 preparation evidence

Transfer root `/nfs-medical3/zyh/fewclick_v2/fair_comparison/transfer_1025_131856`; read `docs/运行记录/模型实验/运行记录_迁移任务隔离与近期参考接口审计_1025_131856.md` and `benchmarks/v5/transfer_comparison/README.md`. Fixed935 source-query pairs ×5 point groups=4675; same/cross verified WSI ×present/absent, identical L-table support points, full4x separation. Query repeats605 pairs, unique query330; do not claim935 independent WSIs. Actual GT/sign/geometry audit passed. Separate input/scorer APIs; unavailable BOTH source/query GT tests passed on real FewClick graph/prototype and SAM feature outputs; query prompt maps zero. Streaming/scoring16-row gates passed, with independent direct prediction-count replay.

Official MultiverSeg v1 native context gate passed8 cases (predicted vs full-GT support; CRC/TIGER; present/absent), native forward byte-identical and incomplete GT rejected. V1 checkpoint SHA and three official Git commits fixed; tabulate0.9.0 installed only into isolated NFS target. Full-GT reference requires complete source annotation, prediction-context does not relabel unknown pixels as human background. These gates are not full accuracy or matched-prior evidence.

G3 all four actual workers have passed1000 warmup into nonzero backbone-gradient joint training, finite losses; no run is complete. Next G4 CPU full retained-M3 graph/same-feature diagnostic and SAM-public feature retrieval; current retained diagnostics must stay separate from clean/adapted transfer. G5 replication, G6 retrained mechanisms, G7 heldouts/TLS, G8 WSI/efficiency and G9 paper closure are still required.

G4 full sparse scoring launched at13:38:33, implementation674bfeb. CPU4 threads each, GPUhidden. Retained M3 graph+same-feature prototype workerPID940942; SAM-public feature prototype workerPID940947. Each method18700 budget rows, actual rows/status confirmed. Logs and tmux names in registry. New adapted-weight transfer and full MultiverSeg tracks still required. Current record: `docs/运行记录/模型实验/运行记录_迁移任务隔离与近期参考接口审计_1025_131856.md`. Do not confuse these scoring workers with G3 training workers.

## User provenance clarification (2026-10-25)

XCell was trained on the internal dataset, not the current CRC/TIGER datasets (explicit user statement). This supersedes earlier unresolved-source wording. Preserve old machine-audit/runtime snapshots; add the source clarification in final reporting. A missing archived file manifest is a reproducibility-documentation gap, not contrary evidence or an allegation of overlap. CTransPath public image pretraining is a distinct upstream prior. Cell ablations remain contribution tests.

## G6 mechanism controls prepared

Root `/nfs-medical3/zyh/fewclick_v2/fair_comparison/mechanisms_1025_134753`; source and rules in `benchmarks/v5/mechanism_comparison/README.md`. Seven real CPU gates passed4 updates/fresh optimizer+model resume and actual backbone/head changes: M3 none/morphology/xcell, M3 shared encoder, M1 xcell, M3 no pre-graph hierarchy (cross-scale graph retained), Dense morphology. Four-task SHA identical. Morphology/xcell have equal125.91M registered parameters and73.19M joint trainable; shared45.56M registered/27.98M trainable vs single44.77M/27.20M. Noncell initial state matches across the three cell controls. Registered unused semantic parameters are separately disclosed. CPU tests are not formal GPU performance evidence.

User-confirmed XCell internal-data provenance is recorded in the updated source audit; historical runtime snapshots remain immutable. Formal queue must wait for each G3 GPU slot to finish train+evaluation and become actually idle, then run native GPU gates and new generic-init10k training, followed by6040 score rows. No GPU0/2 use. No goal phase is complete from CPU gates alone.

G6 coordinatorPID1107196 is alive in tmux `fewclick_mechanism_formal_1025_141300`, output `mechanisms_1025_134753/mechanism_formal_1025_141300`. Contract/code1ebe4c8 locked, all four slots currentlywaiting_dependency. This is a queued coordinator, not a GPU training worker. It will require G3 train+evaluation completion and actual idle GPU, then native GPU gate→new generic-init formal training→full scoring. Source code remains immutable. When presenting final results, overlay user-confirmed internal XCell training provenance on older runtime strings that still say not-audited; preserve raw snapshots and cite the clarification rather than propagating stale uncertainty.
