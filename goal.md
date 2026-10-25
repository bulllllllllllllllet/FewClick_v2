# FewClick supplementary experiment goal

Updated: 2026-10-25 12:55 CST. Native thread goal: active.

## Objective

Complete the versioned supplementary experiment route for the current physical-multiscale FewClick system. First establish auditable comparison conditions and complete local interaction baselines (SAM/WSI-SAM public and fine-tuned, closed-set supervised reference, controlled FewClick). Then proceed through transfer, statistical replication, cell ablations, held-out concepts/domains, TLS, WSI and efficiency. Success means reproducible evidence and accurately bounded conclusions; it does not require FewClick to win.

## Resume here

1. Read [registry](docs/运行记录/模型实验/公平对比进度_1025_124208.json).
2. Read [current run record](docs/运行记录/模型实验/运行记录_公平对比协议固定与首批基线_1025_124208.md).
3. Inspect the registry's active workers and actual status/log files before launching or repeating a job.
4. Continue the first incomplete phase whose prerequisites pass. Update the records when evidence changes.

Current phase: **G2: public local baselines (running)**. G1 audit, metric tests, immutable local manifests and prior/input ledger passed. Strict equal-total-supervision eligibility remains a separate G3 gate. SAM and WSI-SAM passed real CRC/TIGER inference gates (4 tasks × 4 budgets each); full scoring is running on GPU1/3 (worker PIDs729815/729820), 6040 scored budget-tasks per method. Inspect registry/logs before taking the next step. Native goal is thread-persistent; project files preserve the route for a new session.

Evidence root: `/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208`. Local validation: 302 image-target pairs × 5 point groups = 1510 tasks, four nested budgets. Training: 40000 fixed identity/seed recipes; full realized-coordinate parity must be checked by trainers. Read the [fairness contract](benchmarks/v5/fair_comparison/README.md) before reporting a comparison.

## Frozen decisions

- User2026-10-25: use the newly implemented multiscale model and redesigned data; old 140/30/30 and old paper scores are not requirements for the new experiment.
- Primary model candidates: M3 (`multi_region`) and M3D (`multi_dense`); M1/M1D remain internal controls.
- Existing run: `/nfs-medical3/zyh/fewclick_v2/paper_multiscale/rerun_1025_033600/` (implementation `e41c060`). Four arms each completed10k updates/40k identical tasks; current results are single-seed development evidence.
- New data: CRC train1200/val300, TIGER train156/val81; full context train/val footprints disjoint. IGNITE has no eligible full-context val; PUMA/test excluded from current development.
- Current models have historical J5 target-data supervision. They are retained-system comparisons, not clean-initialization architecture evidence.
- Keep local interaction and remote support-query transfer separate. Do not copy support point coordinates into a different query image.
- New metrics use model-independent eligibility; a GT boundary missed entirely by prediction scores BF1=0. Legacy reports remain available. Do not silently replace old metrics or claim different BF1 definitions are interchangeable.
- For strict target-supervision controls, initialize the backbone from audited generic public weights and new heads; add an image-only FewClick control. XCell's historical training overlap remains explicit until audited.
- Same cases may appear across spatial splits; maintain full-footprint isolation and report residual dependence. No patient-disjoint split requirement.

## Route and completion evidence

| Phase | Status | Acceptance evidence |
|---|---|---|
| G0 four-arm screen | complete | Original run complete, matched task/val SHA, checkpoint reload, report |
| G1 fairness and manifests | complete (qualified) | Machine audit, input/exposure ledger, versioned metrics, frozen tasks and selection rules |
| G2 local public baselines | running; real gates passed | SAM/WSI-SAM loaders, exact image/point geometry, full local report |
| G3 local adapted controls | pending | Real gradient/checkpoint/resume gates; SAM-FT/WSI-SAM-FT/supervised reference and controlled FewClick |
| G4 transfer comparisons | pending | Matching source-query inputs, prototype/control systems; no query GT |
| G5 replication/statistics | pending | Source/space cluster intervals, independent training seeds and cost |
| G6 cells and mechanisms | pending | Retrained no-cell/morphology/XCell and appropriate hierarchy/capacity controls |
| G7 unseen concepts/domains and TLS | pending | Exposure audit, valid context/GT, matched baselines, strict held-out initialization |
| G8 WSI and efficiency | pending | New-model real WSI path, complete coverage/coordinates, runtime/cache decomposition |
| G9 paper evidence closure | pending | Challenge-to-evidence map, corrections and limitations; submission actions require separate user instruction |

## Fairness verdict so far

The four internal arms share incremental training tasks, validation tasks, update counts, seed and shared pretrained components. Extra views and parameter counts differ by design. They do not establish superiority over equally trained external methods, independent generalization or an isolated hierarchy benefit. Legacy BF1 excludes some missed-boundary cases and needs a separate corrected report; Dice and FPA are unaffected by that denominator issue.

## Latest audit evidence

All four arms have identical incremental training/validation SHA and budget; largest-context cross-split overlap count is 0. Legacy square-tolerance BF1 corrected to common n=72: M1 .3043, M3 .2899, M1D .3274, M3D .3583. Dice is unchanged. Checkpoint selection used Dice−.25×FPA, so BF1 correction does not alter selected checkpoints. Old reports are preserved.
