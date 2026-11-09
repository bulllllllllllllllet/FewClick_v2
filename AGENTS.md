# FewClick_v2 experiment continuity

For FewClick experiments, read `goal.md` first, then the linked experiment registry and current run record. The user's latest instructions take precedence over these records.

The active CVPR roadmap is `docs/建议/模型与训练/路线_CVPR细胞词汇分割阶段合同_1030_174832.md`; its registry is `docs/运行记录/模型实验/CVPR研究进度_1030_174832.json` and current record is `docs/运行记录/模型实验/运行记录_WSI效率点击稳健性与空间确认_1109_090821.md`. The previous AAAI goal/registry is archived historical evidence only: do not resume its pending queues. Follow the current phase and acceptance gates; update this pointer if the active registry changes.

- Keep `benchmarks/v4/` and `/home/zhaoyh/FewClick/` read-only. Develop new experiments in `benchmarks/v5/` and use the `aligner` environment.
- Preserve existing data, checkpoints, caches and reports. New artifacts go in timestamped directories under `/nfs-medical3/zyh/fewclick_v2/`.
- A phase is complete only with its recorded acceptance evidence. Distinguish planned, preparing, running, failed and complete. A successful process launch is not a completed experiment.
- Read the fairness contract before comparing methods. Keep local interaction and support-query transfer in separate tables. Record all target-data pretraining, supervision, input views, prompts, model selection and metric rules.
- Never label retained-J5 comparisons as equal-total-target-supervision experiments. Do not use query GT for inference, prompt candidate selection or thresholds.
- Preserve the user's spatial-block splitting policy; do not impose patient-disjoint splits. Audit the full input footprint and preserve patient/slide provenance for reporting.
- After each milestone, update the registry, `goal.md`, the current run record and `IMPLEMENTATION_STATUS.md`. Record commands, code/data/weight hashes, real worker PIDs, logs, results and remaining gaps so another session can continue.

For FewClick model training, retraining or collapse diagnosis, consult `/home/zhaoyh/.codex/skills/fewclick-training-quality/SKILL.md` (versioned source: `benchmarks/v5/architecture_bridge/training_quality_skill/SKILL.md`). Follow the active experiment contract for concrete budgets and functional gates; finite loss/checkpoint replay alone is not functional learning acceptance.

Latest user GPU policy (2026-11-03T13:41:20.162219+08:00): sharing an allowed GPU is authorized when free memory is sufficient; do not wait solely because another process exists. Current pixel study uses GPU3 with at least 12288 MiB free before launching each GPU stage. Keep one own GPU training job, exclude GPU0/2, and never terminate other users' processes. This overrides older exclusive-GPU waiting notes.

Latest resource override (2026-11-04T18:50:08.879038+08:00): user authorized multi-GPU execution. For the active size-aware study, allow at most two own independent-arm training workers: preserve precision on GPU3 and opportunistically run detail on GPU1/4/5 when free memory is at least12288MiB. GPU0/2 remain excluded. Do not alter the frozen training recipe or duplicate/restart an active arm. Resource contract: `docs/建议/模型与训练/方案_单模型三方案双卡并行调度_1104_184916.md`; additional scheduler `/nfs-medical3/zyh/fewclick_v2/cvpr_research/size_parallel_1104_184916`. This overrides the earlier one-own-training-worker restriction for this study.

Latest resource override (2026-11-05T11:50:54.212850+08:00): user explicitly authorizes all six physical GPUs 0/1/2/3/4/5. Previous GPU0/2 exclusions are revoked. Sharing is allowed when free memory is sufficient; choose by actual free memory and compute load, without requiring an idle GPU. Preserve running workers and frozen recipes; do not terminate other users or duplicate active arms. This expands eligible devices, not experiment scope or worker-count budgets.

Latest WSI resource override (2026-11-07): user authorizes all available GPUs for WSI. Active WSI quality study permits up to six read-only inference workers, one per physical GPU0–5 with >=12288MiB free, no new training or additional seeds. Three frozen methods × two disjoint shards, gated before fan-out; record docs/运行记录/模型实验/运行记录_WSISAM自动稀疏提示迁移桥_1109_035845.md.

Latest task scope override (2026-11-08): SAME-WSI remote transfer only. The prior cross-slide-dominated11000task study is user-terminated; never resume. New same-slide tasks require different physical patches and disjoint full4x footprints, six read-only GPUworkers authorized after gates; no new training/seeds.

Latest adaptation override (2026-11-08): User explicitly authorizes matched SAM2/WSI-SAM adaptation before inference. Up to two own training workers, frozen8000task/2000update recipe, seed20261103, public initialization, train-only functional gates. This overrides no-new-training for these two baseline arms only; no additional training seeds or SAM3, no cross-slide evaluation.

Latest diagnostic override (2026-11-09): User authorizes WSI-SAM native local queryGT-simulated point/box diagnostic only. Keep it separate from SAME-WSI remote zero-query-human-prompt primary tables; no queryGT-based remote prompt/model/threshold selection. No training authorized by this diagnostic.

Latest SANSA adaptation authorization (2026-11-09): SANSA Universal adapter-only matched8000tasks/2000updates seed20261103; functional gate before fresh formal training; same-slide evaluation only. Up to two own training workers including the ongoing WSI arm. Current root /nfs-medical3/zyh/fewclick_v2/cvpr_research/sansa_adaptation_1109_043250. Finish the already-running WSI-SAM sparse bridge and freeze all WSI-SAM variants with hashes; no additional WSI-SAM candidate/recipe/seed training.
