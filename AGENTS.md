# FewClick_v2 experiment continuity

For FewClick experiments, read `goal.md` first, then the linked experiment registry and current run record. The user's latest instructions take precedence over these records.

The active CVPR roadmap is `docs/建议/模型与训练/路线_CVPR细胞词汇分割阶段合同_1030_174832.md`; its registry is `docs/运行记录/模型实验/CVPR研究进度_1030_174832.json` and current record is `docs/运行记录/模型实验/运行记录_细胞缓存并行加速_1102_194118.md`. The previous AAAI goal/registry is archived historical evidence only: do not resume its pending queues. Follow the current phase and acceptance gates; update this pointer if the active registry changes.

- Keep `benchmarks/v4/` and `/home/zhaoyh/FewClick/` read-only. Develop new experiments in `benchmarks/v5/` and use the `aligner` environment.
- Preserve existing data, checkpoints, caches and reports. New artifacts go in timestamped directories under `/nfs-medical3/zyh/fewclick_v2/`.
- A phase is complete only with its recorded acceptance evidence. Distinguish planned, preparing, running, failed and complete. A successful process launch is not a completed experiment.
- Read the fairness contract before comparing methods. Keep local interaction and support-query transfer in separate tables. Record all target-data pretraining, supervision, input views, prompts, model selection and metric rules.
- Never label retained-J5 comparisons as equal-total-target-supervision experiments. Do not use query GT for inference, prompt candidate selection or thresholds.
- Preserve the user's spatial-block splitting policy; do not impose patient-disjoint splits. Audit the full input footprint and preserve patient/slide provenance for reporting.
- After each milestone, update the registry, `goal.md`, the current run record and `IMPLEMENTATION_STATUS.md`. Record commands, code/data/weight hashes, real worker PIDs, logs, results and remaining gaps so another session can continue.

For FewClick model training, retraining or collapse diagnosis, consult `/home/zhaoyh/.codex/skills/fewclick-training-quality/SKILL.md` (versioned source: `benchmarks/v5/architecture_bridge/training_quality_skill/SKILL.md`). Follow the active experiment contract for concrete budgets and functional gates; finite loss/checkpoint replay alone is not functional learning acceptance.
