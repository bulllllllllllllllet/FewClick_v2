# FewClick_v2 experiment continuity

For FewClick experiments, read `goal.md` first, then the linked experiment registry and current run record. The user's latest instructions take precedence over these records.

- Keep `benchmarks/v4/` and `/home/zhaoyh/FewClick/` read-only. Develop new experiments in `benchmarks/v5/` and use the `aligner` environment.
- Preserve existing data, checkpoints, caches and reports. New artifacts go in timestamped directories under `/nfs-medical3/zyh/fewclick_v2/`.
- A phase is complete only with its recorded acceptance evidence. Distinguish planned, preparing, running, failed and complete. A successful process launch is not a completed experiment.
- Read the fairness contract before comparing methods. Keep local interaction and support-query transfer in separate tables. Record all target-data pretraining, supervision, input views, prompts, model selection and metric rules.
- Never label retained-J5 comparisons as equal-total-target-supervision experiments. Do not use query GT for inference, prompt candidate selection or thresholds.
- Preserve the user's spatial-block splitting policy; do not impose patient-disjoint splits. Audit the full input footprint and preserve patient/slide provenance for reporting.
- After each milestone, update the registry, `goal.md`, the current run record and `IMPLEMENTATION_STATUS.md`. Record commands, code/data/weight hashes, real worker PIDs, logs, results and remaining gaps so another session can continue.
