# Published reference-guided segmentation pilot

The selected model is the official **SANSA Universal** (NeurIPS2025) adapter,
with original SAM2.0 Large initialization, not our trained automatic-mask bridge.
The Universal promptable checkpoint differs from the mask-only paper benchmark.
Official source: https://github.com/ClaudiaCuttano/SANSA
Source commit: f562b5642940e08a57e3da2608b804b99af3763b.

Migration manifest (2026-11-09): new orchestration and dataset adapter code;
external author source is pinned under the timestamped NFS root. No historical
v4 files, checkpoints, masks or datasets were copied into this package.

`prepare` installs small dependencies into a run-specific directory, downloads
original official base/adapter weights, and records byte counts/SHA256. It does
not change aligner global packages. `pipeline` waits for preparation, stops on
failure, checks GPU3 free memory >=12288MiB, and launches exactly one readonly
`pilot`. Named tmux and stage/log/PID records survive the interactive session.

Inference receives only native fine images and the existing support3positive/
2negative coordinates. No support mask, query mask, query human prompt or text
class name is supplied. Each independent support/query pair becomes a two-frame
pseudo-video and clears the official model's memory. The native prompt encoder
supports labels0/1; both signs are retained. Official demo preprocessing uses
640-square normalized frames, then internal1024 encoding; final logits resize
to512 and threshold0. No custom automatic prompt bridge is added.

The fixed SHA-ordered110task pilot stratifies by class/size/presence from the
existing same-slide11000 tasks. Full4x support/query footprints remain disjoint.
Query GT is used only for fixed task strata and scoring. It does not affect
inference/candidate/checkpoint/threshold selection. Two repeated real predictions,
unchanged weight state, saved-mask metric replay, source/weight/input identities
are required for acceptance. Reference row hashes are checked and paired scores
come from precisely the same selected tasks.

This is **zero target-data updates** public-reference inference. Its upstream
adapter supervision, views and budgets differ from our2000update methods; it
must not be called an equal-total-supervision or matched-adaptation comparison.
Keep full/large-area/absent results and sample counts, and do not treat this
stratified development pilot as population performance or independent validation.
No automatic expansion to full11000, extra training, seeds, SAM3 or cross-slide.
