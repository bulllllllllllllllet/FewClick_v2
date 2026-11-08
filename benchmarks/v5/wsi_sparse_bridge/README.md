# WSI-SAM automatic sparse transfer bridge

Single-factor candidate relative to `benchmarks/v5/wsi_author_freeze`:
remote support-derived dense logits additionally produce at most three positive
and two negative query points. The point selector never accepts query labels.
The dense path remains differentiable; extrema selection uses detached logits.
The original local/support prompt path and author-frozen decoder mask remain.

Algorithm: interpolate 256px dense logits to 64px; sigmoid; select positive
extrema >=0.7 and negative extrema <=0.3 with Chebyshev radius4 suppression.
Use pixel centers, scale to native1024 coordinates, map low-view points by
`xy/2+256`. Without an eligible positive point, provide no sparse prompt.
This is a fixed confidence abstention, not an exception fallback.

The real native prompt encoder receives points via a temporary pre-forward hook
for its high/low calls; the hook is removed in `finally`. `auto_enabled=False`
replays the original dense-only model without altering its state dictionary.

Validation order: synthetic GPU selection checks and six real train tasks;
fixed32 train-only functional gate256 updates and exact optimizer continuation;
smoke4; fresh-public formal2000x4; frozen same-slide11000 evaluation. Gate
failure stops the sequence. Functional acceptance is not improved generalization.

## Migration manifest — 2026-11-09

Source revision: f743eb2; copied functional/train/evaluate/prepare_eval/aggregate/
workflow/pipeline from `benchmarks/v5/wsi_author_freeze`, changing factory module
and scheduler devices. The new model imports the frozen-control model and adds
only sparse prompt injection. Integration validation is specific to this change.
Weights, datasets, caches, generated reports and historical v4 code were not
copied. Artifacts and source hashes are stored in
`/nfs-medical3/zyh/fewclick_v2/cvpr_research/wsi_sparse_bridge_1109_035845`.

Command (aligner):
`python -m benchmarks.v5.wsi_sparse_bridge.pipeline --root <artifact-root>`.
GPU1 training and GPU1/5 inference; stage launch requires >=12288MiB free.
This candidate is an explicit automatic-prompt adaptation of WSI-SAM, not the
original author's remote-transfer method or a claim about its performance limit.
