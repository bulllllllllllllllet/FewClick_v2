# Matched SANSA Universal target adaptation

Official SANSA Universal (NeurIPS2025), pinned public base and adapter, native
support-to-query memory path. Only240original adapter tensors train; no new
matcher, object-presence head or query prompt. Frozen modules run in eval mode
while adapter gradients remain enabled. The input is fine-only512 -> PIL640 ->
GPU normalization -> official1024 backbone, identical to the public pilot.

Same fixed seed20261103, actual8000episodes/2000updates x4, AdamW1e-4/wd1e-4,
original mask/click/support objective and fixed0.5 threshold. Train-only32task
functional gate256 updates includes six local/remote-present/remote-absent domain
groups and exact fresh optimizer continuation. Success queues smoke4 -> fresh
public formal2000 -> six same-slide forward checks ->11000same-slide evaluation.
Gate failure stops. Adapter updates inside the encoder are allowed; all other
encoder/decoder weights must remain unchanged. Upstream public supervision,
capacity, views and trainable parameter counts differ from other baselines.

`encode` caches only preprocessed frames, not learned image embeddings. Do not
interpret evaluation runtime as cached WSI interaction latency. Public110pilot
is retained; the full11000trained result is separate.

Migration manifest2026-11-09: new native SANSA wrapper and integration; copy
functional/train/prepare_eval/evaluate/aggregate/workflow/pipeline from
wsi_sparse_bridge at d8e5660, adapt factory and nonadapter freeze audit. External
source stays at its pinned timestamped NFS directory. No v4 changes or weights,
data, caches and generated reports copied into the repository.

Active artifacts: /nfs-medical3/zyh/fewclick_v2/cvpr_research/sansa_adaptation_1109_043250.
Previous042824 preserved after controller-session interruption of functional
training; same recipe restarted in a new directory, no active arm duplicated.
One GPU3 training worker, later GPU3/5 readonly inference after >=12GiB checks.
WSI-SAM's existing automatic-point experiment continues independently and freezes
at accepted completion, with no further bridge/seed/recipe exploration.
