---
name: fewclick-training-quality
description: Diagnose or run FewClick prompt-conditioned segmentation training, including functional learning gates, collapse detection, matched retraining and actual worker monitoring. Use with the repository continuity contract.
---
# FewClick training quality

Read the repository goal, registry, current run record and active experiment contract. User requests override historical decisions. Apply to FewClick training and diagnosis; do not impose these dataset rules on unrelated projects.

Before a replacement comparison, trace the actual baseline forward and loss code. Record each change to initialization, prompts, input views, cell context, backbone freezing, BN/dropout, training objective, supervision and updates. A shared decoder test cannot establish a replacement in the full original architecture. Separate retained-J5 target supervision from public initialization.

A finite loss, nonzero gradient, successful launch or exact checkpoint reload does not prove learning. Use fixed train-only representative tasks to test spatial target fitting and prompt responsiveness before an expensive run. Preregister the subset, budget and acceptance rules. Include local and remote, present/absent and both available domains in engineering gates; a focused positive-task overfit test may additionally diagnose learning. Do not use validation GT to select prompts, thresholds, repairs or checkpoints.

For FewClick bridge training, compare against the original episode objective: balanced pixel BCE, Dice/absent handling, boundary term, support-click loss, remote support-mask loss and assignment regularization. If changing it, disclose and test the change. A nearly constant predictor can have a plausible mask-only loss. Measure probability spatial variation, positive/negative separation, prompt-swap effect, empty/full-mask rates, assignments and activation scales on fixed tasks; retain all failures. A zero-initialized output head can legitimately block upstream gradients at the first update.

Verify actual fresh model and optimizer continuation. CUDA grid_sample backward may be nondeterministic: diagnose the specific operator and test equivalent deterministic sampling, preserve the old failure, and rerun gates in a new directory. Do not silently relax exact-replay checks or alter frozen v4 source.

After a matched repair, rerun relevant arms under one preregistered recipe. Do not claim superiority over an all-empty control. Functional success is not generalization; formal results still need full mask replay, original quality/cost guards and conditional seed replication. Keep threshold and split policies fixed by the active contract.

Launch long runs in named tmux with timestamped NFS artifacts, source/data/weight identities, stage/PID/log/exit status and terminal acceptance. Poll live handles before restarting. When authorized to run a sequence, queue dependent steps so successful gates flow into training; stop on failure without unrecorded retries. Report preparing/running/failed/complete separately from quality acceptance, and sync continuity records after milestones.
