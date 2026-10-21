# Dynamic multi-domain FewClick training

This v5 path initializes the retained J5 + dense128 checkpoints and trains on
the immutable, quality-matched InstanSeg/CTP/XCell inputs. It does not modify v4
or replace the deployed checkpoint. Input images cover approximately 256 um at
512 square pixels. Training/evaluation are patch based; support and query may
belong to different WSI, which does not make this a full-WSI evaluation.

Run with the `aligner` environment and timestamped new NFS output directories:

```bash
python -m benchmarks.v5.multidomain_training.catalog --output NEW_CATALOG
python -m unittest benchmarks.v5.multidomain_training.test_training -v
CUDA_VISIBLE_DEVICES=2 python -m benchmarks.v5.multidomain_training.smoke --catalog NEW_CATALOG --output NEW_SMOKE
CUDA_VISIBLE_DEVICES=2 python -m benchmarks.v5.multidomain_training.train --catalog NEW_CATALOG --output NEW_BENCHMARK --arm C3 --steps 200 --warmup 20 --eval-every 200 --validation-plans NEW_SMOKE/validation_plans.json --benchmark
CUDA_VISIBLE_DEVICES=2 python -m benchmarks.v5.multidomain_training.train --catalog NEW_CATALOG --output NEW_C3 --arm C3 --steps 2000 --warmup 200 --validation-plans NEW_SMOKE/validation_plans.json
```

`C0` uses CRC/original; `C1` CRC/quality augmentation; `C2` four domains/original;
`C3` four domains/quality augmentation. All arms start from the same retained
weights, accumulate four episodes per optimizer update, and use the same fixed
160 development tasks. The 200-update throughput run is a gate, not a warm-start
for one of the matched arms. Train uses 1–5 positive and negative simulated
support clicks; validation uses 3 positive and 2 negative clicks. Positive
targets retain each dataset's meaning and exclude designated background/composite
classes; the other labeled tissues remain valid negatives. PUMA and test are
excluded from catalogue label reading, training and development selection.

`catalog.py` verifies label hashes, detector recipe identity, source geometry and
known cross-split footprint overlap. It computes a conservative 2 um interior
for clicks. `episodes.py` balances domains, organs and feasible target classes.
Local/present-remote/absent-remote proportions are 30/50/20 percent. Remote is
split 70/30 between known same-slide/ROI and cross-slide/image contexts. Missing
protocol coverage raises an error; it never substitutes the source patch as a
remote query. Cross-image IGNITE pairs use distinct identified patients, are
not reported as verified cross-WSI, and do not impose patient-disjoint splits.
Same-image quality variants are always local. All remote query click maps are
zero. Full/absent query targets are legal; unknown pixels remain ignored.

Only image/cell tensors and positive/negative support points enter model forward;
query GT, target label IDs and organ identities remain in the simulator/loss.
Actual pixel clicks read region evidence through the assignment at that pixel.
The centroid reader exists for exact replay of the old inference bridge. No
task token, base prediction, trainable image feature or point map is cached.
The frozen cell inputs are bound to their quality variant and source hashes.

`model.py` keeps BN statistics fixed, trains cell fusion/prompt/graph/dense during
warmup, then additionally unfreezes region heads and backbone layer3/layer4.
`losses.py` supports mixed, empty and full targets with valid-area BCE/Dice,
an absent-area penalty and an ignore-safe physical boundary band. Source mask
and signed-point losses act on the source image; they do not label the same xy
in a remote query. BF16 autocast uses float32 losses and finite-gradient gates.

Validation reports target-present Dice, the all-foreground baseline, BF1 with
physical 2 um tolerance (plus historical 2 px), and absent-query false-positive
area. Undefined boundary scores remain null with explicit evaluable counts.
Report strata include dataset, target, organ, protocol and image quality.
Best-checkpoint selection uses development present Dice minus 0.25 times absent
false-positive fraction. This short-screen score does not authorize deployment
or imply the separate CRC/low-quality/worst-organ acceptance gates have passed.
CRC validation retains historical initialization exposure and is not blind
patient-independent testing. PUMA remains a future unseen-disease check.

Every run writes input/source hashes, fixed validation tasks, sampled training
episodes, per-update loss/gradient norms, status, per-checkpoint validation rows,
optimizer/RNG checkpoints and exact reload verification. A numerical or coverage
failure stops explicitly without retrying or replacing a protocol/model.

Design and run record:
`docs/运行记录/模型实验/运行记录_多域动态点击提示与可微重训练_1021_122836.md`.

Long runs can use an expanded validation manifest and explicit learning-rate
scheduling while the default constant schedule reproduces the short screen:

```bash
python -m benchmarks.v5.multidomain_training.prepare_longrun --catalog CATALOG --anchor-plans ORIGINAL_SMOKE/validation_plans.json --output NEW_VALIDATION
CUDA_VISIBLE_DEVICES=2 python -m benchmarks.v5.multidomain_training.train --catalog CATALOG --output NEW_LONG_RUN --arm C2 --steps 20000 --warmup 2000 --eval-every 2000 --validation-plans NEW_VALIDATION/validation_plans.json --validation-anchor-count 160 --lr-schedule warmup_cosine --lr-ramp-steps 200 --min-lr-ratio 0.1
```

Expansion preserves the original 160 episodes byte-for-byte as task objects and
adds 800 locked cases, with independent GT-simulated points on val only. Reports
keep the anchor and newly added cases separate. Heads ramp up at run start;
backbone LR starts at zero and ramps separately after unfreezing. Both decay
through the joint stage to 0.1 times their maximum rates. This is a fresh run
from the retained initial checkpoints, not an implicit resume of a short-screen
checkpoint. Coverage reports count unique support/query fields and quality
variants within each arm's configured training pool. Balanced sampling is with
replacement; an 80,000-episode run does not guarantee a complete pass through
every source field or represent 80,000 independent images.


`regression.py` evaluates new checkpoints on the retained 1024 CRC local tasks
with the original images, GT, prompts and cell inputs. A locked JSON plan binds
the legacy run hash and each preselected candidate checkpoint path/step/hash.
It reports the retained centroid reader, retained pixel reader and all planned
candidates, requires exact old-cache replay, verifies saved masks, and provides
class summaries and spatial-block paired intervals. This checks compatibility
with the original deployment input contract; it is separate from the new
0.5 MPP/InstanSeg contract and the multi-domain development score.

Historical prompts are region centroids, which can lie outside the prompted
tissue. Reusing their coordinates with the pixel reader changes their meaning;
the compatibility score must not be described as the loss in real-click accuracy.
`diagnose_regression.py` separates the two contracts. It locks GT-interior points
with the same click counts and size condition, audits signed-point accuracy,
compares retained/C0/C3 with both readers, and optionally compares old cells,
unknown-only identity and InstanSeg cells recomputed on the exact original RGB.
Every condition retains the full query/target set; nonbackground results are
explicit diagnostic strata. `--size-one` adds a same-coordinate control for the
size condition used by dynamic training. It is not chosen by the resulting score.

```bash
python -m benchmarks.v5.multidomain_training.regression --plan PLAN --output NEW_SMOKE --smoke
python -m benchmarks.v5.multidomain_training.regression --plan PLAN --output NEW_SHARD --shard 0 --shards 4
python -m benchmarks.v5.multidomain_training.regression --plan PLAN --output NEW_MERGE --merge SHARD0 SHARD1 SHARD2 SHARD3
python -m unittest benchmarks.v5.multidomain_training.test_diagnosis -v
python -m benchmarks.v5.multidomain_training.diagnose_regression --plan PLAN --output NEW_SMOKE --smoke
python -m benchmarks.v5.multidomain_training.diagnose_regression --plan PLAN --output NEW_96_SHARD --shard 0 --shards 4
python -m benchmarks.v5.multidomain_training.diagnose_regression --plan PLAN --output NEW_1024_SHARD --full --cells legacy --size-one --shard 0 --shards 4
```
