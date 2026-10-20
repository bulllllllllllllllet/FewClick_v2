# Multi-organ standard inputs

This v5 preparation package keeps source-local tissue labels, physical coordinates,
official/spatial splits, image quality and cell inputs explicit. It does not train
or replace the retained J5+dense128 model.

Use the `aligner` Python, with `pyvips` imported before torch. Raw datasets are
read-only. All derived output directories must be new timestamped NFS paths.

```bash
python -m benchmarks.v5.multidomain_data.prepare --mode catalog --output CATALOG_OUTPUT
python -m benchmarks.v5.multidomain_data.prepare --mode render --catalog CATALOG_OUTPUT --output RENDER_OUTPUT --per-group 16
python -m benchmarks.v5.multidomain_data.validate_inputs --inputs RENDER_OUTPUT/inputs.parquet --output QA_OUTPUT
CUDA_VISIBLE_DEVICES=2 python -m benchmarks.v5.multidomain_data.prepare_cells --inputs RENDER_OUTPUT/inputs.parquet --output CELL_OUTPUT --smoke
CUDA_VISIBLE_DEVICES=2 python -m benchmarks.v5.multidomain_data.smoke_model --inputs CELL_OUTPUT/inputs_with_cells.parquet --output MODEL_SMOKE_OUTPUT
python -m unittest benchmarks.v5.multidomain_data.test_preparation benchmarks.v5.multidomain_data.test_cell_preparation -v
```

`--per-group` caps exported tiles per dataset/organ/split; it does not modify the
complete candidate manifest. Initial export reads train/val pixels only. PUMA is
marked `training_allowed=false` even when its reserved within-domain split is
named train. A subsequent explicit model coverage experiment must version that
policy before using it for training. `prepare_cells --limit 100` selects a
deterministic balanced benchmark; `--smoke` instead selects original/0.8 quality
of one fixed tile per dataset. The model smoke checks finite forward/loss and
reload behavior; PUMA receives tensor-only checks without reading its GT.

`sources.parquet` records original files, physical sampling, label format and
source provenance. `tiles.parquet` freezes geometric selection/splits before
label-dependent checks. `inputs.parquet` references 512-square RGB/labels and
records array hashes, actual MPP, categorical counts and invertible transforms.
Each rendered tile has a durable `details.json`; per-tile progress and failure
states survive process termination. `inputs_with_cells.parquet` binds each RGB
quality variant to its newly predicted nuclei/XCell features and weight hashes.
It may cover only a benchmark subset; never substitute it for complete preparation.

Target sampling is 0.5 microns/pixel over approximately 256 microns. Integer
source windows cause small, recorded axis-wise rounding. GT uses nearest pixel
centers, keeps source categories, and maps unannotated/padding/conflicts to 255.
Image downsampling uses Lanczos, quality restoration uses bilinear, and geometric
padding is restored to white after quality augmentation. No target task/base or
click heatmap is cached here: those must be rebuilt dynamically by the trainer.

TIGER uses its released 0.5MPP coordinate grid, globally aligned 512px tiles and
buffered 8192px blocks. Incomplete grid crops and block-border crops are reported
as excluded. Source images whose dimensions disagree with published coordinate
extents are quarantined, never silently stretched. IGNITE original ROIs
are used without duplicate context representations; absent slide identity is
explicit and must not be treated as same-WSI evidence. PUMA rasterization respects
holes/MultiPolygons, leaves uncovered space unknown, and ignores contradictory
specific tissue labels. Cell preparation uses image evidence and geometric
padding only, encodes nuclear identity as unknown in every domain, and stores
log1p area in 0.25MPP-equivalent pixels. XCell reg64 is a model prediction.

Design/run record:
`docs/运行记录/特征缓存/运行记录_多器官标准输入前处理_1020_174937.md`.


Separate detector diagnostics reproduce the legacy recipe explicitly:

```bash
python -m benchmarks.v5.multidomain_data.profile_cells --inputs FIXED_INPUTS --output NEW_PROFILE_OUTPUT
python -m benchmarks.v5.multidomain_data.compare_nuclei_speed --inputs FIXED_INPUTS --checkpoint OFFICIAL_BRIGHTFIELD_TORCHSCRIPT --output NEW_COMPARISON_OUTPUT
python -m benchmarks.v5.multidomain_data.evaluate_nuclei_points --checkpoint OFFICIAL_BRIGHTFIELD_TORCHSCRIPT --output NEW_POINT_QA_OUTPUT
python -m benchmarks.v5.multidomain_data.smoke_model --inputs NEW_COMPARISON_OUTPUT/instanseg_inputs_with_cells.parquet --output NEW_MODEL_SMOKE_OUTPUT
```

Set `CUDA_VISIBLE_DEVICES` explicitly for these CUDA probes. The adapter requires
and verifies the official brightfield_nuclei0.1.1 TorchScript SHA, runs at its
0.5MPP, and maps instances back to the approximately 0.25MPP detector grid.
`CachedCenterCropper` caches centers for one immutable instance-map object and
preserves attended crop pixels. The speed comparison enforces exact Cellpose
mask and feature equality before accepting this optimization. Its measured
compute excludes model loading, RGB I/O and cache compression/writing.
The point probe fixes one OCELOT train cell field per organ and reports
class-agnostic center detection, not nucleus-boundary or tissue segmentation
accuracy. No detector is selected automatically from those six fields.

Experiment record:
`docs/运行记录/模型实验/运行记录_快速核分割替代与细胞缓存剖析_1020_182710.md`.


New multi-organ cell caches default to **InstanSeg brightfield_nuclei0.1.1**
plus cached-center attended crops. This path loads only InstanSeg, CTransPath and
XCell weights; Cellpose is available explicitly through `--detector cellpose`.
Actual axis-wise detector MPP is passed to InstanSeg before physical resampling.
Every row records `cell_recipe_id`, bound to model hashes, feature/crop contract
and source hashes. Existing cache files are never overwritten or reused as a
new detector's output. `--all` selects the full manifest; `--shard-index I
--num-shards N` gives disjoint deterministic tile/quality shards.

```bash
python -m benchmarks.v5.multidomain_data.queue_cells --render-root FULL_RENDER --render-log FULL_RENDER_LOG --gate MODEL_SMOKE/summary.json --output NEW_FULL_CELL_OUTPUT --gpus 0 1 2 3 4 5
```

The queue waits for render `EXIT_CODE=0`, a complete manifest and a passed real
model smoke before launching workers. `--limit 100` instead exercises a balanced
mini-batch through the same six-worker and merge path. Each worker has its own
log, PID, progress, status and feature manifest. A worker failure stops this run;
it does not retry, switch detectors or change thresholds. Merge requires exact
input coverage, unchanged splits/RGB identities and one consistent recipe.
Queued source changes cause an explicit failure before GPU extraction starts.

Adoption record:
`docs/运行记录/特征缓存/运行记录_InstanSeg正式细胞缓存切换_1020_185343.md`.
