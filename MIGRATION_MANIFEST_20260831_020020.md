# FewClick_v2 migration manifest

- Migrated at: `20260831_020020`
- Historical source: `/home/zhaoyh/FewClick`
- Historical revision: `6678d04fb189675d6e31dc3d35f311b8ba2e794d`
- Source state: the historical worktree contained user changes and untracked files; it was read only and left unchanged.

## Copied baseline source

- `benchmarks/v4/` → `benchmarks/v4/` (six-module v4 implementation, configurations, tests, baseline integrations, and whole-slide inference source)
- `module/KFBreader/`, `module/TransPath/`, `module/ModalEncoder/`
- `XCellFormer.py`, `utils.py`, `environment.yml`, `README.md`

## Explicit exclusions

- `.git/` and all Git history
- `module/checkpoint/` and all model weights
- cached features, generated reports, visual summaries, experiment outputs, archives, and logs
- `__pycache__/`, `.pyc`, and notebook checkpoint folders

The copied `benchmarks/v4/` is a frozen reproducibility baseline. New RNA/spatial-omics and FewClick_v2 work belongs under `benchmarks/v5/`.
