# V5 iterative prompt decoder

This branch keeps the v4 J5 Phase 2/3/5/6 model frozen as the initial mask and trains a new correction decoder. It consumes cached region tokens, the previous region logits, every signed prompt token, and the newest click index. A graph block controls update strength; visual similarity to the newest click competes with opposite-sign support. Region updates are sign monotonic. A small pixel residual anchors clicked pixels after soft region projection. Opposite signs at the same pixel return `abstain`.

## Data contract

The existing single-round `PromptEpisodeDataset` supplies initial prompts and target mask. `episodes.select_correction_click` derives the next positive click from a false-negative component or the next negative click from a false-positive component **of the model's own previous prediction**. The full prompt history, new sign/index and prior logits are passed to each training step. GT drives simulated click selection and losses only; it is never passed to the decoder as a previous mask. This simulator is an oracle-click development protocol, not a human-click result.

The v5 spatial split in `tissue_spatial_split_0922_001235` replaces the old case split for these experiments. `prepare_spatial_routing.py` reindexes existing frozen cell features by patch ID without rewriting source features. Train/validation episodes derive from disjoint spatial groups. The frozen v4 J5 initialization has historical exposure to some new validation patches, so validation metrics from this branch are exploratory until upstream initialization is retrained on the new split.

## Entry points

- `train_iterative.py`: frozen J5 online feature extraction, two or more simulated correction rounds, train/validation summaries, strict checkpoint quality gates.
- `evaluate_iterative.py`: reload a saved v5 checkpoint and run paired validation steps. `--historical-source val` restricts to patches outside the v4 baseline's historical train split; those patches still informed the old model's selection and remain exploratory.
- `interactive.py`: one patch/target in-memory state with slide, target, cache and geometry identity checks.
- `whole_slide_interactive.py`: apply a source-tile click token to every prepared tile; only the source tile receives the local pixel anchor. The caller supplies the prepared tile cache and stitches returned tile probabilities using the existing WSI tiling contract.

Whole-slide cross-tile correction is wired at the decoder state level. Its efficacy is not established by the current within-patch training episodes; it requires a paired cross-patch training/evaluation set and human-click review before deployment.
