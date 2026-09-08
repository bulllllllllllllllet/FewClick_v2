#!/usr/bin/env python3
"""Build Phase-2-soft-assignment RNA targets for a bounded paired COAD window.

This creates a contract cache only.  RNA is never an input feature: each target
is a soft-assignment-weighted aggregate of per-cell RNA program scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import torch
from PIL import Image
from torchvision.transforms import functional as tvf

PROJECT_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_ROOT, Path('/home/zhaoyh/FewClick')):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
from benchmarks.v4.phase_1_multiscale.src.common import load_config  # noqa: E402
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder  # noqa: E402
from benchmarks.v4.phase_3_cell_region.src.model import sample_assignment_at_cells  # noqa: E402
from run_xenium_rna_pilot import load_transcripts, normalized_counts, split_gene_views  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/nfs-medical3/fyx/PKU_DATA/Xenium/COAD'))
    parser.add_argument('--xcell-artifact', type=Path, default=Path('/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_xcell_baseline_0831_111806'))
    parser.add_argument('--phase2-config', type=Path, default=Path('/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240/config_snapshot.json'))
    parser.add_argument('--phase2-checkpoint', type=Path, default=Path('/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240/checkpoint_epoch_028.pth'))
    parser.add_argument('--output-root', type=Path, default=Path('/nfs-medical3/zyh/fewclick_v2/region_rna_target_smoke'))
    parser.add_argument('--timestamp')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--x', type=int, default=1000)
    parser.add_argument('--y', type=int, default=2000)
    parser.add_argument('--size', type=int, default=1500)
    parser.add_argument('--patch-size', type=int, default=512)
    parser.add_argument('--grid-size', type=int, default=3)
    parser.add_argument('--min-transcripts', type=int, default=30)
    return parser.parse_args()


def image_tensor(image: np.ndarray) -> torch.Tensor:
    pil = Image.fromarray(image, mode='RGB').resize((512, 512), Image.Resampling.BILINEAR)
    value = tvf.to_tensor(pil)
    return tvf.normalize(value, [.485, .456, .406], [.229, .224, .225])


def fixed_hash_programs(counts: pd.DataFrame, genes: list[str], programs: int = 8) -> tuple[np.ndarray, list[int]]:
    """Gene-set means with fixed membership; no fit uses held-out cells."""
    membership = np.asarray([int(hashlib.sha256(gene.encode()).hexdigest()[:8], 16) % programs for gene in genes])
    values = counts[genes].to_numpy(dtype=np.float32)
    output = np.zeros((len(counts), programs), np.float32)
    sizes = []
    for index in range(programs):
        chosen = membership == index
        if not chosen.any():
            raise RuntimeError(f'empty hash RNA program {index}')
        output[:, index] = values[:, chosen].mean(1)
        sizes.append(int(chosen.sum()))
    return output, sizes


def load_cells_and_rna(args: argparse.Namespace):
    summary = pd.read_csv(args.xcell_artifact / 'cell_summary.csv', dtype={'cell_id': str})
    reg = np.load(args.xcell_artifact / 'xcell_reg64.npy')
    if len(summary) != len(reg) or reg.shape[1] != 64 or not np.isfinite(reg).all():
        raise RuntimeError(f'invalid XCell cache: summary={len(summary)} reg={reg.shape}')
    tx = load_transcripts(args.data_root / 'transcripts/transcripts.parquet', set(summary.cell_id), args.x, args.y, args.size, 20.0)
    counts = normalized_counts(tx, summary.cell_id.tolist(), args.min_transcripts)
    positions = pd.Index(summary.cell_id).get_indexer(counts.index)
    if (positions < 0).any() or len(np.unique(positions)) != len(positions):
        raise RuntimeError('cell/RNA/XCell ID contract failed')
    summary, reg = summary.iloc[positions].reset_index(drop=True), reg[positions].astype(np.float32)
    _, target_genes = split_gene_views(counts.columns.tolist())
    program, program_sizes = fixed_hash_programs(counts, target_genes)
    return summary, reg, program, target_genes, program_sizes


def phase2_model(args: argparse.Namespace, device: torch.device) -> DeepRegionEncoder:
    config = load_config(args.phase2_config)
    model = DeepRegionEncoder(config, len(config['data']['class_map']), int(config['model']['num_regions']), int(config['model']['embedding_dim']))
    state = torch.load(args.phase2_checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval()
    return model


def main() -> None:
    args = parse_args(); stamp = args.timestamp or datetime.now().strftime('%m%d_%H%M%S')
    output = args.output_root / f'region_rna_target_smoke_{stamp}'
    if output.exists(): raise FileExistsError(output)
    device = torch.device(args.device)
    cells, reg, program, target_genes, program_sizes = load_cells_and_rna(args)
    vips = pyvips.Image.new_from_file(str(args.data_root / 'HE.tif'), access='sequential').crop(args.x, args.y, args.size, args.size)
    image = np.ndarray(buffer=vips.write_to_memory(), dtype=np.uint8, shape=(args.size, args.size, vips.bands))[:, :, :3]
    model = phase2_model(args, device)
    starts = np.linspace(0, args.size - args.patch_size, args.grid_size).round().astype(int)
    examples = []
    with torch.inference_mode():
        for top in starts:
            for left in starts:
                selected = (cells.center_x >= args.x + left) & (cells.center_x < args.x + left + args.patch_size) & (cells.center_y >= args.y + top) & (cells.center_y < args.y + top + args.patch_size)
                selected = np.flatnonzero(selected.to_numpy())
                if not len(selected): raise RuntimeError(f'no RNA cells in local patch {left},{top}')
                raw = image[top:top + args.patch_size, left:left + args.patch_size]
                output_p2 = model(image_tensor(raw).unsqueeze(0).to(device), return_full_assignment=False)
                local = cells.iloc[selected]
                xy = np.stack(((local.center_x.to_numpy() - args.x - left) / args.patch_size, (local.center_y.to_numpy() - args.y - top) / args.patch_size), axis=1).astype(np.float32)
                if not ((xy >= 0).all() and (xy < 1).all()): raise RuntimeError('local cell coordinates outside patch')
                mass = sample_assignment_at_cells(output_p2['assignment_low'], torch.from_numpy(xy).unsqueeze(0).to(device))[0].float().cpu().numpy()
                if not np.allclose(mass.sum(0), 1.0, atol=1e-4) or not np.isfinite(mass).all(): raise RuntimeError('invalid Phase-2 assignment at cells')
                score = program[selected]
                coverage = mass.sum(1)
                target = (mass @ score) / coverage[:, None].clip(1e-8)
                area = np.log1p((local.bbox_width.to_numpy() * local.bbox_height.to_numpy()).clip(0)).astype(np.float32)
                # COAD pilot lacks a H&E nucleus-class field; do not substitute an RNA-derived label.
                feature = np.concatenate((xy, area[:, None], np.zeros((len(xy), 7), np.float32), reg[selected]), axis=1)
                examples.append({'left': int(left), 'top': int(top), 'cell_ids': local.cell_id.to_numpy(), 'cells': feature, 'assignment': mass, 'region_tokens': output_p2['region_tokens'][0].float().cpu().numpy(), 'rna_target': target.astype(np.float32), 'rna_coverage': coverage.astype(np.float32)})
    maximum = max(len(x['cells']) for x in examples); count, regions, dim, target_dim = len(examples), examples[0]['assignment'].shape[0], examples[0]['cells'].shape[1], program.shape[1]
    cell_values = np.zeros((count, maximum, dim), np.float32); valid = np.zeros((count, maximum), bool); assignment = np.zeros((count, regions, maximum), np.float32)
    for index, item in enumerate(examples):
        n = len(item['cells']); cell_values[index, :n] = item['cells']; valid[index, :n] = True; assignment[index, :, :n] = item['assignment']
    output.mkdir(parents=True)
    np.savez_compressed(output / 'region_rna_contract.npz', region_tokens=np.stack([x['region_tokens'] for x in examples]), cells=cell_values, cell_valid=valid, assignment_at_cells=assignment, rna_targets=np.stack([x['rna_target'] for x in examples]), rna_coverage=np.stack([x['rna_coverage'] for x in examples]))
    manifest = pd.DataFrame([{'patch_index': i, 'left_px': x['left'], 'top_px': x['top'], 'rna_cells': len(x['cells']), 'cell_id_sha256': __import__('hashlib').sha256('|'.join(x['cell_ids']).encode()).hexdigest()} for i, x in enumerate(examples)])
    manifest.to_csv(output / 'patch_manifest.csv', index=False)
    metadata = {'purpose': 'Phase-2 soft-assignment RNA target contract smoke; RNA is target only', 'patches': count, 'patch_size_px': args.patch_size, 'cell_feature_shape': list(cell_values.shape), 'assignment_shape': list(assignment.shape), 'region_token_shape': list(np.stack([x['region_tokens'] for x in examples]).shape), 'rna_target_shape': [count, regions, target_dim], 'target_gene_count': len(target_genes), 'target_definition': 'eight deterministic hash gene-bucket means; fixed gene membership, no SVD fit across held-out cells', 'target_gene_count_by_program': program_sizes, 'phase2_checkpoint': str(args.phase2_checkpoint), 'xcell_cache': str(args.xcell_artifact), 'nuclei_class': 'unavailable in COAD pilot; seven one-hot slots set to zero, no RNA-derived replacement', 'acceptance': 'all real Phase-2 assignments, RNA targets, tokens and cell features are finite; assignment mass sums to one per valid cell'}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'complete', 'output': str(output), **metadata}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
