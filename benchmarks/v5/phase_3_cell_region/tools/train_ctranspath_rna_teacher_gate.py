#!/usr/bin/env python3
"""Fine-tune the final CTransPath stage with an H&E-only RNA-teacher gate.

RNA and annotations are supervision targets during training only.  Every
evaluation prediction is produced from a soft-attended H&E cell crop alone.
This is deliberately a bounded, spatial-block COAD gate; it does not replace
the frozen v4 CTransPath -> XCellFormer contract or create a FewClick cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
for root in (SCRIPT_DIR, PROJECT_ROOT, Path('/home/zhaoyh/FewClick')):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
from run_phikon_backbone_probe import annotated_cells, build_attended_crops  # noqa: E402
from run_xenium_rna_pilot import load_transcripts, normalized_counts, split_gene_views, svd_features  # noqa: E402
from run_xenium_xcell_baseline import load_ctranspath, read_polygons  # noqa: E402


class CropDataset(Dataset):
    def __init__(self, crops, labels, rna, indices, transform):
        self.crops, self.labels, self.rna = crops, labels, rna
        self.indices, self.transform = np.asarray(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        return self.transform(self.crops[index]), self.labels[index], self.rna[index]


class CTransPathTeacher(nn.Module):
    """Last-stage CTransPath tuning plus a new 768->64 cache-compatible adapter."""
    def __init__(self, backbone: nn.Module, num_classes: int, rna_dim: int):
        super().__init__()
        self.backbone = backbone
        self.adapter = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 64))
        self.classifier = nn.Linear(64, num_classes)
        self.rna_head = nn.Linear(64, rna_dim)

    def forward(self, image):
        embedding = self.adapter(self.backbone(image))
        return embedding, self.classifier(embedding), self.rna_head(embedding)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xcell-artifact', type=Path, default=Path('/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_xcell_baseline_0831_111806'))
    parser.add_argument('--annotations', type=Path, default=Path('/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_celltype_pilot_0831_121408/annotations.csv'))
    parser.add_argument('--data-root', type=Path, default=Path('/nfs-medical3/fyx/PKU_DATA/Xenium/COAD'))
    parser.add_argument('--checkpoint', type=Path, default=Path('/home/zhaoyh/FewClick/module/checkpoint/ctranspath.pth'))
    parser.add_argument('--output-root', type=Path, default=Path('/nfs-medical3/zyh/fewclick_v2/ctranspath_rna_teacher_gate'))
    parser.add_argument('--timestamp')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--head-lr', type=float, default=5e-4)
    parser.add_argument('--rna-weight', type=float, default=.25)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20260908])
    parser.add_argument('--arms', nargs='+', choices=['label_only', 'paired_rna', 'shuffled_rna'], default=['label_only', 'paired_rna', 'shuffled_rna'])
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--max-cells', type=int, default=0, help='Bounded smoke only; keeps the first N cells after contract filtering.')
    return parser.parse_args()


def prepare_data(args):
    summary = pd.read_csv(args.xcell_artifact / 'cell_summary.csv', dtype={'cell_id': str})
    annotation = pd.read_csv(args.annotations, dtype={'cell_id': str})
    cells = annotated_cells(summary, annotation, 30)
    if args.max_cells:
        cells = cells.iloc[:args.max_cells].reset_index(drop=True)
    image_vips = pyvips.Image.new_from_file(str(args.data_root / 'HE.tif'), access='sequential').crop(1000, 2000, 1500, 1500)
    image = np.ndarray(buffer=image_vips.write_to_memory(), dtype=np.uint8, shape=(1500, 1500, image_vips.bands))[:, :, :3]
    polygons = read_polygons(args.data_root / 'segmentation_mask/cell_boundaries.csv', set(cells.cell_id), 1000, 2000)
    if set(cells.cell_id) != set(polygons):
        raise RuntimeError('cell boundary contract failed')
    crops = build_attended_crops(image, cells, polygons, 1000, 2000)
    transcript = load_transcripts(args.data_root / 'transcripts/transcripts.parquet', set(cells.cell_id), 1000, 2000, 1500, 20.0)
    counts = normalized_counts(transcript, cells.cell_id.tolist(), 30)
    order = pd.Index(cells.cell_id).get_indexer(counts.index)
    if (order < 0).any() or len(order) != len(cells):
        raise RuntimeError(f'RNA/cell contract failed: RNA={len(order)} cells={len(cells)}')
    cells, crops = cells.iloc[order].reset_index(drop=True), [crops[i] for i in order]
    genes, _ = split_gene_views(counts.columns.tolist())
    rna = svd_features(counts[genes].to_numpy()).astype(np.float32)
    return cells, crops, rna


def train_fold(args, cells, crops, rna, train_index, test_index, arm, seed, device):
    torch.manual_seed(seed)
    classes = sorted(cells.annotation.unique())
    class_index = {name: index for index, name in enumerate(classes)}
    labels = np.asarray([class_index[name] for name in cells.annotation])
    rna_mean, rna_std = rna[train_index].mean(0), rna[train_index].std(0).clip(1e-6)
    rna = (rna - rna_mean) / rna_std
    if arm == 'shuffled_rna':
        rna = rna.copy(); rna[train_index] = rna[np.random.default_rng(seed).permutation(train_index)]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    train_loader = DataLoader(CropDataset(crops, labels, rna, train_index, transform), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(CropDataset(crops, labels, rna, test_index, transform), batch_size=args.batch_size, num_workers=2, pin_memory=True)
    backbone = load_ctranspath(args.checkpoint, device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    for name, parameter in backbone.named_parameters():
        if name.startswith('layers.3') or name.startswith('norm'):
            parameter.requires_grad_(True)
    model = CTransPathTeacher(backbone, len(classes), rna.shape[1]).to(device)
    tuned = [p for p in model.backbone.parameters() if p.requires_grad]
    head = list(model.adapter.parameters()) + list(model.classifier.parameters()) + list(model.rna_head.parameters())
    optimizer = torch.optim.AdamW([{'params': tuned, 'lr': args.lr}, {'params': head, 'lr': args.head_lr}], weight_decay=1e-4)
    weight = torch.bincount(torch.tensor(labels[train_index]), minlength=len(classes)).float().clamp_min(1)
    weight = (len(train_index) / (len(classes) * weight)).to(device)
    final_loss = None
    model.train()
    for _ in range(args.epochs):
        for image, label, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            _, logits, predicted_rna = model(image.to(device, non_blocking=True))
            loss = torch.nn.functional.cross_entropy(logits, label.to(device), weight=weight)
            if arm != 'label_only':
                loss = loss + args.rna_weight * torch.nn.functional.mse_loss(predicted_rna, target.to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            loss.backward(); optimizer.step(); final_loss = float(loss.detach())
    model.eval(); predicted, truth = [], []
    with torch.inference_mode():
        for image, label, _ in test_loader:
            _, logits, _ = model(image.to(device, non_blocking=True))
            predicted.extend(logits.argmax(1).cpu().tolist()); truth.extend(label.tolist())
    return np.asarray(truth), np.asarray(predicted), final_loss, len(tuned)


def main():
    args = parse_args()
    stamp = args.timestamp or datetime.now().strftime('%m%d_%H%M%S')
    output = args.output_root / f'ctranspath_rna_teacher_gate_{stamp}'
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    device = torch.device(args.device)
    cells, crops, rna = prepare_data(args)
    groups, labels = cells.region.to_numpy(), cells.annotation.to_numpy()
    if len(np.unique(groups)) < args.folds:
        raise RuntimeError('not enough spatial blocks for requested folds')
    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            for fold, (train, test) in enumerate(GroupKFold(n_splits=args.folds).split(cells, labels, groups)):
                truth, predicted, loss, tuned_count = train_fold(args, cells, crops, rna, train, test, arm, seed, device)
                rows.append({'arm': arm, 'seed': seed, 'fold': fold, 'cells_train': len(train), 'cells_test': len(test), 'macro_f1': f1_score(truth, predicted, average='macro', zero_division=0), 'balanced_accuracy': balanced_accuracy_score(truth, predicted), 'final_loss': loss, 'tuned_parameter_tensors': tuned_count})
                print(json.dumps({'event': 'fold_complete', **rows[-1]}, ensure_ascii=False), flush=True)
    result = pd.DataFrame(rows); result.to_csv(output / 'fold_metrics.csv', index=False)
    summary = result.groupby('arm')[['macro_f1', 'balanced_accuracy']].agg(['mean', 'std']).round(6).to_dict()
    metadata = {'purpose': 'CTransPath final-stage H&E-only RNA-teacher spatial-block gate', 'cells': len(cells), 'classes': sorted(np.unique(labels).tolist()), 'rna_dim': int(rna.shape[1]), 'arms': args.arms, 'seeds': args.seeds, 'folds': args.folds, 'epochs': args.epochs, 'checkpoint': str(args.checkpoint), 'acceptance': 'paired_rna must exceed label_only and shuffled_rna; this gate does not authorize v4 replacement.', 'summary': {str(key): value for key, value in summary.items()}}
    (output / 'metrics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'complete', 'output': str(output), **metadata}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
