"""Cache frozen J5 native stem/layer1/layer2 features for the exact pilot manifest."""
import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
import pandas as pd
import torch
from . import module_source
from .boundary_data import load_baseline


EXPECTED_SHAPES = {'stem': (1, 64, 256, 256), 'native_shallow': (1, 256, 128, 128),
                   'middle': (1, 512, 64, 64)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if Path('/nfs-medical3/zyh/fewclick_v2') not in args.output.resolve().parents:
        raise ValueError('artifact root')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'features').mkdir()
    started = time.time()
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    source_path = Path('/nfs-medical3/zyh/fewclick_v2/boundary_refinement/next_stage_1014_200033/finetune_1014_200033/run.json')
    source = json.loads(source_path.read_text())
    refs = copy.deepcopy(source['refs'])
    if args.smoke:
        refs = {s: rows[:4 if s == 'train' else 2] for s, rows in refs.items()}
    train_ids = [r['identity']['patch_id'] for r in refs['train']]
    val_ids = [r['identity']['patch_id'] for r in refs['val']]
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids) or set(train_ids) & set(val_ids):
        raise RuntimeError('unique patch/split identity')
    manifest = pd.read_parquet(Path(source['args']['spatial_data'])/'spatial_manifest.parquet').set_index('patch_id')
    for split, rows in refs.items():
        if any(str(manifest.loc[r['identity']['patch_id'], 'split']) != split for r in rows):
            raise RuntimeError('spatial split identity')
    checkpoint = Path(source['checkpoint'])
    metadata = json.loads((checkpoint.parent/'run_metadata.json').read_text())
    full = load_baseline(metadata, checkpoint, device)
    encoder = full.phase2
    del full
    encoder.requires_grad_(False).eval()
    captured = {}
    def capture(name):
        def hook(module, inputs, output):
            captured[name] = output
        return hook
    hooks = [encoder.backbone.relu.register_forward_hook(capture('stem')),
             encoder.backbone.layer1.register_forward_hook(capture('native_shallow')),
             encoder.backbone.layer2.register_forward_hook(capture('middle'))]
    total_bytes = 0
    max_errors = {'deep': 0., 'shallow': 0.}
    (args.output/'status.json').write_text(json.dumps({'status': 'preparing', 'worker_pid': os.getpid()}))
    for split, rows in refs.items():
        for i, ref in enumerate(rows):
            sample = torch.load(ref['file'], map_location=device, weights_only=False)
            if sample['image'].shape != (1, 3, 512, 512) or not torch.isfinite(sample['image']).all():
                raise ValueError('RGB cache contract')
            with torch.no_grad():
                captured.clear()
                features = encoder.extract_backbone_features(sample['image'].float())
                deep = encoder.embedding(features).half().float()
                shallow = captured['native_shallow'].reshape(1, 32, 8, 128, 128).mean(2).half().float()
                for name, value in [('deep', deep), ('shallow', shallow)]:
                    reference = sample[name].float()
                    max_errors[name] = max(max_errors[name], float((value-reference).abs().max()))
                    if not torch.allclose(value, reference, rtol=2e-3, atol=2e-3):
                        raise RuntimeError(f'encoder/cache mismatch: {name}')
                for name, shape in EXPECTED_SHAPES.items():
                    if tuple(captured[name].shape) != shape or not torch.isfinite(captured[name]).all():
                        raise ValueError(f'feature contract: {name} {captured[name].shape}')
                cached = {k: v.detach().cpu() for k, v in sample.items() if k != 'image'}
                cached.update({k: v.detach().cpu().half() for k, v in captured.items()})
            if not all(torch.isfinite(v).all() for v in cached.values()):
                raise FloatingPointError('native feature cache')
            file = args.output/'features'/f'{split}_{i:05}.pth'
            torch.save(cached, file)
            total_bytes += file.stat().st_size
            ref['source_rgb_file'] = ref['file']
            ref['file'] = str(file)
            event = {'split': split, 'completed': i+1, 'total': len(rows), 'cache_bytes': total_bytes,
                     'seconds': time.time()-started, 'worker_pid': os.getpid()}
            with (args.output/'progress.jsonl').open('a') as f:
                f.write(json.dumps(event)+'\n')
            if i % 50 == 0:
                print('CACHE', json.dumps(event), flush=True)
    for hook in hooks:
        hook.remove()
    run = {'args': {**source['args'], 'smoke': args.smoke}, 'refs': refs, 'source': str(source_path),
           'encoder_checkpoint': str(checkpoint), 'encoder_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
           'cache_version': 'native_stem_layer1_layer2_fp16_v1', 'feature_shapes': EXPECTED_SHAPES,
           'cache_bytes': total_bytes, 'seconds': time.time()-started, 'max_encoder_cache_error': max_errors,
           'worker_pid': os.getpid(), 'test_used': False,
           'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             [Path(__file__), module_source('boundary_data')]},
           'scope': 'fixed spatial manifests; historical encoder exposure; no encoder training'}
    (args.output/'run.json').write_text(json.dumps(run, indent=2))
    (args.output/'status.json').write_text(json.dumps({'status': 'complete', 'worker_pid': os.getpid()}))
    print('COMPLETE', json.dumps({'bytes': total_bytes, 'seconds': run['seconds'], 'max_errors': max_errors}), flush=True)


if __name__ == '__main__':
    main()
