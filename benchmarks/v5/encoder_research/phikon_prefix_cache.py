"""SHA-bound frozen first23-block cache, with separate full-inventory acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .continuous_data import TASKS, sha
from .cell_catalog import load_cell_catalog


WEIGHTS = Path('/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2')
IMAGES = Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/image_pilot_1031_015107/phikon/cache')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--accept', action='store_true')
    parser.add_argument('--gate-only', action='store_true')
    args = parser.parse_args()
    gate = json.loads((args.gate / 'acceptance.json').read_text())
    assert gate['status'] == 'accepted_real_prefix_gradient_reload_gate'
    references = {r['tile_id']: r for r in gate['prefix_bridges']}
    _, catalog = load_cell_catalog(TASKS)
    images = {r['tile_id']: r for r in map(json.loads, (IMAGES / 'index.jsonl').read_text().splitlines())}
    assert images.keys() == catalog.keys() and len(images) == 1737
    image_proof = json.loads((IMAGES / 'acceptance.json').read_text())
    assert image_proof['status'] == 'accepted_all1737_spatial_features'
    assert image_proof['index_sha256'] == sha(IMAGES / 'index.jsonl')
    expected = 14 if args.gate_only else 1737
    selected_ids = sorted(references if args.gate_only else images)
    identity = dict(image_index_sha256=image_proof['index_sha256'], gate_sha256=sha(args.gate / 'acceptance.json'),
                    config_sha256=sha(WEIGHTS / 'config.json'), preprocessor_sha256=sha(WEIGHTS / 'preprocessor_config.json'))
    if args.accept:
        contract = json.loads((args.output / 'contract.json').read_text())
        assert all(contract[k] == v for k, v in identity.items())
        assert contract['weight_sha256'] == '261ae680fa699b3b951597fd57aa19c02ef735805acb104b93af69b36d928569'
        records = [json.loads(line) for line in (args.output / 'index.jsonl').read_text().splitlines()]
        assert contract['gate_only'] == args.gate_only
        assert len(records) == expected and len({r['tile_id'] for r in records}) == expected
        assert {r['tile_id'] for r in records} == set(selected_ids)
        counts, replays = {}, 0
        for i, row in enumerate(records):
            original, meta = images[row['tile_id']], catalog[row['tile_id']]
            assert row['image_sha256'] == original['image_sha256'] == sha(original['image'])
            assert row['split'] == meta['split'] and row['dataset'] == meta['dataset']
            assert row['file_sha256'] == sha(row['file'])
            array = np.load(row['file'], allow_pickle=False)
            assert array.dtype == np.float32 and array.shape == (1025, 1024) and array.flags.c_contiguous
            assert np.isfinite(array).all() and hashlib.sha256(array.tobytes()).hexdigest() == row['array_sha256']
            if row['tile_id'] in references:
                ref = references[row['tile_id']]
                assert sha(ref['prefix']) == ref['prefix_sha256']
                assert np.array_equal(array, np.load(ref['prefix'], allow_pickle=False)[0])
                replays += 1
            key = meta['dataset'] + '|' + meta['split']
            counts[key] = counts.get(key, 0) + 1
            if i % 100 == 0:
                print(json.dumps(dict(stage='accept', count=i+1)), flush=True)
        assert replays == len(references) == 14
        if not args.gate_only:
            assert counts == {'crc|train': 1200, 'crc|val': 300, 'tiger|train': 156, 'tiger|val': 81}
        (args.output / 'independent_acceptance.json').write_text(json.dumps(dict(
            status='accepted_gate14_frozen23block_prefixes' if args.gate_only else 'accepted_all1737_frozen23block_prefixes', worker_pid=os.getpid(), count=expected,
            split_counts=counts, real_gate_prefixes_exact=14, no_GT_read=True, no_fitted_statistics=True,
            index_sha256=sha(args.output / 'index.jsonl'), contract_sha256=sha(args.output / 'contract.json'),
            source_sha256=sha(__file__), quality_result=False), indent=2)+'\n')
        return

    args.output.mkdir(exist_ok=False)
    started = time.time()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Stream the large immutable weight instead of reading it all into memory.
    digest = hashlib.sha256()
    with (WEIGHTS / 'model.safetensors').open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            digest.update(block)
    assert digest.hexdigest() == '261ae680fa699b3b951597fd57aa19c02ef735805acb104b93af69b36d928569'
    import transformers
    from transformers import AutoModel, AutoImageProcessor
    assert transformers.__version__ == '4.49.0'
    model = AutoModel.from_pretrained(str(WEIGHTS), local_files_only=True).cuda().eval().requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(str(WEIGHTS), local_files_only=True)
    (args.output / 'contract.json').write_text(json.dumps(dict(
        **identity, weight_sha256=digest.hexdigest(), worker_pid=os.getpid(), source_sha256=sha(__file__),
        version=transformers.__version__, prefix_shape=[1025,1024], dtype='float32', no_GT_read=True, gate_only=args.gate_only,
        feature_definition='embeddings then encoder.layer[0:23], includes CLS; block23 and final norm excluded'), indent=2)+'\n')
    with (args.output / 'index.jsonl').open('w') as index, torch.no_grad():
        for i, tile in enumerate(selected_ids):
            item, meta = images[tile], catalog[tile]
            assert sha(item['image']) == item['image_sha256']
            assert json.loads(meta['variants_json'])['original']['image_sha256'] == item['image_sha256']
            with Image.open(item['image']) as im:
                rgb = np.asarray(im.convert('RGB')).copy()
            assert rgb.shape == (512, 512, 3)
            pixels = processor(images=[rgb], do_resize=False, do_center_crop=False, return_tensors='pt')['pixel_values'].cuda()
            value = model.embeddings(pixels)
            for layer in model.encoder.layer[:23]:
                value = layer(value)[0]
            array = np.ascontiguousarray(value[0].cpu().numpy())
            assert array.shape == (1025, 1024) and array.dtype == np.float32 and np.isfinite(array).all()
            path = args.output / (tile + '.npy')
            np.save(path, array)
            row = dict(tile_id=tile, dataset=meta['dataset'], split=meta['split'], file=str(path),
                       file_sha256=sha(path), array_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                       image=item['image'], image_sha256=item['image_sha256'])
            index.write(json.dumps(row)+'\n'); index.flush()
            if i % 50 == 0:
                print(json.dumps(dict(stage='build', count=i+1, total=expected, seconds=time.time()-started)), flush=True)
    (args.output / 'build_complete.json').write_text(json.dumps(dict(count=expected, worker_pid=os.getpid(),
        index_sha256=sha(args.output / 'index.jsonl'), seconds=time.time()-started))+'\n')


if __name__ == '__main__':
    main()
