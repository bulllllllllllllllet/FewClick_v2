"""Extract LEMON while preserving every original retained cell and XCell vector."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .cell_inputs import load_continuous_cell_inputs
from .cell_catalog import load_cell_catalog
from .lemon_features import FrozenLemon, sha


TASKS = Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/tasks_1025_125400')
INPUT_GATE = Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/cell_downstream_input_gate_1031_021255/acceptance.json')


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate', action='store_true')
    parser.add_argument('--gate-receipt', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    catalog, catalog_rows = load_cell_catalog(TASKS)
    fixture = json.loads(INPUT_GATE.read_text())
    assert fixture['status'] == 'accepted_same_mask_order_RGB_cell_input_gate'
    fixture_rows = {r['tile_id']: r for r in fixture['rows']}
    identities = list(fixture_rows) if args.gate else sorted(catalog_rows)
    if not args.gate:
        assert len(identities) == 1737 and args.gate_receipt is not None
        prior = json.loads(args.gate_receipt.read_text())
        assert prior['status'] == 'accepted_raw16_continuous_cell_gate'
        assert prior['extraction_source_sha256'] == sha(__file__)
        assert prior['cell_input_source_sha256'] == sha(Path(__file__).with_name('cell_inputs.py'))
        assert prior['lemon_source_sha256'] == sha(Path(__file__).with_name('lemon_features.py'))
        assert prior['encoder_identity'] == FrozenLemon.identity()
        assert prior['acceptance_source_sha256'] == sha(Path(__file__).with_name('accept_continuous_cells.py'))
        assert prior['catalog_source_sha256'] == sha(Path(__file__).with_name('cell_catalog.py'))
    start = time.time()
    status = dict(status='extracting', worker_pid=os.getpid(), completed=0, total=len(identities))
    def save():
        status['elapsed_seconds'] = time.time() - start
        (args.output / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    save()
    encoder = FrozenLemon()
    reload_images, reload_reference = None, None
    entries = []
    with (args.output / 'index.jsonl').open('w') as stream:
        for identity in identities:
            row = catalog_rows[identity]
            # Only image/instance input fields are consumed by this helper, never labels.
            item = load_continuous_cell_inputs(row)
            crops = item.pop('crops')
            if identity in fixture_rows:
                reference = fixture_rows[identity]
                assert len(crops) == reference['retained']
                for crop, saved in zip(crops, reference['crop_examples']):
                    with Image.open(saved['path']) as im:
                        assert np.array_equal(np.asarray(crop), np.asarray(im.convert('RGB')))
            batches = [encoder.encode(crops[offset:offset + 64]) for offset in range(0, len(crops), 64)]
            features = np.concatenate(batches) if batches else np.empty((0, 384), np.float32)
            assert features.shape == (len(item['instance_ids']), 384)
            if args.gate and reload_images is None and crops:
                reload_images = crops[:8]
                reload_reference = encoder.encode(reload_images)
            arrays = {key: item[key] for key in ['instance_ids', 'positions', 'metadata10', 'old_reg64']}
            arrays['lemon384'] = features
            path = args.output / (identity + '.npz')
            np.savez_compressed(path, **arrays)
            variant = json.loads(row['variants_json'])['original']
            entry = dict(tile_id=identity, dataset=row['dataset'], split=row['split'], file=str(path),
                         file_sha256=sha(path), arrays={k: array_sha(v) for k, v in arrays.items()},
                         retained=len(features), total_detected=item['total'], edge_padded=item['edge_padded'],
                         actual_crop_um=item['actual_crop_um'], native_information_mpp=item['native_information_mpp'],
                         sampling_mpp=item['sampling_mpp'], source_cells_sha256=variant['cells_sha256'],
                         source_image_sha256=variant['image_sha256'])
            entries.append(entry)
            stream.write(json.dumps(entry) + '\n')
            stream.flush()
            status['completed'] = len(entries)
            save()
    reload_error = None
    if args.gate:
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        fresh = FrozenLemon()
        actual = fresh.encode(reload_images)
        reload_error = float(np.max(np.abs(actual - reload_reference)))
        assert np.allclose(actual, reload_reference, atol=1e-5, rtol=1e-5)
    report = dict(status='complete_raw_continuous_cell_cache_unaccepted', gate=args.gate,
                  worker_pid=os.getpid(), frames=len(entries), retained_cells=sum(r['retained'] for r in entries),
                  total_detected=sum(r['total_detected'] for r in entries), edge_padded=sum(r['edge_padded'] for r in entries),
                  catalog_sha256=sha(catalog / 'tiles.parquet'), input_gate_sha256=sha(INPUT_GATE),
                  index_sha256=sha(args.output / 'index.jsonl'), extraction_source_sha256=sha(__file__),
                  cell_input_source_sha256=sha(Path(__file__).with_name('cell_inputs.py')),
                  lemon_source_sha256=sha(Path(__file__).with_name('lemon_features.py')),
                  catalog_source_sha256=sha(Path(__file__).with_name('cell_catalog.py')),
                  encoder_identity=FrozenLemon.identity(), fresh_reload_max_abs_error=reload_error,
                  gate_receipt_sha256=sha(args.gate_receipt) if args.gate_receipt else None,
                  elapsed_seconds=time.time() - start, GT_read=False, detector_changed=False,
                  normalization_fitted=False, qualification='raw cell features only; no tissue quality or tokenizer result')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    status.update(status='complete_unaccepted', completed=len(entries))
    save()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
