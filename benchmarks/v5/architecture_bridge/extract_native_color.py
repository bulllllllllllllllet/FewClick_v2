"""Native mask-attended CTransPath plus XCell context on fixed color inputs."""
import argparse
import inspect
import json
import os
from pathlib import Path
import time

import pyvips
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .prepare_native_color import CONDITIONS
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.multidomain_data.fast_nuclei import CachedCenterCropper
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_feature_models, WEIGHTS


def read_inputs(folder):
    proof = json.loads((folder / 'independent_acceptance.json').read_text())
    assert proof['status'] in ['accepted_fixed_mask_native_color_input_gate', 'accepted_fixed_mask_native_color_inputs']
    assert proof['report_sha256'] == sha(folder / 'report.json')
    report = json.loads((folder / 'report.json').read_text())
    assert report['conditions'] == CONDITIONS
    return proof, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--accept', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(2)
    proof, inputs = read_inputs(a.inputs)
    provenance = json.loads((a.inputs.parent / 'native_recipe_provenance.json').read_text())
    weights = {name: sha(WEIGHTS[name]) for name in ['ctranspath', 'xcell']}
    assert all(weights[name] == provenance['recipe']['weights'][name]['sha256'] for name in weights)
    identity = dict(input_report_sha256=proof['report_sha256'], source_sha256=sha(__file__),
                    feature_source_sha256=sha(inspect.getfile(load_feature_models)),
                    cropper_source_sha256=sha(inspect.getfile(CachedCenterCropper)), weights=weights)
    if a.accept:
        result = json.loads((a.output / 'report.json').read_text())
        assert result['identity'] == identity
        assert result['inputs_gate'] == inputs['identity']['gate']
        assert len(result['frames']) == len(inputs['records'])
        observations = 0
        for row, saved in zip(inputs['records'], result['frames']):
            assert saved['tile_id'] == row['tile_id']
            assert sha(row['geometry']) == row['geometry_sha256']
            path = a.output / saved['file']
            assert sha(path) == saved['sha256']
            with np.load(row['geometry'], allow_pickle=False) as g, np.load(path, allow_pickle=False) as f:
                assert np.array_equal(f['instance_ids'], g['instance_ids'])
                assert list(f['conditions']) == list(CONDITIONS)
                values = f['features']
                assert values.shape == (5, row['retained_cells'], 64) and np.isfinite(values).all()
                reference = g['features'][:, 10:]
                assert np.allclose(values[0], reference, rtol=1e-5, atol=1e-5)
                error = float(np.max(np.abs(values[0] - reference))) if len(reference) else 0.
                assert error == saved['identity_max_abs_error']
                observations += values.shape[0] * values.shape[1]
        (a.output / 'independent_acceptance.json').write_text(json.dumps(dict(
            status='accepted_native_color_feature_gate' if result['inputs_gate'] else 'accepted_native_color_features',
            report_sha256=sha(a.output / 'report.json'), frames=len(inputs['records']),
            observations=observations, worker_pid=os.getpid(),
            code_stability_accepted=False, tissue_quality_accepted=False), indent=2) + '\n')
        return
    a.output.mkdir(parents=True, exist_ok=False)
    ctp, xcell, loading = load_feature_models(torch.device('cuda:0'))
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    def encode(rgb, instances, ids, cropper):
        if not len(ids):
            return np.empty((0, 64), dtype=np.float32)
        tensors = []
        for label in ids:
            crop = cropper(rgb, instances, int(label))
            value = torch.from_numpy(crop.copy()).permute(2, 0, 1).float()[None] / 255
            value = F.interpolate(value, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)[0]
            tensors.append((value - mean) / std)
        raw = torch.cat([ctp(torch.stack(tensors[i:i+32]).cuda()).float() for i in range(0, len(ids), 32)])
        padded = raw.new_zeros((1, 255, 768)); valid = raw.new_zeros((1, 255))
        padded[0, :len(ids)] = raw; valid[0, :len(ids)] = 1
        return xcell(None, padded, valid)[1][0, :len(ids)].float().cpu().numpy()
    records = []
    with torch.inference_mode():
        for row in inputs['records']:
            assert sha(row['geometry']) == row['geometry_sha256']
            with np.load(row['geometry'], allow_pickle=False) as f:
                instances, ids, reference = f['instances'], f['instance_ids'], f['features'][:, 10:]
            cropper = CachedCenterCropper(); values = []; timing = []
            assert [v['condition'] for v in row['variants']] == list(CONDITIONS)
            for variant in row['variants']:
                assert sha(variant['path']) == variant['file_sha256']
                with Image.open(variant['path']) as im:
                    rgb = np.asarray(im.convert('RGB')).copy()
                import hashlib
                assert hashlib.sha256(rgb.tobytes()).hexdigest() == variant['pixel_sha256']
                torch.cuda.synchronize(); start = time.perf_counter()
                value = encode(rgb, instances, ids, cropper)
                torch.cuda.synchronize(); elapsed = time.perf_counter() - start
                assert value.shape == (len(ids), 64) and np.isfinite(value).all()
                if variant['condition'] == 'identity':
                    error = float(np.max(np.abs(value-reference))) if len(ids) else 0.
                    assert np.allclose(value, reference, rtol=1e-5, atol=1e-5), (row['tile_id'], error)
                values.append(value); timing.append(dict(condition=variant['condition'], seconds=elapsed))
            path = a.output / (row['tile_id'] + '.npz')
            np.savez_compressed(path, features=np.stack(values), instance_ids=ids, conditions=np.array(list(CONDITIONS)))
            records.append(dict(tile_id=row['tile_id'], file=path.name, sha256=sha(path),
                                identity_max_abs_error=error, timing=timing))
            print(json.dumps(dict(frames=len(records), total=len(inputs['records']), retained=len(ids),
                                  identity_max_abs_error=error)), flush=True)
    result = dict(status='native_color_features_unaccepted', worker_pid=os.getpid(), identity=identity,
                  frames=records, loading=loading, inputs_gate=inputs['identity']['gate'],
                  detector_rerun=False, tissue_GT_read=False,
                  scope='Frozen fine-view native XCell with fixed instances and context; timing excludes detection, RGB IO and loading',
                  code_stability_accepted=False, WSI_speed_claim=False)
    (a.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
