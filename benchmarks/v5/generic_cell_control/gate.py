"""Extract pre-alignment H&E tokens while replaying existing cell identities."""
import argparse
import inspect
import json
import os
from pathlib import Path
import time
import traceback

import pyvips
import numpy as np
import pandas as pd
from PIL import Image
import torch

from benchmarks.v5.evidence_closure.audit_external import require
from benchmarks.v5.fair_comparison.audit import digest, write
from benchmarks.v5.multidomain_data.prepare_cells import prepare_variant
from benchmarks.v5.multidomain_data.fast_nuclei import InstanSegDetector, CachedCenterCropper, DEFAULT_INSTANSEG_CHECKPOINT
from benchmarks.v5.paper_multiscale.data import native_context
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_feature_models, WEIGHTS


class Capture(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__(); self.encoder = encoder; self.parts = []

    def forward(self, value):
        result = self.encoder(value)
        self.parts.append(result.detach().float().cpu().numpy().copy())
        return result


class ExistingInstances:
    """Reuse the original fine-image nuclei, never tissue class annotations."""
    def __init__(self, instances): self.instances = instances

    def eval(self, image, **kwargs):
        require(image.shape[:2] == self.instances.shape, 'fine nuclear mask/image geometry')
        return [self.instances.copy()]


def run(catalog, output):
    output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4)
    device = torch.device('cuda:0'); started = time.time()
    require(torch.cuda.is_available(), 'native GPU cell gate required')
    rows = pd.read_parquet(catalog/'tiles.parquet').to_dict('records')
    selected = [min((r for r in rows if r['dataset'] == d and r['split'] == 'train'),
                    key=lambda r: r['tile_id']) for d in ('crc', 'tiger')]
    # Prevent labels or GT-derived eligibility from being available to feature preparation.
    for row in selected:
        row['labels'] = '/unavailable/tissue_GT.png'
        for key in ('labels_array_sha256', 'counts_json', 'eligible_json', 'interior_counts_json'):
            row.pop(key, None)
    write(output/'selection.json', selected)
    encoder, reference_xcell, compatibility = load_feature_models(device)
    encoder = Capture(encoder).eval(); detector = InstanSegDetector(DEFAULT_INSTANSEG_CHECKPOINT, device)
    cropper = CachedCenterCropper()
    source_files = [Path(__file__), Path(inspect.getfile(prepare_variant)), Path(inspect.getfile(load_feature_models)),
                    Path(inspect.getfile(InstanSegDetector)), Path(inspect.getfile(native_context))]
    write(output/'run.json', dict(worker_pid=os.getpid(), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        catalog=str(catalog), catalog_sha256=digest(catalog/'tiles.parquet'), compatibility=compatibility,
        sources={str(p): digest(p) for p in source_files}, weights={k:dict(path=str(v),sha256=digest(v)) for k,v in WEIGHTS.items() if k != 'cpsam'},
        detector_sha256=digest(DEFAULT_INSTANSEG_CHECKPOINT),
        feature_scope='Frozen public CTransPath768 before XCell; reference XCell outputs used only for parity audit',
        XCell_provenance='Internal dataset, not current CRC/TIGER; upstream CTransPath separately recorded',
        tissue_GT_available=False, transform_fitted=False, gate=True))
    records = []
    with torch.inference_mode():
        for row in selected:
            variant = json.loads(row['variants_json'])['original']; geometry = json.loads(row['multiscale_json'])
            require(digest(variant['cells']) == variant['cells_sha256'] and
                    digest(row['multiscale_cache']) == row['multiscale_sha256'], 'reference cell cache SHA')
            for scale in (0, 1, 2):
                if scale == 0:
                    require(digest(variant['image']) == variant['image_sha256'], 'fine RGB identity')
                    rgb = np.asarray(Image.open(variant['image']).convert('RGB')).copy()
                    mpp = json.loads(row['input_mpp_xy_json'])
                    with np.load(variant['cells'], allow_pickle=False) as cache:
                        reference = cache['features'].copy(); total = int(cache['total'])
                        active_detector = ExistingInstances(cache['instances'].copy())
                else:
                    rgb = native_context(row, scale); mpp = geometry['native_mpp']; active_detector = detector
                    with np.load(row['multiscale_cache'], allow_pickle=False) as cache:
                        reference = cache[f'cells_{scale}'].copy(); total = int(cache[f'total_{scale}'])
                h, w = rgb.shape[:2]
                metadata = dict(input_to_image_affine=[w/512, 0, 0, 0, h/512, 0], source_image_size=[w, h])
                encoder.parts.clear()
                arrays, details = prepare_variant(rgb, mpp, metadata, active_detector, encoder,
                                                   reference_xcell, device, cropper=cropper)
                raw = np.concatenate(encoder.parts, axis=0) if encoder.parts else np.empty((0,768), np.float32)
                require(raw.shape == (len(reference), 768) and np.isfinite(raw).all(), 'raw generic H&E shape/finite')
                require(int(arrays['total']) == total and arrays['features'].shape == reference.shape, 'same nuclei count/order')
                require(np.array_equal(arrays['features'][:, :10], reference[:, :10]), 'same geometry/type identity')
                error = float(np.max(np.abs(arrays['features']-reference))) if len(reference) else 0.
                require(error <= 2e-5, 'XCell reference replay differs; do not substitute cell identities')
                path = output/f'{row["tile_id"]}_scale{scale}.npz'
                np.savez_compressed(path, raw768=raw, geometry10=reference[:, :10], total=np.asarray(total),
                                    reference74=reference, instances=arrays['instances'])
                with np.load(path, allow_pickle=False) as saved:
                    require(np.array_equal(saved['raw768'], raw) and np.array_equal(saved['geometry10'], reference[:, :10]), 'raw/geometry disk replay')
                records.append(dict(tile_id=row['tile_id'], dataset=row['dataset'], split='train', scale=scale,
                    cells=len(raw), total=total, raw_shape=list(raw.shape), original_feature_max_error=error,
                    output=str(path), output_sha256=digest(path), details=details))
                write(output/'status.json', dict(status='running', worker_pid=os.getpid(), completed=len(records), total=6))
                print(json.dumps(records[-1]), flush=True)
    require(len(records) == 6, 'two domains and three physical views')
    write(output/'summary.json', dict(status='passed_input_gate', worker_pid=os.getpid(), records=records,
        seconds=time.time()-started, tissue_GT_used=False, transform_fitted=False, controlled_training_complete=False,
        scope='Raw generic H&E extraction and cell-identity parity only; no full-cache or final model claim'))
    write(output/'status.json', dict(status='passed_input_gate', worker_pid=os.getpid(), completed=6))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    try: run(a.catalog, a.output)
    except Exception:
        if a.output.is_dir(): write(a.output/'failure.json', dict(status='failed', traceback=traceback.format_exc()))
        raise
