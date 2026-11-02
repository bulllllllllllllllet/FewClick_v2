"""Read-only P2 evidence, remote strata and retained-cell geometry audit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

BASE = Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def mean(rows, key):
    values = [r['metrics'][key] for r in rows if r['metrics'][key] is not None]
    return float(np.mean(values)) if values else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    closeout = json.loads((BASE / 'p2_closeout_1102_174122/acceptance.json').read_text())
    for path, digest in closeout['evidence'].items():
        assert sha(path) == digest, path
    comparison_path = BASE / 'p2_post_training_audit_1102_124900/comparison/comparison.json'
    comparison = json.loads(comparison_path.read_text())
    arms = {}
    for arm in ['continuous', 'fixed', 'ema']:
        root = BASE / 'p2_vocabulary_pilot_1102_122100' / arm
        for filename, digest in comparison['evidence'][arm].items():
            assert sha(root / filename) == digest, filename
        raw = json.loads((root / 'rows.json').read_text())
        remote = [r for r in raw if r['scope'] == 'remote' and r['budget'] == '3+2']
        present = [r for r in remote if r['metrics']['positive'] > 0]
        absent = [r for r in remote if r['metrics']['positive'] == 0]
        targets = sorted({(r['plan']['dataset'], r['plan']['target']) for r in present})
        macro = float(np.mean([mean([r for r in present if (r['plan']['dataset'], r['plan']['target']) == t], 'dice') for t in targets]))
        assert abs(macro - comparison['values'][arm]['remote_macro_Dice']) < 1e-12
        fpa = mean(absent, 'false_positive_fraction')
        assert abs(fpa - comparison['values'][arm]['absent_FPA']) < 1e-12
        strata = {}
        for name, subset in {
            'remote_small': [r for r in present if r['metrics']['positive'] / r['metrics']['valid'] < .05],
            'remote_fat': [r for r in present if r['plan']['dataset'] == 'crc' and r['plan']['target'] == 10],
        }.items():
            strata[name] = dict(n=len(subset), dice=mean(subset, 'dice'))
        domain_fpa = {d: dict(n=len([r for r in absent if r['plan']['dataset'] == d]),
                             fpa=mean([r for r in absent if r['plan']['dataset'] == d], 'false_positive_fraction')) for d in ['crc', 'tiger']}
        arms[arm] = dict(remote_macro_Dice=macro, absent_FPA=fpa, diagnostic_strata=strata, domain_FPA=domain_fpa)
    # These 16 fields were selected before P2 outcomes for the native color audit.
    geometry = []
    for path in sorted((BASE / 'p2_native_color_inputs_1102_125400/full').glob('*_geometry.npz')):
        with np.load(path, allow_pickle=False) as f:
            inst, retained, features = f['instances'], f['instance_ids'], f['features']
        ids = np.unique(inst); ids = ids[ids > 0]
        yx = np.asarray(ndimage.center_of_mass(np.ones(inst.shape, np.uint8), inst, ids)).reshape(-1, 2)
        selected = np.isin(ids, retained)
        assert set(retained) == set(ids[:255])
        normalized = (yx + .5) / np.array(inst.shape)
        quad = (normalized[:, 0] >= .5).astype(int) * 2 + (normalized[:, 1] >= .5).astype(int)
        all_hist = np.bincount(quad, minlength=4) / len(ids)
        selected_hist = np.bincount(quad[selected], minlength=4) / len(retained)
        # Stored xy uses integer-index centroids divided by source raster size.
        selected_yx = yx[selected]
        selected_yx = selected_yx[np.lexsort((selected_yx[:, 1], selected_yx[:, 0]))]
        assert np.array_equal((selected_yx[:, ::-1] / np.array(inst.shape[::-1])).astype(np.float32), features[:, :2])
        geometry.append(dict(tile=path.stem.removesuffix('_geometry'), source_sha256=sha(path),
                             detected=len(ids), retained=len(retained), source_shape=list(inst.shape),
                             cap_active=len(ids) > 255, selected_first_ids=True,
                             quadrant_fraction_all=all_hist.tolist(), quadrant_fraction_retained=selected_hist.tolist(),
                             quadrant_total_variation=float(np.abs(all_hist - selected_hist).sum() / 2),
                             normalized_centroid_shift_yx=(normalized[selected].mean(0) - normalized.mean(0)).tolist(),
                             missing_center_offset_in_512_xy=(.5 / np.array(inst.shape[::-1]) * 512).tolist()))
    assert len(geometry) == 16
    result = dict(status='complete_bounded_P2_followup_audit', source_sha256=sha(__file__),
                  closeout_evidence_hashes_verified=len(closeout['evidence']),
                  comparison_sha256=sha(comparison_path), arms=arms, geometry=geometry,
                  limitations=['remote strata are post-hoc diagnostics, not new P2 decision guards',
                               '16 fixed fine fields only; quadrant discrepancy is not soft-region density error or causal segmentation loss',
                               'hash/aggregate replay does not re-infer final checkpoints or prove convergence',
                               'original P2 guard decision and artifacts remain unchanged'])
    (a.output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(arms=arms, geometry=[r for r in geometry if r['cap_active']]), indent=2))


if __name__ == '__main__':
    main()
