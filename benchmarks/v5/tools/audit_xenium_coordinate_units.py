"""Compare historical pixel assumption with a physical-coordinate hypothesis.

Exports diagnostic coordinates only; neither hypothesis grants registration or
permission to use transcriptomic labels as aligned H&E cell supervision.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyvips


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(a.manifest)
    summary = []
    diagnostics = []
    for cohort, group in frame[frame.platform == 'Xenium'].groupby('cohort_id'):
        im = pyvips.Image.new_from_file(group.he_path.iloc[0], access='random')
        with h5py.File(group.h5ad_path.iloc[0], 'r') as h:
            mpp = np.asarray(h['uns']['H&E resolution'][()], dtype=float)
        assert mpp.shape == (2,) and np.isfinite(mpp).all() and (mpp > 0).all()
        raw = group[['x','y']].to_numpy(float)
        projected = raw / mpp
        dims = np.array([im.width, im.height])
        assert np.isfinite(raw).all()
        summary.append(dict(cohort=cohort, cells=len(group), image_wh=dims.tolist(),
            resolution_metadata_xy=mpp.tolist(), candidate_scale_xy=(1/mpp).tolist(),
            direct_pixel_extent_fraction=(raw.max(axis=0)/dims).tolist(),
            candidate_um_to_pixel_extent_fraction=(projected.max(axis=0)/dims).tolist(),
            candidate_inside_count=int(((projected >= 0) & (projected < dims)).all(axis=1).sum()),
            direct_inside_count=int(((raw >= 0) & (raw < dims)).all(axis=1).sum()),
            old_256px_crop_candidate_um=(256*mpp).tolist()))
        part = group[['cohort_id','cell_id','he_path','x','y']].copy()
        part['hypothesis_x_HE_px'] = projected[:,0]
        part['hypothesis_y_HE_px'] = projected[:,1]
        part['registration_status'] = 'unaccepted_zero_origin_scale_only_hypothesis'
        part['training_allowed'] = False
        diagnostics.append(part)
    pd.concat(diagnostics).to_csv(a.output/'diagnostic_coordinates_not_for_training.csv',index=False)
    report = dict(status='complete_unit_extent_audit_not_registration',
        source=str(a.manifest), source_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        official_schema='https://www.10xgenomics.com/support/software/xenium-onboard-analysis/2.0/analysis/xoa-output-understanding-outputs',
        official_vertex_units='micrometers',
        interpretation='Local extents support a missing unit conversion in the old pixel assumption; scale-only coordinates remain a hypothesis until local transform/registration is proven.',
        rows=summary, next_gate='Verify local export transform, origin/orientation and cell/nucleus landmarks before training or spatial split generation')
    (a.output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
