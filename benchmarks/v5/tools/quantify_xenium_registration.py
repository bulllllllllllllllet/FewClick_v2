"""Fixed-sample HE/DAPI correspondence proxy; not expert landmark ground truth."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import h5py
import numpy as np
import pandas as pd
from skimage.color import rgb2hed
from benchmarks.v5.tools.render_xenium_unit_diagnostics import read_crop


def match(he, dapi):
    scores=cv2.matchTemplate(he.astype('float32'),dapi.astype('float32'),cv2.TM_CCOEFF_NORMED)
    # 40um HE versus20um DAPI, both0.5um/pixel; search +/-4um.
    window=scores[12:29,12:29];iy,ix=np.unravel_index(np.argmax(window),window.shape)
    dx,dy=(ix-8)*.5,(iy-8)*.5
    return dict(zero=float(scores[20,20]),peak=float(window[iy,ix]),dx_um=dx,dy_um=dy,
                displacement_um=float(np.hypot(dx,dy)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.time()
    manifest=Path('/nfs-medical3/zyh/fewclick_v2/multisection_cell_manifest/multisection_cell_manifest_0907_161850/sampled_cells.csv')
    f=pd.read_csv(manifest);rows=[];cohorts=[]
    for cohort,g in f[f.platform=='Xenium'].groupby('cohort_id'):
        g=g.copy();g['key']=[hashlib.sha256(f'{cohort}|{v}'.encode()).hexdigest() for v in g.cell_id]
        selected=g.sort_values('key').head(32)
        with h5py.File(g.h5ad_path.iloc[0]) as h:
            mpp=np.asarray(h['uns']['H&E resolution'][()]);dmpp=float(h['uns']['DAPI resolution'][()])
        features=[]
        for r in selected.itertuples():
            raw=np.array([r.x,r.y]);hs=round(40/mpp[0]);ds=round(20/dmpp)
            he=read_crop(r.he_path,raw/mpp,hs);old=read_crop(r.he_path,raw,hs)
            dapi=read_crop(Path(r.he_path).parent/'DAPI.tif',raw/dmpp,ds)
            # Fixed stain transform, no learned fit or class labels.
            hmap=cv2.resize(rgb2hed(he)[...,0],(80,80)).astype('float32')
            omap=cv2.resize(rgb2hed(old)[...,0],(80,80)).astype('float32')
            dmap=cv2.resize(dapi.astype('float32'),(40,40))
            features.append((r.cell_id,hmap,omap,dmap))
        for i,(identity,hmap,omap,dmap) in enumerate(features):
            paired=match(hmap,dmap);legacy=match(omap,dmap)
            shuffled=match(hmap,features[(i+1)%len(features)][3])
            rows.append(dict(cohort=cohort,cell_id=identity,paired=paired,legacy=legacy,shuffled=shuffled,
                passes_proxy_gate=paired['peak']>=.35 and paired['displacement_um']<=2,
                training_allowed=False))
        group=rows[-32:]
        cohorts.append(dict(cohort=cohort,n=32,proxy_pass_fraction=float(np.mean([r['passes_proxy_gate'] for r in group])),
            median_paired_zero=float(np.median([r['paired']['zero'] for r in group])),
            median_legacy_zero=float(np.median([r['legacy']['zero'] for r in group])),
            median_shuffled_zero=float(np.median([r['shuffled']['zero'] for r in group])),
            median_displacement_um=float(np.median([r['paired']['displacement_um'] for r in group]))))
        (a.output/'partial.json').write_text(json.dumps(dict(rows=rows,cohorts=cohorts),indent=2)+'\n')
        print(json.dumps(cohorts[-1]),flush=True)
    decision=all(c['proxy_pass_fraction']>=.75 and c['median_paired_zero']>c['median_shuffled_zero'] for c in cohorts)
    report=dict(status='complete_quantitative_registration_proxy',proxy_gate_passed=decision,rows=rows,cohorts=cohorts,
        elapsed_seconds=time.time()-start,registration_accepted=False,
        limitation='Correlation peak is an image-structure proxy, not a ground-truth cell identity or landmark error. No per-cell inferred shift is applied.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],proxy_gate_passed=decision,seconds=report['elapsed_seconds'])))


if __name__=='__main__':main()
