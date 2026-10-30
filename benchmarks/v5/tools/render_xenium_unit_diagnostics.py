"""Fast bounded TIFF reads for diagnostic HE/DAPI coordinate comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import tifffile
import zarr


def read_crop(path, center, side):
    x,y = (round(float(v)-side/2) for v in center)
    with tifffile.TiffFile(path) as t:
        store=t.pages[0].aszarr()
        try:
            a=zarr.open(store,mode='r')
            assert x>=0 and y>=0 and x+side<=a.shape[1] and y+side<=a.shape[0]
            return np.asarray(a[y:y+side,x:x+side]).copy()
        finally:
            store.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=Path('/nfs-medical3/zyh/fewclick_v2/multisection_cell_manifest/multisection_cell_manifest_0907_161850/sampled_cells.csv')
    frame=pd.read_csv(manifest);rows=[];start=time.time()
    for cohort,g in frame[frame.platform=='Xenium'].groupby('cohort_id'):
        g=g.copy();g['key']=[hashlib.sha256(f'{cohort}|{v}'.encode()).hexdigest() for v in g.cell_id]
        with h5py.File(g.h5ad_path.iloc[0]) as h:
            mpp=np.asarray(h['uns']['H&E resolution'][()],float)
            dmpp=float(h['uns']['DAPI resolution'][()])
        assert abs(mpp[0]-mpp[1])<1e-4
        montage=Image.new('RGB',(768,3*280),'white');draw=ImageDraw.Draw(montage)
        for j,r in enumerate(g.sort_values('key').head(3).itertuples()):
            raw=np.array([r.x,r.y]);he=read_crop(r.he_path,raw,128)
            old=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/p0_inventory_1030_180457')/f'{cohort}_{r.cell_id}.png'
            parity=None
            if old.exists():
                parity=bool(np.array_equal(np.asarray(Image.open(old)),he));assert parity
            # All panels display approximately20um, but raw center is the historical pixel assumption.
            side=round(20/mpp[0]);ds=round(20/dmpp)
            dapi_path=Path(r.he_path).parent/'DAPI.tif'
            views=[read_crop(r.he_path,raw,side),read_crop(r.he_path,raw/mpp,side),read_crop(dapi_path,raw/dmpp,ds)]
            for k,v in enumerate(views):
                if v.ndim==2:
                    lo,hi=np.percentile(v,[1,99]);v=(255*np.clip((v.astype(float)-lo)/max(hi-lo,1),0,1)).astype('uint8')
                panel=Image.fromarray(v).convert('RGB').resize((256,256))
                pdrawing=ImageDraw.Draw(panel);pdrawing.line((120,128,136,128),fill='red');pdrawing.line((128,120,128,136),fill='red')
                montage.paste(panel,(k*256,j*280+24))
                draw.text((k*256+2,j*280+2),['old HE px','candidate HE um/MPP','candidate DAPI um/MPP'][k],fill='black')
            rows.append(dict(cohort=cohort,cell_id=r.cell_id,raw_xy=raw.tolist(),candidate_he_xy=(raw/mpp).tolist(),
                raw_crop_reader_pixel_parity=parity,he_side=side,dapi_side=ds,display_um_approx=20,
                training_allowed=False,registration_accepted=False))
        montage.save(a.output/f'{cohort}_coordinate_comparison.png')
    report=dict(status='complete_diagnostic_views_not_registration',rows=rows,elapsed_seconds=time.time()-start,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        assumptions='candidate assumes common physical origin and no rotation; validate local export alignment before use',
        annotation='red cross marks assumed cell center; DAPI percentile contrast per crop for inspection only')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'seconds':report['elapsed_seconds'],'instances':len(rows)}))


if __name__=='__main__':main()
