"""Fixed synthetic RGB optical-density stress on all186 accepted validation cells."""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/p0_cell_probe_manifest_1030_182424/cells.jsonl')
    rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    expected={r['cell_id']:r['canonical_pixels_sha256'] for r in json.loads((manifest.parent/'all570_decode_acceptance.json').read_text())['cells']}
    rows=[r for r in rows if r['split']=='val'];assert len(rows)==186
    conditions={'identity':None,'OD_RG_plus':[1.1,.9,1.],'OD_RG_minus':[.9,1.1,1.],'OD_light':[.9,.9,.9],'OD_dark':[1.1,1.1,1.1]}
    @lru_cache(maxsize=8)
    def image(path,digest):
        assert sha(path)==digest
        with Image.open(path) as im: return im.convert('RGB').copy()
    output=[]
    for i,row in enumerate(rows):
        im=image(row['image'],row['image_sha256']);mx,my=row['mpp_xy'];x,y=row['xy']
        w,h=round(10/mx),round(10/my);left,top=round(x-w/2),round(y-h/2)
        patch=im.crop((left,top,left+w,top+h)).resize((40,40),Image.Resampling.BICUBIC)
        rgb=np.asarray(patch).copy();assert hashlib.sha256(rgb.tobytes()).hexdigest()==expected[row['cell_id']]
        for name,factors in conditions.items():
            value=rgb if factors is None else np.rint(256*np.exp(-(-np.log((rgb.astype(np.float64)+1)/256))*np.asarray(factors))-1).clip(0,255).astype(np.uint8)
            path=args.output/(str(i)+'_'+name+'.png');Image.fromarray(value).save(path)
            with Image.open(path) as fresh: replay=np.asarray(fresh.convert('RGB')).copy()
            assert np.array_equal(replay,value) and replay.shape==(40,40,3)
            if factors is not None:
                independent=np.empty_like(rgb)
                for c,factor in enumerate(factors):
                    independent[:,:,c]=np.rint(256*((rgb[:,:,c].astype(float)+1)/256)**factor-1).clip(0,255).astype(np.uint8)
                assert np.array_equal(independent,replay)
            output.append(dict(cell_id=row['cell_id'],split='val',condition=name,crop=str(path),file_sha256=sha(path),
                               pixel_sha256=hashlib.sha256(value.tobytes()).hexdigest(),source_pixel_sha256=expected[row['cell_id']],
                               changed_channels=int(np.count_nonzero(value!=rgb))))
    assert len(output)==930 and len({(r['cell_id'],r['condition']) for r in output})==930
    (args.output/'variants.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in output))
    result=dict(status='accepted_186cell_930color_variant_pixels',worker_pid=os.getpid(),cells=186,variants=930,
                manifest_sha256=sha(manifest),variants_sha256=sha(args.output/'variants.jsonl'),source_sha256=sha(__file__),conditions=conditions,
                crop_um=10,canonical_pixels=40,no_GT_raster_read=True,no_fitted_statistics=True,
                scope='synthetic color input preparation only; no encoder/probe/tissue quality result')
    (args.output/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
