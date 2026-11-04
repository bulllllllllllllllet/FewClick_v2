"""Scoring-only size/components/FP-distance analysis of the accepted reference."""
import argparse,json,os,time
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label,distance_transform_edt
from benchmarks.v5.encoder_research.continuous_data import TASKS,sha
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    s=json.loads((a.root/'settings.json').read_text());source=Path(s['reference_match']);proof=json.loads((source/'independent_acceptance.json').read_text())
    for name,key in [('rows.json','rows_sha256'),('masks.npz','masks_sha256')]:assert sha(source/name)==proof[key]
    rows=json.loads((source/'rows.json').read_text());masks=np.load(source/'masks.npz')['masks'];_,catalog=load_cell_catalog(TASKS)
    output=a.root/'diagnostic';output.mkdir(exist_ok=False);results=[];start=time.monotonic()
    @lru_cache(maxsize=256)
    def truth_data(identity,target):
        r=catalog[identity];gt=np.asarray(Image.open(r['labels']));assert sha_array(gt)==r['labels_array_sha256']
        valid=gt!=255;truth=(gt==target)&valid;mpp=np.asarray(json.loads(r['input_mpp_xy_json']))
        components,n=label(truth,np.ones((3,3)));areas=np.bincount(components.ravel())[1:]
        distance=distance_transform_edt(~truth,sampling=mpp[::-1]) if truth.any() else None
        return valid,truth,mpp,components,areas,distance
    for index,row in enumerate(rows):
        if row['budget']!='3+2':continue
        task=row['plan'];valid,truth,mpp,components,areas,distance=truth_data(task['query_id'],task['target'])
        mask=np.unpackbits(masks[index],count=512*512).reshape(512,512).astype(bool)&valid;tp=int((mask&truth).sum());fp=int((mask&~truth).sum());fn=int((~mask&truth).sum())
        assert (tp,fp,fn)==tuple(row['metrics'][k] for k in ['tp','fp','fn'])
        overlaps=np.bincount(components[mask],minlength=len(areas)+1)[1:];fractions=overlaps/areas if len(areas) else np.empty(0)
        pos=int(truth.sum());area=pos/int(valid.sum());size='absent' if pos==0 else 'small' if area<.05 else 'large' if area>=.25 else 'medium'
        y,x=np.nonzero(truth);bbox=float((x.max()-x.min()+1)*(y.max()-y.min()+1)*mpp.prod()) if pos else None
        results.append(dict(plan=task,scope=row['scope'],size=size,area_fraction=area,physical_area_um2=pos*float(mpp.prod()),
            GT_component_count=len(areas),GT_component_area_um2_median=float(np.median(areas)*mpp.prod()) if len(areas) else None,
            component_any_overlap_recall=float((overlaps>0).mean()) if len(areas) else None,
            component_10pct_overlap_recall=float((fractions>=.1).mean()) if len(areas) else None,
            FP_beyond8um_fraction=float(((mask&~truth)&(distance>8)).sum()/fp) if fp and distance is not None else None,
            GT_bbox_area_um2=bbox,GT_fill_bbox=pos*float(mpp.prod())/bbox if bbox else None,metrics=row['metrics']))
        if len(results)%50==0:print(json.dumps(dict(stage='size_diagnostic',tasks=len(results),total=1237,worker_pid=os.getpid())),flush=True)
    assert len(results)==1237
    groups={}
    for domain in ['crc','tiger']:
        for scope in ['local','remote']:
            for size in ['small','medium','large','absent']:
                rr=[r for r in results if r['plan']['dataset']==domain and r['scope']==scope and r['size']==size]
                groups[domain+'|'+scope+'|'+size]=dict(n=len(rr),**{field:float(np.mean([r[field] for r in rr if r[field] is not None])) if any(r[field] is not None for r in rr) else None for field in ['component_10pct_overlap_recall','FP_beyond8um_fraction','GT_fill_bbox']})
    report=dict(status='accepted_scoring_only_size_components_diagnostic',rows=results,groups=groups,rows_count=len(results),
                labels_used_only_for_scoring=True,components_8_connected=True,all_components_retained=True,FP_distance_um=8,
                reference_receipt_sha256=sha(source/'independent_acceptance.json'),source_sha256=sha(__file__),worker_pid=os.getpid(),seconds=time.monotonic()-start)
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(stage='complete',tasks=len(results))),flush=True)


def sha_array(x):
    import hashlib
    return hashlib.sha256(x.tobytes()).hexdigest()


if __name__=='__main__':main()
