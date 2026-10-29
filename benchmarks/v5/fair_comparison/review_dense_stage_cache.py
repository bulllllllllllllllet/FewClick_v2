"""Compare saved graph probability and original final output, without new inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def counts(probability, truth, valid):
    mask=probability>=.5
    tp=int((mask&truth&valid).sum());fp=int((mask&~truth&valid).sum());fn=int((~mask&truth&valid).sum())
    return dict(tp=tp,fp=fp,fn=fn,dice=2*tp/(2*tp+fp+fn),
                precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn))


def main(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2);a.output.mkdir(exist_ok=False)
    paths=sorted((a.pilot/'val_cache').glob('case*.pth'));assert len(paths)==32
    bindings={};rows=[]
    for p in paths:
        c=torch.load(p,map_location='cpu',weights_only=False)
        q=a.pilot/'original'/(p.stem+'.npz')
        bindings[str(p)]=sha(p);bindings[str(q)]=sha(q)
        with np.load(q,allow_pickle=False) as data:
            truth=c['truth'][0].numpy();valid=c['valid'][0].numpy()
            assert np.array_equal(data['truth'],truth) and np.array_equal(data['valid'],valid)
            base=c['features']['base'][0].numpy();final=data['probability']
            assert base.shape==final.shape==truth.shape==(512,512)
            assert np.isfinite(base).all() and np.isfinite(final).all()
            rows.append(dict(case=p.stem,dataset=c['plan']['dataset'],target=c['plan']['target'],
                query_id=c['plan']['query_id'],GT_fraction=float((truth&valid).sum()/valid.sum()),
                base=counts(base,truth,valid),dense=counts(final,truth,valid),
                feature_shapes={k:list(v.shape) for k,v in c['features'].items()}))
    strata={}
    for label,selected in [('all',rows),('small',[r for r in rows if r['GT_fraction']<.05]),
                           ('large',[r for r in rows if r['GT_fraction']>.5])]:
        strata[label]=dict(n=len(selected),dense_dice_improved=sum(r['dense']['dice']>r['base']['dice'] for r in selected),
            **{stage:{name:float(np.mean([r[stage][name] for r in selected])) for name in ('dice','precision','recall','fp')}
               for stage in ('base','dense')})
    result=dict(status='complete_saved_stage_comparison_pending_independent_recount',worker_pid=os.getpid(),
        source_sha256=sha(__file__),bindings=bindings,rows=rows,strata=strata,
        limitations='32 posthoc class-area-extreme development pairs, single seed CPU FP32. Base is intermediate within the Dense-trained model, NOT a separately trained region-only control or a performance bound.')
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(worker_pid=os.getpid(),strata=strata)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
