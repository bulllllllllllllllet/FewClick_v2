"""Fit fine-train statistics, export all three repaired cell views, replay them."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmarks.v5.encoder_research.continuous_data import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    proof=json.loads((a.raw/'acceptance.json').read_text());assert proof['status']=='accepted_repaired_raw_cells'
    assert proof['report_sha256']==sha(a.raw/'report.json')
    report=json.loads((a.raw/'report.json').read_text());rows=report['frames'];assert len(rows)==1737
    total=np.zeros(64,np.float64);squares=total.copy();count=0
    for row in rows:
        assert sha(row['file'])==row['sha256']
        if row['split']=='train':
            with np.load(row['file']) as f:x=f['reg_0'].astype(np.float64)
            total+=x.sum(0);squares+=(x*x).sum(0);count+=len(x)
    assert count>0
    mean=total/count;std=np.sqrt(np.maximum(squares/count-mean**2,0));std[std==0]=1
    a.output.mkdir(parents=True,exist_ok=False);np.savez(a.output/'transforms.npz',xcell_mean=mean,xcell_std=std)
    exported=[]
    for row in rows:
        arrays={}
        with np.load(row['file']) as f:
            for s in range(3):
                cells=f[f'cells_{s}'];value=np.concatenate([cells[:,:10],(cells[:,10:].astype(np.float64)-mean)/std],1).astype(np.float32)
                assert np.array_equal(value[:,:10],cells[:,:10]) and np.isfinite(value).all()
                arrays[f'cells_{s}']=value
        path=a.output/(row['tile_id']+'.npz');np.savez_compressed(path,**arrays)
        exported.append(row|dict(normalized_file=str(path),normalized_sha256=sha(path)))
    # Independent fitting replay uses direct concatenation/std, not the streaming moments.
    blocks=[]
    for row in rows:
        if row['split']=='train':
            with np.load(row['file']) as f:blocks.append(f['reg_0'].astype(np.float64))
    matrix=np.concatenate(blocks);ref_mean=matrix.mean(0);ref_std=matrix.std(0);ref_std[ref_std==0]=1
    assert np.allclose(mean,ref_mean,atol=1e-12,rtol=1e-10) and np.allclose(std,ref_std,atol=1e-12,rtol=1e-10)
    for row in exported:
        assert sha(row['normalized_file'])==row['normalized_sha256']
        with np.load(row['file']) as raw,np.load(row['normalized_file']) as saved:
            for s in range(3):
                expected=np.concatenate([raw[f'cells_{s}'][:,:10],(raw[f'reg_{s}'].astype(np.float64)-mean)/std],1).astype(np.float32)
                assert np.array_equal(saved[f'cells_{s}'],expected)
    (a.output/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in exported))
    (a.output/'acceptance.json').write_text(json.dumps(dict(status='accepted_repaired_normalized_cells',frames=len(rows),fine_train_fitting_cells=count,validation_fitting_cells=0,
        index_sha256=sha(a.output/'index.jsonl'),transforms_sha256=sha(a.output/'transforms.npz'),raw_acceptance_sha256=sha(a.raw/'acceptance.json'),source_sha256=sha(__file__),
        functional_learning_accepted=False,P3_training_allowed=False),indent=2)+'\n')


if __name__=='__main__':main()
