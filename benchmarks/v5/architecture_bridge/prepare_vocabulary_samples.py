"""Train-only matched XCell/LEMON sampling inventory; no codebook or labels fitted."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from .data import BridgeData
from benchmarks.v5.encoder_research.continuous_data import sha


def priority(tile,scale,index):
    return hashlib.sha256(f'P2sample20261103|{tile}|{scale}|{index}'.encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',action='store_true');p.add_argument('--accept',action='store_true');a=p.parse_args()
    data=BridgeData();train=[data.rows[k] for k in sorted(data.rows) if data.rows[k]['split']=='train'];assert len(train)==1356
    if a.gate:train=[min([r for r in train if r['dataset']==d],key=lambda r:r['tile_id']) for d in ['crc','tiger']]
    if a.accept:
        report=json.loads((a.output/'report.json').read_text());assert report['identity']==data.identity and report['source_sha256']==sha(__file__) and report['gate']==a.gate
        records=[json.loads(s) for s in (a.output/'samples.jsonl').read_text().splitlines()]
        arrays=np.load(a.output/'samples.npz',allow_pickle=False);assert arrays['xcell'].shape==arrays['lemon'].shape==(len(records),64)
        assert np.isfinite(arrays['xcell']).all() and np.isfinite(arrays['lemon']).all()
        groups={}
        for i,r in sorted(enumerate(records),key=lambda pair:(pair[1]["tile_id"],pair[1]["scale"],pair[1]["cell_row"])):
            assert data.rows[r['tile_id']]['split']=='train' and r['key']==priority(r['tile_id'],r['scale'],r['cell_row'])
            suffix='' if r['scale']==0 else '_'+str(r['scale']);key='cells'+suffix
            for arm in ['xcell','lemon']:
                expected=data.inputs(r['tile_id'],arm)[key][0,r['cell_row'],10:].numpy()
                assert np.array_equal(expected,arrays[arm][i])
            groups[r['dataset']+'|'+str(r['scale'])]=groups.get(r['dataset']+'|'+str(r['scale']),0)+1
        assert len(groups)==6 and all(n<=5000 for n in groups.values())
        (a.output/'independent_acceptance.json').write_text(json.dumps(dict(status='accepted_train_only_matched_vocabulary_samples',worker_pid=os.getpid(),samples=len(records),groups=groups,validation_samples=0,index_sha256=sha(a.output/'samples.jsonl'),array_sha256=sha(a.output/'samples.npz'),gate=a.gate,scope='matched input sampling only; no codebook fitting or encoder choice acceptance'),indent=2)+'\n');return
    a.output.mkdir(exist_ok=False);groups={}
    for row in train:
        for scale in range(3):
            suffix='' if scale==0 else '_'+str(scale);key='cells'+suffix
            left=data.inputs(row['tile_id'],'xcell')[key][0].numpy();right=data.inputs(row['tile_id'],'lemon')[key][0].numpy()
            assert left.shape==right.shape and np.array_equal(left[:,:10],right[:,:10])
            indices=sorted(range(len(left)),key=lambda i:priority(row['tile_id'],scale,i))[:64]
            group=groups.setdefault(row['dataset']+'|'+str(scale),[])
            for i in indices:group.append((priority(row['tile_id'],scale,i),row['tile_id'],scale,i,left[i,10:].copy(),right[i,10:].copy()))
    chosen=[]
    for key,values in sorted(groups.items()):
        for item in sorted(values,key=lambda x:x[0])[:5000]:chosen.append((key,*item))
    records=[dict(dataset=g.split('|')[0],key=k,tile_id=tile,scale=s,cell_row=i) for g,k,tile,s,i,_,_ in chosen]
    np.savez_compressed(a.output/'samples.npz',xcell=np.stack([x[-2] for x in chosen]),lemon=np.stack([x[-1] for x in chosen]))
    (a.output/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (a.output/'report.json').write_text(json.dumps(dict(status='complete_matched_train_only_samples_unaccepted',identity=data.identity,source_sha256=sha(__file__),gate=a.gate,train_frames=len(train),samples=len(chosen),per_frame_scale_cap=64,per_dataset_scale_cap=5000,labels_read=False,fit_performed=False),indent=2)+'\n')


if __name__=='__main__':main()
