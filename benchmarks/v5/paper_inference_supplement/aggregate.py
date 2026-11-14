import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from .common import verify
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump


def macro(items,key):
    groups=defaultdict(list)
    for task,m in items:
        if m.get(key) is not None:groups[task['target']].append(m[key])
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


def summary(items):
    pos=[(t,m) for t,m in items if m['positive']];neg=[(t,m) for t,m in items if not m['positive']]
    return dict(n=len(items),present=len(pos),absent=len(neg),class_macro_Dice=macro(pos,'dice'),class_macro_BF1=macro(pos,'bf1_um2'),macro_absent_FPA=macro(neg,'false_positive_fraction'),boundary_evaluable=sum(m['boundary_evaluable'] for _,m in items))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;c=verify(r)
    results={'local':{},'robust':{},'timing':{}};accepted=[];input_hashes={}
    for stage in results:
        tasks=json.loads((r/(stage+'_tasks.json')).read_text());methods=c['robust_methods'] if stage=='robust' else c['methods']
        for method in methods:
            out=r/f'{stage}_{method}';acc=json.loads((out/'acceptance.json').read_text())
            assert acc['status']=='complete_accepted' and acc['rows']==len(tasks) and acc['weights_unchanged']
            assert acc['contract_sha256']==sha(r/'contract.json') and acc['checkpoint_sha256']==c['checkpoints'][method]['sha256']
            assert acc['tasks_sha256']==sha(r/(stage+'_tasks.json'))
            for key,file in [('rows_sha256','rows.json'),('masks_sha256','masks.npy'),('input_hashes_sha256','input_hashes.json'),('timing_rows_sha256','timing_rows.json'),('timing_summary_sha256','timing_summary.json')]:
                if key in acc:assert acc[key]==sha(out/file),(method,file)
            for identity,h in json.loads((out/'input_hashes.json').read_text()).items():
                if identity in input_hashes:assert input_hashes[identity]==h
                input_hashes[identity]=h
            accepted.append({'path':str(out/'acceptance.json'),'sha256':sha(out/'acceptance.json')})
            if stage=='timing':
                timing=json.loads((out/'timing_rows.json').read_text());saved=json.loads((out/'timing_summary.json').read_text())
                assert len(timing)==100 and [t['index'] for t in timing]==[t['index'] for t in tasks]
                for key in timing[0]:
                    if key!='index':assert saved[key]==float(np.mean([t[key] for t in timing]))
                results[stage][method]=saved;continue
            rows=json.loads((out/'rows.json').read_text());assert [v['index'] for v in rows]==[t['index'] for t in tasks]
            items=[(t,row['metrics']) for t,row in zip(tasks,rows)]
            if stage=='local':
                results[stage][method]={d:summary([(t,m) for t,m in items if t['dataset']==d]) for d in ['crc','tiger']}
            else:
                results[stage][method]={cfg:summary([(t,m) for t,m in items if t['config']==cfg]) for cfg in ['3+2','1+1','3+1','jitter2px','jitter5px']}
                # Independently bind unchanged3+2 control to the accepted full11000 run.
                ev=Path(c['checkpoints'][method]['path']).parent.parent/'evaluation';previous={};packed={}
                for shard in [0,1]:
                    old=ev/f'{method}_{shard}';receipt=json.loads((old/'acceptance.json').read_text())
                    assert receipt['checkpoint_sha256']==c['checkpoints'][method]['sha256']
                    assert receipt['rows_sha256']==sha(old/'rows.json') and receipt['masks_sha256']==sha(old/'masks.npy')
                    mm=np.load(old/'masks.npy',mmap_mode='r')
                    for i,row in enumerate(json.loads((old/'rows.json').read_text())):previous[row['index']]=row['metrics'];packed[row['index']]=mm[i]
                actual=np.load(out/'masks.npy',mmap_mode='r');n=0
                for i,(t,m) in enumerate(items):
                    if t['config']=='3+2':
                        assert previous[t['source_index']]==m,(method,t['source_index'])
                        assert np.array_equal(packed[t['source_index']],actual[i]);n+=1
                assert n==330
                results[stage][method]['control_exact_reproduced']=n
    dump(r/'report.json',dict(status='complete_accepted',results=results,coverage=json.loads((r/'coverage.json').read_text()),receipts=accepted,notes='302 group0 local: current trained local interfaces; CRC/TIGER separate. Remote robustness330 fixed stratified subset, same-slide, changed support prompts. Runtime100 remote tasks sameGPU3, arithmetic means, includes IO in first stage; cached features per support-query pair, not whole-slide persistent cache. One frozen training seed.'))
    dump(r/'acceptance.json',dict(status='complete_accepted',report_sha256=sha(r/'report.json'),contract_sha256=sha(r/'contract.json'),accepted_jobs=len(accepted),shared_input_hashes=len(input_hashes),robust_baseline_exact=660,timing_tasks_per_method=100))
    print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
