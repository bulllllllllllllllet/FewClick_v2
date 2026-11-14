import json, hashlib, shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict, Counter
import numpy as np
from benchmarks.v5.encoder_research.continuous_data import TASKS, sha
from benchmarks.v5.support_match_study.model import DecoderData
from benchmarks.v5.paper_multiscale.data import intersects
from benchmarks.v5.wsi_quality_study.prepare import dump

BASE=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')

def order(task):
    return hashlib.sha256(f"paper-inference-v1|{task.get('index',task.get('number'))}".encode()).hexdigest()

def main():
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%m%d_%H%M%S')
    root=BASE/f'paper_inference_{stamp}';root.mkdir()
    ev=BASE/'hg_combination_1112_115539/HG_mix/evaluation'
    cache=BASE/'cell_repair_rebalance_1102_210552/normalized'
    data=DecoderData(cache,'match');base_rows=dict(data.rows)
    data.rows.update(json.loads((ev/'query_rows.json').read_text()))
    local=[]
    for t in json.loads((TASKS/'validation_maximum.json').read_text()):
        if t['prompt_group']!=0:continue
        t=dict(t,index=len(local),local=True,positive_xy=t['positive_xy'][:3],negative_xy=t['negative_xy'][:2])
        gt,_=data.labels(t['query_id']);t['query_positive_pixels']=int((gt==t['target']).sum())
        local.append(t)
    remote=json.loads((ev/'tasks.json').read_text());groups=defaultdict(list)
    for t in remote:
        gt,_=data.labels(t['query_id']);valid=int((gt!=255).sum());pos=t['query_positive_pixels']
        assert pos==int((gt==t['target']).sum())
        band='absent' if pos==0 else 'large' if pos/valid>=.2 else 'small'
        groups[t['target'],band].append(dict(t,size_band=band,valid_pixels=valid))
    chosen=[]
    for key,items in sorted(groups.items()):
        assert len(items)>=10,key
        chosen.extend(sorted(items,key=order)[:10])
    assert len(chosen)==330 and len(groups)==33
    variants=[]
    for t in sorted(chosen,key=lambda t:t['index']):
        for config in ['3+2','1+1','3+1','jitter2px','jitter5px']:
            v=dict(t,source_index=t['index'],config=config,index=len(variants))
            if '+' in config:
                a,b=map(int,config.split('+'));v.update(positive_xy=t['positive_xy'][:a],negative_xy=t['negative_xy'][:b])
            else:
                radius=int(config[6:-2]);rng=np.random.default_rng(20261114+t['index'])
                for sign in ['positive','negative']:
                    xy=np.array(t[sign+'_xy']);theta=rng.uniform(0,2*np.pi,len(xy));offset=np.stack([np.cos(theta),np.sin(theta)],axis=1)*radius/512
                    v[sign+'_xy']=np.clip(xy+offset,.5/512,511.5/512).tolist()
            variants.append(v)
    timing=sorted([t for t in chosen if t['query_positive_pixels']],key=order)[:50]+sorted([t for t in chosen if not t['query_positive_pixels']],key=order)[:50]
    timing=sorted(timing,key=order);assert len(timing)==100
    # Disclose sign flips caused by geometric perturbation; do not repair using GT.
    flips=Counter(); totals=Counter()
    for t in variants:
        if not t['config'].startswith('jitter'):continue
        gt,_=data.labels(t['support_id'])
        for sign in ['positive','negative']:
            for x,y in t[sign+'_xy']:
                value=int(gt[min(511,int(y*512)),min(511,int(x*512))]);totals[t['config'],sign]+=1
                flips[t['config'],sign]+=int(value==255 or ((value==t['target']) != (sign=='positive')))
    # Verify all actual current training inputs against every selected evaluation input.
    train_source=BASE/'same_slide_training_1110_191022/training_consumption.jsonl'
    train=list(map(json.loads,train_source.read_text().splitlines()))
    train_ids={t[k] for t in train for k in ['support_id','query_id']}
    by_space=defaultdict(list)
    for identity in train_ids:
        row=base_rows[identity];by_space[row['space_key']].append(json.loads(row['multiscale_json'])['context_um'])
    used={t[k] for t in local+chosen for k in ['support_id','query_id']}
    for identity in used:
        row=data.rows[identity];assert row['split']=='val'
        assert not any(intersects(json.loads(row['multiscale_json'])['context_um'],b) for b in by_space[row['space_key']]),identity
    for t in chosen:
        s,q=[data.rows[t[k]] for k in ['support_id','query_id']]
        assert s['space_key']==q['space_key'] and s['tile_id']!=q['tile_id']
        assert not intersects(json.loads(s['multiscale_json'])['context_um'],json.loads(q['multiscale_json'])['context_um'])
    roots={'HG_mix':BASE/'hg_combination_1112_115539/HG_mix','sansa':BASE/'same_slide_retrain_1110_204945/sansa',
           **{m:BASE/'sam_current_fair_1112_171010'/m for m in ['sam','sam2','wsi']},'med2d':BASE/'sam_med2d_current_1112_173751/med2d'}
    checkpoints={}
    for m,p in roots.items():
        a=json.loads((p/'formal/acceptance.json').read_text());ck=p/'formal/checkpoint.pt'
        assert a['status']=='complete_accepted_training' and a['updates']==2375 and a['consumed']==9500
        assert a['training_consumed_sha256']==sha(train_source) and a['checkpoint_sha256']==sha(ck)
        checkpoints[m]={'path':str(ck),'sha256':a['checkpoint_sha256'],'training_acceptance':str(p/'formal/acceptance.json')}
    for name,value in [('local_tasks',local),('robust_tasks',variants),('timing_tasks',timing),('rows',{i:data.rows[i] for i in used})]:dump(root/(name+'.json'),value)
    coverage={'local_tasks':len(local),'local_dataset_counts':dict(Counter(t['dataset'] for t in local)), 'local_classes':{d:sorted({t['target'] for t in local if t['dataset']==d}) for d in ['crc','tiger']},'remote_bases':len(chosen),'remote_configs':5,'remote_per_config':dict(Counter(t['config'] for t in variants)), 'timing_tasks':100,'jitter_label_crossings':{str(k):{'count':flips[k],'total':v} for k,v in totals.items()},'train_evaluation_4x_overlaps':0,'same_slide_remote_4x_overlaps':0,'training_source':str(train_source),'training_sha256':sha(train_source)}
    dump(root/'coverage.json',coverage)
    sources={str(p):sha(p) for base in ['benchmarks/v4','benchmarks/v5'] for p in Path(base).rglob('*.py') if '__pycache__' not in str(p)}
    c={'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'status':'prepared','cache':str(cache),'checkpoints':checkpoints,'sources':sources,'files':{name:sha(root/name) for name in ['local_tasks.json','robust_tasks.json','timing_tasks.json','rows.json','coverage.json']},'resource_pool':str(root/'leases'),'threshold':.5,'methods':list(roots),'robust_methods':['HG_mix','sansa'],'local_protocol':'302 fixed group0 validation tasks, 3 positive/2 negative, current trained local=True interface; separate CRC/TIGER class macro','robust_protocol':'330 fixed hash-selected remote tasks: 10 per 11class x small/large/absent; 3+2,1+1,3+1,2px and5px radial jitter; signs retained; no query GT in prompts','timing_protocol':'100 same remote tasks (50present/50absent), three warmup tasks; same physical GPU3 serial six methods; synchronized wall time. First input preparation+encode+decode, then changed support1+1 prompts with image features reused, including rebuilding support memory. No model-load/scoring in timer. Mean; tensor cache bytes; peak allocated/reserved bytes. No new training.'}
    dump(root/'contract.json',c)
    Path('/tmp/fewclick_inference_root').write_text(str(root))
    print(json.dumps({'root':str(root),'coverage':coverage},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
