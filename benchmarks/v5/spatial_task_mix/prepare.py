"""Train-only supervision ledger and balanced local recipe; no model or val use."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS,sha


def key(text):return hashlib.sha256(text.encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    _,inventory=load_cell_catalog(TASKS);rows={k:r for k,r in inventory.items() if r['split']=='train'};assert len(rows)==1356
    labels={};counts={};bank=defaultdict(list)
    for identity,row in rows.items():
        with Image.open(row['labels']) as im:gt=np.asarray(im).copy()
        assert hashlib.sha256(gt.tobytes()).hexdigest()==row['labels_array_sha256'];labels[identity]=gt
        valid=int((gt!=255).sum());counter=Counter(gt.ravel().tolist());counts[identity]=(valid,counter)
        for target in json.loads(row['eligible_json']):
            if 0<counter[target]<valid:bank[(row['dataset'],target)].append(identity)
    for (d,t),values in bank.items():values.sort(key=lambda identity:key(f'P3local20261105|{d}|{t}|{identity}'))
    source=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/phikon_joint_tasks_1031_222610/data/training4000.jsonl');mixed=list(map(json.loads,source.read_text().splitlines()));assert len(mixed)==4000
    cursors=Counter();local=[]
    for i,task in enumerate(mixed):
        assert task['support_id'] in rows and task['query_id'] in rows
        gt=labels[task['support_id']]
        for sign in ['positive','negative']:
            for x,y in task[sign+'_xy']:
                assert 0<x<1 and 0<y<1;label=gt[int(y*gt.shape[0]),int(x*gt.shape[1])]
                assert label!=255 and ((label==task['target'])==(sign=='positive'))
        group=(task['dataset'],task['target']);pool=bank[group];assert pool,group
        identity=pool[cursors[group]%len(pool)];cursors[group]+=1
        local.append(dict(number=i,dataset=group[0],target=group[1],kind='local',protocol='local',split='train',support_id=identity,query_id=identity,
                          positive_count=len(task['positive_xy']),negative_count=len(task['negative_xy']),click_seed=int(key(f'P3click20261105|{i}|{identity}')[:8],16),
                          source_space=rows[identity]['space_key'],query_space=rows[identity]['space_key'],parent_source_index=i,needs_click_realization=True))
    def ledger(tasks,draft):
        result=dict(source_tasks=len(tasks),epochs=2,unique_input_images=len({t[k] for t in tasks for k in ['query_id','support_id']}),
                    unique_query_images=len({t['query_id'] for t in tasks}),unique_image_targets=len({(t[k],t['target']) for t in tasks for k in ['query_id','support_id']}),
                    unique_spaces=len({rows[t[k]]['space_key'] for t in tasks for k in ['query_id','support_id']}),domain_target_quota=dict(Counter(f"{t['dataset']}|{t['target']}" for t in tasks)))
        masks=valid_pixels=positive_pixels=extra_support=clicks=0
        for t in tasks:
            identities=[t['query_id']]+([t['support_id']] if t['query_id']!=t['support_id'] else [])
            extra_support+=len(identities)-1;masks+=len(identities)
            for identity in identities:
                valid,counter=counts[identity];valid_pixels+=valid;positive_pixels+=counter[t['target']]
            clicks+=(t['positive_count']+t['negative_count']) if draft else len(t['positive_xy'])+len(t['negative_xy'])
        result.update(mask_occurrences_two_epochs=masks*2,extra_support_masks_two_epochs=extra_support*2,valid_pixel_occurrences_two_epochs=valid_pixels*2,
                      target_pixel_occurrences_two_epochs=positive_pixels*2,click_occurrences_two_epochs=clicks*2,clicks_realized=not draft)
        return result
    ml,ll=ledger(mixed,False),ledger(local,True);assert ml['domain_target_quota']==ll['domain_target_quota'] and ml['click_occurrences_two_epochs']==ll['click_occurrences_two_epochs']
    path=a.output/'local_recipe.jsonl';path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in local));assert len(path.read_text().splitlines())==4000
    (a.output/'report.json').write_text(json.dumps(dict(status='accepted_train_supervision_ledger_and_unrealized_local_recipe',mixed=ml,local_draft=ll,
        source_sha256=sha(source),implementation_sha256=sha(__file__),local_recipe_sha256=sha(path),bank_counts={f'{d}|{t}':len(v) for (d,t),v in bank.items()},
        seed=20261105,val_labels_read=0,training_allowed=False,remaining='realize and audit actual local clicks; final architecture and fair supervision scope contract',
        limitation='same label pool/target quota/steps; unequal image-target and mask/pixel supervision exposure; local recipe has no actual click coordinates'),indent=2)+'\n')


if __name__=='__main__':main()
