"""Spatially separated episodes; only the support GT produces simulated clicks."""
from collections import defaultdict,Counter
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path

import pyvips
import numpy as np
from PIL import Image
from scipy.ndimage import label as component_labels
import torch

from benchmarks.v5.multidomain_data.prepare import read_rows,sha
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor
from .catalog import interior_labels,disjoint,TARGETS

KINDS=['local','same_present','cross_present','same_absent','cross_absent']
KIND_PROBS=[.30,.35,.15,.14,.06]
DATASETS=['crc','tiger','ignite','ocelot']
DATA_PROBS=[.30,.25,.25,.20]


def spread_clicks(mask,count,rng):
    components,n=component_labels(mask)
    if int(mask.sum())<count:raise ValueError('insufficient interior pixels for requested click budget')
    selected=[]
    for identity in rng.permutation(n)+1:
        positions=np.argwhere(components==identity);selected.append(positions[rng.integers(len(positions))])
        if len(selected)==count:break
    if len(selected)<count:
        available=mask.copy()
        for y,x in selected:available[y,x]=False
        candidates=np.argwhere(available)
        selected.extend(candidates[rng.choice(len(candidates),count-len(selected),replace=False)])
    return [[(float(x)+.5)/mask.shape[1],(float(y)+.5)/mask.shape[0]] for y,x in selected]


class EpisodeCorpus:
    def __init__(self,catalog,seed=20261021,prompt_policy='representative'):
        if prompt_policy not in ('legacy','representative'):raise ValueError('unknown prompt policy')
        self.prompt_policy=prompt_policy
        self.catalog=Path(catalog);self.seed=seed;self.rows=read_rows(self.catalog/'tiles.parquet')
        policy_path=self.catalog/'training_policy.json'
        self.training_policy_sha256=sha(policy_path.read_bytes()) if policy_path.exists() else None
        self.training_policy=json.loads(policy_path.read_text()) if policy_path.exists() else None
        if self.training_policy is not None:
            policy=self.training_policy;datasets=policy['datasets'];probs=policy['probabilities']
            if (policy['catalog_sha256']!=sha((self.catalog/'tiles.parquet').read_bytes())
                    or not datasets or len(set(datasets))!=len(datasets) or not set(datasets)<=set(DATASETS)
                    or len(probs)!=len(datasets) or not np.isfinite(probs).all()
                    or any(p<=0 for p in probs) or not np.isclose(sum(probs),1.)
                    or policy['qualities']!=['original']):raise ValueError('invalid original-only training policy')
            for row in self.rows:
                if row['split']=='train' and (row['dataset'] not in datasets or not row['training_allowed']
                        or set(json.loads(row['variants_json']))!={'original'}):raise ValueError('training catalog violates quality policy')
        if any(r['dataset']=='puma' or r['split']=='test' for r in self.rows):raise ValueError('heldout data entered episode corpus')
        self.index={r['tile_id']:i for i,r in enumerate(self.rows)}
        if len(self.index)!=len(self.rows):raise ValueError('duplicate episode tile')
        self.counts=[{int(k):int(v) for k,v in json.loads(r['counts_json']).items()} for r in self.rows]
        self.eligible=[set(json.loads(r['eligible_json'])) for r in self.rows]
        self.boxes=np.array([[r[k] for k in ['x0_um','y0_um','x1_um','y1_um']] for r in self.rows])
        self.keys=np.array([r['space_key'] for r in self.rows],object)
        self.patients=np.array([r['patient_id'] for r in self.rows],object)
        self.verified=np.array([r['slide_verified'] for r in self.rows],bool)
        self.queries={};self.pools={};coverage=[]
        for split in ['train','val']:
            self.pools[split]={}
            for dataset in DATASETS:
                base=[i for i,r in enumerate(self.rows) if r['dataset']==dataset and r['split']==split]
                pools={kind:defaultdict(lambda:defaultdict(list)) for kind in KINDS}
                for target in TARGETS[dataset]:
                    support=[i for i in base if target in self.eligible[i]]
                    for absent in [False,True]:
                        q=np.array([i for i in base if (self.counts[i].get(target,0)==0 if absent else self.counts[i].get(target,0)>=64)],np.int64)
                        self.queries[split,dataset,target,absent]=q
                        by_key=defaultdict(list)
                        for qi in q:by_key[self.keys[qi]].append(qi)
                        verified_count=int(self.verified[q].sum());key_count=Counter(self.keys[q[self.verified[q]]])
                        known_patient=[self.patients[qi] for qi in q if self.patients[qi] is not None];patient_count=Counter(known_patient)
                        for i in support:
                            for relation in ['same','cross']:
                                if relation=='same':possible=self.candidates(i,np.array(by_key[self.keys[i]],np.int64),relation).size>0
                                elif self.verified[i]:possible=verified_count-key_count[self.keys[i]]>0
                                else:possible=self.patients[i] is not None and len(known_patient)-patient_count[self.patients[i]]>0
                                if possible:
                                    kind=relation+('_absent' if absent else '_present')
                                    pools[kind][self.rows[i]['organ']][target].append(i)
                    for i in support:pools['local'][self.rows[i]['organ']][target].append(i)
                self.pools[split][dataset]=pools
                for kind,organs in pools.items():
                    coverage.append(dict(split=split,dataset=dataset,kind=kind,eligible_supports=sum(len(ids) for targets in organs.values() for ids in targets.values()),organs=list(organs)))
        self.coverage=coverage

    def candidates(self,support,indices,relation):
        if not len(indices):return indices
        if relation=='same':
            near=indices[self.keys[indices]==self.keys[support]]
            return near[disjoint(self.boxes[support],self.boxes[near])] if len(near) else near
        if self.verified[support]:
            keep=self.verified[indices]&(self.keys[indices]!=self.keys[support])
        else:
            # IGNITE does not provide WSI IDs. Different identified patients
            # ensure distinct specimens, still reported only as cross-image.
            keep=(self.keys[indices]!=self.keys[support])&(self.patients[indices]!=self.patients[support])
            if self.patients[support] is None:keep[:]=False
            keep&=np.array([p is not None for p in self.patients[indices]])
        return indices[keep]

    def plan(self,number,*,split='train',crc_only=False,quality_aug=True,kind=None,dataset=None,fixed_clicks=False):
        rng=np.random.default_rng(self.seed+int(number));quality_rng=np.random.default_rng(self.seed+100000000+int(number))
        policy=getattr(self,'training_policy',None) if split=='train' else None
        datasets=policy['datasets'] if policy else DATASETS;probs=policy['probabilities'] if policy else DATA_PROBS
        dataset=dataset or ('crc' if crc_only else str(rng.choice(datasets,p=probs)))
        if policy and dataset not in datasets:raise ValueError('excluded training dataset requested')
        kind=kind or str(rng.choice(KINDS,p=KIND_PROBS))
        pool=self.pools[split][dataset][kind]
        if not pool:raise ValueError(f'no feasible {split}/{dataset}/{kind}; never substitute a different protocol')
        organ=str(rng.choice(sorted(pool)));target=int(rng.choice(sorted(pool[organ])))
        s=int(rng.choice(pool[organ][target]));q=s
        if kind!='local':
            candidates=self.candidates(s,self.queries[split,dataset,target,kind.endswith('absent')],kind.split('_')[0])
            q=int(rng.choice(candidates))
        qualities=['original','0.8','1.0'];augment=quality_aug and policy is None
        sq=str(quality_rng.choice(qualities,p=[.5,.25,.25])) if augment else 'original'
        qq=str(quality_rng.choice(qualities,p=[.5,.25,.25])) if augment else 'original'
        click_rng=np.random.default_rng(self.seed+200000000+int(number))
        positive,negative=(3,2) if fixed_clicks else tuple(map(int,click_rng.integers(1,6,size=2)))
        gt=self.labels(s);inside=interior_labels(gt,json.loads(self.rows[s]['input_mpp_xy_json']))
        p=spread_clicks(inside==target,positive,click_rng);n=spread_clicks((inside!=target)&(inside!=255),negative,click_rng)
        same=kind=='local';verified=self.rows[s]['slide_verified'] and self.rows[q]['slide_verified']
        protocol='local' if same else ('same_wsi' if kind.startswith('same') and verified else 'same_roi' if kind.startswith('same') else 'cross_wsi' if verified else 'cross_image')
        plan=dict(number=int(number),support_id=self.rows[s]['tile_id'],query_id=self.rows[q]['tile_id'],dataset=dataset,target=target,
            support_quality=sq,query_quality=qq,kind=kind,protocol=protocol,split=split,positive_xy=p,negative_xy=n,
            support_organ=self.rows[s]['organ'],query_organ=self.rows[q]['organ'],gt_simulated_clicks=True)
        if getattr(self,'prompt_policy','representative')=='representative':
            from .click_policy import repoint
            return repoint(self,plan)
        return plan

    @lru_cache(maxsize=128)
    def labels(self,index):
        row=self.rows[index];gt=np.asarray(Image.open(row['labels'])).copy()
        if sha(gt.tobytes())!=row['labels_array_sha256']:raise ValueError('GT identity changed')
        return gt

    @lru_cache(maxsize=128)
    def inputs(self,index,quality):
        row=self.rows[index];v=json.loads(row['variants_json'])[quality]
        rgb_bytes=Path(v['image']).read_bytes();cell_bytes=Path(v['cells']).read_bytes()
        if sha(rgb_bytes)!=v['image_sha256'] or sha(cell_bytes)!=v['cells_sha256']:raise ValueError('image/cell identity changed')
        image=np.asarray(Image.open(BytesIO(rgb_bytes)).convert('RGB')).copy()
        with np.load(BytesIO(cell_bytes),allow_pickle=False) as data:cells=data['features'].copy();total=int(data['total'])
        if cells.shape!=(len(cells),74) or not np.isfinite(cells).all() or (len(cells) and not np.all(cells[:,3]==1)):raise ValueError('cell feature contract')
        return dict(image=he_to_tensor(image)[None],cells=torch.from_numpy(cells)[None],cell_valid=torch.ones((1,len(cells)),dtype=torch.bool),total_cell_count=torch.tensor([total],dtype=torch.float32))

    def materialize(self,plan,device):
        policy=getattr(self,'training_policy',None)
        if policy and plan['split']=='train' and (plan['dataset'] not in policy['datasets']
                or plan['support_quality']!='original' or plan['query_quality']!='original'):
            raise ValueError('excluded low-resolution training input requested')
        s=self.index[plan['support_id']];q=self.index[plan['query_id']]
        if self.rows[s]['split']!=plan['split'] or self.rows[q]['split']!=plan['split']:raise ValueError('episode split changed')
        source=self.labels(s);query=self.labels(q);target=plan['target']
        if plan['kind'].endswith('absent') and ((query==target)&(query!=255)).any():raise ValueError('absent query contains target')
        for sign,mask in [('positive',source==target),('negative',(source!=target)&(source!=255))]:
            points=np.asarray(plan[sign+'_xy']);positions=np.floor(points*np.array(source.shape[::-1])).astype(int)
            if not mask[positions[:,1],positions[:,0]].all():raise ValueError('incorrect simulated click sign')
        def move(inputs):return {k:v.to(device) for k,v in inputs.items()}
        prompts={}
        for sign in ['positive','negative']:
            prompts[sign+'_xy']=torch.tensor([plan[sign+'_xy']],device=device,dtype=torch.float32)
            prompts[sign+'_mask']=torch.ones((1,len(plan[sign+'_xy'])),device=device,dtype=torch.bool)
        prompts['prompt_size_id']=torch.ones(1,device=device,dtype=torch.long)
        supervision=dict(truth=torch.tensor(query==target,device=device)[None],valid=torch.tensor(query!=255,device=device)[None],
            support_truth=torch.tensor(source==target,device=device)[None],support_valid=torch.tensor(source!=255,device=device)[None],
            mpp=json.loads(self.rows[q]['input_mpp_xy_json']),support_mpp=json.loads(self.rows[s]['input_mpp_xy_json']))
        local=plan['support_id']==plan['query_id'];shared=local and plan['support_quality']==plan['query_quality']
        return dict(support=move(self.inputs(s,plan['support_quality'])),query=move(self.inputs(q,plan['query_quality'])),prompts=prompts,local=local,shared=shared),supervision
