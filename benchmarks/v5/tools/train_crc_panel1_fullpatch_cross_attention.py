#!/usr/bin/env python3
"""Compare target 32 px, 512 px patch, and all-nucleus cross-attention."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.v5.tools.run_crc_panel1_cell_identity import CLASSES, score

BASE = Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3')
SMALL = BASE/'panel1_small_cell_crop_full_1008_143551'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def unit(value):
    value = value.astype(np.float32)
    return value/np.maximum(np.linalg.norm(value,axis=-1,keepdims=True),1e-8)


def prepare(shards, device, max_patches=None):
    rows = pd.read_csv(SMALL/'cells.csv')
    with np.load(SMALL/'features.npz',allow_pickle=False) as cache:
        target = cache['features_32'].copy()
    if target.shape != (len(rows),768) or rows[['patch','he_id']].duplicated().any():
        raise ValueError('invalid target table or feature dimensions')
    if rows.groupby('patch').arm_split.nunique().max() != 1:
        raise ValueError('a patch spans splits')
    expected = sorted(rows.patch.unique())
    if max_patches is not None:
        expected = expected[:max_patches]
        keep = rows.patch.isin(expected).to_numpy()
        rows = rows.loc[keep].reset_index(drop=True)
        target = target[keep]
    records = {}
    for directory in shards:
        audit = json.loads((directory/'audit.json').read_text())
        if audit['status'] != 'complete' or audit['source_small_cells_sha256'] != digest(SMALL/'cells.csv'):
            raise ValueError(f'bad shard audit: {directory}')
        for rec in audit['records']:
            if rec['patch'] in records:
                raise ValueError('duplicate patch across shards')
            records[rec['patch']] = directory/ f"{rec['patch']}.npz"
            if digest(records[rec['patch']]) != rec['sha256']:
                raise ValueError(f"cache hash mismatch: {rec['patch']}")
    if set(expected) - set(records):
        raise ValueError(f'missing {len(set(expected)-set(records))} patches')
    cell_blocks, patch_features = [], np.empty((len(rows),768),np.float32)
    # A dense index table is small; the feature bank remains one contiguous tensor.
    per_row = [None]*len(rows)
    max_peers = 0
    for patch, group in rows.groupby('patch',sort=False):
        with np.load(records[patch],allow_pickle=False) as z:
            ids, xy, features = z['he_ids'].copy(), z['he_xy'].copy(), z['features32'].copy()
            global_feature = z['patch512'].copy()
        if len(ids) == 0 or len(np.unique(ids)) != len(ids) or features.shape != (len(ids),768):
            raise ValueError(f'invalid cache: {patch}')
        if not np.isfinite(features).all() or not np.isfinite(global_feature).all():
            raise ValueError(f'nonfinite cache: {patch}')
        start = sum(len(block) for block in cell_blocks)
        cell_blocks.append(unit(features))
        lookup = {int(v):i for i,v in enumerate(ids)}
        ids_all = np.arange(start,start+len(ids),dtype=np.int32)
        for row in group.itertuples():
            if int(row.he_id) not in lookup:
                raise ValueError(f'missing target: {patch}/{row.he_id}')
            local = lookup[int(row.he_id)]
            if np.linalg.norm(xy[local]-np.array([row.he_x,row.he_y])) > .1:
                raise ValueError(f'coordinate mismatch: {patch}/{row.he_id}')
            if np.dot(unit(features[local]),unit(target[row.Index])) < .999:
                raise ValueError(f'target feature mismatch: {patch}/{row.he_id}')
            peers = np.delete(ids_all,local)
            offsets = np.delete((xy-xy[local])/512.,local,axis=0)
            per_row[row.Index] = (peers,offsets)
            max_peers = max(max_peers,len(peers))
            patch_features[row.Index] = unit(global_feature)
    bank = np.concatenate(cell_blocks)
    neighbor = np.zeros((len(rows),max_peers),np.int32)
    valid = np.zeros((len(rows),max_peers),bool)
    offset = np.zeros((len(rows),max_peers,2),np.float32)
    counts = np.empty(len(rows),np.int32)
    for i,(ids,xy) in enumerate(per_row):
        n=len(ids); neighbor[i,:n]=ids; valid[i,:n]=True; offset[i,:n]=xy; counts[i]=n
    if not np.isfinite(target).all() or not np.isfinite(bank).all():
        raise ValueError('nonfinite input')
    data = tuple(torch.as_tensor(x,device=device) for x in
                 (unit(target),patch_features,bank,neighbor,valid,offset))
    return rows,data,counts


class Model(nn.Module):
    def __init__(self, arm, hidden=128):
        super().__init__()
        self.arm = arm
        self.target = nn.Sequential(nn.Linear(768,hidden),nn.LayerNorm(hidden),nn.GELU())
        if arm != 'target32':
            self.patch = nn.Sequential(nn.Linear(768,hidden),nn.LayerNorm(hidden),nn.GELU())
        if arm == 'cross_attention':
            self.cell = nn.Sequential(nn.Linear(768,hidden),nn.LayerNorm(hidden),nn.GELU())
            self.position = nn.Sequential(nn.Linear(2,32),nn.GELU(),nn.Linear(32,hidden))
            self.attention = nn.MultiheadAttention(hidden,4,dropout=.1,batch_first=True)
            self.norm = nn.LayerNorm(hidden)
            self.gate = nn.Linear(hidden*2,hidden)
        width = hidden*(1+(arm!='target32')+(arm=='cross_attention'))
        self.head = nn.Sequential(nn.Linear(width,hidden),nn.LayerNorm(hidden),nn.GELU(),
                                  nn.Dropout(.25),nn.Linear(hidden,len(CLASSES)))

    def forward(self,data,index,context=True):
        target,patch,bank,neighbors,valid,offset = data
        x = self.target(target[index]); tokens = [x]
        if self.arm != 'target32':
            tokens.append(self.patch(patch[index]))
        if self.arm == 'cross_attention':
            ok = valid[index]
            safe = ok.clone()
            empty = ~ok.any(1)
            safe[empty,0] = True
            peers = self.cell(bank[neighbors[index]])
            attended,_ = self.attention(x[:,None],peers+self.position(offset[index]),peers,
                                        key_padding_mask=~safe,need_weights=False)
            attended = self.norm(attended[:,0])
            attended = torch.sigmoid(self.gate(torch.cat([x,attended],1)))*attended
            tokens.append(torch.where(empty[:,None],torch.zeros_like(attended),attended)
                          if context else torch.zeros_like(attended))
        return self.head(torch.cat(tokens,1))


def predict(model,data,index,batch,context=True):
    model.eval(); chunks=[]
    with torch.inference_mode():
        for ids in index.split(batch):
            chunks.append(model(data,ids,context).argmax(1).cpu().numpy())
    return np.concatenate(chunks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shards',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--seeds',default='11,23,47')
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--max-patches',type=int)
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(BASE) or min(a.epochs,a.batch_size,a.patience)<1:
        p.error('invalid output or training budget')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    device=torch.device(a.device)
    rows,data,counts=prepare(a.shards,device,a.max_patches)
    labels=torch.tensor(rows.label.to_numpy(int),device=device)
    split=rows.arm_split.to_numpy(str)
    indices={s:torch.as_tensor(np.flatnonzero(split==s),device=device) for s in
             ('train','val','test','supplement_val','supplement_test')}
    if any(len(ids)==0 for ids in indices.values()):
        raise ValueError('empty evaluation split')
    if set(rows.loc[rows.arm_split.eq('train'),'group']) & set(rows.loc[rows.arm_split.ne('train'),'group']):
        raise ValueError('spatial group leak')
    class_count=torch.bincount(labels[indices['train']],minlength=len(CLASSES)).float()
    if (class_count==0).any():
        raise ValueError('missing training class')
    weights=class_count.rsqrt(); weights/=weights.mean()
    report={'status':'running','classes':CLASSES,'rows':len(rows),'patches':rows.patch.nunique(),
            'cells_in_bank':len(data[2]),'peer_quantiles':np.quantile(counts,[0,.25,.5,.75,1]).tolist(),
            'split_counts':{s:len(v) for s,v in indices.items()},'runs':[]}
    truth=rows.label.to_numpy(int)
    for seed in map(int,a.seeds.split(',')):
        for arm in ('target32','target32_patch512','cross_attention'):
            torch.manual_seed(seed); np.random.seed(seed)
            model=Model(arm).to(device)
            optim=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-2)
            best=(-1.,None,-1); stale=0
            for epoch in range(a.epochs):
                model.train()
                order=indices['train'][torch.randperm(len(indices['train']),device=device)]
                for part in order.split(a.batch_size):
                    optim.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(model(data,part),labels[part],weight=weights)
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite loss')
                    loss.backward(); optim.step()
                pred=predict(model,data,indices['val'],a.batch_size)
                value=score(truth[indices['val'].cpu().numpy()],pred)['macro_f1']
                if value>best[0]+1e-6:
                    best=(value,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch)
                    stale=0
                else:
                    stale+=1
                print(json.dumps({'seed':seed,'arm':arm,'epoch':epoch,'val_macro_f1':value,'best':best[0]}),flush=True)
                if stale>=a.patience: break
            model.load_state_dict(best[1])
            result={'seed':seed,'arm':arm,'best_epoch':best[2],'parameters':sum(x.numel() for x in model.parameters()),'metrics':{}}
            predictions=rows[['patch','he_id','arm_split','label']].copy()
            predictions['pred']=-1
            for name,index in indices.items():
                ids=index.cpu().numpy(); pred=predict(model,data,index,a.batch_size)
                predictions.loc[ids,'pred']=pred
                result['metrics'][name]=score(truth[ids],pred)
            if arm=='cross_attention':
                result['no_context_metrics']={name:score(truth[index.cpu().numpy()],predict(model,data,index,a.batch_size,False))
                                              for name,index in indices.items() if name!='train'}
            torch.save({'state_dict':best[1],'arm':arm,'seed':seed,'classes':CLASSES,
                        'source_small_cells_sha256':digest(SMALL/'cells.csv')},a.output/f'{arm}_seed{seed}.pth')
            predictions.to_csv(a.output/f'{arm}_seed{seed}_predictions.csv',index=False)
            report['runs'].append(result)
            (a.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'event':'arm_complete','seed':seed,'arm':arm,
                              'test_macro_f1':result['metrics']['test']['macro_f1'],
                              'supplement_test_macro_f1':result['metrics']['supplement_test']['macro_f1']}),flush=True)
    report['status']='complete'
    (a.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'event':'complete','output':str(a.output)}),flush=True)


if __name__=='__main__':
    main()
