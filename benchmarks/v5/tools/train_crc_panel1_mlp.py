#!/usr/bin/env python3
"""Train lightweight MLP probes on frozen Panel1 CTransPath features."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import StandardScaler
from benchmarks.v5.tools.run_crc_panel1_cell_identity import CLASSES, score, write_json

class LocalMLP(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.net=torch.nn.Sequential(torch.nn.Linear(768,256),torch.nn.LayerNorm(256),torch.nn.GELU(),torch.nn.Dropout(.25),torch.nn.Linear(256,5))
    def forward(self,x): return self.net(x)

class GatedMLP(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.local=torch.nn.Sequential(torch.nn.Linear(768,256),torch.nn.LayerNorm(256),torch.nn.GELU()); self.ctx=torch.nn.Sequential(torch.nn.Linear(768,128),torch.nn.LayerNorm(128),torch.nn.GELU()); self.head=torch.nn.Sequential(torch.nn.Linear(256+128+128,256),torch.nn.GELU(),torch.nn.Dropout(.25),torch.nn.Linear(256,5))
    def forward(self,x):
        a=self.local(x[:,:768]); b=self.ctx(x[:,768:]); return self.head(torch.cat([a,b,a[:,:128]*b],1))

def load_features(root, kind):
    with np.load(root/'features.npz',allow_pickle=False) as z: return z['features'] if kind=='128' else z['features']

def train_one(model, x, y, ids, seed, epochs):
    torch.manual_seed(seed); np.random.seed(seed); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-2)
    counts=torch.bincount(torch.as_tensor(y[ids['train']]),minlength=5).float(); weights=counts.rsqrt(); weights/=weights.mean(); lossfn=torch.nn.CrossEntropyLoss(weight=weights)
    best=(-1,None,None)
    for ep in range(epochs):
        model.train(); order=torch.randperm(len(ids['train']))
        for batch in order.split(256):
            ii=torch.as_tensor(ids['train'])[batch]; opt.zero_grad(); loss=lossfn(model(x[ii]),torch.as_tensor(y[ii])); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): pred=model(x[torch.as_tensor(ids['val'])]).argmax(1).numpy()
        metric=score(y[ids['val']],pred)
        if metric['macro_f1']>best[0]: best=(metric['macro_f1'],{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},ep)
    model.load_state_dict(best[1]); return model,best[0],best[2]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--features128',type=Path,required=True); p.add_argument('--features256',type=Path,required=True); p.add_argument('--supplement',type=Path,required=True); p.add_argument('--supplement-features',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--epochs',type=int,default=80); p.add_argument('--seeds',type=int,nargs='+',default=[11,23,47]); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8)
    data=pd.read_csv(a.manifest/'cells.csv'); meta=json.loads((a.manifest/'manifest.json').read_text());
    if hashlib.sha256((a.manifest/'cells.csv').read_bytes()).hexdigest()!=meta['manifest_sha256']: raise ValueError('manifest mismatch')
    raw128=load_features(a.features128,'128'); raw256=load_features(a.features256,'256')
    if raw128.shape!=(len(data),768) or raw256.shape!=raw128.shape: raise ValueError('feature shape mismatch')
    ids={s:np.flatnonzero(data.split.eq(s)) for s in ('train','val','test')}; y=data.label.to_numpy(int)
    s128,s256=StandardScaler(),StandardScaler(); s128.fit(raw128[ids['train']]); s256.fit(raw256[ids['train']]); z128=s128.transform(raw128); z256=s256.transform(raw256)
    arms={'128_mlp':(z128,LocalMLP),'gated_128_256_mlp':(np.concatenate([z128,z256],1),GatedMLP)}; results={}
    for name,(arr,ctor) in arms.items():
        runs=[]
        for seed in a.seeds:
            model,best,ep=train_one(ctor(),torch.tensor(arr,dtype=torch.float32),y,ids,seed,a.epochs); model.eval()
            with torch.no_grad(): pred=model(torch.tensor(arr,dtype=torch.float32)).argmax(1).numpy()
            runs.append({'seed':seed,'best_epoch':ep,'val':score(y[ids['val']],pred[ids['val']]),'test':score(y[ids['test']],pred[ids['test']])})
            torch.save({'model':model.state_dict(),'arm':name,'manifest_sha256':meta['manifest_sha256'],'seed':seed},a.output/f'{name}_seed{seed}.pth')
        results[name]={'runs':runs,'selected_seed':max(runs,key=lambda r:r['val']['macro_f1'])['seed']}
    sup=pd.read_csv(a.supplement/'cells.csv');
    with np.load(a.supplement_features/'features.npz',allow_pickle=False) as z: sup128,sup256=z['features128'],z['features256']
    zs128=s128.transform(sup128); zs256=s256.transform(sup256); sy=sup.label.to_numpy(int)
    for name,(arr,ctor) in arms.items():
        seed=results[name]['selected_seed']; state=torch.load(a.output/f'{name}_seed{seed}.pth',map_location='cpu')['model']; model=ctor(); model.load_state_dict(state); model.eval(); sx= np.concatenate([zs128,zs256],1) if name.startswith('gated') else zs128
        with torch.no_grad(): pred=model(torch.tensor(sx,dtype=torch.float32)).argmax(1).numpy()
        results[name]['supplement']={s:score(sy[sup.split.eq(s)],pred[sup.split.eq(s)]) for s in ('val','test')}
    write_json(a.output/'metrics.json',{'status':'complete','classes':CLASSES,'manifest_sha256':meta['manifest_sha256'],'classifier':'train-only StandardScaler + class-balanced 2-layer MLP','arms':results})
    print(json.dumps({k:{'val':max(v['runs'],key=lambda r:r['val']['macro_f1'])['val']['accuracy'],'supp_test':v['supplement']['test']['accuracy']} for k,v in results.items()}),flush=True)
if __name__=='__main__': main()
