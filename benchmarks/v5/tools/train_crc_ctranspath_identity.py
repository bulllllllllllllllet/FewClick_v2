#!/usr/bin/env python3
"""Train a five-class CRC identity MLP directly on frozen CTransPath-768 crops."""
from __future__ import annotations

import argparse, json, random, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES=["CD4_T","CD8_T","Treg","B_cell","CD163_Macrophage"]

class Head(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,256),nn.GELU(),nn.Dropout(.15),nn.Linear(256,5))
    def forward(self,x): return self.net(x)

def load_split(cache, manifest):
    spec=json.loads(manifest.read_text()); by_split={"train":[],"val":[]}
    for row in spec["samples"]:
        if row["split"] not in by_split: continue
        name=row["sample_id"].removesuffix("_identity")+".npz"; path=cache/name
        if not path.is_file(): raise FileNotFoundError(path)
        with np.load(path,allow_pickle=False) as x:
            valid=x["identity_verified"].astype(bool); feat=x["ctranspath_features"]; label=x["candidate_class"].astype(np.int64)
        if feat.shape!=(len(valid),768) or label.shape != valid.shape or not np.isfinite(feat).all(): raise ValueError(f"invalid cache {path.name}")
        if ((label[valid] < 0) | (label[valid] >= len(CLASSES))).any(): raise ValueError(f"invalid verified class index {path.name}")
        # A patch may contain no doctor-verified cells from the five-class ontology.
        # It is valid input, but contributes no supervised cell to this identity task.
        if valid.any(): by_split[row["split"]].append((feat[valid].astype(np.float32),label[valid],np.full(valid.sum(),row["case_id"])))
    result={}
    for split,values in by_split.items():
        if not values: raise ValueError(f"no doctor-verified five-class cells in {split}")
        x,y,cases=map(np.concatenate,zip(*values)); result[split]=(torch.from_numpy(x),torch.from_numpy(y),cases)
    return result

def macro_f1(logits, labels):
    pred=logits.argmax(1); scores=[]; per=[]
    for c in range(5):
        tp=int(((pred==c)&(labels==c)).sum()); fp=int(((pred==c)&(labels!=c)).sum()); fn=int(((pred!=c)&(labels==c)).sum())
        f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.; scores.append(f); per.append({"class":CLASSES[c],"f1":f,"support":int((labels==c).sum())})
    return float(np.mean(scores)),per

def patient_macro_f1(logits, labels, cases):
    patient={}
    for case in sorted(set(cases.tolist())):
        keep=torch.from_numpy(cases == case)
        score, per=macro_f1(logits[keep],labels[keep])
        patient[case]={"macro_f1":score,"per_class":per,"cells":int(keep.sum())}
    return float(np.mean([item["macro_f1"] for item in patient.values()])),patient

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--cache",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--timestamp",required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda:1"); p.add_argument("--epochs",type=int,default=15); p.add_argument("--batch-size",type=int,default=512); p.add_argument("--threads",type=int,default=16); p.add_argument("--seed",type=int,default=20260916); p.add_argument("--drw-start",type=int,default=3); p.add_argument("--alpha",type=float,default=.5); a=p.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}",a.timestamp) or min(a.epochs,a.batch_size,a.threads)<1 or not 0<a.alpha<=1: raise ValueError("invalid budget")
    torch.set_num_threads(a.threads); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); a.output.mkdir(parents=True,exist_ok=False)
    data=load_split(a.cache,a.manifest); train_x,train_y,_=data["train"]; val_x,val_y,val_cases=data["val"]; counts=torch.bincount(train_y,minlength=5).float(); weights=(counts.pow(-a.alpha)/counts.pow(-a.alpha).mean()).float()
    device=torch.device(a.device); model=Head().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); best=-1.; history=[]
    for epoch in range(a.epochs):
        model.train(); order=torch.randperm(len(train_y)); loss_sum=0.; steps=0
        active=weights.to(device) if epoch>=a.drw_start else None
        for ids in order.split(a.batch_size):
            opt.zero_grad(set_to_none=True); loss=F.cross_entropy(model(train_x[ids].to(device)),train_y[ids].to(device),weight=active); loss.backward(); opt.step(); loss_sum+=float(loss.detach()); steps+=1
        model.eval()
        with torch.no_grad(): logits=torch.cat([model(v.to(device)).cpu() for v in val_x.split(a.batch_size)])
        global_score,per=macro_f1(logits,val_y); score,patients=patient_macro_f1(logits,val_y,val_cases); row={"epoch":epoch,"loss":loss_sum/steps,"patient_macro_f1":score,"global_macro_f1":global_score,"drw":epoch>=a.drw_start,"per_class":per,"patients":patients}; history.append(row)
        if score>best: best=score; torch.save({"model":model.state_dict(),"epoch":epoch,"classes":CLASSES,"manifest":str(a.manifest)},a.output/"best.pth")
        print(json.dumps(row),flush=True)
    (a.output/"metrics.json").write_text(json.dumps({"status":"complete","best_patient_macro_f1":best,"train_counts":counts.tolist(),"weights":weights.tolist(),"history":history,"config":vars(a),"scope":"pooled internal patch classification; not patient-independent"},default=str,indent=2))
if __name__=="__main__": main()
