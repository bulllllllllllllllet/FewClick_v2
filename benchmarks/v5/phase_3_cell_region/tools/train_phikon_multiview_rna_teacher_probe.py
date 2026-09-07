#!/usr/bin/env python3
"""Train a dual-context (256px + 384px) H&E-only Phikon RNA-teacher probe."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_xenium_rna_pilot import load_transcripts, normalized_counts, split_gene_views, svd_features  # noqa: E402


class DualViewAdapter(torch.nn.Module):
    def __init__(self, input_dim: int = 2048):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, 512), torch.nn.GELU(), torch.nn.Dropout(.1), torch.nn.Linear(512, 256))
    def forward(self, x): return self.net(x)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration-dir", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/context_calibration/phikon_context_0907_123105"))
    p.add_argument("--data-root", type=Path, default=Path("/nfs-medical3/fyx/PKU_DATA/Xenium/COAD"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[20260907, 20260908, 20260909])
    p.add_argument("--epochs", type=int, default=250); p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--rna-weight", type=float, default=.25); p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def run_fold(xtr, ytr, rtr, xte, classes, seed, epochs, lr, rna_weight, shuffled, device):
    torch.manual_seed(seed); adapter = DualViewAdapter(xtr.shape[1]).to(device); head = torch.nn.Linear(256, len(classes)).to(device); rhead = torch.nn.Linear(256, rtr.shape[1]).to(device)
    lookup = {name:i for i,name in enumerate(classes)}; label = torch.tensor([lookup[v] for v in ytr], device=device)
    weight = (len(label)/(len(classes)*torch.bincount(label,minlength=len(classes)).float().clamp_min(1))).to(device)
    mean, std = xtr.mean(0,keepdims=True), xtr.std(0,keepdims=True).clip(1e-6); xtr,xte=(xtr-mean)/std,(xte-mean)/std
    rtr=(rtr-rtr.mean(0,keepdims=True))/rtr.std(0,keepdims=True).clip(1e-6)
    target=rtr[np.random.default_rng(seed).permutation(len(rtr))] if shuffled else rtr
    x, r = torch.tensor(xtr,device=device), torch.tensor(target,dtype=torch.float32,device=device); opt=torch.optim.AdamW([*adapter.parameters(),*head.parameters(),*rhead.parameters()],lr=lr,weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True); z=adapter(x); loss=torch.nn.functional.cross_entropy(head(z),label,weight=weight)+rna_weight*torch.nn.functional.mse_loss(rhead(z),r); loss.backward(); opt.step()
    with torch.inference_mode(): pred=head(adapter(torch.tensor(xte,device=device))).argmax(1).cpu().numpy()
    return np.array([classes[i] for i in pred])


def main() -> None:
    a=parse()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True); device=torch.device(a.device)
    cells=pd.read_csv(a.calibration_dir/'cells.csv',dtype={'cell_id':str}); feature=np.concatenate([np.load(a.calibration_dir/'phikon_1024_crop256.npy'),np.load(a.calibration_dir/'phikon_1024_crop384.npy')],axis=1).astype(np.float32)
    tx=load_transcripts(a.data_root/'transcripts/transcripts.parquet',set(cells.cell_id),1000,2000,1500,20.0); counts=normalized_counts(tx,cells.cell_id.tolist(),30); index=pd.Index(cells.cell_id).get_indexer(counts.index); cells,feature=cells.iloc[index].reset_index(drop=True),feature[index]
    genes,_=split_gene_views(counts.columns.tolist()); rna=svd_features(counts[genes].to_numpy()).astype(np.float32); labels=cells.annotation.to_numpy(); groups=cells.region.to_numpy(); classes=sorted(set(labels)); rows=[]
    for arm,weight,shuffled in [('label_only',0.,False),('paired_rna_teacher',a.rna_weight,False),('shuffled_rna_teacher',a.rna_weight,True)]:
        for seed in a.seeds:
            truth=[]; pred=[]
            for tr,te in GroupKFold(n_splits=4).split(feature,labels,groups):
                pred.extend(run_fold(feature[tr],labels[tr],rna[tr],feature[te],classes,seed,a.epochs,a.lr,weight,shuffled,device)); truth.extend(labels[te])
            rows.append({'arm':arm,'seed':seed,'macro_f1':f1_score(truth,pred,average='macro',zero_division=0),'balanced_accuracy':balanced_accuracy_score(truth,pred)})
    frame=pd.DataFrame(rows); frame.to_csv(a.output_dir/'seed_metrics.csv',index=False); grouped=frame.groupby('arm')[['macro_f1','balanced_accuracy']].agg(['mean','std'])
    summary={arm:{metric:{stat:float(grouped.loc[arm,(metric,stat)]) for stat in ('mean','std')} for metric in ('macro_f1','balanced_accuracy')} for arm in grouped.index}
    metric={'purpose':'dual-view 256px+384px H&E-only RNA-teacher probe','cells':len(cells),'feature_dim':int(feature.shape[1]),'seeds':a.seeds,'summary':summary,'acceptance':'paired RNA must exceed label-only and shuffled RNA'}
    (a.output_dir/'metrics.json').write_text(json.dumps(metric,indent=2),encoding='utf-8'); print(json.dumps({'event':'complete','output':str(a.output_dir),**metric},ensure_ascii=False))
if __name__=='__main__': main()
