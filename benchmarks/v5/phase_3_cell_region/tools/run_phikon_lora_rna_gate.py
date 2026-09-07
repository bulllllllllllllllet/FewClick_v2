#!/usr/bin/env python3
"""Small real-image LoRA gate: dual-view H&E, RNA teacher only during training."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

HERE=Path(__file__).resolve().parent; SRC=HERE.parent/'src'; sys.path[:0]=[str(HERE),str(SRC)]
from phikon_v2 import PhikonV2Encoder  # noqa
from calibrate_phikon_context_probe import crops_for_size  # noqa
from run_phikon_backbone_probe import annotated_cells  # noqa
from run_xenium_xcell_baseline import read_polygons  # noqa
from run_xenium_rna_pilot import load_transcripts, normalized_counts, split_gene_views, svd_features  # noqa
import pyvips

class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, rank: int=8, alpha: int=16):
        super().__init__(); self.base=base; self.scale=alpha/rank
        for p in base.parameters(): p.requires_grad_(False)
        self.a=torch.nn.Parameter(torch.empty(rank,base.in_features,dtype=base.weight.dtype)); self.b=torch.nn.Parameter(torch.zeros(base.out_features,rank,dtype=base.weight.dtype)); torch.nn.init.kaiming_uniform_(self.a,a=5**.5)
    def forward(self,x): return self.base(x)+F.linear(F.linear(x,self.a),self.b)*self.scale

class Heads(torch.nn.Module):
    def __init__(self,nclass:int,nrna:int):
        super().__init__(); self.adapter=torch.nn.Sequential(torch.nn.LayerNorm(2048),torch.nn.Linear(2048,512),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(512,256)); self.cls=torch.nn.Linear(256,nclass); self.rna=torch.nn.Linear(256,nrna)
    def forward(self,a,b):
        z=self.adapter(torch.cat((a,b),1).float()); return self.cls(z),self.rna(z)

def parse():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--data-root',type=Path,default=Path('/nfs-medical3/fyx/PKU_DATA/Xenium/COAD')); p.add_argument('--model-dir',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2')); p.add_argument('--xcell-artifact',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_xcell_baseline_0831_111806')); p.add_argument('--annotations',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_celltype_pilot_0831_121408/annotations.csv')); p.add_argument('--epochs',type=int,default=8); p.add_argument('--batch-size',type=int,default=12); p.add_argument('--lr',type=float,default=5e-5); p.add_argument('--lora-rank',type=int,default=8); p.add_argument('--device',default='cuda:0'); p.add_argument('--seed',type=int,default=20260907); return p.parse_args()

def inject(model,rank):
    for p in model.parameters(): p.requires_grad_(False)
    for block in model.backbone.encoder.layer[-2:]:
        att=block.attention.attention; att.query=LoRALinear(att.query,rank); att.value=LoRALinear(att.value,rank)

def main():
    a=parse();
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True); torch.manual_seed(a.seed); dev=torch.device(a.device)
    summary=pd.read_csv(a.xcell_artifact/'cell_summary.csv',dtype={'cell_id':str}); ann=pd.read_csv(a.annotations,dtype={'cell_id':str}); cells=annotated_cells(summary,ann,30)
    v=pyvips.Image.new_from_file(str(a.data_root/'HE.tif'),access='sequential').crop(1000,2000,1500,1500); image=np.ndarray(buffer=v.write_to_memory(),dtype=np.uint8,shape=(1500,1500,v.bands))[:,:,:3]; polygons=read_polygons(a.data_root/'segmentation_mask/cell_boundaries.csv',set(cells.cell_id),1000,2000)
    view256=crops_for_size(image,cells,polygons,256); view384=crops_for_size(image,cells,polygons,384)
    tx=load_transcripts(a.data_root/'transcripts/transcripts.parquet',set(cells.cell_id),1000,2000,1500,20.0); count=normalized_counts(tx,cells.cell_id.tolist(),30); order=pd.Index(cells.cell_id).get_indexer(count.index); cells=cells.iloc[order].reset_index(drop=True); view256=[view256[i] for i in order]; view384=[view384[i] for i in order]; genes,_=split_gene_views(count.columns.tolist()); rna=svd_features(count[genes].to_numpy()).astype(np.float32)
    labels=cells.annotation.to_numpy(); groups=cells.region.to_numpy(); classes=sorted(set(labels)); lookup={x:i for i,x in enumerate(classes)}; result=[]
    for arm,weight,shuffle in [('label_only',0.,False),('paired_rna_teacher',.25,False),('shuffled_rna_teacher',.25,True)]:
        truth=[]; pred=[]
        for tr,te in GroupKFold(n_splits=4).split(labels,labels,groups):
            enc=PhikonV2Encoder(a.model_dir,device=dev,dtype=torch.bfloat16); inject(enc,a.lora_rank); enc.to(dev); heads=Heads(len(classes),rna.shape[1]).to(dev); opt=torch.optim.AdamW([p for p in list(enc.parameters())+list(heads.parameters()) if p.requires_grad],lr=a.lr,weight_decay=1e-4)
            y=torch.tensor([lookup[x] for x in labels],device=dev); cw=(len(tr)/(len(classes)*torch.bincount(y[tr],minlength=len(classes)).float().clamp_min(1))).to(dev); target=rna[np.random.default_rng(a.seed).permutation(len(rna))] if shuffle else rna
            enc.backbone.train(); heads.train()
            for _ in range(a.epochs):
                for start in range(0,len(tr),a.batch_size):
                    ind=tr[start:start+a.batch_size]; x1=enc.preprocess([view256[i] for i in ind],dev); x2=enc.preprocess([view384[i] for i in ind],dev); f1=enc(x1); f2=enc(x2); logit,reg=heads(f1,f2); loss=F.cross_entropy(logit,y[ind],weight=cw)+weight*F.mse_loss(reg,torch.tensor(target[ind],device=dev)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            enc.backbone.eval(); heads.eval()
            with torch.inference_mode():
                out=[]
                for start in range(0,len(te),a.batch_size):
                    ind=te[start:start+a.batch_size]; logit,_=heads(enc(enc.preprocess([view256[i] for i in ind],dev)),enc(enc.preprocess([view384[i] for i in ind],dev))); out.extend(logit.argmax(1).cpu().numpy())
            pred.extend([classes[i] for i in out]); truth.extend(labels[te])
        result.append({'arm':arm,'macro_f1':f1_score(truth,pred,average='macro',zero_division=0),'balanced_accuracy':balanced_accuracy_score(truth,pred)})
    pd.DataFrame(result).to_csv(a.output_dir/'metrics.csv',index=False); metric={'purpose':'real-image last-two-block Q/V LoRA dual-view RNA teacher gate; H&E only at test','cells':len(cells),'epochs':a.epochs,'lora_rank':a.lora_rank,'results':result}; (a.output_dir/'metrics.json').write_text(json.dumps(metric,indent=2),encoding='utf-8'); print(json.dumps({'event':'complete','output':str(a.output_dir),**metric},ensure_ascii=False))
if __name__=='__main__': main()
