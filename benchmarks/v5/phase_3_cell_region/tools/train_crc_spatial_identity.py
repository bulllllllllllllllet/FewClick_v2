#!/usr/bin/env python3
"""Train/evaluate the explicit spatial-context CRC identity probe."""
from __future__ import annotations
import argparse,json,random,re,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from benchmarks.v5.phase_3_cell_region.src.semantic_contract import SemanticDataset,sha256_file
from benchmarks.v5.phase_3_cell_region.src.semantic_losses import partial_identity_loss
from benchmarks.v5.phase_3_cell_region.src.semantic_training import summarize_confusion
from benchmarks.v5.phase_3_cell_region.src.spatial_identity import SpatialContextIdentity

@torch.no_grad()
def evaluate(model,data,device):
    model.eval(); cases={}; classes=len(data.spec['identity_classes'])
    for item in data:
        x={k:v.to(device) for k,v in item['inputs'].items()}; y={k:v.to(device) for k,v in item['targets'].items()}
        out=model(x['region_tokens'],x['cells'],x['cell_valid'],x['assignment_at_cells']); keep=y['identity_valid']&x['cell_valid']&(y['identity_allowed'].sum(-1)==1)
        truth=y['identity_allowed'].long().argmax(-1)[keep]; prob=out['identity_probabilities'][keep]; pred=prob.argmax(-1); rec=cases.setdefault(item['case_id'],{'matrix':torch.zeros(classes,classes,dtype=torch.long),'brier':0.,'count':0})
        rec['matrix']+=torch.bincount((truth*classes+pred).cpu(),minlength=classes**2).reshape(classes,classes);rec['brier']+=float(((prob-torch.nn.functional.one_hot(truth,classes))**2).sum());rec['count']+=len(truth)
    result={};
    for case,rec in cases.items(): result[case]={**summarize_confusion(rec['matrix']),'brier':rec['brier']/rec['count']}
    return {'cases':result,'patient_macro_f1':sum(x['macro_f1'] for x in result.values())/len(result),'scope':'pooled internal patch classification; not patient-independent or final click segmentation'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--timestamp',required=True);p.add_argument('--output-root',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3'));p.add_argument('--device',default='cuda:1');p.add_argument('--epochs',type=int,default=60);p.add_argument('--patience',type=int,default=12);p.add_argument('--neighbors',type=int,default=12);p.add_argument('--seed',type=int,default=20260915);a=p.parse_args()
    if not re.fullmatch(r'\d{4}_\d{6}',a.timestamp):raise ValueError('timestamp must be MMDD_HHMMSS')
    if not a.output_root.resolve().is_relative_to(Path('/nfs-medical3/zyh/fewclick_v2')):raise ValueError('FewClick output only')
    torch.manual_seed(a.seed);rng=random.Random(a.seed);train,val=SemanticDataset(a.manifest,'train'),SemanticDataset(a.manifest,'val');device=torch.device(a.device)
    for data in (train,val):
        for _ in data:pass
    model=SpatialContextIdentity(len(train.spec['identity_classes']),neighbors=a.neighbors).to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);out=a.output_root/f'spatial_identity_{a.timestamp}';out.mkdir(parents=True,exist_ok=False)
    meta={'schema':'crc_spatial_identity_v1','manifest_sha256':train.manifest_hash,'identity_classes':train.spec['identity_classes'],'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'input':'H&E-only reg64, xy, log_area, frozen phase2 region token, xy-KNN; no labels or marker values'};(out/'metadata.json').write_text(json.dumps(meta,indent=2));best=-1;stale=0
    for epoch in range(a.epochs):
        model.train();loss_sum=0.;steps=0
        for _ in range(len(train)):
            item=train[rng.randrange(len(train))];x={k:v.to(device) for k,v in item['inputs'].items()};y={k:v.to(device) for k,v in item['targets'].items()};opt.zero_grad();pred=model(x['region_tokens'],x['cells'],x['cell_valid'],x['assignment_at_cells']);loss,count=partial_identity_loss(pred['identity_logits'],y['identity_allowed'],y['identity_valid']&x['cell_valid'])
            if not count or not torch.isfinite(loss):raise RuntimeError('invalid identity supervision/loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();loss_sum+=float(loss.detach());steps+=1
        metrics=evaluate(model,val,device);row={'epoch':epoch,'loss':loss_sum/steps,**metrics};(out/'metrics.jsonl').open('a').write(json.dumps(row)+'\n');score=metrics['patient_macro_f1'];print(json.dumps({'epoch':epoch,'loss':row['loss'],'patient_macro_f1':score}),flush=True)
        if score>best:best=score;stale=0;torch.save({**meta,'epoch':epoch,'model':model.state_dict()},out/f'best_epoch_{epoch:03d}.pth')
        else:stale+=1
        if stale>=a.patience:break
    (out/'complete.json').write_text(json.dumps({'status':'complete','best_internal_macro_f1':best,'patient_independent':False},indent=2))
if __name__=='__main__':main()
