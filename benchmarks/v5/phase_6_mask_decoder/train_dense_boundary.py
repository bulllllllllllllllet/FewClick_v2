"""Unique-patch paired dense boundary pilot; encoder cached once, no query GT input."""
import argparse
import functools
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .boundary_data import load_baseline, make_loader, move_batch
from .dense_prompt_decoder import DensePromptDecoder, prompt_maps
from .probe_boundary_refinement import metrics, summarize, prompt_violations
from .boundary_refinement import dilate
from benchmarks.v4.phase_6_mask_decoder.src.evaluation import binary_boundary
from benchmarks.v4.phase_6_mask_decoder.src.joint_losses import balanced_pixel_bce, soft_pixel_dice


def select_indices(dataset, count, seed, reference=None):
    rng=np.random.default_rng(seed); ep=dataset.episodes
    frame=pd.DataFrame({'index':np.arange(len(ep)),'patch':ep.patch_ids,'target':ep.target_classes})
    selected=[];used=set()
    if reference:
        for row in reference:
            index=row['episode_index']
            if str(ep.patch_ids[index])!=row['patch_id']: raise RuntimeError('regression episode identity mismatch')
            selected.append(index);used.add(str(ep.patch_ids[index]))
    buckets={c:rng.permutation(frame[frame.target==c]['index']) for c in sorted(frame.target.unique())}
    cursor={c:0 for c in buckets}
    while len(selected)<count:
        progress=False
        for c,indices in buckets.items():
            while cursor[c]<len(indices):
                index=int(indices[cursor[c]]);cursor[c]+=1
                patch=str(ep.patch_ids[index])
                if patch not in used:
                    selected.append(index);used.add(patch);progress=True;break
            if len(selected)==count: break
        if not progress: raise ValueError('not enough unique eligible patches')
    return selected


def points(probability, sample):
    total=wrong=0
    for name,sign in [('positive',True),('negative',False)]:
        xy=sample[name+'_xy'][0][sample[name+'_mask'][0]]
        x=(xy[:,0].clamp(0,1)*512).long().clamp(max=511)
        y=(xy[:,1].clamp(0,1)*512).long().clamp(max=511)
        credible=sample['valid'][0,y,x] & (sample['truth'][0,y,x]==sign)
        total+=int(credible.sum());wrong+=int((((probability[0,y,x]>=.5)!=sign)&credible).sum())
    return total,wrong


def evaluate(head, refs, device, output=None):
    rows=[]; predictions=[]
    with torch.no_grad():
        for ref in refs:
            sample=torch.load(ref['file'],map_location=device,weights_only=False)
            p=head(sample)['pixel_probability'] if head is not None else sample['base'].float()
            row={**ref['identity'],**metrics(p,sample['truth'],sample['valid'])}
            row['prompt_violations']=prompt_violations(p,sample)
            row['credible_prompts'],row['credible_violations']=points(p,sample)
            rows.append(row)
            if output is not None and ref.get('regression'): predictions.append((ref['identity']['patch_id'],p.cpu().numpy()[0]))
    summary=summarize(rows)
    summary['credible_prompts']=sum(r['credible_prompts'] for r in rows)
    summary['credible_violations']=sum(r['credible_violations'] for r in rows)
    if output is not None:
        output.write_text(json.dumps(rows))
        if predictions: np.savez_compressed(output.with_suffix('.npz'),probability=np.stack([p for _,p in predictions]),patch_id=[k for k,_ in predictions])
    return summary,rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-samples',type=int,default=3000)
    parser.add_argument('--val-samples',type=int,default=1024)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20261014)
    args=parser.parse_args()
    if Path('/nfs-medical3/zyh/fewclick_v2') not in args.output.resolve().parents: raise ValueError('artifact root')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'cache').mkdir()
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    device=torch.device('cuda:0')
    baseline_root=Path('/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142455')
    metadata=json.loads((baseline_root/'run_metadata.json').read_text())
    args.spatial_data=Path('/nfs-medical3/zyh/fewclick_v2/extensible_prompt/tissue_spatial_split_0922_001235')
    args.routing=Path('/nfs-medical3/zyh/fewclick_v2/iterative_decoder/spatial_routing_0923_154545')
    regression=json.loads(Path('/nfs-medical3/zyh/fewclick_v2/boundary_refinement/development_1014_172905/run.json').read_text())['manifest']['val']
    regression=regression if args.val_samples>=64 else regression[:args.val_samples]
    model=load_baseline(metadata,baseline_root/'checkpoint_epoch_004.pth',device)
    captured={}
    hooks=[model.phase2.embedding.register_forward_hook(lambda m,i,o:captured.update(deep=o)),
           model.phase2.backbone.layer1.register_forward_hook(lambda m,i,o:captured.update(shallow=o))]
    refs={};started=time.time();image_bytes=0
    for split,count in [('train',args.train_samples),('val',args.val_samples)]:
        dataset=make_loader(args,metadata,split).dataset.dataset
        indices=select_indices(dataset,count,args.seed+(split=='val'),regression if split=='val' else None)
        loader=DataLoader(Subset(dataset,indices),batch_size=1,num_workers=0,collate_fn=make_loader(args,metadata,split).collate_fn)
        refs[split]=[]
        for i,batch in enumerate(loader):
            batch=move_batch(batch,device)
            with torch.no_grad():
                out=model(batch);base=out['pixel_probability'].float()
                truth=batch['pixel_gt']==batch['target_class'][:,None,None];valid=batch['pixel_gt']!=255
                if not (truth & valid).any() or not (~truth & valid).any(): raise RuntimeError('invalid binary GT episode')
                shallow=captured['shallow'];shallow=shallow.reshape(1,32,8,*shallow.shape[-2:]).mean(2)
                sample={'deep':captured['deep'].cpu().half(),'shallow':shallow.cpu().half(),
                        'semantic':F.interpolate(out['semantic_logits'],size=(64,64),mode='bilinear',align_corners=False).cpu().half(),
                        'task':out['task_token'].cpu().float(),'base':base.cpu(),
                        'prompt_maps':prompt_maps(batch).cpu().half(),'truth':truth.cpu(),'valid':valid.cpu()}
                sample.update({k:batch[k].cpu() for k in ['positive_xy','negative_xy','positive_mask','negative_mask']})
                if not all(torch.isfinite(v).all() for v in sample.values()): raise FloatingPointError('cache')
            file=args.output/'cache'/f'{split}_{i:05d}.pth';torch.save(sample,file);image_bytes+=file.stat().st_size
            ref={'file':str(file),'identity':{'patch_id':str(batch['patch_id'][0]),'target_class':int(batch['target_class'][0]),'episode_index':int(batch['episode_index'][0])},'regression':split=='val' and i<len(regression)}
            refs[split].append(ref)
            with (args.output/'cache_progress.jsonl').open('a') as f:f.write(json.dumps({'split':split,'completed':i+1,'ref':ref})+'\n')
            if i%50==0: print('CACHE',split,i+1,'seconds',round(time.time()-started,1),'pid',os.getpid(),flush=True)
    for hook in hooks:hook.remove()
    del model;torch.cuda.empty_cache()
    if set(r['identity']['patch_id'] for r in refs['train']) & set(r['identity']['patch_id'] for r in refs['val']):raise RuntimeError('train/val overlap')
    run={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'refs':refs,'worker_pid':os.getpid(),
         'checkpoint':str(baseline_root/'checkpoint_epoch_004.pth'),'test_used':False,'scope':'development spatial validation, historical initialization exposure',
         'cache_seconds':time.time()-started,'cache_bytes':image_bytes,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('dense_prompt_decoder.py'),Path(__file__).with_name('boundary_data.py')]}}
    (args.output/'run.json').write_text(json.dumps(run,indent=2))
    baseline,base_rows=evaluate(None,refs['val'],device,args.output/'baseline_rows.json')
    result={'baseline':baseline,'arms':{}};print('BASELINE',json.dumps(baseline),flush=True)
    for name,high in [('dense64',False),('dense128',True)]:
        torch.manual_seed(args.seed);head=DensePromptDecoder(high).to(device)
        real=torch.load(refs['val'][0]['file'],map_location=device,weights_only=False)
        with torch.no_grad(): error=(head(real)['pixel_probability']-real['base']).abs().max().item()
        if error>2e-6:raise RuntimeError('zero initialization')
        optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=1e-4);history=[]
        for epoch in range(args.epochs):
            head.train();losses=[];start=time.time()
            for index in np.random.default_rng(args.seed+epoch).permutation(len(refs['train'])):
                s=torch.load(refs['train'][index]['file'],map_location=device,weights_only=False)
                out=head(s);p=out['pixel_probability'];logits=out['pixel_logits'];truth=s['truth'];valid=s['valid']
                loss=balanced_pixel_bce(logits,truth,valid)+soft_pixel_dice(p,truth,valid)
                band=dilate(binary_boundary(truth,valid),4)&valid&~dilate(~valid,4)
                if band.any():loss=loss+.5*F.binary_cross_entropy_with_logits(logits[band],truth[band].float())
                if not torch.isfinite(loss):raise FloatingPointError('loss')
                optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5)
                if not torch.isfinite(norm):raise FloatingPointError('gradient')
                optimizer.step();losses.append(float(loss.detach()))
            head.eval();summary,rows=evaluate(head,refs['val'],device,args.output/f'{name}_epoch_{epoch:03}_rows.json')
            summary.update(epoch=epoch,train_loss=float(np.mean(losses)),seconds=time.time()-start)
            summary['eligible']=summary['bf1_2']>=baseline['bf1_2']+.05 and summary['dice']>=baseline['dice']-.005 and summary['micro_dice']>=baseline['micro_dice']-.005 and summary['credible_violations']<=baseline['credible_violations']
            history.append(summary);torch.save({'model':head.state_dict(),'high_resolution':high,'epoch':epoch,'summary':summary,'run':run},args.output/f'{name}_epoch_{epoch:03}.pth')
            result['arms'][name]={'history':history};(args.output/'summary.json').write_text(json.dumps(result,indent=2))
            (args.output/'status.json').write_text(json.dumps({'status':'training','arm':name,'epoch':epoch,'worker_pid':os.getpid()}))
            print('EPOCH',name,json.dumps(summary),flush=True)
        loaded=DensePromptDecoder(high).to(device);loaded.load_state_dict(torch.load(args.output/f'{name}_epoch_{args.epochs-1:03}.pth',map_location=device,weights_only=False)['model'],strict=True)
        with torch.no_grad(): reload_error=(loaded(real)['pixel_probability']-head(real)['pixel_probability']).abs().max().item()
        if reload_error!=0:raise RuntimeError('checkpoint round trip')
        result['arms'][name]['reload_max_error']=reload_error
    (args.output/'summary.json').write_text(json.dumps(result,indent=2));(args.output/'status.json').write_text(json.dumps({'status':'complete','worker_pid':os.getpid()}))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
