"""Fixed-manifest seed replay, nested expansion and matched layer4 fine-tuning."""
import argparse
import copy
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
from .train_dense_boundary import select_indices, evaluate
from .boundary_refinement import dilate
from benchmarks.v4.phase_6_mask_decoder.src.evaluation import binary_boundary
from benchmarks.v4.phase_6_mask_decoder.src.joint_losses import balanced_pixel_bce, soft_pixel_dice
from benchmarks.v4.phase_1_multiscale.src.data import read_he_patch
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor


class DenseContextModel(torch.nn.Module):
    def __init__(self, phase2=None):
        super().__init__()
        self.head=DensePromptDecoder(True)
        self.phase2=phase2
        if phase2 is not None:
            phase2.requires_grad_(False);phase2.backbone.layer4.requires_grad_(True);phase2.eval()

    def train(self, mode=True):
        self.training=mode;self.head.train(mode)
        if self.phase2 is not None:self.phase2.eval()  # Frozen BN statistics in both stages.
        return self

    def forward(self, sample):
        if self.phase2 is None:return self.head(sample)
        features=self.phase2.extract_backbone_features(sample['image'])
        deep=self.phase2.embedding(features)
        semantic=F.interpolate(self.phase2.semantic(features),size=sample['base'].shape[-2:],mode='bilinear',align_corners=False)
        semantic=F.interpolate(semantic,size=(64,64),mode='bilinear',align_corners=False)
        # Match the original FP16 feature-cache representation; casts retain gradients.
        updated={**sample,'deep':deep.half().float(),'semantic':semantic.half().float()}
        return self.head(updated)


def dense_loss(output,sample):
    p=output['pixel_probability'];logits=output['pixel_logits'];truth=sample['truth'];valid=sample['valid']
    loss=balanced_pixel_bce(logits,truth,valid)+soft_pixel_dice(p,truth,valid)
    band=dilate(binary_boundary(truth,valid),4)&valid&~dilate(~valid,4)
    if band.any():loss=loss+.5*F.binary_cross_entropy_with_logits(logits[band],truth[band].float())
    return loss


def expanded_cache(args, source, refs, metadata, device):
    data_args=argparse.Namespace(**source['args'])
    data_args.spatial_data=Path(data_args.spatial_data);data_args.routing=Path(data_args.routing)
    data_args.train_samples=args.train_samples
    loader=make_loader(data_args,metadata,'train');dataset=loader.dataset.dataset
    indices=select_indices(dataset,args.train_samples,int(source['args']['seed']))
    previous=[r['identity']['episode_index'] for r in refs['train']]
    if indices[:len(previous)]!=previous:raise RuntimeError('expanded manifest is not an exact prefix extension')
    remaining=indices[len(previous):]
    baseline=load_baseline(metadata,Path(source['checkpoint']),device)
    captured={};hooks=[baseline.phase2.embedding.register_forward_hook(lambda m,i,o:captured.update(deep=o)),
                       baseline.phase2.backbone.layer1.register_forward_hook(lambda m,i,o:captured.update(shallow=o))]
    data=DataLoader(Subset(dataset,remaining),batch_size=1,num_workers=0,collate_fn=loader.collate_fn)
    for i,batch in enumerate(data):
        batch=move_batch(batch,device)
        with torch.no_grad():
            out=baseline(batch);shallow=captured['shallow'];shallow=shallow.reshape(1,32,8,*shallow.shape[-2:]).mean(2)
            sample={'deep':captured['deep'].cpu().half(),'shallow':shallow.cpu().half(),
                    'semantic':F.interpolate(out['semantic_logits'],size=(64,64),mode='bilinear',align_corners=False).cpu().half(),
                    'task':out['task_token'].cpu().float(),'base':out['pixel_probability'].cpu().float(),
                    'prompt_maps':prompt_maps(batch).cpu().half(),
                    'truth':(batch['pixel_gt']==batch['target_class'][:,None,None]).cpu(),'valid':(batch['pixel_gt']!=255).cpu()}
            sample.update({k:batch[k].cpu() for k in ['positive_xy','negative_xy','positive_mask','negative_mask']})
            if not all(torch.isfinite(v).all() for v in sample.values()):raise FloatingPointError('expanded feature cache')
            if not (sample['truth']&sample['valid']).any() or not (~sample['truth']&sample['valid']).any():raise ValueError('invalid expanded binary target')
        path=args.output/'cache'/f'expanded_{i:05d}.pth';torch.save(sample,path)
        ref={'file':str(path),'identity':{'patch_id':str(batch['patch_id'][0]),'target_class':int(batch['target_class'][0]),'episode_index':int(batch['episode_index'][0])},'regression':False}
        refs['train'].append(ref)
        with (args.output/'cache_progress.jsonl').open('a') as f:f.write(json.dumps({'added':i+1,'total_new':len(remaining),'ref':ref})+'\n')
        if i%50==0:print('EXPAND_CACHE',i+1,'/',len(remaining),'pid',os.getpid(),flush=True)
    for h in hooks:h.remove()
    del baseline;torch.cuda.empty_cache()


def image_cache(args,refs,source):
    table=pd.read_parquet(Path(source['args']['spatial_data'])/'patches_index.parquet').set_index('patch_id')
    for split in ['train','val']:
        for i,ref in enumerate(refs[split]):
            row=table.loc[ref['identity']['patch_id']]
            rgb=read_he_patch(Path(row.wsi_path),int(row.x_level0),int(row.y_level0),int(row.width_level0),int(row.height_level0),int(row.width_10x))
            sample=torch.load(ref['file'],map_location='cpu',weights_only=False)
            sample['image']=he_to_tensor(rgb)[None]
            if sample['image'].shape!=(1,3,512,512) or not torch.isfinite(sample['image']).all():raise ValueError('RGB cache contract')
            path=args.output/'cache'/f'online_{split}_{i:05d}.pth';torch.save(sample,path)
            ref['source_feature_file']=ref['file'];ref['file']=str(path)
            with (args.output/'cache_progress.jsonl').open('a') as f:f.write(json.dumps({'split':split,'completed':i+1,'ref':ref})+'\n')
            if i%50==0:print('IMAGE_CACHE',split,i+1,'pid',os.getpid(),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['replay','expand','finetune'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/boundary_refinement/dense_pilot_1014_190217'))
    parser.add_argument('--seed',type=int,default=20261015)
    parser.add_argument('--train-samples',type=int,default=12000)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if Path('/nfs-medical3/zyh/fewclick_v2') not in args.output.resolve().parents:raise ValueError('artifact root')
    if json.loads((args.source/'status.json').read_text())['status']!='complete':raise RuntimeError('source cache incomplete')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'cache').mkdir()
    torch.set_num_threads(4);device=torch.device('cuda:0')
    source=json.loads((args.source/'run.json').read_text());refs=copy.deepcopy(source['refs'])
    metadata=json.loads((Path(source['checkpoint']).parent/'run_metadata.json').read_text())
    started=time.time()
    (args.output/'status.json').write_text(json.dumps({'status':'preparing','worker_pid':os.getpid(),'mode':args.mode}))
    if args.mode=='expand':
        if args.smoke:args.train_samples=len(refs['train'])+4
        expanded_cache(args,source,refs,metadata,device)
    if args.smoke:
        refs['train']=(refs['train'][:2]+refs['train'][-2:]) if args.mode=='expand' else refs['train'][:4]
        refs['val']=refs['val'][:2]
    if args.mode=='finetune':image_cache(args,refs,source)
    train_ids=[r['identity']['patch_id'] for r in refs['train']];val_ids=[r['identity']['patch_id'] for r in refs['val']]
    if len(set(train_ids))!=len(train_ids) or len(set(val_ids))!=len(val_ids) or set(train_ids)&set(val_ids):raise RuntimeError('unique-patch split audit')
    run={'args':{**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'spatial_data':source['args']['spatial_data']},
         'refs':refs,'worker_pid':os.getpid(),'checkpoint':source['checkpoint'],'source_cache':str(args.source),
         'data_seed':source['args']['seed'],'training_seed':args.seed,'test_used':False,'preparation_seconds':time.time()-started,
         'scope':'developmental spatial validation; historical encoder exposure; fixed query prompts',
         'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('dense_prompt_decoder.py'),Path(__file__).with_name('train_dense_boundary.py')]}}
    (args.output/'run.json').write_text(json.dumps(run,indent=2))
    baseline,_=evaluate(None,refs['val'],device,args.output/'baseline_rows.json')
    candidate_payload=torch.load(args.source/'dense128_epoch_007.pth',map_location=device,weights_only=False)
    candidate=DenseContextModel().to(device);candidate.head.load_state_dict(candidate_payload['model'],strict=True);candidate.eval()
    candidate_summary,_=evaluate(candidate,refs['val'],device,args.output/'reference_candidate_rows.json')
    result={'baseline':baseline,'reference_candidate':candidate_summary,'arms':{}}
    phase2=None
    if args.mode=='finetune':
        full=load_baseline(metadata,Path(source['checkpoint']),device);phase2=copy.deepcopy(full.phase2);del full
    arms=['head_continue','layer4_continue'] if args.mode=='finetune' else ['dense128']
    for name in arms:
        torch.manual_seed(args.seed)
        model=DenseContextModel(copy.deepcopy(phase2) if name=='layer4_continue' else None).to(device)
        if args.mode=='finetune':model.head.load_state_dict(candidate_payload['model'],strict=True)
        real=torch.load(refs['val'][0]['file'],map_location=device,weights_only=False)
        with torch.no_grad():
            initial=model(real)['pixel_probability'];expected=candidate(real)['pixel_probability'] if args.mode=='finetune' else real['base']
            initial_error=float((initial-expected).abs().max())
        if initial_error>2e-6:raise RuntimeError(f'initial cache/online mismatch: {initial_error}')
        groups=[{'params':model.head.parameters(),'lr':1e-5 if args.mode=='finetune' else 1e-4}]
        original_layer4=None
        if model.phase2 is not None:
            groups.append({'params':model.phase2.backbone.layer4.parameters(),'lr':1e-6})
            original_layer4={k:v.detach().clone() for k,v in model.phase2.backbone.layer4.state_dict().items()}
        optimizer=torch.optim.AdamW(groups,weight_decay=1e-4);history=[];max_backbone_gradient=0.
        for epoch in range(args.epochs):
            model.train();losses=[];start=time.time()
            for index in np.random.default_rng(args.seed+epoch).permutation(len(refs['train'])):
                sample=torch.load(refs['train'][index]['file'],map_location=device,weights_only=False)
                loss=dense_loss(model(sample),sample)
                if not torch.isfinite(loss):raise FloatingPointError('loss')
                optimizer.zero_grad();loss.backward()
                if model.phase2 is not None:
                    norm=sum(float(p.grad.detach().square().sum()) for p in model.phase2.backbone.layer4.parameters() if p.grad is not None)**.5
                    if not np.isfinite(norm):raise FloatingPointError('layer4 gradient')
                    max_backbone_gradient=max(max_backbone_gradient,norm)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                if not torch.isfinite(norm):raise FloatingPointError('gradient')
                optimizer.step();losses.append(float(loss.detach()))
            model.eval();summary,_=evaluate(model,refs['val'],device,args.output/f'{name}_epoch_{epoch:03}_rows.json')
            summary.update(epoch=epoch,train_loss=float(np.mean(losses)),seconds=time.time()-start,optimizer_steps=(epoch+1)*len(refs['train']))
            summary['eligible']=summary['bf1_2']>=baseline['bf1_2']+.05 and summary['dice']>=baseline['dice']-.005 and summary['micro_dice']>=baseline['micro_dice']-.005 and summary['credible_violations']<=baseline['credible_violations']
            history.append(summary)
            torch.save({'model':model.state_dict(),'mode':args.mode,'arm':name,'epoch':epoch,'summary':summary,'run':run},args.output/f'{name}_epoch_{epoch:03}.pth')
            result['arms'][name]={'history':history,'initial_max_error':initial_error}
            (args.output/'summary.json').write_text(json.dumps(result,indent=2))
            (args.output/'status.json').write_text(json.dumps({'status':'training','arm':name,'epoch':epoch,'worker_pid':os.getpid()}))
            print('EPOCH',name,json.dumps(summary),flush=True)
        loaded=DenseContextModel(copy.deepcopy(phase2) if model.phase2 is not None else None).to(device)
        loaded.load_state_dict(torch.load(args.output/f'{name}_epoch_{args.epochs-1:03}.pth',map_location=device,weights_only=False)['model'],strict=True);loaded.eval()
        with torch.no_grad():error=float((loaded(real)['pixel_probability']-model(real)['pixel_probability']).abs().max())
        if error!=0:raise RuntimeError('checkpoint round trip')
        result['arms'][name]['reload_max_error']=error
        if original_layer4 is not None:
            changed=max(float((v-original_layer4[k]).abs().max()) for k,v in model.phase2.backbone.layer4.state_dict().items() if v.is_floating_point())
            if max_backbone_gradient<=0 or changed<=0:raise RuntimeError('layer4 was not actually trained')
            result['arms'][name].update(layer4_max_gradient=max_backbone_gradient,layer4_max_parameter_change=changed)
        del loaded,model;torch.cuda.empty_cache()
    (args.output/'summary.json').write_text(json.dumps(result,indent=2))
    (args.output/'status.json').write_text(json.dumps({'status':'complete','worker_pid':os.getpid()}));print('COMPLETE',flush=True)


if __name__=='__main__':main()
