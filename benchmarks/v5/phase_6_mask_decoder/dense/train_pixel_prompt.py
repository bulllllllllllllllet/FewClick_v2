"""Matched warm-start controls for per-click visual-to-dense interaction."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from . import module_source
from .continue_dense_boundary import DenseContextModel, dense_loss
from .pixel_prompt_decoder import PixelPromptDenseModel
from .train_dense_boundary import evaluate
from .train_multiscale_dense import decoder_milliseconds


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    artifact_root=Path('/nfs-medical3/zyh/fewclick_v2')
    if args.epochs<1 or artifact_root not in args.output.resolve().parents:
        raise ValueError('epochs/artifact root')
    if args.smoke and args.epochs!=1:
        raise ValueError('smoke requires exactly one epoch')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);device=torch.device('cuda:0');seed=20261014
    source=artifact_root/'boundary_refinement/next_stage_1014_200033/finetune_1014_200033'
    source_run=json.loads((source/'run.json').read_text())
    if json.loads((source/'status.json').read_text())['status']!='complete':
        raise RuntimeError('source cache incomplete')
    refs=copy.deepcopy(source_run['refs'])
    for rows in refs.values():
        for ref in rows:
            ref['file']=ref['source_feature_file']
    if args.smoke:
        refs={s:rows[:(4 if s=='train' else 2)] for s,rows in refs.items()}
    ids={s:[r['identity']['patch_id'] for r in rows] for s,rows in refs.items()}
    if set(ids['train']) & set(ids['val']) or any(len(v)!=len(set(v)) for v in ids.values()):
        raise RuntimeError('patch/split identity mismatch')
    if not args.smoke and (len(refs['train']),len(refs['val']))!=(3000,1024):
        raise RuntimeError('formal manifest budget mismatch')
    checkpoint=artifact_root/'boundary_refinement/next_stage_1014_200033/matched96k_1014_221710/dense128_epoch_031.pth'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
    modules=['train_pixel_prompt','pixel_prompt_decoder','dense_prompt_decoder','continue_dense_boundary','train_dense_boundary']
    run={'args':{**source_run['args'],'mode':'pixel_prompt','epochs':args.epochs,'seed':seed,'smoke':args.smoke,
                 'train_samples':len(refs['train']),'val_samples':len(refs['val'])},
         'refs':refs,'coarse_checkpoint':str(checkpoint),
         'coarse_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
         'source_run':str(source/'run.json'),'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'source_sha256':{str(module_source(m)):hashlib.sha256(module_source(m).read_bytes()).hexdigest() for m in modules},
         'worker_pid':os.getpid(),'test_used':False,'baseline_label':'Frozen matched96k dense128',
         'comparison_pairs':[['per_click','control'],['pooled','control'],['per_click','pooled']],
         'optimizer':{'coarse_lr':1e-5,'interaction_lr':1e-4,'weight_decay':1e-4},
         'budget_note':'same seed, warm-start, fixed prompts, data order, loss and updates; pooled/per_click identical parameters; control has fewer parameters',
         'scope':'developmental spatial validation; historical encoder exposure; fixed query prompts',
         'additional_reference_rows':{}}
    for label,path in [
        ('native_final',artifact_root/'boundary_refinement/multiscale_1015_133617/pilot_1015_133617/native_epoch_007_rows.json'),
        ('point_global_final',artifact_root/'boundary_refinement/point_objective_1015_125213/pilot_1015_125213/global_epoch_007_rows.json')]:
        lookup={r['patch_id']:r for r in json.loads(path.read_text())}
        rows=[lookup[r['identity']['patch_id']] for r in refs['val']]
        if any(row['target_class']!=ref['identity']['target_class'] or row['episode_index']!=ref['identity']['episode_index'] for row,ref in zip(rows,refs['val'])):
            raise RuntimeError('external reference identity mismatch')
        filename=label+'_reference_rows.json';(args.output/filename).write_text(json.dumps(rows))
        run['additional_reference_rows'][label]=filename
    (args.output/'run.json').write_text(json.dumps(run,indent=2))
    (args.output/'status.json').write_text(json.dumps({'status':'baseline','worker_pid':os.getpid()}))
    coarse=DenseContextModel().to(device);coarse.load_state_dict(state,strict=True);coarse.eval().requires_grad_(False)
    baseline,_=evaluate(coarse,refs['val'],device,args.output/'baseline_rows.json')
    if not args.smoke and (abs(baseline['bf1_2']-.5570402079429186)>1e-7 or abs(baseline['dice']-.8034157349685337)>1e-7):
        raise RuntimeError('coarse replay mismatch')
    result={'baseline':baseline,'arms':{}}
    real=torch.load(refs['val'][0]['file'],map_location=device,weights_only=False)
    with torch.no_grad():expected=coarse(real)['pixel_probability']
    print('BASELINE',json.dumps(baseline),flush=True)
    initialized_hash=None
    for mode in ['control','pooled','per_click']:
        torch.manual_seed(seed)
        model=PixelPromptDenseModel(state,mode).to(device).eval()
        with torch.no_grad():
            initial_error=float((model(real)['pixel_probability']-expected).abs().max())
            if initial_error>2e-6:raise RuntimeError('coarse initialization mismatch')
        groups=[{'params':model.coarse.parameters(),'lr':1e-5}]
        gradient_sums={}
        if mode!='control':
            groups.append({'params':model.interaction.parameters(),'lr':1e-4})
            digest=hashlib.sha256()
            for key,value in model.interaction.state_dict().items():
                digest.update(key.encode());digest.update(value.cpu().numpy().tobytes())
            if initialized_hash is not None and digest.hexdigest()!=initialized_hash:
                raise RuntimeError('pooled/per_click initialization differs')
            initialized_hash=digest.hexdigest()
            gradient_sums={key:0. for key,_ in model.interaction.named_parameters()}
        optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
        history=[];max_gradient=0.;torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(args.epochs):
            (args.output/'status.json').write_text(json.dumps({'status':'training','arm':mode,'epoch':epoch,'worker_pid':os.getpid()}))
            start=time.time();model.train();losses=[]
            for step,index in enumerate(np.random.default_rng(seed+epoch).permutation(len(refs['train']))):
                sample=torch.load(refs['train'][index]['file'],map_location=device,weights_only=False)
                output=model(sample);loss=dense_loss(output,sample)
                if not torch.isfinite(loss):raise FloatingPointError('loss')
                optimizer.zero_grad();loss.backward()
                if mode!='control' and (args.smoke or (epoch==0 and step<8)):
                    for key,param in model.interaction.named_parameters():
                        if param.grad is not None:gradient_sums[key]+=float(param.grad.detach().abs().sum())
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                if not torch.isfinite(norm):raise FloatingPointError('gradient')
                max_gradient=max(max_gradient,float(norm));optimizer.step();losses.append(float(loss.detach()))
                if (step+1)%1000==0:print('TRAIN',mode,epoch,step+1,'/',len(refs['train']),flush=True)
            train_seconds=time.time()-start;model.eval()
            summary,_=evaluate(model,refs['val'],device,args.output/f'{mode}_epoch_{epoch:03}_rows.json')
            with torch.no_grad():
                altered={**real,'truth':~real['truth'],'valid':~real['valid'],'target_class':torch.tensor([11],device=device)}
                if not torch.equal(model(real)['pixel_probability'],model(altered)['pixel_probability']):
                    raise RuntimeError('GT-dependent inference')
            summary.update(epoch=epoch,optimizer_steps=(epoch+1)*len(refs['train']),train_loss=float(np.mean(losses)),
                           train_seconds=train_seconds,seconds=time.time()-start,
                           trainable_parameters=sum(p.numel() for p in model.parameters()),peak_cuda_bytes=torch.cuda.max_memory_allocated(device))
            summary['eligible']=(summary['bf1_2']>=baseline['bf1_2']+.01 and summary['dice']>=baseline['dice']-.002
                                 and summary['micro_dice']>=baseline['micro_dice']-.002 and summary['credible_violations']<=baseline['credible_violations'])
            history.append(summary)
            result['arms'][mode]={'history':history,'initial_error':initial_error,'max_gradient_before_clip':max_gradient,
                                  'interaction_gradient_sums':gradient_sums,'interaction_initial_sha256':initialized_hash if mode!='control' else None}
            torch.save({'model':model.state_dict(),'mode':mode,'epoch':epoch,'summary':summary,'run':run},args.output/f'{mode}_epoch_{epoch:03}.pth')
            (args.output/'summary.json').write_text(json.dumps(result,indent=2))
            print('EPOCH',mode,json.dumps(summary),flush=True)
        if not max_gradient:raise RuntimeError('missing gradients')
        if mode!='control' and any(not np.isfinite(v) or v<=0 for v in gradient_sums.values()):
            raise RuntimeError('untrained interaction parameter: '+str(gradient_sums))
        reloaded=PixelPromptDenseModel(state,mode).to(device)
        reloaded.load_state_dict(torch.load(args.output/f'{mode}_epoch_{args.epochs-1:03}.pth',map_location=device,weights_only=False)['model'],strict=True)
        reloaded.eval()
        with torch.no_grad():reload_error=float((reloaded(real)['pixel_probability']-model(real)['pixel_probability']).abs().max())
        if reload_error!=0:raise RuntimeError('checkpoint reload mismatch')
        result['arms'][mode].update(reload_max_error=reload_error,decoder_ms=decoder_milliseconds(model,real))
        del reloaded,model,optimizer;torch.cuda.empty_cache()
    (args.output/'summary.json').write_text(json.dumps(result,indent=2))
    (args.output/'status.json').write_text(json.dumps({'status':'complete','worker_pid':os.getpid()}))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
