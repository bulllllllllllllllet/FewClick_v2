"""Uniform vs uncertainty-sampled refinement of a frozen dense coarse mask."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from .point_boundary_detail import PointBoundaryDetail, training_indices
from .train_dense_boundary import evaluate
from .continue_dense_boundary import DenseContextModel
from benchmarks.v4.phase_6_mask_decoder.src.joint_losses import balanced_pixel_bce,soft_pixel_dice


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=8);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if Path('/nfs-medical3/zyh/fewclick_v2') not in args.output.resolve().parents:raise ValueError('artifact root')
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path('/nfs-medical3/zyh/fewclick_v2/boundary_refinement/next_stage_1014_200033')
    source=json.loads((root/'finetune_1014_200033/run.json').read_text());refs=source['refs']
    if args.smoke:refs={split:rows[:4 if split=='train' else 2] for split,rows in refs.items()}
    checkpoint=root/'matched96k_1014_221710/dense128_epoch_031.pth'
    coarse_state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
    torch.set_num_threads(4);device=torch.device('cuda:0');seed=20261014
    coarse=DenseContextModel().to(device);coarse.load_state_dict(coarse_state,strict=True);coarse.eval().requires_grad_(False)
    baseline,_=evaluate(coarse,refs['val'],device,args.output/'baseline_rows.json')
    run={'args':{'spatial_data':source['args']['spatial_data'],'epochs':args.epochs,'smoke':args.smoke,'seed':seed},
         'refs':refs,'coarse_checkpoint':str(checkpoint),'worker_pid':os.getpid(),'test_used':False,
         'training_points':4096,'inference_points':65536,
         'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('point_boundary_detail.py')]},
         'scope':'developmental same-val selection, historical encoder exposure; GT only for training sampling/loss'}
    (args.output/'run.json').write_text(json.dumps(run,indent=2))
    result={'baseline':baseline,'arms':{}}
    real=torch.load(refs['val'][0]['file'],map_location=device,weights_only=False)
    for strategy in ['uniform','uncertain']:
        torch.manual_seed(seed);model=PointBoundaryDetail(coarse_state).to(device)
        with torch.no_grad():
            initial=model(real);error=float((initial['pixel_probability']-initial['coarse_probability']).abs().max())
            tampered={**real,'truth':~real['truth'],'valid':~real['valid']}
            if not torch.equal(initial['indices'],model(tampered)['indices']):raise RuntimeError('GT-dependent inference selection')
        if error>2e-6:raise RuntimeError('zero initialization')
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
        history=[]
        for epoch in range(args.epochs):
            model.train();losses=[];start=time.time()
            for index in np.random.default_rng(seed+epoch).permutation(len(refs['train'])):
                sample=torch.load(refs['train'][index]['file'],map_location=device,weights_only=False)
                with torch.no_grad():probability=coarse(sample)['pixel_probability']
                indices=training_indices(probability,sample['truth'],sample['valid'],strategy,seed+epoch*100000+int(index))
                output=model(sample,indices)
                truth=sample['truth'].flatten(1).gather(1,indices);valid=torch.ones_like(truth)
                loss=balanced_pixel_bce(output['point_logits'],truth,valid)+soft_pixel_dice(output['point_probability'],truth,valid)
                if not torch.isfinite(loss):raise FloatingPointError('point loss')
                optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5)
                if not torch.isfinite(norm):raise FloatingPointError('point gradient')
                optimizer.step();losses.append(float(loss.detach()))
            model.eval();summary,_=evaluate(model,refs['val'],device,args.output/f'{strategy}_epoch_{epoch:03}_rows.json')
            with torch.no_grad():
                output=model(real);keep=torch.ones_like(output['pixel_probability'],dtype=torch.bool).flatten(1)
                keep.scatter_(1,output['indices'],False)
                if not torch.equal(output['pixel_probability'].flatten(1)[keep],output['coarse_probability'].flatten(1)[keep]):raise RuntimeError('outside selected pixels modified')
            summary.update(epoch=epoch,train_loss=float(np.mean(losses)),seconds=time.time()-start)
            summary['eligible']=summary['bf1_2']>=baseline['bf1_2']+.01 and summary['dice']>=baseline['dice']-.002 and summary['micro_dice']>=baseline['micro_dice']-.002 and summary['credible_violations']<=baseline['credible_violations']
            history.append(summary);result['arms'][strategy]={'history':history}
            torch.save({'model':model.state_dict(),'epoch':epoch,'summary':summary,'run':run},args.output/f'{strategy}_epoch_{epoch:03}.pth')
            (args.output/'summary.json').write_text(json.dumps(result,indent=2))
            (args.output/'status.json').write_text(json.dumps({'status':'training','arm':strategy,'epoch':epoch,'worker_pid':os.getpid()}))
            print('EPOCH',strategy,json.dumps(summary),flush=True)
        loaded=PointBoundaryDetail(coarse_state).to(device)
        loaded.load_state_dict(torch.load(args.output/f'{strategy}_epoch_{args.epochs-1:03}.pth',map_location=device,weights_only=False)['model'],strict=True);loaded.eval()
        with torch.no_grad():error=float((loaded(real)['pixel_probability']-model(real)['pixel_probability']).abs().max())
        if error!=0:raise RuntimeError('point checkpoint round trip')
        result['arms'][strategy]['reload_max_error']=error
    (args.output/'summary.json').write_text(json.dumps(result,indent=2))
    (args.output/'status.json').write_text(json.dumps({'status':'complete','worker_pid':os.getpid()}));print('COMPLETE',flush=True)


if __name__=='__main__':main()
