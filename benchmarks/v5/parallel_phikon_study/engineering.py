"""Four real groups, exact baseline init for E/H, G footprint/interface audit."""
import argparse,json,os,hashlib,time
from pathlib import Path
import torch
from .model import DecoderModel,DecoderData,SelfSupport
from benchmarks.v5.support_match_study.model import DecoderModel as Parent
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.same_slide_retrain.model import frozen_backbone_state
from benchmarks.v5.wsi_quality_study.prepare import dump


def self_support_unit():
    # Isolated numerical test: actual learning acceptance still uses the fixed32 real tasks.
    module=SelfSupport().cuda();sf=torch.zeros(1,64,64,64,device='cuda');sf[:,0,:,:32]=1;sf[:,1,:,32:]=1
    sf.requires_grad_();qf=sf.detach().clone().requires_grad_();p=torch.full((1,512,512),.1,device='cuda');p[:,:,:256]=.9
    prompts={}
    for sign,x in [('positive',.25),('negative',.75)]:
        prompts[sign+'_xy']=torch.tensor([[[x,.5]]],device='cuda');prompts[sign+'_mask']=torch.ones(1,1,device='cuda',dtype=torch.bool)
    opt=torch.optim.AdamW(module.parameters(),lr=1e-4);grads=[]
    for _ in range(3):
        opt.zero_grad(set_to_none=True);sf.grad=None;qf.grad=None;out,diag=module(sf,qf,prompts,p);assert diag['active']==1
        (out-1).square().mean().backward();grads.append(float(qf.grad.norm()));opt.step()
    assert grads[-1]>0
    empty,diag=module(sf,qf,prompts,torch.full_like(p,.5));assert diag['active']==0 and torch.count_nonzero(empty)==0
    return {'synthetic_active_branch_backward':True,'zero_correction_without_evidence':True,'query_gradient_norm':grads}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args();r=a.root;c=json.load(open(r/'contract.json'));started=time.monotonic()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=DecoderModel(a.arm);parent=Parent('match');d=DecoderData(Path(c['cache']),a.arm);plans=json.load(open(c['functional_source']));groups=[];fine_checks=[]
    if a.arm in ['E','H']:assert all(torch.equal(v,m.state_dict()[k]) for k,v in parent.state_dict().items())
    else:
        for k,v in parent.state_dict().items():
            if '.backbone.' not in k:assert torch.equal(v,m.state_dict()[k]),k
        assert tree_sha(m.base.core.encoders[1].backbone.state_dict())==tree_sha(parent.base.core.encoders[1].backbone.state_dict())
    frozen=tree_sha(frozen_backbone_state(m));m.train()
    with torch.no_grad():
        for domain,kind in sorted({(t['dataset'],t['kind']) for t in plans}):
            t=next(t for t in plans if (t['dataset'],t['kind'])==(domain,kind));assert d.rows[t['support_id']]['split']==d.rows[t['query_id']]['split']=='train'
            x=d.model_inputs(t,'xcell','cuda:0');out=m(**x)
            assert out['query']['pixel_probability'].shape==(1,512,512) and torch.isfinite(episode_loss(out,d,t,x['prompts']))
            if a.arm in ['E','H']:
                old=parent(**x);assert torch.equal(out['query']['pixel_probability'],old['query']['pixel_probability']);assert torch.equal(out['support']['pixel_probability'],old['support']['pixel_probability'])
            else:
                calls=[];bb=m.base.core.encoders[1].backbone;hook=bb.encoder.register_forward_hook(lambda module,args,result:calls.append(tuple(result.last_hidden_state.shape)))
                try:encoded=m.encode(m.base.transform(x['support']))
                finally:hook.remove()
                assert len(calls)==6 and all(v==(1,1025,1024) for v in calls)
                for index in [1,2]:
                    image=x['support']['image_'+str(index)]
                    assert torch.equal(m.base.core.encoders[index].backbone(image)['out'],parent.base.core.encoders[index].backbone(image)['out'])
                image=x['support']['image'];tiled=m.base.core.encoders[0].backbone(image)['out'];full=parent.base.core.encoders[0].backbone(image)['out']
                assert tiled.shape==full.shape==(1,2048,64,64) and not torch.equal(tiled,full)
                fine_checks.append({'group':[domain,kind],'encoder_calls_per_image':len(calls),'native_fine_grid':[64,64],'context_exact':True,'fine_mean_abs_change':float((tiled-full).abs().mean())})
            se=m.encode(m.base.transform(x['support']));qe=se if x['shared'] else m.encode(m.base.transform(x['query']))
            cached=m.cached_decode(se,qe,x['prompts'],x['local'])[0];assert torch.equal(cached['pixel_probability'],out['query']['pixel_probability']);groups.append([domain,kind]);print('input passed',domain,kind,flush=True)
    del parent;torch.cuda.empty_cache();t=next(t for t in plans if t['kind']=='same_present');x=d.model_inputs(t,'xcell','cuda:0');opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);grad=[]
    for step in range(3):
        opt.zero_grad(set_to_none=True);out=m(**x);loss=episode_loss(out,d,t,x['prompts']);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        grad.append({k:float(p.grad.norm()) for k,p in m.named_parameters() if p.grad is not None and (k=='pixel.project.weight' or k.startswith('extension.head'))});opt.step()
    assert grad[-1]['pixel.project.weight']>0
    if a.arm=='H':assert grad[-1]['extension.head.2.weight']>0
    assert tree_sha(frozen_backbone_state(m))==frozen
    with torch.no_grad():
        se=m.encode(m.base.transform(x['support']));qe=m.encode(m.base.transform(x['query']));out=m.cached_decode(se,qe,x['prompts'],False)[0];assert torch.equal(out['pixel_probability'],m(**x)['query']['pixel_probability'])
        prompts=dict(x['prompts'])
        for k in ['xy','mask']:prompts['positive_'+k],prompts['negative_'+k]=prompts['negative_'+k],prompts['positive_'+k]
        other=m.cached_decode(se,qe,prompts,False)[0];dependency=float((out['pixel_probability']-other['pixel_probability']).abs().mean());assert dependency>0
    unit=self_support_unit() if a.arm=='E' else None
    dump(r/a.arm/'engineering.json',dict(status='accepted',pid=os.getpid(),groups=groups,parent_initial_state_and_outputs_exact=a.arm in ['E','H'],G_fine_checks=fine_checks,cached_full_before_after_exact=True,gradients=grad,support_swap_mean=dependency,all_frozen_state_unchanged=True,self_support_unit=unit,trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),seconds=time.monotonic()-started))

if __name__=='__main__':main()
