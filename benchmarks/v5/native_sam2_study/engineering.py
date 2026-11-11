"""Real four-domain/protocol input checks and native decoder gradient audit."""
import argparse,json,os
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import DecoderModel,DecoderData
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.additional_sam.model import rgb


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args();r=a.root;c=json.load(open(r/'contract.json'))
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=DecoderModel(a.arm);m.train();data=DecoderData(Path(c['cache']),a.arm);plans=json.load(open(c['functional_source']));groups=[];native_gradient=[]
    frozen=tree_sha(m.sam.state_dict())
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor=SAM2ImagePredictor(m.sam)
    for domain,kind in sorted({(t['dataset'],t['kind']) for t in plans}):
        plan=next(t for t in plans if (t['dataset'],t['kind'])==(domain,kind))
        assert data.rows[plan['support_id']]['split']==data.rows[plan['query_id']]['split']=='train'
        x=data.model_inputs(plan,'xcell','cuda:0')
        with torch.no_grad():
            out=m(**x);se=m.encode(m.base.transform(x['support']));qe=se if x['shared'] else m.encode(m.base.transform(x['query']))
            cached=m.cached_decode(se,qe,x['prompts'],x['local'])[0]
            assert torch.equal(cached['pixel_probability'],out['query']['pixel_probability'])
            assert torch.isfinite(episode_loss(out,data,plan,x['prompts'])) and cached['pixel_probability'].shape==(1,512,512)
            # Independent official predictor image preparation and native mask forward.
            predictor.set_image(rgb(x['query']['image']))
            for v,w in zip(predictor._features['high_res_feats'],qe['native']['high_res_feats']):assert torch.equal(v,w)
            assert torch.equal(predictor._features['image_embed'],qe['native']['image_embed'])
            xy=lab=None
            if x['local']:
                xy=torch.cat([x['prompts'][sign+'_xy'][0][x['prompts'][sign+'_mask'][0].bool()] for sign in ['positive','negative']])[None]*1024
                lab=torch.cat([torch.full((int(x['prompts'][sign+'_mask'].sum()),),value,device='cuda:0',dtype=torch.int32) for sign,value in [('positive',1),('negative',0)]])[None]
            dense=F.interpolate(cached['coarse_logits'][:,None],size=(256,256),mode='bilinear',align_corners=False)
            reference=predictor._predict(xy,lab,mask_input=dense,multimask_output=False,return_logits=True)[0][:,0]
            assert torch.equal(reference,cached['native_logits']),float((reference-cached['native_logits']).abs().max())
            assert torch.equal(cached['pixel_logits'],cached['native_logits']+cached['coarse_logits'])
        # Frozen decoder must still transmit gradient to dense conditions.
        out=m(**x);q=out['query'];g=torch.autograd.grad(q['native_logits'].mean(),q['coarse_logits'],retain_graph=True)[0]
        assert torch.isfinite(g).all() and g.abs().sum()>0
        loss=episode_loss(out,data,plan,x['prompts']);m.zero_grad(set_to_none=True);loss.backward()
        assert all(v.grad is None for v in m.sam.parameters())
        assert m.pixel.output.weight.grad is not None and m.pixel.output.weight.grad.norm()>0
        assert all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None)
        native_gradient.append(float(g.norm()));groups.append([domain,kind]);print('passed',domain,kind,flush=True)
    del out,q,g,loss,se,qe,cached
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    plan=next(t for t in plans if t['kind']=='same_present');x=data.model_inputs(plan,'xcell','cuda:0');grads=[]
    for step in range(3):
        opt.zero_grad(set_to_none=True);loss=episode_loss(m(**x),data,plan,x['prompts']);loss.backward();grads.append(float(m.pixel.project.weight.grad.norm()));opt.step()
    assert grads[-1]>0 and tree_sha(m.sam.state_dict())==frozen
    with torch.no_grad():
        expected=m(**x)['query']['pixel_probability'];se=m.encode(m.base.transform(x['support']));qe=m.encode(m.base.transform(x['query']));actual=m.cached_decode(se,qe,x['prompts'],False,with_support=False)[0]['pixel_probability'];assert torch.equal(actual,expected)
        swapped=dict(x['prompts'])
        for key in ['xy','mask']:swapped['positive_'+key],swapped['negative_'+key]=swapped['negative_'+key],swapped['positive_'+key]
        other=m.cached_decode(se,qe,swapped,False,with_support=False)[0]['pixel_probability'];dependency=float((other-actual).abs().mean());assert dependency>0
    dump(r/a.arm/'engineering.json',dict(status='accepted',pid=os.getpid(),groups=groups,official_predictor_features_and_native_logits_exact=True,cached_full_before_after_exact=True,native_dense_gradient_norm=native_gradient,matching_project_gradients=grads,support_sign_swap_mean=dependency,all_sam_state_frozen_sha256=frozen,trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),query_native_points=None,point_coordinates_native_scale=1024))

if __name__=='__main__':main()
