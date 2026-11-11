"""Five bounded frozen-checkpoint train-only diagnostics; never primary evaluation."""
import argparse,json,os,time
from pathlib import Path
import torch,numpy as np
from torch.nn import functional as F
from benchmarks.v5.same_slide_retrain.model import build
from benchmarks.v5.region_evidence_study.model import EvidenceModel,DecoderData
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump
MODES=['few_region_oracle','matcher_routes','prompt_budget','region_purity']
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=MODES,required=True);a=p.parse_args();r=a.root;c=json.load(open(r/'contract.json'));out=r/a.mode;out.mkdir(exist_ok=False)
 for path,h in c['sources'].items():assert sha(path)==h,path
 assert sha(r/'tasks.json')==c['tasks_sha256']
 for path,h in c['checkpoints'].items():assert sha(path)==h,path
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 dump(out/'worker.json',dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],mode=a.mode))
 plans=json.load(open(r/'tasks.json'));arm='original';prior=json.load(open(Path(c['baseline_diagnostic'])/'original_rows.json'));data=DecoderData(Path(c['cache']),'match');ck=c['weights'][arm]
 m=EvidenceModel() if a.mode=='few_region_oracle' else build(arm)
 m.load_state_dict(torch.load(ck,weights_only=True,map_location='cuda:0')['model']);m.eval();rows=[];masks=[];start=time.monotonic()
 def metric(prob,t,side='query'):
  g,mp=data.labels(t[side+'_id']);pred=prob[0].detach().cpu().numpy()>=.5;masks.append(np.packbits(pred));return score(pred,g==t['target'],g!=255,mp)
 with torch.no_grad():
  for i,t in enumerate(plans):
   assert data.rows[t['support_id']]['split']==data.rows[t['query_id']]['split']=='train'
   x=data.model_inputs(t,'xcell','cuda:0');pr=x['prompts'];row=dict(index=i,task=t,variants={})
   if a.mode=='sansa_mask_oracle':
    se=m.encode(x['support']);qe=se if x['shared'] else m.encode(x['query']);base=m.decode(se,qe,pr,x['local'])['pixel_probability'];row['variants']['points']=metric(base,t)
    sg,_=data.labels(t['support_id']);mask=torch.from_numpy((sg==t['target']).astype('float32'))[None,None].cuda();mask=F.interpolate(mask,size=(640,640),mode='nearest');video=(se[None] if x['local'] else torch.stack([se,qe]))[None]
    prompt={0:{0:dict(prompt_type='mask',prompt=mask)},'shots':1};logits=m.network(video,prompt)['pred_masks'];prob=F.interpolate(logits[:,None],size=(512,512),mode='bilinear',align_corners=False)[-1].sigmoid();row['variants']['support_GT_mask_stronger_supervision']=metric(prob,t)
   else:
    se=m.encode(m.base.transform(x['support']));qe=se if x['shared'] else m.encode(m.base.transform(x['query']));task=m.base.core.support_task(se,pr);source=m.base.core.decode(se,task,pr,True)[0];source=m.pixel(se['pixel_embedding'],se['pixel_embedding'],pr,source);base=m.base.core.decode(qe,task,pr,x['local'])[0]
    full=m.pixel(se['pixel_embedding'],qe['pixel_embedding'],pr,base);row['variants']['points']=metric(full['pixel_probability'],t);row['support']=metric(source['pixel_probability'],t,'support')
    if a.mode=='few_region_oracle':
     sg,_=data.labels(t['support_id']);truth=torch.from_numpy((sg==t['target']).astype('float32'))[None].cuda()
     for name,prob in [('predicted_region',source['pixel_probability']),('support_GT_pooling_stronger_supervision',truth)]:
      pred=m.pixel(se['pixel_embedding'],qe['pixel_embedding'],pr,base,prob);row['variants'][name]=metric(pred['pixel_probability'],t)
    elif a.mode=='matcher_routes':
     row['variants']['region_only_bypass_no_retraining']=metric(base['pixel_probability'],t);row['residual_abs_mean']=float(full['residual'].abs().mean());row['base_probability_std']=float(base['pixel_probability'].std())
    elif a.mode=='prompt_budget':
     for pos,neg in [(1,1),(3,1)]:
      pp={k:(v[:,:pos] if k.startswith('positive') else v[:,:neg] if k.startswith('negative') else v) for k,v in pr.items()};tt=m.base.core.support_task(se,pp);bb=m.base.core.decode(qe,tt,pp,x['local'])[0];pred=m.pixel(se['pixel_embedding'],qe['pixel_embedding'],pp,bb);row['variants'][f'{pos}+{neg}']=metric(pred['pixel_probability'],t)
    elif a.mode=='region_purity':
     stats={}
     for side,e in [('support',se),('query',qe)]:
      gt,_=data.labels(t[side+'_id']);valid=torch.from_numpy(gt!=255).cuda();truth=torch.from_numpy(gt==t['target']).cuda();aa=F.interpolate(e['assignment_low'].float(),size=(512,512),mode='bilinear',align_corners=False)[0];aa=aa/aa.sum(0,keepdim=True).clamp_min(1e-8);mass=aa[:,valid].sum(1);purity=(aa[:,valid]*truth[valid]).sum(1)/mass.clamp_min(1e-8);z={'target_fraction':float(truth[valid].float().mean()),'max_region_purity':float(purity.max())}
      if side=='support':
       for sign in ['positive','negative']:
        w=sample_assignment_at_cells(e['assignment_low'].float(),pr[sign+'_xy'])[0].T;w=w/w.sum(1,keepdim=True).clamp_min(1e-8);z[sign+'_target_purity']=(w@purity).cpu().tolist()
      stats[side]=z
     row['assignment']=stats
   from benchmarks.v5.support_match_study.model import DecoderModel as Parent
   reference=Parent.forward(m,**x)['query']['pixel_probability']
   gt,mp=data.labels(t['query_id']);expected=score(reference[0].cpu().numpy()>=.5,gt==t['target'],gt!=255,mp)
   assert row['variants']['points']==expected, 'Current full/cached baseline mismatch'
   row['historical_baseline_identical']=row['variants']['points']==prior[i]['query']
   row['historical_dice_delta']=None if expected['dice'] is None else expected['dice']-prior[i]['query']['dice']
   rows.append(row);dump(out/'progress.json',dict(pid=os.getpid(),rows=len(rows),total=len(plans),seconds=time.monotonic()-start));dump(out/'rows.json',rows)
 np.save(out/'masks.npy',np.stack(masks));dump(out/'acceptance.json',dict(status='complete_train_only_diagnostic',rows=len(rows),rows_sha256=sha(out/'rows.json'),masks_sha256=sha(out/'masks.npy'),tasks_sha256=c['tasks_sha256'],GT_support_only_oracle=a.mode=='few_region_oracle',no_query_GT_inference=True,primary_comparison=False,training=False))
if __name__=='__main__':main()
