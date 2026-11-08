"""Oracle local prompts versus fixed remote support, never pooled in one metric."""
import os,json,hashlib,gc
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
from PIL import Image
from scipy.ndimage import label
from torch.nn import functional as F
from benchmarks.v5.sam_matched_adaptation.model import DecoderModel,DecoderData,points
from benchmarks.v5.additional_sam.model import rgb
from benchmarks.v5.paper_multiscale.data import native_context
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor
from benchmarks.v5.fair_comparison.protocol import resolve_clicks
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.wsi_quality_study.prepare import dump

def prompt_tensor(plan):
 out={}
 for sign in ['positive','negative']:
  out[sign+'_xy']=torch.tensor([plan[sign+'_xy']],dtype=torch.float32,device='cuda:0');out[sign+'_mask']=torch.ones(1,len(plan[sign+'_xy']),dtype=torch.bool,device='cuda:0')
 out['prompt_size_id']=torch.ones(1,dtype=torch.long,device='cuda:0');return out

def native(m,qe,point_prompts=None,box=None):
 outputs=[]
 for i,(embedding,intermediate) in enumerate(qe['native']):
  b=None if box is None else torch.tensor([box],dtype=torch.float32,device='cuda:0')*(1024/512)
  if b is not None and i:b=b/2+256
  sparse,dense=m.wrapped.sam.prompt_encoder(points=points(point_prompts,low=bool(i)) if point_prompts is not None else None,boxes=b,masks=None);net=m.wrapped.net_high if i==0 else m.wrapped.net_low
  outputs.append(net(image_embeddings=embedding,image_pe=[m.wrapped.sam.prompt_encoder.get_dense_pe()],sparse_prompt_embeddings=[sparse],dense_prompt_embeddings=[dense],multimask_output=False,wsi_token_only=False,interm_embeddings=intermediate))
 mask,_,token,feat=outputs[0];logits=((token+outputs[1][2])@feat).view(1,-1,256,256)+mask;return F.interpolate(logits,size=(512,512),mode='bilinear',align_corners=False)[:,0]

def bbox(mask):
 yy,xx=np.where(mask);x0,x1=int(xx.min()),int(xx.max()+1);y0,y1=int(yy.min()),int(yy.max()+1);px=max(1,round(.1*(x1-x0)));py=max(1,round(.1*(y1-y0)));return [max(0,x0-px),max(0,y0-py),min(512,x1+px),min(512,y1+py)]

def main():
 root=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/wsi_native_prompt_control_1109_034133');src=root.parent/'wsi_same_slide_1108_013913';tasks=json.load(open(src/'tasks.json'));qr=json.load(open(src/'query_rows.json'));cache=Path(json.load(open(src/'contract.json'))['cache']);data=DecoderData(cache,'wsi_sam_ft');data.rows.update(qr);selected=[]
 for target in [0,1,3,4,5,6,7,8,9,10,11]:
  pool=[t for t in tasks if t['target']==target and t['query_positive_pixels']>0];pool.sort(key=lambda t:hashlib.sha256(json.dumps(t,sort_keys=True).encode()).hexdigest());seen=set();chosen=[]
  for large in [False,True]:
   n=0
   for t in pool:
    if t['query_id'] in seen or (t['query_positive_pixels']/(512*512)>=.2)!=large:continue
    chosen.append(t);seen.add(t['query_id']);n+=1
    if n==5:break
  for t in pool:
   if len(chosen)==10:break
   if t['query_id'] not in seen:chosen.append(t);seen.add(t['query_id'])
  assert len(chosen)==10;selected+=chosen
 dump(root/'plans.json',selected);states={'public_init':None,'adapt_full_decoder':root.parent/'sam_matched_adaptation_1108_142826/wsi_sam_ft/formal/checkpoint.pt','adapt_author_freeze':root.parent/'wsi_author_freeze_1108_204150/wsi_sam_ft/formal/checkpoint.pt'};dump(root/'contract.json',dict(worker_pid=os.getpid(),plans_sha256=sha(root/'plans.json'),source_tasks_sha256=sha(src/'tasks.json'),checkpoints={k:sha(v) if v else 'strict_public_config_weights' for k,v in states.items()},source_sha256=sha(__file__),query_GT_used_for_local_oracle_prompts=True,remote_query_GT_prompts=False,training_updates=0))
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;allrows=[];skips=[];out=root/'masks';out.mkdir()
 def inputs(identity):
  row=data.rows[identity];x={k:v.cuda() for k,v in data.inputs(row.get('template_id',identity),'xcell').items()}
  for scale in range(3):x['image'+('' if scale==0 else '_'+str(scale))]=he_to_tensor(np.asarray(Image.fromarray(native_context(row,scale)).resize((512,512),Image.Resampling.BILINEAR)).copy())[None].cuda()
  return x
 for state,ck in states.items():
  m=DecoderModel('wsi_sam_ft')
  if ck:m.load_state_dict(torch.load(ck,map_location='cuda:0',weights_only=True)['model'])
  m.eval();before=tree_sha(m.state_dict());rows=[];masklist=[];normal_checks=0
  with torch.no_grad():
   for j,t in enumerate(selected):
    row=qr[t['query_id']];gt,mpp=data.labels(t['query_id']);truth=gt==t['target'];assert truth.any();valid=gt!=255;q=inputs(t['query_id']);qe=m.encode(q);se=m.encode(inputs(t['support_id']));pp=prompt_tensor(t)
    modes={'remote_fixed_support':(m.decode(se,qe,pp,False)['pixel_logits'],truth)}
    try:local=resolve_clicks(dict(dataset='crc',target=t['target'],click_seed=20261109+t['index']),row,maximum=(3,2))
    except ValueError as e:skips.append(dict(state=state,task=j,mode='local_points',reason=str(e)));local=None
    if local is not None:
     prompts=prompt_tensor(local);logits=native(m,qe,point_prompts=prompts);modes['local_points']=(logits,truth)
     if normal_checks<2:
      raw={key:torch.as_tensor(rgb(q['image'+suffix]),device='cuda:0').permute(2,0,1)[None].float() for key,suffix in [('rgb',''),('rgb_1','_1')]};reference=m.wrapped(raw,prompts)['pixel_logits'];assert torch.equal(logits,reference),'native direct path differs from existing point forward';normal_checks+=1
    modes['local_box_all_target']=(native(m,qe,box=bbox(truth)),truth);components,n=label(truth);sizes=np.bincount(components.ravel());sizes[0]=0;component=components==int(sizes.argmax());modes['local_box_largest_component']=(native(m,qe,box=bbox(component)),component)
    for mode,(logits,target) in modes.items():
     assert torch.isfinite(logits).all();pred=(logits[0]>=0).cpu().numpy();rows.append(dict(state=state,case=j,original_task=t['index'],target=t['target'],mode=mode,area_fraction=float(target.sum()/valid.sum()),metrics=score(pred,target,valid,mpp)));masklist.append(np.packbits(pred))
    if j%10==0:print(json.dumps(dict(state=state,cases=j+1,total=len(selected),pid=os.getpid())),flush=True)
  assert tree_sha(m.state_dict())==before;np.save(out/f'{state}.npy',np.stack(masklist));dump(root/f'{state}_rows.json',rows);stored=np.load(out/f'{state}.npy',mmap_mode='r')
  for r,packed in zip(rows,stored):
   t=selected[r['case']];gt,mpp=data.labels(t['query_id']);target=gt==t['target']
   if r['mode']=='local_box_largest_component':cc,n=label(target);sz=np.bincount(cc.ravel());sz[0]=0;target=cc==int(sz.argmax())
   pred=np.unpackbits(packed,count=512*512).reshape(512,512).astype(bool);assert score(pred,target,gt!=255,mpp)==r['metrics']
  allrows+=rows;dump(root/f'{state}_acceptance.json',dict(status='complete_accepted',rows=len(rows),normal_point_forward_exact=normal_checks,weights_unchanged=True,state_sha256=before,masks_sha256=sha(out/f'{state}.npy'),rows_sha256=sha(root/f'{state}_rows.json')));del m;gc.collect();torch.cuda.empty_cache()
 summary={}
 for state in states:
  summary[state]={}
  for mode in ['remote_fixed_support','local_points','local_box_all_target','local_box_largest_component']:
   group=[r for r in allrows if r['state']==state and r['mode']==mode];by=defaultdict(list)
   for r in group:by[r['target']].append(r['metrics']['dice'])
   summary[state][mode]=dict(n=len(group),task_mean_Dice=float(np.mean([r['metrics']['dice'] for r in group])),class_macro_Dice=float(np.mean([np.mean(v) for v in by.values()])),class_count=len(by))
 dump(root/'report.json',dict(status='complete_accepted_readonly_oracle_local_diagnostic',summary=summary,skips=skips,limitations='Local queryGT oracle prompts diagnostic only; remote fixed support no queryGT. Component-box track changes scoring target. Stratified110case selection is not the11000task population estimate; no model/threshold selection.'))
 dump(root/'acceptance.json',dict(status='complete_accepted',report_sha256=sha(root/'report.json'),cases=len(selected),rows=len(allrows)))
if __name__=='__main__':main()
