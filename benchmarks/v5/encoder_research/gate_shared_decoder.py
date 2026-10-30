"""Real frozen encoder, signed decoder training and fresh optimizer reload gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from .image_features import FrozenImageEncoder,CHANNELS
from .shared_prompt import SharedPromptDecoder
from benchmarks.v5.fair_comparison.local_data import LocalData
from benchmarks.v5.transfer_comparison.data import TransferData


def loss_fn(logits,truth,valid):
    target=truth.float();weight=valid.float();prob=logits.sigmoid()
    bce=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*weight).sum()/weight.sum()
    dice=1-(2*(prob*target*weight).sum()+1)/((prob*weight).sum()+(target*weight).sum()+1)
    return bce+dice


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',required=True,choices=list(CHANNELS))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(20261031);start=time.time();device='cuda:0'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    status=dict(status='loading',worker_pid=os.getpid(),arm=a.arm)
    def save():(a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save();contract=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/p0_tissue_contract_1031_011514')
    plans=[json.loads(x) for x in (contract/'pilot_training4000.jsonl').read_text().splitlines()]
    data=LocalData('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/tasks_1025_125400')
    selected=[]
    for dataset in ['crc','tiger']:
        subset=[p for p in plans if p['dataset']==dataset];used=set()
        for plan in subset:
            if plan['query_id'] not in used:selected.append(plan);used.add(plan['query_id'])
            if len(used)==2:break
    assert len(selected)==4
    encoder=FrozenImageEncoder(a.arm,device)
    torch.manual_seed(20261031)
    head=SharedPromptDecoder(CHANNELS[a.arm]).to(device)
    before={k:v.detach().clone() for k,v in head.state_dict().items()}
    common_init_sha=hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for k,v in sorted(before.items()) if not k.startswith('project.'))).hexdigest()
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001);samples=[];losses=[]
    for index,plan in enumerate(selected):
        raw,prompts,supervision=data.materialize(plan,device,contexts=False,cells=False)
        feature=encoder(raw['rgb']).clone()
        sample=(feature,raw['rgb'].float()/255,prompts['positive_xy'],prompts['negative_xy'],supervision)
        samples.append(sample);optimizer.zero_grad()
        logits=head(*sample[:4]);assert tuple(logits.shape)==(1,512,512) and torch.isfinite(logits).all()
        loss=loss_fn(logits,supervision['truth'],supervision['valid']);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        optimizer.step();losses.append(float(loss.detach()));status.update(status='training_gate',updates=index+1);save()
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    changed=sum(not torch.equal(before[k],v) for k,v in head.state_dict().items());assert changed>0
    torch.save({'model':head.state_dict(),'optimizer':optimizer.state_dict()},a.output/'checkpoint.pt')
    fresh=SharedPromptDecoder(CHANNELS[a.arm]).to(device);fresh_optimizer=torch.optim.AdamW(fresh.parameters(),lr=.001,weight_decay=.0001)
    checkpoint=torch.load(a.output/'checkpoint.pt',map_location=device,weights_only=True);fresh.load_state_dict(checkpoint['model'],strict=True);fresh_optimizer.load_state_dict(checkpoint['optimizer'])
    with torch.no_grad():
        reference=head(*samples[0][:4]);restored=fresh(*samples[0][:4]);assert torch.equal(reference,restored)
    # Prove optimizer state by one deterministic continuation of both instances.
    for model,opt in [(head,optimizer),(fresh,fresh_optimizer)]:
        opt.zero_grad();value=model(*samples[0][:4]);loss_fn(value,samples[0][4]['truth'],samples[0][4]['valid']).backward();opt.step()
    continuation_errors={k:float((v-fresh.state_dict()[k]).abs().max()) for k,v in head.state_dict().items()}
    (a.output/'continuation_diagnostics.json').write_text(json.dumps(continuation_errors,indent=2)+'\n')
    assert all(error==0 for error in continuation_errors.values()),continuation_errors
    remote=TransferData('/nfs-medical3/zyh/fewclick_v2/fair_comparison/transfer_1025_131856/tasks_1025_132300')
    plan=remote.plans[0];inputs=remote.materialize_inputs(plan,device,contexts=False,cells=False)
    support=encoder(inputs['support']['rgb']).clone();query=encoder(inputs['query']['rgb']).clone()
    with torch.no_grad():
        output=head(query,inputs['query']['rgb'].float()/255,inputs['prompts']['positive_xy'][:,:3],inputs['prompts']['negative_xy'][:,:2],support_feature=support,local=False)
        assert output.shape==(1,512,512) and torch.isfinite(output).all()
    np.savez_compressed(a.output/'gate_mask.npz',logits=reference.cpu().numpy(),truth=samples[0][4]['truth'].cpu().numpy(),valid=samples[0][4]['valid'].cpu().numpy())
    report=dict(status='complete_real_shared_decoder_gate_unaccepted',worker_pid=os.getpid(),arm=a.arm,losses=losses,
        changed_state_tensors=changed,frozen_encoder=True,exact_reload=True,exact_optimizer_continuation=True,
        remote_source_only_prompt_gate=True,head_parameters=sum(p.numel() for p in head.parameters()),
        adapter_parameters=sum(p.numel() for p in head.project.parameters()),common_decoder_parameters=sum(p.numel() for p in head.parameters())-sum(p.numel() for p in head.project.parameters()),
        inputs=[dict(id=p['query_id'],dataset=p['dataset'],target=p['target']) for p in selected],
        common_initialization_sha256=common_init_sha,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_seconds=time.time()-start,
        quality_claim=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');status.update(status='complete');save();print(json.dumps(report),flush=True)


if __name__=='__main__':main()
