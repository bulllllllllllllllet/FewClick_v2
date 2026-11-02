"""Real P2 three-view functional gradients, EMA boundaries and exact checkpoint continuation."""
import argparse,json,os
from pathlib import Path
import pyvips
import numpy as np
import torch
from .vocabulary_model import VocabularyFewClick
from .data import BridgeData,BASE
from .restored_loss import episode_loss
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.encoder_research.train_cell_only_pilot import gate_indices
from benchmarks.v5.fair_comparison.metrics import score


def common_state(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith('vocabulary.')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['continuous','fixed','ema'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=BridgeData();source=BASE/'phikon_joint_tasks_1031_222610/data/training4000.jsonl';lines=source.read_text().splitlines();plans=[json.loads(lines[i]) for i in gate_indices(lines)]
    model=VocabularyFewClick(a.mode);common=tree_sha(common_state(model));trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer=lambda m:torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    opt=optimizer(model);original_centers=None if model.vocabulary.centers is None else model.vocabulary.centers.clone()
    with torch.no_grad():
        inputs=data.model_inputs(plans[0],'xcell','cuda:0');output=model(**inputs)
        if a.mode=='continuous':assert torch.equal(output['query']['pixel_logits'],model.core(**inputs)['query']['pixel_logits'])
    def step(m,o,batch):
        m.train();o.zero_grad(set_to_none=True);losses=[]
        for plan in batch:
            assert data.rows[plan['query_id']]['split']==data.rows[plan['support_id']]['split']=='train'
            inputs=data.model_inputs(plan,'xcell','cuda:0');loss=episode_loss(m(**inputs),data,plan,inputs['prompts']);(loss/len(batch)).backward();losses.append(float(loss.detach()))
        assert m.vocabulary.mapper.weight.grad is not None and torch.isfinite(m.vocabulary.mapper.weight.grad).all()
        o.step();m.after_optimizer_step();assert not m.pending;return losses
    losses=[step(model,opt,plans[:4]),step(model,opt,plans[4:])]
    assert model.vocabulary.mapper.weight.grad.abs().sum()>0
    if a.mode=='fixed':assert torch.equal(model.vocabulary.centers,original_centers)
    if a.mode=='ema':assert not torch.equal(model.vocabulary.centers,original_centers)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),a.output/'checkpoint.pt')
    fresh=VocabularyFewClick(a.mode);restored=optimizer(fresh);state=torch.load(a.output/'checkpoint.pt',map_location='cuda:0',weights_only=True);fresh.load_state_dict(state['model'],strict=True);restored.load_state_dict(state['optimizer'])
    model.eval();fresh.eval();before=tree_sha(model.state_dict());rows=[];masks=[]
    with torch.no_grad():
        for plan in plans:
            inputs=data.model_inputs(plan,'xcell','cuda:0');left=model(**inputs)['query']['pixel_logits'];right=fresh(**inputs)['query']['pixel_logits'];assert torch.equal(left,right)
            mask=(left[0]>=0).cpu().numpy();gt,mpp=data.labels(plan['query_id']);rows.append(score(mask,gt==plan['target'],gt!=255,mpp));masks.append(np.packbits(mask))
    assert tree_sha(model.state_dict())==before and not model.pending
    try:model.after_optimizer_step();raise AssertionError('eval EMA update allowed')
    except RuntimeError:pass
    np.savez_compressed(a.output/'masks.npz',masks=np.stack(masks));saved=np.load(a.output/'masks.npz',allow_pickle=False)['masks']
    for i,plan in enumerate(plans):
        gt,mpp=data.labels(plan['query_id']);mask=np.unpackbits(saved[i],count=512*512).reshape(512,512).astype(bool);assert rows[i]==score(mask,gt==plan['target'],gt!=255,mpp)
    assert step(model,opt,plans[:4])==step(fresh,restored,plans[:4])
    assert tree_sha(model.state_dict())==tree_sha(fresh.state_dict()) and tree_sha(opt.state_dict())==tree_sha(restored.state_dict())
    result=dict(status='accepted_P2_vocabulary_model_gate',mode=a.mode,worker_pid=os.getpid(),common_initial_sha256=common,trainable_parameters=trainable,mapper_parameters=4160,losses=losses,mask_rows=8,exact_fresh_optimizer_EMA_continuation=True,eval_state_immutable=True,
                inputs=data.identity,exact_optimizer_continuation=True,initial_common_heads_sha256=common,
                sources={n:sha(Path(__file__).with_name(n)) for n in ['vocabulary_model.py','cell_vocabulary.py','gate_vocabulary_model.py','restored_loss.py']},scope='real8task engineering gate, not functional fitting or formal P2 quality')
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
