"""Matched four-arm real training, checkpoint and optimizer continuation gate."""
import argparse,json,os,copy
from pathlib import Path
import pyvips
import numpy as np
import torch
from .data import BridgeData,BASE
from .image_interface import build_bridge_image
from .gate_image_zero_head import heads
from benchmarks.v5.encoder_research.train_cell_only_pilot import gate_indices
from benchmarks.v5.encoder_research.continuous_prompt import mask_loss,tree_sha
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.fair_comparison.metrics import score

ARMS={'A':('resnet','xcell'),'B':('phikon','xcell'),'C':('resnet','lemon'),'D':('phikon','lemon')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=BASE/'phikon_joint_tasks_1031_222610/data/training4000.jsonl';lines=source.read_text().splitlines();plans=[json.loads(lines[i]) for i in gate_indices(lines)]
    data=BridgeData();image_arm,cell_arm=ARMS[a.arm];model,identity=build_bridge_image(image_arm);initial_heads=tree_sha(heads(model))
    make_opt=lambda m:torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    opt=make_opt(model);frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if '.backbone.' in k}
    def step(m,o,batch):
        o.zero_grad(set_to_none=True);losses=[]
        for plan in batch:
            assert data.rows[plan['support_id']]['split']==data.rows[plan['query_id']]['split']=='train'
            output=m(**data.model_inputs(plan,cell_arm,'cuda:0'))
            truth,_=data.labels(plan['query_id']);loss=mask_loss(output['query']['pixel_logits'],torch.from_numpy(truth==plan['target'])[None].cuda(),torch.from_numpy(truth!=255)[None].cuda())+output['assignment_regularization']
            assert torch.isfinite(loss);(loss/len(batch)).backward();losses.append(float(loss.detach()))
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in m.decoder.parameters())
        assert all(p.grad is None for e in m.encoders for p in e.backbone.parameters());o.step();return losses
    losses=[step(model,opt,plans[:4]),step(model,opt,plans[4:])]
    assert tree_sha(heads(model))!=initial_heads
    assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in frozen.items())
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),a.output/'checkpoint.pt')
    fresh,_=build_bridge_image(image_arm);restored=make_opt(fresh);state=torch.load(a.output/'checkpoint.pt',map_location='cuda:0',weights_only=True)
    fresh.load_state_dict(state['model'],strict=True);restored.load_state_dict(state['optimizer']);assert tree_sha(restored.state_dict())==tree_sha(opt.state_dict())
    metrics=[];packed=[]
    with torch.no_grad():
        for plan in plans:
            inputs=data.model_inputs(plan,cell_arm,'cuda:0');left=model(**inputs)['query']['pixel_logits'];right=fresh(**inputs)['query']['pixel_logits']
            assert torch.equal(left,right)
            mask=(left[0]>=0).cpu().numpy();truth,mpp=data.labels(plan['query_id'])
            metrics.append(score(mask,truth==plan['target'],truth!=255,mpp));packed.append(np.packbits(mask))
    np.savez_compressed(a.output/'masks.npz',masks=np.stack(packed))
    saved=np.load(a.output/'masks.npz',allow_pickle=False)['masks']
    for i,plan in enumerate(plans):
        truth,mpp=data.labels(plan['query_id']);mask=np.unpackbits(saved[i],count=512*512).reshape(512,512).astype(bool)
        assert score(mask,truth==plan['target'],truth!=255,mpp)==metrics[i]
    assert step(model,opt,plans[:4])==step(fresh,restored,plans[:4])
    assert tree_sha(model.state_dict())==tree_sha(fresh.state_dict()) and tree_sha(opt.state_dict())==tree_sha(restored.state_dict())
    result=dict(status='accepted_bridge_four_arm_component_gate',arm=a.arm,worker_pid=os.getpid(),identity=identity,inputs=data.identity,initial_common_heads_sha256=initial_heads,
                plans=plans,training_sha256=sha(source),losses=losses,metrics=metrics,exact_disk_logits=True,exact_optimizer_continuation=True,backbone_frozen=True,
                sources={p:sha(Path(__file__).with_name(p)) for p in ['gate_four_arms.py','data.py','image_interface.py']},
                checkpoint_sha256=sha(a.output/'checkpoint.pt'),scope='8 real training tasks including local/remote; 2 updates; formal training and full validation remain pending')
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
