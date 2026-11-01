"""Real two-domain three-view Phikon-to-region forward/backward gate."""
import argparse,json,os
from pathlib import Path
import pyvips
import torch
from .image_interface import build_bridge_image
from benchmarks.v5.fair_comparison.local_data import LocalData,fewclick_inputs
from benchmarks.v5.encoder_research.continuous_data import TASKS,sha
from benchmarks.v5.encoder_research.continuous_prompt import mask_loss,tree_sha


def heads(model):
    return {k:v for k,v in model.state_dict().items() if '.backbone.' not in k}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(2)
    data=LocalData(TASKS)
    source=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/realized_train_1025_125928/training_plans.jsonl')
    plans=[json.loads(s) for s in source.read_text().splitlines()];plans=[next(p for p in plans if p['dataset']==d) for d in ['crc','tiger']]
    baseline,_=build_bridge_image('resnet');common=tree_sha(heads(baseline));shapes=[]
    with torch.no_grad():
        for plan in plans:
            raw,_,_=data.materialize(plan,'cuda:0',True,True)
            for scale,encoder in enumerate(baseline.encoders):
                value=encoder.extract_backbone_features(raw['image' if scale==0 else 'image_'+str(scale)])
                assert tuple(value.shape)==(1,2048,64,64);shapes.append(list(value.shape))
    del baseline;torch.cuda.empty_cache()
    model,identity=build_bridge_image('phikon');assert tree_sha(heads(model))==common
    backbone=model.encoders[0].backbone;before=tree_sha(backbone.state_dict())
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    rows=[]
    for step,plan in enumerate(plans*2):
        raw,prompts,gt=data.materialize(plan,'cuda:0',True,True);inputs=fewclick_inputs(raw)
        # Recover original bytes identically at each physical scale.
        for scale in range(3):
            suffix='' if scale==0 else '_'+str(scale);x=raw['image'+suffix]
            recovered=((x*x.new_tensor([.229,.224,.225])[None,:,None,None]+x.new_tensor([.485,.456,.406])[None,:,None,None])*255).round().clamp(0,255).byte()
            assert torch.equal(recovered,raw['rgb'+suffix].byte())
        optimizer.zero_grad(set_to_none=True);output=model(inputs,inputs,prompts,True,True)
        logits=output['query']['pixel_logits'];assert logits.shape==(1,512,512) and torch.isfinite(logits).all()
        loss=mask_loss(logits,gt['truth'],gt['valid'])+output['assignment_regularization'];loss.backward()
        norms=[sum(float(p.grad.abs().sum()) for p in e.embedding.parameters() if p.grad is not None) for e in model.encoders]
        assert all(v==0 for v in norms) if step==0 else all(v>0 for v in norms)
        assert all(p.grad is None for p in backbone.parameters());optimizer.step()
        rows.append(dict(dataset=plan['dataset'],loss=float(loss.detach()),tile_id=plan['query_id'],step=step,embedding_gradient_L1=norms))
    assert tree_sha(backbone.state_dict())==before
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),a.output/'checkpoint.pt')
    state=torch.load(a.output/'checkpoint.pt',map_location='cpu',weights_only=True);assert tree_sha(state['model'])==tree_sha(model.state_dict())
    report=dict(status='accepted_Phikon_three_view_region_interface',worker_pid=os.getpid(),rows=rows,identity=identity,
                common_initial_heads_sha256=common,original_feature_shapes=shapes,backbone_unchanged=True,RGB_roundtrip_exact=True,
                source_sha256=sha(__file__),interface_sha256=sha(Path(__file__).with_name('image_interface.py')),
                scope='image replacement engineering gate only; multiscaleLEMON/four-arm fresh reload and formal quality remain pending')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
