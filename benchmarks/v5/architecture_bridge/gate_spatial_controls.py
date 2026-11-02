"""Real-input P3 control invariants; not a functional learning acceptance."""
import argparse
import gc
import json
from pathlib import Path
import pyvips
import torch
from .spatial_controls import ARMS,SpatialData,SpatialModel,SpatialCell
from .repaired_model import RepairedContinuous
from .restored_loss import episode_loss
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import transformers
    assert transformers.__version__=='4.49.0'
    source=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/phikon_joint_tasks_1031_222610/data/training4000.jsonl')
    plans=list(map(json.loads,source.read_text().splitlines()))
    selected=[next(x for x in plans if x['dataset']==d and x['kind']==k) for d in ['crc','tiger'] for k in ['local','cross_present','cross_absent']]
    data=SpatialData(a.cache,'S11');baseline=RepairedContinuous();baseline.eval()
    initial=tree_sha(baseline.state_dict());references=[]
    with torch.no_grad():
        for plan in selected:references.append(baseline(**data.model_inputs(plan,'xcell','cuda:0'))['query']['pixel_logits'].cpu())
    del baseline;gc.collect();torch.cuda.empty_cache();receipts=[]
    for arm in ARMS:
        data=SpatialData(a.cache,arm);model=SpatialModel(arm);model.eval();assert tree_sha(model.state_dict())==initial
        for plan_index,plan in enumerate(selected):
            inputs=data.model_inputs(plan,'xcell','cuda:0')
            with torch.no_grad():output=model(**inputs);logits=output['query']['pixel_logits'];assert torch.isfinite(logits).all()
            if arm=='S11':assert torch.equal(logits.cpu(),references[plan_index])
            if arm=='I':
                def perturb(x):
                    z=dict(x)
                    for s in ['', '_1','_2']:
                        z['cells'+s]=torch.full_like(z['cells'+s],float('nan'))
                        z['cell_valid'+s]=~z['cell_valid'+s]
                        count='total_cell_count' if not s else 'total'+s;z[count]=z[count]+9999
                    return z
                with torch.no_grad():changed=model(**dict(inputs,support=perturb(inputs['support']),query=perturb(inputs['query'])))['query']['pixel_logits']
                assert torch.equal(logits,changed)
        # Actual data -> model -> signed episode loss -> gradient path, each arm.
        model.train();inputs=data.model_inputs(selected[0],'xcell','cuda:0');loss=episode_loss(model(**inputs),data,selected[0],inputs['prompts']);loss.backward()
        assert torch.isfinite(loss) and any(x.grad is not None and x.grad.count_nonzero() for x in model.parameters())
        if arm=='I':assert all(x.grad is None for x in model.representation.parameters())
        # Module-level orthogonality uses actual retained cells and independent mass.
        cells=inputs['support']['cells'];valid=inputs['support']['cell_valid'];n=cells.shape[1]
        region=torch.randn(1,64,256,device='cuda');mass=torch.softmax(torch.randn(1,64,n,device='cuda'),1);total=inputs['support']['total_cell_count']
        snapshots={};hook=model.core.cell.cell.register_forward_pre_hook(lambda m,args:snapshots.update(mlp=args[0].detach().clone()))
        with torch.no_grad():result=model.core.cell(region,mass,cells,valid,total)
        hook.remove()
        if arm in ['S00','S01']:assert torch.equal(snapshots['mlp'][...,:2],torch.full_like(cells[...,:2],.5))
        if arm in ['S00','S10']:assert torch.allclose(result['region_cell_count'],total[:,None].expand(-1,64)/64,atol=1e-4)
        if arm in ['S01','S11','H11']:
            expected=(mass/mass.sum(1,keepdim=True))*valid[:,None,:];expected=expected.sum(-1)*total[:,None]/valid.sum(1)[:,None]
            assert torch.allclose(result['region_cell_count'],expected,atol=1e-4)
        # Empty-cell path remains finite for every control.
        with torch.no_grad():empty=model.core.cell(region,mass[:,:,:0],cells[:,:0],valid[:,:0],total*0)
        assert torch.isfinite(empty['fused_tokens']).all()
        if arm=='H11':
            original=SpatialData(a.cache,'S11').inputs(selected[0]['support_id'],'xcell');x=data.inputs(selected[0]['support_id'],'xcell');again=data.inputs(selected[0]['support_id'],'xcell')
            for s in ['', '_1','_2']:
                assert torch.equal(x['cells'+s],again['cells'+s]) and torch.equal(x['cells'+s][...,:10],original['cells'+s][...,:10])
                assert torch.equal(x['cells'+s][...,10:].sort(dim=1).values,original['cells'+s][...,10:].sort(dim=1).values)
        receipts.append(dict(arm=arm,finite_real_tasks=6,loss=float(loss.detach()),initial_sha256=initial));print(json.dumps(receipts[-1]),flush=True)
        del model,output,result,inputs;gc.collect();torch.cuda.empty_cache()
    second=SpatialModel('S11',20261104);assert tree_sha(second.state_dict())!=initial
    (a.output/'report.json').write_text(json.dumps(dict(status='accepted_P3_control_forward_and_invariants',arms=receipts,seed_change_verified=True,functional_learning_accepted=False,optimizer_continuation_accepted=False,formal_training_allowed=False),indent=2)+'\n')


if __name__=='__main__':main()
