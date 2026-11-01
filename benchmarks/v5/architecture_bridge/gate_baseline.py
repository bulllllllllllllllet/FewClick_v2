"""Real three-view region/cell/hierarchy/graph bridge baseline check."""
import argparse
import json
import os
from pathlib import Path

import pyvips
import numpy as np
import torch

from benchmarks.v5.fair_comparison.adapted_models import clean_fewclick
from benchmarks.v5.fair_comparison.local_data import LocalData,fewclick_inputs
from benchmarks.v5.encoder_research.continuous_data import TASKS,sha
from benchmarks.v5.encoder_research.continuous_prompt import mask_loss,tree_sha
from benchmarks.v5.paper_multiscale.model import load_model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(20261101)
    data=LocalData(TASKS);source=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/realized_train_1025_125928/training_plans.jsonl')
    plans=[json.loads(s) for s in source.read_text().splitlines()];plans=[next(p for p in plans if p['dataset']==d) for d in ['crc','tiger']]
    records=[]
    for label,loader in [('retained_J5_reference',lambda:load_model('cuda:0',True,False)),('controlled_COCO_baseline',lambda:clean_fewclick('cuda:0','xcell',True,False))]:
        model,weights=loader();model.eval();model.set_stage(False)
        assert model.multiscale and not model.use_dense and len(model.encoders)==3
        parameters=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-4)
        rows=[]
        for plan in plans:
            raw,prompts,supervision=data.materialize(plan,'cuda:0',contexts=True,cells=True);inputs=fewclick_inputs(raw)
            opt.zero_grad(set_to_none=True)
            output=model(inputs,inputs,prompts,local=True,shared=True)
            logits=output['query']['pixel_logits'];assert logits.shape==(1,512,512) and torch.isfinite(logits).all()
            loss=mask_loss(logits,supervision['truth'],supervision['valid'])+output['assignment_regularization']
            assert torch.isfinite(loss);loss.backward()
            assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.decoder.parameters())
            before=tree_sha(model.decoder.state_dict());opt.step();assert tree_sha(model.decoder.state_dict())!=before
            rows.append(dict(dataset=plan['dataset'],tile_id=plan['query_id'],loss=float(loss.detach()),cells=[int(inputs[k].shape[1]) for k in ['cells','cells_1','cells_2']],views=3))
        path=a.output/(label+'.pt');torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),path)
        state=torch.load(path,map_location='cpu',weights_only=True);assert tree_sha(state['model'])==tree_sha(model.state_dict())
        records.append(dict(label=label,weights=weights,rows=rows,checkpoint_sha256=sha(path),registered_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in parameters)))
        del model,opt,output,loss,parameters;torch.cuda.empty_cache()
    (a.output/'report.json').write_text(json.dumps(dict(status='accepted_two_domain_three_view_baseline_path',worker_pid=os.getpid(),records=records,source_sha256=sha(__file__),training_sha256=sha(source),quality_claim=False,scope='baseline forward/backward/disk-state gate only; replacement arms and full fresh reload/continuation still pending'),indent=2)+'\n')


if __name__=='__main__':main()
