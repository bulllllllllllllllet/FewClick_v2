"""Bounded CPU validation on real train images while the GPU queue waits."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import torch

from benchmarks.v5.pixel_decoder_study.model import DecoderModel, DecoderData
from benchmarks.v5.architecture_bridge.train_monitor import fixed_train_plans
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.encoder_research.continuous_data import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    s=json.loads((root/'settings.json').read_text());out=root/'cpu_gate';out.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    source=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/phikon_joint_tasks_1031_222610/data/training4000.jsonl')
    plans=fixed_train_plans(source);tasks=[next(x for x in plans if x['dataset']==d and x['kind']=='local') for d in ['crc','tiger']]
    data=DecoderData(s['cache'],'pixel');model=DecoderModel('pixel',device='cpu').eval()
    initial=tree_sha(model.state_dict());rows=[];start=time.monotonic()
    for task in tasks:
        assert data.rows[task['query_id']]['split']=='train'
        inputs=data.model_inputs(task,'xcell','cpu');support=model.base.transform(inputs['support'])
        with torch.no_grad():
            encoded=model.encode(support);prompt_task=model.base.core.support_task(encoded,inputs['prompts'])
            base,maps=model.base.core.decode(encoded,prompt_task,inputs['prompts'],True)
            output=model.pixel(encoded['pixel_embedding'],encoded['pixel_rgb'],prompt_task['task_token'],base)
        assert torch.equal(output['pixel_logits'],base['pixel_logits'])
        assert float((output['pixel_probability']-base['pixel_probability']).abs().max())<=2e-7
        assert output['pixel_logits'].shape==(1,512,512)
        # The copy has a nonzero output solely to expose upstream gradients at init.
        probe=copy.deepcopy(model.pixel);torch.nn.init.constant_(probe.output.weight,.01)
        prediction=probe(encoded['pixel_embedding'],encoded['pixel_rgb'],prompt_task['task_token'],base)
        whole=dict(query=prediction,support=prediction,assignment_regularization=encoded['regularization'])
        loss=episode_loss(whole,data,task,inputs['prompts']);assert torch.isfinite(loss);loss.backward()
        grad={key:float(param.grad.abs().sum()) for key,param in probe.named_parameters() if param.grad is not None}
        assert grad['rgb2.0.weight']>0 and grad['rgb4.0.weight']>0 and grad['deep.0.weight']>0 and grad['output.weight']>0
        assert all(torch.isfinite(param.grad).all() for param in probe.parameters() if param.grad is not None)
        rows.append(dict(task=task,loss=float(loss.detach()),embedding_shape=list(encoded['pixel_embedding'].shape),
                         image_shape=list(encoded['pixel_rgb'].shape),gradient_abs_sums=grad,zero_residual_logits_exact=True))
        status=dict(stage='real_CPU_gate',done=len(rows),total=len(tasks),worker_pid=os.getpid(),seconds=time.monotonic()-start)
        (out/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
        del probe,prediction,whole,loss
    assert tree_sha(model.state_dict())==initial
    result=dict(status='accepted_real_CPU_pixel_head_gate',rows=rows,model_state_unchanged=True,optimizer_updates=0,
                source_sha256=sha(__file__),model_source_sha256=sha('benchmarks/v5/pixel_decoder_study/model.py'),
                GPU_readiness_accepted=False,formal_training_allowed=False)
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
