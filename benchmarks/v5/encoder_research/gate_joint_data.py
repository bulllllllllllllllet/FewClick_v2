"""Integrate accepted prefixes, loss, saved tail/head checkpoints and metric replay."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from .continuous_data import sha
from .continuous_prompt import mask_loss, tree_sha
from .gate_phikon_joint import Tail
from .joint_data import JointData
from .shared_prompt import SharedPromptDecoder
from benchmarks.v5.fair_comparison.metrics import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(20261031)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    base = Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')
    previous = base/'phikon_joint_gate_1031_223143/data'
    proof = json.loads((previous/'acceptance.json').read_text())
    assert proof['status'] == 'accepted_real_prefix_gradient_reload_gate'
    data = JointData(base/'phikon_prefix_1031_223355/full')
    full = AutoModel.from_pretrained('/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2',local_files_only=True)
    template = Tail(full); del full
    results = {}
    for arm in ['frozen','joint']:
        saved = torch.load(previous/(arm+'.pt'),map_location='cpu',weights_only=True)
        import copy
        tail=copy.deepcopy(template).cuda().eval().requires_grad_(arm=='joint')
        tail.load_state_dict(saved['tail'],strict=True)
        head=SharedPromptDecoder(1024).cuda().eval(); head.load_state_dict(saved['model'],strict=True)
        rows=[]; masks=[]
        for plan in proof['selected_tasks']:
            with torch.no_grad():
                inputs=data.model_inputs(plan,tail,'cuda:0'); logits=head(**inputs)
                refs={r['tile_id']:r for r in proof['prefix_bridges']}
                for key in ['query_id','support_id']:
                    original=np.load(refs[plan[key]]['prefix'],allow_pickle=False)
                    assert np.array_equal(data.inputs(plan[key])[0].numpy(),original)
                qp=torch.from_numpy(np.load(refs[plan['query_id']]['prefix'],allow_pickle=False)).cuda()
                sp=torch.from_numpy(np.load(refs[plan['support_id']]['prefix'],allow_pickle=False)).cuda()
                direct=dict(inputs,query_feature=tail(qp),support_feature=tail(sp))
                assert torch.equal(logits,head(**direct))
                mask=(logits[0]>=0).cpu().numpy()
            gt,mpp=data.labels(plan['query_id'])
            rows.append(score(mask,gt==plan['target'],gt!=255,mpp)); masks.append(np.packbits(mask))
        np.savez_compressed(args.output/(arm+'_masks.npz'),masks=np.stack(masks))
        packed=np.load(args.output/(arm+'_masks.npz'),allow_pickle=False)['masks']
        for i,plan in enumerate(proof['selected_tasks']):
            replay=np.unpackbits(packed[i],count=512*512).reshape(512,512).astype(bool)
            gt,mpp=data.labels(plan['query_id'])
            assert score(replay,gt==plan['target'],gt!=255,mpp)==rows[i]
        opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=1e-4)
        if arm=='joint': opt.add_param_group(dict(params=list(tail.parameters()),lr=1e-5,weight_decay=1e-4))
        opt.load_state_dict(saved['optimizer']); assert tree_sha(opt.state_dict())==tree_sha(saved['optimizer'])
        before=tree_sha(tail.state_dict()); opt.zero_grad()
        plan=proof['selected_tasks'][0]; truth,valid=data.supervision(plan,'cuda:0')
        loss=mask_loss(head(**data.model_inputs(plan,tail,'cuda:0')),truth,valid)
        assert torch.isfinite(loss); loss.backward(); opt.step()
        assert (tree_sha(tail.state_dict())!=before)==(arm=='joint')
        results[arm]=dict(masks_replayed=8,rows=rows,finite_training_loss=float(loss.detach()),tail_changed=arm=='joint')
    (args.output/'acceptance.json').write_text(json.dumps(dict(status='accepted_joint_data_loss_checkpoint_metric_integration',worker_pid=os.getpid(),inputs=data.identity,arms=results,source_sha256=sha(__file__),data_source_sha256=sha(Path(__file__).with_name('joint_data.py')),formal_training_started=False,scope='integration gate only; formal trainer and full3711 evaluation still pending'),indent=2)+'\n')


if __name__=='__main__': main()
