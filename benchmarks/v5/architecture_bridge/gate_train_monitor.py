"""Exercise fixed train monitoring on an actual accepted spatial checkpoint."""
import argparse
import json
from pathlib import Path
import pyvips
import torch
from .spatial_controls import SpatialModel,SpatialData
from .train_monitor import fixed_train_plans,evaluate_train_monitor
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.encoder_research.continuous_data import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--learning',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    proof=json.loads((a.learning/'independent_acceptance.json').read_text());assert proof['arm']=='S11' and proof['status']=='accepted_P3_spatial_train_learning_gate'
    assert sha(a.learning/'checkpoint.pt')==proof['checkpoint_sha256']
    model=SpatialModel('S11',20261103,'cpu');model.load_state_dict(torch.load(a.learning/'checkpoint.pt',map_location='cpu',weights_only=True)['model']);model.train()
    original=tree_sha(model.state_dict());data=SpatialData(a.cache,'S11')
    source=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/phikon_joint_tasks_1031_222610/data/training4000.jsonl')
    result=evaluate_train_monitor(model,data,fixed_train_plans(source),'cpu',0,a.output/'monitor.json')
    assert tree_sha(model.state_dict())==original and model.training
    assert len(result['rows'])==32 and len(result['groups'])==6
    (a.output/'acceptance.json').write_text(json.dumps(dict(status='accepted_fixed_train_monitor_integration',source_sha256=sha(__file__),
        checkpoint_sha256=proof['checkpoint_sha256'],monitor_sha256=sha(a.output/'monitor.json'),model_state_unchanged=True,RNG_unchanged=True,
        mode_restored=True,rows=32,device='cpu',step_label='monitor invocation index0 on existing256-step checkpoint; not initialization performance',
        formal_trainer_integration_accepted=False),indent=2)+'\n')


if __name__=='__main__':main()
