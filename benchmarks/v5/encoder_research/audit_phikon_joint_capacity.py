"""Read fixed weights and parameter shapes for the planned P1I joint-adaptation gate."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from safetensors import safe_open


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/encoder_inventory_1030_182424/local_weight_identities.json')
    identity = json.loads(inventory.read_text())['phikon_v2']
    weights = Path(identity['path'])
    assert weights.stat().st_size == identity['bytes'] and sha(weights) == identity['sha256']
    config = json.loads((weights.parent / 'config.json').read_text())
    assert (config['model_type'], config['num_hidden_layers'], config['hidden_size'], config['patch_size']) == ('dinov2', 24, 1024, 16)
    tensors = {}
    with safe_open(weights, framework='np') as file:
        for name in file.keys():
            shape = file.get_slice(name).get_shape()
            tensors[name] = dict(shape=shape, elements=math.prod(shape))
    subset = {name: row for name, row in tensors.items() if name.startswith('encoder.layer.23.') or name.startswith('layernorm.')}
    assert len(subset) == 20
    assert subset['encoder.layer.23.attention.attention.query.weight']['shape'] == [1024, 1024]
    assert subset['layernorm.weight']['shape'] == [1024]
    count = sum(row['elements'] for row in subset.values())
    result = dict(status='accepted_fixed_weight_header_capacity_audit', worker_pid=os.getpid(), source_sha256=sha(__file__),
                  inventory_sha256=sha(inventory), weights_sha256=identity['sha256'], config_sha256=sha(weights.parent / 'config.json'),
                  model_type=config['model_type'], total_weight_tensors=len(tensors), total_weight_elements=sum(r['elements'] for r in tensors.values()),
                  num_hidden_layers=24, candidate_subset='last encoder block23 plus final layernorm; subset not yet trained',
                  candidate_tensors=subset, candidate_encoder_parameters=count,
                  fp32_parameter_gradient_two_AdamW_moments_lower_bound_bytes=count * 16,
                  device='CPU metadata only; no encoder inference or GPU allocation', training_updates=0,
                  joint_adaptation_accepted=False, new_image_candidate_added=False,
                  limitations=['lower bound excludes other frozen weights, head, activations and workspaces; not measured GPU peak',
                               'parameter availability does not prove gradient/reload correctness or quality',
                               'formal matched frozen-versus-adapted budget, tasks, optimization and real-input gate remain to be registered'])
    (args.output / 'capacity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'candidate_tensors'}), flush=True)


if __name__ == '__main__':
    main()
