import argparse
import json
import os
from pathlib import Path
import torch
from .model import build, decode_variants, MODES
from benchmarks.v5.parallel_phikon_study.model import DecoderModel, DecoderData
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.wsi_quality_study.prepare import dump


def configure():
    import transformers
    assert transformers.__version__ == '4.49.0', transformers.__version__
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    c = json.loads((root/'contract.json').read_text())
    configure()
    import transformers
    dump(root/'environment.json', dict(transformers_version=transformers.__version__,
                                       transformers_source=transformers.__file__,
                                       pythonpath=os.environ.get('PYTHONPATH'),
                                       cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],
                                       physical_gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    model = build(c['checkpoints']['H']['path'])
    reference = DecoderModel('H')
    reference.load_state_dict(torch.load(c['checkpoints']['H']['path'], weights_only=True)['model'])
    reference.eval()
    assert tree_sha(model.state_dict()) == tree_sha(reference.state_dict())
    initial = tree_sha(model.state_dict())
    plans = json.loads(Path(c['functional_source']).read_text())
    data = DecoderData(Path(c['cache']), 'H')
    rows = []
    for domain, kind in sorted({(t['dataset'], t['kind']) for t in plans}):
        plan = next(t for t in plans if (t['dataset'], t['kind']) == (domain, kind))
        assert data.rows[plan['support_id']]['split'] == data.rows[plan['query_id']]['split'] == 'train'
        x = data.model_inputs(plan, 'xcell', 'cuda:0')
        se = model.encode(model.base.transform(x['support']))
        qe = se if x['shared'] else model.encode(model.base.transform(x['query']))
        predictions = decode_variants(model, se, qe, x['prompts'], x['local'])
        assert torch.equal(predictions['control'], reference(**x)['query']['pixel_probability'])
        differences = {}
        for mode in MODES:
            model.extension.mode = mode
            full = model(**x)['query']['pixel_probability']
            assert torch.equal(full, predictions[mode]), mode
            assert torch.isfinite(full).all() and ((full >= 0) & (full <= 1)).all()
            differences[mode] = float((full-predictions['control']).abs().mean())
            if x['local']:
                assert differences[mode] == 0
        model.extension.mode = 'control'
        # Directly verify the two interventions rather than infer them from Dice.
        if not x['local']:
            captured = {}
            def hook(module, args):
                captured[model.extension.mode] = args[0].clone()
            handle = model.extension.head.register_forward_pre_hook(hook)
            try:
                decode_variants(model, se, qe, x['prompts'])
            finally:
                handle.remove()
            # Channels 3: are unchanged when only the support evidence is altered.
            assert torch.equal(captured['control'][:, 2:], captured['no_anchor'][:, 2:])
            # Coarse transport is unchanged when only the fine prior is removed.
            assert torch.equal(captured['control'][:, 1:2], captured['no_coarse_prior'][:, 1:2])
            # Both interventions must actually change the intended evidence channels.
            assert not torch.equal(captured['control'][:, :2], captured['no_anchor'][:, :2])
            assert not torch.equal(captured['control'][:, :1], captured['no_coarse_prior'][:, :1])
        rows.append(dict(dataset=domain, kind=kind, cached_full_exact=True,
                         control_matches_original=True, mean_probability_change=differences))
        print(json.dumps(rows[-1]), flush=True)
    assert tree_sha(model.state_dict()) == initial
    dump(root/'engineering.json', dict(status='accepted', pid=os.getpid(), rows=rows,
                                      weights_unchanged=True, train_only=True,
                                      no_optimizer_or_training=True))


if __name__ == '__main__':
    main()
