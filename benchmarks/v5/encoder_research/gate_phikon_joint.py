"""Real train-only prefix/gradient/reload gate; no model quality claim."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from transformers import AutoModel, AutoImageProcessor

from .cell_catalog import load_cell_catalog
from .continuous_data import TASKS, sha
from .continuous_prompt import mask_loss, tree_sha
from .shared_prompt import SharedPromptDecoder
from .train_cell_only_pilot import gate_indices


class Tail(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.block = copy.deepcopy(model.encoder.layer[23])
        self.norm = copy.deepcopy(model.layernorm)

    def forward(self, prefix):
        x = self.norm(self.block(prefix)[0])[:, 1:]
        return x.transpose(1, 2).reshape(len(x), 1024, 32, 32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    started = time.time()
    torch.set_num_threads(2)
    torch.manual_seed(20261031)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda:0'
    receipt = json.loads((args.tasks / 'independent_acceptance.json').read_text())
    assert receipt['status'] == 'accepted_novel_train_only4000_joint_tasks'
    assert sha(args.tasks / 'training4000.jsonl') == receipt['training_sha256']
    lines = (args.tasks / 'training4000.jsonl').read_text().splitlines()
    plans = [json.loads(lines[i]) for i in gate_indices(lines)]
    _, rows = load_cell_catalog(TASKS)
    weights = '/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2'
    model = AutoModel.from_pretrained(weights, local_files_only=True).to(device).eval().requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(weights, local_files_only=True)
    frozen_hash = tree_sha(model.state_dict())
    initial_tail = Tail(model).to(device).eval()
    assert sum(p.numel() for p in initial_tail.parameters()) == 12600320
    image_index = Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/image_pilot_1031_015107/phikon/cache/index.jsonl')
    images = {r['tile_id']: r for r in map(json.loads, image_index.read_text().splitlines())}
    cache, bridges = {}, []
    for identity in dict.fromkeys(p[k] for p in plans for k in ['query_id', 'support_id']):
        assert rows[identity]['split'] == 'train'
        item = images[identity]
        assert sha(item['image']) == item['image_sha256']
        with Image.open(item['image']) as im:
            rgb = np.asarray(im.convert('RGB')).copy()
        pixels = processor(images=[rgb], do_resize=False, do_center_crop=False, return_tensors='pt')['pixel_values'].to(device)
        with torch.no_grad():
            full = model(pixel_values=pixels).last_hidden_state
            prefix = model.embeddings(pixels)
            for layer in model.encoder.layer[:23]:
                prefix = layer(prefix)[0]
            assert tuple(prefix.shape) == (1, 1025, 1024)
            segmented = initial_tail(prefix)
            direct = full[:, 1:].transpose(1, 2).reshape(1, 1024, 32, 32)
            assert torch.equal(segmented, direct), (segmented - direct).abs().max().item()
        path = args.output / ('prefix_' + str(len(cache)) + '.npy')
        np.save(path, prefix.cpu().numpy())
        prefix_reload = torch.from_numpy(np.load(path, allow_pickle=False)).to(device)
        assert torch.equal(initial_tail(prefix_reload), direct)
        legacy = np.load(item['feature'], allow_pickle=False)
        assert __import__('hashlib').sha256(legacy.tobytes()).hexdigest() == item['array_sha256']
        delta = (direct.cpu() - torch.from_numpy(legacy.astype(np.float32))[None]).abs()
        bridges.append(dict(tile_id=identity, dataset=rows[identity]['dataset'], prefix=str(path),
                            prefix_sha256=sha(path), image_sha256=item['image_sha256'],
                            full_vs_segment_max_error=0., legacy_fp16_max_error=float(delta.max()),
                            legacy_fp16_mean_error=float(delta.mean())))
        cache[identity] = (prefix_reload.detach(), torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device) / 255)
        print(json.dumps(dict(stage='prefix', count=len(cache), tile_id=identity)), flush=True)

    parent = Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/task_diagnostic_run_1031_023222/mixed/checkpoint.pt')
    assert sha(parent) == 'd8d7c4d0828ac1b95b97e0791f64800be6a84246476c066d026aed720e4d6843'
    state = torch.load(parent, map_location='cpu', weights_only=True)

    def build(adapt):
        head = SharedPromptDecoder(1024).to(device)
        head.load_state_dict(state['model'], strict=True)
        tail = copy.deepcopy(initial_tail).requires_grad_(adapt)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-4)
        opt.load_state_dict(copy.deepcopy(state['optimizer']))
        for group in opt.param_groups:
            group['lr'] = 1e-4
        if adapt:
            opt.add_param_group(dict(params=list(tail.parameters()), lr=1e-5, weight_decay=1e-4))
        return head, tail, opt

    def forward(head, tail, plan):
        qp, rgb = cache[plan['query_id']]
        sp, _ = cache[plan['support_id']]
        qf = tail(qp)
        sf = qf if plan['query_id'] == plan['support_id'] else tail(sp)
        return head(qf, rgb, torch.tensor([plan['positive_xy']], device=device),
                    torch.tensor([plan['negative_xy']], device=device), sf,
                    local=plan['query_id'] == plan['support_id'])

    def update(head, tail, opt, batch):
        opt.zero_grad(set_to_none=True)
        losses = []
        for plan in batch:
            logits = forward(head, tail, plan)
            with Image.open(rows[plan['query_id']]['labels']) as im:
                gt = np.asarray(im).copy()
            assert __import__('hashlib').sha256(gt.tobytes()).hexdigest() == rows[plan['query_id']]['labels_array_sha256']
            loss = mask_loss(logits, torch.from_numpy(gt == plan['target'])[None].to(device),
                             torch.from_numpy(gt != 255)[None].to(device))
            assert torch.isfinite(loss)
            (loss / len(batch)).backward()
            losses.append(float(loss.detach()))
        gradients = {k: float(p.grad.norm()) for k, p in tail.named_parameters() if p.grad is not None}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        opt.step()
        return losses, gradients

    a, at, ao = build(False)
    b, bt, bo = build(True)
    with torch.no_grad():
        for plan in plans:
            assert torch.equal(forward(a, at, plan), forward(b, bt, plan))
    arm_results = {}
    for name, head, tail, opt, adapt in [('frozen', a, at, ao, False), ('joint', b, bt, bo, True)]:
        before_head, before_tail = tree_sha(head.state_dict()), tree_sha(tail.state_dict())
        updates = [update(head, tail, opt, plans[i:i + 4]) for i in [0, 4]]
        changed = [k for k, v in tail.state_dict().items() if not torch.equal(v, initial_tail.state_dict()[k])]
        assert tree_sha(head.state_dict()) != before_head
        assert (tree_sha(tail.state_dict()) != before_tail) == adapt
        assert len(changed) == (20 if adapt else 0)
        if adapt:
            assert all(len(g) == 20 and all(v > 0 for v in g.values()) for _, g in updates)
        checkpoint = args.output / (name + '.pt')
        torch.save(dict(model=head.state_dict(), tail=tail.state_dict(), optimizer=opt.state_dict()), checkpoint)
        fresh, ft, fo = build(adapt)
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        fresh.load_state_dict(saved['model']); ft.load_state_dict(saved['tail']); fo.load_state_dict(saved['optimizer'])
        assert tree_sha(fo.state_dict()) == tree_sha(opt.state_dict())
        with torch.no_grad():
            for plan in plans:
                assert torch.equal(forward(head, tail, plan), forward(fresh, ft, plan))
        original = update(head, tail, opt, plans[:4])
        replayed = update(fresh, ft, fo, plans[:4])
        assert original == replayed
        assert tree_sha(head.state_dict()) == tree_sha(fresh.state_dict())
        assert tree_sha(tail.state_dict()) == tree_sha(ft.state_dict())
        assert tree_sha(opt.state_dict()) == tree_sha(fo.state_dict())
        arm_results[name] = dict(changed_tail_tensors=changed, updates=updates, checkpoint_sha256=sha(checkpoint),
                                 reload_logits_exact=True, continued_model_optimizer_exact=True)
        print(json.dumps(dict(stage='arm_complete', arm=name)), flush=True)
    assert tree_sha(model.state_dict()) == frozen_hash
    report = dict(status='accepted_real_prefix_gradient_reload_gate', worker_pid=os.getpid(),
                  selected_tasks=plans, training_sha256=receipt['training_sha256'], domains=['crc', 'tiger'],
                  prefix_bridges=bridges, initial_arm_logits_exact=True, frozen_full_encoder_unchanged=True,
                  trainable_tail_parameters=12600320, arms=arm_results,
                  GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(), seconds=time.time()-started,
                  source_sha256=sha(__file__), quality_result=False,
                  scope='bounded engineering gate; full1737 prefix acceptance and formal training still pending')
    (args.output / 'acceptance.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
