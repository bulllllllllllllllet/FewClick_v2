"""Synchronous gradient averaging with dynamic trainable/unused parameters."""
from collections import defaultdict
import hashlib
import random

import numpy as np
import torch
import torch.distributed as dist

from .evaluate import summarize
from .scoring import scoring_policy,split_target_tasks


def task_numbers(step,rank,world,global_batch):
    if global_batch<1 or not 0<=rank<world or global_batch%world:raise ValueError('global batch must divide evenly across ranks')
    local=global_batch//world
    return list(range(step*global_batch+rank*local,step*global_batch+(rank+1)*local))


def synchronize_gradients(optimizer,average=True):
    """Reduce gradients; globally unused parameters retain grad=None."""
    params=[p for g in optimizer.param_groups for p in g['params'] if p.requires_grad]
    if not params:raise ValueError('no trainable parameters')
    flags=torch.tensor([p.grad is not None for p in params],device=params[0].device,dtype=torch.int32)
    dist.all_reduce(flags,op=dist.ReduceOp.MAX)
    selected=[p for p,used in zip(params,flags.tolist()) if used]
    if selected:
        flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in selected])
        dist.all_reduce(flat)
        if average:flat.div_(dist.get_world_size())
        offset=0
        for p in selected:
            p.grad=flat[offset:offset+p.numel()].view_as(p);offset+=p.numel()
    return len(selected)


def rng_state(device):
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device),numpy=np.random.get_state(),python=random.getstate())


def restore_rng(value,device):
    torch.set_rng_state(value['torch'].cpu());torch.cuda.set_rng_state(value['cuda'].cpu(),device)
    np.random.set_state(value['numpy']);random.setstate(value['python'])


def state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def optimizer_digest(state):
    """Hash optimizer values independently of their CPU/CUDA placement."""
    h=hashlib.sha256()
    def visit(value):
        if isinstance(value,torch.Tensor):
            h.update(str((value.dtype,tuple(value.shape))).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value,dict):
            for key in sorted(value,key=str):h.update(repr(key).encode());visit(value[key])
        elif isinstance(value,(tuple,list)):
            for item in value:visit(item)
        else:h.update(repr(value).encode())
    visit(state);return h.hexdigest()


def merge_rows(shards,plans,anchor_count=0):
    if not 0<=anchor_count<=len(plans):raise ValueError('invalid global validation anchor')
    indices={(p['number'],p['quality_condition']):i for i,p in enumerate(plans)}
    rows=[r for shard in shards for r in shard]
    keys=[(r['number'],r['quality_condition']) for r in rows]
    if len(rows)!=len(plans) or len(set(keys))!=len(rows) or set(keys)!=set(indices):raise ValueError('validation shard identities missing/duplicated')
    rows.sort(key=lambda r:indices[r['number'],r['quality_condition']]);groups=defaultdict(list)
    for r in rows:
        for k in ('dataset','protocol','quality_condition','query_organ'):groups[k+'|'+r[k]].append(r)
        groups[f'class|{r["dataset"]}|{r["target"]}'].append(r)
    _,excluded=split_target_tasks(rows)
    report=dict(overall=summarize(rows),legacy_all_targets=summarize(rows,include_background_targets=True),
        excluded_background_target_tasks=summarize(excluded,include_background_targets=True),
        strata={k:summarize(v) for k,v in sorted(groups.items())},threshold=.5,scoring_policy=scoring_policy(),
        scope='fixed GT-simulated clicks; spatial development; distributed per-task merge; not PUMA/test')
    if anchor_count:report.update(anchor=summarize(rows[:anchor_count]),expanded_only=summarize(rows[anchor_count:]),anchor_episodes=anchor_count)
    return rows,report
