"""Global-task ownership, unused-gradient semantics and weighted merge controls."""
import tempfile
from pathlib import Path
import unittest

import pyvips
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .distributed_utils import task_numbers,synchronize_gradients,merge_rows
from .evaluate import summarize


def gradient_worker(rank,world,store):
    dist.init_process_group('gloo',init_method='file://'+store,rank=rank,world_size=world)
    a=torch.nn.Parameter(torch.tensor([10.]));b=torch.nn.Parameter(torch.tensor([20.]));c=torch.nn.Parameter(torch.tensor([30.]));frozen=torch.nn.Parameter(torch.tensor([40.]),requires_grad=False)
    optimizer=torch.optim.SGD([a,b,c,frozen],lr=1.)
    a.grad=torch.tensor([float(rank+1)])
    if rank==1:b.grad=torch.tensor([8.])
    synchronize_gradients(optimizer)
    assert torch.equal(a.grad,torch.tensor([2.5]))
    assert torch.equal(b.grad,torch.tensor([2.]))
    assert c.grad is None and frozen.grad is None
    optimizer.step()
    assert a.item()==7.5 and b.item()==18. and c.item()==30. and frozen.item()==40.
    optimizer.zero_grad(set_to_none=True)
    a.grad=torch.tensor([float(rank+1)/4])
    if rank==1:b.grad=torch.tensor([2.])
    synchronize_gradients(optimizer,average=False)
    assert a.grad.item()==2.5 and b.grad.item()==2.
    assert c.grad is None and frozen.grad is None
    dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_four_ranks_cover_same_four_serial_episodes_once(self):
        for step in (0,1,1999,2000,4000):
            tasks=[n for rank in range(4) for n in task_numbers(step,rank,4,4)]
            self.assertEqual(tasks,task_numbers(step,0,1,4));self.assertEqual(len(set(tasks)),4)
        with self.assertRaises(ValueError):task_numbers(0,0,6,4)

    def test_collective_mean_includes_zero_contributions_but_keeps_all_unused_none(self):
        with tempfile.TemporaryDirectory(prefix='fewclick_dist_test_') as tmp:
            mp.spawn(gradient_worker,args=(4,str(Path(tmp)/'store')),nprocs=4,join=True)

    def test_merged_scores_use_tasks_instead_of_rank_means_and_preserve_anchor(self):
        plans=[dict(number=i,quality_condition='original') for i in range(3)]
        def row(i,dice):return dict(plans[i],dataset='crc',target=0,positive=1,dice=dice,all_foreground_dice=.5,bf1_um2=.4,
                                   false_positive_fraction=.1,protocol='local',query_organ='colon')
        a,b,c=row(0,.1),row(1,.3),row(2,.9)
        ordered,report=merge_rows([[c,a],[b]],plans,anchor_count=1)
        self.assertEqual([r['number'] for r in ordered],[0,1,2])
        self.assertAlmostEqual(report['overall']['dice'],(.1+.3+.9)/3)
        self.assertAlmostEqual(report['anchor']['dice'],.1)
        self.assertAlmostEqual(report['expanded_only']['dice'],.6)
        self.assertEqual(report['overall'],summarize([a,b,c]))
        with self.assertRaises(ValueError):merge_rows([[a,a],[b]],plans)
        with self.assertRaises(ValueError):merge_rows([[a],[b]],plans)
        with self.assertRaises(ValueError):merge_rows([[a,b,c]],plans,anchor_count=4)


if __name__=='__main__':unittest.main()
