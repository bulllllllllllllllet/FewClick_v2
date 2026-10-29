"""Count, sequence, checkpoint and native Dense replay acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import numpy as np
import torch

from .diagnose_local_errors import read,write,sha
from .dense_hardnegative_repair import state_sha
from benchmarks.v5.phase_6_mask_decoder.dense.dense_prompt_decoder import DensePromptDecoder


def run(root):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    c=read(root/'contract.json');s=read(root/'summary.json');plans=read(root/'training_plans.json');hist=read(root/'history.json')
    producer='benchmarks/v5/fair_comparison/dense_generalization.py'
    assert sha(producer)==c['source_sha256'] and sha(root/'training_plans.json')==c['training_plans_sha256']
    assert len(plans)==c['actual_tasks_per_arm'] and len(hist)==c['updates_per_arm'] and hist[-1]['tasks']==len(plans)
    native=read(root/'native_checks.json');assert set(native)=={'crc','tiger'}
    assert all(all(d.values()) for d in native.values())
    if not c['gate']:
        source=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/realized_train_1025_125928/training_plans.jsonl')
        assert sha(source)==c['training_source_sha256'];unique={}
        for line in source.open():
            p=json.loads(line);k=p['dataset'],p['query_id'],p['target']
            if k not in unique:unique[k]=p
        expected=sorted(unique.values(),key=lambda p:hashlib.sha256(f"dense-generalization-v1|{p['dataset']}|{p['query_id']}|{p['target']}".encode()).hexdigest())
        assert plans==expected and len({p['query_id'] for p in plans})==1356 and len(plans)==4733
    for arm in ('continued_original_loss','color_augmentation'):
        assert sha(root/arm/'consumed.json')==sha(root/'training_plans.json')
        assert s['checks'][arm]['reload_exact']
        if c['gate']:assert s['checks'][arm]['optimizer_continuation_exact']
        assert sha(root/arm/'checkpoint.pth')==s['checks'][arm]['checkpoint_sha256']
        assert all(np.isfinite(h[arm]['loss']) and h[arm]['gradient_norm']>0 for h in hist)
    folders=['development']+([] if c['gate'] else ['matched_training'])
    masks=0;replays=0;allrows={}
    heads={}
    for arm in ('original','continued_original_loss','color_augmentation'):
        head=DensePromptDecoder(high_resolution=True).cuda().eval()
        saved=torch.load(root/'initial_dense.pth' if arm=='original' else root/arm/'checkpoint.pth',map_location='cpu',weights_only=False)
        head.load_state_dict(saved if arm=='original' else saved['dense'],strict=True);heads[arm]=head
        if arm=='original':assert state_sha(head)==s['initial_dense_sha256']
    for folder in folders:
        rr=read(root/folder/'rows.json');allrows[folder]=rr
        expected=2 if c['gate'] else 302 if folder=='development' else 29
        assert len(rr)==expected*3
        assert len({(x['index'],x['arm']) for x in rr})==len(rr)
        for r in rr:
            path=root/folder/f"{r['index']:04d}_{r['arm']}.npz";assert sha(path)==r['artifact_sha256']
            with np.load(path,allow_pickle=False) as z:
                mask=z['probability']>=.5;t=z['truth'];v=z['valid'];tp=int(np.count_nonzero(mask&t&v));fp=int(np.count_nonzero(mask&~t&v));fn=int(np.count_nonzero(~mask&t&v))
                assert (tp,fp,fn)==(r['tp'],r['fp'],r['fn'])
                assert abs(2*tp/(2*tp+fp+fn)-r['dice'])<1e-12
                assert abs(t[v].sum()/v.sum()-r['GT_fraction'])<1e-12
            masks+=1
        for p in sorted((root/folder).glob('replay_features_*.pth')):
            i=int(p.stem.split('_')[-1]);f={k:v.cuda() for k,v in torch.load(p,map_location='cpu',weights_only=False).items()}
            with torch.no_grad():
                for arm,head in heads.items():
                    expected_prob=np.load(root/folder/f'{i:04d}_{arm}.npz')['probability']
                    got=head(f)['pixel_probability'][0].cpu().numpy();assert np.array_equal(got,expected_prob);replays+=1
    for arm,g in s['groups'].items():
        for label,metrics in g.items():
            rr=[r for r in allrows['development'] if r['arm']==arm and (label=='all' or (r['GT_fraction']<.05 if label=='small' else r['GT_fraction']>=.5))]
            assert len(rr)==metrics['n'] and abs(statistics.mean(r['dice'] for r in rr)-metrics['dice'])<1e-12
    result=dict(status='accepted_dense_generalization_gate' if c['gate'] else 'accepted_dense_generalization_training_qualified',worker_pid=os.getpid(),
        producer_source_sha256=c['source_sha256'],audit_source_sha256=sha(__file__),summary_sha256=sha(root/'summary.json'),
        mask_counts_recounted=masks,fresh_GPU_head_replays=replays,groups=s['groups'],
        limitations='Extra Dense-only single-seed development experiment; no automatic original/WSI replacement; counts replay does not independently redo BF1.')
    write(root/'independent_acceptance.json',result);print(json.dumps(result),flush=True)
    if not c['gate']:
        groups=s['groups'];control=groups['continued_original_loss'];aug=groups['color_augmentation']
        positive=(aug['all']['dice']>control['all']['dice'] and aug['small']['dice']>control['small']['dice'])
        report='# Dense完整训练池颜色增强对照\n\n状态：独立计数与fresh GPU head重放接受。候选主权重未自动替换。\n\n'
        report+='原4733个image-target/1356视野各一遍，两臂均1184更新、同起点/同原任务/点/损失/优化器，仅图像颜色增强不同；额外训练单seed开发实验，不并入原matched40k或WSI主表。完整302对group0/3+2，最终一次评价，无阈值/中途选模。\n\n'
        report+='| 方法 | Dice | 小目标Dice | 大目标Dice | recall | BF1 |\n|---|---:|---:|---:|---:|---:|\n'
        for arm,g in groups.items():
            report+=f"|{arm}|{g['all']['dice']:.2%}|{g['small']['dice']:.2%}|{g['large']['dice']:.2%}|{g['all']['recall']:.2%}|{g['all']['bf1']:.2%}|\n"
        report+=f"\n增强减继续原loss：总体{100*(aug['all']['dice']-control['all']['dice']):+.3f}pp，小目标{100*(aug['small']['dice']-control['small']['dice']):+.3f}pp，大目标{100*(aug['large']['dice']-control['large']['dice']):+.3f}pp。\n"
        report+=('增强在两项预定Dice指标上均有正向变化；还需评估效应大小、召回/边界/大目标代价及多seed/未见测试，不自动认定修复成功。' if positive else '本增强候选未同时改善预定的总体与小目标Dice，不视为成功修复。')
        report+='\n\n本试验只检验当前轻度染色增强与单轮Dense继续训练，不能由无效结果否定过拟合假设或所有泛化策略。原训练图像/标签均有历史暴露，开发组已用于旧选模。本次未增加分子/目标监督，没有GT用于推理或候选选择。原主15WSI模型保持。\n'
        report+=f"\n接受mask计数{masks}份、fresh GPU head概率重放{replays}次；BF1复用既有固定metrics实现，本接受不声称独立重写BF1。worker与命令见contract、summary、独立接受及pipeline日志。\n"
        (root/'report_zh.md').write_text(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
