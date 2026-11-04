"""Bind both final runs before evaluating fixed localization and safety criteria."""
import argparse
import json
from pathlib import Path

import torch

from .model import ARMS
from benchmarks.v5.architecture_bridge.spatial_quality_guards import compare_rows
from benchmarks.v5.architecture_bridge.compare_vocabulary import summarize_arm
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.encoder_research.continuous_data import sha, TASKS
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.compare_continuous_pilots import paired_interval


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    torch.set_num_threads(2)
    assert json.loads((root/'stage.json').read_text())['status']=='complete_runs'
    c=json.loads((root/'contract.json').read_text()); rows={};reports={};evidence={}
    assert c['updates']==1000 and c['batch']==4 and c['seed']==20261103
    for arm in ARMS:
        folder=root/arm;proof=json.loads((folder/'independent_acceptance.json').read_text());r=json.loads((folder/'report.json').read_text())
        assert proof['status']=='accepted_P3_spatial_evaluation' and not proof['smoke'] and proof['arm']==arm
        assert r['updates']==1000 and r['consumed_tasks']==4000 and r['fresh_prediction_replay_count']==12
        assert r['execution_contract_sha256']==sha(root/'contract.json')
        assert r['initial_common_sha256']==c['initial_common_sha256']
        for filename,key in [('report.json','report_sha256'),('checkpoint.pt','checkpoint_sha256'),('rows.json','rows_sha256'),('masks.npz','masks_sha256')]:
            assert sha(folder/filename)==proof[key]
        assert sha(folder/'training_consumed.jsonl')==c['consumption_sha256']
        checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert checkpoint['updates']==1000 and checkpoint['arm']==arm and checkpoint['seed']==c['seed']
        assert all(torch.isfinite(t).all() for t in checkpoint['model'].values())
        assert all(int(s['step'])==1000 for s in checkpoint['optimizer']['state'].values())
        assert all(g['lr']==g['weight_decay']==1e-4 for g in checkpoint['optimizer']['param_groups'])
        del checkpoint
        reports[arm]=r;rows[arm]=json.loads((folder/'rows.json').read_text());assert len(rows[arm])==3711
        evidence[arm]={name:sha(folder/name) for name in ['independent_acceptance.json','report.json','masks.npz','training_consumed.jsonl']}
    for arm in ARMS[1:]:
        for field in ['seed','inputs','initial_common_sha256','source_sha256','protocol_sha256']:
            assert reports['control'][field]==reports[arm][field]
    values={arm:summarize_arm(rows[arm])[2] for arm in ARMS}
    comparisons={}
    def large(rr,scope):
        g=[r['metrics']['dice'] for r in rr if r['budget']=='3+2' and r['scope']==scope and r['metrics']['positive']/r['metrics']['valid']>=.25]
        return sum(g)/len(g)
    for arm in ARMS[1:]:
        guards=compare_rows(rows['control'],rows[arm])
        delta=values[arm]['small_Dice']-values['control']['small_Dice']
        large_delta={scope:large(rows[arm],scope)-large(rows['control'],scope) for scope in ['local','remote']}
        comparisons[arm]=dict(safety_guards=guards,small_Dice_gain=delta,small_mechanism_gain_ge5pp=delta>=.05,
                              large_Dice_delta=large_delta,large_guard=all(v>=-.02 for v in large_delta.values()),
                              candidate=guards['pass_all'] and delta>=.05 and all(v>=-.02 for v in large_delta.values()))
    s=json.loads((root/'settings.json').read_text())
    old=torch.load(Path(s['reference_match'])/'checkpoint.pt',map_location='cpu',weights_only=True)
    new=torch.load(root/'control/checkpoint.pt',map_location='cpu',weights_only=True)
    assert tree_sha(old['model'])==tree_sha(new['model']), 'Reference reproduction must match old first-seed model exactly'
    assert rows['control']==json.loads((Path(s['reference_match'])/'rows.json').read_text())
    result=dict(status='accepted_size_aware_single_model_comparison',values=values,comparisons=comparisons,
                reference_final_state_and_rows_exact=True,evidence=evidence,contract_sha256=sha(root/'contract.json'),
                source_sha256=sha(__file__),P3_complete=False,WSI_promoted=False,
                limitation='one exposed development seed; no expert routing, size GT only used in training loss or scoring; no automatic next experiment')
    out=root/'comparison';out.mkdir(exist_ok=False);(out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
