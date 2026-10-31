"""Matched continuation task-factor effects and full positive/absent guard."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    reports={};rows={}
    for arm in ['local_control','mixed']:
        accepted=json.loads((a.root/arm/'independent_acceptance.json').read_text());assert accepted['status']=='accepted_fixed_image_pilot' and accepted['total_optimizer_updates']==2000
        reports[arm]=accepted;rows[arm]=json.loads((a.root/arm/'rows.json').read_text())
    assert reports['local_control']['initial_checkpoint_sha256']==reports['mixed']['initial_checkpoint_sha256']
    assert reports['local_control']['start_model_state_sha256']==reports['mixed']['start_model_state_sha256']
    frozen=json.loads((a.root/'contract.json').read_text())
    tasks=json.loads(Path(frozen['training_factors']).read_text())
    if 'initial_identity_binding' in frozen:
        initial=json.loads((a.root/frozen['initial_identity_binding']).read_text())
        expected_initial=initial['checkpoint_sha256']
    else:expected_initial=frozen['initial_checkpoint_sha256']
    for arm in ['local_control','mixed']:
        assert reports[arm]['initial_checkpoint_sha256']==expected_initial
        assert reports[arm]['actual4000_task_SHA']==tasks['files'][arm+'.jsonl']
    raw={arm:json.loads((a.root/arm/'report.json').read_text()) for arm in reports}
    assert raw['local_control']['cache_index_sha256']==raw['mixed']['cache_index_sha256']
    assert raw['local_control']['source_sha256']==raw['mixed']['source_sha256']
    key=lambda group:[(r['scope'],r['budget'],r['query_id'],r['support_id'],r['target']) for r in group]
    assert key(rows['local_control'])==key(rows['mixed'])
    guard_path=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/image_pilot_comparison_1031_022128/decision_contract.json')
    guard=json.loads(guard_path.read_text())['frozen_guard_values'];addendum=json.loads((a.root/'positive_remote_guard_addendum.json').read_text())
    summary=reports['mixed']['summary'];primary=[r for r in rows['mixed'] if r['scope']=='local' and r['budget']=='3+2']
    small=[r for r in primary if r['positive']/r['valid']<.05];fat=[r for r in primary if r['dataset']=='crc' and r['target']==10]
    strata={'small':float(np.mean([r['dice'] for r in small])),'fat':float(np.mean([r['dice'] for r in fat]))}
    checks={'localDice':summary['local|3+2']['dice']>=guard['localDice_floor'],
        'localBF1':summary['local|3+2']['BF1']>=guard['localBF1_floor'],
        'smallDice':strata['small']>=guard['smallDice_floor'],'fatDice':strata['fat']>=guard['fatDice_floor'],
        'remote_absent_FPA':summary['remote|3+2']['absent_FPA']<=guard['remote_absent_FPA_ceiling'],
        'remote_presentDice':summary['remote|3+2']['dice']>=addendum['remote_presentDice_floor'],
        'remote_presentBF1':summary['remote|3+2']['BF1']>=addendum['remote_presentBF1_floor']}
    effects={k:{m:(reports['mixed']['summary'][k][m]-reports['local_control']['summary'][k][m] if reports['mixed']['summary'][k][m] is not None else None) for m in ['dice','BF1','absent_FPA']} for k in summary}
    result=dict(status='complete_accepted_matched_task_diagnostic',summaries={k:v['summary'] for k,v in reports.items()},
        mixed_minus_control=effects,strata=strata,guard_checks=checks,guard_passed=all(checks.values()),
        common_initial_checkpoint_sha256=reports['mixed']['initial_checkpoint_sha256'],
        interpretation='task condition effect under same encoder/head and incremental update budget; source/query input exposure differs by design',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_fullscale_without_second_seed=True,phase_completion=False)
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
