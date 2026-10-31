"""Fixed Phikon comparison after both full independent acceptances."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

from .accept_continuous_pilot import summarize
from .cell_catalog import load_cell_catalog
from .continuous_data import TASKS, sha
from .compare_continuous_pilots import summary_values, resource_guards, paired_interval


BOUNDS=dict(primary_Dice_gain=.01,remote_BF1_max_drop=.01,absent_FPA_max_increase=.01,
            local_Dice_max_drop=.02,small_Dice_max_drop=.02,fat_Dice_max_drop=.02)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-gate',action='store_true'); args=p.parse_args()
    assert json.loads((args.run/'stage.json').read_text())['status']=='complete'
    receipts={}; reports={}; rows={}; summaries={}; evidence={}
    for arm in ['frozen','joint']:
        d=args.run/arm
        receipt=json.loads((d/'independent_acceptance.json').read_text())
        report=json.loads((d/'report.json').read_text())
        assert receipt['status']==('accepted_joint_training_gate' if args.verify_gate else 'accepted_joint_pilot')
        assert receipt['arm']==report['arm']==arm and report['gate']==args.verify_gate
        assert receipt['report_sha256']==sha(d/'report.json')
        assert receipt['checkpoint_sha256']==report['checkpoint_sha256']==sha(d/'checkpoint.pt')
        assert receipt['sources']==report['sources'] and receipt['inputs']==report['inputs']
        for name,value in receipt['sources'].items(): assert sha(Path(__file__).with_name(name))==value
        if not args.verify_gate: assert sha(d/'training_consumed.jsonl')==receipt['training_sha256']
        flat=[]
        for item in json.loads((d/'rows.json').read_text()):
            flat.append(dict(item['plan'],scope=item['scope'],budget=item['budget'],**item['metrics']))
        assert len(flat)==receipt['rows_replayed']==(20 if args.verify_gate else 3711)
        groups=defaultdict(list)
        for item in flat: groups[item['scope']+'|'+item['budget']].append(item)
        local=groups['local|3+2']
        summaries[arm]=dict(summary={key:summarize(value) for key,value in groups.items()},
                            local_strata=dict(small=summarize([r for r in local if r['positive']/r['valid']<.05]),
                                              fat=summarize([r for r in local if r['dataset']=='crc' and r['target']==10])))
        receipts[arm]=receipt; reports[arm]=report; rows[arm]=flat
        evidence[arm]={name:sha(d/name) for name in ['report.json','independent_acceptance.json','rows.json','packed_masks.npz','checkpoint.pt','training_consumed.jsonl']}
    for key in ['sources','inputs','initial_state','initial_logits_sha256','training_sha256']:
        assert receipts['frozen'][key]==receipts['joint'][key]
    for a,b in zip(rows['frozen'],rows['joint']):
        for key in ['scope','budget','dataset','target','support_id','query_id','kind','positive_xy','negative_xy','positive','valid']:
            assert a[key]==b[key]
    result=dict(status='accepted_joint_comparison_gate' if args.verify_gate else 'complete_joint_comparison',
                worker_pid=os.getpid(),source_sha256=sha(__file__),evidence=evidence,summaries=summaries,
                bounds=BOUNDS,eligible_for_second_seed=False,fullWSI_promoted=False,
                limitations=['exposed development data; not independent confirmation','fixed support bank, reused queries and same-patient dependence remain','one seed; gated recipe and matched updates do not imply equal FLOPs','TIGER3 lacks present tasks; keep per-target counts and NA'])
    if not args.verify_gate:
        values={arm:summary_values(summaries[arm]) for arm in summaries}
        decision=resource_guards(values['frozen'],values['joint'],BOUNDS)
        _,inventory=load_cell_catalog(TASKS)
        result.update(values=values,comparison=decision,eligible_for_second_seed=decision['pass_all'],
                      conditional_intervals=dict(
                          remote_macro_Dice=paired_interval(rows['frozen'],rows['joint'],inventory,'remote','dice',True,macro=True),
                          absent_FPA=paired_interval(rows['frozen'],rows['joint'],inventory,'remote','false_positive_fraction',False)))
    args.output.mkdir(exist_ok=False)
    (args.output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],eligible_for_second_seed=result['eligible_for_second_seed'])),flush=True)


if __name__=='__main__': main()
