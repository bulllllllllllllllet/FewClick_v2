#!/usr/bin/env python3
"""Read paired-run masks/metrics and locate task-level cross-prompt failures."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v5.phase_5_prompt_encoder.src.tissue_error_analysis import aggregate_task_errors, paired_delta, seed_task_mean

CLASS_NAMES = ('tumor_epithelium','tumor_stroma','background','necrosis','normal_gland','normal_stroma',
               'submucosa_serosa','muscle','lymphocyte_aggregate','mucus','fat','blood')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(line) for line in (a.run/'episode_metrics.jsonl').read_text().splitlines()]
    report=dict(run=str(a.run),metric_rows=len(rows),methods={})
    for method in ('phase5','single','multi'):
        method_rows=[x for x in rows if x['method']==method and x['protocol']=='cross']
        by_arm={arm:aggregate_task_errors([x for x in method_rows if x['arm']==arm]) for arm in ('local','cross')}
        delta=paired_delta(by_arm['local'],by_arm['cross'])
        for task, value in delta.items(): value['class_name']=CLASS_NAMES[int(task)]
        seed={arm:seed_task_mean([x for x in method_rows if x['arm']==arm]) for arm in ('local','cross')}
        ranked=sorted(delta.values(),key=lambda x: x['dice_delta'] if x['dice_delta'] is not None else 0)
        report['methods'][method]=dict(tasks=delta,seed_tasks=seed,most_harmed=ranked[:3],most_improved=ranked[-3:])
    (a.run/'error_analysis.json').write_text(json.dumps(report,indent=2))
    for method, data in report['methods'].items():
        print(method)
        for row in data['most_harmed']+data['most_improved']:
            print(json.dumps(dict(class_name=row['class_name'],dice_delta=row['dice_delta'],fnr_delta=row['fnr_delta'],fpr_delta=row['fpr_delta'])))

if __name__=='__main__': main()
