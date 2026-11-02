"""Remote subgroup safety tables; P2 use is diagnostic, P3 rules freeze before runs."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize_remote(rows):
    remote=[dict(r['plan'],**r['metrics']) for r in rows if r['scope']=='remote' and r['budget']=='3+2']
    present=[r for r in remote if r['positive']>0];absent=[r for r in remote if r['positive']==0]
    def values(group,key):
        x=[r[key] for r in group if r[key] is not None]
        return dict(n=len(group),mean=float(np.mean(x)) if x else None)
    result=dict(small=values([r for r in present if r['positive']/r['valid']<.05],'dice'),
                fat=values([r for r in present if r['dataset']=='crc' and r['target']==10],'dice'))
    result['per_domain']={d:values([r for r in absent if r['dataset']==d],'false_positive_fraction') for d in ['crc','tiger']}
    result['per_target']={f'{d}|{t}':values([r for r in absent if r['dataset']==d and r['target']==t],'false_positive_fraction') for d,t in sorted({(r['dataset'],r['target']) for r in absent})}
    fractions=[r['false_positive_fraction'] for r in absent]
    result['absent_tail']=dict(n=len(absent),quantiles95_99=np.quantile(fractions,[.95,.99]).tolist(),fraction_tasks_above_1pct=float(np.mean(np.array(fractions)>.01)),fraction_tasks_above_10pct=float(np.mean(np.array(fractions)>.1)))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False)
    results={m:summarize_remote(json.loads((a.run/m/'rows.json').read_text())) for m in ['continuous','fixed','ema']}
    assert all(r['small']['n']==222 and r['fat']['n']==34 and r['absent_tail']['n']==465 for r in results.values())
    (a.output/'report.json').write_text(json.dumps(dict(status='complete_remote_diagnostic_tables',results=results,changes_P2_decision=False),indent=2)+'\n')


if __name__=='__main__':main()
