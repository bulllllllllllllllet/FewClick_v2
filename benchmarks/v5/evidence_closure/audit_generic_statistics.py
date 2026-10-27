"""Recount saved case means and every bootstrap draw using scalar sums."""
import argparse
from collections import Counter,defaultdict
import csv
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .audit_heldout_local import read,sha,require,close


def quantile(values,p):
    ordered=sorted(values);h=(len(ordered)-1)*p;i=math.floor(h);j=math.ceil(h)
    return ordered[i]+(ordered[j]-ordered[i])*(h-i)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid()
    def dump(name,d):
        (a.output/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    try:
        while True:
            try:state=read(a.report/'status.json')
            except json.JSONDecodeError:time.sleep(1);continue
            if state['status']=='complete_qualified_generic_cell_statistics':
                log=a.report.with_suffix('.log').read_text().rstrip()
                if log.endswith('EXIT_CODE=0'):break
                require('EXIT_CODE=' not in log.splitlines()[-1],'report nonzero exit');time.sleep(1);continue
            require(state['status']!='failed','report failed')
            proc=Path(f'/proc/{state["worker_pid"]}/cmdline')
            require(proc.exists() and b'report_generic_cells' in proc.read_bytes(),'specific report worker not live')
            dump('status.json',dict(status='waiting_live_report',worker_pid=pid,producer_pid=state['worker_pid'],observed_at=time.time()));time.sleep(5)
        report=read(a.report/'summary.json')
        require(sha('benchmarks/v5/evidence_closure/report_generic_cells.py')==report['source_sha256'],'report source SHA')
        for p,v in report['upstream_acceptance_sha256'].items():require(sha(p)==v,'upstream acceptance SHA')
        cluster_ids={k:i for i,k in enumerate(report['clusters'])}
        indices=np.load(a.report/'bootstrap_cluster_indices.npy',allow_pickle=False)
        require(indices.shape==(2000,75) and np.issubdtype(indices.dtype,np.integer) and indices.min()==0 and indices.max()==74,'bootstrap draw shape/range')
        multiplicity=[Counter(map(int,row)) for row in indices]
        rows=list(csv.DictReader((a.report/'case_means.csv').open()))
        require(len(rows)==6040,'five methods/four budgets/302 cases')
        groups=defaultdict(list)
        for r in rows:groups[r['method'],r['budget']].append(r)
        require(len(groups)==20,'all fixed method/budget groups')
        verified={};max_error=0.
        with np.load(a.report/'bootstrap_values.npz',allow_pickle=False) as stored:
            for (method,budget),rr in groups.items():
                require(len(rr)==302 and len({r['pair_id'] for r in rr})==302,'unique cases')
                for metric in ('dice','bf1'):
                    values=defaultdict(list)
                    for r in rr:
                        if r[metric]!='':values[cluster_ids[r['space_key']]].append(float(r[metric]))
                    sums={c:math.fsum(v) for c,v in values.items()};counts={c:len(v) for c,v in values.items()}
                    average=math.fsum(sums.values())/sum(counts.values())
                    m=report['metrics'][method][budget][metric];close(average,m['mean'],'case macro mean')
                    require(sum(counts.values())==m['pairs'],'metric denominator')
                    samples=[]
                    for draw in multiplicity:
                        numerator=math.fsum(sums.get(c,0.)*n for c,n in draw.items())
                        denominator=sum(counts.get(c,0)*n for c,n in draw.items())
                        require(denominator>0,'defined draw');samples.append(numerator/denominator)
                    k=method+'|'+budget+'|'+metric
                    error=max(abs(x-y) for x,y in zip(samples,stored[k]));require(error<1e-12,'all2000 scalar bootstrap draws')
                    max_error=max(max_error,error);verified[k]=samples
                    close(quantile(samples,.025),m['ci95'][0],'lower percentile');close(quantile(samples,.975),m['ci95'][1],'upper percentile')
        require(len(verified)==40 and len(report['contrasts'])==80,'all metrics and pairwise contrasts')
        for c in report['contrasts']:
            suffix='|'+c['budget']+'|'+c['metric'];a_values=verified[c['a']+suffix];b_values=verified[c['b']+suffix]
            delta=[b-a for a,b in zip(a_values,b_values)]
            close(quantile(delta,.025),c['ci95'][0],'paired lower percentile');close(quantile(delta,.975),c['ci95'][1],'paired upper percentile')
            close(report['metrics'][c['b']][c['budget']][c['metric']]['mean']-report['metrics'][c['a']][c['budget']][c['metric']]['mean'],c['b_minus_a'],'contrast direction')
        dump('summary.json',dict(status='passed_independent_generic_statistics',worker_pid=pid,case_rows=6040,
            bootstrap_series=40,draws_per_series=2000,contrasts=80,maximum_scalar_recount_error=max_error,
            report_sha256=sha(str(a.report/'summary.json')),source_sha256=sha(__file__),
            scope='Numerical statistics acceptance; all original single-seed/development/prior/dependence qualifications remain'))
        dump('status.json',dict(status='complete',worker_pid=pid));print('Passed scalar recount: 80000 draws and80 contrasts',flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
