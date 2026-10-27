"""Independently recount matched score cases, scalar intervals and WSI dependencies."""
import argparse
from collections import defaultdict,Counter
import csv
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np

from .audit_heldout_local import read,sha,require,close
from .audit_generic_statistics import quantile


def select(rows,name):
    if name=='overall':return rows
    if name.startswith('same_wsi_distance_'):
        def bin_name(r):
            if r['distance_um']=='':return None
            distance=float(r['distance_um'])
            return 'same_wsi_distance_'+('lt2mm' if distance<2000 else '2to5mm' if distance<5000 else 'ge5mm')
        return [r for r in rows if bin_name(r)==name]
    if '|' in name:
        domain,kind=name.split('|');return [r for r in rows if r['dataset']==domain and r['kind']==kind]
    if name in ('crc','tiger'):return [r for r in rows if r['dataset']==name]
    if name in ('same_wsi','cross_wsi'):return [r for r in rows if r['protocol']==name]
    return [r for r in rows if r['kind']==name]


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid()
    def dump(name,d):
        (a.output/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    dump('status.json',dict(status='preparing',worker_pid=pid))
    try:
        report=read(a.report/'summary.json')
        require(report['status']=='complete_qualified_heldout_statistics' and a.report.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0'),'completed producer')
        require(sha('benchmarks/v5/evidence_closure/report_heldout_statistics.py')==report['source_sha256'],'report source binding')
        for p,v in report['input_acceptance_sha256'].items():require(sha(p)==v,'upstream acceptance binding')
        rows=list(csv.DictReader((a.report/'case_means.csv').open()));require(len(rows)==9664,'full case coverage')
        groups=defaultdict(list);lookup={}
        for r in rows:
            group=tuple(r[k] for k in ('fold','mode','role','method','budget'));groups[group].append(r)
            k=group+(int(r['pair_id']),);require(k not in lookup,'duplicate case CSV');lookup[k]=r
        require(len(groups)==168,'all method/budget groups')
        arithmetic_rows=0
        for identity,evidence in report['evidence'].items():
            fold,mode,role,method=identity.split('|');folder=Path(evidence['folder'])
            require(sha(str(folder/'rows.jsonl'))==evidence['rows_sha256'] and sha(str(folder/'run.json'))==evidence['metadata_sha256'],'actual score/metadata identity')
            raw=defaultdict(list)
            for line in (folder/'rows.jsonl').open():
                r=json.loads(line);raw[r['click_budget'],r['pair_id']].append(r);arithmetic_rows+=1
            for (budget,pair),rr in raw.items():
                require(len(rr)==5 and {r['prompt_group'] for r in rr}==set(range(5)),'five raw point groups')
                target=lookup[fold,mode,role,method,budget,pair];present=rr[0]['positive']>0
                require((target['present']=='True')==present,'presence value')
                require(target['cluster']==rr[0]['space_key' if mode=='local' else 'query_space'],'case cluster identity')
                dice=[2*r['tp']/(2*r['tp']+r['fp']+r['fn']) for r in rr] if present else []
                bf=[r['bf1_um2'] for r in rr if r['bf1_um2'] is not None] if present else []
                fp=[r['fp']/r['valid'] for r in rr] if not present else []
                for metric,values in [('dice',dice),('bf1',bf),('absent_FPA',fp)]:
                    if values:close(float(target[metric]),math.fsum(values)/len(values),'independent raw-to-case mean')
                    else:require(target[metric]=='','undefined case metric')
        require(arithmetic_rows==48320,'all raw score rows recounted')
        draws={};counters={};cluster_ids={}
        for name,info in report['draws'].items():
            clusters=info['clusters'];sample=np.load(info['path'],allow_pickle=False);rng=np.random.default_rng(20261025);strata=[]
            for domain in sorted(set(info['cluster_dataset'].values())):
                ids=np.array([i for i,c in enumerate(clusters) if info['cluster_dataset'][c]==domain]);strata.append(rng.choice(ids,size=(2000,len(ids)),replace=True))
            require(np.array_equal(sample,np.concatenate(strata,axis=1)),'exact prespecified stratified draw sequence')
            draws[name]=sample;counters[name]=[Counter(map(int,r)) for r in sample];cluster_ids[name]={c:i for i,c in enumerate(clusters)}
        verified={};intervals=0;max_error=0.
        with np.load(a.report/'bootstrap_values.npz',allow_pickle=False) as stored:
            for (fold,mode,role,method,budget),rr in groups.items():
                name=fold+'|'+mode;ids=cluster_ids[name]
                for stratum,summary in report['metrics'][name][role+'|'+method][budget].items():
                    subset=select(rr,stratum)
                    require(len(subset)==summary['pairs'] and sum(r['present']=='True' for r in subset)==summary['present_pairs'],'stratum/presence coverage')
                    require(sum(r['partially_annotated_query']=='True' for r in subset)==summary['partially_annotated_pairs'],'partial annotation count')
                    for metric in ('dice','bf1','absent_FPA'):
                        usable=[r for r in subset if r[metric]!=''];m=summary[metric]
                        if not usable:require(m is None,'undefined metric');continue
                        values=defaultdict(list)
                        for r in usable:values[ids[r['cluster']]].append(float(r[metric]))
                        sums={c:math.fsum(v) for c,v in values.items()};counts={c:len(v) for c,v in values.items()}
                        close(math.fsum(sums.values())/len(usable),m['mean'],'independent stratum mean')
                        require(m['pairs']==len(usable) and m['contributing_clusters']==len(values),'metric denominator')
                        samples=[]
                        for draw in counters[name]:
                            denominator=sum(counts.get(c,0)*n for c,n in draw.items())
                            numerator=math.fsum(sums.get(c,0.)*n for c,n in draw.items())
                            samples.append(numerator/denominator if denominator else math.nan)
                        good=[v for v in samples if math.isfinite(v)]
                        require(len(good)==m['valid_draws'],'defined draw count')
                        close(quantile(good,.025),m['conditional_cluster_ci95'][0],'scalar lower CI')
                        close(quantile(good,.975),m['conditional_cluster_ci95'][1],'scalar upper CI');intervals+=1
                        if stratum=='overall':
                            identity='|'.join([fold,mode,role,method,budget,metric]);actual=stored[identity]
                            require(np.array_equal(np.isnan(samples),np.isnan(actual)),'stored bootstrap eligibility')
                            error=max(abs(x-y) for x,y in zip(samples,actual) if math.isfinite(x));require(error<1e-12,'all stored draws independently recounted')
                            max_error=max(max_error,error);verified[identity]=samples
                dump('status.json',dict(status='scalar_interval_recount',worker_pid=pid,fold=fold,mode=mode,overall_series=len(verified),stratified_intervals=intervals))
        require(len(verified)==432 and len(report['contrasts'])==792,'all overall series/contrasts')
        for c in report['contrasts']:
            left='|'.join([c['fold'],c['mode'],c['a_role'],c['a_method'],c['budget'],c['metric']])
            right='|'.join([c['fold'],c['mode'],c['b_role'],c['b_method'],c['budget'],c['metric']])
            delta=[y-x for x,y in zip(verified[left],verified[right]) if math.isfinite(x) and math.isfinite(y)]
            require(len(delta)==c['valid_draws'],'paired valid draw count')
            close(quantile(delta,.025),c['conditional_paired_ci95'][0],'paired lower CI');close(quantile(delta,.975),c['conditional_paired_ci95'][1],'paired upper CI')
            data=report['metrics'][c['fold']+'|'+c['mode']]
            close(data[c['b_role']+'|'+c['b_method']][c['budget']]['overall'][c['metric']]['mean']-data[c['a_role']+'|'+c['a_method']][c['budget']]['overall'][c['metric']]['mean'],c['b_minus_a'],'paired direction')
        components={}
        for fold in ('necrosis','adipose','lymphoid_related'):
            rr=next(v for k,v in groups.items() if k[0]==fold and k[1]=='remote' and k[2]=='heldout' and k[4]=='3+2')
            adjacency=defaultdict(set)
            for r in rr:adjacency[r['source_space']].add(r['query_space']);adjacency[r['query_space']].add(r['source_space'])
            labels={};label=0
            for node in sorted(adjacency):
                if node in labels:continue
                pending=[node];labels[node]=label
                while pending:
                    here=pending.pop()
                    for neighbour in adjacency[here]:
                        if neighbour not in labels:labels[neighbour]=label;pending.append(neighbour)
                label+=1
            sizes=sorted(Counter(labels[r['source_space']] for r in rr).values());declared=report['dependencies'][fold]
            require(sizes==declared['component_pair_counts'] and label==declared['components'],'independent BFS support-query dependencies')
            components[fold]=sizes
        result=dict(status='passed_independent_heldout_statistics',worker_pid=pid,arithmetic_score_rows=arithmetic_rows,case_rows=9664,
            overall_series=len(verified),draws_per_series=2000,stratified_intervals=intervals,paired_contrasts=792,components=components,
            maximum_scalar_draw_error=max_error,report_sha256=sha(str(a.report/'summary.json')),source_sha256=sha(__file__),
            scope='Numerical and dependency audit only; all single-seed, conditional support/query, annotation and noncausal-reference qualifications remain')
        dump('summary.json',result);dump('status.json',dict(status='complete',worker_pid=pid));print(json.dumps(result),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
