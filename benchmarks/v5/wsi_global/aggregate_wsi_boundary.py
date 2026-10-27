"""Bind independently accepted BF1 shards and join every original task row."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(test,why):
    if not test:raise ValueError(why)


def run(a):
    a.output.mkdir(exist_ok=False,parents=True)
    require(len(a.sources)==len(a.acceptances)==(1 if a.gate else 4),'all one-fixture or four formal disjoint BF1 shards')
    prediction=read(a.predictions/'summary.json');source_rows=read(a.predictions/'rows.json');pred_receipt=read(a.prediction_acceptance/'summary.json')
    require(sha(a.predictions/'rows.json')==prediction['rows_sha256'] and pred_receipt['source_summary_sha256']==sha(a.predictions/'summary.json'),'accepted complete native prediction task rows')
    seen={};ranks=[];evidence=[];signature=None
    for source,acceptance in zip(a.sources,a.acceptances):
        s=read(source/'summary.json');g=read(acceptance/'summary.json');contract=read(source/'contract.json');rows=read(source/'rows.json');checks=read(acceptance/'checks.json')
        require(s['status']==('passed_WSI_boundary_driver_fixture' if a.gate else 'complete_WSI_boundary_shard_unaccepted') and g['status']==('accepted_WSI_boundary_fixture' if a.gate else 'accepted_WSI_boundary_shard'),'all actual BF1 producer and independent results complete')
        require(g['source_summary_sha256']==sha(source/'summary.json') and g['checks_sha256']==sha(acceptance/'checks.json') and s['rows_sha256']==sha(source/'rows.json'),'every independent/source output SHA')
        require(g['recipe']['producer_signature']==s['signature'] and g['recipe']['source_sha256']==sha('benchmarks/v5/wsi_global/accept_wsi_boundary.py'),'same independently accepted metric/auditor')
        require(signature is None or signature==s['signature'],'same exact BF1 definition and originalGT grid');signature=s['signature']
        require(contract['source_summary_sha256']==g['prediction_summary_sha256']==sha(a.predictions/'summary.json') and contract['acceptance_sha256']==sha(a.prediction_acceptance/'summary.json'),'all shards use same accepted underlying prediction source')
        require(s['source_rows']==len(source_rows) and s['shards']==len(a.sources) and len(rows)==len(checks)==s['rows'],'every assigned row and independent check')
        ranks.append(s['rank'])
        for row,check in zip(rows,checks):
            index=row['source_row_index'];require(index not in seen and index==check['source_row_index'] and index%s['shards']==s['rank'],'unique real fixed shard membership')
            require(all(row['physical_BF1'][k]==v for k,v in check.items() if k!='source_row_index'),'independent complete scalar/count confirmation')
            seen[index]=row['physical_BF1']
        evidence.append(dict(source=str(source),source_summary_sha256=sha(source/'summary.json'),acceptance=str(acceptance),acceptance_summary_sha256=sha(acceptance/'summary.json'),worker_pid=s['worker_pid'],auditor_pid=g['worker_pid']))
    require(sorted(ranks)==list(range(len(a.sources))) and set(seen)==set(range(len(source_rows))),'complete row/rank coverage without omissions or repetitions')
    joined=[dict(row,physical_BF1=seen[i]) for i,row in enumerate(source_rows)]
    (a.output/'rows.json').write_text(json.dumps(joined,indent=2,allow_nan=False)+'\n')
    result=dict(status='accepted_WSI_boundary_aggregation_fixture' if a.gate else 'accepted_complete_WSI_physical_BF1_rows',worker_pid=os.getpid(),signature=signature,rows=len(joined),tissue_rows=sum(r['target']!=2 for r in joined),background_rows=sum(r['target']==2 for r in joined),boundary_evaluable=sum(r['physical_BF1']['boundary_evaluable'] for r in joined),GT_boundary_NA=sum(not r['physical_BF1']['boundary_evaluable'] for r in joined),rows_sha256=sha(a.output/'rows.json'),prediction_summary_sha256=sha(a.predictions/'summary.json'),source_sha256=sha(__file__),evidence=evidence,
                scope='All accepted original prediction/GT rows joined with independently recounted physical BF1; row counts are not independent patient/sample or prompt repetitions',remaining='Paired WSI/target uncertainty and complete methods including full-cell predictions; no overall goal completion')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,nargs='+',required=True);p.add_argument('--acceptances',type=Path,nargs='+',required=True)
    for n in ('predictions','prediction-acceptance','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gate',action='store_true');run(p.parse_args())
