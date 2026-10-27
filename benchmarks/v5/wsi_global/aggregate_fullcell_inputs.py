"""Accept exact fullcell input-shard coverage and all actual zero-cell records."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(test,message):
    if not test:raise ValueError(message)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);indices=set();checks=[];zero=[];total=retained=0;signature=None;sources=[]
    for path in a.audits:
        summary=read(path/'summary.json');contract=read(path/'contract.json');entries=read(path/'entries.json')
        require(summary['status']==('passed_actual_fullcell_input_gate' if a.gate else 'accepted_fullcell_input_shard'),'actual completed input audit')
        require(sha(path/'entries.json')==summary['entries_sha256'] and sha(path/'input_snapshot.json')==summary['input_snapshot_sha256'],'all actual audit output SHA')
        require(signature is None or signature==summary['signature'],'same producer/actual auditor source');signature=summary['signature']
        require(summary['tiles']==len(entries)==len(contract['tiles']),'audit row count')
        for entry in entries:
            tile=entry['tile'];require(tile not in indices and tile in contract['tiles'],'all tile indices unique and assigned')
            if not a.gate:require(tile%summary['shards']==summary['rank'],'fixed audit shard membership')
            require(len(entry['views'])==3 and {v['scale'] for v in entry['views']}=={0,1,2},'all three scales exactly once');indices.add(tile)
            for v in entry['views']:
                require(v['features_exact'] and v['retained']==min(255,v['total']),'actual independently audited cell geometry and counts')
                total+=v['total'];retained+=v['retained']
                if not v['total']:zero.append(dict(tile=tile,scale=v['scale']))
        require(sum(v['total'] for e in entries for v in e['views'])==summary['sum_true_instances_across_views'] and sum(v['retained'] for e in entries for v in e['views'])==summary['sum_retained_across_views'],'local true and retained sums')
        sources.append(dict(path=str(path),summary_sha256=sha(path/'summary.json'),worker_pid=summary['worker_pid']))
    require(signature['producer_contract_sha256']==sha(a.source/'contract.json'),'actual producer bound')
    key=lambda z:(z['tile'],z['scale']);zero=sorted(zero,key=key)
    if a.gate:
        require(indices=={e['tile'] for e in read(a.audits[0]/'input_snapshot.json')},'all actual gate indices')
    else:
        done=read(a.source/'summary.json');require(done['status']=='complete_fullcell_GPU_WSI_execution' and indices==set(range(3050)),'complete3050 not subset')
        require(len(a.audits)==4 and {read(p/'summary.json')['rank'] for p in a.audits}=={0,1,2,3} and all(read(p/'summary.json')['shards']==4 for p in a.audits),'all4 fixed CPU audit partitions')
        require(zero==sorted(done['zero_cell_cases'],key=key),'every actual zero-cell view independently confirmed')
    result=dict(status='accepted_actual_fullcell_input_gate_aggregate' if a.gate else 'accepted_all3050_fullcell_geometry_and_true_counts',worker_pid=os.getpid(),signature=signature,tiles=len(indices),views=3*len(indices),zero_cell_cases=zero,sum_true_instances_across_overlapping_views=total,sum_retained_across_views=retained,audit_sources=sources,source_sha256=sha(__file__),remaining='Native encoder/zero-cell density and complete probability/cache/output acceptance separate; TLS semantics unconfirmed')
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=result['status'],tiles=len(indices),views=result['views'],zero_views=len(zero))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--audits',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',action='store_true');run(p.parse_args())
