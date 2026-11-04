"""Strict serial dependency chain: engineering -> learning -> smoke -> formal."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmarks.v5.architecture_bridge.run_spatial_study import sha, read


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;s=read(root/'settings.json')
    def status(**value):
        value.update(observed_at=time.time(),controller_pid=os.getpid())
        temporary=root/'stage.tmp';temporary.write_text(json.dumps(value));temporary.replace(root/'stage.json')
        with (root/'history.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    def command(label,module,args,gpu=True):
        for path,h in s['sources'].items():assert sha(path)==h,path
        if gpu:
            while True:
                free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free>=s['minimum_free_memory_MiB']:break
                status(status='waiting_GPU_memory',stage=label,free_MiB=free,required_MiB=s['minimum_free_memory_MiB']);time.sleep(10)
        cmd=[sys.executable,'-u','-m',module,*map(str,args)]
        with (root/(label+'.log')).open('w') as log:
            worker=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            status(status='running',stage=label,worker_pid=worker.pid,command=cmd)
            rc=worker.wait();log.write('\nEXIT_CODE='+str(rc)+'\n');assert rc==0,(label,rc)
    try:
        assert not (root/'history.jsonl').exists(),'Never restart an existing study automatically'
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
        for path,h in s['weights'].items():assert sha(path)==h,path
        protocol=read(Path(s['protocol'])/'acceptance.json');cache=read(Path(s['cache'])/'acceptance.json')
        assert protocol['status']=='accepted_spatial_task_protocol_only' and cache['status']=='accepted_repaired_normalized_cells'
        assert sha(Path(s['protocol'])/'training_consumption.jsonl')==protocol['consumption_sha256']
        assert sha(Path(s['protocol'])/'evaluation.jsonl')==protocol['evaluation_sha256']
        assert sha(Path(s['cache'])/'index.jsonl')==cache['index_sha256']
        assert sha(Path(s['cache'])/'transforms.npz')==cache['transforms_sha256']
        if s.get('engineering_reuse'):
            previous=Path(s['engineering_reuse']);old=read(previous/'settings.json')
            for key in ['cache','protocol','seed','arms','weights']:assert s[key]==old[key],key
            for path,h in old['sources'].items():
                if path!='benchmarks/v5/support_match_replication/pipeline.py':assert s['sources'][path]==h,path
            proof=read(previous/'engineering/acceptance.json')
            assert proof['status']=='accepted_support_match_engineering'
            assert proof['source_sha256']==sha('benchmarks/v5/support_match_replication/gate.py')
            (root/'engineering').symlink_to(previous/'engineering',target_is_directory=True)
            status(status='accepted_engineering_reuse',source=str(previous/'engineering'))
        else:
            command('engineering','benchmarks.v5.support_match_replication.gate',['--root',root])
        engineering=read(root/'engineering/acceptance.json');assert engineering['status']=='accepted_support_match_engineering' and engineering['seed']==s['seed']
        diagnostic_args=['--study',s['sam_study'],'--output',root/'sam_diagnostic']
        command('sam_diagnostic','benchmarks.v5.support_match_replication.diagnose_sam',diagnostic_args)
        command('sam_diagnostic_accept','benchmarks.v5.support_match_replication.diagnose_sam',diagnostic_args+['--accept'],False)
        assert read(root/'sam_diagnostic/acceptance.json')['status']=='accepted_frozen_SAM_train_diagnostic'
        (root/'learning').mkdir()
        for arm in s['arms']:
            folder=root/'learning'/arm
            command(arm+'_learning','benchmarks.v5.support_match_replication.functional',['--cache',s['cache'],'--output',folder,'--arm',arm,'--seed',s['seed']])
            command(arm+'_learning_accept','benchmarks.v5.architecture_bridge.accept_spatial_learning',['--cache',s['cache'],'--run',folder],False)
            assert read(folder/'independent_acceptance.json')['status']=='accepted_P3_spatial_train_learning_gate'
        for arm in s['arms']:
            args=['--arm',arm,'--cache',s['cache'],'--protocol',s['protocol'],'--learning',root/'learning','--output',root/(arm+'_smoke'),'--smoke','--seed',s['seed']]
            command(arm+'_smoke','benchmarks.v5.support_match_replication.train',args)
            command(arm+'_smoke_accept','benchmarks.v5.support_match_replication.train',args+['--accept'],False)
            assert read(root/(arm+'_smoke')/'independent_acceptance.json')['smoke']
        evidence={}
        for arm in s['arms']:
            for f in [root/'engineering'/(arm+'_benchmark.json'),root/'learning'/arm/'independent_acceptance.json',root/(arm+'_smoke')/'independent_acceptance.json']:
                evidence[str(f)]=sha(f)
        execution=dict(updates=1000,batch=4,seed=s['seed'],arms=s['arms'],sources=s['sources'],weights=s['weights'],
                       initial_common_sha256=engineering['common_initial_sha256'],old_formal=s['old_formal'],
                       protocol_sha256=sha(Path(s['protocol'])/'acceptance.json'),consumption_sha256=s['pilot_consumption_sha256'],
                       evidence=evidence,selection='fixed final; no dev threshold/checkpoint selection',new_formal_runs=2)
        for arm in s['arms']:
            assert read(root/(arm+'_smoke')/'report.json')['initial_common_sha256']==execution['initial_common_sha256']
        assert not (root/'contract.json').exists();(root/'contract.json').write_text(json.dumps(execution,indent=2)+'\n')
        for arm in s['arms']:
            args=['--arm',arm,'--cache',s['cache'],'--protocol',s['protocol'],'--learning',root/'learning','--inventory',s['inventory'],
                  '--output',root/arm,'--execution-contract',root/'contract.json','--benchmark-receipt',root/'engineering'/(arm+'_benchmark.json'),'--monitor-receipt',s['monitor'],'--seed',s['seed']]
            command(arm+'_formal','benchmarks.v5.support_match_replication.train',args)
            command(arm+'_formal_accept','benchmarks.v5.support_match_replication.train',args+['--accept'],False)
        status(status='complete_runs',P3_complete=False)
        # Comparator requires this terminal training state and all independent receipts.
        cmd=[sys.executable,'-u','-m','benchmarks.v5.support_match_replication.compare','--root',str(root)]
        with (root/'comparison.log').open('w') as log:
            rc=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT);log.write('\nEXIT_CODE='+str(rc)+'\n');assert rc==0,('comparison',rc)
        status(status='complete_accepted',P3_complete=False,next_action='report to user; no automatic next experiment')
    except Exception as exc:
        status(status='failed',error=repr(exc),retry=False);raise


if __name__=='__main__':main()
