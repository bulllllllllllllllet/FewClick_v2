"""Wait for actual accepted startup evidence, then run the fixed first-seed study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ORDER=['S11','I','S00','H11','S10','S01']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    # Observation retries only: a non-atomic status writer may briefly expose empty JSON.
    for attempt in range(10):
        try:return json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            if attempt==9:raise
            time.sleep(.1)


def validate(settings):
    evidence={}
    def accepted(path,status):
        value=read(path);assert value['status']==status,(path,value['status']);evidence[str(path)]=sha(path);return value
    for path,digest in settings['sources'].items():assert sha(path)==digest,path
    for path,digest in settings['weights'].items():assert sha(path)==digest,path
    protocol=accepted(Path(settings['protocol'])/'acceptance.json','accepted_spatial_task_protocol_only')
    cache=accepted(Path(settings['cache'])/'acceptance.json','accepted_repaired_normalized_cells')
    assert sha(Path(settings['cache'])/'index.jsonl')==cache['index_sha256']
    assert sha(Path(settings['cache'])/'transforms.npz')==cache['transforms_sha256']
    assert protocol['input_identity']==dict(repaired_index_sha256=cache['index_sha256'],transforms_sha256=cache['transforms_sha256'])
    assert sha(Path(settings['protocol'])/'training_consumption.jsonl')==protocol['consumption_sha256']
    assert sha(Path(settings['protocol'])/'evaluation.jsonl')==protocol['evaluation_sha256']
    inventory=accepted(settings['inventory'],'accepted_full_spatial_inventory')
    assert inventory['fields']==1737 and inventory['views']==5211 and inventory['all_image_views_pixel_exact'] and inventory['physical_extents_checked']
    assert inventory['identity']['index_sha256']==cache['index_sha256']
    assert sha(Path(settings['inventory']).parent/'frames.jsonl')==inventory['frames_sha256']
    monitor=accepted(settings['monitor'],'accepted_fixed_train_monitor_integration')
    assert monitor['rows']==32 and monitor['RNG_unchanged'] and monitor['model_state_unchanged'] and monitor['mode_restored']
    assert sha(Path(settings['monitor']).parent/'monitor.json')==monitor['monitor_sha256']
    benchmark=accepted(settings['benchmark'],'accepted_S11_GPU_throughput_probe')
    assert benchmark['updates']==20 and benchmark['inference_tasks']==100 and benchmark['visible_devices']=='3'
    assert len(benchmark['train_seconds'])==20 and len(benchmark['inference_seconds'])==100
    assert benchmark['protocol_sha256']==sha(Path(settings['protocol'])/'acceptance.json')
    initials=set();tasks=None
    for arm in ORDER:
        folder=Path(settings['learning'])/arm;receipt=accepted(folder/'independent_acceptance.json','accepted_P3_spatial_train_learning_gate')
        assert receipt['arm']==arm and receipt['seed']==20261103 and receipt['masks_replayed']==32
        for filename,key in [('report.json','report_sha256'),('masks.npz','masks_sha256'),('checkpoint.pt','checkpoint_sha256'),('contract.json','contract_sha256')]:assert sha(folder/filename)==receipt[key]
        report=read(folder/'report.json');assert report['exact_optimizer_continuation'] and report['fresh_final_predictions_exact']
        tasklist=read(folder/'contract.json')['tasks']
        if tasks is None:tasks=tasklist
        assert tasks==tasklist and all(x['passed'] for x in report['groups'].values())
        smoke=Path(settings['smokes'])/(arm+'_smoke');receipt=accepted(smoke/'independent_acceptance.json','accepted_P3_spatial_evaluation')
        assert receipt['arm']==arm and receipt['smoke'] and receipt['seed']==20261103 and receipt['rows']==12
        for filename,key in [('report.json','report_sha256'),('masks.npz','masks_sha256'),('checkpoint.pt','checkpoint_sha256'),('rows.json','rows_sha256')]:assert sha(smoke/filename)==receipt[key]
        report=read(smoke/'report.json');assert report['device']=='cuda:0' and report['source_sha256']==settings['sources']['benchmarks/v5/architecture_bridge/train_spatial.py']
        assert report['train_monitor_nodes']==[0] and report['fresh_prediction_replay_count']==12
        assert sha(smoke/'train_monitor_0.json')==report['train_monitor_sha256']['0']
        assert report['inputs']==protocol['input_identity'] and report['protocol_sha256']==sha(Path(settings['protocol'])/'acceptance.json')
        initials.add(report['initial_state_sha256'])
    assert len(initials)==1
    return dict(updates=2000,batch=4,seed=20261103,arms=ORDER,protocol_sha256=sha(Path(settings['protocol'])/'acceptance.json'),
                consumption_sha256=protocol['consumption_sha256'],sources=settings['sources'],weights=settings['weights'],evidence=evidence,
                initial_state_sha256=initials.pop(),formal_budget_runs=6,model_selection='fixed final; no dev threshold/checkpoint selection')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--audit-only',action='store_true');a=p.parse_args();root=a.root;settings=read(root/'settings.json')
    def status(**value):
        value['observed_at']=time.time();(root/'stage.json').write_text(json.dumps(value))
        with (root/'history.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    try:
        while True:
            state=read(Path(settings['dependency_root'])/'stage.json')
            if state['status']=='complete':break
            assert state['status'] not in ['failed','stopped_dependency_failed'],state
            os.kill(settings['dependency_controller'],0)
            status(status='waiting_verified_dependency',controller=settings['dependency_controller'],dependency=state)
            if a.audit_only:return
            time.sleep(10)
        import transformers
        assert transformers.__version__=='4.49.0' and os.environ.get('CUDA_VISIBLE_DEVICES')=='3' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        execution=validate(settings)
        if a.audit_only:status(status='accepted_readiness_audit_only');return
        assert not (root/'contract.json').exists(),'Never overwrite an existing execution contract'
        (root/'contract.json').write_text(json.dumps(execution,indent=2)+'\n')
        for arm in ORDER:(root/(arm+'_smoke')).symlink_to(Path(settings['smokes'])/(arm+'_smoke'),target_is_directory=True)
        for arm in ORDER:
            cmd=[sys.executable,'-u','-m','benchmarks.v5.architecture_bridge.train_spatial','--arm',arm,'--cache',settings['cache'],
                 '--protocol',settings['protocol'],'--learning',settings['learning'],'--inventory',settings['inventory'],
                 '--output',str(root/arm),'--execution-contract',str(root/'contract.json'),'--benchmark-receipt',settings['benchmark'],'--monitor-receipt',settings['monitor']]
            for label,command in [(arm+'_train',cmd),(arm+'_accept',cmd+['--accept'])]:
                for path,digest in settings['sources'].items():assert sha(path)==digest,path
                if label.endswith('_train'):
                    uuid=subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
                    while True:
                        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
                        busy=[line for line in apps if line.split(',')[0].strip()==uuid]
                        if not busy:break
                        status(status='waiting_GPU3_occupied',processes=busy);time.sleep(10)
                with (root/(label+'.log')).open('w') as log:
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);status(status='running',stage=label,worker_pid=process.pid,command=command)
                    rc=process.wait();log.write('\nEXIT_CODE='+str(rc)+'\n');assert rc==0,(label,rc)
        status(status='complete',comparison_pending=True,P3_complete=False)
        with (root/'comparison.log').open('w') as log:
            rc=subprocess.call([sys.executable,'-m','benchmarks.v5.architecture_bridge.compare_spatial','--run',str(root),'--output',str(root/'comparison')],stdout=log,stderr=subprocess.STDOUT)
            assert rc==0,('comparison',rc)
        status(status='complete',comparison_pending=False,P3_complete=False)
    except Exception as exc:status(status='failed',error=repr(exc),retry=False);raise


if __name__=='__main__':main()
