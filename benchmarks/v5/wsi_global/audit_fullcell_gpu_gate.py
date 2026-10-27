"""Fresh selected model replay and independent native cell geometry/count audit."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import traceback

import pyvips
import numpy as np
from PIL import Image
from scipy import ndimage
import torch

from benchmarks.v5.evidence_closure.audit_heldout_local import read,sha,require,close
from benchmarks.v5.evidence_closure.audit_generic_statistics import quantile
from benchmarks.v5.evidence_closure.audit_full_wsi import tree_error
from .fullcell_gpu_gate import transfer
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.external_evaluation.wsi_reader import StrictSlide
from benchmarks.v5.fair_comparison.local_data import LocalData,fewclick_inputs
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid();torch.set_num_threads(4)
    def dump(name,d):
        (a.output/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    dump('status.json',dict(status='loading',worker_pid=pid))
    try:
        done=read(a.gate/'summary.json');meta=read(a.gate/'run.json')
        require(done['status']=='passed_real_fullcell_GPU_cache_and_click_gate' and a.gate.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0'),'completed real gate')
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='5','only allowed GPU5')
        for p,v in meta['source_sha256'].items():require(sha(p)==v,'same native gate source')
        training=read(a.training/'best_checkpoint.json');require(sha(training['path'])==meta['checkpoint_sha256']==training['checkpoint_sha256'],'selected model SHA')
        device=torch.device('cuda:0');model=load('fewclick_m3d_xcell',Path(training['path']),device,meta['development_protocol_sha256'])
        data=LocalData(a.development);maximum=next(p for p in read(a.development/'validation_maximum.json') if p['dataset']=='crc' and p['prompt_group']==0)
        raw,_,_=data.materialize(nested(maximum,'3+2'),device,True,True)
        with torch.inference_mode():source=model.core.encode_image(fewclick_inputs(raw))
        record=read(a.plan/'slides_private.json')[0];slide=StrictSlide(record['provenance']['wsi_path']);checks=[];geometry_checks=[]
        try:
            for q in done['queries']:
                require(sha(q['encoded_path'])==q['encoded_sha256'],'saved encoded cache SHA')
                payload=torch.load(q['encoded_path'],map_location='cpu',weights_only=False);cached=transfer(payload['encoded'],device)
                require(payload['checkpoint_sha256']==meta['checkpoint_sha256'],'encoded selected model binding')
                raw={};totals=[]
                for cell,box in zip(q['cell_records'],payload['geometry']['native_boxes']):
                    require(sha(cell['path'])==cell['sha256'],'saved cell cache SHA')
                    with np.load(cell['path'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
                    inst=arrays['instances'];ids=np.unique(inst);ids=ids[ids>0];kept=ids[:255]
                    require(len(ids)==int(arrays['total'])==cell['details']['total_cell_count'],'true nuclei count')
                    yx=np.asarray(ndimage.center_of_mass(np.ones_like(inst,np.uint8),inst,kept)) if len(kept) else np.empty((0,2))
                    order=np.lexsort((yx[:,1],yx[:,0]));kept=kept[order];yx=yx[order]
                    expected=np.zeros((len(kept),74),np.float32)
                    if len(kept):
                        expected[:,0]=yx[:,1]/inst.shape[1];expected[:,1]=yx[:,0]/inst.shape[0]
                        mx,my=cell['details']['detection_mpp_xy'];expected[:,2]=np.log1p(np.bincount(inst.ravel())[kept]*mx*my/(.25*.25));expected[:,3]=1
                    expected[:,10:]=arrays['reg'];require(np.array_equal(expected,arrays['features']),'independent74 encoding/geometry/area/unknown class')
                    require(len(kept)==cell['details']['retained_cell_count'],'retained count')
                    native=slide.read(box)
                    import hashlib
                    require(hashlib.sha256(native.tobytes()).hexdigest()==cell['native_RGB_sha256'],'native RGB source identity')
                    image=np.asarray(Image.fromarray(native).resize((512,512),Image.Resampling.BILINEAR));scale=cell['scale'];suffix='' if scale==0 else f'_{scale}'
                    raw['image'+suffix]=he_to_tensor(image.copy())[None].to(device);raw['cells'+suffix]=torch.from_numpy(arrays['features'])[None].to(device)
                    raw['cell_valid'+suffix]=torch.ones(1,len(kept),dtype=torch.bool,device=device)
                    raw['total_cell_count' if scale==0 else f'total_{scale}']=torch.tensor([len(ids)],dtype=torch.float32,device=device);totals.append(len(ids))
                    geometry_checks.append(dict(tile=q['tile'],scale=scale,total=len(ids),retained=len(kept),geometry_exact=True))
                counts=[];hook=model.core.cell.register_forward_hook(lambda module,args,out:counts.append(float(out['region_cell_count'].sum())))
                try:
                    with torch.inference_mode():fresh=model.core.encode_image(raw)
                finally:hook.remove()
                require(len(counts)==3 and all(math.isclose(x,y,abs_tol=1e-3,rel_tol=1e-6) for x,y in zip(counts,totals)),'true total distributed across regions, not capped retained count')
                error=tree_error(fresh,cached);require(error<=2e-6,'fresh selected model encoded cache parity')
                with torch.inference_mode():
                    for budget in ('1+1','3+2'):
                        _,prompts,_=data.materialize(nested(maximum,budget),device,True,True);task=model.core.support_task(source,prompts)
                        result,maps=model.core.decode(cached,task,prompts,False);require(not maps.count_nonzero(),'query maps zero')
                        actual=result['pixel_probability'][0].cpu().numpy();saved=np.load(a.gate/f'probability_{q["tile"]:06d}_{budget}.npy',allow_pickle=False)
                        delta=float(np.abs(actual-saved).max());require(delta<=2e-6 and np.array_equal(actual>=.5,saved>=.5),'fresh model saved probability/binary replay')
                        c=next(c for c in done['checks'] if c['tile']==q['tile'] and c['budget']==budget);timings=c['timings'];require(len(timings)==20,'all repetitions retained')
                        for name in ('task_ms','decode_ms','wall_ms'):close(statistics.median(t[name] for t in timings),c[name+'_median'],'timing median')
                        close(quantile([t['wall_ms'] for t in timings],.95),c['wall_ms_p95'],'timing p95')
                        checks.append(dict(tile=q['tile'],budget=budget,fresh_encoded_error=error,saved_probability_error=delta,binary_exact=True,region_count_sums=counts))
        finally:slide.close()
        require(done['online_query_encode_calls']==done['online_detector_calls']==0 and done['detector_calls']==6,'recorded preparation/online separation')
        report=dict(status='passed_independent_fullcell_GPU_gate_audit',worker_pid=pid,geometry_checks=geometry_checks,checks=checks,source_sha256=sha(__file__),
            gate_summary_sha256=sha(str(a.gate/'summary.json')),limits=['Two actual windows only, includes near-white but actual fine cells5, not an actual zero-cell test',
            'No detector rerun/partition-invariant detection proof; saved instances independently recounted and fresh model/source pixels replayed',
            'Per-window warm resident-cache timing; full-slide cell preparation, IO/stitch and GPU WSI latency remain pending'])
        dump('summary.json',report);dump('status.json',dict(status=report['status'],worker_pid=pid));print(json.dumps(report),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('gate','training','development','plan','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
