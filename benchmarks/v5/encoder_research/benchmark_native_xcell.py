"""Native mask-attended CTransPath/XCell component timing on four accepted fields."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import pyvips
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_cell_models,attended_crop,cells_from_instances


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    base=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research');prior=base/'xcell_point_gate_1031_005709'
    acceptance=json.loads((prior/'independent_acceptance.json').read_text());assert acceptance['status']=='accepted_real_legacy_branch_input_and_point_matching_gate'
    manifest=base/'p0_cell_probe_manifest_1030_182424/cells.jsonl';catalog=[json.loads(s) for s in manifest.read_text().splitlines()]
    old=json.loads((prior/'data/report.json').read_text())
    def timed(fn):
        torch.cuda.synchronize();start=time.perf_counter();result=fn();torch.cuda.synchronize()
        return result,time.perf_counter()-start
    start=time.perf_counter();detector,ctp,xcell,loading=load_cell_models(torch.device('cuda:0'));torch.cuda.synchronize();load_time=time.perf_counter()-start
    records=[]
    for row in old['rows']:
        identity=row['field'];source=next(s for s in catalog if s['field_id']==identity)
        assert sha(source['image'])==source['image_sha256']
        with Image.open(source['image']) as im:rgb=np.asarray(im.convert('RGB')).copy()
        mx,my=source['mpp_xy'];size=(round(rgb.shape[1]*mx/.25),round(rgb.shape[0]*my/.25))
        high=np.asarray(Image.fromarray(rgb).resize(size,Image.Resampling.BILINEAR)).copy()
        reference=np.load(prior/'data'/(identity+'.npz'),allow_pickle=False)
        inst=reference['instances'];metadata,labels,total=cells_from_instances(inst)
        assert np.array_equal(labels,reference['instance_ids'])
        # Detector timing is separate; measured instances never substitute the fixed accepted masks.
        detected,detector_seconds=timed(lambda:np.asarray(detector.eval(high,diameter=18,batch_size=1)[0],np.int32))
        def crops():
            embeddings=[]
            for label in labels:
                patch=attended_crop(high,inst,int(label))
                value=torch.from_numpy(patch.copy()).permute(2,0,1).float()[None]/255
                value=F.interpolate(value,size=(224,224),mode='bilinear',align_corners=False,antialias=True)[0]
                embeddings.append((value-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None])
            return embeddings
        def encode(embeddings):return torch.cat([ctp(torch.stack(embeddings[i:i+32]).cuda()).float() for i in range(0,len(embeddings),32)])
        def context(raw):
            padded=raw.new_zeros((1,255,768));valid=raw.new_zeros((1,255));padded[0,:len(raw)]=raw;valid[0,:len(raw)]=1
            return xcell(None,padded,valid)[1][0,:len(raw)].cpu().numpy()
        timings=[]
        with torch.inference_mode():
            warm=context(encode(crops()));assert warm.shape==reference['features'].shape
            assert np.allclose(warm,reference['features'],rtol=1e-5,atol=1e-5)
            for repeat in range(3):
                embeddings,crop_s=timed(crops);raw,encode_s=timed(lambda:encode(embeddings));features,context_s=timed(lambda:context(raw))
                assert np.allclose(features,warm,rtol=1e-5,atol=1e-5) and np.isfinite(features).all()
                timings.append(dict(crop_seconds=crop_s,ctranspath_seconds=encode_s,xcell_seconds=context_s,
                                    feature_path_seconds=crop_s+encode_s+context_s))
        np.savez_compressed(a.output/(identity+'.npz'),features=warm,instance_ids=labels,detected_instances=detected)
        records.append(dict(field=identity,retained=len(labels),detected_before_cap=total,detector_seconds=detector_seconds,
                            detector_matches_accepted_masks=bool(np.array_equal(detected,inst)),warm_repeats=timings,
                            original_feature_max_abs_error=float(np.max(np.abs(warm-reference['features']))),
                            fixed_instances_sha256=sha(prior/'data'/(identity+'.npz')),image_sha256=source['image_sha256']))
        print(json.dumps(records[-1]),flush=True)
    assert sum(r['retained'] for r in records)==654
    result=dict(status='accepted_native_XCell_component_cost',worker_pid=os.getpid(),records=records,model_load_seconds=load_time,
                loading=loading,source_sha256=sha(__file__),prior_acceptance_sha256=sha(prior/'independent_acceptance.json'),
                manifest_sha256=sha(manifest),GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                scope='four fixed train fields; detector cold/serial observed separately; cached accepted masks used for three warm feature-path repeats; excludes file decode/resample and expert GT; not matched FLOPs/WSI/online-click latency',
                GT_used=False,quality_claim=False)
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
