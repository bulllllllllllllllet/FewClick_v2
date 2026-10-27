"""Freeze real native support/query inputs for the full XCell branch.

This prepares input geometry and validates the single-native-read HE bridge.
No nuclei are detected here, and no cell features or model scores are claimed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import PIL
import pyvips


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def require(test,why):
    if not test:
        raise ValueError(why)


def native_views(slide,record):
    """One actual crop per scale supplies both HE512 and native cell RGB."""
    for scale,(x,y,w,h) in enumerate(record['native_boxes']):
        crop=slide.crop(x,y,w,h).extract_band(0,n=3)
        require(crop.format=='uchar','canonical uint8 native RGB')
        native=np.frombuffer(crop.write_to_memory(),np.uint8).reshape(h,w,3).copy()
        image=np.asarray(Image.fromarray(native).resize((512,512),Image.Resampling.BILINEAR)).copy()
        # Same metadata and canonical-size calculation as prepare_variant.
        metadata=dict(input_to_image_affine=[w/512,0,0,0,h/512,0],source_image_size=[w,h])
        size=[max(1,round(w*record['native_mpp'][0]/.25)),max(1,round(h*record['native_mpp'][1]/.25))]
        actual_mpp=[w*record['native_mpp'][0]/size[0],h*record['native_mpp'][1]/size[1]]
        yield scale,native,image,dict(metadata=metadata,detection_size=size,detection_mpp_xy=actual_mpp)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid();start=time.perf_counter();pyvips.cache_set_max(0)
    def dump(n,d):
        (a.output/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    try:
        root=a.root;protocol=root/'support_query_full_1027_173800';data=root/'main30_full_1027_171918'
        accepted=read(root/'support_query_acceptance_1027_173800/summary.json');da=read(root/'main30_acceptance_1027_171918/summary.json')
        require(accepted['status']=='accepted_source_band_query_protocol' and accepted['protocol_summary_sha256']==sha(protocol/'summary.json'),'all30 source/query protocol independently accepted')
        require(da['source_summary_sha256']==sha(data/'summary.json'),'accepted original native data/geometry source')
        signature=dict(source_sha256=sha(__file__),protocol_sha256=sha(protocol/'summary.json'),data_acceptance_sha256=sha(root/'main30_acceptance_1027_171918/summary.json'),numpy=np.__version__,PIL=PIL.__version__,vips=[pyvips.version(i) for i in range(3)],geometry='native three views; CPU PIL512 HE and native prepare_variant cells share the same RGB crop')
        if not a.gate:
            require(a.verified_gate is not None,'actual representative three-view native-read bridge gate required')
            g=read(a.verified_gate/'summary.json');require(g['status']=='passed_main30_fullcell_native_input_bridge' and g['signature']==signature,'same validated input recipe')
        old=read(root/'native_remote_gate_exact_1027_175300/input_identities.json');references=set(old)
        inputs=read(protocol/'query_inputs.json');records=[]
        if a.gate:
            # Include the first odd-native-width support, selected by geometry
            # alone, to exercise the frozen half-pixel integer rounding policy.
            odd=next(r for r in inputs if r['support_inputs'][0]['images'][0]['native_box'][2]%2)
            references.add(odd['support_inputs'][0]['support_id'])
        for r in inputs:
            sid=r['wsi_id'];require(sha(r['grid'])==r['grid_sha256'],'fixed complete remote grid bytes')
            for s in r['support_inputs']:
                key=s['support_id']
                if a.gate and key not in references:continue
                records.append(dict(index=len(records),key=key,kind='support',wsi_id=sid,source_wsi=r['source_wsi'],native_wh=r['native_wh'],native_mpp=r['native_mpp'],native_boxes=[im['native_box'] for im in s['images']],source_cut_native=r['source_cut_native'],gap_native=r['gap_native'],HE_reference=s['images']))
            for t in (json.loads(line) for line in Path(r['grid']).open()):
                key=sid+':query:'+str(t['number'])
                if a.gate and key not in references:continue
                records.append(dict(index=len(records),key=key,kind='query',tile=t['number'],wsi_id=sid,source_wsi=r['source_wsi'],native_wh=r['native_wh'],native_mpp=r['native_mpp'],native_boxes=t['native_boxes'],source_cut_native=r['source_cut_native'],gap_native=r['gap_native'],HE_reference=old[key]['images'] if a.gate else None))
        require(len(records)==(8 if a.gate else 20456) and len({r['key'] for r in records})==len(records),'all unique native source/query inputs')
        counts={kind:sum(r['kind']==kind for r in records) for kind in ('support','query')}
        require((counts['support'],counts['query'])==((4,4) if a.gate else (248,20208)),'fixed fullcell support/query coverage')
        for r in records:
            w,h=r['native_wh'];require(len(r['native_boxes'])==3 and len(r['native_mpp'])==2 and min(r['native_mpp'])>0,'all physical three-view metadata')
            centres=[]
            for x,y,bw,bh in r['native_boxes']:
                require(min(x,y)>=0 and min(bw,bh)>0 and x+bw<=w and y+bh<=h,'every full physical input footprint stays inside native slide')
                centres.append([x+bw/2,y+bh/2])
            require(all(max(abs(c[0]-centres[0][0]),abs(c[1]-centres[0][1]))<=.5 for c in centres),'preserve frozen view centres within integer-rounding half-pixel tolerance')
            coarse=r['native_boxes'][2]
            require(coarse[0]+coarse[2]<=r['source_cut_native'] if r['kind']=='support' else coarse[0]>=r['source_cut_native']+r['gap_native'],'full4x source/query separation preserved')
        dump('contract.json',dict(worker_pid=pid,signature=signature,gate=a.gate,input_count=len(records),views=3*len(records),query_GT_access=False,tissue_GT_to_cell_class=False,cell_recipe='Existing prepare_variant/InstanSeg/CTP/XCell; native RGB and physical metadata only; no detector/features executed by this module',detector_order='manifest index then scale0/1/2, one continuous serial RNG stream; no partition-invariance claim'))
        with (a.output/'inputs.jsonl').open('x') as f:
            for r in records:f.write(json.dumps(r)+'\n')
        evidence=[]
        if a.gate:
            current=None;slide=None
            for r in records:
                if current!=r['wsi_id']:
                    slide=pyvips.Image.new_from_file(r['source_wsi'],access='random');current=r['wsi_id']
                    require([slide.width,slide.height]==r['native_wh'],'actual native header dimensions')
                folder=a.output/r['key'].replace(':','_');folder.mkdir()
                views=[]
                for scale,native,image,details in native_views(slide,r):
                    d=r['HE_reference'][scale];require(sha(d['path'])==d['sha256'],'previously accepted nativeHE reference bytes')
                    reference=np.asarray(Image.open(d['path']).convert('RGB'))
                    require(np.array_equal(image,reference),'single native read HE512 equals accepted separate-read path exactly')
                    path=folder/f'HE512_scale{scale}.png';Image.fromarray(image).save(path)
                    views.append(dict(scale=scale,native_box=r['native_boxes'][scale],native_RGB_sha256=hashlib.sha256(native.tobytes()).hexdigest(),HE_RGB_sha256=hashlib.sha256(image.tobytes()).hexdigest(),path=str(path),sha256=sha(path),HE_exact=True,**details))
                evidence.append(dict(key=r['key'],views=views));dump('status.json',dict(status='validating_actual_native_cell_inputs',worker_pid=pid,completed=len(evidence),total=len(records)))
            dump('native_evidence.json',evidence)
        result=dict(status='passed_main30_fullcell_native_input_bridge' if a.gate else 'complete_main30_fullcell_input_manifest',worker_pid=pid,signature=signature,inputs=len(records),native_views=3*len(records),counts=counts,input_manifest_sha256=sha(a.output/'inputs.jsonl'),native_evidence_sha256=sha(a.output/'native_evidence.json') if a.gate else None,seconds=time.perf_counter()-start,
                    scope='Accepted physical inputs and one-read native-to-HE equivalence only; not cell segmentation, XCell features, encoded model caches or quality results',remaining='Real GPU detector/74D feature preparation, fullcell native model/cache/decode gates, independent acceptance and complete main30 quality')
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps(result),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',action='store_true');p.add_argument('--verified-gate',type=Path)
    run(p.parse_args())
