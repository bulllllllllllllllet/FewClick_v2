"""Independent all-grid partition/reuse identities and actual native RGB audit."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pyvips


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def require(ok, why):
    if not ok: raise ValueError(why)


def run(a):
    a.output.mkdir(exist_ok=False,parents=True);started=time.perf_counter();pyvips.cache_set_max(0)
    source=read(a.source/'summary.json');gate=read(a.gate/'summary.json')
    require(source['status']=='complete_full_valid_input_manifest_unaccepted' and gate['status']=='passed_full_valid_input_native_gate' and source['signature']==gate['signature'], 'same-source full input and real native gate')
    require(source['signature']['source_sha256']==sha('benchmarks/v5/wsi_global/full_valid_inputs.py') and source['signature']['native_bridge_sha256']==sha('benchmarks/v5/wsi_global/cell_cache_inputs.py'), 'unchanged manifest/native bridge code')
    for path,key in [('inputs.jsonl','inputs_sha256'),('coverage.json','coverage_sha256')]:
        require(sha(a.source/path)==source[key] and sha(a.gate/path)==gate[key], 'all complete ordered input/coverage bytes and gate/full identity')
    entries=[json.loads(line) for line in (a.source/'inputs.jsonl').open()]
    require(len(entries)==28452 and [r['index'] for r in entries]==list(range(28452)), 'every canonical full-grid input slot')
    by_key={r['key']:r for r in entries};require(len(by_key)==28452, 'all unique native fields')
    data=a.root/'main30_full_1027_171918';protocol=a.root/'support_query_full_1027_173800';old=a.root/'cell_native_inputs_full_1028_012700'
    require(sha(data/'summary.json')==source['signature']['data_summary_sha256'] and sha(protocol/'summary.json')==source['signature']['protocol_summary_sha256'] and sha(old/'summary.json')==source['signature']['original_input_summary_sha256'], 'original independent source receipts bound')
    original=[json.loads(line) for line in (old/'inputs.jsonl').open()]
    all_old={r['key']:r for r in original};prior_by_boxes={(r['wsi_id'],tuple(map(tuple,r['native_boxes']))):r for r in original}
    queries={r['wsi_id']:r for r in read(protocol/'query_inputs.json')};coverage={r['wsi_id']:r for r in read(a.source/'coverage.json')}
    order=[];counts=Counter();reused=set();odd=set();partition_counts=Counter()
    for s in read(data/'slides_private.json'):
        sid=s['wsi_id'];q=queries[sid];c=coverage[sid]
        require(sha(s['grid'])==s['files'][s['grid']]['sha256'] and sha(q['grid'])==q['grid_sha256'], 'all original full and remote grid bytes')
        full=list(map(json.loads,Path(s['grid']).open()));remote={t['number']:t for t in map(json.loads,Path(q['grid']).open())}
        numbers=[t['number'] for t in full];require(numbers==list(range(len(full))), 'original complete tile numbering')
        xs=sorted({t['native_boxes'][0][0] for t in full});ys=sorted({t['native_boxes'][0][1] for t in full});fw,fh=s['geometry']['fine_native_wh'];w,h=s['native_wh']
        require(len(full)==len(xs)*len(ys) and {(t['native_boxes'][0][0],t['native_boxes'][0][1]) for t in full}=={(x,y) for y in ys for x in xs}, 'complete Cartesian coverage, no filtered tissue')
        require(all(b-a<=fw for a,b in zip(xs,xs[1:])) and all(b-a<=fh for a,b in zip(ys,ys[1:])), 'no fine-domain holes')
        bounds=[xs[0],ys[0],xs[-1]+fw,ys[-1]+fh]
        require(c['full_tiles']==len(full) and c['valid_fine_bounds_native']==bounds==s['geometry']['valid_fine_bounds_xyxy'], 'exact complete native valid domain')
        gw,gh=s['gt_wh'];sx,sy=w/gw,h/gh
        require(c['GT_full_cell_domain']==[math.ceil(bounds[0]/sx),math.ceil(bounds[1]/sy),math.floor(bounds[2]/sx),math.floor(bounds[3]/sy)] and c['native_area_fraction']==(bounds[2]-bounds[0])*(bounds[3]-bounds[1])/(w*h), 'whole GT-cell extent and exact native rectangle coverage, no GT pixels used')
        require((c['patient_id_as_recorded'],c['split'])==(s['patient_id_as_recorded'],s['split']), 'preserved provenance, no imposed patient split')
        local=Counter();zones=Counter();remote_left=min(t['native_boxes'][0][0] for t in remote.values())
        for t in full:
            key=f'{sid}:full_valid:{t["number"]}';order.append(key);r=by_key[key]
            require((r['wsi_id'],r['source_wsi'],r['native_wh'],r['native_mpp'],r['native_boxes'],r['tile'])==(sid,s['source_wsi'],s['native_wh'],s['native_mpp'],t['native_boxes'],t['number']), 'every exact original frame/three boxes/MPP identity')
            x,y,bw,bh=r['native_boxes'][0];require((bw,bh)==(fw,fh), 'original native fine dimensions')
            for scale,(xx,yy,ww,hh) in enumerate(r['native_boxes']):
                require(ww==fw*(2**scale) and hh==fh*(2**scale) and min(xx,yy)>=0 and xx+ww<=w and yy+hh<=h, 'all physical sizes/full input footprint')
                require(max(abs(xx+ww/2-t['centre_level0'][0]),abs(yy+hh/2-t['centre_level0'][1]))<=.5, 'frozen half-pixel centre geometry')
            if fw%2 or fh%2:odd.add(sid)
            prior=prior_by_boxes.get((sid,tuple(map(tuple,r['native_boxes']))))
            kind='remote_reuse_candidate' if t['number'] in remote else 'support_reuse_candidate' if prior is not None else 'new_input'
            require(r['preparation']==kind and r['source_cut_native']==q['source_cut_native'] and r['gap_native']==q['gap_native'], 'independent preparation and split policy')
            if prior is not None:
                require(r['existing_input_key']==prior['key'] and r['existing_input_index']==prior['index'], 'actual original cache input candidate identity')
                require(prior['kind']==('query' if kind=='remote_reuse_candidate' else 'support'), 'no cross-kind fabricated aliases');reused.add(prior['key'])
            else:
                require(r['existing_input_key'] is None and r['existing_input_index'] is None, 'unprepared inputs have no fake alias')
            if kind!='remote_reuse_candidate':require(x+bw<=remote_left, 'new fine footprints do not change original remote pixels')
            coarse=r['native_boxes'][2]
            expected='source_band' if coarse[0]+coarse[2]<=q['source_cut_native'] else 'strict_remote' if coarse[0]>=q['source_cut_native']+q['gap_native'] else 'context_crosses_source_or_gap'
            require(r['zone']==expected and (expected=='strict_remote')==(t['number'] in remote), 'complete4x source/query/gap partition')
            local[kind]+=1;zones[expected]+=1
        require(dict(local)==c['counts'] and dict(zones)==c['zones'], 'all per-slide partition counts')
        counts.update(local);partition_counts.update(zones)
    require(order==[r['key'] for r in entries] and reused==set(all_old) and counts==dict(remote_reuse_candidate=20208,support_reuse_candidate=248,new_input=7996), 'all28452/full20456 candidates conserved, no omitted fields')
    require(sha(a.gate/'native_evidence.json')==gate['native_evidence_sha256'], 'native gate pixel evidence bound')
    probes=read(a.gate/'native_evidence.json');require(len(probes)==8 and len({r['key'] for r in probes})==8, 'all eight distinct actual native probes')
    checks=[]
    for p in probes:
        r=by_key[p['key']];slide=pyvips.Image.new_from_file(r['source_wsi'],access='random');require([slide.width,slide.height]==r['native_wh'], 'actual original slide header')
        for v in p['views']:
            scale=v['scale'];x,y,w,h=r['native_boxes'][scale]
            native=np.frombuffer(slide.crop(x,y,w,h).extract_band(0,n=3).write_to_memory(),np.uint8).reshape(h,w,3)
            image=np.asarray(Image.fromarray(native).resize((512,512),Image.Resampling.BILINEAR))
            require(hashlib.sha256(native.tobytes()).hexdigest()==v['native_RGB_sha256'] and hashlib.sha256(image.tobytes()).hexdigest()==v['HE_array_sha256'], 'fresh independent native/PIL pixels exact')
            require(sha(v['path'])==v['sha256'] and np.array_equal(image,np.asarray(Image.open(v['path']).convert('RGB'))), 'actual disk HE pixels exact')
            size=[max(1,round(w*r['native_mpp'][0]/.25)),max(1,round(h*r['native_mpp'][1]/.25))]
            require(size==v['detection_size'] and v['detection_mpp_xy']==[w*r['native_mpp'][0]/size[0],h*r['native_mpp'][1]/size[1]], 'future detector physical metadata, no actual detection')
            checks.append(dict(key=r['key'],scale=scale,native_and_HE_exact=True))
    require(len(checks)==24 and any(by_key[p['key']]['native_boxes'][0][2]%2 for p in probes) and {p['preparation'] for p in probes}==set(counts), 'all24 native views, odd sizes and all three preparation kinds')
    (a.output/'native_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    result=dict(status='accepted_all28452_full_valid_inputs_and_native24_bridge',worker_pid=os.getpid(),source_sha256=sha(__file__),source_summary_sha256=sha(a.source/'summary.json'),gate_summary_sha256=sha(a.gate/'summary.json'),inputs=28452,views=85356,counts=dict(counts),zones=dict(partition_counts),existing_inputs_preserved=len(reused),odd_slides=sorted(odd),native_probes=8,seconds=time.perf_counter()-started,
                scope='All original physical fields/partitions/provenance and24 fresh native pixel replays; geometric cache-reuse candidates only, no accepted encoder/cache reuse, detection/features, model scores or full tissue coverage',remaining='Actual per-method cache alias gates and new7996 interactive or8244 closed inputs, full-valid output/stitch/scoring and remote invariance')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','source','gate','output'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
