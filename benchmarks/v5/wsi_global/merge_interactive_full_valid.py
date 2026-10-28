"""Preserve accepted source/query caches, alias supports, append new valid fields."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(ok,why):
    if not ok:raise ValueError(why)


def run(a):
    a.output.mkdir(exist_ok=False,parents=True)
    base=read(a.base_cache/'summary.json');extension=read(a.extension/'summary.json');ba=read(a.base_acceptance/'summary.json');ea=read(a.extension_acceptance/'summary.json')
    method=base['signature']['method'];views={'fewclick_m3d_image':3,'sam_ft':1,'wsi_sam_ft':2}[method]
    require(ba['status']==('accepted_interactive_cache_fixture' if a.fixture else 'accepted_interactive_main30_cache_qualified') and ea['status']==('accepted_interactive_valid_extension_gate' if a.fixture else 'accepted_interactive_valid_extension_qualified'),'actual independently accepted complete input caches')
    for cache,summary,receipt in [(a.base_cache,base,ba),(a.extension,extension,ea)]:
        require(receipt['source_summary_sha256']==sha(cache/'summary.json') and summary['cache_manifest_sha256']==sha(cache/'cache_manifest.jsonl'),'original actual source/output acceptance identity')
    sig=extension['signature'];require((sig['method'],sig['device'],sig['weight_sha256'],sig['base_summary_sha256'],sig['base_acceptance_sha256'])==(method,base['signature']['device'],base['signature']['weight_sha256'],sha(a.base_cache/'summary.json'),sha(a.base_acceptance/'summary.json')),'same selected method/device/accepted base encoding recipe')
    full=a.root/'full_valid_inputs_full_1028_034700';require(sha(full/'summary.json')==sig['input_summary_sha256'],'same all28452 original grid')
    recipe=dict(source_sha256=sha(__file__),method=method,input_summary_sha256=sig['input_summary_sha256'])
    if not a.fixture:
        g=read(a.verified_fixture/'summary.json') if a.verified_fixture else {}
        require(g.get('status')=='passed_interactive_full_valid_index_fixture' and g['recipe']==recipe,'same original-order candidate/alias merging fixture')
    rows=[json.loads(line) for line in (full/'inputs.jsonl').open()]
    old={e['identity']['key']:e for e in map(json.loads,(a.base_cache/'cache_manifest.jsonl').open())};fresh={e['identity']['key']:e for e in map(json.loads,(a.extension/'cache_manifest.jsonl').open())}
    require(len(old)==base['items'] and len(fresh)==extension['items'],'unique complete source manifests')
    if not a.fixture:require((len(old),len(fresh))==(20456,7996),'all required original and additional fields')
    rgb_receipt=read(a.root/'support_query_acceptance_1027_173800/summary.json')
    require(rgb_receipt['actual_native_RGB_views']==744 and rgb_receipt['protocol_summary_sha256']==base['signature']['protocol_sha256'],'independent all744 support native/PIL source equality before aliasing')
    require(read(full/'summary.json')['signature']['protocol_summary_sha256']==base['signature']['protocol_sha256'],'same source band, selected original support frames and full native grid')
    merged=[];used_old=set();used_new=set();ignored=[];counts=Counter();source_RGB_files=0
    for r in rows:
        key=r['existing_input_key'];origin='base' if key in old else 'extension' if r['key'] in fresh else None
        if origin is None:
            require(a.fixture,'every full28452 input must have exactly one accepted encoded cache');continue
        e=old[key] if origin=='base' else fresh[r['key']];d=e['identity'];geometry=d['geometry']
        require(d['wsi_id']==r['wsi_id'] and Path(e['path']).stat().st_size==e['bytes'],'real same native frame and original accepted file reference')
        if origin=='base':
            require(key not in used_old,'each accepted base cache used once');used_old.add(key)
            if d['kind']=='support':
                require(r['preparation']=='support_reuse_candidate','only exact physical support fields aliased as queries')
                images=geometry['source_record']['images'][:views]
                require(len(images)==len(d['RGB_array_sha256'])==views,'all method-specific source RGB views present')
                require([im['native_box'] for im in images]==r['native_boxes'][:views],'all method-specific true support view boxes')
                for im,digest in zip(images,d['RGB_array_sha256']):
                    require(sha(im['path'])==im['sha256'] and hashlib.sha256(np.asarray(Image.open(im['path']).convert('RGB')).tobytes()).hexdigest()==digest,'all support RGB bytes/digests originally independently tied to native views');source_RGB_files+=1
                counts['support_as_query_alias']+=1
            else:
                require(d['kind']=='query' and r['preparation']=='remote_reuse_candidate' and geometry==dict(tile=r['tile'],native_boxes=r['native_boxes'][:views],native_mpp=r['native_mpp'],source_wsi=r['source_wsi']),'all remote cache geometry unchanged');counts['original_remote']+=1
            if r['key'] in fresh:
                require(a.fixture,'no newly prepared duplicate formal field');ignored.append(r['key']);used_new.add(r['key'])
        else:
            require(r['preparation']=='new_input' and d['kind']=='full_valid_query' and geometry==dict(tile=r['tile'],native_boxes=r['native_boxes'][:views],native_mpp=r['native_mpp'],source_wsi=r['source_wsi']),'every truly new native field matched');used_new.add(r['key']);counts['additional']+=1
        merged.append(dict(e,full_valid_key=r['key'],cache_origin=origin,original_cache_key=d['key'],full_valid_geometry=dict(tile=r['tile'],native_boxes=r['native_boxes'],native_mpp=r['native_mpp'],source_wsi=r['source_wsi'])))
    require(used_old==set(old) and used_new==set(fresh),'all original and new inputs accounted, no orphan/omitted cache')
    require(len(merged)==(11 if a.fixture else 28452) and len({e['full_valid_key'] for e in merged})==len(merged),'unique complete full-grid ordering')
    require(len(ignored)==4 if a.fixture else counts==dict(original_remote=20208,support_as_query_alias=248,additional=7996),'representative alias intersection or all full28452 parts')
    with (a.output/'cache_manifest.jsonl').open('x') as f:
        for e in merged:f.write(json.dumps(e)+'\n')
    result=dict(status='passed_interactive_full_valid_index_fixture' if a.fixture else 'accepted_interactive_full_valid_cache_index_qualified',worker_pid=os.getpid(),recipe=recipe,device=sig['device'],weight_sha256=sig['weight_sha256'],items=len(merged),counts=dict(counts),support_RGB_files=source_RGB_files,ignored_fixture_duplicate_aliases=ignored,cache_manifest_sha256=sha(a.output/'cache_manifest.jsonl'),
        base_cache=str(a.base_cache),base_summary_sha256=sha(a.base_cache/'summary.json'),base_acceptance=str(a.base_acceptance),base_acceptance_sha256=sha(a.base_acceptance/'summary.json'),extension=str(a.extension),extension_summary_sha256=sha(a.extension/'summary.json'),extension_acceptance=str(a.extension_acceptance),extension_acceptance_sha256=sha(a.extension_acceptance/'summary.json'),
        scope='Accepted immutable encoded caches referenced once in original full-grid order. Native RGB equivalence of all support images inherited from744-view audit; same-method source/query encoder equivalence gated. Original payload identities preserved, no fresh all248 encodes/detectors or copied/recomputed features.',remaining='Full-valid per-method predictions/native output/GT scoring and remote invariance; fullcell XCell path requires separate actual cells/cache evidence')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','base-cache','base-acceptance','extension','extension-acceptance','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--fixture',action='store_true');p.add_argument('--verified-fixture',type=Path);run(p.parse_args())
