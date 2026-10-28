"""Join independently accepted old/new logits in immutable original grid order."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


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
    expected=[json.loads(line) for line in (a.root/'full_valid_inputs_full_1028_034700/inputs.jsonl').open()]
    by_number={(r['wsi_id'],r['tile']):r for r in expected}
    recipe=dict(source_sha256=sha(__file__),input_summary_sha256=sha(a.root/'full_valid_inputs_full_1028_034700/summary.json'),weight_sha256='0aa11b35866262648d2b56340a0d4501165da0516cf21df7bdfebf8da05d06a4')
    if not a.fixture:
        g=read(a.verified_fixture/'summary.json') if a.verified_fixture else {}
        require(g.get('status')=='passed_closed12_full_valid_index_fixture' and g['recipe']==recipe,'same original-order merging gate')
    seen={};duplicates=[];receipts=[];counts={}
    for kind,path,acceptance in [('remote',a.remote,a.remote_acceptance),('extension',a.extension,a.extension_acceptance)]:
        source=read(path/'summary.json');accepted=read(acceptance/'summary.json')
        wanted=('accepted_closed12_cache_gate' if a.fixture else 'accepted_closed12_full_WSI_cache_qualified') if kind=='remote' else ('accepted_closed12_extension_gate' if a.fixture else 'accepted_closed12_extension_cache_qualified')
        require(accepted['status']==wanted and accepted['source_summary_sha256']==sha(path/'summary.json'),'independent completed source cache receipt')
        require(source['signature']['weight_sha256']==recipe['weight_sha256'] and source['cache_manifest_sha256']==sha(path/'cache_manifest.jsonl'),'same selected weight and full input cache manifests')
        entries=[json.loads(line) for line in (path/'cache_manifest.jsonl').open()];counts[kind]=len(entries)
        require(len(entries)==source['tiles'],'source manifests complete')
        for e in entries:
            d=e['identity'];key=(d['wsi_id'],d['tile']);r=by_number[key]
            require((d['source_wsi'],d['native_box'],d['native_mpp'])==(r['source_wsi'],r['native_boxes'][0],r['native_mpp']),'every actual grid identity, all three-view context bounds preserved in full input manifest')
            require(Path(e['path']).stat().st_size==e['bytes'],'referenced immutable cache file exists at original size')
            if not a.fixture:require((r['preparation']=='remote_reuse_candidate')==(kind=='remote'),'formal disjoint source partition')
            if key in seen:
                require(a.fixture and kind=='extension','no duplicate formal tile or source-manifest entries')
                old=seen[key];require(old['identity']==d,'same real duplicate input identity in representative gate')
                for entry in (old,e):require(sha(entry['path'])==entry['sha256'],'actual gate duplicate cache bytes')
                left=torch.load(old['path'],map_location='cpu',weights_only=False);right=torch.load(e['path'],map_location='cpu',weights_only=False)
                require(left['checkpoint_sha256']==right['checkpoint_sha256']==recipe['weight_sha256'] and torch.equal(left['logits'],right['logits']),'native original and extension duplicate logits exactly identical')
                duplicates.append(list(key));continue
            seen[key]=dict(e,full_valid_key=r['key'],cache_origin=kind)
        receipts.append(dict(kind=kind,source=str(path),source_summary_sha256=sha(path/'summary.json'),acceptance=str(acceptance),acceptance_summary_sha256=sha(acceptance/'summary.json')))
    require(counts==dict(remote=4,extension=8) if a.fixture else counts==dict(remote=20208,extension=8244),'fixed complete source counts')
    require(bool(duplicates) if a.fixture else not duplicates and set(seen)==set(by_number),'actual representative cross-path duplicate or complete disjoint28452 coverage')
    merged=[seen[(r['wsi_id'],r['tile'])] for r in expected if (r['wsi_id'],r['tile']) in seen]
    with (a.output/'cache_manifest.jsonl').open('x') as f:
        for entry in merged:f.write(json.dumps(entry)+'\n')
    result=dict(status='passed_closed12_full_valid_index_fixture' if a.fixture else 'accepted_closed12_full_valid_cache_index_qualified',worker_pid=os.getpid(),recipe=recipe,tiles=len(merged),counts=counts,gate_duplicate_inputs=duplicates,cache_manifest_sha256=sha(a.output/'cache_manifest.jsonl'),receipts=receipts,
                scope='Original full-grid order and disjoint accepted cache identities/selected weight; original file references/digests retained, no logits re-encoded or copied. Fresh full-cache bytes and native encoder evidence inherited from each independent accepted source, not rehashed exhaustively at index join.',
                remaining='Complete28452 native-probability raster/GT scoring and independent output acceptance; strict remote benchmark remains separate')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','remote','remote-acceptance','extension','extension-acceptance','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--fixture',action='store_true');p.add_argument('--verified-fixture',type=Path);run(p.parse_args())
