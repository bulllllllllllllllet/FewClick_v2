"""Actual accepted model outputs: lossless packed masks and all probability hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .packed_prediction import PackedSampledStripe,unpack_rows


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def require(test,reason):
    if not test:
        raise ValueError(reason)


def verify(folder,case,records,native_wh,GT_wh):
    bounds=case['native_coverage']['bounds_level0']
    writer=PackedSampledStripe(folder,bounds,records[0]['native_box'][2:],native_wh,GT_wh)
    for d in records:
        require(sha(d['path'])==d['sha256'],'actual accepted512 probability bytes')
        writer.push(np.load(d['path'],allow_pickle=False),d['native_box'])
    summary=writer.finish()
    require(summary['GT_cell_bounds']==case['sampled_coverage']['GT_cell_bounds'] and summary['weighted_GT_cells']==case['sampled_coverage']['weighted_GT_cells'],'same complete-cell geometry and overlap accounting')
    files=case['files']['sampled']
    for d in files.values():
        require(sha(d['path'])==d['sha256'],'independently accepted full sampled output bytes')
    p=np.load(files['probability.npy']['path'],mmap_mode='r');count=np.load(files['overlap.npy']['path'],mmap_mode='r');mask=np.load(files['mask.npy']['path'],mmap_mode='r')
    width=summary['GT_shape'][1];height=summary['GT_shape'][0];visited=0
    for block in summary['blocks']:
        start=block['start_GT_y']-summary['GT_cell_bounds'][1];rows=block['rows'];end=start+rows
        require(start==visited,'every finalized row exactly once');visited=end
        require(hashlib.sha256(np.ascontiguousarray(p[start:end]).tobytes()).hexdigest()==block['probability_float32_sha256'],'every prethreshold float32 probability unchanged')
        require(hashlib.sha256(np.ascontiguousarray(count[start:end]).tobytes()).hexdigest()==block['overlap_uint16_sha256'],'every native overlap count unchanged')
        expected=np.asarray(mask[start:end]);restored=unpack_rows(summary['mask_path'],slice(start,end),width)
        require(np.array_equal(expected,restored) and hashlib.sha256(expected.tobytes()).hexdigest()==block['binary_bool_sha256'],'all Boolean predictions restored exactly')
        bits=np.asarray(writer.packed[start:end]);require(hashlib.sha256(bits.tobytes()).hexdigest()==block['packed_uint8_sha256'],'saved packed block bytes match')
        if summary['padding_bits_per_row']:
            require(np.all((bits[:,-1]&((1<<summary['padding_bits_per_row'])-1))==0),'unused row padding is zero and excluded from metrics')
    require(visited==height,'all GT rows completed')
    summary.update(mask_sha256=sha(summary['mask_path']),exact_mask=True,exact_probabilities=True,exact_overlap=True)
    (Path(folder)/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def run(a):
    a.output.mkdir(exist_ok=False,parents=True);start=time.perf_counter();pid=os.getpid()
    receipt=read(a.acceptance/'summary.json');source=read(a.source/'summary.json')
    require(receipt['status']=='accepted_actual_interactive_native_GT_projection' and receipt['source_summary_sha256']==sha(a.source/'summary.json'),'independently accepted real model decoder/native output set')
    require(sha(a.source/'results.json')==source['results_sha256'] and sha(a.source/'outputs.json')==source['outputs_sha256'],'all frozen gate input records')
    results=read(a.source/'results.json');outputs=read(a.source/'outputs.json')
    data=read(a.root/'main30_full_1027_171918/slides_private.json')[0];evidence=[]
    def key(d):
        return tuple(d[k] for k in ('method','window','target','budget','variant'))
    for i,result in enumerate(results):
        records=[d for d in outputs if key(d)==key(result)];require(len(records)==4,'all fixed native contributions')
        summary=verify(a.output/f'case{i:03d}',result,records,data['native_wh'],data['gt_wh'])
        evidence.append(dict(identity=key(result),GT_cells=summary['GT_cells'],mask_path=summary['mask_path'],mask_sha256=summary['mask_sha256'],padding=summary['padding_bits_per_row']))
        (a.output/'status.json').write_text(json.dumps(dict(status='actual_packed_mask_and_probability_hash_gate',worker_pid=pid,completed=i+1,total=64))+'\n')
    records=[]
    for i,(y,x) in enumerate((y,x) for y in (5,13,19) for x in (5,13,21,24)):
        path=a.source/'anisotropic_fixture'/f'tile{i:02d}.npy';records.append(dict(path=str(path),sha256=sha(path),native_box=[x,y,8,8]))
    synthetic=verify(a.output/'anisotropic',read(a.source/'synthetic.json'),records,[37,29],[19,14])
    require(len(evidence)==64,'all methods/variants/budgets/windows')
    result=dict(status='passed_lossless_packed_prediction_gate',worker_pid=pid,cases=len(evidence),GT_cells=sum(r['GT_cells'] for r in evidence),sources={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/v5/wsi_global/packed_prediction.py'),Path('benchmarks/v5/wsi_global/sampled_stripe.py')]},source_acceptance_sha256=sha(a.acceptance/'summary.json'),evidence=evidence,synthetic_GT_cells=synthetic['GT_cells'],seconds=time.perf_counter()-start,
                scope='Lossless mask storage only; exact native probabilities/overlap block digests, same .5 decisions and every fixed scoring pixel; no GT labels, size filtering, probabilities quantized or speed claim',remaining='Integrate full30 interactive decoder and independent complete output/scoring acceptance, BF1, full-cell main30')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','source','acceptance','output'):
        p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
