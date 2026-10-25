"""Fit a fixed PCA64 from training-cell raw features, never validation features."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from benchmarks.v5.evidence_closure.audit_external import require
from benchmarks.v5.fair_comparison.audit import digest, write


def read(path): return json.loads(Path(path).read_text())


def project(raw, model, whiten=False):
    value = (np.asarray(raw, np.float64)-model['mean']) @ model['components'].T
    if whiten: value = value/np.sqrt(model['eigenvalues'])
    require(value.shape == (len(raw),64) and np.isfinite(value).all(), 'invalid genericHE projection')
    return value.astype(np.float32)


def run(catalog, shards, output, gate=False, wait=False):
    output.mkdir(parents=True, exist_ok=False)
    if wait:
        require(not gate, 'fit gate must already have finished raw inputs')
        while True:
            complete = True
            for shard in shards:
                path = shard/'status.json'
                if not path.exists():
                    raise ValueError('raw shard has not started: '+str(shard))
                try: state = read(path)
                except json.JSONDecodeError:
                    complete = False; continue
                require(state['status'] != 'failed', 'raw shard failed: '+str(shard))
                terminal = state['status'] == 'complete' and shard.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0')
                if not terminal:
                    require(Path('/proc/'+str(state['worker_pid'])).exists(), 'raw shard worker missing before terminal evidence')
                    complete = False
            if complete: break
            write(output/'status.json',dict(status='waiting_live_raw_shards',worker_pid=os.getpid(),shards=list(map(str,shards)),observed_at=time.time()))
            time.sleep(5)
    fields = {r['tile_id']:r for r in pd.read_parquet(catalog/'tiles.parquet').to_dict('records')}
    require(len(fields) == 1737, 'fixed catalog required')
    records = {}; recipe = None
    for shard in shards:
        result = read(shard/'summary.json'); meta = read(shard/'run.json')
        reused_complete_original_gate = (not gate and result['gate'] and
            meta.get('history_name') == 'context_gate_1025_035900' and result['fields'] == 4)
        require(result['status'] == 'complete' and (result['gate'] == gate or reused_complete_original_gate) and
                result['rows_sha256'] == digest(shard/'rows.jsonl'), 'raw shard incomplete/changed')
        require(meta['recipe']['catalog_sha256'] == digest(catalog/'tiles.parquet'), 'raw catalog identity')
        if recipe is None: recipe = meta['recipe']
        else: require(recipe == meta['recipe'], 'raw preparation recipes differ')
        selection = read(shard/'selection.json')
        require(digest(shard/'selection.json') == meta['selection_sha256'] == result['selection_sha256'], 'raw selection SHA')
        selected_ids = {r['tile_id'] for r in selection}
        if reused_complete_original_gate:
            original_gate_ids = {tile for tile,field in fields.items()
                if Path(field['multiscale_cache']).parent.name == 'context_gate_1025_035900'}
            require(selected_ids == original_gate_ids and len(original_gate_ids) == 4,
                    'only the complete four-field original preparation gate can be reused')
        shard_records = [json.loads(line) for line in (shard/'rows.jsonl').open()]
        require(len(shard_records) == result['scale_rows'] == 3*len(selected_ids), 'raw shard row count')
        require({(r['tile_id'],r['scale']) for r in shard_records} == {(tile,s) for tile in selected_ids for s in range(3)}, 'raw field/scale coverage')
        for record in shard_records:
            tile = record['tile_id']; k = tile, record['scale']
            require(k not in records and tile in fields and record['dataset'] == fields[tile]['dataset'] and
                    record['split'] == fields[tile]['split'], 'raw duplicate/split provenance')
            require(record['reference_max_error'] <= 2e-5, 'raw identity parity')
            records[k] = record
    expected = {(tile,s) for tile in fields for s in range(3)}
    if gate:
        require(len(records) == 12 and {r['split'] for r in records.values()} == {'train','val'}, 'gate subset')
    else: require(set(records) == expected, 'all1737 fields and5211 scales required')
    training = [r for k,r in sorted(records.items()) if fields[k[0]]['split'] == 'train']
    write(output/'training_inputs.json',training)
    write(output/'run.json',dict(worker_pid=os.getpid(),gate=gate,catalog=str(catalog),catalog_sha256=digest(catalog/'tiles.parquet'),
        raw_shards=list(map(str,shards)),raw_recipe=recipe,training_inputs_sha256=digest(output/'training_inputs.json'),
        source_sha256=digest(__file__),fit_scope='Only train raw768 is opened for fitting; no XCell/reference74/GT features loaded',
        solver='float64 centered batch covariance merge + symmetric eigh top64; deterministic loading signs',
        prespecified_variants=['PCA64 unwhitened','same PCA64 whitened sensitivity'],validation_model_selection=False))
    n = 0; mean = np.zeros(768,np.float64); centered_sum = np.zeros((768,768),np.float64)
    for index,record in enumerate(training):
        require(fields[record['tile_id']]['split'] == record['split'] == 'train', 'validation leakage into PCA')
        require(digest(record['output']) == record['output_sha256'], 'training raw cache changed')
        with np.load(record['output'],allow_pickle=False) as cache:
            # Do not load reference74: it contains the aligned representation being controlled.
            raw = cache['raw768'].astype(np.float64)
        require(raw.shape == (record['cells'],768) and np.isfinite(raw).all(), 'training raw geometry/finite')
        if len(raw):
            batch_mean = raw.mean(0); x = raw-batch_mean; batch_centered = x.T@x
            delta = batch_mean-mean; count = len(raw)
            centered_sum += batch_centered + np.outer(delta,delta)*(n*count/(n+count))
            mean += delta*(count/(n+count)); n += count
        if index%50 == 0:
            write(output/'status.json',dict(status='fitting_train_only',worker_pid=os.getpid(),train_rows=index+1,
                total_train_rows=len(training),train_cells=n))
    require(n > 768, 'insufficient training cells for fixed PCA64 covariance audit')
    covariance = centered_sum/(n-1)
    eigenvalues,eigenvectors = np.linalg.eigh(covariance)
    selected = np.argsort(eigenvalues)[::-1][:64]; values = eigenvalues[selected]; components = eigenvectors[:,selected].T
    require((values > 0).all() and np.isfinite(values).all(), 'nonpositive leading PCA variance')
    for row in components:
        if row[np.argmax(np.abs(row))] < 0: row *= -1
    require(np.allclose(components@components.T,np.eye(64),atol=1e-10,rtol=0), 'PCA orthogonality')
    projected_covariance = components@covariance@components.T
    require(np.allclose(projected_covariance,np.diag(values),atol=1e-10,rtol=1e-8), 'PCA covariance replay')
    whitened_covariance = projected_covariance/np.sqrt(np.outer(values,values))
    require(np.allclose(whitened_covariance,np.eye(64),atol=1e-8,rtol=0), 'whitening covariance replay')
    path = output/'pca64.npz';np.savez(path,mean=mean,components=components,eigenvalues=values,train_cells=np.asarray(n))
    with np.load(path,allow_pickle=False) as cache:
        model = {k:cache[k].copy() for k in cache.files}
    require(np.array_equal(model['mean'],mean) and np.array_equal(model['components'],components), 'PCA disk reload')
    transformed = []
    for record in training[:2]:
        with np.load(record['output'],allow_pickle=False) as cache: raw=cache['raw768']
        plain = project(raw,model,False); white = project(raw,model,True)
        require(np.allclose(white*np.sqrt(values),plain,atol=1e-7,rtol=1e-5), 'projection variants replay')
        transformed.append(dict(tile_id=record['tile_id'],scale=record['scale'],cells=len(raw),
            plain_norm_mean=float(np.linalg.norm(plain,axis=1).mean()) if len(raw) else None,
            whitened_norm_mean=float(np.linalg.norm(white,axis=1).mean()) if len(raw) else None))
    write(output/'summary.json',dict(status='complete_gate' if gate else 'complete_train_fit',gate=gate,worker_pid=os.getpid(),
        pca_sha256=digest(path),training_input_manifest_sha256=digest(output/'training_inputs.json'),
        train_scale_rows=len(training),train_cells=n,validation_raw_files_opened_for_fit=0,
        explained_variance_fraction=float(values.sum()/np.trace(covariance)),eigenvalues=values.tolist(),
        projection_checks=transformed,source_sha256=digest(__file__),full_model_training_complete=False,
        scope='Only training-cell transform fitted; full raw validation cache/model/score acceptance remains separate'))
    write(output/'status.json',dict(status='complete_gate' if gate else 'complete_train_fit',worker_pid=os.getpid(),train_cells=n))
    print(json.dumps(dict(status='complete_gate' if gate else 'complete_train_fit',train_cells=n,output=str(output))),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--shards',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--wait-for-shards',action='store_true');a=p.parse_args()
    if a.output.exists(): raise SystemExit('output already exists; preserve previous artifacts')
    try:run(a.catalog,a.shards,a.output,a.gate,a.wait_for_shards)
    except Exception:
        if a.output.is_dir():
            failure=dict(status='failed',worker_pid=os.getpid(),traceback=traceback.format_exc())
            write(a.output/'failure.json',failure);write(a.output/'status.json',failure)
        raise
