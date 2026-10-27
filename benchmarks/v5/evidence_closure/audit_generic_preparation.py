"""Independent full generic-HE cache, training-only PCA and projection acceptance."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from .audit_heldout_local import read, sha, require


ARMS = ('generic_he_pca64','generic_he_white64')


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.time();pid=os.getpid()
    def dump(name,d):
        (a.output/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    dump('status.json',dict(status='preparing',worker_pid=pid))
    try:
        fit=a.root/'pca_full_train_1025_234700'
        assembly=a.root/'formal_controls_1026_001500/assembly_1026_001500'
        fm=read(fit/'run.json');fs=read(fit/'summary.json');manifest=read(assembly/'manifest.json')
        require(fs['status']=='complete_train_fit' and not fs['gate'],'full train PCA')
        require(sha(str(fit/'pca64.npz'))==fs['pca_sha256']==read(assembly/'summary.json')['pca_sha256'],'PCA identity')
        require(sha('benchmarks/v5/generic_cell_control/fit.py')==fm['source_sha256'],'frozen fitter source')
        require(sha('benchmarks/v5/generic_cell_control/assemble.py')==read(assembly/'summary.json')['source_sha256'],'frozen assembly source')
        for p,v in fm['raw_recipe']['sources'].items():require(sha(p)==v,'raw source changed')
        for info in fm['raw_recipe']['weights'].values():require(sha(info['path'])==info['sha256'],'raw weight changed')
        catalog=Path(fm['catalog'])/'tiles.parquet'
        require(sha(str(catalog))==fm['catalog_sha256'],'original catalog identity')
        fields={r['tile_id']:r for r in pd.read_parquet(catalog).to_dict('records')}
        require(len(fields)==1737,'full fields')
        records={}
        for folder in map(Path,fm['raw_shards']):
            done=read(folder/'summary.json');meta=read(folder/'run.json')
            require(done['status']=='complete' and sha(str(folder/'rows.jsonl'))==done['rows_sha256'],'raw manifest identity')
            require(meta['recipe']==fm['raw_recipe'],'common raw recipe')
            require(sha(str(folder/'selection.json'))==meta['selection_sha256']==done['selection_sha256'],'raw selection identity')
            selected={r['tile_id'] for r in read(folder/'selection.json')}
            shard=[json.loads(line) for line in (folder/'rows.jsonl').open()]
            require(len(shard)==3*len(selected) and {(r['tile_id'],r['scale']) for r in shard}=={(t,s) for t in selected for s in range(3)},'per-shard coverage')
            for r in shard:
                k=r['tile_id'],r['scale'];require(k not in records,'raw duplicates');records[k]=r
        require(set(records)=={(t,s) for t in fields for s in range(3)},'all5211 scales')
        train_records=[r for k,r in sorted(records.items()) if fields[k[0]]['split']=='train']
        require(read(fit/'training_inputs.json')==train_records,'exact train-only fitting list')
        require(sha(str(fit/'training_inputs.json'))==fs['training_input_manifest_sha256']==fm['training_inputs_sha256'],'fitting list SHA')
        with np.load(fit/'pca64.npz',allow_pickle=False) as z:model={k:z[k].copy() for k in z.files}
        mean=model['mean'];components=model['components'];values=model['eigenvalues']
        require(mean.shape==(768,) and components.shape==(64,768) and values.shape==(64,) and (values>0).all(),'PCA shape/positive variance')
        require(np.allclose(components@components.T,np.eye(64),atol=1e-10,rtol=0),'orthogonality')
        signature=dict(source_sha256=sha(__file__),catalog_sha256=fm['catalog_sha256'],pca_sha256=fs['pca_sha256'],assembly_manifest_sha256=sha(str(assembly/'manifest.json')))
        if not a.gate:
            require(a.verified_gate is not None,'real representative gate required')
            gate=read(a.verified_gate/'summary.json')
            require(gate['status']=='passed_preparation_gate' and gate['signature']==signature,'same source/data full gate')
        changed={}
        for arm in ARMS:
            info=manifest[arm];p=Path(info['catalog'])/'tiles.parquet';tc=read(Path(info['tasks'])/'contract.json')
            require(sha(str(p))==tc['catalog_sha256'] and tc['generic_HE_arm']==arm,'new catalog/arm identity')
            require(sha(str(Path(info['realized'])/'training_plans.jsonl'))==info['coordinate_sha256']=='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe','original actual40k identity')
            require(sha(str(Path(info['tasks'])/'validation_maximum.json'))==info['validation_sha256'],'unchanged validation')
            changed[arm]={r['tile_id']:r for r in pd.read_parquet(p).to_dict('records')}
            require(set(changed[arm])==set(fields),'arm field coverage')
        identities=sorted(fields)
        if a.gate:
            identities=[min(t for t,r in fields.items() if r['dataset']==d and r['split']==s) for d in ('crc','tiger') for s in ('train','val')]
        dump('run.json',dict(worker_pid=pid,gate=a.gate,root=str(a.root),signature=signature,fields=len(identities),
            covariance_scope='Independent fixed-mean central second moment plus independently summed mean; train only',
            covariance_validation_inputs=0,projection_validation_inputs='all val checked in full run; never used for covariance'))
        n=0;total=np.zeros(768,np.float64);centered=np.zeros((768,768),np.float64)
        max_reference_error=0.;max_projection={arm:0. for arm in ARMS};scale_rows=0;projections=0
        with (a.output/'checked_fields.jsonl').open('x') as handle:
            for index,tile in enumerate(identities):
                original=fields[tile];variant=json.loads(original['variants_json'])['original']
                require(sha(variant['cells'])==variant['cells_sha256'] and sha(original['multiscale_cache'])==original['multiscale_sha256'],'original cache SHA')
                with np.load(variant['cells'],allow_pickle=False) as z:fine={k:z[k] for k in ('features','total','instances')}
                with np.load(original['multiscale_cache'],allow_pickle=False) as z:context={k:z[k] for k in z.files}
                prepared={}
                for arm in ARMS:
                    row=changed[arm][tile]
                    allowed={'variants_json','multiscale_cache','multiscale_sha256','cell_recipe_id'}
                    for k,v in original.items():
                        if k not in allowed:require(row[k]==v,'unchanged catalog field: '+k)
                    new_variant=json.loads(row['variants_json'])['original']
                    require({k:v for k,v in new_variant.items() if k not in ('cells','cells_sha256')}=={k:v for k,v in variant.items() if k not in ('cells','cells_sha256')},'unchanged fine RGB provenance')
                    require(sha(new_variant['cells'])==new_variant['cells_sha256'] and sha(row['multiscale_cache'])==row['multiscale_sha256'],'new cache SHA')
                    with np.load(new_variant['cells'],allow_pickle=False) as z:nf={k:z[k] for k in z.files}
                    with np.load(row['multiscale_cache'],allow_pickle=False) as z:nc={k:z[k] for k in z.files}
                    require(set(nc)==set(context),'context key parity')
                    for k,v in context.items():
                        if k not in ('cells_1','cells_2'):require(np.array_equal(nc[k],v),'unchanged context '+k)
                    require(int(nf['total'])==int(fine['total']),'fine total unchanged')
                    prepared[arm]=nf,nc
                for scale in range(3):
                    record=records[tile,scale]
                    require(record['split']==original['split'] and record['dataset']==original['dataset'],'raw provenance')
                    require(sha(record['output'])==record['output_sha256'],'raw cache SHA')
                    with np.load(record['output'],allow_pickle=False) as z:
                        raw=z['raw768'].astype(np.float64);geometry=z['geometry10'];reference=z['reference74'];count=int(z['total'])
                        if scale==0:require(np.array_equal(z['instances'],fine['instances']),'fine nuclei identities')
                    old=fine['features'] if scale==0 else context[f'cells_{scale}']
                    old_total=int(fine['total']) if scale==0 else int(context[f'total_{scale}'])
                    require(raw.shape==(len(old),768) and geometry.shape==(len(old),10) and np.isfinite(raw).all(),'raw768 shape/finite')
                    require(count==old_total==record['total'] and len(raw)==record['cells'],'nucleus counts')
                    require(np.array_equal(geometry,old[:,:10]),'same retained geometric identities')
                    error=float(np.max(np.abs(reference-old))) if old.size else 0.
                    require(error<=2e-5 and error==record['reference_max_error'],'reference74 replay')
                    max_reference_error=max(max_reference_error,error)
                    centered_raw=raw-mean
                    if original['split']=='train':
                        total+=raw.sum(0);centered+=centered_raw.T@centered_raw;n+=len(raw)
                    projection=centered_raw@components.T
                    for arm in ARMS:
                        expected=projection/np.sqrt(values) if arm=='generic_he_white64' else projection
                        nf,nc=prepared[arm];observed=nf['features'] if scale==0 else nc[f'cells_{scale}']
                        require(observed.shape==old.shape and np.isfinite(observed).all() and np.array_equal(observed[:,:10],geometry),'same74 interface')
                        delta=float(np.max(np.abs(observed[:,10:]-expected))) if expected.size else 0.
                        require(np.allclose(observed[:,10:],expected,atol=2e-6,rtol=1e-5),'independent PCA projection')
                        max_projection[arm]=max(max_projection[arm],delta);projections+=1
                    scale_rows+=1
                handle.write(json.dumps(dict(tile_id=tile,split=original['split'],scales=3,projections=6))+'\n');handle.flush()
                if index%25==0:
                    dump('status.json',dict(status='checking_cache_and_train_covariance',worker_pid=pid,fields=index+1,total_fields=len(identities),scale_rows=scale_rows,projections=projections))
                    print(json.dumps(dict(fields=index+1,scale_rows=scale_rows,train_cells=n)),flush=True)
        covariance_report=None
        if not a.gate:
            require(n==fs['train_cells']==int(model['train_cells'])==901192 and len(train_records)==fs['train_scale_rows']==4068,'training cells/scales')
            covariance=centered/(n-1)
            mean_error=float(np.max(np.abs(total/n-mean)))
            require(mean_error<2e-10,'independent training mean')
            eigenvalues=np.linalg.eigvalsh(covariance)[::-1][:64]
            require(np.allclose(eigenvalues,values,atol=1e-10,rtol=1e-7),'independent top64 variances')
            residual=float(np.max(np.abs(covariance@components.T-components.T*values)))
            require(residual<1e-10,'PCA eigenvector/covariance residual')
            explained=float(values.sum()/np.trace(covariance))
            require(abs(explained-fs['explained_variance_fraction'])<1e-10,'explained variance')
            covariance_report=dict(train_cells=n,train_scale_rows=len(train_records),validation_covariance_rows=0,mean_max_error=mean_error,
                eigenvalue_max_error=float(np.max(np.abs(eigenvalues-values))),eigenvector_residual=residual,explained_variance_fraction=explained)
        result=dict(status='passed_preparation_gate' if a.gate else 'accepted_full_generic_preparation_qualified',worker_pid=pid,signature=signature,
            fields=len(identities),scale_rows=scale_rows,projections=projections,max_reference_error=max_reference_error,max_projection_errors=max_projection,
            independent_covariance=covariance_report,seconds=time.time()-start,
            limits=['No native GPU feature extraction rerun; raw source/weight hashes and saved identity parity checked',
                    'XCell reference74 used only for identity parity, not transform input','PCA fits current training RGB-derived features; not equal total pretraining',
                    'This accepts preparation, not paired statistics or all G6 requirements'])
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps(result),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--verified-gate',type=Path);run(p.parse_args())
