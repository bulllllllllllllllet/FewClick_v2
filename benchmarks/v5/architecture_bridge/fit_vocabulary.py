"""Fit a single frozen train-only K128 vocabulary; separate independent distance replay."""
import argparse,json,os
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from benchmarks.v5.encoder_research.build_joint_task_manifest import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',action='store_true');p.add_argument('--accept',action='store_true');a=p.parse_args()
    receipt=json.loads((a.samples/'independent_acceptance.json').read_text());assert receipt['validation_samples']==0 and not receipt['gate']
    assert sha(a.samples/'samples.npz')==receipt['array_sha256'] and sha(a.samples/'samples.jsonl')==receipt['index_sha256']
    with np.load(a.samples/'samples.npz',allow_pickle=False) as f:x=f['xcell'].copy()
    if a.gate:x=x[:2048]
    assert x.shape[1]==64 and len(x)>=128 and np.isfinite(x).all()
    if a.accept:
        report=json.loads((a.output/'report.json').read_text());assert report['samples_sha256']==receipt['array_sha256'] and report['source_sha256']==sha(__file__)
        with np.load(a.output/'vocabulary.npz',allow_pickle=False) as f:centers=f['centers'];labels=f['labels']
        assert centers.shape==(128,64) and len(labels)==len(x) and np.isfinite(centers).all()
        inertia=0.
        for i in range(0,len(x),256):
            distance=((x[i:i+256,None].astype(np.float64)-centers[None].astype(np.float64))**2).sum(2)
            assigned=distance[np.arange(len(distance)),labels[i:i+256]]
            assert np.all(assigned<=distance.min(1)+1e-5);inertia+=float(assigned.sum())
        assert abs(inertia-report['inertia'])/max(1.,inertia)<1e-5
        (a.output/'independent_acceptance.json').write_text(json.dumps(dict(status='accepted_train_only_K128_gate' if a.gate else 'accepted_train_only_K128_vocabulary',samples=len(x),validation_samples=0,encoder='xcell_standardized64',vocabulary_sha256=sha(a.output/'vocabulary.npz'),independent_inertia=inertia,occupied_codes=int(len(np.unique(labels))),worker_pid=os.getpid()),indent=2)+'\n');return
    a.output.mkdir(exist_ok=False);model=KMeans(n_clusters=128,n_init=10,max_iter=100,tol=1e-4,random_state=20261103,algorithm='lloyd')
    labels=model.fit_predict(x);np.savez_compressed(a.output/'vocabulary.npz',centers=model.cluster_centers_,labels=labels)
    with np.load(a.output/'vocabulary.npz',allow_pickle=False) as f:assert np.array_equal(f['centers'],model.cluster_centers_) and np.array_equal(f['labels'],labels)
    (a.output/'report.json').write_text(json.dumps(dict(status='complete_train_only_K128_unaccepted',samples=len(x),gate=a.gate,encoder='xcell_standardized64',samples_sha256=receipt['array_sha256'],source_sha256=sha(__file__),inertia=float(model.inertia_),iterations=int(model.n_iter_),validation_samples=0,worker_pid=os.getpid()),indent=2)+'\n')


if __name__=='__main__':main()
