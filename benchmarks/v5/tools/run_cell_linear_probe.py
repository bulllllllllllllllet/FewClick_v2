"""Fixed train-only PCA64 and equal-budget paired/shuffled linear probes."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score,balanced_accuracy_score
import torch
import torch.nn.functional as F


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);results=[];reference=None
    protocol=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/cell_probe_protocol_1031_010248/contract.json')
    assert json.loads(protocol.read_text())['linear_training']['updates']==500
    for arm in ['ctranspath','lemon','phikon']:
        acceptance=json.loads((a.features/'independent_acceptance.json').read_text());assert acceptance['status']=='accepted_three570_common_input_features'
        with np.load(a.features/arm/'features.npz',allow_pickle=False) as f:
            features=f['features'].astype('float64');ids=f['cell_ids'].copy();splits=f['splits'].copy();labels=f['labels'].copy()-1;organs=f['organs'].copy()
        if reference is None:reference=ids
        else:assert np.array_equal(reference,ids)
        train=splits=='train';dev=splits=='val';assert (train.sum(),dev.sum())==(384,186)
        mean=features[train].mean(0);std=features[train].std(0);std[std==0]=1
        standardized=(features-mean)/std;pca=PCA(n_components=64,svd_solver='full');pca.fit(standardized[train])
        projected=pca.transform(standardized).astype('float32')
        folder=a.output/arm;folder.mkdir()
        np.savez_compressed(folder/'train_only_transform.npz',mean=mean,std=std,pca_mean=pca.mean_,components=pca.components_)
        x=torch.from_numpy(projected[train]);v=torch.from_numpy(projected[dev]);true=labels[dev]
        for condition in ['paired','shuffled']:
            torch.manual_seed(20261031);target=labels[train].copy()
            if condition=='shuffled':target=target[np.random.default_rng(20261031).permutation(len(target))]
            y=torch.tensor(target,dtype=torch.long);model=torch.nn.Linear(64,2);before=model.weight.detach().clone()
            optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.0001)
            for step in range(500):
                optimizer.zero_grad();loss=F.cross_entropy(model(x),y);assert torch.isfinite(loss)
                loss.backward();assert torch.isfinite(model.weight.grad).all();optimizer.step()
            assert not torch.equal(before,model.weight)
            model.eval()
            with torch.inference_mode():logits=model(v);pred=logits.argmax(1).numpy()
            checkpoint=folder/f'{condition}.pt';torch.save(model.state_dict(),checkpoint)
            restored=torch.nn.Linear(64,2);restored.load_state_dict(torch.load(checkpoint,weights_only=True));restored.eval()
            with torch.inference_mode():assert torch.equal(logits,restored(v))
            gallery=projected[train];query=projected[dev]
            cosine=(query/np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-12))@(gallery/np.maximum(np.linalg.norm(gallery,axis=1,keepdims=True),1e-12)).T
            neighbors=np.argsort(-cosine,axis=1)[:,:5];p5=float(np.mean(target[neighbors]==true[:,None]))
            row=dict(arm=arm,condition=condition,updates=500,train_tasks_consumed=500*384,seed=20261031,
                macro_F1=float(f1_score(true,pred,labels=[0,1],average='macro',zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(true,pred)),precision_at5_train_gallery=p5,
                per_organ={str(o):dict(n=int((organs[dev]==o).sum()),macro_F1=float(f1_score(true[organs[dev]==o],pred[organs[dev]==o],labels=[0,1],average='macro',zero_division=0))) for o in np.unique(organs[dev])},
                final_training_loss=float(loss.detach()),head_parameters=sum(p.numel() for p in model.parameters()),
                exact_disk_reload=True,checkpoint_sha256=sha(checkpoint),feature_sha256=sha(a.features/arm/'features.npz'))
            np.savez_compressed(folder/f'{condition}_predictions.npz',cell_ids=ids[dev],true=true,pred=pred,logits=logits.numpy())
            results.append(row);(a.output/'partial.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(row),flush=True)
    report=dict(status='complete_fixed_cell_probe_unaccepted',worker_pid=os.getpid(),rows=results,
                source_sha256=sha(__file__),protocol_sha256=sha(protocol),
                qualification='BC/TC developer probe only; not unchanged original XCell or tissue segmentation superiority; no threshold/checkpoint tuning')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
