"""Evaluate synthetic colors using immutable existing PCA, probe and train gallery."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score,balanced_accuracy_score
import torch

from .continuous_data import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert json.loads((a.features/'stage.json').read_text())['status']=='complete'
    a.output.mkdir(exist_ok=False);torch.set_num_threads(2);results={}
    base=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')
    for arm in ['ctranspath','lemon','phikon']:
        folder=base/'cell_linear_probe_1031_012400/data'/arm
        report=json.loads((a.features/arm/'report.json').read_text())
        assert report['feature_sha256']==sha(a.features/arm/'features.npz')
        old_path=base/'cell_features_1031_012400'/arm/'features.npz'
        assert report['prior_feature_sha256']==sha(old_path)
        old=np.load(old_path,allow_pickle=False);new=np.load(a.features/arm/'features.npz',allow_pickle=False)
        transform=np.load(folder/'train_only_transform.npz',allow_pickle=False)
        def project(x):
            z=(x.astype(np.float64)-transform['mean'])/transform['std']
            return ((z-transform['pca_mean'])@transform['components'].T).astype(np.float32)
        train=old['splits']=='train';val=old['splits']=='val';true=old['labels'][val]-1;labels=old['labels'][train]-1;ids=old['cell_ids'][val];organs=old['organs'][val]
        assert (train.sum(),val.sum())==(384,186)
        gallery=project(old['features'][train]);gallery/=np.maximum(np.linalg.norm(gallery,axis=1,keepdims=True),1e-12)
        head=torch.nn.Linear(64,2);head.load_state_dict(torch.load(folder/'paired.pt',weights_only=True));head.eval()
        prior=np.load(folder/'paired_predictions.npz',allow_pickle=False)
        conditions=['identity','OD_RG_plus','OD_RG_minus','OD_light','OD_dark'];rows=[];saved={};baseline=None;original=None
        for condition in conditions:
            subset=new['conditions']==condition;features=new['features'][subset]
            assert subset.sum()==186 and np.array_equal(new['cell_ids'][subset],ids) and np.isfinite(features).all()
            projected=project(features)
            with torch.no_grad(): logits=head(torch.from_numpy(projected)).numpy()
            pred=logits.argmax(1)
            if condition=='identity':
                assert np.allclose(features,old['features'][val],atol=1e-5,rtol=1e-5)
                assert np.array_equal(pred,prior['pred']) and np.allclose(logits,prior['logits'],atol=1e-5,rtol=1e-5)
                baseline=pred;original=features
            query=projected/np.maximum(np.linalg.norm(projected,axis=1,keepdims=True),1e-12)
            neighbors=np.argsort(-(query@gallery.T),axis=1)[:,:5]
            cosine=(features*original).sum(1)/np.maximum(np.linalg.norm(features,axis=1)*np.linalg.norm(original,axis=1),1e-12)
            rows.append(dict(condition=condition,n=186,macro_F1=float(f1_score(true,pred,labels=[0,1],average='macro',zero_division=0)),
                             balanced_accuracy=float(balanced_accuracy_score(true,pred)),precision_at5=float(np.mean(labels[neighbors]==true[:,None])),
                             prediction_flip_fraction=float(np.mean(pred!=baseline)),feature_cosine_mean=float(cosine.mean()),
                             per_organ={str(o):dict(n=int((organs==o).sum()),macro_F1=float(f1_score(true[organs==o],pred[organs==o],labels=[0,1],average='macro',zero_division=0))) for o in np.unique(organs)}))
            saved[condition+'_pred']=pred;saved[condition+'_logits']=logits
        np.savez_compressed(a.output/(arm+'_predictions.npz'),cell_ids=ids,true=true,**saved)
        disk=np.load(a.output/(arm+'_predictions.npz'),allow_pickle=False)
        for row in rows:
            pred=disk[row['condition']+'_pred'];assert row['macro_F1']==float(f1_score(disk['true'],pred,labels=[0,1],average='macro',zero_division=0))
        results[arm]=dict(rows=rows,timing=report['timing'],identity_predictions_exact=True,
                         bindings={str(p):sha(p) for p in [folder/'paired.pt',folder/'train_only_transform.npz',a.features/arm/'features.npz',a.features/arm/'report.json',old_path]})
    (a.output/'report.json').write_text(json.dumps(dict(status='accepted_fixed_color_probe_evaluation',worker_pid=os.getpid(),source_sha256=sha(__file__),arms=results,
        scope='synthetic color dev probe; 186 cells are not independent patients; CTransPath point control is not native XCell; no tissue segmentation or external stain generalization claim'),indent=2)+'\n')


if __name__=='__main__':main()
