#!/usr/bin/env python3
"""Fixed-regularization linear control for target-32 and target-32+patch-512."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize

from benchmarks.v5.tools.run_crc_panel1_cell_identity import CLASSES, score
from benchmarks.v5.tools.train_crc_panel1_fullpatch_cross_attention import BASE, SMALL, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shards', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not a.output.resolve().is_relative_to(BASE):
        p.error('output must be under artifact root')
    a.output.mkdir(parents=True, exist_ok=False)
    rows = pd.read_csv(SMALL/'cells.csv')
    with np.load(SMALL/'features.npz',allow_pickle=False) as z:
        target = normalize(z['features_32'].astype(np.float32))
    if target.shape != (len(rows),768):
        raise ValueError('invalid target features')
    lookup = {}
    for shard in a.shards:
        audit = json.loads((shard/'audit.json').read_text())
        if audit['status'] != 'complete' or audit['source_small_cells_sha256'] != digest(SMALL/'cells.csv'):
            raise ValueError(f'invalid shard: {shard}')
        for rec in audit['records']:
            if rec['patch'] in lookup:
                raise ValueError('duplicate patch')
            file = shard/f"{rec['patch']}.npz"
            if digest(file) != rec['sha256']:
                raise ValueError(f'cache mismatch: {file}')
            lookup[rec['patch']] = file
    if set(rows.patch) != set(lookup):
        raise ValueError('patch coverage mismatch')
    global_features = np.empty_like(target)
    for patch, group in rows.groupby('patch'):
        with np.load(lookup[patch],allow_pickle=False) as z:
            value = normalize(z['patch512'].astype(np.float32)[None])[0]
        global_features[group.index] = value
    split = rows.arm_split.to_numpy(str)
    train = np.flatnonzero(split=='train')
    labels = rows.label.to_numpy(int)
    report = {'status':'complete','classes':CLASSES,'rows':len(rows),'train':len(train),
              'target_manifest_sha256':digest(SMALL/'cells.csv'),'classifier':'train-only StandardScaler + balanced LogisticRegression(C=1)',
              'arms':{}}
    preds = rows[['patch','he_id','arm_split','label']].copy()
    for arm,x in (('target32',target),('target32_patch512',np.concatenate([target,global_features],axis=1))):
        scaler = StandardScaler().fit(x[train])
        x = scaler.transform(x)
        clf = LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=20261008).fit(x[train],labels[train])
        prediction = clf.predict(x)
        preds[f'pred_{arm}'] = prediction
        report['arms'][arm] = {name:score(labels[split==name],prediction[split==name])
                               for name in ('val','test','supplement_val','supplement_test')}
        print(json.dumps({'arm':arm,'test':report['arms'][arm]['test']['macro_f1'],
                          'supplement_test':report['arms'][arm]['supplement_test']['macro_f1']}),flush=True)
    preds.to_csv(a.output/'predictions.csv',index=False)
    (a.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'event':'complete','output':str(a.output)}),flush=True)


if __name__=='__main__':
    main()
