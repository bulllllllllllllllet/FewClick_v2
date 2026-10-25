"""CPU public baselines on actual native external fields; GT stays in the scorer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
from PIL import Image
import torch
import yaml

from .prepare import digest, write
from benchmarks.v5.fair_comparison.metrics import score, VERSION
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v5.fair_comparison.public_baselines import SharedContextWSISAM, summarize
from benchmarks.v4.baseline.adapters.sam_adapter import SAMAdapter
from benchmarks.v4.baseline.adapters.base import EpisodeRequest


def inputs(row, contexts):
    """No labels, dataset, target ID, valid mask, or GT-dependent feature enters."""
    images = json.loads(row['images_json'])
    result = {}
    for scale, key in ((0,'fine'),(1,'low')) if contexts else ((0,'fine'),):
        entry = images[scale]
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('external RGB changed')
        image = np.asarray(Image.open(entry['path']).convert('RGB')).copy()
        if image.shape != (512,512,3):
            raise ValueError('external image shape')
        result[key] = image
    return result


def supervision(row, target):
    labels = np.asarray(Image.open(row['labels'])).copy()
    if hashlib.sha256(labels.tobytes()).hexdigest() != row['labels_array_sha256']:
        raise ValueError('external GT changed')
    return labels == target, labels != 255


def prompt_arrays(plan):
    pos = np.asarray(plan['positive_xy'],np.float32)*512
    neg = np.asarray(plan['negative_xy'],np.float32)*512
    if not np.isfinite(np.concatenate((pos,neg))).all():
        raise ValueError('nonfinite points')
    return pos,neg


class PublicModel:
    def __init__(self, method):
        self.method = method
        path = Path('benchmarks/v4/baseline/configs')/('sam_zero_shot.yaml' if method=='sam' else 'wsi_sam_zero_shot.yaml')
        self.config = yaml.safe_load(path.read_text())
        self.config['device'] = 'cpu'
        self.config_sha256 = digest(path)
        self.cached_id = None
        if method == 'sam':
            self.model = SAMAdapter(self.config)
            self.prior = self.model.provenance()
        else:
            for checkpoint in self.config['checkpoints'].values():
                if digest(checkpoint['path']) != checkpoint['sha256']:
                    raise ValueError('public checkpoint changed')
            self.model = SharedContextWSISAM(self.config)
            self.prior = self.model.load_report

    @torch.inference_mode()
    def predict(self, data, pos, neg, identity):
        if self.method == 'wsi_sam':
            result = self.model.predict(data,pos,neg,'point',None)
            return result['probability'],result['candidates']
        if self.cached_id != identity:
            self.model.predictor.set_image(data['fine'])
            self.cached_id = identity
        coords = np.concatenate((pos,neg)).astype(np.float32)
        signs = np.concatenate((np.ones(len(pos),np.int32),np.zeros(len(neg),np.int32)))
        masks, quality, _ = self.model.predictor.predict(point_coords=coords,point_labels=signs,
            box=None,multimask_output=True,return_logits=True)
        if not np.isfinite(quality).all():
            raise ValueError('invalid public mask quality')
        selected = int(np.argmax(quality))
        probability = (1/(1+np.exp(-np.asarray(masks[selected],np.float32)))).astype(np.float32)
        return probability,dict(selection_rule='argmax_predicted_iou',selected_index=selected,
                                predicted_iou=np.asarray(quality).tolist(),candidate_count=len(masks))


def run(tasks, method, output, gate=False, verified_gate=None):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    started = time.time()
    sources = {str(p):digest(p) for p in Path(__file__).parent.glob('*.py')}
    try:
        contract = json.loads((tasks/'contract.json').read_text())
        if digest(tasks/'fields.parquet') != contract['fields_sha256'] or digest(tasks/'local_maximum.json') != contract['clicks_sha256']:
            raise ValueError('external contract changed')
        if not gate:
            if verified_gate is None:
                raise ValueError('formal scoring requires real gate')
            evidence = json.loads((verified_gate/'summary.json').read_text())
            if evidence['status'] != 'complete' or not evidence['gate'] or evidence['method'] != method:
                raise ValueError('gate incomplete')
            previous = json.loads((verified_gate/'run.json').read_text())
            if previous['source_sha256'] != sources or previous['protocol_sha256'] != digest(tasks/'contract.json'):
                raise ValueError('code/protocol changed since gate')
        fields = pd.read_parquet(tasks/'fields.parquet').to_dict('records')
        by_id = {r['tile_id']:r for r in fields}
        plans = json.loads((tasks/'local_maximum.json').read_text())
        if gate:
            chosen, seen = [], set()
            for plan in plans:
                if plan['query_id'] not in seen:
                    chosen.append(plan)
                    seen.add(plan['query_id'])
                if len(chosen) == 2:
                    break
            plans = chosen
        model = PublicModel(method)
        write(output/'run.json',dict(method=method,gate=gate,worker_pid=os.getpid(),device='cpu',
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),protocol_sha256=digest(tasks/'contract.json'),
            source_sha256=sources,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            config=model.config,original_config_sha256=model.config_sha256,prior=model.prior,
            metric_version=VERSION,adaptation_steps=0,GT_in_forward=False,
            timing='CPU including input read; SAM image embedding cached per field; not GPU/WSI efficiency'))
        results, equivalence = [], []
        with (output/'rows.jsonl').open('x') as stream:
            for original in plans:
                if original['support_id'] != original['query_id'] or original['kind'] != 'local':
                    raise ValueError('external local identity mismatch')
                row = by_id[original['query_id']]
                # This real gate makes both loader and forward run with an invalid GT path.
                input_row = row | dict(labels='/deliberately_unavailable/external_GT.png') if gate else row
                for budget in contract['report_budgets']:
                    plan = nested(original,budget)
                    tick = time.perf_counter()
                    data = inputs(input_row,method=='wsi_sam')
                    pos,neg = prompt_arrays(plan)
                    probability,candidates = model.predict(data,pos,neg,row['tile_id'])
                    elapsed = 1000*(time.perf_counter()-tick)
                    if probability.shape != (512,512) or not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
                        raise ValueError('invalid external probability')
                    if gate and method=='sam' and budget=='3+2':
                        request = EpisodeRequest(str(plan['number']),data['fine'],pos,neg,'point')
                        request.validate()
                        direct = model.model.predict(request).probability
                        error = float(np.max(np.abs(probability-direct)))
                        if error>1e-7 or not np.array_equal(probability>=.5,direct>=.5):
                            raise ValueError('cached/native SAM path differs')
                        equivalence.append(error)
                    # GT is accessed only after forward returns.
                    truth,valid = supervision(row,plan['target'])
                    result = score(probability>=.5,truth,valid,json.loads(row['input_mpp_xy_json']))
                    if gate:
                        path = output/f'prediction_{plan["pair_id"]}_{budget}.npy'
                        np.save(path,probability)
                        saved = np.load(path,allow_pickle=False)>=.5
                        direct_counts = dict(tp=int((saved&truth&valid).sum()),fp=int((saved&~truth&valid).sum()),fn=int((~saved&truth&valid).sum()))
                        if any(result[key]!=value for key,value in direct_counts.items()):
                            raise ValueError('saved prediction independent counts differ')
                    result.update(plan,method=method,latency_ms=elapsed,candidates=candidates,
                        source_id=row['source_id'],space_key=row['source_id'],spatial_group_id=row['source_id'])
                    results.append(result)
                    stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
                    stream.flush()
                write(output/'status.json',dict(status='running',worker_pid=os.getpid(),completed=len(results),total=len(plans)*4,seconds=time.time()-started))
                print(json.dumps(dict(method=method,completed=len(results),total=len(plans)*4)),flush=True)
        write(output/'summary.json',dict(status='complete',worker_pid=os.getpid(),method=method,gate=gate,
            rows=len(results),rows_sha256=digest(output/'rows.jsonl'),metrics=summarize(results),
            GT_unavailable_forward_passed=gate,native_sam_max_errors=equivalence,
            independent_saved_count_replay=gate,seconds=time.time()-started,
            scope='external small native-context ROI subset; zero adaptation; different public priors/views'))
        write(output/'status.json',dict(status='complete',worker_pid=os.getpid(),rows=len(results)))
    except Exception:
        write(output/'status.json',dict(status='failed',worker_pid=os.getpid(),traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks',type=Path,required=True)
    parser.add_argument('--method',choices=['sam','wsi_sam'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gate',action='store_true')
    parser.add_argument('--verified-gate',type=Path)
    args = parser.parse_args()
    run(args.tasks,args.method,args.output,args.gate,args.verified_gate)
