"""Cached support-conditioned full-valid WSI output, separate from remote scores."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from .interactive_prediction import move, score_task
from .native_remote_gate import prompts
from .packed_prediction import PackedSampledStripe
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.transfer_comparison.prototypes import dense_prototype, region_prototype


METHODS = ('fewclick_m3d_image', 'sam_ft', 'wsi_sam_ft')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); pid = os.getpid()
    def dump(n, d):
        (a.output/n).write_text(json.dumps(d, indent=2, allow_nan=False)+'\n')
    try:
        # Match the accepted original CPU4 decoder; thread reductions can
        # change FP32 probability bytes even when thresholded masks agree.
        torch.set_num_threads(4)
        root = a.root; index = read(a.index/'summary.json'); method = index['recipe']['method']
        require(method in METHODS and (not a.cpu_fixture or a.gate), 'three image methods; CPU fixture only for a gate')
        require(index['status'] == ('passed_interactive_full_valid_index_fixture' if a.cpu_fixture else 'accepted_interactive_full_valid_cache_index_qualified'), 'accepted actual full-valid cache index')
        require(sha(a.index/'cache_manifest.jsonl') == index['cache_manifest_sha256'], 'bound complete cache index')
        device = torch.device(index['device'])
        require((device.type == 'cpu') == a.cpu_fixture, 'formal branch uses actual CUDA caches')
        if device.type == 'cuda':
            require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('1','3','4','5'), 'single allowed physical GPU')
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        else:
            require(not torch.cuda.is_available(), 'hide CUDA for CPU fixtures')
        protocol = root/'full_valid_protocol_1028_154400'; accepted = root/'full_valid_protocol_acceptance_1028_154500'
        ps = read(protocol/'summary.json'); pa = read(accepted/'summary.json')
        require(pa['status'] == 'accepted_full_valid_metadata_protocol_not_predictions' and pa['source_summary_sha256'] == sha(protocol/'summary.json'), 'all30 frozen full-valid task/domain protocol')
        for name, digest in ps['files'].items():
            require(sha(protocol/name) == digest, 'all frozen protocol files')
        base = Path(index['base_cache']); base_summary = read(base/'summary.json')
        require(sha(base/'summary.json') == index['base_summary_sha256'] and sha(Path(index['base_acceptance'])/'summary.json') == index['base_acceptance_sha256'], 'same independently accepted source/query caches')
        require(sha(Path(index['extension'])/'summary.json') == index['extension_summary_sha256'] and sha(Path(index['extension_acceptance'])/'summary.json') == index['extension_acceptance_sha256'], 'same accepted additional fields')
        sources = {str(p):sha(p) for p in [Path(__file__), Path('benchmarks/v5/wsi_global/interactive_prediction.py'), Path('benchmarks/v5/wsi_global/packed_prediction.py'), Path('benchmarks/v5/wsi_global/sampled_stripe.py'), Path('benchmarks/v5/wsi_global/native_remote_gate.py'), Path('benchmarks/v5/wsi_global/retrieval_stream.py'), Path('benchmarks/v5/wsi_global/retrieval_metrics.py'), Path('benchmarks/v5/transfer_comparison/prototypes.py')]}
        signature = dict(sources=sources, method=method, device=str(device), torch_cpu_threads=4, weight_sha256=index['weight_sha256'], index_summary_sha256=sha(a.index/'summary.json'), protocol_summary_sha256=sha(protocol/'summary.json'), evaluation_scope='complete_valid4x_including_support_separate_diagnostic', primary_summary_sha256=None, primary_acceptance_sha256=None)
        if not a.cpu_fixture:
            require(a.primary is not None and a.primary_acceptance is not None, 'accepted original primary predictions before formal full-valid branch')
            primary = read(a.primary/'summary.json'); receipt = read(a.primary_acceptance/'summary.json')
            require(receipt['status'] == 'accepted_interactive_main30_GT_scores_qualified' and receipt['source_summary_sha256'] == sha(a.primary/'summary.json'), 'independently accepted original all30 output')
            require(primary['signature']['method'] == method and primary['signature']['weight_sha256'] == index['weight_sha256'] and primary['signature']['device'] == str(device), 'same selected method/weight/device as original remote')
            signature.update(primary_summary_sha256=sha(a.primary/'summary.json'), primary_acceptance_sha256=sha(a.primary_acceptance/'summary.json'))
        if not a.gate:
            require(a.verified_gate is not None and a.gate_acceptance is not None, 'same actual GPU driver gate and independent acceptance')
            g = read(a.verified_gate/'summary.json'); ga = read(a.gate_acceptance/'summary.json')
            require(g['status'] == 'passed_full_valid_prediction_driver_gate' and g['signature'] == signature and ga['status'] == 'accepted_full_valid_prediction_gate' and ga['source_summary_sha256'] == sha(a.verified_gate/'summary.json'), 'same full input/weight/device/code gate accepted')
        entries = [json.loads(line) for line in (a.index/'cache_manifest.jsonl').open()]
        require(len(entries) == (11 if a.cpu_fixture else 28452), 'complete accepted fixture or formal full grid')
        base_entries = {e['identity']['key']:e for e in map(json.loads, (base/'cache_manifest.jsonl').open())}
        require(sha(base/'cache_manifest.jsonl') == base_summary['cache_manifest_sha256'], 'unchanged original support manifest')
        plans = read(root/'support_query_full_1027_173800/support_tasks_maximum.json')
        tasks = {(t['wsi_id'],t['target'],t['prompt_group'],t['budget']):t for t in map(json.loads,(protocol/'tasks.jsonl').open())}
        data = {r['wsi_id']:r for r in read(root/'main30_full_1027_171918/slides_private.json')}
        sid0 = next(iter(data)); first = [e for e in entries if e['identity']['wsi_id'] == sid0]
        if a.gate:
            work = [('new_left', sid0, [next(e for e in first if e['cache_origin'] == 'extension')]), ('source_alias', sid0, [next(e for e in first if e['identity']['kind'] == 'support')]), ('remote_single', sid0, [next(e for e in first if e['identity']['kind'] == 'query')])]
        else:
            work = [(sid,sid,[e for e in entries if e['identity']['wsi_id'] == sid]) for sid in data]
            for _, sid, group in work:
                require([e['full_valid_geometry']['tile'] for e in group] == [t['number'] for t in map(json.loads,Path(data[sid]['grid']).open())], 'complete original row-major full-valid domain')
        variants = ('graph_dense','region_prototype') if method == 'fewclick_m3d_image' else ('feature_prototype',)
        expected_rows = (24 if a.gate else 5520)*len(variants)
        dump('contract.json', dict(worker_pid=pid, signature=signature, index=str(a.index), primary=str(a.primary) if a.primary else None, primary_acceptance=str(a.primary_acceptance) if a.primary_acceptance else None, gate=a.gate, cpu_fixture=a.cpu_fixture, expected_rows=expected_rows, windows=[dict(item=n,wsi_id=sid,tiles=[e['full_valid_geometry']['tile'] for e in es]) for n,sid,es in work], inference_GT='No GT enters model, prompts, thresholds or candidate selection. Each task mask sealed before its labels opened.', failure_policy='stop preserve; no retry/resume/fallback'))
        adapt = Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208')
        best = read(adapt/f'adaptation_1025_131451/train_{method}_1025_131451/best_checkpoint.json')
        require(sha(best['path']) == best['checkpoint_sha256'] == index['weight_sha256'], 'actual fixed development-selected model')
        model = load(method,Path(best['path']),device,sha(adapt/'tasks_1025_125400/contract.json'))
        calls = [0]
        def hook(module, args):
            calls[0] += 1
        encoders = [e.backbone for e in model.core.encoders] if method == 'fewclick_m3d_image' else [model.sam.image_encoder]
        handles = [e.register_forward_pre_hook(hook) for e in encoders]
        rows = []; outputs = []
        with torch.inference_mode(), (a.output/'tile_probability_digests.jsonl').open('x') as mf:
            for item_no,(name,sid,group) in enumerate(work):
                r = data[sid]; selected = [p for p in plans if p['wsi_id'] == sid and (not a.gate or p['target'] in (0,10) and p['prompt_group'] == 0)]
                boxes = [e['full_valid_geometry']['native_boxes'][0] for e in group]
                bounds = [min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[0]+b[2] for b in boxes),max(b[1]+b[3] for b in boxes)]
                sx,sy = r['native_wh'][0]/r['gt_wh'][0],r['native_wh'][1]/r['gt_wh'][1]
                estimate = len(selected)*4*len(variants)*(int((bounds[3]-bounds[1])/sy)+1)*(int((bounds[2]-bounds[0])/sx)+8)/8
                require(shutil.disk_usage(a.output).free > 1.5*estimate, 'actual packed-mask disk headroom')
                if not a.gate:
                    available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
                    require(available > 3*sum(e['bytes'] for e in group)+8*1024**3, 'actual one-WSI cache RAM headroom')
                query_cpu = {}
                for e in group:
                    require(sha(e['path']) == e['sha256'], 'all actual accepted full-valid cache bytes')
                    payload = torch.load(e['path'],map_location='cpu',weights_only=False)
                    require(payload['identity'] == e['identity'] and payload['checkpoint_sha256'] == index['weight_sha256'], 'original identities preserved including support aliases')
                    query_cpu[e['full_valid_key']] = payload['encoded']
                for plan in selected:
                    e = base_entries[plan['support_id']]; require(sha(e['path']) == e['sha256'], 'same original support cache bytes')
                    payload = torch.load(e['path'],map_location='cpu',weights_only=False)
                    source_device = move(payload['encoded'],device)
                    for budget in ('3+2','1+1','3+1','5+1'):
                        for path,digest in sources.items():
                            require(sha(path) == digest, 'frozen prediction/scoring path')
                        p = move(prompts(plan,budget),device)
                        task = model.core.support_task(source_device,p) if method == 'fewclick_m3d_image' else None
                        folder = a.output/name/f'target{plan["target"]}_group{plan["prompt_group"]}_{budget}'
                        writers = {v:PackedSampledStripe(folder/v,bounds,boxes[0][2:],r['native_wh'],r['gt_wh']) for v in variants}
                        for entry in group:
                            query = move(query_cpu[entry['full_valid_key']],device)
                            if method == 'fewclick_m3d_image':
                                value,maps = model.core.decode(query,task,p,False)
                                require(not maps.count_nonzero(), 'no local query point maps even in source band')
                                probabilities = dict(graph_dense=value['pixel_probability'][0], region_prototype=region_prototype(source_device,query,p)['pixel_probability'][0])
                            else:
                                probabilities = dict(feature_prototype=dense_prototype(source_device,query,p)['pixel_probability'][0])
                            for variant,value in probabilities.items():
                                array = value.detach().float().cpu().numpy(); tile = entry['full_valid_geometry']['tile']
                                require(array.shape == (512,512) and np.isfinite(array).all(), 'actual finite512 decoder output')
                                writers[variant].push(array,entry['full_valid_geometry']['native_boxes'][0])
                                d = dict(item=name,wsi_id=sid,target=plan['target'],prompt_group=plan['prompt_group'],budget=budget,variant=variant,tile=tile,full_valid_key=entry['full_valid_key'],probability512_float32_sha256=hashlib.sha256(array.tobytes()).hexdigest())
                                if a.gate:
                                    path = folder/variant/f'tile{tile}_probability.npy'; np.save(path,array,allow_pickle=False)
                                    d.update(gate_probability_path=str(path),gate_probability_file_sha256=sha(path))
                                mf.write(json.dumps(d)+'\n')
                        mf.flush()
                        frozen = tasks[sid,plan['target'],plan['prompt_group'],budget]
                        for variant,writer in writers.items():
                            prediction = writer.finish(); prediction['mask_sha256'] = sha(prediction['mask_path'])
                            pred_file = folder/variant/'prediction.json'; pred_file.write_text(json.dumps(prediction,indent=2)+'\n')
                            # The score kernel is called only after the mask and probability digests are sealed.
                            scores = score_task(root,r,dict(positive_component_ids=frozen['scoring_only_positive_component_ids']),prediction,plan,budget)
                            comp = scores['components']; path = folder/variant/'component_hits.npz'
                            np.savez_compressed(path,ids=np.array(comp.pop('ids'),np.uint32),areas=np.array(comp.pop('areas'),np.int64),hits=np.array(comp.pop('hits'),np.int64));comp.update(path=str(path),sha256=sha(path))
                            rows.append(dict(item=name,wsi_id=sid,method=method,variant=variant,target=plan['target'],prompt_group=plan['prompt_group'],budget=budget,evaluation_scope=signature['evaluation_scope'],positive_native_xy=frozen['positive_native_xy'],negative_native_xy=frozen['negative_native_xy'],prediction=str(pred_file),prediction_sha256=sha(pred_file),scores=scores))
                            outputs.append(dict(prediction=str(pred_file),sha256=sha(pred_file),mask_sha256=prediction['mask_sha256']))
                        dump('rows.json',rows); dump('status.json',dict(status='full_valid_cached_prediction_and_GT_scoring',worker_pid=pid,method=method,completed_rows=len(rows),total_rows=expected_rows,item=name,items_complete=item_no,total_items=len(work),online_encoder_calls=calls[0],seconds=time.perf_counter()-started))
                        del writers
                    del source_device
                del query_cpu; gc.collect()
                print(json.dumps(dict(item=name,rows=len(rows),online_encoder_calls=calls[0])),flush=True)
        for h in handles:
            h.remove()
        require(calls[0] == 0 and len(rows) == expected_rows, 'all fixed tasks and zero online encoding')
        dump('outputs.json',outputs)
        result = dict(status='passed_full_valid_prediction_driver_gate' if a.gate else 'complete_full_valid_predictions_scores_unaccepted',worker_pid=pid,signature=signature,rows=len(rows),items=len(work),online_query_encode_calls=calls[0],rows_sha256=sha(a.output/'rows.json'),outputs_sha256=sha(a.output/'outputs.json'),tile_digests_sha256=sha(a.output/'tile_probability_digests.jsonl'),seconds=time.perf_counter()-started,remaining='Independent native probability/gate and saved-mask GT/component/distance acceptance, every remote digest/mask crop invariance, full-valid BF1/statistics and fullcell input/decode path.')
        dump('summary.json',result); dump('status.json',dict(status=result['status'],worker_pid=pid)); print(json.dumps(result),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ('root','index','output'):
        p.add_argument('--'+n,type=Path,required=True)
    for n in ('primary','primary-acceptance','verified-gate','gate-acceptance'):
        p.add_argument('--'+n,type=Path)
    p.add_argument('--gate',action='store_true'); p.add_argument('--cpu-fixture',action='store_true')
    run(p.parse_args())
