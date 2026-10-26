"""Accept generic HE local training/score bindings; preparation audit is separate."""
import argparse
import json
from pathlib import Path
import os
import traceback

from .audit_heldout_local import read, sha, require, close, checker, means


def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    save('status.json', dict(status='running', worker_pid=os.getpid()))
    results = {}
    try:
        completed = read(root/'summary.json')
        require(completed['status'] == 'complete_execution' and root.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0'), 'upstream execution')
        for method in ('generic_he_pca64', 'generic_he_white64'):
            train = Path(completed['complete']['train|'+method])
            scored = Path(completed['complete']['evaluate|'+method])
            gate = Path(completed['complete']['gate|'+method])
            tm = read(train/'run.json'); tr = read(train/'summary.json'); tc = tm['contract']
            em = read(scored/'run.json'); er = read(scored/'summary.json'); best = read(train/'best_checkpoint.json')
            tasks = Path(tm['args']['tasks']); realized = Path(tm['args']['realized'])
            _, expected, validate = checker(tasks)
            require(tr['status'] == er['status'] == 'complete' and tr['gate'] is False and er['gate'] is False, 'formal full scope')
            require(tm['args']['resume'] is None and not tm['args']['gate'], 'fresh training')
            require(tc['steps'] == tr['step'] == 10000 and tc['accumulate'] == 4 and tc['warmup'] == 1000, 'training budget')
            actual = realized/'training_plans.jsonl'
            require(sha(str(actual)) == sha(str(train/'training_consumed.jsonl')) == tr['training_consumed_sha256'] == tc['training_plans_sha256'] == 'fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe', 'same original actual40k tasks')
            require(sum(1 for _ in actual.open()) == 40000, 'actual task count')
            require(tc['protocol_sha256'] == em['protocol_sha256'] == sha(str(tasks/'contract.json')), 'protocol binding')
            require(tr['full_manifest_parity'] and tr['checkpoint_reload_exact'] and tr['nonzero_joint_backbone_gradient'], 'training checks')
            prior = tc['initialization']
            require(not prior['J5_loaded'] and not prior['XCell64_used'] and prior['whitened'] == (method == 'generic_he_white64'), 'generic cell prior')
            for sources in (tc['source_sha256'], prior['generic_sources'], em['source_sha256']):
                for p, value in sources.items():
                    require(sha(p) == value, 'frozen source changed: '+p)
            gr = read(gate/'summary.json')
            require(gr['gate'] and gr['status'] == 'complete' and gr['fresh_gate_resume'] and gr['checkpoint_reload_exact'] and gr['nonzero_joint_backbone_gradient'], 'native training gate')
            history = tr['history']
            require([h['step'] for h in history] == list(range(1000, 10001, 1000)), 'selection schedule')
            selection = {k for k in expected if k[1] == 0 and k[2] == '3+2'}
            for h in history:
                rows = read(train/f'validation_{h["step"]:05d}_1025/rows.json')
                require(len(rows) == len(selection) == 302 and {validate(r) for r in rows} == selection, 'selection coverage')
                close(means(rows)['3+2']['dice'], h['score'], 'selection recount')
            chosen = max(history, key=lambda h: (h['score'], h['step']))
            require(best['step'] == chosen['step'] and best['score'] == chosen['score'], 'development best selection')
            require(Path(best['path']).resolve() == Path(em['checkpoint']).resolve() and sha(best['path']) == best['checkpoint_sha256'] == em['checkpoint_sha256'], 'selected weight SHA')
            require(sha(str(scored/'rows.jsonl')) == er['rows_sha256'], 'score SHA')
            rows = [json.loads(line) for line in (scored/'rows.jsonl').open()]
            require(len(rows) == len(expected) == 6040 and {validate(r) for r in rows} == set(expected), 'complete scoring and GT recount')
            metrics = means(rows)
            for b, value in metrics.items():
                close(value['dice'], er['metrics'][b]['macro_pair_dice'], 'reported Dice')
                close(value['bf1'], er['metrics'][b]['bf1_um2'], 'reported BF1 average')
            for p in (train, scored, gate):
                require(p.with_suffix('.log').read_text().rstrip().endswith('EXIT_CODE=0'), 'stage exit')
            results[method] = dict(metrics=metrics, selected_step=best['step'], checkpoint_sha256=best['checkpoint_sha256'], rows_sha256=er['rows_sha256'])
            print(json.dumps(dict(method=method, primary=metrics['3+2'])), flush=True)
        save('summary.json', dict(status='accepted_generic_local_training_and_counts_qualified', conditions=results,
            worker_pid=os.getpid(), score_rows=12080, source_sha256=sha(__file__),
            limits=['Full raw cache/PCA/projection independent audit remains separate', 'Single seed; paired statistics pending',
                    'No full prediction-mask replay/BF1 distance recount', 'PCA fits current training images; not equal total pretraining',
                    'Both prespecified projections reported; no posthoc best variant selection']))
        save('status.json', dict(status='accepted', worker_pid=os.getpid()))
    except Exception:
        save('status.json', dict(status='failed', worker_pid=os.getpid(), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.output)
