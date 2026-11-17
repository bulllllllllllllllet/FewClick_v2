"""Frozen four-method extension to 30 WSIs; reuse all accepted first-15 outputs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump

ARMS = ["HG_mix", "proto_ms", "M", "no_cross_scale"]


def read(path):
    return json.loads(Path(path).read_text())


def contract(root):
    c = read(root / "contract.json")
    assert sha(root / "cases.json") == c["cases_sha256"]
    for path, digest in c["sources"].items():
        assert sha(path) == digest, path
    return c


def run_worker(root, method, index, gate_only):
    import torch
    from . import wsi
    from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
    c = contract(root)
    assert method in ARMS and 15 <= index < 30
    case = read(root / "cases.json")[index]
    assert case["eligible"] and not any(case["membership"].values())
    checkpoint = c["checkpoints"][method]
    assert sha(checkpoint["path"]) == checkpoint["sha256"]
    out = (root / "gates" / method if gate_only else root / "wsi" / method / f"case_{index:02}")
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "worker.json", dict(pid=os.getpid(), gpu=os.environ["CUDA_VISIBLE_DEVICES"],
         method=method, case_index=index, checkpoint=checkpoint, gate_only=gate_only))
    wsi.raster.configure()
    model = wsi.load(method, checkpoint["path"])
    state = tree_sha(model.state_dict())
    wsi.accumulation_gate()
    wsi.raster.encode, wsi.raster.decode = wsi.encode, wsi.decode
    with torch.no_grad():
        wsi.raster.engineering(model, method, case, case["supports"][0], out)
        if not gate_only:
            wsi.infer(model, method, case, case["supports"], out / "readouts")
    assert tree_sha(model.state_dict()) == state
    if not gate_only:
        # The reused scorer retains a legacy first-three description. Correct
        # metadata only in this new output; its inference/scoring code is frozen.
        report_path = out / "readouts/report.json"
        report = read(report_path)
        report.update(status="complete_accepted_WSI", cohort="fixed_original_30_WSI_pool",
                      case_index=index, reused_scorer="benchmarks.v5.reviewer_sample.wsi.infer",
                      limitations="Fixed original 30-slide pool; no audited target train/val membership; historical evaluation exposure retained. No training or new seed.")
        dump(report_path, report)
        acc_path = out / "readouts/acceptance.json"
        acc = read(acc_path)
        acc.update(status="complete_accepted_WSI", report_sha256=sha(report_path))
        dump(acc_path, acc)
    dump(out / "acceptance.json", dict(status="accepted_engineering_only" if gate_only else "complete_accepted",
         checkpoint_sha256=checkpoint["sha256"], cases_sha256=c["cases_sha256"],
         unchanged_model_state=True, case_index=index, method=method))


def read_result(spec, method, checkpoint):
    path = Path(spec["path"])
    assert sha(path) == spec["sha256"], path
    report = read(path)
    if spec["kind"] == "original":
        assert report["status"] == "complete_accepted_whole_slide_Dice"
        assert report["full_output_covered"] and report["independent_disk_score_exact"]
        values = report["targets"]
        values = values.values() if isinstance(values, dict) else values
        by = {v["target"]: {"all": v["whole_WSI"], "remote": v["remote_only_WSI"]} for v in values}
    else:
        folder = path.parent.parent
        acc = read(folder / "acceptance.json")
        assert acc["status"] == "complete_accepted" and acc["unchanged_model_state"]
        assert acc["checkpoint_sha256"] == checkpoint["sha256"]
        inner = read(path.parent / "acceptance.json")
        assert sha(path) == inner["report_sha256"]
        assert sha(path.parent / "windows.jsonl") == inner["windows_sha256"]
        assert report["independent_window_and_whole_disk_replay"] and report["full_coverage"]
        by = {int(t): v["fused"] for t, v in report["targets"].items()}
    assert report["method"] == method
    return report, by


def aggregate(rows, slides):
    import numpy as np
    methods = list(rows)
    tissues = sorted({r["target"] for r in rows["HG_mix"]})
    assert len(tissues) == 11
    scores, matrices, slide_scores = {}, {}, {}
    for method in methods:
        scores[method], matrices[method], slide_scores[method] = {}, {}, {}
        for domain in ["all", "remote"]:
            matrix = np.full((slides, len(tissues)), np.nan)
            for row in rows[method]:
                value = row[domain]["dice"]
                if value is not None:
                    matrix[row["case_index"], tissues.index(row["target"])] = value
            per_tissue = np.nanmean(matrix, axis=0)
            scores[method][domain] = dict(macro_Dice=float(per_tissue.mean()),
                per_tissue={str(t): float(v) for t, v in zip(tissues, per_tissue)})
            matrices[method][domain] = matrix
            slide_scores[method][domain] = np.nanmean(matrix, axis=1).tolist()
    rng = np.random.default_rng(20261116)
    sampled = rng.integers(0, slides, size=(10000, slides))
    paired = {}
    for method in methods:
        if method == "HG_mix":
            continue
        paired[method] = {}
        for domain in ["all", "remote"]:
            difference = matrices["HG_mix"][domain] - matrices[method][domain]
            values = difference[sampled]
            count = np.isfinite(values).sum(axis=1)
            keep = (count > 0).all(axis=1)
            gains = (np.nansum(values[keep], axis=1) / count[keep]).mean(axis=1)
            paired[method][domain] = dict(
                macro_gain=scores["HG_mix"][domain]["macro_Dice"] - scores[method][domain]["macro_Dice"],
                CI95=np.percentile(gains, [2.5, 97.5]).tolist(), retained_replicates=int(keep.sum()),
                per_slide_mean_tissue_gain=np.nanmean(difference, axis=1).tolist())
    return dict(scores=scores, per_slide_mean_tissue_Dice=slide_scores, paired_FewClick_minus=paired)


def collect(root, count):
    c = contract(root)
    cases = read(root / "cases.json")[:count]
    rows, evidence = {m: [] for m in ARMS}, {}
    for method in ARMS:
        for index, case in enumerate(cases):
            if index < 15:
                spec = c["reuse_reports"][method][str(index)]
            else:
                path = root / "wsi" / method / f"case_{index:02}" / "readouts/report.json"
                spec = dict(path=str(path), sha256=sha(path), kind="readouts")
            report, by = read_result(spec, method, c["checkpoints"][method])
            assert report["slide"] == case["wsi_id"]
            assert set(by) == {v["target"] for v in case["supports"]}
            evidence[spec["path"]] = spec["sha256"]
            for target, metrics in sorted(by.items()):
                rows[method].append(dict(case_index=index, slide=case["wsi_id"], target=target, **metrics))
    expected = sum(len(case["supports"]) for case in cases)
    assert expected == {15:152, 30:307}[count]
    assert all(len(v) == expected for v in rows.values())
    for method in ARMS[1:]:
        for ref, row in zip(rows["HG_mix"], rows[method]):
            assert (ref["slide"], ref["target"]) == (row["slide"], row["target"])
            for domain in ["all", "remote"]:
                for key in ["valid", "positive"]:
                    assert ref[domain][key] == row[domain][key], (method, row["slide"], key)
    return rows, evidence


def validate_history(root):
    c = contract(root)
    rows, evidence = collect(root, 15)
    result = aggregate(rows, 15)
    old = read(c["original_summary"])
    for key, by_method in result.items():
        for method, value in by_method.items():
            assert value == old[key][method], (key, method)
    dump(root / "historical_replay.json", dict(status="accepted_exact_historical_15_replay",
         methods=ARMS, report_count=len(evidence), target_method_rows=608,
         scores_and_bootstrap_exact=True, source_reports=evidence))


def summarize(root):
    contract(root)
    assert read(root / "historical_replay.json")["scores_and_bootstrap_exact"]
    rows, evidence = collect(root, 30)
    result = aggregate(rows, 30)
    # Report the extension separately as well; do not hide the already observed first cohort.
    extra = {m: [dict(v, case_index=v["case_index"]-15) for v in rr if v["case_index"] >= 15]
             for m, rr in rows.items()}
    dump(root / "summary.json", dict(status="complete_accepted_full30", slides=30, targets=307,
         methods=ARMS, **result, rows=rows, extension15=aggregate(extra, 15),
         source_report_SHA256=evidence,
         bootstrap=dict(slide_clusters=30, replicates=10000, seed=20261116, fixed_predictions=True),
         scope="Four frozen methods on the original 30-slide eligible pool; first15 reused, remaining15 inferred. No new training or seed. Historical evaluation exposure retained."))
    dump(root / "acceptance.json", dict(status="complete_accepted_full30", methods=4, slides=30,
         model_slide_pairs=120, newly_computed_pairs=60, reused_pairs=60,
         target_method_evaluations=1228, report_sha256=sha(root / "summary.json")))


def queue(root):
    c = contract(root)
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root / "jobs.json").exists(), "Preserve an existing queue; no automatic restart"
    jobs = [dict(method=m, case=15, gate_only=True, status="pending") for m in ARMS]
    tasks = [dict(method=m, case=i, gate_only=False, status="pending",
                  estimated_seconds=c["seconds_per_window"][m] * c["windows"][str(i)])
             for m in ARMS for i in range(15, 30)]
    jobs.extend(sorted(tasks, key=lambda j: -j["estimated_seconds"]))
    running, blocked = {}, set()
    (root / "logs").mkdir(exist_ok=False)

    def save():
        dump(root / "jobs.json", dict(controller_pid=os.getpid(), updated_at=time.time(), jobs=jobs))

    save()
    while any(j["status"] in ["pending", "running"] for j in jobs):
        for gpu, (proc, log, job) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.write(f"\nEXIT_CODE={code}\n")
            log.close()
            folder = (root / "gates" / job["method"] if job["gate_only"] else
                      root / "wsi" / job["method"] / f"case_{job['case']:02}")
            acc = folder / "acceptance.json"
            expected = "accepted_engineering_only" if job["gate_only"] else "complete_accepted"
            passed = code == 0 and acc.exists() and read(acc)["status"] == expected
            job.update(status="complete" if passed else "failed", exit_code=code, finished_at=time.time())
            if not passed:
                blocked.add(job["method"])
            del running[gpu]
        for job in jobs:
            if job["status"] == "pending" and job["method"] in blocked:
                job["status"] = "blocked_by_failed_arm"
        gates_ready = all(j["status"] == "complete" for j in jobs if j["gate_only"])
        if any(j["gate_only"] and j["status"] in ["failed", "blocked_by_failed_arm"] for j in jobs):
            for job in jobs:
                if job["status"] == "pending" and not job["gate_only"]:
                    job["status"] = "blocked_by_engineering_failure"
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                          "--format=csv,noheader,nounits"], text=True)
        devices = [tuple(int(v.strip()) for v in line.split(",")) for line in output.splitlines()]
        for gpu, free, _ in sorted(devices, key=lambda d: (-d[1], d[2])):
            if gpu not in c["gpus"] or gpu in running or free < 12288:
                continue
            if shutil.disk_usage(root).free < 120 * 1024**3:
                break
            candidates = [j for j in jobs if j["status"] == "pending" and (j["gate_only"] or gates_ready)]
            if not candidates:
                break
            contract(root)
            job = candidates[0]
            label = "gate" if job["gate_only"] else f"case_{job['case']:02}"
            log_path = root / "logs" / f"{job['method']}_{label}.log"
            log = log_path.open("x")
            cmd = [sys.executable, "-u", "-m", "benchmarks.v5.reviewer_sample.full30", "worker",
                   "--root", str(root), "--method", job["method"], "--case-index", str(job["case"])]
            if job["gate_only"]:
                cmd.append("--gate-only")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
            job.update(status="running", gpu=gpu, worker_pid=proc.pid, command=cmd,
                       log=str(log_path), started_at=time.time())
            running[gpu] = proc, log, job
            save()
        save()
        if any(j["status"] in ["pending", "running"] for j in jobs):
            time.sleep(15)
    if all(j["status"] == "complete" for j in jobs):
        summarize(root)
        dump(root / "terminal.json", dict(status="complete_accepted_full30", time=time.time()))
    else:
        dump(root / "terminal.json", dict(status="incomplete_failed_or_blocked", time=time.time(), jobs=jobs))
        raise RuntimeError("One or more controls failed; no automatic retries")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["worker", "queue", "summarize", "validate-history"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--method", choices=ARMS)
    p.add_argument("--case-index", type=int)
    p.add_argument("--gate-only", action="store_true")
    a = p.parse_args()
    if a.mode == "worker":
        run_worker(a.root, a.method, a.case_index, a.gate_only)
    elif a.mode == "queue":
        queue(a.root)
    elif a.mode == "validate-history":
        validate_history(a.root)
    else:
        summarize(a.root)


if __name__ == "__main__":
    main()
