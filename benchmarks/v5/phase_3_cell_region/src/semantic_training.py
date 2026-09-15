"""Stage A/B training helpers and patient-first classification metrics."""
from __future__ import annotations

import torch

from .semantic_losses import partial_identity_loss, distribution_loss


def training_loss(model, output, targets, cell_valid, identity_class_weights=None):
    if model.stage == "identity":
        return partial_identity_loss(output["identity_logits"], targets["identity_allowed"],
                                     targets["identity_valid"] & cell_valid, identity_class_weights)
    tissue, nt = distribution_loss(output["tissue_logits"], targets["tissue_target"], targets["tissue_valid"])
    composition, nc = distribution_loss(output["composition_logits"], targets["composition_target"],
                                         targets["composition_valid"] & output["evidence_valid"])
    return tissue + 0.2 * composition, nt + nc


def summarize_confusion(matrix):
    matrix = matrix.double()
    support = matrix.sum(1)
    precision = matrix.diag() / matrix.sum(0).clamp_min(1)
    recall = matrix.diag() / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    supported = support > 0
    return {"macro_f1": float(f1[supported].mean()) if supported.any() else None,
            "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(),
            "support": support.tolist(), "confusion": matrix.tolist()}


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    cases = {}
    classes = len(dataset.spec["identity_classes"] if model.stage == "identity" else dataset.spec["tissue_classes"])
    for item in dataset:
        inputs = {k: v.to(device) for k, v in item["inputs"].items()}
        targets = {k: v.to(device) for k, v in item["targets"].items()}
        result = model(**inputs)
        record = cases.setdefault(item["case_id"], {"matrix": torch.zeros(classes, classes, dtype=torch.long),
                        "brier_sum": 0., "count": 0, "bins": torch.zeros(15, 3, dtype=torch.float64),
                        "composition_error_sum": 0., "composition_count": 0})
        if model.stage == "identity":
            keep = targets["identity_valid"] & inputs["cell_valid"] & (targets["identity_allowed"].sum(-1) == 1)
            truth = targets["identity_allowed"].long().argmax(-1)[keep]
            prob = result["identity_probabilities"][keep]
        else:
            keep = targets["tissue_valid"]
            truth = targets["tissue_target"].argmax(-1)[keep]
            prob = result["tissue_logits"].softmax(-1)[keep]
        pred = prob.argmax(-1)
        record["matrix"] += torch.bincount((truth * classes + pred).cpu(), minlength=classes**2).reshape(classes, classes)
        if len(truth):
            record["brier_sum"] += float(((prob - torch.nn.functional.one_hot(truth, classes))**2).sum(-1).sum())
            record["count"] += len(truth)
            conf = prob.max(-1).values
            bins = (conf * 15).long().clamp_max(14)
            for idx in range(15):
                selected = bins == idx
                record["bins"][idx] += torch.tensor([int(selected.sum()), float(conf[selected].sum()), float((pred[selected] == truth[selected]).sum())])
        valid_comp = targets["composition_valid"] & result["evidence_valid"]
        if valid_comp.any():
            error = (result["composition"][valid_comp] - targets["composition_target"][valid_comp]).abs().mean(-1)
            record["composition_error_sum"] += float(error.sum())
            record["composition_count"] += len(error)
    summaries = {}
    for case, record in cases.items():
        count = record["count"]
        bins = record["bins"]
        value = summarize_confusion(record["matrix"])
        value.update(brier=record["brier_sum"] / count if count else None,
                     ece=float((bins[:, 1] - bins[:, 2]).abs().sum() / count) if count else None,
                     composition_mae=record["composition_error_sum"] / record["composition_count"] if record["composition_count"] else None)
        summaries[case] = value
    scores = [r["macro_f1"] for r in summaries.values() if r["macro_f1"] is not None]
    return {"cases": summaries, "patient_macro_f1": sum(scores) / len(scores) if scores else None,
            "stage": model.stage, "scope": "classification probe; not final click segmentation"}
