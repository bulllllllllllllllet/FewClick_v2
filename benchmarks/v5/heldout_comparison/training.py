"""Use the frozen shared trainer with an explicit heldout supervision contract."""
import json
from pathlib import Path

from .audit import FOLDS
from .supervision import FoldData
from benchmarks.v5.fair_comparison import train_adapted
from benchmarks.v5.fair_comparison.audit import digest

METHODS = ('fewclick_m3d_image', 'sam_ft', 'wsi_sam_ft')
BASE_FINGERPRINT = train_adapted.fingerprint


class TrainingData(FoldData):
    def __init__(self, tasks, realized=None):
        contract = json.loads((Path(tasks)/'contract.json').read_text())
        super().__init__(tasks, contract['heldout_fold'], realized)
        if contract.get('heldout_excluded_labels') != FOLDS[self.fold]['exclude']:
            raise ValueError('heldout label definition mismatch')
        for plan in self.validation + (self.training or []):
            if plan['target'] in FOLDS[self.fold]['exclude'][plan['dataset']]:
                raise ValueError('excluded target used for training or selection')


def fingerprint(model, data, method, steps, warmup, accumulate, seed):
    if method not in METHODS:raise ValueError('heldout method not declared')
    base = BASE_FINGERPRINT(model, data, method, steps, warmup, accumulate, seed)
    return base | dict(heldout_fold=data.fold, heldout_excluded_labels=FOLDS[data.fold]['exclude'],
        checkpoint_selection='seen masked-label development only; no heldout target scores',
        heldout_source_sha256={str(p):digest(p) for p in (Path(__file__), Path(__file__).with_name('supervision.py'), Path(__file__).with_name('audit.py'))})


if __name__ == '__main__':
    train_adapted.METHODS = METHODS
    train_adapted.LocalData = TrainingData
    train_adapted.fingerprint = fingerprint
    train_adapted.main()
