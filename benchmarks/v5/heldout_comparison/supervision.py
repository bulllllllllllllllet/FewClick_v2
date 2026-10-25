"""Exclude heldout labels from all current local supervision, including selection."""
from functools import lru_cache
import numpy as np

from .audit import FOLDS
from benchmarks.v5.fair_comparison.local_data import LocalData


class FoldData(LocalData):
    def __init__(self,tasks,fold,realized=None,gate=False):
        if fold not in FOLDS:raise ValueError('unknown heldout fold')
        super().__init__(tasks,realized);self.fold=fold
        if not gate:
            if self.contract.get('heldout_fold')!=fold or self.contract.get('heldout_excluded_labels')!=FOLDS[fold]['exclude']:
                raise ValueError('versioned heldout task contract required for training/seen selection')

    @lru_cache(maxsize=32)
    def raw(self,identity,contexts=False,cells=False):
        raw,labels=super().raw(identity,contexts,cells)
        excluded=FOLDS[self.fold]['exclude'][self.rows[identity]['dataset']]
        masked=labels.copy();masked[np.isin(masked,excluded)]=255
        return raw,masked
