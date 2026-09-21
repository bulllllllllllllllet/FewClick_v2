"""Extensible binary tissue definitions independent of model output channels."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TissueTask:
    task_id: str
    positive_ids: tuple[int, ...]
    negative_ids: tuple[int, ...]

    def __post_init__(self):
        if not self.task_id or not self.positive_ids or not self.negative_ids:
            raise ValueError('task ID and explicit positive/negative label sets required')
        if set(self.positive_ids) & set(self.negative_ids):
            raise ValueError('positive/negative labels overlap')
        if 255 in (*self.positive_ids, *self.negative_ids):
            raise ValueError('ignore label cannot define supervision')

    def masks(self, labels):
        labels = np.asarray(labels)
        positive = np.isin(labels, self.positive_ids)
        valid = positive | np.isin(labels, self.negative_ids)
        return positive, valid


def tasks_from_class_map(class_map):
    """One-vs-known-rest tasks; unknown labels remain ignored."""
    ids = tuple(int(c['id']) for c in class_map)
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate class ID')
    return [TissueTask('tissue.'+c['name'], (int(c['id']),),
                       tuple(i for i in ids if i != int(c['id']))) for c in class_map]
