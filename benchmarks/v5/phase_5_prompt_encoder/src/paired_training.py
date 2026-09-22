"""Controls shared by local and cross-patch training arms."""
import hashlib
import numpy as np


def support_counts(item):
    return tuple(int(item[sign + '_mask'].sum()) for sign in ('positive', 'negative'))


def shuffled_order(count, length, seed=20260922):
    if count < 1 or length < 1:
        raise ValueError('positive count and length required')
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(count) for _ in range((length+count-1)//count)])[:length].tolist()


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
