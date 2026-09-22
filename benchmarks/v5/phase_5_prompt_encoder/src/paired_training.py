"""Controls shared by local and cross-patch training arms."""
import hashlib
import numpy as np
import torch


def support_counts(item):
    return tuple(int(item[sign + '_mask'].sum()) for sign in ('positive', 'negative'))


def shuffled_order(count, length, seed=20260922):
    if count < 1 or length < 1:
        raise ValueError('positive count and length required')
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(count) for _ in range((length+count-1)//count)])[:length].tolist()


def mixed_cross_mask(length, fraction, seed=20260923):
    if length < 1 or not 0 < fraction < 1:
        raise ValueError('positive length and fraction strictly between zero and one required')
    result = np.zeros(length, dtype=bool)
    rng = np.random.default_rng(seed)
    result[rng.permutation(length)[:round(length*fraction)]] = True
    return result


def rejection_task_masks(length, absent_fraction, cross_present_fraction, seed=20260925):
    """Return disjoint absent and remote-present episode masks for a fixed budget.

    The remainder is a local, target-present episode.  Sampling one joint
    permutation prevents an accidental overlap between the two remote modes.
    """
    if (length < 1 or not 0 < absent_fraction < 1 or not 0 <= cross_present_fraction < 1
            or absent_fraction + cross_present_fraction >= 1):
        raise ValueError('invalid rejection task fractions')
    order = np.random.default_rng(seed).permutation(length)
    # A smoke run must exercise every task type; full runs retain exact rounded
    # fractions (e.g. 1200/1800 of 12000 episodes).
    absent_count = max(1, round(length * absent_fraction))
    cross_count = max(1, round(length * cross_present_fraction)) if cross_present_fraction else 0
    if absent_count + cross_count >= length:
        raise ValueError('episode budget too small for local, cross-present and absent tasks')
    absent = np.zeros(length, dtype=bool)
    cross = np.zeros(length, dtype=bool)
    absent[order[:absent_count]] = True
    cross[order[absent_count:absent_count + cross_count]] = True
    return absent, cross


def absent_query_target(item):
    """Keep geometry/validity, but supervise every valid query region as negative."""
    result = dict(item)
    target = result['binary_target'].clone()
    result['binary_target'] = torch.where(target == 255, target, torch.zeros_like(target))
    # The query's original prompt exclusion must not suppress negative evidence.
    result['all_prompted_regions'] = torch.zeros_like(result['all_prompted_regions'], dtype=torch.bool)
    return result


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
