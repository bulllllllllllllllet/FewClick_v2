"""Explicit support/query separation with patch-local coordinate frames."""


def separated_contexts(a, b, gap=512):
    """Axis-separated xyxy boxes, in source level-0 pixels."""
    if gap < 0 or any(len(x) != 4 or x[2] <= x[0] or x[3] <= x[1] for x in (a, b)):
        raise ValueError('invalid boxes or gap')
    return a[2]+gap <= b[0] or b[2]+gap <= a[0] or a[3]+gap <= b[1] or b[3]+gap <= a[1]


def transfer_support(query, support):
    """Transfer ONLY annotated support inputs; never query labels or masks.

    Phase5 consumes support-local and query-local geometry separately. We retain
    these frames; this is a transfer diagnostic, not global-coordinate inference.
    """
    result = dict(query)
    for sign in ('positive', 'negative'):
        for suffix in ('tokens', 'xy', 'mask'):
            result[f'{sign}_{suffix}'] = support[f'{sign}_{suffix}']
    result['prompt_size_id'] = support['prompt_size_id']
    return result
