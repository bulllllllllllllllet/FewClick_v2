"""Deterministic spatial isolation in explicit level-0 pixels, not microns."""
import hashlib
import math


def assign_context(source, boxes, block_size=16384, margin=256, seed=20260922):
    """Keep the union of all contexts strictly inside a buffered spatial block.

    Patient identity and labels are deliberately absent. Equal physical source
    paths must use the same canonical source key. Different rescans need an
    external registration audit; this function cannot establish equivalence.
    """
    if block_size <= 2 * margin or margin < 0:
        raise ValueError('invalid block size or margin')
    if not boxes or any(len(b) != 4 or not all(math.isfinite(v) for v in b)
                        or b[2] <= 0 or b[3] <= 0 for b in boxes):
        raise ValueError('finite positive xywh boxes required')
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    bx = math.floor((x0 + x1) / (2 * block_size))
    by = math.floor((y0 + y1) / (2 * block_size))
    if (x0 < bx * block_size + margin or y0 < by * block_size + margin
            or x1 > (bx + 1) * block_size - margin
            or y1 > (by + 1) * block_size - margin):
        return None
    group = f'{source}|{bx}|{by}'
    bucket = int(hashlib.sha256(f'{seed}|{group}'.encode()).hexdigest()[:16], 16) % 100
    return dict(spatial_group_id=group, split='train' if bucket < 70 else 'val' if bucket < 85 else 'test',
                context_x0=x0, context_y0=y0, context_x1=x1, context_y1=y1)
