"""White-padded physical views and fixed full-coverage grid; no query labels."""
import json
import math
from functools import lru_cache
import numpy as np
import pyvips
import torch
from PIL import Image
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor


@lru_cache(maxsize=3)
def image(path):
    return pyvips.Image.new_from_file(path, access='random').extract_band(0, n=3)


def rgb_box(path, box):
    x, y, w, h = map(int, box)
    im = image(path)
    left, top, right, bottom = max(0, x), max(0, y), min(im.width, x+w), min(im.height, y+h)
    assert left < right and top < bottom
    crop = im.crop(left, top, right-left, bottom-top)
    padded = crop.embed(left-x, top-y, w, h, extend='background', background=[255, 255, 255])
    rgb = np.frombuffer(padded.write_to_memory(), np.uint8).reshape(h, w, 3).copy()
    return rgb


def boxes_at(x, y, bw, bh):
    cx, cy = x+bw/2, y+bh/2
    return [[round(cx-bw*f/2), round(cy-bh*f/2), bw*f, bh*f] for f in [1, 2, 4]]


def inputs(path, boxes, device='cuda:0'):
    result = {}
    for i, box in enumerate(boxes):
        rgb = np.asarray(Image.fromarray(rgb_box(path, box)).resize((512, 512), Image.Resampling.BILINEAR)).copy()
        suffix = '' if i == 0 else '_'+str(i)
        result['image'+suffix] = he_to_tensor(rgb)[None].to(device)
        result['cells'+suffix] = torch.zeros(1, 0, 74, device=device)
        result['cell_valid'+suffix] = torch.zeros(1, 0, device=device, dtype=torch.bool)
        result['total_cell_count' if i == 0 else 'total_'+str(i)] = torch.zeros(1, device=device)
    return result


def prompts(support, device='cuda:0'):
    result = {'prompt_size_id': torch.ones(1, dtype=torch.long, device=device)}
    for sign in ['positive', 'negative']:
        result[sign+'_xy'] = torch.tensor([support[sign+'_xy']], device=device, dtype=torch.float32)
        result[sign+'_mask'] = torch.ones(1, len(support[sign+'_xy']), device=device, dtype=torch.bool)
    return result


def starts(length, size):
    assert length >= size
    values = list(range(0, length-size+1, max(1, size//2)))
    if values[-1] != length-size:
        values.append(length-size)
    return values


def gt_extent(box, case):
    x, y, w, h = box
    ax, ay = case['gt_affine'][:2]
    return [max(0, math.ceil(x*ax-.5)), max(0, math.ceil(y*ay-.5)),
            min(case['gt_width'], math.ceil((x+w)*ax-.5)),
            min(case['gt_height'], math.ceil((y+h)*ay-.5))]


def grid(case, supports):
    first = supports[0]['geometry']['native_boxes'][0]
    bw, bh = first[2:]
    assert all(s['geometry']['native_boxes'][0][2:] == [bw, bh] for s in supports)
    groups = {y: set(starts(case['width'], bw)) for y in starts(case['height'], bh)}
    for s in supports:
        x, y, _, _ = s['geometry']['native_boxes'][0]
        groups.setdefault(y, set()).add(x)
    return [(y, sorted(xs)) for y, xs in sorted(groups.items())], bw, bh
