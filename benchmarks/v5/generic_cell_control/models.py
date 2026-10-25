"""The same fresh three-view Dense architecture with generic H&E provenance."""
from pathlib import Path

from .assemble import ARMS
from benchmarks.v5.fair_comparison.adapted_models import FreshFewClick
from benchmarks.v5.fair_comparison.audit import digest


def build(arm,device):
    if arm not in ARMS:raise ValueError('unknown fixed genericHE arm')
    model=FreshFewClick(device,'xcell')
    model.prior.update(cell_mode=arm,appearance='frozen public CTransPath768 -> train-only PCA64',
        XCell_overlap='XCell64 not used in this control; reference offline parity only',
        XCell64_used=False,whitened=arm=='generic_he_white64',
        generic_sources={str(p):digest(p) for p in [Path(__file__),Path(__file__).with_name('assemble.py'),
            Path(__file__).with_name('train.py'),Path(__file__).with_name('evaluate.py')]})
    return model
