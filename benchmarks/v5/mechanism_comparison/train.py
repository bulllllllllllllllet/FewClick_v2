"""Reuse the frozen audited trainer with explicit versioned mechanism builders."""
from benchmarks.v5.fair_comparison import train_adapted
from .models import VARIANTS,build


if __name__=='__main__':
    train_adapted.METHODS=tuple(VARIANTS)
    train_adapted.build=build
    train_adapted.main()
