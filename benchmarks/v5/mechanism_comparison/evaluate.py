"""Use the frozen local checkpoint scorer with explicit mechanism architectures."""
from benchmarks.v5.fair_comparison import evaluate_checkpoint
from .models import VARIANTS,build


if __name__=='__main__':
    evaluate_checkpoint.METHODS=tuple(VARIANTS)
    evaluate_checkpoint.build=build
    evaluate_checkpoint.main()
