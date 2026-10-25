"""Unchanged audited local checkpoint scorer with genericHE builders."""
from benchmarks.v5.fair_comparison import evaluate_checkpoint
from .models import ARMS,build

if __name__=='__main__':
    evaluate_checkpoint.METHODS=ARMS;evaluate_checkpoint.build=build;evaluate_checkpoint.main()
