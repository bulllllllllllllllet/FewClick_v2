"""Unchanged audited trainer with explicitly named genericHE builders."""
from benchmarks.v5.fair_comparison import train_adapted
from .models import ARMS,build

if __name__=='__main__':
    train_adapted.METHODS=ARMS;train_adapted.build=build;train_adapted.main()
