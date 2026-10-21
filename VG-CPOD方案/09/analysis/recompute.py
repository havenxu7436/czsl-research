"""Recompute H1 and historical comparisons from archived results without training."""
from pathlib import Path
import runpy

if __name__=='__main__':
    output=Path(__file__).resolve().parent/'recomputed'
    output.mkdir(exist_ok=True)
    for name in ('recompute_metrics.py','recompute_figures.py'):
        runpy.run_path(str(Path(__file__).resolve().parent/name),run_name='__main__')
