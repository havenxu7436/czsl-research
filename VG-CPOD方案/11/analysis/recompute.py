"""Recompute archived J1 metrics, optimizer measurements and figures without training."""
from pathlib import Path
import runpy

if __name__=='__main__':
    folder=Path(__file__).resolve().parent
    (folder/'recomputed').mkdir(exist_ok=True)
    for name in ('recompute_metrics.py','recompute_supplement.py','recompute_figures.py'):
        runpy.run_path(str(folder/name),run_name='__main__')
