"""Recompute archived K1/K2 predictions, updates, error decomposition and figures."""
from pathlib import Path
import runpy
import sys

if __name__=='__main__':
    folder=Path(__file__).resolve().parent
    sys.path.insert(0,str(folder))
    (folder/'recomputed').mkdir(exist_ok=True)
    for name in ('recompute_metrics.py','recompute_supplement.py','recompute_figures.py'):
        runpy.run_path(str(folder/name),run_name='__main__')
