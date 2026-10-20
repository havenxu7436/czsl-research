"""Recompute archived metrics, probability-factor diagnostics and figures."""
from pathlib import Path
import runpy

if __name__=='__main__':
    for name in ('recompute_metrics.py','recompute_mechanism.py','recompute_figures.py'):
        runpy.run_path(str(Path(__file__).resolve().parent/name),run_name='__main__')
