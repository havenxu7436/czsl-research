"""Recompute the G archive metrics, static-gate equivalence and figures."""
from pathlib import Path
import runpy
import shutil

if __name__=='__main__':
    for name in ('recompute_metrics.py','recompute_supplementary.py','recompute_figures.py'):
        runpy.run_path(str(Path(__file__).resolve().parent/name),run_name='__main__')
    stage=Path(__file__).resolve().parents[1]
    for name in ('ranking_diagnostics.csv','ranking_diagnostics_mean_sd.csv',
                 'ranking_diagnostic_deltas.csv','ranking_diagnostic_paired_mean_sd.csv'):
        shutil.copy2(stage/'results/raw_export/roundG_20261020'/name,
                     stage/'analysis/recomputed/tables'/name)
