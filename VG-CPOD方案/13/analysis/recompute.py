from pathlib import Path
import runpy
if __name__ == '__main__':
    folder=Path(__file__).resolve().parent
    for name in ['recompute_metrics.py','recompute_candidates.py','check_next_operator.py']:
        runpy.run_path(str(folder/name),run_name='__main__')
