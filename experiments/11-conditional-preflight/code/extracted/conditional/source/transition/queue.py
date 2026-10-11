"""Sequential single-GPU plans with explicit handling of completed/partial runs."""
import subprocess
import sys
from pathlib import Path

import yaml


def run_plan(args):
    for filename in args.configs:
        path = Path(filename).resolve()
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        command = "train-branch" if "branch_mode" in cfg["MODEL"] else "run-semantic"
        run = Path(cfg["TRAIN"]["kcc_run_dir"])
        argv = [sys.executable, "-u", "-m", "transition.cli", command, "--cfg", str(path)]
        if (run / "train_complete.json").is_file():
            # Validates config, source, checkpoint and completed evaluations first.
            argv.append("--finish-only")
            print("VERIFY / FINISH", run.name, flush=True)
        elif run.exists() and any(run.iterdir()):
            raise FileExistsError("Partial training exists. Preserve it and use a new run version: " + str(run))
        else:
            print("START", run.name, flush=True)
        subprocess.run(argv, check=True)
    print("RUN PLAN COMPLETE", flush=True)
