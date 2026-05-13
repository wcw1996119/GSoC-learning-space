"""Sequential orchestrator for Paper A experiments.

Order:
    1. synthetic_recovery     (W4 hard checkpoint — stops the run if it fails)
    2. gnn_ablation           (W8 GO/NO-GO — also stops on failure)
    3. choice_set_ablation
    4. temporal_holdout

All scripts log to evaluation_outputs/paper_a/run_log_<timestamp>.json.

CLI
---
  python run_all_paper_a_experiments.py --n_seeds 5 --device cpu
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = V2_ROOT / "experiments" / "paper_a"
OUT_DIR = V2_ROOT / "evaluation_outputs" / "paper_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)


SCRIPTS = [
    ("synthetic_recovery",   "synthetic_recovery.py",    True),   # hard gate
    ("gnn_ablation",         "gnn_ablation.py",          True),   # W8 gate
    ("choice_set_ablation",  "choice_set_ablation.py",   False),
    ("temporal_holdout",     "temporal_holdout.py",      False),
]


def _child_env() -> dict:
    """Build a child-process environment with PYTHONPATH including V2_ROOT
    so that ``import models_lib...`` resolves in the child interpreter.
    """
    return {
        **os.environ,
        "PYTHONPATH": str(V2_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }


def run_script(script_path: Path, n_seeds: int, device: str, resume: bool) -> dict:
    cmd = [sys.executable, str(script_path), "--n_seeds", str(n_seeds), "--device", device]
    if resume:
        cmd.append("--resume")
    print(f"\n>>> {' '.join(cmd)}")
    started = dt.datetime.utcnow().isoformat()
    proc = subprocess.run(cmd, cwd=str(V2_ROOT), env=_child_env())
    finished = dt.datetime.utcnow().isoformat()
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "started_utc": started,
        "finished_utc": finished,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", action="store_true",
                        help="pass --resume to each child script")
    args = parser.parse_args()

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_path = OUT_DIR / f"run_log_{timestamp}.json"
    log: dict = {
        "n_seeds": args.n_seeds,
        "device": args.device,
        "started_utc": dt.datetime.utcnow().isoformat(),
        "steps": [],
    }

    for name, fname, is_gate in SCRIPTS:
        script_path = EXP_DIR / fname
        if not script_path.exists():
            print(f"[run_all] missing {script_path}; skipping")
            log["steps"].append({"name": name, "skipped": True})
            continue
        result = run_script(script_path, args.n_seeds, args.device, args.resume)
        result["name"] = name
        result["is_gate"] = is_gate
        log["steps"].append(result)
        log_path.write_text(json.dumps(log, indent=2))
        if is_gate and result["returncode"] != 0:
            log["aborted"] = True
            log["abort_reason"] = (
                f"{name} failed with returncode={result['returncode']}; "
                f"this is a hard checkpoint — downstream scripts will not run"
            )
            log_path.write_text(json.dumps(log, indent=2))
            print(f"\n[run_all] HARD CHECKPOINT FAILURE in {name}; aborting")
            sys.exit(result["returncode"])

    log["finished_utc"] = dt.datetime.utcnow().isoformat()
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\n[run_all] complete; log at {log_path}")


if __name__ == "__main__":
    main()
