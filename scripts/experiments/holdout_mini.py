"""Adds the mini model (EQUS.MINI to 15:58, current universe) to the holdout check: waits for
holdout.py, runs mini's walk-forward with the holdout, then redoes the comparison with mini included.

    python scripts/experiments/holdout_mini.py WORKDIR SECTORS_CSV
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("holdout", ROOT / "scripts/experiments/holdout.py")
holdout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(holdout)
log, stage, edit = holdout.log, holdout.stage, holdout.l50.edit


def mini_config() -> None:
    def change(c):
        c["version"]["evaluation"] = "mini_v1_rerun20261008"  # keeps the earlier mini results
        c["walk_forward"]["variants"] = [v for v in c["walk_forward"]["variants"] if v["name"] == holdout.s2.BASE_VARIANT]
        c["live"]["model_path"] = "models/experiments/mini_v1_rerun20261008.joblib"
        c["model"]["path"] = "models/experiments/mini_v1_rerun20261008_single.joblib"
    edit("config/train/train_model_mini.yaml", "train_model_mini_from2025-06.yaml", change)


def wait_for_holdout() -> None:
    while "holdout compare" not in holdout.DONE.read_text():
        time.sleep(60)


if __name__ == "__main__":
    holdout.MODELS["mini (current universe)"] = ("train_model_mini_from2025-06.yaml", holdout.s2.BASE_VARIANT)
    try:
        stage("mini config", mini_config)
        stage("mini wait for the 10 models", wait_for_holdout)
        stage("holdout walk-forward mini (current universe)", lambda: holdout.walk_forward("mini (current universe)"))
        stage("holdout compare with mini", holdout.compare)
        log("ALL DONE mini")
    except Exception as error:
        import traceback
        traceback.print_exc()
        log(f"FAILED mini: {error}")
        try:
            from src.notify import telegram
            telegram.send(f"⚠️ Holdout check (mini) stopped: {str(error)[:300]}")
        except Exception:
            pass
        raise
