"""Select canonical Phase 2B black-box checkpoints without using test performance.

Rule:
For each model × prefix, compute the mean validation Macro-F1 across the five formal
seeds and choose the seed whose validation Macro-F1 is closest to that mean.
Ties are broken by the smaller seed.

Run from project root:
    python scripts/10_select_canonical_blackboxes.py
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT / "src"))

from etce.phase2.data import PREFIXES
from etce.utils.io import read_json, sha256, write_json

RESULTS2B = ROOT / "results/phase2b"
RESULTS3 = ROOT / "results/phase3"
MODELS = ("RF", "XGBoost", "MLP", "LSTM")
MODEL_FILES = {"RF": "rf", "XGBoost": "xgboost", "MLP": "mlp", "LSTM": "lstm"}
FORMAL_SEEDS = (13, 21, 42, 73, 101)


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    metrics_path = RESULTS2B / "metrics/formal_metrics_per_seed.csv"
    validation_path = RESULTS2B / "logs/validation_phase2b.json"
    phase2a_configs = ROOT / "results/phase2/configs/best_configs.json"

    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    validation = read_json(validation_path)
    if validation.get("status") != "passed":
        raise RuntimeError("Phase 2B must be validated before canonical selection")

    phase2a = read_json(phase2a_configs)
    metrics = rows(metrics_path)

    out_dir = RESULTS3 / "canonical_blackboxes"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "selection_rule": (
            "For each model/prefix, select the formal seed whose validation Macro-F1 "
            "is closest to the five-seed validation mean; tie -> smaller seed. "
            "Test metrics are not read for selection."
        ),
        "formal_seeds": list(FORMAL_SEEDS),
        "source_phase2b_validation_status": validation["status"],
        "source_phase2b_metrics_sha256": sha256(metrics_path),
        "source_phase2a_configs_sha256": sha256(phase2a_configs),
        "items": {},
    }

    for model in MODELS:
        manifest["items"][model] = {}
        stem = MODEL_FILES[model]
        for prefix in PREFIXES:
            group = [
                r for r in metrics
                if r["model"] == model
                and int(r["prefix"]) == int(prefix)
                and r["split"] == "val"
            ]
            if sorted(int(r["seed"]) for r in group) != list(FORMAL_SEEDS):
                raise AssertionError(f"Formal validation seed set mismatch: {model}/P{prefix}")

            mean = sum(float(r["macro_f1"]) for r in group) / len(group)
            ranked = sorted(
                group,
                key=lambda r: (abs(float(r["macro_f1"]) - mean), int(r["seed"])),
            )
            selected = ranked[0]
            seed = int(selected["seed"])

            suffix = "joblib" if model in ("RF", "XGBoost") else "pt"
            source = RESULTS2B / f"checkpoints/seed_{seed}/{stem}_prefix{prefix}.{suffix}"
            if not source.is_file():
                raise FileNotFoundError(source)
            target = ckpt_dir / f"{stem}_prefix{prefix}_seed{seed}.{suffix}"
            shutil.copy2(source, target)

            hyper = phase2a["models"][model][str(prefix)]["hyperparameters"]
            manifest["items"][model][str(prefix)] = {
                "selected_seed": seed,
                "validation_macro_f1": float(selected["macro_f1"]),
                "validation_mean_macro_f1": float(mean),
                "absolute_deviation_from_mean": abs(float(selected["macro_f1"]) - mean),
                "phase2a_config_id": phase2a["models"][model][str(prefix)]["config_id"],
                "hyperparameters": hyper,
                "source_checkpoint": source.relative_to(ROOT).as_posix(),
                "canonical_checkpoint": target.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": sha256(target),
            }

    write_json(out_dir / "canonical_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
