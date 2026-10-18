"""Independent validation for ETCE Phase 2B formal multi-seed artifacts.

Expected project placement:
    ETCE/scripts/09_validate_phase2b.py

Run:
    python scripts/09_validate_phase2b.py
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import torch

from etce.phase2.data import COHORT_HASHES, EXPECTED_COUNTS, PREFIXES, load_frozen_common_cohort
from etce.phase2.metrics import binary_metrics
from etce.utils.io import read_json, sha256, write_json

RESULTS = ROOT / "results/phase2b"
PHASE2A_CONFIG = ROOT / "results/phase2/configs/best_configs.json"
SEEDS = (13, 21, 42, 73, 101)
MODELS = ("RF", "XGBoost", "MLP", "LSTM")
MODEL_FILES = {"RF": "rf", "XGBoost": "xgboost", "MLP": "mlp", "LSTM": "lstm"}
METRICS = ("accuracy", "macro_f1", "roc_auc", "balanced_accuracy")
T_CRIT_95_DF4 = 2.7764451051977987


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(left, right, tolerance=1e-10):
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def strip_randomness(model_name, config):
    out = json.loads(json.dumps(config))
    if model_name in ("RF", "XGBoost"):
        out.pop("random_state", None)
    else:
        out.pop("seed", None)
    return out


def validate():
    checks = {}

    required = [
        RESULTS / "PHASE2B_REPORT.md",
        RESULTS / "metrics/formal_metrics_per_seed.csv",
        RESULTS / "metrics/formal_metrics_summary.csv",
        RESULTS / "configs/frozen_phase2a_configs_used.json",
        RESULTS / "logs/test_access_log_phase2b.json",
        RESULTS / "run_manifest_phase2b.json",
    ]
    require(all(p.is_file() for p in required), "Required Phase 2B output missing")
    checks["required_outputs_exist"] = True

    manifest = read_json(RESULTS / "run_manifest_phase2b.json")
    require(manifest["formal_seeds"] == list(SEEDS), "Formal seed list changed")
    require(manifest["prefixes"] == list(PREFIXES), "Prefix list changed")
    require(manifest["models"] == list(MODELS), "Model list changed")
    require(manifest["cohort_hashes"] == COHORT_HASHES, "Cohort hashes changed")
    require(manifest["hyperparameter_search_performed"] is False, "Unexpected Phase 2B search")
    require(manifest["test_used_for_selection"] is False, "Test used for selection")
    require(manifest["phase2a_best_config_sha256"] == sha256(PHASE2A_CONFIG),
            "Frozen Phase 2A config changed")
    checks["manifest_and_frozen_config_match"] = True

    # Re-derive frozen cohort independently.
    cases, audit = load_frozen_common_cohort(ROOT)
    require(audit["raw_sha256"] == manifest["raw_sha256"], "Raw hash mismatch")
    by_split = {
        split: [c for c in cases if c.split == split]
        for split in ("train", "val", "test")
    }
    for split in ("train", "val", "test"):
        require(len(by_split[split]) == EXPECTED_COUNTS[split],
                f"Frozen cohort count mismatch: {split}")
    checks["frozen_cohort_rederived"] = True

    phase2a = read_json(PHASE2A_CONFIG)
    frozen_used = read_json(RESULTS / "configs/frozen_phase2a_configs_used.json")
    require(frozen_used["source_sha256"] == sha256(PHASE2A_CONFIG),
            "Recorded Phase 2A source hash mismatch")

    # Confirm hyperparameters differ only in seed/random_state.
    for model in MODELS:
        for prefix in PREFIXES:
            base = phase2a["models"][model][str(prefix)]["hyperparameters"]
            require(
                strip_randomness(model, base)
                == strip_randomness(model, frozen_used["models"][model][str(prefix)]["hyperparameters"]),
                f"Frozen base config mismatch: {model}/P{prefix}",
            )
    checks["hyperparameters_frozen"] = True

    per_seed = rows(RESULTS / "metrics/formal_metrics_per_seed.csv")
    expected_metric_rows = len(SEEDS) * len(MODELS) * len(PREFIXES) * 2
    require(len(per_seed) == expected_metric_rows,
            f"Expected {expected_metric_rows} per-seed metric rows")
    index = {
        (r["model"], int(r["prefix"]), r["split"], int(r["seed"])): r
        for r in per_seed
    }
    require(len(index) == expected_metric_rows, "Duplicate/missing metric key")

    prediction_files = 0
    constant_test = []
    expected_ids = {
        split: sorted(c.case_id for c in by_split[split])
        for split in ("val", "test")
    }

    for seed in SEEDS:
        pred_dir = RESULTS / "predictions" / f"seed_{seed}"
        require(pred_dir.is_dir(), f"Missing prediction dir seed {seed}")

        for model in MODELS:
            stem = MODEL_FILES[model]
            for prefix in PREFIXES:
                for split in ("val", "test"):
                    path = pred_dir / f"{stem}_prefix{prefix}_{split}.csv"
                    require(path.is_file(), f"Missing prediction file: {path}")
                    pred = rows(path)
                    prediction_files += 1
                    require(len(pred) == EXPECTED_COUNTS[split], f"Row count mismatch: {path}")
                    require([r["case_id"] for r in pred] == expected_ids[split],
                            f"Case ID/order mismatch: {path}")
                    require(all(int(r["seed"]) == seed for r in pred),
                            f"Seed metadata mismatch: {path}")
                    require(all(int(r["prefix_k"]) == prefix for r in pred),
                            f"Prefix metadata mismatch: {path}")
                    require(all(r["model_name"] == model for r in pred),
                            f"Model metadata mismatch: {path}")

                    truth = np.asarray([int(r["true_label"]) for r in pred], dtype=int)
                    prob = np.asarray([float(r["predicted_probability"]) for r in pred], dtype=float)
                    yhat = np.asarray([int(r["predicted_label"]) for r in pred], dtype=int)
                    require(np.isfinite(prob).all() and np.all((prob >= 0) & (prob <= 1)),
                            f"Invalid probability: {path}")
                    require(np.array_equal(yhat, (prob >= 0.5).astype(int)),
                            f"Threshold mismatch: {path}")

                    if split == "test" and len(set(yhat.tolist())) < 2:
                        constant_test.append(f"seed{seed}/{model}/P{prefix}")

                    calc = binary_metrics(truth, prob)
                    rec = index[(model, prefix, split, seed)]
                    require(int(rec["n_cases"]) == len(pred), f"n_cases mismatch: {path}")
                    for metric, value in calc.items():
                        require(close(rec[metric], value),
                                f"Metric mismatch {metric}: {path}")

    require(prediction_files == expected_metric_rows,
            "Prediction inventory count mismatch")
    require(not constant_test, f"Constant test predictions: {constant_test}")
    checks["predictions_and_metrics_reproduced"] = True

    # Test must be accessed exactly once for each seed/model/prefix.
    access = read_json(RESULTS / "logs/test_access_log_phase2b.json")
    expected_test_evals = len(SEEDS) * len(MODELS) * len(PREFIXES)
    require(len(access) == expected_test_evals,
            f"Expected {expected_test_evals} test access entries")
    require(all(item["access_count"] == 1 for item in access),
            "A test set was evaluated more than once")
    require(
        len({(int(x["seed"]), x["model"], int(x["prefix"])) for x in access})
        == expected_test_evals,
        "Duplicate test access key",
    )
    checks["single_test_evaluation_per_run"] = True

    # Check checkpoint inventory/loadability.
    checkpoint_count = 0
    for seed in SEEDS:
        ckpt_dir = RESULTS / "checkpoints" / f"seed_{seed}"
        for model in MODELS:
            stem = MODEL_FILES[model]
            for prefix in PREFIXES:
                suffix = "joblib" if model in ("RF", "XGBoost") else "pt"
                path = ckpt_dir / f"{stem}_prefix{prefix}.{suffix}"
                require(path.is_file() and path.stat().st_size > 0,
                        f"Missing checkpoint: {path}")
                checkpoint_count += 1
                if suffix == "joblib":
                    loaded = joblib.load(path)
                    require(hasattr(loaded, "predict_proba"),
                            f"Invalid classical checkpoint: {path}")
                    # Verify only random_state differs from the frozen base config.
                    expected = phase2a["models"][model][str(prefix)]["hyperparameters"]
                    params = loaded.get_params()
                    for key, val in expected.items():
                        if key == "random_state":
                            require(int(params[key]) == seed,
                                    f"random_state mismatch: {path}")
                        elif key in params:
                            require(params[key] == val,
                                    f"Frozen parameter changed ({key}): {path}")
                else:
                    loaded = torch.load(path, map_location="cpu", weights_only=False)
                    require(loaded["model_name"] == model, f"Model metadata mismatch: {path}")
                    require(int(loaded["prefix"]) == prefix, f"Prefix metadata mismatch: {path}")
                    require(int(loaded["seed"]) == seed, f"Seed metadata mismatch: {path}")
                    expected = phase2a["models"][model][str(prefix)]["hyperparameters"]
                    require(
                        strip_randomness(model, loaded["config"])
                        == strip_randomness(model, expected),
                        f"Frozen neural config changed: {path}",
                    )
    require(checkpoint_count == expected_test_evals, "Checkpoint count mismatch")
    checks["all_checkpoints_load_and_match_frozen_configs"] = True

    # Recompute mean/std/CI exactly.
    summary = rows(RESULTS / "metrics/formal_metrics_summary.csv")
    expected_summary_rows = len(MODELS) * len(PREFIXES) * 2
    require(len(summary) == expected_summary_rows,
            f"Expected {expected_summary_rows} summary rows")
    sindex = {
        (r["model"], int(r["prefix"]), r["split"]): r
        for r in summary
    }
    require(len(sindex) == expected_summary_rows, "Duplicate summary key")

    for model in MODELS:
        for prefix in PREFIXES:
            for split in ("val", "test"):
                group = [
                    index[(model, prefix, split, seed)]
                    for seed in SEEDS
                ]
                sr = sindex[(model, prefix, split)]
                require(int(sr["n_seeds"]) == len(SEEDS), "n_seeds mismatch")
                require(sr["seeds"] == ",".join(map(str, SEEDS)), "Seed serialization mismatch")

                for metric in METRICS:
                    values = np.asarray([float(r[metric]) for r in group], dtype=float)
                    mean = float(values.mean())
                    std = float(values.std(ddof=1))
                    half = T_CRIT_95_DF4 * std / math.sqrt(len(values))
                    require(close(sr[f"{metric}_mean"], mean), f"{metric} mean mismatch")
                    require(close(sr[f"{metric}_std"], std), f"{metric} std mismatch")
                    require(close(sr[f"{metric}_ci95_low"], mean-half), f"{metric} CI low mismatch")
                    require(close(sr[f"{metric}_ci95_high"], mean+half), f"{metric} CI high mismatch")
    checks["summary_mean_std_ci_reproduced"] = True

    result = {
        "scope": "ETCE Phase 2B",
        "status": "passed",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "formal_seeds": list(SEEDS),
        "models": list(MODELS),
        "prefixes": list(PREFIXES),
        "checks": checks,
        "prediction_files": prediction_files,
        "checkpoint_count": checkpoint_count,
        "metric_rows": len(per_seed),
        "summary_rows": len(summary),
        "test_evaluations": expected_test_evals,
    }
    write_json(RESULTS / "logs/validation_phase2b.json", result)

    report_path = RESULTS / "PHASE2B_REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    old = "<!-- VALIDATION_START -->\nAwaiting Phase 2B validation.\n<!-- VALIDATION_END -->"
    new = (
        "<!-- VALIDATION_START -->\n"
        "Post-run validation: **PASSED**. Per-seed predictions, frozen configurations, "
        "single test access, checkpoints, and mean/std/95% CI were independently reproduced.\n"
        "<!-- VALIDATION_END -->"
    )
    require(old in report, "Report validation marker missing")
    report_path.write_text(report.replace(old, new), encoding="utf-8")

    manifest = read_json(RESULTS / "run_manifest_phase2b.json")
    manifest["status"] = "validated_passed"
    manifest["validated_utc"] = result["validated_utc"]
    manifest["validation_sha256"] = sha256(RESULTS / "logs/validation_phase2b.json")
    manifest["report_sha256_after_validation"] = sha256(report_path)
    write_json(RESULTS / "run_manifest_phase2b.json", manifest)

    return result


if __name__ == "__main__":
    try:
        outcome = validate()
    except Exception as exc:
        failure = {
            "scope": "ETCE Phase 2B",
            "status": "failed",
            "validated_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
        write_json(RESULTS / "logs/validation_phase2b.json", failure)
        raise

    print(json.dumps(outcome, ensure_ascii=False, indent=2))
