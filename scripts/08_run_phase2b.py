"""ETCE Phase 2B formal multi-seed black-box experiment.

This script reuses the already validated Phase 2A data/encoding/model implementation.
It DOES NOT retune hyperparameters. It freezes Phase 2A best configs and changes only
the random seed/random_state for formal repeated runs.

Expected project placement:
    ETCE/scripts/08_run_phase2b.py

Run:
    python scripts/08_run_phase2b.py
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import csv
import json
import math
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier

from etce.phase2.data import COHORT_HASHES, EXPECTED_COUNTS, PREFIXES, load_frozen_common_cohort
from etce.phase2.encoding import fit_encode_prefix
from etce.phase2.metrics import binary_metrics
from etce.phase2.models import predict_torch, set_pilot_seed, train_torch_candidate
from etce.utils.io import read_json, sha256, write_csv, write_json

SEEDS = (13, 21, 42, 73, 101)
MODELS = ("RF", "XGBoost", "MLP", "LSTM")
MODEL_FILES = {"RF": "rf", "XGBoost": "xgboost", "MLP": "mlp", "LSTM": "lstm"}
THRESHOLD = 0.5
PHASE2A_CONFIG = ROOT / "results/phase2/configs/best_configs.json"
RESULTS = ROOT / "results/phase2b"

# Two-sided 95% Student-t critical value for df=4 (n=5 seeds).
T_CRIT_95_DF4 = 2.7764451051977987


def deep_copy(obj):
    return json.loads(json.dumps(obj))


def prediction_rows(model, prefix, split, data, probability, config_id, seed):
    predicted = (np.asarray(probability) >= THRESHOLD).astype(int)
    return [
        {
            "case_id": case_id,
            "prefix_k": prefix,
            "true_label": int(label),
            "predicted_label": int(pred),
            "predicted_probability": float(prob),
            "split": split,
            "model_name": model,
            "config_id": config_id,
            "seed": int(seed),
        }
        for case_id, label, pred, prob in zip(
            data.case_ids, data.labels, predicted, probability
        )
    ]


def metric_row(model, prefix, split, data, probability, config_id, seed):
    row = {
        "model": model,
        "prefix": int(prefix),
        "split": split,
        "n_cases": int(len(data.labels)),
        "positive": int(data.labels.sum()),
        "negative": int(len(data.labels) - data.labels.sum()),
        "seed": int(seed),
        "config_id": config_id,
    }
    row.update(binary_metrics(data.labels, np.asarray(probability)))
    return row


def frozen_seeded_config(model_name, prefix, seed, frozen):
    """Copy Phase 2A winning hyperparameters and change ONLY randomness fields."""
    item = frozen["models"][model_name][str(prefix)]
    config = deep_copy(item["hyperparameters"])

    if model_name in ("RF", "XGBoost"):
        config["random_state"] = int(seed)
    else:
        config["seed"] = int(seed)

    return item["config_id"], config


def save_checkpoint(model_name, prefix, seed, model, config, config_id, encoder, extra):
    out_dir = RESULTS / "checkpoints" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = MODEL_FILES[model_name]

    if model_name in ("RF", "XGBoost"):
        joblib.dump(model, out_dir / f"{stem}_prefix{prefix}.joblib")
    else:
        torch.save(
            {
                "phase": "2B",
                "model_name": model_name,
                "prefix": int(prefix),
                "config_id": config_id,
                "config": config,
                "state_dict": model.state_dict(),
                "encoder_vocabulary_size": len(encoder["vocabulary"]) + 1,
                "seed": int(seed),
                "training": extra,
            },
            out_dir / f"{stem}_prefix{prefix}.pt",
        )


def summarize(rows):
    """Aggregate per-seed metrics to mean/std/t-based 95% CI."""
    metric_names = ("accuracy", "macro_f1", "roc_auc", "balanced_accuracy")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], int(row["prefix"]), row["split"])].append(row)

    summary = []
    for (model, prefix, split), group in sorted(grouped.items()):
        group = sorted(group, key=lambda r: int(r["seed"]))
        seeds = [int(r["seed"]) for r in group]
        if tuple(seeds) != SEEDS:
            raise AssertionError(
                f"Seed set mismatch for {model}/P{prefix}/{split}: {seeds}"
            )

        base = {
            "model": model,
            "prefix": prefix,
            "split": split,
            "n_seeds": len(group),
            "seeds": ",".join(map(str, seeds)),
            "n_cases_per_seed": int(group[0]["n_cases"]),
            "positive_per_seed": int(group[0]["positive"]),
            "negative_per_seed": int(group[0]["negative"]),
        }

        for metric in metric_names:
            values = np.asarray([float(r[metric]) for r in group], dtype=float)
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            half = float(T_CRIT_95_DF4 * std / math.sqrt(len(values)))
            base[f"{metric}_mean"] = mean
            base[f"{metric}_std"] = std
            base[f"{metric}_ci95_low"] = mean - half
            base[f"{metric}_ci95_high"] = mean + half

        summary.append(base)

    return summary


def build_report(summary, started):
    test = [r for r in summary if r["split"] == "test"]
    test = sorted(test, key=lambda r: (int(r["prefix"]), r["model"]))

    lines = []
    for r in test:
        lines.append(
            "| {model} | {prefix} | "
            "{acc:.4f} ± {accs:.4f} | "
            "{f1:.4f} ± {f1s:.4f} | "
            "{auc:.4f} ± {aucs:.4f} | "
            "{ba:.4f} ± {bas:.4f} | "
            "[{lo:.4f}, {hi:.4f}] |".format(
                model=r["model"],
                prefix=r["prefix"],
                acc=r["accuracy_mean"],
                accs=r["accuracy_std"],
                f1=r["macro_f1_mean"],
                f1s=r["macro_f1_std"],
                auc=r["roc_auc_mean"],
                aucs=r["roc_auc_std"],
                ba=r["balanced_accuracy_mean"],
                bas=r["balanced_accuracy_std"],
                lo=r["macro_f1_ci95_low"],
                hi=r["macro_f1_ci95_high"],
            )
        )

    text = f"""# ETCE Phase 2B Formal Multi-seed Report

## Protocol

- Frozen BPIC2017 common cohort from Phase 1.2.
- Prefixes: {list(PREFIXES)}.
- Formal seeds: {list(SEEDS)}.
- Hyperparameters are frozen from Phase 2A validation selection.
- Phase 2B does **not** perform a new hyperparameter search.
- For RF/XGBoost only `random_state` changes with seed.
- For MLP/LSTM only `seed` changes with seed; validation remains available only for early stopping/checkpoint selection under the already frozen architecture and optimizer settings.
- Prediction threshold: {THRESHOLD}.
- Test is evaluated once per `(seed, model, prefix)`.
- 95% CI: Student-t interval over the 5 formal seeds (df=4).
- Started UTC: {started}.

## Test results

| Model | Prefix | Accuracy | Macro-F1 | ROC-AUC | Balanced Acc. | Macro-F1 95% CI |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

Detailed per-seed results are in `metrics/formal_metrics_per_seed.csv`.
Full mean/std/95% CI results are in `metrics/formal_metrics_summary.csv`.

## Scope

This phase evaluates only the four frozen black-box predictors. No counterfactual generation,
surrogate extraction, GA, BPA, D-S fusion, process-validity filter, or ETCE explanation experiment
is run here.

<!-- VALIDATION_START -->
Awaiting Phase 2B validation.
<!-- VALIDATION_END -->
"""
    (RESULTS / "PHASE2B_REPORT.md").write_text(text, encoding="utf-8")


def main():
    started = datetime.now(timezone.utc).isoformat()

    if not PHASE2A_CONFIG.is_file():
        raise FileNotFoundError(
            f"Missing frozen Phase 2A config: {PHASE2A_CONFIG}"
        )

    frozen = read_json(PHASE2A_CONFIG)
    if frozen.get("pilot_seed") != 0:
        raise AssertionError("Unexpected Phase 2A pilot seed")
    if float(frozen.get("prediction_threshold")) != THRESHOLD:
        raise AssertionError("Unexpected Phase 2A prediction threshold")

    for directory in (
        "artifacts",
        "checkpoints",
        "configs",
        "logs",
        "metrics",
        "predictions",
    ):
        (RESULTS / directory).mkdir(parents=True, exist_ok=True)

    try:
        from xgboost import XGBClassifier
        import xgboost
    except ImportError as exc:
        raise RuntimeError(
            "Official xgboost is required; no substitute permitted"
        ) from exc

    # Re-derive and validate frozen Phase 1.2 cohort through the existing validated loader.
    cases, data_audit = load_frozen_common_cohort(ROOT)

    by_split = {
        split: [case for case in cases if case.split == split]
        for split in ("train", "val", "test")
    }
    for split in ("train", "val", "test"):
        if len(by_split[split]) != EXPECTED_COUNTS[split]:
            raise AssertionError(f"{split} frozen cohort count mismatch")

    # Encode each prefix ONCE; encoding is train-fit-only and does not depend on seed.
    encoded_by_prefix = {}
    encoder_by_prefix = {}
    for prefix in PREFIXES:
        encoded, encoder = fit_encode_prefix(
            cases, prefix, RESULTS / "artifacts"
        )
        encoded_by_prefix[int(prefix)] = encoded
        encoder_by_prefix[int(prefix)] = encoder

    metric_rows = []
    run_log = []
    test_access = []

    total = len(SEEDS) * len(PREFIXES) * len(MODELS)
    done = 0

    for seed in SEEDS:
        for prefix in PREFIXES:
            encoded = encoded_by_prefix[int(prefix)]
            encoder = encoder_by_prefix[int(prefix)]
            train, val, test = (
                encoded["train"],
                encoded["val"],
                encoded["test"],
            )

            for model_name in MODELS:
                done += 1
                set_pilot_seed(int(seed))
                base_config_id, config = frozen_seeded_config(
                    model_name, prefix, seed, frozen
                )
                config_id = f"{base_config_id}_seed{seed}"
                t0 = time.perf_counter()
                extra = {}

                print(
                    f"[{done:02d}/{total}] seed={seed} "
                    f"{model_name} prefix={prefix}",
                    flush=True,
                )

                if model_name == "RF":
                    model = RandomForestClassifier(**config)
                    model.fit(train.tabular, train.labels)
                    val_probability = model.predict_proba(val.tabular)[:, 1]

                elif model_name == "XGBoost":
                    model = XGBClassifier(**config)
                    model.fit(train.tabular, train.labels)
                    val_probability = model.predict_proba(val.tabular)[:, 1]

                elif model_name == "MLP":
                    model, extra = train_torch_candidate(
                        "mlp", train, val, config, int(seed)
                    )
                    val_probability = predict_torch(
                        model, "mlp", val, config["batch_size"]
                    )

                else:
                    model, extra = train_torch_candidate(
                        "lstm",
                        train,
                        val,
                        config,
                        int(seed),
                        len(encoder["vocabulary"]) + 1,
                    )
                    val_probability = predict_torch(
                        model, "lstm", val, config["batch_size"]
                    )

                # Test is touched only after the frozen model has been trained.
                if model_name in ("RF", "XGBoost"):
                    test_probability = model.predict_proba(test.tabular)[:, 1]
                else:
                    test_probability = predict_torch(
                        model,
                        model_name.lower(),
                        test,
                        config["batch_size"],
                    )

                elapsed = time.perf_counter() - t0

                save_checkpoint(
                    model_name,
                    prefix,
                    seed,
                    model,
                    config,
                    config_id,
                    encoder,
                    extra,
                )

                pred_dir = RESULTS / "predictions" / f"seed_{seed}"
                pred_dir.mkdir(parents=True, exist_ok=True)
                stem = MODEL_FILES[model_name]

                for split, data, probability in (
                    ("val", val, val_probability),
                    ("test", test, test_probability),
                ):
                    write_csv(
                        pred_dir / f"{stem}_prefix{prefix}_{split}.csv",
                        prediction_rows(
                            model_name,
                            prefix,
                            split,
                            data,
                            probability,
                            config_id,
                            seed,
                        ),
                    )
                    metric_rows.append(
                        metric_row(
                            model_name,
                            prefix,
                            split,
                            data,
                            probability,
                            config_id,
                            seed,
                        )
                    )

                test_access.append(
                    {
                        "seed": int(seed),
                        "model": model_name,
                        "prefix": int(prefix),
                        "config_id": config_id,
                        "access_count": 1,
                        "purpose": "single final evaluation using frozen Phase 2A configuration",
                    }
                )
                run_log.append(
                    {
                        "seed": int(seed),
                        "model": model_name,
                        "prefix": int(prefix),
                        "config_id": config_id,
                        "training_seconds": float(elapsed),
                        "training_metadata": extra,
                    }
                )

                print(
                    f"    done in {elapsed:.1f}s "
                    f"val_F1={binary_metrics(val.labels, val_probability)['macro_f1']:.4f} "
                    f"test_F1={binary_metrics(test.labels, test_probability)['macro_f1']:.4f}",
                    flush=True,
                )

    summary_rows = summarize(metric_rows)

    write_csv(RESULTS / "metrics/formal_metrics_per_seed.csv", metric_rows)
    write_csv(RESULTS / "metrics/formal_metrics_summary.csv", summary_rows)
    write_json(RESULTS / "logs/training_log_phase2b.json", run_log)
    write_json(RESULTS / "logs/test_access_log_phase2b.json", test_access)

    write_json(
        RESULTS / "configs/frozen_phase2a_configs_used.json",
        {
            "source": PHASE2A_CONFIG.relative_to(ROOT).as_posix(),
            "source_sha256": sha256(PHASE2A_CONFIG),
            "formal_seeds": list(SEEDS),
            "prediction_threshold": THRESHOLD,
            "models": frozen["models"],
            "note": (
                "All hyperparameters are frozen from Phase 2A. "
                "Only random_state/seed is replaced by the formal run seed."
            ),
        },
    )

    build_report(summary_rows, started)

    source_files = sorted((ROOT / "src/etce/phase2").glob("*.py"))
    source_files += [ROOT / "src/etce/utils/io.py", Path(__file__).resolve()]
    source_files = [p for p in source_files if p.is_file()]

    output_files = [
        p
        for p in RESULTS.rglob("*")
        if p.is_file() and p.name != "run_manifest_phase2b.json"
    ]

    manifest = {
        "phase": "2B",
        "status": "formal_pending_validation",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "formal_seeds": list(SEEDS),
        "prefixes": list(PREFIXES),
        "models": list(MODELS),
        "raw_sha256": data_audit["raw_sha256"],
        "cohort_hashes": COHORT_HASHES,
        "expected_counts": EXPECTED_COUNTS,
        "phase2a_best_config_sha256": sha256(PHASE2A_CONFIG),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "source_hashes": {
            p.relative_to(ROOT).as_posix(): sha256(p)
            for p in source_files
        },
        "output_hashes_before_validation": {
            p.relative_to(ROOT).as_posix(): sha256(p)
            for p in sorted(output_files)
        },
        "test_evaluations": len(test_access),
        "hyperparameter_search_performed": False,
        "test_used_for_selection": False,
        "paper_statistics": True,
    }
    write_json(RESULTS / "run_manifest_phase2b.json", manifest)

    print("\nPhase 2B finished.")
    print("Report:", RESULTS / "PHASE2B_REPORT.md")
    print("Per-seed:", RESULTS / "metrics/formal_metrics_per_seed.csv")
    print("Summary:", RESULTS / "metrics/formal_metrics_summary.csv")
    print("Run scripts/09_validate_phase2b.py next.")


if __name__ == "__main__":
    main()
