#!/usr/bin/env python3
"""Stage 67: audit frozen Phase-2B predictions and build the black-box test table.

This script is intentionally evaluation-only. It does not load model checkpoints,
train models, select hyperparameters, or tune thresholds on the test set.

Expected input layout (relative to --project-root):
  results/phase2b/
    configs/frozen_phase2a_configs_used.json
    logs/test_access_log_phase2b.json
    predictions/seed_<seed>/<model>_prefix<prefix>_test.csv

The frozen prediction CSV schema is:
  case_id, prefix_k, true_label, predicted_label, predicted_probability,
  split, model_name, config_id, seed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


MODEL_ORDER = ["RF", "XGBoost", "MLP", "LSTM"]
MODEL_FILE_STEMS = {
    "RF": "rf",
    "XGBoost": "xgboost",
    "MLP": "mlp",
    "LSTM": "lstm",
}
PREFIX_ORDER = [5, 10, 15, 20]
REQUIRED_COLUMNS = {
    "case_id",
    "prefix_k",
    "true_label",
    "predicted_label",
    "predicted_probability",
    "split",
    "model_name",
    "config_id",
    "seed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Macro-F1, ROC-AUC, and accuracy from the frozen Phase-2B "
            "BPIC2017 TEST predictions and generate CSV/JSON/LaTeX outputs."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"D:\ETCE"),
        help=r"ETCE project root (default: D:\ETCE).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Default: "
            "<project-root>/results/stage67_blackbox_performance"
        ),
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    require(path.is_file(), f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cohort_hash(frame: pd.DataFrame) -> str:
    canonical = (
        frame.loc[:, ["case_id", "true_label"]]
        .assign(case_id=lambda x: x["case_id"].astype(str))
        .sort_values(["case_id", "true_label"], kind="mergesort")
    )
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_config_id(config: dict[str, Any], model: str, prefix: int) -> str:
    entry = config["models"][model][str(prefix)]
    base = str(entry["config_id"])
    return base


def frozen_threshold(config: dict[str, Any], model: str, prefix: int) -> float:
    entry = config["models"][model][str(prefix)]
    threshold = float(entry.get("threshold", config.get("prediction_threshold", 0.5)))
    require(math.isfinite(threshold), f"Non-finite threshold for {model}, P{prefix}")
    require(0.0 <= threshold <= 1.0, f"Threshold outside [0,1] for {model}, P{prefix}")
    return threshold


def validate_config(config: dict[str, Any]) -> list[int]:
    require("formal_seeds" in config, "Missing formal_seeds in frozen config")
    require("models" in config, "Missing models in frozen config")
    seeds = [int(seed) for seed in config["formal_seeds"]]
    require(len(seeds) == len(set(seeds)), "formal_seeds contains duplicates")
    require(seeds == [13, 21, 42, 73, 101], f"Unexpected formal_seeds: {seeds}")

    for model in MODEL_ORDER:
        require(model in config["models"], f"Frozen config is missing model {model}")
        for prefix in PREFIX_ORDER:
            key = str(prefix)
            require(key in config["models"][model], f"Missing config for {model}, P{prefix}")
            entry = config["models"][model][key]
            require(
                entry.get("selection_metric") == "validation_macro_f1",
                f"Unexpected selection metric for {model}, P{prefix}: "
                f"{entry.get('selection_metric')}",
            )
            require(
                entry.get("test_used_for_selection") is False,
                f"Test appears to have been used for selection: {model}, P{prefix}",
            )
            frozen_threshold(config, model, prefix)
    return seeds


def validate_access_log(
    access_log: list[dict[str, Any]], seeds: list[int], config: dict[str, Any]
) -> None:
    expected_keys = {
        (seed, model, prefix)
        for seed in seeds
        for model in MODEL_ORDER
        for prefix in PREFIX_ORDER
    }
    observed: dict[tuple[int, str, int], dict[str, Any]] = {}
    for item in access_log:
        key = (int(item["seed"]), str(item["model"]), int(item["prefix"]))
        require(key not in observed, f"Duplicate test-access entry: {key}")
        observed[key] = item

    require(
        set(observed) == expected_keys,
        "Test-access log does not exactly cover the expected 80 seed/model/prefix combinations",
    )
    for (seed, model, prefix), item in observed.items():
        require(int(item.get("access_count", -1)) == 1, f"Unexpected access_count: {item}")
        require(
            item.get("purpose") == "single final evaluation using frozen Phase 2A configuration",
            f"Unexpected TEST access purpose: {item}",
        )
        expected = f"{expected_config_id(config, model, prefix)}_seed{seed}"
        require(item.get("config_id") == expected, f"Access-log config mismatch: {item}")


def evaluate_one_file(
    path: Path,
    seed: int,
    model: str,
    prefix: int,
    threshold: float,
    config_base_id: str,
) -> tuple[dict[str, Any], str]:
    require(path.is_file(), f"Missing frozen TEST prediction file: {path}")
    frame = pd.read_csv(path)
    require(not frame.empty, f"Prediction file is empty: {path}")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    require(not missing, f"Missing columns {sorted(missing)} in {path}")
    require(frame["case_id"].notna().all(), f"Missing case_id in {path}")
    require(frame["case_id"].astype(str).is_unique, f"Duplicate case_id in {path}")
    require(set(frame["split"].astype(str).str.lower()) == {"test"}, f"Non-TEST rows in {path}")
    require(set(frame["prefix_k"].astype(int)) == {prefix}, f"prefix_k mismatch in {path}")
    require(set(frame["model_name"].astype(str)) == {model}, f"model_name mismatch in {path}")
    require(set(frame["seed"].astype(int)) == {seed}, f"seed mismatch in {path}")

    expected_id = f"{config_base_id}_seed{seed}"
    require(set(frame["config_id"].astype(str)) == {expected_id}, f"config_id mismatch in {path}")

    y_true = pd.to_numeric(frame["true_label"], errors="raise").to_numpy(dtype=int)
    y_saved = pd.to_numeric(frame["predicted_label"], errors="raise").to_numpy(dtype=int)
    y_prob = pd.to_numeric(frame["predicted_probability"], errors="raise").to_numpy(dtype=float)
    require(set(np.unique(y_true)).issubset({0, 1}), f"Non-binary true_label in {path}")
    require(set(np.unique(y_saved)).issubset({0, 1}), f"Non-binary predicted_label in {path}")
    require(np.isfinite(y_prob).all(), f"Non-finite predicted_probability in {path}")
    require(((0.0 <= y_prob) & (y_prob <= 1.0)).all(), f"Probability outside [0,1] in {path}")
    require(len(np.unique(y_true)) == 2, f"ROC-AUC undefined because TEST labels have one class: {path}")

    y_recomputed = (y_prob >= threshold).astype(int)
    require(
        np.array_equal(y_saved, y_recomputed),
        f"Saved predictions are not reproduced by frozen threshold={threshold}: {path}",
    )

    metrics = {
        "seed": seed,
        "prefix": prefix,
        "model": model,
        "n_test": int(len(frame)),
        "n_negative": int((y_true == 0).sum()),
        "n_positive": int((y_true == 1).sum()),
        "positive_rate": float((y_true == 1).mean()),
        "threshold": threshold,
        "macro_f1": float(f1_score(y_true, y_saved, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_saved)),
        "config_id": expected_id,
        "prediction_file": str(path.resolve()),
        "prediction_sha256": sha256_file(path),
    }
    return metrics, cohort_hash(frame)


def build_summary(seedwise: pd.DataFrame) -> pd.DataFrame:
    grouped = seedwise.groupby(["prefix", "model"], sort=False)
    rows: list[dict[str, Any]] = []
    for (prefix, model), group in grouped:
        require(group["seed"].nunique() == 5, f"Expected five seeds for {model}, P{prefix}")
        for column in ["n_test", "n_negative", "n_positive", "threshold"]:
            require(group[column].nunique() == 1, f"{column} differs across seeds for {model}, P{prefix}")
        row: dict[str, Any] = {
            "prefix": int(prefix),
            "model": str(model),
            "n_test": int(group["n_test"].iloc[0]),
            "n_negative": int(group["n_negative"].iloc[0]),
            "n_positive": int(group["n_positive"].iloc[0]),
            "threshold": float(group["threshold"].iloc[0]),
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in ["macro_f1", "roc_auc", "accuracy"]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["_prefix_order"] = pd.Categorical(summary["prefix"], PREFIX_ORDER, ordered=True)
    summary["_model_order"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    return (
        summary.sort_values(["_prefix_order", "_model_order"])
        .drop(columns=["_prefix_order", "_model_order"])
        .reset_index(drop=True)
    )


def format_pm(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def build_paper_csv(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            {
                "Prefix": row.prefix,
                "Predictor": row.model,
                "N": row.n_test,
                "Macro-F1 (mean ± std)": format_pm(row.macro_f1_mean, row.macro_f1_std),
                "ROC-AUC (mean ± std)": format_pm(row.roc_auc_mean, row.roc_auc_std),
                "Accuracy (mean ± std)": format_pm(row.accuracy_mean, row.accuracy_std),
                "Frozen threshold": f"{row.threshold:.2f}",
            }
        )
    return pd.DataFrame(rows)


def latex_pm(mean: float, std: float) -> str:
    return f"${mean:.4f} \\pm {std:.4f}$"


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Predictive performance of the frozen black-box models on the complete held-out BPIC2017 test split. Values are mean $\pm$ standard deviation over the five predeclared seeds. The binary decision threshold was fixed at 0.50 before test evaluation.}",
        r"\label{tab:blackbox_test_performance}",
        r"\small",
        r"\begin{tabular}{clrrrrr}",
        r"\toprule",
        r"Prefix & Predictor & $N$ & Macro-F1 & ROC-AUC & Accuracy & Threshold \\",
        r"\midrule",
    ]
    last_prefix: int | None = None
    for row in summary.itertuples(index=False):
        if last_prefix is not None and row.prefix != last_prefix:
            lines.append(r"\midrule")
        lines.append(
            f"{row.prefix} & {row.model} & {row.n_test} & "
            f"{latex_pm(row.macro_f1_mean, row.macro_f1_std)} & "
            f"{latex_pm(row.roc_auc_mean, row.roc_auc_std)} & "
            f"{latex_pm(row.accuracy_mean, row.accuracy_std)} & "
            f"{row.threshold:.2f} \\\\"
        )
        last_prefix = row.prefix
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    phase2b_root = project_root / "results" / "phase2b"
    prediction_root = phase2b_root / "predictions"
    config_path = phase2b_root / "configs" / "frozen_phase2a_configs_used.json"
    access_log_path = phase2b_root / "logs" / "test_access_log_phase2b.json"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / "results" / "stage67_blackbox_performance"
    )

    config = read_json(config_path)
    seeds = validate_config(config)
    access_log = read_json(access_log_path)
    require(isinstance(access_log, list), "TEST access log must be a JSON list")
    validate_access_log(access_log, seeds, config)

    rows: list[dict[str, Any]] = []
    cohort_hashes: dict[str, str] = {}
    for seed in seeds:
        for prefix in PREFIX_ORDER:
            for model in MODEL_ORDER:
                stem = MODEL_FILE_STEMS[model]
                path = prediction_root / f"seed_{seed}" / f"{stem}_prefix{prefix}_test.csv"
                threshold = frozen_threshold(config, model, prefix)
                base_id = expected_config_id(config, model, prefix)
                row, current_hash = evaluate_one_file(
                    path=path,
                    seed=seed,
                    model=model,
                    prefix=prefix,
                    threshold=threshold,
                    config_base_id=base_id,
                )
                rows.append(row)
                hash_key = f"P{prefix}"
                if hash_key in cohort_hashes:
                    require(
                        cohort_hashes[hash_key] == current_hash,
                        f"TEST cohort or labels differ across models/seeds for prefix P{prefix}",
                    )
                else:
                    cohort_hashes[hash_key] = current_hash

    require(len(rows) == 80, f"Expected 80 metric rows, found {len(rows)}")
    seedwise = pd.DataFrame(rows)
    seedwise["_prefix_order"] = pd.Categorical(seedwise["prefix"], PREFIX_ORDER, ordered=True)
    seedwise["_model_order"] = pd.Categorical(seedwise["model"], MODEL_ORDER, ordered=True)
    seedwise = (
        seedwise.sort_values(["_prefix_order", "_model_order", "seed"])
        .drop(columns=["_prefix_order", "_model_order"])
        .reset_index(drop=True)
    )
    summary = build_summary(seedwise)
    paper = build_paper_csv(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    seedwise_path = output_dir / "blackbox_test_metrics_by_seed.csv"
    summary_path = output_dir / "blackbox_test_metrics_summary.csv"
    paper_csv_path = output_dir / "blackbox_test_performance_paper_table.csv"
    latex_path = output_dir / "blackbox_test_performance_table.tex"
    audit_path = output_dir / "blackbox_test_performance_audit.json"

    seedwise.to_csv(seedwise_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    paper.to_csv(paper_csv_path, index=False, encoding="utf-8-sig")
    write_latex_table(summary, latex_path)

    thresholds = sorted({float(value) for value in summary["threshold"]})
    audit = {
        "stage": 67,
        "purpose": "evaluation-only black-box TEST performance table",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "input_prediction_root": str(prediction_root),
        "frozen_config": str(config_path),
        "frozen_config_sha256": sha256_file(config_path),
        "test_access_log": str(access_log_path),
        "test_access_log_sha256": sha256_file(access_log_path),
        "models": MODEL_ORDER,
        "prefixes": PREFIX_ORDER,
        "seeds": seeds,
        "prediction_file_count": len(rows),
        "frozen_thresholds_observed": thresholds,
        "selection_metric": "validation_macro_f1",
        "test_used_for_selection": False,
        "cohort_hash_by_prefix": cohort_hashes,
        "checks": {
            "all_80_prediction_files_present": True,
            "all_rows_are_test_split": True,
            "saved_labels_reproduced_by_frozen_thresholds": True,
            "same_test_cohort_and_labels_across_models_and_seeds_within_prefix": True,
            "test_access_count_exactly_one_per_combination": True,
            "no_training_or_threshold_tuning_performed": True,
        },
        "outputs": {
            "seedwise_csv": str(seedwise_path),
            "summary_csv": str(summary_path),
            "paper_csv": str(paper_csv_path),
            "latex_table": str(latex_path),
        },
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("ETCE STAGE 67 - FROZEN BLACK-BOX TEST PERFORMANCE")
    print("=" * 100)
    print(f"Prediction files audited : {len(rows)}")
    print(f"Formal seeds             : {seeds}")
    print(f"Models                   : {MODEL_ORDER}")
    print(f"Prefixes                 : {PREFIX_ORDER}")
    print(f"Frozen threshold(s)      : {thresholds}")
    print("Training performed       : False")
    print("Threshold tuning         : False")
    print("\nPaper-ready results:\n")
    print(paper.to_string(index=False))
    print(f"\nOutputs: {output_dir}")
    print("STAGE 67 PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"STAGE 67 FAILED: {exc}", file=sys.stderr)
        raise
