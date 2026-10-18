#!/usr/bin/env python3
"""Stage 68A: frozen outcome-proxy masking sensitivity for BPIC2017 black boxes.

This stage changes one thing only: within each already frozen Prefix-k input,
O_Accepted, O_Refused, and O_Cancelled are mapped to one neutral activity
token, O_MASKED_OUTCOME. Event positions, timestamps, case labels, temporal
splits, common-cohort membership, model hyperparameters, formal seeds, and the
0.50 decision threshold remain unchanged.

The models are retrained because the input representation changes. No
hyperparameter search or threshold tuning is performed. Prefix 5 is audited
but not rerun because the frozen common cohort contains no masked activity in
its first five events.

Run from the ETCE project root:
    python -u scripts/68a_outcome_proxy_masked_blackbox.py --project-root D:\\ETCE

Use --overwrite only to archive an earlier Stage-68A output and start again.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from datetime import datetime, timezone


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="ETCE project root (default: parent of scripts directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Default: "
            "<project-root>/results/stage68_outcome_proxy_masking/blackbox"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Archive an existing output directory before starting a fresh run.",
    )
    return parser.parse_args()


ARGS = parse_args()
ROOT = ARGS.project_root.expanduser().resolve()
LOCAL_DEPS = ROOT / ".deps"
if sys.platform == "win32" and LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
import xgboost  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier  # noqa: E402

from etce.phase2.data import (  # noqa: E402
    COHORT_HASHES,
    EXPECTED_COUNTS,
    FrozenCase,
    load_frozen_common_cohort,
)
from etce.phase2.encoding import fit_encode_prefix  # noqa: E402
from etce.phase2.models import (  # noqa: E402
    predict_torch,
    set_pilot_seed,
    train_torch_candidate,
)


STAGE = "68A"
MODEL_ORDER = ("RF", "XGBoost", "MLP", "LSTM")
MODEL_STEMS = {
    "RF": "rf",
    "XGBoost": "xgboost",
    "MLP": "mlp",
    "LSTM": "lstm",
}
RERUN_PREFIXES = (10, 15, 20)
ALL_AUDIT_PREFIXES = (5, 10, 15, 20)
MASKED_ACTIVITIES = frozenset(
    {"O_Accepted", "O_Refused", "O_Cancelled"}
)
MASK_TOKEN = "O_MASKED_OUTCOME"
EXPECTED_AFFECTED_CASES = {5: 0, 10: 124, 15: 474, 20: 907}
THRESHOLD = 0.50

PHASE2B = ROOT / "results" / "phase2b"
FROZEN_CONFIG = PHASE2B / "configs" / "frozen_phase2a_configs_used.json"
ORIGINAL_CANONICAL = (
    ROOT / "results" / "phase3" / "canonical_blackboxes" / "canonical_manifest.json"
)
ORIGINAL_STAGE67 = (
    ROOT / "results" / "stage67_blackbox_performance" / "blackbox_test_metrics_summary.csv"
)
OUT = (
    ARGS.output_dir.expanduser().resolve()
    if ARGS.output_dir is not None
    else ROOT / "results" / "stage68_outcome_proxy_masking" / "blackbox"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path):
    require(path.is_file(), f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def prepare_output() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        require(
            ARGS.overwrite,
            f"Output already exists: {OUT}. Use --overwrite to archive it first.",
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = OUT.with_name(f"{OUT.name}_backup_{stamp}")
        require(not backup.exists(), f"Backup path already exists: {backup}")
        shutil.move(str(OUT), str(backup))
        print(f"Archived previous output: {backup}", flush=True)
    for name in (
        "artifacts",
        "checkpoints",
        "predictions",
        "metrics",
        "logs",
        "audit",
        "canonical_blackboxes/checkpoints",
    ):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def validate_frozen_config(config: dict) -> list[int]:
    seeds = [int(value) for value in config.get("formal_seeds", [])]
    require(seeds == [13, 21, 42, 73, 101], f"Unexpected formal seeds: {seeds}")
    require(math.isclose(float(config.get("prediction_threshold")), THRESHOLD),
            "Frozen prediction threshold is not 0.50")
    for model in MODEL_ORDER:
        require(model in config.get("models", {}), f"Missing frozen model: {model}")
        for prefix in RERUN_PREFIXES:
            entry = config["models"][model].get(str(prefix))
            require(entry is not None, f"Missing frozen config for {model}/P{prefix}")
            require(entry.get("selection_metric") == "validation_macro_f1",
                    f"Unexpected selection metric for {model}/P{prefix}")
            require(entry.get("test_used_for_selection") is False,
                    f"TEST was used for original selection: {model}/P{prefix}")
            require(math.isclose(float(entry.get("threshold", THRESHOLD)), THRESHOLD),
                    f"Unexpected threshold for {model}/P{prefix}")
    return seeds


def mask_cases(cases: list[FrozenCase]):
    require(
        all(MASK_TOKEN not in case.activities for case in cases),
        f"Mask token already occurs in original data: {MASK_TOKEN}",
    )
    affected_case_counts = {}
    affected_event_counts = {}
    affected_by_split = {}
    for prefix in ALL_AUDIT_PREFIXES:
        affected_case_counts[prefix] = sum(
            any(a in MASKED_ACTIVITIES for a in case.activities[:prefix])
            for case in cases
        )
        affected_event_counts[prefix] = sum(
            sum(a in MASKED_ACTIVITIES for a in case.activities[:prefix])
            for case in cases
        )
        affected_by_split[prefix] = {
            split: sum(
                case.split == split
                and any(a in MASKED_ACTIVITIES for a in case.activities[:prefix])
                for case in cases
            )
            for split in ("train", "val", "test")
        }
    require(
        affected_case_counts == EXPECTED_AFFECTED_CASES,
        "Historical outcome audit differs from the frozen manuscript protocol: "
        f"observed={affected_case_counts}, expected={EXPECTED_AFFECTED_CASES}",
    )

    masked = []
    total_replacements = 0
    for case in cases:
        activities = tuple(
            MASK_TOKEN if activity in MASKED_ACTIVITIES else activity
            for activity in case.activities
        )
        total_replacements += sum(
            old != new for old, new in zip(case.activities, activities)
        )
        masked_case = FrozenCase(
            case_id=case.case_id,
            split=case.split,
            label=case.label,
            label_event_index=case.label_event_index,
            activities=activities,
            timestamps=case.timestamps,
        )
        require(masked_case.case_id == case.case_id, "Masking changed case ID")
        require(masked_case.split == case.split, "Masking changed split")
        require(masked_case.label == case.label, "Masking changed label")
        require(masked_case.timestamps == case.timestamps, "Masking changed timestamps")
        require(len(masked_case.activities) == len(case.activities), "Masking changed length")
        masked.append(masked_case)
    require(total_replacements > 0, "No historical outcome activity was replaced")
    require(
        all(
            not any(a in MASKED_ACTIVITIES for a in case.activities[:20])
            for case in masked
        ),
        "At least one prohibited activity identity remains after masking",
    )
    return masked, {
        "mask_token": MASK_TOKEN,
        "masked_activity_identities": sorted(MASKED_ACTIVITIES),
        "affected_case_counts": affected_case_counts,
        "affected_event_counts": affected_event_counts,
        "affected_case_counts_by_split": affected_by_split,
        "total_replacements_within_prefix20": total_replacements,
        "prefix5_unchanged_by_construction": affected_case_counts[5] == 0,
    }


def metric_values(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = (probabilities >= THRESHOLD).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
    }


def prediction_frame(
    model: str,
    prefix: int,
    split: str,
    seed: int,
    data,
    probabilities: np.ndarray,
    config_id: str,
) -> pd.DataFrame:
    predicted = (probabilities >= THRESHOLD).astype(int)
    return pd.DataFrame(
        {
            "case_id": data.case_ids.astype(str),
            "prefix_k": prefix,
            "true_label": data.labels.astype(int),
            "predicted_label": predicted.astype(int),
            "predicted_probability": probabilities.astype(float),
            "split": split,
            "model_name": model,
            "config_id": config_id,
            "seed": seed,
            "mask_protocol": "shared_neutral_token",
            "mask_token": MASK_TOKEN,
        }
    )


def train_one(model_name, prefix, seed, train, val, test, encoder, frozen_entry):
    config = copy.deepcopy(frozen_entry["hyperparameters"])
    config_id = f"{frozen_entry['config_id']}_outcome_mask_seed{seed}"
    training_metadata = {}
    set_pilot_seed(seed)

    if model_name == "RF":
        config["random_state"] = seed
        model = RandomForestClassifier(**config)
        model.fit(train.tabular, train.labels)
        val_probability = model.predict_proba(val.tabular)[:, 1]
        test_probability = model.predict_proba(test.tabular)[:, 1]
    elif model_name == "XGBoost":
        config["random_state"] = seed
        model = XGBClassifier(**config)
        model.fit(train.tabular, train.labels)
        val_probability = model.predict_proba(val.tabular)[:, 1]
        test_probability = model.predict_proba(test.tabular)[:, 1]
    elif model_name == "MLP":
        config["seed"] = seed
        model, training_metadata = train_torch_candidate(
            "mlp", train, val, config, seed
        )
        val_probability = predict_torch(model, "mlp", val, config["batch_size"])
        test_probability = predict_torch(model, "mlp", test, config["batch_size"])
    elif model_name == "LSTM":
        config["seed"] = seed
        model, training_metadata = train_torch_candidate(
            "lstm",
            train,
            val,
            config,
            seed,
            len(encoder["vocabulary"]) + 1,
        )
        val_probability = predict_torch(model, "lstm", val, config["batch_size"])
        test_probability = predict_torch(model, "lstm", test, config["batch_size"])
    else:
        raise ValueError(model_name)

    return (
        model,
        config,
        config_id,
        training_metadata,
        np.asarray(val_probability, dtype=float),
        np.asarray(test_probability, dtype=float),
    )


def save_checkpoint(model_name, prefix, seed, model, config, training, encoder):
    stem = MODEL_STEMS[model_name]
    directory = OUT / "checkpoints" / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    if model_name in ("RF", "XGBoost"):
        path = directory / f"{stem}_prefix{prefix}.joblib"
        joblib.dump(model, path)
    else:
        path = directory / f"{stem}_prefix{prefix}.pt"
        torch.save(
            {
                "model_name": model_name,
                "prefix": prefix,
                "config": config,
                "state_dict": model.state_dict(),
                "encoder_vocabulary_size": len(encoder["vocabulary"]) + 1,
                "seed": seed,
                "training": training,
                "outcome_proxy_masking": {
                    "masked_activities": sorted(MASKED_ACTIVITIES),
                    "mask_token": MASK_TOKEN,
                    "positions_preserved": True,
                },
            },
            path,
        )
    return path


def summarize(seedwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (prefix, model), group in seedwise.groupby(["prefix", "model"], sort=False):
        row = {
            "prefix": int(prefix),
            "model": model,
            "n_test": int(group["n_test"].iloc[0]),
            "threshold": THRESHOLD,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in ("macro_f1", "roc_auc", "accuracy", "balanced_accuracy"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["prefix"] = frame["prefix"].astype(int)
    model_rank = {name: index for index, name in enumerate(MODEL_ORDER)}
    frame["_model_rank"] = frame["model"].map(model_rank)
    return frame.sort_values(["prefix", "_model_rank"]).drop(columns="_model_rank").reset_index(drop=True)


def build_comparison(masked_summary: pd.DataFrame) -> pd.DataFrame:
    require(ORIGINAL_STAGE67.is_file(), f"Stage 67 summary not found: {ORIGINAL_STAGE67}")
    original = pd.read_csv(ORIGINAL_STAGE67, encoding="utf-8-sig")
    required = {"prefix", "model", "macro_f1_mean", "macro_f1_std"}
    require(required.issubset(original.columns), "Unexpected Stage 67 summary schema")
    original = original[original["prefix"].astype(int).isin(RERUN_PREFIXES)].copy()
    merged = masked_summary.merge(
        original[["prefix", "model", "macro_f1_mean", "macro_f1_std"]],
        on=["prefix", "model"],
        how="left",
        suffixes=("_masked", "_original"),
        validate="one_to_one",
    )
    require(merged["macro_f1_mean_original"].notna().all(), "Missing original comparison rows")
    merged["delta_macro_f1"] = (
        merged["macro_f1_mean_masked"] - merged["macro_f1_mean_original"]
    )
    return merged[
        [
            "prefix",
            "model",
            "n_test",
            "macro_f1_mean_original",
            "macro_f1_std_original",
            "macro_f1_mean_masked",
            "macro_f1_std_masked",
            "delta_macro_f1",
            "threshold",
        ]
    ]


def write_report(summary: pd.DataFrame, comparison: pd.DataFrame, audit: dict) -> None:
    table_rows = []
    for row in comparison.itertuples(index=False):
        table_rows.append(
            f"| P{row.prefix} | {row.model} | "
            f"{row.macro_f1_mean_original:.4f} ± {row.macro_f1_std_original:.4f} | "
            f"{row.macro_f1_mean_masked:.4f} ± {row.macro_f1_std_masked:.4f} | "
            f"{row.delta_macro_f1:+.4f} |"
        )
    affected = audit["mask_audit"]["affected_case_counts_by_split"]
    affected_rows = [
        f"| P{prefix} | {affected[prefix]['train']} | {affected[prefix]['val']} | "
        f"{affected[prefix]['test']} | {audit['mask_audit']['affected_case_counts'][prefix]} |"
        for prefix in ALL_AUDIT_PREFIXES
    ]
    text = f"""# ETCE Stage 68A Outcome-Proxy Masking Sensitivity

Status: **PASSED**

## Frozen protocol

- Masked identities: `O_Accepted`, `O_Refused`, `O_Cancelled`.
- Shared neutral token: `{MASK_TOKEN}`.
- Masking occurs after the original Prefix-k endpoint is fixed; event positions,
  timestamps, labels, case IDs, and temporal partitions remain unchanged.
- Prefixes rerun: P10, P15, P20. P5 is unchanged by construction.
- Models: RF, XGBoost, MLP, LSTM.
- Formal seeds: {audit['formal_seeds']}.
- Threshold: {THRESHOLD:.2f}.
- Hyperparameter search: **not performed**.
- TEST-based tuning: **not performed**.

## Affected common-cohort cases

| Prefix | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
{os.linesep.join(affected_rows)}

## Black-box test Macro-F1

| Prefix | Predictor | Original | Masked | Difference |
|---|---|---:|---:|---:|
{os.linesep.join(table_rows)}

The original canonical seed for each model-prefix pair was retained for the
subsequent explanation sensitivity stage. No masked TEST result was used to
select a seed, model, epoch, threshold, or fusion parameter.
"""
    (OUT / "STAGE68A_REPORT.md").write_text(text, encoding="utf-8")


def main() -> int:
    started_wall = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    require(FROZEN_CONFIG.is_file(), f"Missing frozen Phase 2B config: {FROZEN_CONFIG}")
    require(ORIGINAL_CANONICAL.is_file(), f"Missing original canonical manifest: {ORIGINAL_CANONICAL}")
    prepare_output()

    frozen_config = read_json(FROZEN_CONFIG)
    seeds = validate_frozen_config(frozen_config)
    original_canonical = read_json(ORIGINAL_CANONICAL)
    cases, source_audit = load_frozen_common_cohort(ROOT)
    require(len(cases) == EXPECTED_COUNTS["all"], "Unexpected frozen cohort size")
    masked_cases, mask_audit = mask_cases(cases)

    write_json(OUT / "audit" / "masking_audit.json", mask_audit)
    rows = []
    training_log = []
    test_access_log = []
    checkpoint_index = {}

    total_jobs = len(RERUN_PREFIXES) * len(MODEL_ORDER) * len(seeds)
    job_index = 0
    for prefix in RERUN_PREFIXES:
        encoded, encoder = fit_encode_prefix(masked_cases, prefix, OUT / "artifacts")
        train, val, test = encoded["train"], encoded["val"], encoded["test"]
        require(
            (len(train.labels), len(val.labels), len(test.labels))
            == (EXPECTED_COUNTS["train"], EXPECTED_COUNTS["val"], EXPECTED_COUNTS["test"]),
            f"Encoded split counts changed at P{prefix}",
        )
        require(MASK_TOKEN in encoder["vocabulary"], f"Mask token absent from P{prefix} train vocabulary")

        for model_name in MODEL_ORDER:
            frozen_entry = frozen_config["models"][model_name][str(prefix)]
            for seed in seeds:
                job_index += 1
                label = f"[{job_index:02d}/{total_jobs}] {model_name} P{prefix} seed={seed}"
                print(f"{label} | training", flush=True)
                one_started = time.perf_counter()
                (
                    model,
                    config,
                    config_id,
                    training_metadata,
                    val_probability,
                    test_probability,
                ) = train_one(
                    model_name, prefix, seed, train, val, test, encoder, frozen_entry
                )
                checkpoint = save_checkpoint(
                    model_name, prefix, seed, model, config, training_metadata, encoder
                )
                checkpoint_index[(model_name, prefix, seed)] = checkpoint

                for split, data, probability in (
                    ("val", val, val_probability),
                    ("test", test, test_probability),
                ):
                    metrics = metric_values(data.labels, probability)
                    pred_dir = OUT / "predictions" / f"seed_{seed}"
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    pred_path = pred_dir / f"{MODEL_STEMS[model_name]}_prefix{prefix}_{split}.csv"
                    prediction_frame(
                        model_name,
                        prefix,
                        split,
                        seed,
                        data,
                        probability,
                        config_id,
                    ).to_csv(pred_path, index=False, encoding="utf-8-sig", float_format="%.12f")
                    if split == "test":
                        rows.append(
                            {
                                "seed": seed,
                                "prefix": prefix,
                                "model": model_name,
                                "n_test": len(data.labels),
                                "threshold": THRESHOLD,
                                **metrics,
                                "config_id": config_id,
                                "checkpoint": relative(checkpoint),
                                "prediction_file": relative(pred_path),
                            }
                        )
                        test_access_log.append(
                            {
                                "seed": seed,
                                "model": model_name,
                                "prefix": prefix,
                                "access_count": 1,
                                "purpose": (
                                    "single frozen outcome-proxy masking sensitivity evaluation"
                                ),
                                "test_used_for_tuning": False,
                            }
                        )
                    else:
                        val_metrics = metrics

                elapsed = time.perf_counter() - one_started
                training_log.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "prefix": prefix,
                        "original_config_id": frozen_entry["config_id"],
                        "masked_config_id": config_id,
                        "hyperparameters": config,
                        "validation_metrics": val_metrics,
                        "training_metadata": training_metadata,
                        "training_seconds": elapsed,
                        "test_evaluations": 1,
                    }
                )
                print(
                    f"{label} | val Macro-F1={val_metrics['macro_f1']:.6f} | "
                    f"test Macro-F1={rows[-1]['macro_f1']:.6f} | {elapsed:.1f}s",
                    flush=True,
                )

    require(len(rows) == 60, f"Expected 60 masked TEST metric rows, found {len(rows)}")
    seedwise = pd.DataFrame(rows)
    summary = summarize(seedwise)
    require(len(summary) == 12, f"Expected 12 summary rows, found {len(summary)}")
    comparison = build_comparison(summary)

    seedwise.to_csv(
        OUT / "metrics" / "masked_test_metrics_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10f",
    )
    summary.to_csv(
        OUT / "metrics" / "masked_test_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10f",
    )
    comparison.to_csv(
        OUT / "metrics" / "masked_vs_original_macro_f1.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10f",
    )
    write_json(OUT / "logs" / "training_log.json", training_log)
    write_json(OUT / "logs" / "test_access_log.json", test_access_log)

    canonical_items = {model: {} for model in MODEL_ORDER}
    for model in MODEL_ORDER:
        for prefix in RERUN_PREFIXES:
            original_entry = original_canonical["items"][model][str(prefix)]
            canonical_seed = int(original_entry["selected_seed"])
            source = checkpoint_index[(model, prefix, canonical_seed)]
            suffix = source.suffix
            destination = (
                OUT
                / "canonical_blackboxes"
                / "checkpoints"
                / f"{MODEL_STEMS[model]}_prefix{prefix}_seed{canonical_seed}{suffix}"
            )
            shutil.copy2(source, destination)
            canonical_items[model][str(prefix)] = {
                "selected_seed": canonical_seed,
                "selection_rule": "same canonical seed as the original unmasked experiment",
                "source_checkpoint": relative(source),
                "canonical_checkpoint": relative(destination),
                "checkpoint_sha256": sha256_file(destination),
                "encoder_metadata": relative(OUT / "artifacts" / f"encoder_prefix{prefix}.json"),
                "tabular_scaler": relative(OUT / "artifacts" / f"tabular_scaler_prefix{prefix}.joblib"),
                "sequence_time_scaler": relative(OUT / "artifacts" / f"sequence_time_scaler_prefix{prefix}.joblib"),
                "hyperparameters": frozen_config["models"][model][str(prefix)]["hyperparameters"],
                "phase2a_config_id": frozen_config["models"][model][str(prefix)]["config_id"],
                "mask_token": MASK_TOKEN,
            }

    masked_canonical = {
        "stage": STAGE,
        "purpose": "outcome-proxy masking sensitivity",
        "models": list(MODEL_ORDER),
        "prefixes": list(RERUN_PREFIXES),
        "formal_seeds": seeds,
        "canonical_seed_policy": (
            "retain the original canonical seed for every model-prefix pair; "
            "masked validation/test performance cannot change seed selection"
        ),
        "test_used_for_selection": False,
        "mask_protocol": {
            "masked_activities": sorted(MASKED_ACTIVITIES),
            "mask_token": MASK_TOKEN,
            "prefix_endpoint_fixed_before_masking": True,
            "event_positions_preserved": True,
            "timestamps_preserved": True,
        },
        "items": canonical_items,
    }
    write_json(
        OUT / "canonical_blackboxes" / "canonical_manifest_masked.json",
        masked_canonical,
    )

    audit = {
        "stage": STAGE,
        "status": "passed",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started_wall,
        "runtime_seconds": time.perf_counter() - started,
        "project_root": str(ROOT),
        "output_dir": str(OUT),
        "python": platform.python_version(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "source_cohort_hashes": COHORT_HASHES,
        "source_data_audit": source_audit,
        "frozen_config": relative(FROZEN_CONFIG),
        "frozen_config_sha256": sha256_file(FROZEN_CONFIG),
        "original_canonical_manifest": relative(ORIGINAL_CANONICAL),
        "original_canonical_manifest_sha256": sha256_file(ORIGINAL_CANONICAL),
        "formal_seeds": seeds,
        "models": list(MODEL_ORDER),
        "rerun_prefixes": list(RERUN_PREFIXES),
        "threshold": THRESHOLD,
        "mask_audit": mask_audit,
        "checks": {
            "same_case_ids": True,
            "same_temporal_splits": True,
            "same_labels": True,
            "same_prefix_positions": True,
            "same_timestamps": True,
            "three_outcome_identities_removed": True,
            "shared_neutral_token_used": True,
            "p5_unchanged_by_construction": True,
            "frozen_hyperparameters_used": True,
            "all_five_formal_seeds_used": True,
            "threshold_remained_0_50": True,
            "hyperparameter_search_performed": False,
            "threshold_tuning_performed": False,
            "test_based_selection_performed": False,
            "original_canonical_seeds_retained": True,
            "all_60_test_evaluations_completed_once": True,
        },
    }
    write_json(OUT / "audit" / "validation_stage68a_masked_blackbox.json", audit)
    write_report(summary, comparison, audit)

    print("=" * 108)
    print("ETCE STAGE 68A - OUTCOME-PROXY MASKED BLACK-BOX SENSITIVITY")
    print("=" * 108)
    print(f"Masked activities        : {sorted(MASKED_ACTIVITIES)}")
    print(f"Neutral token            : {MASK_TOKEN}")
    print(f"Rerun prefixes           : {list(RERUN_PREFIXES)}")
    print(f"Formal seeds             : {seeds}")
    print(f"Completed TEST fits      : {len(rows)}")
    print("Hyperparameter tuning    : False")
    print("Threshold tuning         : False")
    print("TEST-based selection     : False")
    print("\nOriginal vs masked Macro-F1:\n")
    printable = comparison.copy()
    for column in (
        "macro_f1_mean_original",
        "macro_f1_std_original",
        "macro_f1_mean_masked",
        "macro_f1_std_masked",
        "delta_macro_f1",
    ):
        printable[column] = printable[column].map(lambda value: f"{value:.6f}")
    print(printable.to_string(index=False))
    print(f"\nOutputs: {OUT}")
    print("STAGE 68A PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"STAGE 68A FAILED: {exc}", file=sys.stderr)
        raise
