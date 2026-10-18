#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage60
BPIC2018 External Canonical Black-box Binding

Purpose
-------
Bind the 1600 Stage59 frozen explanation targets to the canonical
black-box instance under the already-established ETCE rule:

    canonical black-box seed = 42

This rule comes from the frozen BPIC2017 main protocol. Stage60 does NOT
select a seed based on BPIC2018 TEST performance.

Inputs
------
Stage58:
  - final_test_predictions.csv.gz
  - final_test_metrics_seedwise.csv
  - checkpoints/*
  - protocol_frozen.json
  - consistency_checks.json
Stage59:
  - formal_targets.csv
  - selected_cases.csv
  - protocol_frozen.json
  - consistency_checks.json
  - dataset_summary.json

Outputs
-------
canonical_blackbox_bindings.csv
canonical_checkpoint_map.csv
canonical_binding_summary.json
consistency_checks.json
protocol_frozen.json
sha256_manifest.json
STAGE60_CANONICAL_BLACKBOX_BINDING.md

Strict rules
------------
- canonical seed is hard-coded to 42 BEFORE reading any Stage58 performance.
- no model/prefix-specific seed selection.
- no TEST-metric-based model selection.
- no threshold tuning.
- no target reselection.
- no surrogate / counterfactual / D-S processing yet.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

STAGE = 60
CANONICAL_SEED = 42
THRESHOLD = 0.5
PREFIXES = (5, 10, 15, 20)
MODELS = ("RF", "XGB", "MLP", "LSTM")
EXPECTED_TARGETS = 1600
EXPECTED_CASES = 100


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        items = list(path.iterdir())
        if items and not overwrite:
            raise FileExistsError(
                f"Output directory already contains files: {path}\n"
                "Use --overwrite only for an exact frozen-protocol rebuild."
            )
        if overwrite:
            import shutil
            for p in items:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
    path.mkdir(parents=True, exist_ok=True)


def verify_stage58(stage58_dir: Path) -> Dict[str, Any]:
    required = (
        "protocol_frozen.json",
        "consistency_checks.json",
        "final_test_predictions.csv.gz",
        "final_test_metrics_seedwise.csv",
        "sha256_manifest.json",
    )
    for name in required:
        if not (stage58_dir / name).exists():
            raise FileNotFoundError(stage58_dir / name)

    protocol = read_json(stage58_dir / "protocol_frozen.json")
    checks = read_json(stage58_dir / "consistency_checks.json")

    if protocol.get("stage") != 58:
        raise RuntimeError("Stage60 expects Stage58 formal external TEST.")
    if not protocol.get("all_hard_checks_pass", False):
        raise RuntimeError("Stage58 is not frozen/PASS.")
    if protocol.get("test_based_tuning", True):
        raise RuntimeError("Stage58 reports TEST-based tuning; protocol invalid.")
    if any(not v.get("pass", False) for v in checks.values()):
        raise RuntimeError("Stage58 consistency checks are not fully PASS.")

    return {"protocol": protocol, "checks": checks}


def verify_stage59(stage59_dir: Path) -> Dict[str, Any]:
    required = (
        "formal_targets.csv",
        "selected_cases.csv",
        "protocol_frozen.json",
        "consistency_checks.json",
        "dataset_summary.json",
        "sha256_manifest.json",
    )
    for name in required:
        if not (stage59_dir / name).exists():
            raise FileNotFoundError(stage59_dir / name)

    protocol = read_json(stage59_dir / "protocol_frozen.json")
    checks = read_json(stage59_dir / "consistency_checks.json")
    summary = read_json(stage59_dir / "dataset_summary.json")

    if protocol.get("stage") != 59:
        raise RuntimeError("Stage60 expects Stage59 formal target freeze.")
    if not protocol.get("all_hard_checks_pass", False):
        raise RuntimeError("Stage59 protocol is not PASS.")
    if not summary.get("all_hard_checks_pass", False):
        raise RuntimeError("Stage59 summary is not PASS.")
    if any(not v.get("pass", False) for v in checks.values()):
        raise RuntimeError("Stage59 consistency checks are not fully PASS.")

    if int(summary.get("selected_cases", -1)) != EXPECTED_CASES:
        raise RuntimeError("Stage59 selected case count changed.")
    if int(summary.get("formal_targets", -1)) != EXPECTED_TARGETS:
        raise RuntimeError("Stage59 formal target count changed.")

    return {"protocol": protocol, "checks": checks, "summary": summary}


def load_stage59_targets(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXPECTED_TARGETS:
        raise RuntimeError(
            f"Expected {EXPECTED_TARGETS} Stage59 targets; got {len(rows)}"
        )
    return rows


def load_seed42_checkpoints(metrics_path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Resolve exactly one canonical checkpoint for each (model, prefix) at seed=42.
    No metric comparison is performed.
    """
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}

    with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["seed"]) != CANONICAL_SEED:
                continue

            model = row["model"].strip()
            prefix = int(row["prefix"])
            key = (model, prefix)

            if key in out:
                raise RuntimeError(f"Duplicate Stage58 seed42 metric row for {key}")

            checkpoint = Path(row["checkpoint"])
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Canonical checkpoint does not exist: {checkpoint}"
                )

            actual_hash = sha256_file(checkpoint)
            recorded_hash = row["checkpoint_sha256"].strip()
            if actual_hash != recorded_hash:
                raise RuntimeError(
                    f"Checkpoint SHA256 mismatch for {model} P{prefix} seed42"
                )

            out[key] = {
                "model": model,
                "prefix": prefix,
                "seed": CANONICAL_SEED,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": actual_hash,
            }

    expected_keys = {(m, p) for m in MODELS for p in PREFIXES}
    if set(out) != expected_keys:
        missing = sorted(expected_keys - set(out))
        extra = sorted(set(out) - expected_keys)
        raise RuntimeError(
            f"Canonical seed42 checkpoint map mismatch. missing={missing}, extra={extra}"
        )

    return out


def load_seed42_predictions(
    predictions_path: Path,
    selected_case_ids: set,
) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    """
    Load only seed=42 rows for the already-frozen 100 selected cases.
    Prediction values are used only AFTER the seed and target cohort are frozen.
    """
    out: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

    with gzip.open(predictions_path, "rt", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)

        # Stage58 uses either `year` or `case_year` depending on version.
        for row in r:
            if int(row["seed"]) != CANONICAL_SEED:
                continue

            case_id = str(row["case_id"])
            if case_id not in selected_case_ids:
                continue

            prefix = int(row["prefix"])
            model = row["model"].strip()
            key = (case_id, prefix, model)

            if key in out:
                raise RuntimeError(f"Duplicate canonical prediction row: {key}")

            prob = float(row["prob_positive"])
            pred_col = "pred_0_5" if "pred_0_5" in row else "pred_threshold_0_5"
            pred = int(row[pred_col])

            expected_pred = int(prob >= THRESHOLD)
            if pred != expected_pred:
                raise RuntimeError(
                    f"Threshold reproduction mismatch for {key}: "
                    f"prob={prob}, stored_pred={pred}, expected={expected_pred}"
                )

            year_col = "year" if "year" in row else "case_year"
            year = int(row[year_col])
            if year != 2017:
                raise RuntimeError(f"Canonical prediction is not 2017 for {key}")

            out[key] = {
                "case_id": case_id,
                "prefix": prefix,
                "model": model,
                "seed": CANONICAL_SEED,
                "prob_positive": prob,
                "predicted_class": pred,
                "year": year,
            }

    expected = len(selected_case_ids) * len(PREFIXES) * len(MODELS)
    if len(out) != expected:
        raise RuntimeError(
            f"Expected {expected} canonical predictions for selected targets; got {len(out)}"
        )

    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ETCE Stage60 - BPIC2018 External Canonical Black-box Binding"
    )
    ap.add_argument("--project_root", type=Path, default=Path(r"D:\ETCE"))
    ap.add_argument("--stage58_dir", type=Path, default=None)
    ap.add_argument("--stage59_dir", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.project_root.resolve()

    stage58_dir = (
        args.stage58_dir.resolve()
        if args.stage58_dir
        else root / "results" / "phase4" / "phase4e_bpic2018_external" / "stage58_formal_external_test"
    )
    stage59_dir = (
        args.stage59_dir.resolve()
        if args.stage59_dir
        else root / "results" / "phase4" / "phase4e_bpic2018_external" / "stage59_formal_targets"
    )
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else root / "results" / "phase4" / "phase4e_bpic2018_external" / "stage60_canonical_blackbox"
    )

    ensure_clean_dir(out_dir, args.overwrite)

    print("=" * 120)
    print("ETCE STAGE60 - BPIC2018 EXTERNAL CANONICAL BLACK-BOX BINDING")
    print("=" * 120)
    print(f"Stage58        : {stage58_dir}")
    print(f"Stage59        : {stage59_dir}")
    print(f"Output         : {out_dir}")
    print(f"Canonical seed : {CANONICAL_SEED}")
    print(f"Threshold      : {THRESHOLD}")
    print("Seed selection : FROZEN FROM BPIC2017 MAIN PROTOCOL")
    print("TEST metric use: NONE FOR SEED SELECTION")
    print("=" * 120)

    s58 = verify_stage58(stage58_dir)
    print("[Stage58 gate] PASS")

    s59 = verify_stage59(stage59_dir)
    print("[Stage59 gate] PASS")

    targets = load_stage59_targets(stage59_dir / "formal_targets.csv")
    selected_ids = {str(r["case_id"]) for r in targets}

    if len(selected_ids) != EXPECTED_CASES:
        raise RuntimeError(
            f"Stage59 target file should contain {EXPECTED_CASES} unique cases; "
            f"got {len(selected_ids)}"
        )

    checkpoint_map = load_seed42_checkpoints(
        stage58_dir / "final_test_metrics_seedwise.csv"
    )
    print("[Canonical seed42 checkpoints] PASS - 16/16 resolved")

    prediction_map = load_seed42_predictions(
        stage58_dir / "final_test_predictions.csv.gz",
        selected_ids,
    )
    print("[Canonical seed42 target predictions] PASS - 1600/1600 resolved")

    # ------------------------------------------------------------------
    # Write canonical checkpoint map.
    # ------------------------------------------------------------------
    checkpoint_map_path = out_dir / "canonical_checkpoint_map.csv"
    with checkpoint_map_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "prefix",
                "canonical_seed",
                "checkpoint",
                "checkpoint_sha256",
            ]
        )
        for model in MODELS:
            for prefix in PREFIXES:
                x = checkpoint_map[(model, prefix)]
                w.writerow(
                    [
                        model,
                        prefix,
                        CANONICAL_SEED,
                        x["checkpoint"],
                        x["checkpoint_sha256"],
                    ]
                )

    # ------------------------------------------------------------------
    # Bind all 1600 Stage59 targets.
    # ------------------------------------------------------------------
    binding_path = out_dir / "canonical_blackbox_bindings.csv"
    bound_rows: List[Dict[str, Any]] = []

    with binding_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "target_id",
            "case_rank",
            "case_id",
            "case_year",
            "prefix",
            "model",
            "canonical_seed",
            "checkpoint",
            "checkpoint_sha256",
            "prob_positive",
            "predicted_class",
            "threshold",
            "binding_rule",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for t in targets:
            case_id = str(t["case_id"])
            prefix = int(t["prefix"])
            model = t["model"].strip()

            pkey = (case_id, prefix, model)
            ckey = (model, prefix)

            if pkey not in prediction_map:
                raise RuntimeError(f"Missing prediction for Stage59 target {pkey}")
            if ckey not in checkpoint_map:
                raise RuntimeError(f"Missing checkpoint for Stage59 target {ckey}")

            pred = prediction_map[pkey]
            ck = checkpoint_map[ckey]

            row = {
                "target_id": t["target_id"],
                "case_rank": int(t["case_rank"]),
                "case_id": case_id,
                "case_year": int(t["case_year"]),
                "prefix": prefix,
                "model": model,
                "canonical_seed": CANONICAL_SEED,
                "checkpoint": ck["checkpoint"],
                "checkpoint_sha256": ck["checkpoint_sha256"],
                "prob_positive": f"{pred['prob_positive']:.12g}",
                "predicted_class": pred["predicted_class"],
                "threshold": THRESHOLD,
                "binding_rule": "BPIC2017_CANONICAL_SEED_42",
            }
            w.writerow(row)
            bound_rows.append(row)

    # ------------------------------------------------------------------
    # Audits.
    # ------------------------------------------------------------------
    model_prefix_counts = Counter(
        (r["model"], int(r["prefix"])) for r in bound_rows
    )
    model_counts = Counter(r["model"] for r in bound_rows)
    prefix_counts = Counter(int(r["prefix"]) for r in bound_rows)
    predicted_class_counts = Counter(int(r["predicted_class"]) for r in bound_rows)

    checks = {
        "stage58_pass": {"pass": True},
        "stage59_pass": {"pass": True},
        "canonical_seed_exactly_42": {
            "pass": CANONICAL_SEED == 42,
            "actual": CANONICAL_SEED,
            "expected": 42,
        },
        "canonical_seed_rule_source": {
            "pass": True,
            "actual": "BPIC2017_MAIN_PROTOCOL",
            "expected": "BPIC2017_MAIN_PROTOCOL",
        },
        "no_test_metric_based_seed_selection": {
            "pass": True,
            "actual": False,
            "expected": False,
        },
        "checkpoint_count": {
            "pass": len(checkpoint_map) == 16,
            "actual": len(checkpoint_map),
            "expected": 16,
        },
        "formal_binding_count": {
            "pass": len(bound_rows) == EXPECTED_TARGETS,
            "actual": len(bound_rows),
            "expected": EXPECTED_TARGETS,
        },
        "unique_target_ids": {
            "pass": len({r["target_id"] for r in bound_rows}) == EXPECTED_TARGETS,
        },
        "unique_case_prefix_model_keys": {
            "pass": len(
                {
                    (r["case_id"], int(r["prefix"]), r["model"])
                    for r in bound_rows
                }
            )
            == EXPECTED_TARGETS,
        },
        "all_100_cases_preserved": {
            "pass": len({r["case_id"] for r in bound_rows}) == EXPECTED_CASES,
            "actual": len({r["case_id"] for r in bound_rows}),
            "expected": EXPECTED_CASES,
        },
        "each_model_has_400_targets": {
            "pass": all(model_counts[m] == 400 for m in MODELS),
            "actual": dict(model_counts),
            "expected_each": 400,
        },
        "each_prefix_has_400_targets": {
            "pass": all(prefix_counts[p] == 400 for p in PREFIXES),
            "actual": dict(prefix_counts),
            "expected_each": 400,
        },
        "each_model_prefix_has_100_targets": {
            "pass": all(
                model_prefix_counts[(m, p)] == 100
                for m in MODELS
                for p in PREFIXES
            ),
            "actual": {
                f"{m}_P{p}": model_prefix_counts[(m, p)]
                for m in MODELS
                for p in PREFIXES
            },
            "expected_each": 100,
        },
        "all_thresholds_0_5": {
            "pass": all(float(r["threshold"]) == THRESHOLD for r in bound_rows),
        },
        "all_case_year_2017": {
            "pass": all(int(r["case_year"]) == 2017 for r in bound_rows),
        },
    }

    all_pass = all(v.get("pass", False) for v in checks.values())
    write_json(out_dir / "consistency_checks.json", checks)

    # ------------------------------------------------------------------
    # Freeze protocol and summary.
    # ------------------------------------------------------------------
    stage59_set_hash = s59["summary"].get("selected_case_set_sha256")

    protocol = {
        "stage": STAGE,
        "name": "BPIC2018 External Canonical Black-box Binding",
        "dataset": "BPIC2018",
        "created_local": datetime.now().astimezone().isoformat(),
        "formal_test_year": 2017,
        "selected_cases": EXPECTED_CASES,
        "formal_targets": EXPECTED_TARGETS,
        "prefixes": list(PREFIXES),
        "models": list(MODELS),
        "canonical_seed": CANONICAL_SEED,
        "canonical_seed_source": "BPIC2017 frozen main ETCE protocol",
        "prediction_threshold": THRESHOLD,
        "seed_selection_uses_bpic2018_test_metrics": False,
        "case_selection_source_stage": 59,
        "selected_case_set_sha256": stage59_set_hash,
        "surrogate_generated": False,
        "counterfactual_generated": False,
        "ds_fusion_performed": False,
        "all_hard_checks_pass": all_pass,
    }
    write_json(out_dir / "protocol_frozen.json", protocol)

    summary = {
        "stage": STAGE,
        "dataset": "BPIC2018",
        "canonical_seed": CANONICAL_SEED,
        "selected_cases": EXPECTED_CASES,
        "formal_targets": len(bound_rows),
        "checkpoint_bindings": len(checkpoint_map),
        "model_counts": dict(model_counts),
        "prefix_counts": {str(k): v for k, v in prefix_counts.items()},
        "predicted_class_counts_on_frozen_targets": {
            str(k): v for k, v in sorted(predicted_class_counts.items())
        },
        "all_hard_checks_pass": all_pass,
    }
    write_json(out_dir / "canonical_binding_summary.json", summary)

    md = f"""# ETCE Stage60 — BPIC2018 Canonical Black-box Binding

## Frozen canonical rule
Canonical black-box seed: **42**

This seed is inherited from the frozen BPIC2017 main ETCE protocol.
It is not selected from BPIC2018 formal TEST metrics.

## Binding
- Cases: **{EXPECTED_CASES}**
- Prefixes: **P5 / P10 / P15 / P20**
- Models: **RF / XGB / MLP / LSTM**
- Canonical checkpoints: **16**
- Formal target bindings: **{len(bound_rows)}**
- Threshold: **{THRESHOLD}**

Stage59 case-set SHA256:
`{stage59_set_hash}`

Predicted-class counts on the already-frozen targets:
`{dict(sorted(predicted_class_counts.items()))}`

These prediction counts are diagnostic only and did not affect target or seed selection.

Gate: **{"PASS" if all_pass else "FAIL"}**
"""
    (out_dir / "STAGE60_CANONICAL_BLACKBOX_BINDING.md").write_text(
        md, encoding="utf-8"
    )

    files = sorted(
        p for p in out_dir.iterdir()
        if p.is_file() and p.name != "sha256_manifest.json"
    )
    write_json(
        out_dir / "sha256_manifest.json",
        {
            "stage": STAGE,
            "algorithm": "SHA256",
            "generated_local": datetime.now().astimezone().isoformat(),
            "files": {
                str(p): {
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
                for p in files
            },
        },
    )

    print("\n" + "=" * 120)
    print("ETCE STAGE60 SUMMARY")
    print("=" * 120)
    print(f"Canonical seed             : {CANONICAL_SEED}")
    print(f"Canonical checkpoints      : {len(checkpoint_map)}")
    print(f"Selected cases preserved   : {len(selected_ids)}")
    print(f"Formal target bindings     : {len(bound_rows)}")
    print(f"Predicted class counts     : {dict(sorted(predicted_class_counts.items()))}")
    print("TEST metric seed selection : NO")
    print(f"Hard gates                 : {'PASS' if all_pass else 'FAIL'}")
    print("=" * 120)

    if not all_pass:
        print("STAGE60 FAILED. Do not proceed to surrogate/rule generation.")
        return 2

    print(
        "\nSTAGE60 CANONICAL BLACK-BOX BINDING FROZEN SUCCESSFULLY.\n"
        "NEXT: Stage61 may build the external local neighborhoods / surrogate rule bank "
        "for these exact 1600 canonical targets, without retuning on BPIC2018 TEST."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
