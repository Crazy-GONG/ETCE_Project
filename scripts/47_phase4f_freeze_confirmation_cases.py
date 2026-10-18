#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4F-0: freeze a new untouched BPIC2017 confirmation cohort.

Protocol
--------
1. Reconstruct the same deterministic SHA-256 ranking used by Phase 4D.
2. Verify that the original 100-case cohort is exactly ranks 1--100.
3. Exclude those 100 cases and freeze ranks 101--300 as 200 new cases.
4. Verify that the selective Stage-3 gate was already frozen at 0.50.

Selection never uses labels, black-box outputs, rule quality, explanation
success, surrogate outputs, or any Phase-4D/4E performance result.

Run from the project root:
    python -u scripts/47_phase4f_freeze_confirmation_cases.py \
        --project_root "D:\\ETCE"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


MODELS = ("RF", "XGBoost", "MLP", "LSTM")
ORIGINAL_CASE_COUNT = 100
CONFIRMATION_CASE_COUNT = 200
EXPECTED_TEST_CASE_COUNT = 5276
EXPECTED_PREFIXES = (5, 10, 15, 20)
EXPECTED_THRESHOLD = 0.50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_root",
        default=None,
        help="ETCE project root; defaults to the parent of scripts/",
    )
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def stable_rank(case_id):
    return hashlib.sha256(str(case_id).encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "confirmation_rank",
                "global_sha256_rank",
                "case_id",
                "case_id_sha256",
                "split",
                "excluded_from_original_phase4d",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = (
        Path(args.project_root).expanduser().resolve()
        if args.project_root
        else Path(__file__).resolve().parents[1]
    )
    sys.path.insert(0, str(root / "src"))

    from etce.phase2.data import PREFIXES, load_frozen_common_cohort

    frozen_ds_path = root / "configs/phase4c_ds_frozen_config.json"
    original_selection_path = (
        root
        / "results/phase4/phase4d_formal_test/formal_test_case_selection.json"
    )
    adaptive_config_path = (
        root
        / "results/phase4/phase4e_selective_stage3"
        / "selective_stage3_frozen_config.json"
    )
    out = root / "results/phase4/phase4f_confirmation"
    selection_json = out / "confirmation_case_selection.json"
    selection_csv = out / "confirmation_case_selection.csv"

    require(frozen_ds_path.is_file(), f"Missing D-S config: {frozen_ds_path}")
    require(
        original_selection_path.is_file(),
        f"Missing original Phase-4D selection: {original_selection_path}",
    )
    require(
        adaptive_config_path.is_file(),
        f"Missing frozen selective gate: {adaptive_config_path}",
    )

    ds = read_json(frozen_ds_path)
    require(ds.get("status") == "frozen", "D-S protocol is not frozen")
    require(float(ds.get("lambda")) == 5.0, "Frozen lambda is not 5")
    require(
        ds.get("selection_scope") == "validation_only",
        "D-S protocol was not selected on validation only",
    )

    adaptive = read_json(adaptive_config_path)
    require(
        adaptive.get("status") == "validation_frozen_formal_exploratory",
        "Unexpected selective-gate status",
    )
    gate = adaptive.get("gate", {})
    require(
        float(gate.get("pre_mean_fidelity_threshold")) == EXPECTED_THRESHOLD,
        "Selective Stage-3 threshold is not frozen at 0.50",
    )

    original = read_json(original_selection_path)
    original_ids = [str(x) for x in original.get("selected_case_ids", [])]
    require(
        len(original_ids) == ORIGINAL_CASE_COUNT,
        f"Expected 100 original cases, found {len(original_ids)}",
    )
    require(len(set(original_ids)) == ORIGINAL_CASE_COUNT, "Duplicate original case")

    cases, _ = load_frozen_common_cohort(root)
    test_cases = [c for c in cases if c.split == "test"]
    require(
        len(test_cases) == EXPECTED_TEST_CASE_COUNT,
        f"Unexpected frozen test cohort size: {len(test_cases)}",
    )
    require(
        tuple(PREFIXES) == EXPECTED_PREFIXES,
        f"Unexpected formal prefixes: {PREFIXES}",
    )

    ranked = sorted(test_cases, key=lambda c: (stable_rank(c.case_id), c.case_id))
    expected_original_ids = [str(c.case_id) for c in ranked[:ORIGINAL_CASE_COUNT]]
    require(
        original_ids == expected_original_ids,
        "Original Phase-4D cohort is not exactly SHA-256 ranks 1--100",
    )

    start = ORIGINAL_CASE_COUNT
    stop = ORIGINAL_CASE_COUNT + CONFIRMATION_CASE_COUNT
    selected = ranked[start:stop]
    selected_ids = [str(c.case_id) for c in selected]

    require(len(selected) == CONFIRMATION_CASE_COUNT, "Selection size mismatch")
    require(
        len(set(selected_ids)) == CONFIRMATION_CASE_COUNT,
        "Duplicate confirmation case_id",
    )
    require(
        set(selected_ids).isdisjoint(original_ids),
        "Confirmation cohort overlaps original Phase-4D cohort",
    )
    require(all(c.split == "test" for c in selected), "Non-test case selected")

    rows = [
        {
            "confirmation_rank": i + 1,
            "global_sha256_rank": start + i + 1,
            "case_id": c.case_id,
            "case_id_sha256": stable_rank(c.case_id),
            "split": c.split,
            "excluded_from_original_phase4d": True,
        }
        for i, c in enumerate(selected)
    ]

    payload = {
        "schema_version": "etce-phase4f-confirmation-selection-v1",
        "status": "frozen",
        "scope": "selective_stage3_untouched_confirmation_case_selection",
        "source_split": "test",
        "source_test_case_count": len(test_cases),
        "original_phase4d_case_count": ORIGINAL_CASE_COUNT,
        "selected_case_count": CONFIRMATION_CASE_COUNT,
        "models": list(MODELS),
        "prefixes": [int(x) for x in PREFIXES],
        "confirmation_target_count": (
            CONFIRMATION_CASE_COUNT * len(MODELS) * len(PREFIXES)
        ),
        "selection_rule": (
            "reconstruct the Phase-4D ordering by SHA256(case_id), tie-break "
            "by case_id; verify ranks 1--100 equal the frozen Phase-4D "
            "cohort; freeze global ranks 101--300"
        ),
        "global_sha256_rank_start": 101,
        "global_sha256_rank_end": 300,
        "overlap_with_original_phase4d": 0,
        "selection_uses_true_label": False,
        "selection_uses_blackbox_prediction": False,
        "selection_uses_blackbox_probability": False,
        "selection_uses_explanation_output": False,
        "selection_uses_surrogate_output": False,
        "selection_uses_phase4d_or_phase4e_performance": False,
        "selected_case_ids": selected_ids,
        "frozen_lambda": float(ds["lambda"]),
        "selective_stage3_pre_fidelity_threshold": EXPECTED_THRESHOLD,
        "original_selection_path": str(original_selection_path.relative_to(root)),
        "original_selection_sha256": sha256_file(original_selection_path),
        "frozen_ds_config_path": str(frozen_ds_path.relative_to(root)),
        "frozen_ds_config_sha256": sha256_file(frozen_ds_path),
        "adaptive_config_path": str(adaptive_config_path.relative_to(root)),
        "adaptive_config_sha256": sha256_file(adaptive_config_path),
        "confirmatory_analysis_rule": (
            "accept Stage 3 iff the mean held-out fidelity of its Stage-2 "
            "inputs is <= 0.50; otherwise retain Stage 2"
        ),
        "next_phase": (
            "build the unchanged multi-rule evidence bank for these 200 "
            "cases; evaluate the frozen selective rule exactly once"
        ),
    }

    if selection_json.is_file():
        existing = read_json(selection_json)
        require(
            existing == payload,
            "A different Phase-4F selection is already present; refusing overwrite",
        )

    out.mkdir(parents=True, exist_ok=True)
    write_csv(selection_csv, rows)
    write_json(selection_json, payload)

    print("=" * 100)
    print("ETCE PHASE 4F-0 UNTOUCHED CONFIRMATION COHORT FROZEN")
    print("=" * 100)
    print(f"Frozen test cohort: {len(test_cases)}")
    print(f"Original Phase-4D cases excluded: {len(original_ids)}")
    print(f"Selected new cases: {len(selected)}")
    print("Global deterministic ranks: 101--300")
    print("Overlap with original Phase 4D: 0")
    print(f"Prefixes: {list(PREFIXES)}")
    print(f"Models: {list(MODELS)}")
    print(
        "Confirmation targets: "
        f"{CONFIRMATION_CASE_COUNT} × {len(PREFIXES)} × {len(MODELS)} = "
        f"{payload['confirmation_target_count']}"
    )
    print("Selective Stage-3 threshold: 0.50 (already frozen)")
    print("Selection uses labels or model/explanation outputs: False")
    print(f"JSON: {selection_json}")
    print(f"CSV:  {selection_csv}")
    print()
    print("NEXT: build the confirmation multi-rule bank. Do not change this selection.")


if __name__ == "__main__":
    main()
