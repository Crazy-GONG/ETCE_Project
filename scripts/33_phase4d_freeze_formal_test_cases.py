#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-0: freeze formal test explanation cases.

Current rebuilt protocol:
- frozen common cohort from Phase 1.2;
- formal prefixes = 5 / 10 / 15 / 20;
- black-boxes = RF / XGBoost / MLP / LSTM;
- D-S protocol must already be frozen by Phase 4C-v2.

Selection rule
--------------
Select exactly 100 TEST cases by deterministic SHA-256 ranking of case_id.

The selection does NOT use:
- true label,
- black-box prediction,
- probability/confidence,
- explanation success,
- surrogate output.

This script only freezes target case IDs. It does not generate
neighborhoods, rules, or D-S results.

Run:
    python -u scripts/33_phase4d_freeze_formal_test_cases.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etce.phase2.data import PREFIXES, load_frozen_common_cohort

FROZEN_DS = ROOT / "configs/phase4c_ds_frozen_config.json"
OUT = ROOT / "results/phase4/phase4d_formal_test"
SELECTION_JSON = OUT / "formal_test_case_selection.json"
SELECTION_CSV = OUT / "formal_test_case_selection.csv"

MODELS = ("RF", "XGBoost", "MLP", "LSTM")
TEST_CASE_COUNT = 100


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def stable_rank(case_id):
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "selection_rank",
                "case_id",
                "case_id_sha256",
                "split",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def main():
    require(FROZEN_DS.is_file(), f"Missing frozen D-S config: {FROZEN_DS}")

    ds = read_json(FROZEN_DS)
    require(ds.get("status") == "frozen", "D-S protocol is not frozen")
    require(float(ds.get("lambda")) == 5.0, "Frozen lambda is not 5")
    require(
        ds.get("selection_scope") == "validation_only",
        "D-S selection scope is not validation_only",
    )
    require(
        ds.get("formal_test_seen_before_freeze") is False,
        "Frozen config indicates formal test was seen before freeze",
    )

    cases, _ = load_frozen_common_cohort(ROOT)
    test_cases = [c for c in cases if c.split == "test"]

    require(
        len(test_cases) == 5276,
        f"Unexpected frozen test cohort size: {len(test_cases)}",
    )
    require(
        tuple(PREFIXES) == (5, 10, 15, 20),
        f"Unexpected formal prefixes: {PREFIXES}",
    )

    ranked = sorted(
        test_cases,
        key=lambda c: (stable_rank(c.case_id), c.case_id),
    )
    selected = ranked[:TEST_CASE_COUNT]

    require(len(selected) == TEST_CASE_COUNT, "Selection size mismatch")
    require(
        len({c.case_id for c in selected}) == TEST_CASE_COUNT,
        "Duplicate selected case_id",
    )
    require(
        all(c.split == "test" for c in selected),
        "Non-test case selected",
    )

    rows = [
        {
            "selection_rank": i + 1,
            "case_id": c.case_id,
            "case_id_sha256": stable_rank(c.case_id),
            "split": c.split,
        }
        for i, c in enumerate(selected)
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(SELECTION_CSV, rows)

    payload = {
        "schema_version": "etce-phase4d-formal-test-selection-v1",
        "status": "frozen",
        "scope": "formal_test_explanation_case_selection",
        "source_split": "test",
        "source_test_case_count": len(test_cases),
        "selected_case_count": TEST_CASE_COUNT,
        "models": list(MODELS),
        "prefixes": [int(x) for x in PREFIXES],
        "formal_target_count": (
            TEST_CASE_COUNT * len(MODELS) * len(PREFIXES)
        ),
        "selection_rule": (
            "sort TEST cases by SHA256(case_id), tie-break by case_id; "
            "take first 100"
        ),
        "selection_uses_true_label": False,
        "selection_uses_blackbox_prediction": False,
        "selection_uses_blackbox_probability": False,
        "selection_uses_explanation_output": False,
        "selection_uses_surrogate_output": False,
        "selected_case_ids": [c.case_id for c in selected],
        "frozen_ds_config": str(FROZEN_DS),
        "frozen_lambda": float(ds["lambda"]),
        "formal_test_results_used_for_selection": False,
        "next_phase": (
            "build formal test multi-rule bank for these 100 frozen cases "
            "using prefixes 5/10/15/20 and the already frozen Phase 4C-v2 protocol"
        ),
    }
    write_json(SELECTION_JSON, payload)

    print("=" * 96)
    print("ETCE PHASE 4D-0 FORMAL TEST CASE SELECTION FROZEN")
    print("=" * 96)
    print(f"Frozen test cohort: {len(test_cases)}")
    print(f"Selected cases: {TEST_CASE_COUNT}")
    print(f"Prefixes: {list(PREFIXES)}")
    print(f"Models: {list(MODELS)}")
    print(
        "Formal targets: "
        f"{TEST_CASE_COUNT} × {len(PREFIXES)} × {len(MODELS)} = "
        f"{payload['formal_target_count']}"
    )
    print("Selection uses labels: False")
    print("Selection uses black-box outputs: False")
    print("Selection uses explanation/surrogate outputs: False")
    print(f"JSON: {SELECTION_JSON}")
    print(f"CSV:  {SELECTION_CSV}")
    print()
    print("NEXT: build the formal test multi-rule bank. Do not change this selection.")


if __name__ == "__main__":
    main()
