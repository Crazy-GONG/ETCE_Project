#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4C-v2: freeze validation-selected D-S protocol.

This script freezes:
- lambda = 5
- support-aware reliability / BPA
- Stage-1 / Stage-2 / Stage-3 grouping
- deterministic fusion order
- K -> 1 handling
- representative-rule tie breaking
- fused-rule re-evaluation policy

The selection is validation-only. Formal test results must not be read
before this frozen config is written.

Run:
    python -u scripts/32_phase4c_v2_freeze_ds_protocol.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SWEEP_DIR = ROOT / "results/phase4/phase4c_v2_hierarchical_ds_sweep"
SUMMARY_CSV = SWEEP_DIR / "phase4c_v2_ds_lambda_summary.csv"
SWEEP_JSON = SWEEP_DIR / "validation_ds_lambda_sweep.json"

PHASE4B_VALIDATION = (
    ROOT
    / "results/phase4/phase4b_v2_multirule_bank/validation_phase4b_v2.json"
)
GROUP_AUDIT = (
    ROOT
    / "results/phase4/phase4c_v2_target_local_audit/target_local_group_audit_v2.json"
)
CACHE_VALIDATION = (
    ROOT
    / "results/phase4/phase4c_v2_fusion_eval_cache/validation_fusion_eval_cache.json"
)

CONFIG_DIR = ROOT / "configs"
OUT_CONFIG = CONFIG_DIR / "phase4c_ds_frozen_config.json"
OUT_REPORT = SWEEP_DIR / "PHASE4C_V2_FROZEN_PROTOCOL.md"

SELECTED_LAMBDA = 5.0
EXPECTED_LAMBDAS = [1.0, 5.0, 10.0, 20.0, 50.0]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_summary(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["lambda"] = float(r["lambda"])
        for k in (
            "mean_conflict_K",
            "median_conflict_K",
            "p90_conflict_K",
            "max_conflict_K",
            "fused_mean_fidelity",
            "fused_cf_validity_rate",
            "fused_mean_model_cf_validity_rate",
            "fused_all_models_cf_valid_rate",
            "fused_mean_reliability",
            "fused_mean_ignorance",
        ):
            r[k] = float(r[k])
        r["total_conflict_fallback_count"] = int(
            r["total_conflict_fallback_count"]
        )
    return rows


def index(rows):
    return {(r["stage"], r["lambda"]): r for r in rows}


def rel_reduction(old, new):
    return (old - new) / old if old != 0 else 0.0


def main():
    required = [
        SUMMARY_CSV,
        SWEEP_JSON,
        PHASE4B_VALIDATION,
        GROUP_AUDIT,
        CACHE_VALIDATION,
    ]
    missing = [str(p) for p in required if not p.is_file()]
    require(not missing, "Missing required inputs:\n" + "\n".join(missing))

    phase4b = read_json(PHASE4B_VALIDATION)
    audit = read_json(GROUP_AUDIT)
    cache = read_json(CACHE_VALIDATION)
    sweep = read_json(SWEEP_JSON)
    rows = read_summary(SUMMARY_CSV)
    idx = index(rows)

    require(phase4b.get("status") == "passed", "Phase 4B-v2 not PASSED")
    require(
        phase4b.get("test_cases_used") is False,
        "Phase 4B-v2 used test data",
    )
    require(
        audit.get("result", {}).get(
            "former_single_rule_degeneracy_resolved"
        ) is True,
        "Stage-1 non-degeneracy not established",
    )
    require(
        audit.get("test_data_used") is False,
        "Grouping audit used test data",
    )
    require(cache.get("status") == "passed", "Fusion cache not PASSED")
    require(
        cache.get("test_cases_used") is False,
        "Fusion cache used test data",
    )
    require(sweep.get("status") == "passed", "D-S sweep not PASSED")
    require(
        sweep.get("test_cases_used") is False,
        "D-S sweep used formal test data",
    )
    require(
        sweep.get("lambda_selected") is None,
        "Sweep unexpectedly selected lambda before freeze",
    )

    observed_lambdas = sorted(set(r["lambda"] for r in rows))
    require(
        observed_lambdas == EXPECTED_LAMBDAS,
        f"Unexpected lambda grid: {observed_lambdas}",
    )
    require(
        len(rows) == 15,
        f"Expected 15 Stage×lambda rows, found {len(rows)}",
    )

    for stage in ("Stage1", "Stage2", "Stage3"):
        for lam in EXPECTED_LAMBDAS:
            require(
                (stage, lam) in idx,
                f"Missing summary row: {stage}, lambda={lam}",
            )

    fallback_total = sum(
        r["total_conflict_fallback_count"] for r in rows
    )
    require(
        fallback_total == 0,
        "Validation sweep contains total-conflict fallbacks; "
        "review before freezing protocol",
    )

    s1_1 = idx[("Stage1", 1.0)]
    s1_5 = idx[("Stage1", 5.0)]
    s2_1 = idx[("Stage2", 1.0)]
    s2_5 = idx[("Stage2", 5.0)]
    s3_1 = idx[("Stage3", 1.0)]
    s3_5 = idx[("Stage3", 5.0)]
    s3_10 = idx[("Stage3", 10.0)]

    diagnostics = {
        "lambda_1_to_5": {
            "stage1_mean_conflict_relative_reduction": rel_reduction(
                s1_1["mean_conflict_K"], s1_5["mean_conflict_K"]
            ),
            "stage2_mean_conflict_relative_reduction": rel_reduction(
                s2_1["mean_conflict_K"], s2_5["mean_conflict_K"]
            ),
            "stage3_mean_conflict_relative_reduction": rel_reduction(
                s3_1["mean_conflict_K"], s3_5["mean_conflict_K"]
            ),
            "stage3_p90_conflict_relative_reduction": rel_reduction(
                s3_1["p90_conflict_K"], s3_5["p90_conflict_K"]
            ),
            "stage3_fidelity_absolute_change": (
                s3_5["fused_mean_fidelity"]
                - s3_1["fused_mean_fidelity"]
            ),
            "stage3_fidelity_relative_change": (
                s3_5["fused_mean_fidelity"]
                / s3_1["fused_mean_fidelity"]
                - 1.0
            ),
            "stage3_all_model_validity_absolute_change": (
                s3_5["fused_all_models_cf_valid_rate"]
                - s3_1["fused_all_models_cf_valid_rate"]
            ),
        },
        "lambda_5_to_10": {
            "stage3_mean_conflict_relative_reduction": rel_reduction(
                s3_5["mean_conflict_K"], s3_10["mean_conflict_K"]
            ),
            "stage3_fidelity_absolute_change": (
                s3_10["fused_mean_fidelity"]
                - s3_5["fused_mean_fidelity"]
            ),
            "stage3_ignorance_absolute_change": (
                s3_10["fused_mean_ignorance"]
                - s3_5["fused_mean_ignorance"]
            ),
            "stage3_all_model_validity_absolute_change": (
                s3_10["fused_all_models_cf_valid_rate"]
                - s3_5["fused_all_models_cf_valid_rate"]
            ),
        },
    }

    protocol = sweep["protocol"]

    frozen = {
        "schema_version": "etce-phase4c-ds-frozen-v1",
        "status": "frozen",
        "selection_scope": "validation_only",
        "formal_test_seen_before_freeze": False,
        "lambda": SELECTED_LAMBDA,
        "lambda_candidate_grid": EXPECTED_LAMBDAS,
        "lambda_selection_basis": (
            "validation-only compromise between rule fidelity, "
            "counterfactual validity, epistemic ignorance, and "
            "Dempster conflict; lambda=5 is the first substantial "
            "conflict-reduction point without the stronger "
            "information discount observed for larger lambdas"
        ),
        "reliability": {
            "rho_formula": "rho_r = n_r / (n_r + lambda)",
            "reliability_formula": "R(r) = rho_r * Fid(r)",
            "support_definition": "n_r = held-out local evaluation support",
            "fidelity_definition": (
                "Fid(r) = black-box-consistent covered eval samples / n_r"
            ),
        },
        "bpa": {
            "m_theta": "R(r)",
            "m_Theta": "1 - R(r)",
            "m_not_theta": 0.0,
        },
        "grouping": {
            "stage1": (
                "case_id × model × prefix × surrogate × "
                "predicted_class × feature_signature"
            ),
            "stage2": (
                "case_id × model × prefix × "
                "predicted_class × feature_signature"
            ),
            "stage3": (
                "case_id × prefix × "
                "predicted_class × feature_signature"
            ),
        },
        "fusion": {
            "stage1_order": protocol["stage1_order"],
            "stage2_order": protocol["stage2_order"],
            "stage3_order": protocol["stage3_order"],
            "K_total_conflict_guard": protocol[
                "K_total_conflict_guard"
            ],
            "total_conflict_policy": protocol[
                "total_conflict_policy"
            ],
            "representative_rule_policy": protocol[
                "representative_rule_policy"
            ],
            "fused_rule_re_evaluation": protocol[
                "fused_rule_re_evaluation"
            ],
        },
        "validation_evidence": {
            "stage1_fusable_groups": int(
                sweep["group_counts"]["stage1_fusable"]
            ),
            "stage2_fusable_groups": int(
                sweep["group_counts"]["stage2_fusable"]
            ),
            "stage3_fusable_groups": int(
                sweep["group_counts"]["stage3_fusable"]
            ),
            "total_conflict_fallback_count_all_stage_lambda_rows": 0,
            "lambda_5_stage1": s1_5,
            "lambda_5_stage2": s2_5,
            "lambda_5_stage3": s3_5,
            "selection_diagnostics": diagnostics,
            "stage3_lambda_stability": sweep.get(
                "stage3_lambda_stability"
            ),
        },
        "provenance": {
            "phase4b_validation_sha256": sha256(PHASE4B_VALIDATION),
            "group_audit_sha256": sha256(GROUP_AUDIT),
            "fusion_cache_validation_sha256": sha256(CACHE_VALIDATION),
            "ds_sweep_json_sha256": sha256(SWEEP_JSON),
            "ds_lambda_summary_sha256": sha256(SUMMARY_CSV),
        },
        "freeze_rule": (
            "This file must not be altered after formal test outputs "
            "are inspected. Any later protocol change requires a new "
            "versioned configuration and must not be tuned on test."
        ),
        "next_phase": (
            "formal test rule-bank construction and D-S fusion using "
            "this exact frozen protocol"
        ),
    }

    write_json(OUT_CONFIG, frozen)

    d = diagnostics["lambda_1_to_5"]
    d2 = diagnostics["lambda_5_to_10"]

    report = f"""# ETCE Phase 4C-v2 Frozen D-S Protocol

## Frozen parameter

**lambda = 5**

The parameter was selected using validation data only. Formal test
results were not inspected before freezing.

## Selection evidence

Relative to lambda=1:

- Stage-1 mean conflict K reduction: {100*d['stage1_mean_conflict_relative_reduction']:.2f}%
- Stage-2 mean conflict K reduction: {100*d['stage2_mean_conflict_relative_reduction']:.2f}%
- Stage-3 mean conflict K reduction: {100*d['stage3_mean_conflict_relative_reduction']:.2f}%
- Stage-3 P90 conflict K reduction: {100*d['stage3_p90_conflict_relative_reduction']:.2f}%
- Stage-3 mean Rule Fidelity change: {d['stage3_fidelity_absolute_change']:+.4f}
  ({100*d['stage3_fidelity_relative_change']:+.2f}% relative)
- Stage-3 all-model CF-validity change:
  {100*d['stage3_all_model_validity_absolute_change']:+.2f} percentage points

Moving from lambda=5 to lambda=10:

- Stage-3 mean conflict continues to decrease by
  {100*d2['stage3_mean_conflict_relative_reduction']:.2f}%
- Stage-3 mean Rule Fidelity changes by
  {d2['stage3_fidelity_absolute_change']:+.4f}
- Stage-3 mean ignorance increases by
  {d2['stage3_ignorance_absolute_change']:+.4f}
- Stage-3 all-model CF validity changes by only
  {100*d2['stage3_all_model_validity_absolute_change']:+.2f} percentage points

Thus lambda=5 is frozen as the validation-selected compromise:
it provides the first large reduction in D-S conflict while retaining
substantially more committed evidence than the larger lambda values.

## Conflict safety

Across all 15 Stage-by-lambda validation rows:

**total-conflict fallback count = 0**

The K->1 safeguard remains frozen for formal test use, but it was not
triggered during validation.

## Frozen hierarchy

- Stage 1:
  case_id × model × prefix × surrogate × predicted_class × feature_signature
- Stage 2:
  case_id × model × prefix × predicted_class × feature_signature
- Stage 3:
  case_id × prefix × predicted_class × feature_signature

Fusion order and conflict/tie handling are stored in:

`configs/phase4c_ds_frozen_config.json`

## Test protocol

After this file is created, formal test may be opened. Lambda,
BPA mapping, grouping, fusion order, conflict handling, representative
rule selection, and fused-rule re-evaluation must not be changed in
response to test results.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")

    print("=" * 96)
    print("ETCE PHASE 4C-v2 D-S PROTOCOL FROZEN")
    print("=" * 96)
    print(f"lambda = {SELECTED_LAMBDA:g}")
    print("Selection scope: validation only")
    print(f"Total-conflict fallbacks in sweep: {fallback_total}")
    print(f"Config: {OUT_CONFIG}")
    print(f"Report: {OUT_REPORT}")
    print()
    print("Formal test is now permitted under this frozen protocol.")
    print("Do NOT modify lambda/BPA/grouping/fusion/tie rules after test inspection.")


if __name__ == "__main__":
    main()
