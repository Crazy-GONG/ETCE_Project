#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4C-v2: target-local hierarchical grouping audit for the
Phase 4B-v2 multi-rule validation bank.

Purpose
-------
Verify that the multi-rule extension really removes the former Stage-1
degeneracy under the CORRECT target-local hierarchy.

No D-S fusion is performed here.
No lambda is selected here.
No test data are read.

Default input:
    results/phase4/phase4b_v2_multirule_bank/canonical_rules_calibration.jsonl

Hierarchy
---------
Stage 1:
    case_id × model × prefix × surrogate × predicted_class × feature_signature

Stage 2:
    case_id × model × prefix × predicted_class × feature_signature
    (cross-surrogate compatibility)

Stage 3:
    case_id × prefix × predicted_class × feature_signature
    (cross-black-box compatibility)

Run:
    python -u scripts/28_phase4c_v2_target_local_group_audit.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = (
    DEFAULT_ROOT
    / "results/phase4/phase4b_v2_multirule_bank/canonical_rules_calibration.jsonl"
)
DEFAULT_VALIDATION = (
    DEFAULT_ROOT
    / "results/phase4/phase4b_v2_multirule_bank/validation_phase4b_v2.json"
)
DEFAULT_OUT = (
    DEFAULT_ROOT
    / "results/phase4/phase4c_v2_target_local_audit"
)


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL at line {line_no}: {path}"
                ) from exc
    return pd.DataFrame(rows)


def feature_signature(conditions):
    """
    Compatibility signature inherited from corrected 25b audit:
    use the set of rule positions referenced by the rule conditions.
    """
    if not isinstance(conditions, list):
        return tuple()

    positions = []
    for cond in conditions:
        if (
            isinstance(cond, dict)
            and cond.get("position") is not None
        ):
            positions.append(int(cond["position"]))

    return tuple(sorted(set(positions)))


def signature_to_str(sig):
    return "[" + ",".join(map(str, sig)) + "]"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rules",
        default=str(DEFAULT_RULES),
        help="Phase 4B-v2 canonical multi-rule JSONL",
    )
    ap.add_argument(
        "--validation",
        default=str(DEFAULT_VALIDATION),
        help="Phase 4B-v2 independent validation JSON",
    )
    ap.add_argument(
        "--out_dir",
        default=str(DEFAULT_OUT),
        help="Output directory",
    )
    args = ap.parse_args()

    rules_path = Path(args.rules).expanduser().resolve()
    validation_path = Path(args.validation).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    require(rules_path.is_file(), f"Rules file missing: {rules_path}")
    require(
        validation_path.is_file(),
        f"Phase 4B-v2 validation file missing: {validation_path}",
    )

    with validation_path.open("r", encoding="utf-8") as f:
        validation = json.load(f)

    require(
        validation.get("status") == "passed",
        "Phase 4B-v2 validator is not PASSED",
    )
    require(
        validation.get("test_cases_used") is False,
        "Phase 4B-v2 unexpectedly used test cases",
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(rules_path)
    require(len(df) > 0, "Canonical rule bank is empty")

    required_columns = {
        "case_id",
        "model",
        "prefix",
        "surrogate",
        "candidate_id",
        "rule_type",
        "predicted_class",
        "conditions",
        "n_r",
        "Fid_r",
        "coverage_eval",
        "eligible_for_reliability",
    }
    missing = sorted(required_columns - set(df.columns))
    require(
        not missing,
        "Canonical rule bank missing required fields: "
        + ", ".join(missing),
    )

    # Counterfactual inventory before eligibility filtering.
    cf_all = df[df["rule_type"] == "counterfactual"].copy()

    # candidate_id must be unique within each target-local surrogate.
    candidate_key = [
        "case_id",
        "model",
        "prefix",
        "surrogate",
        "candidate_id",
        "rule_type",
    ]
    duplicate_candidate_rows = int(
        cf_all.duplicated(candidate_key, keep=False).sum()
    )
    require(
        duplicate_candidate_rows == 0,
        "Duplicate target-local candidate_id detected",
    )

    # Reliability/BPA can only use eligible CF candidates.
    cf = cf_all[
        cf_all["eligible_for_reliability"].astype(bool)
    ].copy()

    cf["feature_signature"] = cf["conditions"].apply(
        feature_signature
    )
    cf["feature_signature_str"] = cf[
        "feature_signature"
    ].apply(signature_to_str)

    # ------------------------------------------------------------------
    # Raw target-local multi-rule inventory BEFORE feature compatibility.
    # This connects directly to Phase 4B-v2 multi_rule_summary.json.
    # ------------------------------------------------------------------
    raw_group_keys = [
        "case_id",
        "model",
        "prefix",
        "surrogate",
        "predicted_class",
    ]
    raw_s1 = (
        cf.groupby(raw_group_keys, dropna=False)
        .size()
        .reset_index(name="eligible_rule_count")
    )
    raw_s1["raw_multirule"] = (
        raw_s1["eligible_rule_count"] >= 2
    )
    raw_s1.to_csv(
        out_dir / "stage1_raw_target_local_inventory.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Stage 1: same fitted local surrogate AND same feature signature.
    # ------------------------------------------------------------------
    s1_keys = [
        "case_id",
        "model",
        "prefix",
        "surrogate",
        "predicted_class",
        "feature_signature_str",
    ]
    s1 = (
        cf.groupby(s1_keys, dropna=False)
        .agg(
            rule_count=("candidate_id", "size"),
            unique_candidate_count=("candidate_id", "nunique"),
            mean_support=("n_r", "mean"),
            mean_fidelity=("Fid_r", "mean"),
            mean_coverage=("coverage_eval", "mean"),
        )
        .reset_index()
    )
    s1["stage1_fusable"] = s1["rule_count"] >= 2
    s1.to_csv(
        out_dir / "stage1_target_local_inventory.csv",
        index=False,
    )

    # Source rules that actually participate in non-degenerate Stage 1.
    stage1_rules_in_fusable_groups = int(
        s1.loc[
            s1["stage1_fusable"],
            "rule_count",
        ].sum()
    )

    # ------------------------------------------------------------------
    # Stage 2: across LORE/DT/BN for SAME target/model/prefix and
    # compatible predicted class + feature signature.
    # ------------------------------------------------------------------
    s2_keys = [
        "case_id",
        "model",
        "prefix",
        "predicted_class",
        "feature_signature_str",
    ]
    s2 = (
        cf.groupby(s2_keys, dropna=False)
        .agg(
            surrogate_count=("surrogate", "nunique"),
            source_rule_count=("candidate_id", "size"),
            candidate_count=("candidate_id", "nunique"),
        )
        .reset_index()
    )
    s2["stage2_fusable"] = s2["surrogate_count"] >= 2
    s2.to_csv(
        out_dir / "stage2_same_target_inter_surrogate.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Stage 3: across black-boxes for SAME case/prefix and compatible
    # predicted class + feature signature.
    # ------------------------------------------------------------------
    s3_keys = [
        "case_id",
        "prefix",
        "predicted_class",
        "feature_signature_str",
    ]
    s3 = (
        cf.groupby(s3_keys, dropna=False)
        .agg(
            blackbox_count=("model", "nunique"),
            surrogate_count=("surrogate", "nunique"),
            source_rule_count=("candidate_id", "size"),
        )
        .reset_index()
    )
    s3["stage3_fusable"] = s3["blackbox_count"] >= 2
    s3.to_csv(
        out_dir / "stage3_same_case_inter_blackbox.csv",
        index=False,
    )

    # Per-surrogate Stage-1 detail for diagnostics.
    surrogate_stage1 = []
    for surrogate in ("LORE", "DT", "BN"):
        g = s1[s1["surrogate"] == surrogate]
        surrogate_stage1.append(
            {
                "surrogate": surrogate,
                "stage1_group_count": int(len(g)),
                "stage1_fusable_group_count": int(
                    g["stage1_fusable"].sum()
                ),
                "stage1_max_group_size": int(
                    g["rule_count"].max()
                ) if len(g) else 0,
                "source_rules_in_stage1_fusable_groups": int(
                    g.loc[
                        g["stage1_fusable"],
                        "rule_count",
                    ].sum()
                ),
            }
        )
    pd.DataFrame(surrogate_stage1).to_csv(
        out_dir / "stage1_by_surrogate.csv",
        index=False,
    )

    raw_multirule_groups = int(raw_s1["raw_multirule"].sum())
    stage1_fusable_groups = int(s1["stage1_fusable"].sum())
    stage1_max = int(s1["rule_count"].max()) if len(s1) else 0

    status = (
        "non_degenerate"
        if stage1_fusable_groups > 0 and stage1_max >= 2
        else "degenerate"
    )

    summary = {
        "scope": "ETCE Phase 4C-v2 target-local hierarchy audit",
        "source_bank": str(rules_path),
        "source_phase4b_v2_validation_status": validation.get(
            "status"
        ),
        "test_data_used": False,
        "lambda_selected": False,
        "ds_fusion_performed": False,
        "canonical_rule_count": int(len(df)),
        "counterfactual_candidate_count": int(len(cf_all)),
        "eligible_counterfactual_candidate_count": int(len(cf)),
        "candidate_ids_unique_within_target_local_surrogate": True,
        "raw_target_local_inventory": {
            "group_count": int(len(raw_s1)),
            "groups_with_ge2_eligible_rules": raw_multirule_groups,
            "max_group_size": int(
                raw_s1["eligible_rule_count"].max()
            ) if len(raw_s1) else 0,
        },
        "stage1": {
            "definition": (
                "case_id × model × prefix × surrogate × "
                "predicted_class × feature_signature"
            ),
            "group_count": int(len(s1)),
            "fusable_group_count_ge2_rules": stage1_fusable_groups,
            "max_group_size": stage1_max,
            "source_rules_in_fusable_groups":
                stage1_rules_in_fusable_groups,
            "non_degenerate": bool(
                stage1_fusable_groups > 0 and stage1_max >= 2
            ),
            "by_surrogate": surrogate_stage1,
        },
        "stage2": {
            "definition": (
                "case_id × model × prefix × predicted_class × "
                "feature_signature"
            ),
            "compatible_sets": int(len(s2)),
            "sets_with_ge2_surrogates": int(
                s2["stage2_fusable"].sum()
            ),
            "sets_with_all_3_surrogates": int(
                (s2["surrogate_count"] == 3).sum()
            ),
            "source_rules_in_fusable_sets": int(
                s2.loc[
                    s2["stage2_fusable"],
                    "source_rule_count",
                ].sum()
            ),
        },
        "stage3": {
            "definition": (
                "case_id × prefix × predicted_class × "
                "feature_signature"
            ),
            "compatible_sets": int(len(s3)),
            "sets_with_ge2_blackboxes": int(
                s3["stage3_fusable"].sum()
            ),
            "sets_with_ge3_blackboxes": int(
                (s3["blackbox_count"] >= 3).sum()
            ),
            "sets_with_all_4_blackboxes": int(
                (s3["blackbox_count"] == 4).sum()
            ),
            "source_rules_in_fusable_sets": int(
                s3.loc[
                    s3["stage3_fusable"],
                    "source_rule_count",
                ].sum()
            ),
        },
        "result": {
            "stage1_status": status,
            "former_single_rule_degeneracy_resolved": bool(
                status == "non_degenerate"
            ),
            "ready_for_lambda_sensitivity": bool(
                status == "non_degenerate"
            ),
        },
    }

    summary_path = out_dir / "target_local_group_audit_v2.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 100)
    print("ETCE PHASE 4C-v2 - TARGET-LOCAL HIERARCHY AUDIT")
    print("=" * 100)
    print(f"Rules: {rules_path}")
    print(f"Phase 4B-v2 validation: {validation.get('status')}")
    print(f"Canonical rules: {len(df)}")
    print(f"CF candidates: {len(cf_all)}")
    print(f"Eligible CF candidates: {len(cf)}")
    print()

    print("[Raw target-local multi-rule inventory]")
    print(f"Groups: {len(raw_s1)}")
    print(f"Groups with >=2 eligible rules: {raw_multirule_groups}")
    print(
        "Maximum raw group size: "
        f"{int(raw_s1['eligible_rule_count'].max()) if len(raw_s1) else 0}"
    )
    print()

    print("[Stage 1: same target-local surrogate + compatible feature signature]")
    print(f"Groups: {len(s1)}")
    print(f"Groups with >=2 rules: {stage1_fusable_groups}")
    print(f"Maximum group size: {stage1_max}")
    print(
        "Source rules in fusable groups: "
        f"{stage1_rules_in_fusable_groups}"
    )
    for item in surrogate_stage1:
        print(
            f"  {item['surrogate']}: "
            f"fusable={item['stage1_fusable_group_count']} "
            f"max={item['stage1_max_group_size']} "
            f"rules={item['source_rules_in_stage1_fusable_groups']}"
        )
    print()

    print("[Stage 2: across surrogates for same target/model/prefix]")
    print(f"Compatible sets: {len(s2)}")
    print(
        "Sets with >=2 surrogates: "
        f"{int(s2['stage2_fusable'].sum())}"
    )
    print(
        "Sets with all 3 surrogates: "
        f"{int((s2['surrogate_count'] == 3).sum())}"
    )
    print()

    print("[Stage 3: across black-boxes for same case/prefix]")
    print(f"Compatible sets: {len(s3)}")
    print(
        "Sets with >=2 black-boxes: "
        f"{int(s3['stage3_fusable'].sum())}"
    )
    print(
        "Sets with >=3 black-boxes: "
        f"{int((s3['blackbox_count'] >= 3).sum())}"
    )
    print(
        "Sets with all 4 black-boxes: "
        f"{int((s3['blackbox_count'] == 4).sum())}"
    )
    print()

    if status == "non_degenerate":
        print("RESULT: Stage 1 is NON-DEGENERATE under the Phase 4B-v2 bank.")
        print("The former single-rule Stage-1 degeneracy is resolved.")
        print("NEXT: rerun lambda sensitivity on the Phase 4B-v2 eligible CF candidates.")
    else:
        print("RESULT: Stage 1 remains DEGENERATE after feature-signature grouping.")
        print("Do NOT proceed to lambda selection or D-S fusion.")

    print("No test data read; no lambda selected; no D-S fusion performed.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
