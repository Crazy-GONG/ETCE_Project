#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4C-B0: Validation rule grouping audit.

Purpose
-------
Verify that the Phase 4B canonical validation bank can instantiate the
three-stage ETCE fusion hierarchy without touching test data.

Grouping follows the paper's fusion compatibility condition:
two counterfactual rules are fusion-compatible when they share
(1) the same predicted class and
(2) the same involved feature set.

In the current sequential-prefix representation, the involved feature set
is represented by the sorted tuple of `position` values in `conditions`.

Hierarchy audited here
----------------------
Stage 1 (intra-group):
    within the same surrogate / white-box model:
    black-box model × prefix × surrogate × predicted class × feature signature

Stage 2 (inter-group):
    across surrogate / white-box models derived from the same black-box:
    black-box model × prefix × predicted class × feature signature

Stage 3 (inter-model):
    across black-box models:
    prefix × predicted class × feature signature

No fusion is performed in this script. It only audits whether compatible
multi-rule / multi-surrogate / multi-black-box groups actually exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"Invalid JSON at line {i}: {exc}") from exc
    return pd.DataFrame(rows)


def feature_signature(conditions):
    """Canonical feature-set signature used only for grouping compatibility."""
    if not isinstance(conditions, list):
        return tuple()
    positions = []
    for c in conditions:
        if not isinstance(c, dict):
            continue
        p = c.get("position")
        if p is not None:
            positions.append(int(p))
    return tuple(sorted(set(positions)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True, help="Phase 4B canonical_rules_calibration.jsonl")
    ap.add_argument("--out_dir", default="results/phase4/phase4c_group_audit")
    args = ap.parse_args()

    rules_path = Path(args.rules)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(rules_path)

    required = {
        "case_id", "model", "prefix", "surrogate", "predicted_class",
        "conditions", "rule_type", "eligible_for_reliability"
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required fields: {sorted(missing)}")

    cf = df[
        (df["rule_type"] == "counterfactual")
        & (df["eligible_for_reliability"] == True)
    ].copy()

    cf["feature_signature"] = cf["conditions"].apply(feature_signature)
    cf["feature_signature_str"] = cf["feature_signature"].apply(
        lambda x: "[" + ",".join(map(str, x)) + "]"
    )

    # ---------------- Stage 1 ----------------
    s1_keys = [
        "model", "prefix", "surrogate", "predicted_class", "feature_signature_str"
    ]
    s1 = (
        cf.groupby(s1_keys, dropna=False)
        .size()
        .reset_index(name="rule_count")
        .sort_values(["rule_count"] + s1_keys, ascending=[False, True, True, True, True, True])
    )
    s1["stage1_fusable"] = s1["rule_count"] >= 2
    s1.to_csv(out_dir / "stage1_intragroup_inventory.csv", index=False)

    # ---------------- Stage 2 ----------------
    # Each Stage-1 group becomes one group-level evidence unit.
    s2_keys = ["model", "prefix", "predicted_class", "feature_signature_str"]
    s2 = (
        s1.groupby(s2_keys, dropna=False)
        .agg(
            surrogate_group_count=("surrogate", "nunique"),
            stage1_group_count=("surrogate", "size"),
            source_rule_count=("rule_count", "sum"),
        )
        .reset_index()
    )
    s2["stage2_fusable"] = s2["surrogate_group_count"] >= 2
    s2 = s2.sort_values(
        ["surrogate_group_count", "source_rule_count"] + s2_keys,
        ascending=[False, False, True, True, True, True],
    )
    s2.to_csv(out_dir / "stage2_intergroup_inventory.csv", index=False)

    # ---------------- Stage 3 ----------------
    # Each black-box contributes at most one Stage-2 output for a compatible signature.
    s3_keys = ["prefix", "predicted_class", "feature_signature_str"]
    s3 = (
        s2.groupby(s3_keys, dropna=False)
        .agg(
            blackbox_model_count=("model", "nunique"),
            model_group_count=("model", "size"),
            source_rule_count=("source_rule_count", "sum"),
        )
        .reset_index()
    )
    s3["stage3_fusable"] = s3["blackbox_model_count"] >= 2
    s3 = s3.sort_values(
        ["blackbox_model_count", "source_rule_count"] + s3_keys,
        ascending=[False, False, True, True, True],
    )
    s3.to_csv(out_dir / "stage3_intermodel_inventory.csv", index=False)

    # Summary
    s1_multi_rules = int(s1.loc[s1["stage1_fusable"], "rule_count"].sum())
    s2_fusable_rules = int(s2.loc[s2["stage2_fusable"], "source_rule_count"].sum())
    s3_fusable_rules = int(s3.loc[s3["stage3_fusable"], "source_rule_count"].sum())

    summary = {
        "scope": "ETCE Phase 4C validation grouping audit",
        "test_data_used": False,
        "eligible_counterfactual_rule_count": int(len(cf)),
        "stage1": {
            "group_count": int(len(s1)),
            "fusable_group_count_ge2_rules": int(s1["stage1_fusable"].sum()),
            "rules_inside_fusable_groups": s1_multi_rules,
            "fraction_rules_inside_fusable_groups": float(s1_multi_rules / len(cf)) if len(cf) else 0.0,
            "max_group_size": int(s1["rule_count"].max()) if len(s1) else 0,
        },
        "stage2": {
            "compatible_sets": int(len(s2)),
            "sets_with_ge2_surrogates": int(s2["stage2_fusable"].sum()),
            "sets_with_all_3_surrogates": int((s2["surrogate_group_count"] == 3).sum()),
            "source_rules_represented_in_fusable_sets": s2_fusable_rules,
        },
        "stage3": {
            "compatible_sets": int(len(s3)),
            "sets_with_ge2_blackboxes": int(s3["stage3_fusable"].sum()),
            "sets_with_ge3_blackboxes": int((s3["blackbox_model_count"] >= 3).sum()),
            "sets_with_all_4_blackboxes": int((s3["blackbox_model_count"] == 4).sum()),
            "source_rules_represented_in_fusable_sets": s3_fusable_rules,
        },
    }

    with (out_dir / "phase4c_group_audit.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 92)
    print("ETCE PHASE 4C-B0 - VALIDATION RULE GROUPING AUDIT")
    print("=" * 92)
    print(f"Rules: {rules_path}")
    print(f"Eligible counterfactual rules: {len(cf)}")
    print()
    print("[Stage 1: intra-group, within same surrogate]")
    print(f"Groups: {len(s1)}")
    print(f"Groups with >=2 rules: {int(s1['stage1_fusable'].sum())}")
    print(f"Rules inside >=2-rule groups: {s1_multi_rules} / {len(cf)} "
          f"({100*s1_multi_rules/len(cf):.2f}%)")
    print(f"Maximum Stage-1 group size: {int(s1['rule_count'].max()) if len(s1) else 0}")
    print()
    print("[Stage 2: inter-group, across surrogates of same black-box]")
    print(f"Compatible sets: {len(s2)}")
    print(f"Sets with >=2 surrogates: {int(s2['stage2_fusable'].sum())}")
    print(f"Sets with all 3 surrogates: {int((s2['surrogate_group_count']==3).sum())}")
    print(f"Source rules represented in fusable sets: {s2_fusable_rules}")
    print()
    print("[Stage 3: inter-model, across black-box models]")
    print(f"Compatible sets: {len(s3)}")
    print(f"Sets with >=2 black-box models: {int(s3['stage3_fusable'].sum())}")
    print(f"Sets with >=3 black-box models: {int((s3['blackbox_model_count']>=3).sum())}")
    print(f"Sets with all 4 black-box models: {int((s3['blackbox_model_count']==4).sum())}")
    print(f"Source rules represented in fusable sets: {s3_fusable_rules}")
    print()
    print(f"Outputs: {out_dir}")
    print("No test data were read. No lambda was selected. No D-S fusion was performed.")


if __name__ == "__main__":
    main()
