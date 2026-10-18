#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4C-B0b: target-local fusion hierarchy audit.

This is the corrected audit for the rebuilt ETCE protocol.

Key principle:
The local surrogate models are fitted separately for each
(case_id, black-box model, prefix) target. Therefore rules from
different case_id values must NOT be treated as rules from the
same fitted surrogate model.

Audit hierarchy:
Stage 1: within the same target-local surrogate instance
         case_id × blackbox × prefix × surrogate × class × feature-set
Stage 2: across LORE/DT/BN for the SAME target, blackbox and prefix
         case_id × blackbox × prefix × class × feature-set
Stage 3: across black-box models for the SAME case and prefix
         case_id × prefix × class × feature-set

No D-S fusion is performed. No test data are read.
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
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def feature_signature(conditions):
    if not isinstance(conditions, list):
        return tuple()
    pos = []
    for c in conditions:
        if isinstance(c, dict) and c.get("position") is not None:
            pos.append(int(c["position"]))
    return tuple(sorted(set(pos)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--out_dir",
                    default="results/phase4/phase4c_target_local_audit")
    args = ap.parse_args()

    rules_path = Path(args.rules)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(rules_path)
    cf = df[
        (df["rule_type"] == "counterfactual") &
        (df["eligible_for_reliability"] == True)
    ].copy()

    cf["feature_signature"] = cf["conditions"].apply(feature_signature)
    cf["feature_signature_str"] = cf["feature_signature"].apply(
        lambda x: "[" + ",".join(map(str, x)) + "]"
    )

    # Stage 1: SAME fitted local surrogate instance.
    s1_keys = [
        "case_id", "model", "prefix", "surrogate",
        "predicted_class", "feature_signature_str"
    ]
    s1 = (
        cf.groupby(s1_keys, dropna=False)
          .size()
          .reset_index(name="rule_count")
    )
    s1["stage1_fusable"] = s1["rule_count"] >= 2
    s1.to_csv(out_dir / "stage1_target_local_inventory.csv", index=False)

    # Stage 2: across surrogates, but SAME target + blackbox + prefix.
    s2_keys = [
        "case_id", "model", "prefix",
        "predicted_class", "feature_signature_str"
    ]
    s2 = (
        cf.groupby(s2_keys, dropna=False)
          .agg(
              surrogate_count=("surrogate", "nunique"),
              source_rule_count=("surrogate", "size"),
          )
          .reset_index()
    )
    s2["stage2_fusable"] = s2["surrogate_count"] >= 2
    s2.to_csv(out_dir / "stage2_same_target_inter_surrogate.csv", index=False)

    # Stage 3: across blackboxes for SAME case + prefix + compatible rule signature.
    s3_input = s2.copy()
    s3_keys = [
        "case_id", "prefix",
        "predicted_class", "feature_signature_str"
    ]
    s3 = (
        s3_input.groupby(s3_keys, dropna=False)
                .agg(
                    blackbox_count=("model", "nunique"),
                    model_group_count=("model", "size"),
                    source_rule_count=("source_rule_count", "sum"),
                )
                .reset_index()
    )
    s3["stage3_fusable"] = s3["blackbox_count"] >= 2
    s3.to_csv(out_dir / "stage3_same_case_inter_blackbox.csv", index=False)

    summary = {
        "scope": "ETCE target-local hierarchy audit",
        "test_data_used": False,
        "eligible_counterfactual_rule_count": int(len(cf)),
        "stage1": {
            "group_count": int(len(s1)),
            "fusable_group_count_ge2_rules": int(s1["stage1_fusable"].sum()),
            "max_group_size": int(s1["rule_count"].max()) if len(s1) else 0,
        },
        "stage2": {
            "compatible_sets": int(len(s2)),
            "sets_with_ge2_surrogates": int(s2["stage2_fusable"].sum()),
            "sets_with_all_3_surrogates": int((s2["surrogate_count"] == 3).sum()),
            "source_rules_in_fusable_sets":
                int(s2.loc[s2["stage2_fusable"], "source_rule_count"].sum()),
        },
        "stage3": {
            "compatible_sets": int(len(s3)),
            "sets_with_ge2_blackboxes": int(s3["stage3_fusable"].sum()),
            "sets_with_ge3_blackboxes": int((s3["blackbox_count"] >= 3).sum()),
            "sets_with_all_4_blackboxes": int((s3["blackbox_count"] == 4).sum()),
            "source_rules_in_fusable_sets":
                int(s3.loc[s3["stage3_fusable"], "source_rule_count"].sum()),
        },
        "interpretation": (
            "If Stage-1 max_group_size == 1, the current canonical bank contains "
            "only one eligible counterfactual rule per fitted target-local surrogate. "
            "A non-degenerate three-stage ETCE implementation therefore requires "
            "exporting multiple compatible CF candidate rules per target-local surrogate."
        )
    }
    with (out_dir / "target_local_group_audit.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 92)
    print("ETCE PHASE 4C-B0b - TARGET-LOCAL FUSION HIERARCHY AUDIT")
    print("=" * 92)
    print(f"Rules: {rules_path}")
    print(f"Eligible counterfactual rules: {len(cf)}")
    print()
    print("[Stage 1: within SAME target-local surrogate instance]")
    print(f"Groups: {len(s1)}")
    print(f"Groups with >=2 rules: {int(s1['stage1_fusable'].sum())}")
    print(f"Maximum group size: {int(s1['rule_count'].max()) if len(s1) else 0}")
    print()
    print("[Stage 2: across surrogates for SAME target/black-box/prefix]")
    print(f"Compatible sets: {len(s2)}")
    print(f"Sets with >=2 surrogates: {int(s2['stage2_fusable'].sum())}")
    print(f"Sets with all 3 surrogates: {int((s2['surrogate_count'] == 3).sum())}")
    print(f"Source rules in fusable sets: "
          f"{int(s2.loc[s2['stage2_fusable'], 'source_rule_count'].sum())}")
    print()
    print("[Stage 3: across black-boxes for SAME case/prefix]")
    print(f"Compatible sets: {len(s3)}")
    print(f"Sets with >=2 black-boxes: {int(s3['stage3_fusable'].sum())}")
    print(f"Sets with >=3 black-boxes: {int((s3['blackbox_count'] >= 3).sum())}")
    print(f"Sets with all 4 black-boxes: {int((s3['blackbox_count'] == 4).sum())}")
    print()
    if len(s1) and int(s1["rule_count"].max()) == 1:
        print("RESULT: Stage 1 is DEGENERATE under the current canonical bank.")
        print("NEXT: export multiple CF candidate rules per target-local surrogate before full 3-stage D-S.")
    else:
        print("RESULT: Stage 1 has multi-rule target-local groups.")
    print("No test data were read; no lambda selected; no D-S fusion performed.")


if __name__ == "__main__":
    main()
