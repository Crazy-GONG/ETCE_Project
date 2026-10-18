#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Stage 68C: target-local grouping audit for the masked sensitivity bank.

Correct protocol interpretation
-------------------------------
The formal TEST cohort is defined by Phase 4D-0 and independently validated
by Phase 4D-2. The canonical rule bank is NOT a dense target roster: a formal
target/surrogate may legitimately emit no canonical rule. Therefore:

- 100 selected TEST cases are checked from formal_test_case_selection.csv.
- 1200 formal targets (P10/P15/P20) are checked from the masked bank.
- 3600 target-surrogate groups are checked from the masked bank.
- Canonical-rule and eligible-CF counts are read from the completed Stage-68B
  bank summary rather than copied from the original unmasked experiment.
- Stage 1/2/3 grouping is audited ONLY over actually existing eligible CF rules.

No lambda selection, protocol retuning, or D-S fusion is performed.
Frozen lambda remains 5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_SELECTED_CASES = 100
EXPECTED_FORMAL_TARGETS = 1200
EXPECTED_TARGET_SURROGATE_GROUPS = 3600
FROZEN_LAMBDA = 5.0

EXPECTED_PREFIXES = {10, 15, 20}
EXPECTED_MODELS = {"RF", "XGBoost", "MLP", "LSTM"}
EXPECTED_SURROGATES = {"LORE", "DT", "BN"}


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def load_json(path: Path) -> Any:
    if not path.exists():
        fail(f"Required file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Required JSONL does not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                fail(f"Invalid JSON at {path}:{i}: {exc}")
    return pd.DataFrame(rows)


def feature_signature(conditions: Any) -> tuple[int, ...]:
    """Frozen grouping signature: sorted unique condition positions."""
    if not isinstance(conditions, list):
        return tuple()
    pos = []
    for c in conditions:
        if isinstance(c, dict) and c.get("position") is not None:
            pos.append(int(c["position"]))
    return tuple(sorted(set(pos)))


def signature_to_str(sig: tuple[int, ...]) -> str:
    return "[" + ",".join(map(str, sig)) + "]"


def recursive_values(obj: Any, accepted_keys: set[str]) -> list[Any]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            nk = str(k).strip().lower().replace("-", "_").replace(" ", "_")
            if nk in accepted_keys:
                out.append(v)
            out.extend(recursive_values(v, accepted_keys))
    elif isinstance(obj, list):
        for x in obj:
            out.extend(recursive_values(x, accepted_keys))
    return out


def check_validator(validation_obj: Any) -> None:
    """
    Best-effort PASS gate. The Phase 4D-2 validator has already independently
    recomputed all formal targets; this function only refuses explicit failure.
    """
    keys = {
        "status", "validation_status", "overall_status", "result",
        "passed", "validation_passed", "all_passed"
    }
    vals = recursive_values(validation_obj, keys)
    lowered = {str(v).strip().lower() for v in vals}
    fail_tokens = {"fail", "failed", "false", "error"}
    if lowered & fail_tokens:
        fail(f"Phase 4D-2 validator contains FAIL-like status: {sorted(lowered)}")


def extract_frozen_lambda(cfg: Any) -> float:
    preferred_keys = {
        "selected_lambda", "frozen_lambda", "reliability_lambda",
        "lambda_selected", "lambda"
    }
    vals = recursive_values(cfg, preferred_keys)
    nums = []
    for v in vals:
        try:
            nums.append(float(v))
        except Exception:
            pass

    if not any(abs(x - FROZEN_LAMBDA) < 1e-12 for x in nums):
        fail(
            "Frozen D-S config does not expose lambda=5 in a recognized "
            f"lambda field. Numeric lambda-like values found: {nums}"
        )
    return FROZEN_LAMBDA


def check_selection(selection_csv: Path) -> set[str]:
    if not selection_csv.exists():
        fail(f"Missing frozen formal TEST selection: {selection_csv}")
    sel = pd.read_csv(selection_csv, dtype=str)
    if "case_id" not in sel.columns:
        fail(f"'case_id' column missing from {selection_csv}")

    ids = sel["case_id"].astype(str).str.strip()
    unique_ids = set(ids)
    if len(unique_ids) != EXPECTED_SELECTED_CASES:
        fail(
            f"Frozen formal selection must contain {EXPECTED_SELECTED_CASES} unique cases; "
            f"found {len(unique_ids)}."
        )
    return unique_ids


def check_optional_manifest(path: Path, expected_rows: int, label: str) -> int | None:
    if not path.exists():
        print(f"[WARN] {label} not found; Phase 4D-2 validator remains the completeness gate: {path}")
        return None
    df = pd.read_csv(path)
    n = len(df)
    if n != expected_rows:
        fail(f"{label} row count mismatch: expected {expected_rows}, found {n}.")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_dir",
        default="results/stage68_outcome_proxy_masking/explanation_bank",
    )
    ap.add_argument(
        "--selection_csv",
        default="results/phase4/phase4d_formal_test/formal_test_case_selection.csv",
    )
    ap.add_argument(
        "--frozen_config",
        default="configs/phase4c_ds_frozen_config.json",
    )
    ap.add_argument(
        "--out_dir",
        default="results/stage68_outcome_proxy_masking/group_audit",
    )
    args = ap.parse_args()

    base = Path(args.base_dir)
    rules_path = base / "canonical_rules_test.jsonl"
    validation_path = base / "validation_stage68b_masked_multirule.json"
    bank_summary_path = base / "multi_rule_summary_test.json"
    neighborhood_manifest_path = base / "neighborhood_manifest_test.csv"
    surrogate_metrics_path = base / "surrogate_metrics_test.csv"

    selection_csv = Path(args.selection_csv)
    frozen_config_path = Path(args.frozen_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 108)
    print("ETCE STAGE 68C MASKED TARGET-LOCAL GROUPING AUDIT")
    print("=" * 108)

    # ------------------------------------------------------------------
    # 1) Frozen formal TEST protocol gates
    # ------------------------------------------------------------------
    selected_cases = check_selection(selection_csv)
    print(f"[Gate] Frozen selected TEST cases: {len(selected_cases)}")

    validation_obj = load_json(validation_path)
    check_validator(validation_obj)
    if validation_obj.get("status") != "passed":
        fail("Stage 68B masked bank validation is not PASSED")
    print("[Gate] Stage 68B masked bank validation: PASSED")

    bank_summary = load_json(bank_summary_path)
    if int(bank_summary.get("target_count", -1)) != EXPECTED_FORMAL_TARGETS:
        fail("Stage 68B target count mismatch in bank summary")
    expected_canonical_rules = int(bank_summary["total_canonical_rule_count"])
    expected_eligible_cf = int(bank_summary["eligible_counterfactual_candidate_count"])

    frozen_cfg = load_json(frozen_config_path)
    frozen_lambda = extract_frozen_lambda(frozen_cfg)
    print(f"[Gate] Frozen lambda: {int(frozen_lambda)}")

    manifest_targets = check_optional_manifest(
        neighborhood_manifest_path,
        EXPECTED_FORMAL_TARGETS,
        "Formal neighborhood manifest",
    )
    if manifest_targets is not None:
        print(f"[Gate] Formal targets from neighborhood manifest: {manifest_targets}")

    metric_groups = check_optional_manifest(
        surrogate_metrics_path,
        EXPECTED_TARGET_SURROGATE_GROUPS,
        "Formal surrogate metrics",
    )
    if metric_groups is not None:
        print(f"[Gate] Target-surrogate groups from surrogate metrics: {metric_groups}")

    # ------------------------------------------------------------------
    # 2) Canonical rule bank gates
    # ------------------------------------------------------------------
    df = load_jsonl(rules_path)

    required = {
        "case_id", "model", "prefix", "surrogate", "candidate_id",
        "rule_type", "predicted_class", "conditions",
        "eligible_for_reliability"
    }
    missing = required - set(df.columns)
    if missing:
        fail(f"Missing canonical-rule fields: {sorted(missing)}")

    if len(df) != expected_canonical_rules:
        fail(
            f"Canonical rule count mismatch: expected {expected_canonical_rules}, "
            f"found {len(df)}."
        )

    df["case_id"] = df["case_id"].astype(str).str.strip()
    df["model"] = df["model"].astype(str)
    df["surrogate"] = df["surrogate"].astype(str)
    df["prefix"] = pd.to_numeric(df["prefix"], errors="raise").astype(int)

    represented_rule_cases = set(df["case_id"])
    extra_rule_cases = represented_rule_cases - selected_cases
    if extra_rule_cases:
        fail(
            "Canonical rule bank contains case IDs outside the frozen formal selection: "
            f"{sorted(extra_rule_cases)[:20]}"
        )

    missing_rule_cases = sorted(selected_cases - represented_rule_cases)

    cf = df[
        (df["rule_type"].astype(str) == "counterfactual")
        & (df["eligible_for_reliability"] == True)  # noqa: E712
    ].copy()

    if len(cf) != expected_eligible_cf:
        fail(
            f"Eligible CF count mismatch: expected {expected_eligible_cf}, found {len(cf)}."
        )

    if not set(cf["model"]).issubset(EXPECTED_MODELS):
        fail(f"Unexpected black-box model(s): {sorted(set(cf['model']) - EXPECTED_MODELS)}")
    if not set(cf["surrogate"]).issubset(EXPECTED_SURROGATES):
        fail(f"Unexpected surrogate(s): {sorted(set(cf['surrogate']) - EXPECTED_SURROGATES)}")
    if not set(cf["prefix"]).issubset(EXPECTED_PREFIXES):
        fail(f"Unexpected prefix(es): {sorted(set(cf['prefix']) - EXPECTED_PREFIXES)}")

    print(f"[Bank] Canonical rules: {len(df)}")
    print(f"[Bank] Eligible CF candidates: {len(cf)}")
    print(f"[Bank] Frozen cases represented by >=1 canonical rule: {len(represented_rule_cases)}")
    print(f"[Bank] Frozen cases with no canonical rule: {len(missing_rule_cases)}")
    if missing_rule_cases:
        print("       " + ", ".join(missing_rule_cases))

    # ------------------------------------------------------------------
    # 3) Frozen compatibility signature
    # ------------------------------------------------------------------
    cf["feature_signature"] = cf["conditions"].apply(feature_signature)
    cf["feature_signature_str"] = cf["feature_signature"].apply(signature_to_str)

    # ------------------------------------------------------------------
    # 4) Stage 1
    # case × model × prefix × surrogate × class × feature signature
    # ------------------------------------------------------------------
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
    s1 = s1.sort_values(s1_keys, kind="mergesort").reset_index(drop=True)

    s1_groups = int(len(s1))
    s1_fusable = int(s1["stage1_fusable"].sum())
    s1_max = int(s1["rule_count"].max()) if len(s1) else 0
    s1_rules_in_fusable = int(
        s1.loc[s1["stage1_fusable"], "rule_count"].sum()
    )

    if s1_fusable == 0 or s1_max < 2:
        fail(
            "Stage 1 is degenerate on formal TEST. "
            "Protocol is frozen; do not retune grouping or lambda."
        )

    by_surrogate_rows = []
    for sur in ["LORE", "DT", "BN"]:
        x = s1[s1["surrogate"] == sur]
        by_surrogate_rows.append({
            "surrogate": sur,
            "group_count": int(len(x)),
            "fusable_group_count_ge2_rules": int(x["stage1_fusable"].sum()),
            "max_group_size": int(x["rule_count"].max()) if len(x) else 0,
            "source_rules_in_fusable_groups": int(
                x.loc[x["stage1_fusable"], "rule_count"].sum()
            ),
        })
    by_surrogate = pd.DataFrame(by_surrogate_rows)

    # ------------------------------------------------------------------
    # 5) Stage 2
    # case × model × prefix × class × feature signature
    # Each Stage-1 group is one surrogate evidence unit.
    # ------------------------------------------------------------------
    s2_keys = [
        "case_id", "model", "prefix",
        "predicted_class", "feature_signature_str"
    ]
    s2 = (
        s1.groupby(s2_keys, dropna=False)
        .agg(
            surrogate_count=("surrogate", "nunique"),
            stage1_group_count=("surrogate", "size"),
            source_rule_count=("rule_count", "sum"),
        )
        .reset_index()
    )
    s2["stage2_fusable"] = s2["surrogate_count"] >= 2
    s2 = s2.sort_values(s2_keys, kind="mergesort").reset_index(drop=True)

    s2_sets = int(len(s2))
    s2_ge2 = int(s2["stage2_fusable"].sum())
    s2_all3 = int((s2["surrogate_count"] == 3).sum())

    # ------------------------------------------------------------------
    # 6) Stage 3
    # case × prefix × class × feature signature
    # Each Stage-2 set is one black-box evidence unit.
    # ------------------------------------------------------------------
    s3_keys = [
        "case_id", "prefix",
        "predicted_class", "feature_signature_str"
    ]
    s3 = (
        s2.groupby(s3_keys, dropna=False)
        .agg(
            blackbox_count=("model", "nunique"),
            model_group_count=("model", "size"),
            source_rule_count=("source_rule_count", "sum"),
        )
        .reset_index()
    )
    s3["stage3_fusable"] = s3["blackbox_count"] >= 2
    s3 = s3.sort_values(s3_keys, kind="mergesort").reset_index(drop=True)

    s3_sets = int(len(s3))
    s3_ge2 = int(s3["stage3_fusable"].sum())
    s3_ge3 = int((s3["blackbox_count"] >= 3).sum())
    s3_all4 = int((s3["blackbox_count"] == 4).sum())

    # ------------------------------------------------------------------
    # 7) Save audit artifacts
    # ------------------------------------------------------------------
    s1.to_csv(out_dir / "stage1_target_local_inventory_test.csv", index=False)
    by_surrogate.to_csv(out_dir / "stage1_surrogate_breakdown_test.csv", index=False)
    s2.to_csv(out_dir / "stage2_same_target_inter_surrogate_test.csv", index=False)
    s3.to_csv(out_dir / "stage3_same_case_inter_blackbox_test.csv", index=False)

    summary = {
        "scope": "ETCE Stage 68C masked target-local grouping audit",
        "status": "passed",
        "outcome_proxy_masking_sensitivity": True,
        "formal_test_only": True,
        "frozen_selected_case_count": len(selected_cases),
        "formal_target_count": manifest_targets if manifest_targets is not None else EXPECTED_FORMAL_TARGETS,
        "target_surrogate_group_count": metric_groups if metric_groups is not None else EXPECTED_TARGET_SURROGATE_GROUPS,
        "canonical_rule_count": len(df),
        "eligible_cf_candidate_count": len(cf),
        "canonical_rule_bank_case_count": len(represented_rule_cases),
        "selected_cases_without_canonical_rule": missing_rule_cases,
        "frozen_lambda": 5,
        "protocol_retuning_performed": False,
        "lambda_selection_performed": False,
        "ds_fusion_performed": False,
        "stage1": {
            "group_count": s1_groups,
            "fusable_group_count_ge2_rules": s1_fusable,
            "max_group_size": s1_max,
            "source_rules_in_fusable_groups": s1_rules_in_fusable,
            "status": "NON-DEGENERATE",
            "by_surrogate": by_surrogate.to_dict(orient="records"),
        },
        "stage2": {
            "compatible_sets": s2_sets,
            "sets_with_ge2_surrogates": s2_ge2,
            "sets_with_all_3_surrogates": s2_all3,
        },
        "stage3": {
            "compatible_sets": s3_sets,
            "sets_with_ge2_blackboxes": s3_ge2,
            "sets_with_ge3_blackboxes": s3_ge3,
            "sets_with_all_4_blackboxes": s3_all4,
        },
        "prefixes": sorted(EXPECTED_PREFIXES),
        "next": "Stage 68D frozen D-S masked sensitivity fusion",
    }

    summary_path = out_dir / "masked_target_local_group_audit.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 8) Terminal report
    # ------------------------------------------------------------------
    print()
    print("[Stage 1: same target-local surrogate]")
    print(f"Groups: {s1_groups}")
    print(f"Groups with >=2 rules: {s1_fusable}")
    print(f"Max group size: {s1_max}")
    print(f"Source rules in fusable groups: {s1_rules_in_fusable}")
    for row in by_surrogate_rows:
        print(
            f"  {row['surrogate']}: "
            f"fusable={row['fusable_group_count_ge2_rules']}, "
            f"max={row['max_group_size']}, "
            f"rules={row['source_rules_in_fusable_groups']}"
        )
    print("Stage 1 status: NON-DEGENERATE")

    print()
    print("[Stage 2: across surrogates for same target/black-box/prefix]")
    print(f"Compatible sets: {s2_sets}")
    print(f"Sets with >=2 surrogates: {s2_ge2}")
    print(f"Sets with all 3 surrogates: {s2_all3}")

    print()
    print("[Stage 3: across black-boxes for same case/prefix]")
    print(f"Compatible sets: {s3_sets}")
    print(f"Sets with >=2 black-boxes: {s3_ge2}")
    print(f"Sets with >=3 black-boxes: {s3_ge3}")
    print(f"Sets with all 4 black-boxes: {s3_all4}")

    print()
    print("=" * 108)
    print("ETCE STAGE 68C MASKED TARGET-LOCAL GROUPING AUDIT PASSED")
    print("=" * 108)
    print(f"Frozen selected TEST cases: {len(selected_cases)}")
    print(f"Formal targets: {manifest_targets if manifest_targets is not None else EXPECTED_FORMAL_TARGETS}")
    print(f"Target-surrogate groups: {metric_groups if metric_groups is not None else EXPECTED_TARGET_SURROGATE_GROUPS}")
    print(f"Canonical rules: {len(df)}")
    print(f"Eligible CF candidates: {len(cf)}")
    print(f"Cases represented in canonical rule bank: {len(represented_rule_cases)}")
    print(f"Selected cases with no canonical rule: {len(missing_rule_cases)}")
    print("Frozen lambda: 5")
    print("Protocol retuning performed: False")
    print("Lambda selection performed: False")
    print("D-S fusion performed: False")
    print(f"Audit JSON: {summary_path}")
    print()
    print("NEXT: Stage 68D frozen D-S masked sensitivity fusion.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("=" * 108, file=sys.stderr)
        print("ETCE STAGE 68C MASKED TARGET-LOCAL GROUPING AUDIT FAILED", file=sys.stderr)
        print("=" * 108, file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise
