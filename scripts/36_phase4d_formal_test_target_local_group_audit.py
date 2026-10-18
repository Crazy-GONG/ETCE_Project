#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-3: Formal TEST target-local grouping audit.

Purpose
-------
Audit the already-frozen three-stage ETCE target-local grouping hierarchy
on the formal TEST multi-rule bank.

Frozen hierarchy
----------------
Stage 1:
    case_id × model × prefix × surrogate × predicted_class × feature_signature

Stage 2:
    case_id × model × prefix × predicted_class × feature_signature

Stage 3:
    case_id × prefix × predicted_class × feature_signature

Important protocol constraints
------------------------------
- Formal TEST inventory/audit only.
- No lambda selection.
- No reliability/BPA retuning.
- No D-S fusion.
- No modification of the frozen Phase 4C-v2 protocol.
- Frozen lambda must remain 5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_CASES = 100
EXPECTED_FORMAL_TARGETS = 1600
EXPECTED_TARGET_SURROGATE_GROUPS = 4800
EXPECTED_CANONICAL_RULES = 9846
EXPECTED_ELIGIBLE_CF = 6172

EXPECTED_PREFIXES = {5, 10, 15, 20}
EXPECTED_MODELS = {"RF", "XGBoost", "MLP", "LSTM"}
EXPECTED_SURROGATES = {"LORE", "DT", "BN"}
FROZEN_LAMBDA = 5.0


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def load_json(path: Path) -> Any:
    if not path.exists():
        fail(f"Required file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Rule bank does not exist: {path}")
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
    """
    Frozen compatibility signature used by the earlier target-local audit:
    sorted unique `position` values appearing in the rule conditions.
    """
    if not isinstance(conditions, list):
        return tuple()

    pos = []
    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        p = cond.get("position")
        if p is None:
            continue
        try:
            pos.append(int(p))
        except Exception as exc:
            fail(f"Non-integer condition position encountered: {p!r} ({exc})")
    return tuple(sorted(set(pos)))


def signature_to_str(sig: tuple[int, ...]) -> str:
    return "[" + ",".join(map(str, sig)) + "]"


def recursive_find_values(obj: Any, accepted_keys: set[str]) -> list[Any]:
    vals = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            nk = str(k).strip().lower().replace("-", "_").replace(" ", "_")
            if nk in accepted_keys:
                vals.append(v)
            vals.extend(recursive_find_values(v, accepted_keys))
    elif isinstance(obj, list):
        for x in obj:
            vals.extend(recursive_find_values(x, accepted_keys))
    return vals


def extract_frozen_lambda(cfg: Any) -> float:
    keys = {
        "lambda",
        "lambda_r",
        "selected_lambda",
        "frozen_lambda",
        "reliability_lambda",
        "λ",
    }
    vals = recursive_find_values(cfg, keys)

    numeric = []
    for v in vals:
        try:
            numeric.append(float(v))
        except Exception:
            pass

    if not numeric:
        fail("Could not find a numeric frozen lambda in the frozen D-S config.")

    unique = sorted(set(numeric))
    if FROZEN_LAMBDA not in unique:
        fail(f"Frozen config does not contain lambda=5. Numeric lambda-like values: {unique}")

    non5 = [x for x in unique if abs(x - FROZEN_LAMBDA) > 1e-12]
    if non5:
        print(
            f"[WARN] Other lambda-like values also appear in config: {non5}. "
            "The frozen selected lambda=5 is present and will remain unchanged."
        )
    return FROZEN_LAMBDA


def validation_pass_hint(obj: Any) -> tuple[bool | None, list[str]]:
    """
    Best-effort read of the Phase 4D-2 validator JSON. We do not assume one
    exact schema, because the independent validator may store status under a
    nested field. Returns:
        True  -> explicit PASS-like status found
        False -> explicit FAIL-like status found
        None  -> no explicit status token found
    """
    tokens = []
    candidate_keys = {
        "status", "validation_status", "overall_status", "result",
        "passed", "validation_passed", "all_passed"
    }
    vals = recursive_find_values(obj, candidate_keys)
    for v in vals:
        tokens.append(str(v))

    lowered = [x.strip().lower() for x in tokens]
    fail_tokens = {"fail", "failed", "false", "0", "error"}
    pass_tokens = {"pass", "passed", "true", "1", "ok", "success", "validated"}

    if any(x in fail_tokens for x in lowered):
        return False, tokens
    if any(x in pass_tokens for x in lowered):
        return True, tokens
    return None, tokens


def audit_selection(selection_csv: Path, bank_case_ids: set[str]) -> dict[str, Any]:
    if not selection_csv.exists():
        fail(f"Formal TEST case selection CSV does not exist: {selection_csv}")

    sel = pd.read_csv(selection_csv, dtype=str)
    if "case_id" not in sel.columns:
        fail(f"{selection_csv} has no 'case_id' column.")

    selected = set(sel["case_id"].astype(str))
    if len(selected) != EXPECTED_CASES:
        fail(
            f"Formal TEST selection must contain {EXPECTED_CASES} unique case_id values; "
            f"found {len(selected)}."
        )

    if selected != bank_case_ids:
        missing_in_bank = sorted(selected - bank_case_ids)[:10]
        extra_in_bank = sorted(bank_case_ids - selected)[:10]
        fail(
            "Rule-bank case IDs do not exactly match the frozen formal TEST selection. "
            f"Missing in bank (first 10): {missing_in_bank}; "
            f"extra in bank (first 10): {extra_in_bank}"
        )

    return {
        "selection_rows": int(len(sel)),
        "unique_selected_cases": int(len(selected)),
        "case_set_matches_rule_bank": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ETCE Phase 4D-3 formal TEST target-local grouping audit."
    )
    ap.add_argument(
        "--rules",
        default=(
            "results/phase4/phase4d_formal_test/"
            "formal_test_multirule_bank/canonical_rules_test.jsonl"
        ),
    )
    ap.add_argument(
        "--validation",
        default=(
            "results/phase4/phase4d_formal_test/"
            "formal_test_multirule_bank/"
            "validation_phase4d_formal_test_multirule.json"
        ),
    )
    ap.add_argument(
        "--selection_csv",
        default=(
            "results/phase4/phase4d_formal_test/"
            "formal_test_case_selection.csv"
        ),
    )
    ap.add_argument(
        "--frozen_config",
        default="configs/phase4c_ds_frozen_config.json",
    )
    ap.add_argument(
        "--out_dir",
        default=(
            "results/phase4/phase4d_formal_test/"
            "formal_test_target_local_group_audit"
        ),
    )
    args = ap.parse_args()

    rules_path = Path(args.rules)
    validation_path = Path(args.validation)
    selection_csv = Path(args.selection_csv)
    frozen_config_path = Path(args.frozen_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Gate 0: path/protocol identity
    # ------------------------------------------------------------------
    normalized_rules = str(rules_path).replace("\\", "/").lower()
    if "phase4d_formal_test" not in normalized_rules:
        fail("Refusing to run: --rules is not under the Phase 4D formal TEST tree.")
    if rules_path.name != "canonical_rules_test.jsonl":
        fail(
            "Refusing to run: expected the formal TEST rule bank "
            "'canonical_rules_test.jsonl'."
        )

    validation_obj = load_json(validation_path)
    pass_hint, pass_tokens = validation_pass_hint(validation_obj)
    if pass_hint is False:
        fail(
            "The Phase 4D-2 validation JSON contains an explicit FAIL-like status: "
            f"{pass_tokens}"
        )
    if pass_hint is None:
        print(
            "[WARN] No explicit PASS token was recognized in the validation JSON; "
            "continuing with strict independent structural gates below."
        )

    frozen_cfg = load_json(frozen_config_path)
    frozen_lambda = extract_frozen_lambda(frozen_cfg)
    if abs(frozen_lambda - FROZEN_LAMBDA) > 1e-12:
        fail(f"Frozen lambda mismatch: expected 5, found {frozen_lambda}")

    # ------------------------------------------------------------------
    # Gate 1: formal TEST bank identity and completeness
    # ------------------------------------------------------------------
    df = load_jsonl(rules_path)

    required = {
        "case_id",
        "model",
        "prefix",
        "surrogate",
        "candidate_id",
        "rule_type",
        "predicted_class",
        "conditions",
        "eligible_for_reliability",
    }
    missing = required - set(df.columns)
    if missing:
        fail(f"Missing required canonical-rule fields: {sorted(missing)}")

    if len(df) != EXPECTED_CANONICAL_RULES:
        fail(
            f"Canonical rule count mismatch: expected {EXPECTED_CANONICAL_RULES}, "
            f"found {len(df)}."
        )

    # Normalize identity columns only for audit consistency.
    df["case_id"] = df["case_id"].astype(str)
    df["model"] = df["model"].astype(str)
    df["surrogate"] = df["surrogate"].astype(str)
    df["prefix"] = pd.to_numeric(df["prefix"], errors="raise").astype(int)

    cases = set(df["case_id"])
    prefixes = set(df["prefix"])
    models = set(df["model"])
    surrogates = set(df["surrogate"])

    if len(cases) != EXPECTED_CASES:
        fail(f"Expected {EXPECTED_CASES} formal TEST cases, found {len(cases)}.")
    if prefixes != EXPECTED_PREFIXES:
        fail(f"Prefix set mismatch: expected {sorted(EXPECTED_PREFIXES)}, found {sorted(prefixes)}.")
    if models != EXPECTED_MODELS:
        fail(f"Model set mismatch: expected {sorted(EXPECTED_MODELS)}, found {sorted(models)}.")
    if surrogates != EXPECTED_SURROGATES:
        fail(
            f"Surrogate set mismatch: expected {sorted(EXPECTED_SURROGATES)}, "
            f"found {sorted(surrogates)}."
        )

    formal_targets = df[["case_id", "model", "prefix"]].drop_duplicates()
    if len(formal_targets) != EXPECTED_FORMAL_TARGETS:
        fail(
            f"Formal target count mismatch: expected {EXPECTED_FORMAL_TARGETS}, "
            f"found {len(formal_targets)}."
        )

    target_surrogate = df[
        ["case_id", "model", "prefix", "surrogate"]
    ].drop_duplicates()
    if len(target_surrogate) != EXPECTED_TARGET_SURROGATE_GROUPS:
        fail(
            "Target-surrogate group count mismatch: expected "
            f"{EXPECTED_TARGET_SURROGATE_GROUPS}, found {len(target_surrogate)}."
        )

    selection_audit = audit_selection(selection_csv, cases)

    # ------------------------------------------------------------------
    # Gate 2: use eligible counterfactual candidates only
    # ------------------------------------------------------------------
    cf = df[
        (df["rule_type"].astype(str) == "counterfactual")
        & (df["eligible_for_reliability"] == True)  # noqa: E712
    ].copy()

    if len(cf) != EXPECTED_ELIGIBLE_CF:
        fail(
            f"Eligible CF count mismatch: expected {EXPECTED_ELIGIBLE_CF}, "
            f"found {len(cf)}."
        )

    cf["feature_signature"] = cf["conditions"].apply(feature_signature)
    cf["feature_signature_str"] = cf["feature_signature"].apply(signature_to_str)

    # Stable unique rule identity check.
    rule_id_cols = [
        "case_id", "model", "prefix", "surrogate", "candidate_id", "rule_type"
    ]
    dup_rule_ids = int(cf.duplicated(rule_id_cols, keep=False).sum())
    if dup_rule_ids:
        fail(
            "Duplicate eligible CF rule identity detected under "
            f"{rule_id_cols}: duplicated rows={dup_rule_ids}"
        )

    # ------------------------------------------------------------------
    # Stage 1:
    # same fitted target-local surrogate + class + feature signature
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
        .size()
        .reset_index(name="rule_count")
    )
    s1["stage1_fusable"] = s1["rule_count"] >= 2
    s1 = s1.sort_values(
        ["case_id", "model", "prefix", "surrogate",
         "predicted_class", "feature_signature_str"],
        kind="mergesort",
    ).reset_index(drop=True)

    stage1_fusable_groups = int(s1["stage1_fusable"].sum())
    stage1_max_group = int(s1["rule_count"].max()) if len(s1) else 0
    stage1_fusable_source_rules = int(
        s1.loc[s1["stage1_fusable"], "rule_count"].sum()
    )

    if stage1_fusable_groups <= 0 or stage1_max_group < 2:
        fail(
            "Stage 1 is degenerate on the formal TEST bank. "
            "This audit does NOT permit protocol redesign or retuning."
        )

    per_surrogate_rows = []
    for surrogate in ["LORE", "DT", "BN"]:
        x = s1[s1["surrogate"] == surrogate]
        per_surrogate_rows.append({
            "surrogate": surrogate,
            "group_count": int(len(x)),
            "fusable_group_count_ge2_rules": int(x["stage1_fusable"].sum()),
            "max_group_size": int(x["rule_count"].max()) if len(x) else 0,
            "source_rules_in_fusable_groups": int(
                x.loc[x["stage1_fusable"], "rule_count"].sum()
            ),
        })
    per_surrogate = pd.DataFrame(per_surrogate_rows)

    # ------------------------------------------------------------------
    # Stage 2:
    # same target + black-box + class + feature signature across surrogates.
    #
    # Each Stage-1 compatible group acts as one surrogate evidence unit.
    # ------------------------------------------------------------------
    s2_keys = [
        "case_id",
        "model",
        "prefix",
        "predicted_class",
        "feature_signature_str",
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
    s2 = s2.sort_values(
        ["case_id", "model", "prefix",
         "predicted_class", "feature_signature_str"],
        kind="mergesort",
    ).reset_index(drop=True)

    stage2_ge2 = int(s2["stage2_fusable"].sum())
    stage2_all3 = int((s2["surrogate_count"] == 3).sum())
    stage2_source_rules = int(
        s2.loc[s2["stage2_fusable"], "source_rule_count"].sum()
    )

    # ------------------------------------------------------------------
    # Stage 3:
    # same case + prefix + class + feature signature across black-boxes.
    #
    # Each Stage-2 compatible set acts as one black-box evidence unit.
    # ------------------------------------------------------------------
    s3_keys = [
        "case_id",
        "prefix",
        "predicted_class",
        "feature_signature_str",
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
    s3 = s3.sort_values(
        ["case_id", "prefix", "predicted_class", "feature_signature_str"],
        kind="mergesort",
    ).reset_index(drop=True)

    stage3_ge2 = int(s3["stage3_fusable"].sum())
    stage3_ge3 = int((s3["blackbox_count"] >= 3).sum())
    stage3_all4 = int((s3["blackbox_count"] == 4).sum())
    stage3_source_rules = int(
        s3.loc[s3["stage3_fusable"], "source_rule_count"].sum()
    )

    # ------------------------------------------------------------------
    # Persist deterministic audit artifacts
    # ------------------------------------------------------------------
    s1.to_csv(out_dir / "stage1_target_local_inventory_test.csv", index=False)
    per_surrogate.to_csv(
        out_dir / "stage1_surrogate_breakdown_test.csv", index=False
    )
    s2.to_csv(
        out_dir / "stage2_same_target_inter_surrogate_test.csv", index=False
    )
    s3.to_csv(
        out_dir / "stage3_same_case_inter_blackbox_test.csv", index=False
    )

    summary = {
        "scope": "ETCE Phase 4D-3 formal TEST target-local grouping audit",
        "status": "passed",
        "formal_test_only": True,
        "phase4d_validator_file": str(validation_path),
        "formal_selection_file": str(selection_csv),
        "formal_rule_bank": str(rules_path),
        "frozen_config": str(frozen_config_path),
        "frozen_lambda": 5,
        "protocol_retuning_performed": False,
        "lambda_selection_performed": False,
        "ds_fusion_performed": False,
        "selection_audit": selection_audit,
        "bank_identity": {
            "formal_cases": int(len(cases)),
            "formal_targets": int(len(formal_targets)),
            "target_surrogate_groups": int(len(target_surrogate)),
            "canonical_rules": int(len(df)),
            "eligible_cf_candidates": int(len(cf)),
            "prefixes": sorted(int(x) for x in prefixes),
            "models": sorted(models),
            "surrogates": sorted(surrogates),
        },
        "stage1": {
            "group_count": int(len(s1)),
            "fusable_group_count_ge2_rules": stage1_fusable_groups,
            "max_group_size": stage1_max_group,
            "source_rules_in_fusable_groups": stage1_fusable_source_rules,
            "status": "NON-DEGENERATE",
            "by_surrogate": per_surrogate.to_dict(orient="records"),
        },
        "stage2": {
            "compatible_sets": int(len(s2)),
            "sets_with_ge2_surrogates": stage2_ge2,
            "sets_with_all_3_surrogates": stage2_all3,
            "source_rules_in_fusable_sets": stage2_source_rules,
        },
        "stage3": {
            "compatible_sets": int(len(s3)),
            "sets_with_ge2_blackboxes": stage3_ge2,
            "sets_with_ge3_blackboxes": stage3_ge3,
            "sets_with_all_4_blackboxes": stage3_all4,
            "source_rules_in_fusable_sets": stage3_source_rules,
        },
        "next": "Phase 4D-4 frozen D-S formal TEST fusion",
    }

    summary_path = out_dir / "formal_test_target_local_group_audit.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report_lines = [
        "# ETCE Phase 4D-3 Formal TEST Target-local Grouping Audit",
        "",
        "- Status: **PASSED**",
        "- Formal TEST only: **True**",
        "- Frozen lambda: **5**",
        "- Protocol retuning performed: **False**",
        "- D-S fusion performed: **False**",
        "",
        "## Bank identity",
        f"- Formal cases: {len(cases)}",
        f"- Formal targets: {len(formal_targets)}",
        f"- Target-surrogate groups: {len(target_surrogate)}",
        f"- Canonical rules: {len(df)}",
        f"- Eligible CF candidates: {len(cf)}",
        "",
        "## Stage 1",
        f"- Groups: {len(s1)}",
        f"- Groups with >=2 rules: {stage1_fusable_groups}",
        f"- Max group size: {stage1_max_group}",
        f"- Source rules in fusable groups: {stage1_fusable_source_rules}",
        "- Status: **NON-DEGENERATE**",
        "",
        "### By surrogate",
    ]
    for row in per_surrogate_rows:
        report_lines.append(
            f"- {row['surrogate']}: "
            f"fusable={row['fusable_group_count_ge2_rules']}, "
            f"max={row['max_group_size']}, "
            f"rules={row['source_rules_in_fusable_groups']}"
        )
    report_lines.extend([
        "",
        "## Stage 2",
        f"- Compatible sets: {len(s2)}",
        f"- Sets with >=2 surrogates: {stage2_ge2}",
        f"- All 3 surrogates: {stage2_all3}",
        "",
        "## Stage 3",
        f"- Compatible sets: {len(s3)}",
        f"- Sets with >=2 black-boxes: {stage3_ge2}",
        f"- Sets with >=3 black-boxes: {stage3_ge3}",
        f"- All 4 black-boxes: {stage3_all4}",
        "",
        "## Protocol statement",
        "This phase is an inventory/audit only. No lambda selection, "
        "no BPA/reliability retuning, and no D-S fusion were performed.",
        "",
        "NEXT: frozen D-S formal TEST fusion.",
    ])
    (out_dir / "PHASE4D_FORMAL_TEST_GROUPING_AUDIT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Terminal report
    # ------------------------------------------------------------------
    print("=" * 108)
    print("ETCE PHASE 4D-3 FORMAL TEST TARGET-LOCAL GROUPING AUDIT PASSED")
    print("=" * 108)
    print(f"Formal TEST cases: {len(cases)}")
    print(f"Formal targets: {len(formal_targets)}")
    print(f"Target-surrogate groups: {len(target_surrogate)}")
    print(f"Canonical rules: {len(df)}")
    print(f"Eligible CF candidates: {len(cf)}")
    print()
    print("[Stage 1: within SAME target-local surrogate instance]")
    print(f"Groups: {len(s1)}")
    print(f"Groups with >=2 rules: {stage1_fusable_groups}")
    print(f"Max group size: {stage1_max_group}")
    print(f"Source rules in fusable groups: {stage1_fusable_source_rules}")
    for row in per_surrogate_rows:
        print(
            f"  {row['surrogate']}: "
            f"fusable={row['fusable_group_count_ge2_rules']}, "
            f"max={row['max_group_size']}, "
            f"rules={row['source_rules_in_fusable_groups']}"
        )
    print("Stage 1 status: NON-DEGENERATE")
    print()
    print("[Stage 2: across surrogates for SAME target/black-box/prefix]")
    print(f"Compatible sets: {len(s2)}")
    print(f"Sets with >=2 surrogates: {stage2_ge2}")
    print(f"Sets with all 3 surrogates: {stage2_all3}")
    print()
    print("[Stage 3: across black-boxes for SAME case/prefix]")
    print(f"Compatible sets: {len(s3)}")
    print(f"Sets with >=2 black-boxes: {stage3_ge2}")
    print(f"Sets with >=3 black-boxes: {stage3_ge3}")
    print(f"Sets with all 4 black-boxes: {stage3_all4}")
    print()
    print("Frozen lambda: 5")
    print("Protocol retuning performed: False")
    print("Lambda selection performed: False")
    print("D-S fusion performed: False")
    print(f"Audit JSON: {summary_path}")
    print()
    print("NEXT: frozen D-S formal TEST fusion.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("=" * 108, file=sys.stderr)
        print("ETCE PHASE 4D-3 FORMAL TEST TARGET-LOCAL GROUPING AUDIT FAILED", file=sys.stderr)
        print("=" * 108, file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise
