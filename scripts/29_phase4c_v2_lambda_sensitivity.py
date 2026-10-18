#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4C-v2: validation-only lambda sensitivity on the validated
Phase 4B-v2 multi-rule bank.

Frozen reliability/BPA formula
------------------------------
rho_r = n_r / (n_r + lambda)
R(r)  = rho_r * Fid(r)

m_r(theta_r) = R(r)
m_r(Theta)   = 1 - R(r)

Important
---------
- Uses ONLY the validation multi-rule bank.
- Requires Phase 4B-v2 validator = PASSED.
- Requires Phase 4C-v2 target-local audit = NON-DEGENERATE.
- Does NOT read formal test data.
- Does NOT automatically freeze/select lambda.
- Candidate grid remains {1, 5, 10, 20, 50} unless explicitly overridden.

Run:
    python -u scripts/29_phase4c_v2_lambda_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
DEFAULT_GROUP_AUDIT = (
    DEFAULT_ROOT
    / "results/phase4/phase4c_v2_target_local_audit/target_local_group_audit_v2.json"
)
DEFAULT_OUT = (
    DEFAULT_ROOT
    / "results/phase4/phase4c_v2_lambda_sensitivity"
)

DEFAULT_LAMBDAS = [1, 5, 10, 20, 50]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
                    f"Invalid JSON at line {line_no}: {path}"
                ) from exc
    return pd.DataFrame(rows)


def q(values, p):
    s = pd.Series(values)
    if len(s) == 0:
        return float("nan")
    return float(s.quantile(p))


def summarize(sub: pd.DataFrame, lam: float) -> dict:
    n = sub["n_r"].astype(float)
    fid = sub["Fid_r"].astype(float)
    rho = n / (n + float(lam))
    rel = rho * fid

    return {
        "lambda": float(lam),
        "rule_count": int(len(sub)),
        "mean_support": float(n.mean()),
        "median_support": float(n.median()),
        "mean_fidelity": float(fid.mean()),
        "median_fidelity": float(fid.median()),
        "mean_rho": float(rho.mean()),
        "median_rho": float(rho.median()),
        "mean_reliability": float(rel.mean()),
        "median_reliability": float(rel.median()),
        "p10_reliability": q(rel, 0.10),
        "p25_reliability": q(rel, 0.25),
        "p75_reliability": q(rel, 0.75),
        "p90_reliability": q(rel, 0.90),
        "mean_ignorance_mass": float((1.0 - rel).mean()),
        "share_R_le_0_1": float((rel <= 0.1).mean()),
        "share_R_le_0_2": float((rel <= 0.2).mean()),
        "share_R_ge_0_5": float((rel >= 0.5).mean()),
        "share_R_ge_0_7": float((rel >= 0.7).mean()),
        "share_R_ge_0_9": float((rel >= 0.9).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rules",
        default=str(DEFAULT_RULES),
        help="Phase 4B-v2 canonical_rules_calibration.jsonl",
    )
    ap.add_argument(
        "--validation",
        default=str(DEFAULT_VALIDATION),
        help="Phase 4B-v2 validation JSON",
    )
    ap.add_argument(
        "--group_audit",
        default=str(DEFAULT_GROUP_AUDIT),
        help="Phase 4C-v2 target-local group audit JSON",
    )
    ap.add_argument(
        "--out_dir",
        default=str(DEFAULT_OUT),
        help="Output directory",
    )
    ap.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=DEFAULT_LAMBDAS,
    )
    args = ap.parse_args()

    rules_path = Path(args.rules).expanduser().resolve()
    validation_path = Path(args.validation).expanduser().resolve()
    group_audit_path = Path(args.group_audit).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    require(rules_path.is_file(), f"Rules missing: {rules_path}")
    require(
        validation_path.is_file(),
        f"Phase 4B-v2 validation missing: {validation_path}",
    )
    require(
        group_audit_path.is_file(),
        f"Phase 4C-v2 group audit missing: {group_audit_path}",
    )

    validation = read_json(validation_path)
    require(
        validation.get("status") == "passed",
        "Phase 4B-v2 validation is not PASSED",
    )
    require(
        validation.get("test_cases_used") is False,
        "Phase 4B-v2 unexpectedly used test cases",
    )

    audit = read_json(group_audit_path)
    result = audit.get("result", {})
    require(
        result.get("former_single_rule_degeneracy_resolved") is True,
        "Stage-1 degeneracy has not been resolved",
    )
    require(
        result.get("ready_for_lambda_sensitivity") is True,
        "Grouping audit does not permit lambda sensitivity",
    )
    require(
        audit.get("test_data_used") is False,
        "Grouping audit unexpectedly used test data",
    )

    lambdas = [float(v) for v in args.lambdas]
    require(
        all(v > 0 for v in lambdas),
        "All lambda candidates must be > 0",
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(rules_path)
    required = {
        "case_id",
        "model",
        "prefix",
        "surrogate",
        "candidate_id",
        "rule_type",
        "predicted_class",
        "n_r",
        "Fid_r",
        "coverage_eval",
        "eligible_for_reliability",
    }
    missing = sorted(required - set(df.columns))
    require(
        not missing,
        "Missing required fields: " + ", ".join(missing),
    )

    # Defensive identity check for multi-rule records.
    key = [
        "case_id",
        "model",
        "prefix",
        "surrogate",
        "candidate_id",
        "rule_type",
    ]
    dup_count = int(df.duplicated(key, keep=False).sum())
    require(dup_count == 0, "Duplicate canonical candidate key detected")

    eligible = df[
        df["eligible_for_reliability"].astype(bool)
    ].copy()

    factual = eligible[
        eligible["rule_type"] == "factual"
    ].copy()

    cf = eligible[
        eligible["rule_type"] == "counterfactual"
    ].copy()

    require(len(cf) > 0, "No eligible counterfactual candidates found")
    require(
        len(cf) == int(
            validation["eligible_counterfactual_candidate_count"]
        ),
        "Eligible CF count differs from Phase 4B-v2 validator",
    )

    # --------------------------------------------------------------
    # Rule-level reliability/BPA grid.
    # Keep all eligible rules for downstream audit/reproduction.
    # --------------------------------------------------------------
    grid_parts = []
    for lam in lambdas:
        tmp = eligible.copy()
        n = tmp["n_r"].astype(float)
        fid = tmp["Fid_r"].astype(float)

        tmp["lambda"] = lam
        tmp["rho_r"] = n / (n + lam)
        tmp["R_r"] = tmp["rho_r"] * fid
        tmp["mass_theta"] = tmp["R_r"]
        tmp["mass_Theta"] = 1.0 - tmp["R_r"]

        grid_parts.append(tmp)

    grid = pd.concat(grid_parts, ignore_index=True)
    grid.to_csv(
        out_dir / "phase4c_v2_rule_reliability_grid.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Global summaries.
    # --------------------------------------------------------------
    summary_rows = []
    for scope_name, sub in (
        ("all", eligible),
        ("factual", factual),
        ("counterfactual", cf),
    ):
        for lam in lambdas:
            row = summarize(sub, lam)
            row["scope"] = scope_name
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        out_dir / "phase4c_v2_lambda_summary.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Counterfactual sensitivity by surrogate.
    # --------------------------------------------------------------
    by_surrogate = []
    for surrogate, sub in cf.groupby(
        "surrogate",
        dropna=False,
    ):
        for lam in lambdas:
            row = summarize(sub, lam)
            row["surrogate"] = surrogate
            by_surrogate.append(row)

    pd.DataFrame(by_surrogate).to_csv(
        out_dir / "phase4c_v2_lambda_by_surrogate.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Counterfactual sensitivity by model.
    # --------------------------------------------------------------
    by_model = []
    for model, sub in cf.groupby("model", dropna=False):
        for lam in lambdas:
            row = summarize(sub, lam)
            row["model"] = model
            by_model.append(row)

    pd.DataFrame(by_model).to_csv(
        out_dir / "phase4c_v2_lambda_by_model.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Counterfactual sensitivity by prefix.
    # --------------------------------------------------------------
    by_prefix = []
    for prefix, sub in cf.groupby("prefix", dropna=False):
        for lam in lambdas:
            row = summarize(sub, lam)
            row["prefix"] = int(prefix)
            by_prefix.append(row)

    pd.DataFrame(by_prefix).to_csv(
        out_dir / "phase4c_v2_lambda_by_prefix.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Counterfactual sensitivity by model + prefix.
    # --------------------------------------------------------------
    by_mp = []
    for (model, prefix), sub in cf.groupby(
        ["model", "prefix"],
        dropna=False,
    ):
        for lam in lambdas:
            row = summarize(sub, lam)
            row["model"] = model
            row["prefix"] = int(prefix)
            by_mp.append(row)

    pd.DataFrame(by_mp).to_csv(
        out_dir / "phase4c_v2_lambda_by_model_prefix.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Support/fidelity diagnostics for eligible CF candidates.
    # --------------------------------------------------------------
    support = {
        "scope": "Phase 4B-v2 eligible counterfactual candidates",
        "eligible_rule_count_all_types": int(len(eligible)),
        "eligible_factual_rule_count": int(len(factual)),
        "eligible_counterfactual_candidate_count": int(len(cf)),
        "support_min": float(cf["n_r"].min()),
        "support_q10": q(cf["n_r"], 0.10),
        "support_q25": q(cf["n_r"], 0.25),
        "support_median": q(cf["n_r"], 0.50),
        "support_q75": q(cf["n_r"], 0.75),
        "support_q90": q(cf["n_r"], 0.90),
        "support_max": float(cf["n_r"].max()),
        "fidelity_min": float(cf["Fid_r"].min()),
        "fidelity_q25": q(cf["Fid_r"], 0.25),
        "fidelity_median": q(cf["Fid_r"], 0.50),
        "fidelity_q75": q(cf["Fid_r"], 0.75),
        "fidelity_max": float(cf["Fid_r"].max()),
        "lambda_candidates": lambdas,
        "lambda_selected": None,
        "selection_status": "NOT_FROZEN_SENSITIVITY_ONLY",
        "source_phase4b_v2_validation_status": "passed",
        "stage1_non_degenerate": True,
        "test_cases_used": False,
    }

    with (
        out_dir / "phase4c_v2_support_and_grid.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            support,
            f,
            ensure_ascii=False,
            indent=2,
        )

    cf_table = summary[
        summary["scope"] == "counterfactual"
    ][
        [
            "lambda",
            "rule_count",
            "mean_rho",
            "mean_reliability",
            "median_reliability",
            "p25_reliability",
            "p75_reliability",
            "mean_ignorance_mass",
            "share_R_le_0_2",
            "share_R_ge_0_5",
            "share_R_ge_0_7",
        ]
    ].copy()

    # --------------------------------------------------------------
    # Human-readable report.
    # --------------------------------------------------------------
    def pct(x):
        return f"{100.0 * float(x):.2f}%"

    lines = []
    lines.append("# ETCE Phase 4C-v2 Lambda Sensitivity")
    lines.append("")
    lines.append(
        "> Validation-only multi-rule sensitivity analysis. "
        "No formal test data are read and lambda is NOT frozen here."
    )
    lines.append("")
    lines.append("## Preconditions")
    lines.append("")
    lines.append("- Phase 4B-v2 independent validation: PASSED.")
    lines.append("- Target-local Stage 1: NON-DEGENERATE.")
    lines.append(
        f"- Eligible CF candidates: {len(cf)}."
    )
    lines.append("")
    lines.append("## Frozen reliability/BPA formula")
    lines.append("")
    lines.append(r"- $\rho_r=n_r/(n_r+\lambda)$")
    lines.append(r"- $R(r)=\rho_r Fid(r)$")
    lines.append(r"- $m_r(\theta_r)=R(r)$")
    lines.append(r"- $m_r(\Theta)=1-R(r)$")
    lines.append("")
    lines.append("## Eligible CF candidate support")
    lines.append("")
    lines.append(
        f"- N={len(cf)}; "
        f"min={support['support_min']:.0f}; "
        f"Q25={support['support_q25']:.0f}; "
        f"median={support['support_median']:.0f}; "
        f"Q75={support['support_q75']:.0f}; "
        f"Q90={support['support_q90']:.0f}; "
        f"max={support['support_max']:.0f}."
    )
    lines.append("")
    lines.append("## Lambda sensitivity on eligible CF candidates")
    lines.append("")
    lines.append(
        "| lambda | N | mean rho | mean R | median R | "
        "P25 R | P75 R | mean m(Theta) | R<=0.2 | R>=0.5 | R>=0.7 |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for _, r in cf_table.iterrows():
        lines.append(
            f"| {r['lambda']:g} | {int(r['rule_count'])} | "
            f"{r['mean_rho']:.4f} | "
            f"{r['mean_reliability']:.4f} | "
            f"{r['median_reliability']:.4f} | "
            f"{r['p25_reliability']:.4f} | "
            f"{r['p75_reliability']:.4f} | "
            f"{r['mean_ignorance_mass']:.4f} | "
            f"{pct(r['share_R_le_0_2'])} | "
            f"{pct(r['share_R_ge_0_5'])} | "
            f"{pct(r['share_R_ge_0_7'])} |"
        )

    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(
        "This step measures how lambda discounts low-support evidence. "
        "It does not select lambda. The next step must run the validation-only "
        "hierarchical D-S fusion/evaluation for every lambda candidate and "
        "freeze one lambda before any formal test rule bank is inspected."
    )

    (
        out_dir / "PHASE4C_V2_LAMBDA_REPORT.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 100)
    print("ETCE PHASE 4C-v2 - VALIDATION-ONLY LAMBDA SENSITIVITY")
    print("=" * 100)
    print(f"Rules: {rules_path}")
    print("Phase 4B-v2 validation: PASSED")
    print("Stage 1 grouping: NON-DEGENERATE")
    print(f"Eligible all rules: {len(eligible)}")
    print(f"Eligible factual rules: {len(factual)}")
    print(f"Eligible CF candidates: {len(cf)}")
    print()
    print(
        "CF support: "
        f"min={support['support_min']:.0f}, "
        f"q25={support['support_q25']:.0f}, "
        f"median={support['support_median']:.0f}, "
        f"q75={support['support_q75']:.0f}, "
        f"q90={support['support_q90']:.0f}, "
        f"max={support['support_max']:.0f}"
    )
    print()
    print(cf_table.to_string(index=False))
    print()
    print(f"Outputs: {out_dir}")
    print("IMPORTANT: lambda_selected = null")
    print("NEXT: hierarchical D-S validation sweep for lambda selection.")
    print("No formal test data were read.")


if __name__ == "__main__":
    main()
