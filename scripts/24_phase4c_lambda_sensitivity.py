#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4C-A: validation-only lambda sensitivity for support-aware rule reliability.

Frozen formula:
    rho_r = n_r / (n_r + lambda)
    R(r)  = rho_r * Fid(r)
    m_r(theta_r) = R(r)
    m_r(Theta)   = 1 - R(r)

This script DOES NOT inspect test data and DOES NOT freeze lambda automatically.
It produces validation-only sensitivity tables for lambda in {1,5,10,20,50}.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_LAMBDAS = [1, 5, 10, 20, 50]

def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Invalid JSON at line {line_no}: {e}") from e
    return pd.DataFrame(rows)

def q(x, p):
    return float(pd.Series(x).quantile(p)) if len(x) else float("nan")

def summarize(sub: pd.DataFrame, lam: float) -> dict:
    n = sub["n_r"].astype(float)
    fid = sub["Fid_r"].astype(float)
    rho = n / (n + lam)
    rel = rho * fid
    return {
        "lambda": lam,
        "rule_count": int(len(sub)),
        "mean_support": float(n.mean()),
        "median_support": float(n.median()),
        "mean_fidelity": float(fid.mean()),
        "mean_rho": float(rho.mean()),
        "mean_reliability": float(rel.mean()),
        "median_reliability": float(rel.median()),
        "p25_reliability": q(rel, 0.25),
        "p75_reliability": q(rel, 0.75),
        "mean_ignorance_mass": float((1.0 - rel).mean()),
        "share_R_le_0_2": float((rel <= 0.2).mean()),
        "share_R_ge_0_5": float((rel >= 0.5).mean()),
        "share_R_ge_0_7": float((rel >= 0.7).mean()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True, help="canonical_rules_calibration.jsonl")
    ap.add_argument("--out_dir", default="outputs/phase4c_lambda_calibration")
    ap.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    args = ap.parse_args()

    rules_path = Path(args.rules)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(rules_path)

    required = {
        "case_id", "model", "prefix", "surrogate", "rule_type",
        "n_r", "Fid_r", "eligible_for_reliability", "predicted_class"
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required fields: {sorted(missing)}")

    # Keep all rules for audit; reliability is computed only for eligible rules.
    eligible = df[df["eligible_for_reliability"] == True].copy()
    cf = eligible[eligible["rule_type"] == "counterfactual"].copy()
    factual = eligible[eligible["rule_type"] == "factual"].copy()

    if len(cf) == 0:
        raise RuntimeError("No eligible counterfactual rules found.")

    # Rule-level grid
    grid_rows = []
    for lam in args.lambdas:
        tmp = eligible.copy()
        tmp["lambda"] = lam
        tmp["rho_r"] = tmp["n_r"].astype(float) / (tmp["n_r"].astype(float) + lam)
        tmp["R_r"] = tmp["rho_r"] * tmp["Fid_r"].astype(float)
        tmp["mass_theta"] = tmp["R_r"]
        tmp["mass_Theta"] = 1.0 - tmp["R_r"]
        grid_rows.append(tmp)

    grid = pd.concat(grid_rows, ignore_index=True)
    grid.to_csv(out_dir / "phase4c_rule_reliability_grid.csv", index=False)

    # Global summaries: all, factual, counterfactual
    summary_rows = []
    for name, sub in [("all", eligible), ("factual", factual), ("counterfactual", cf)]:
        for lam in args.lambdas:
            row = summarize(sub, lam)
            row["scope"] = name
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "phase4c_lambda_summary.csv", index=False)

    # By surrogate
    by_sur = []
    for surrogate, sub in cf.groupby("surrogate", dropna=False):
        for lam in args.lambdas:
            row = summarize(sub, lam)
            row["surrogate"] = surrogate
            by_sur.append(row)
    pd.DataFrame(by_sur).to_csv(out_dir / "phase4c_lambda_by_surrogate.csv", index=False)

    # By model + prefix
    by_mp = []
    for (model, prefix), sub in cf.groupby(["model", "prefix"], dropna=False):
        for lam in args.lambdas:
            row = summarize(sub, lam)
            row["model"] = model
            row["prefix"] = prefix
            by_mp.append(row)
    pd.DataFrame(by_mp).to_csv(out_dir / "phase4c_lambda_by_model_prefix.csv", index=False)

    # Support distribution
    support = {
        "eligible_rule_count": int(len(eligible)),
        "eligible_counterfactual_rule_count": int(len(cf)),
        "support_min": float(cf["n_r"].min()),
        "support_q25": q(cf["n_r"], 0.25),
        "support_median": q(cf["n_r"], 0.50),
        "support_q75": q(cf["n_r"], 0.75),
        "support_q90": q(cf["n_r"], 0.90),
        "support_max": float(cf["n_r"].max()),
        "lambda_candidates": list(args.lambdas),
        "lambda_selected": None,
        "selection_status": "NOT_FROZEN_PHASE4C_A_ONLY"
    }
    with (out_dir / "phase4c_support_and_grid.json").open("w", encoding="utf-8") as f:
        json.dump(support, f, ensure_ascii=False, indent=2)

    # Human-readable report
    cf_table = summary[summary["scope"] == "counterfactual"][
        ["lambda", "mean_rho", "mean_reliability", "median_reliability",
         "mean_ignorance_mass", "share_R_le_0_2", "share_R_ge_0_5", "share_R_ge_0_7"]
    ].copy()

    def pct(x): return f"{100*x:.2f}%"

    lines = []
    lines.append("# ETCE Phase 4C-A Lambda Sensitivity Report\n")
    lines.append("> Validation-only. No test data are used. Lambda is not frozen in this step.\n")
    lines.append("## Frozen reliability/BPA formula\n")
    lines.append(r"- $\rho_r=n_r/(n_r+\lambda)$")
    lines.append(r"- $R(r)=\rho_r Fid(r)$")
    lines.append(r"- $m_r(\theta_r)=R(r)$")
    lines.append(r"- $m_r(\Theta)=1-R(r)$")
    lines.append("\n## Eligible counterfactual rule support\n")
    lines.append(
        f"- N={support['eligible_counterfactual_rule_count']}; "
        f"min={support['support_min']:.0f}; Q25={support['support_q25']:.0f}; "
        f"median={support['support_median']:.0f}; Q75={support['support_q75']:.0f}; "
        f"Q90={support['support_q90']:.0f}; max={support['support_max']:.0f}."
    )
    lines.append("\n## Lambda sensitivity on eligible counterfactual rules\n")
    lines.append("| lambda | mean rho | mean R | median R | mean m(Theta) | R<=0.2 | R>=0.5 | R>=0.7 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in cf_table.iterrows():
        lines.append(
            f"| {r['lambda']:g} | {r['mean_rho']:.4f} | {r['mean_reliability']:.4f} | "
            f"{r['median_reliability']:.4f} | {r['mean_ignorance_mass']:.4f} | "
            f"{pct(r['share_R_le_0_2'])} | {pct(r['share_R_ge_0_5'])} | {pct(r['share_R_ge_0_7'])} |"
        )
    lines.append("\n## Status\n")
    lines.append(
        "This step only quantifies how strongly each candidate discounts low-support rules. "
        "Do not freeze lambda from this table alone. The next step must apply the frozen "
        "hierarchical D-S fusion/evaluation protocol on the validation bank and then choose one "
        "lambda before any formal test rule bank is inspected."
    )
    (out_dir / "PHASE4C_A_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("=" * 88)
    print("ETCE PHASE 4C-A - VALIDATION-ONLY LAMBDA SENSITIVITY")
    print("=" * 88)
    print(f"Rules: {rules_path}")
    print(f"Eligible all: {len(eligible)}")
    print(f"Eligible counterfactual: {len(cf)}")
    print(
        "CF support: "
        f"min={support['support_min']:.0f}, q25={support['support_q25']:.0f}, "
        f"median={support['support_median']:.0f}, q75={support['support_q75']:.0f}, "
        f"max={support['support_max']:.0f}"
    )
    print()
    print(cf_table.to_string(index=False))
    print()
    print(f"Outputs: {out_dir}")
    print("IMPORTANT: lambda_selected = null. Formal selection belongs to Phase 4C-B.")

if __name__ == "__main__":
    main()
