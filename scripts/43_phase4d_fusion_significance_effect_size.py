#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-9 / Script 43
Paired significance and effect-size analysis for formal fusion strategies.

Primary comparison:
    D-S ETCE vs each frozen fusion baseline
on the SAME 529 genuinely fusable Stage-3 compatible groups.

Statistics:
- Rule Fidelity:
    paired mean/median delta
    paired bootstrap 95% CI of mean delta
    Wilcoxon signed-rank two-sided p-value (SciPy if available)
    paired rank-biserial effect size
    wins / ties / losses
    Holm-Bonferroni correction across the 6 primary Fidelity comparisons

- Model-level CF validity:
    same paired continuous analysis

- CF Validity and All-model Validity:
    exact McNemar/binomial test on discordant pairs

No TEST-side parameter tuning is performed.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]

FORMAL = ROOT / "results/phase4/phase4d_formal_test"
BASE_DIR = FORMAL / "formal_test_fusion_baselines"
DS_DIR = FORMAL / "formal_test_frozen_ds_fusion"
OUT = FORMAL / "formal_test_fusion_significance"

BASE_STAGE = BASE_DIR / "formal_fusion_baseline_stage_outputs.csv"
BASE_RESULT = BASE_DIR / "formal_fusion_strategy_comparison.json"
DS_STAGE3 = DS_DIR / "phase4d_stage3_fused_rules.csv"
DS_VALIDATION = DS_DIR / "validation_phase4d_frozen_ds_fusion.json"

BASELINES = [
    "best_single_evidence",
    "validation_best_source",
    "uniform_vote",
    "validation_weighted_vote",
    "fuzzy_probabilistic_or",
    "bayesian_categorical_noise",
]

BOOTSTRAP_B = 20000
BOOTSTRAP_SEED = 20260930
TOL = 1e-12


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def f(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"none", "nan", "null"}:
        return None
    return float(s)


def b(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "1", "yes"}


def key(row):
    return (
        row["case_id"],
        int(float(row["prefix"])),
        int(float(row["predicted_class"])),
        json.dumps(
            json.loads(row["feature_signature"]),
            separators=(",", ":"),
        ),
    )


def mean(xs):
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def median(xs):
    vals = [float(x) for x in xs if x is not None]
    return float(statistics.median(vals)) if vals else None


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    z = (len(sorted_vals) - 1) * p
    lo = int(math.floor(z))
    hi = int(math.ceil(z))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = z - lo
    return float(
        sorted_vals[lo] * (1 - frac)
        + sorted_vals[hi] * frac
    )


def bootstrap_ci_mean(diffs, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    vals = [float(x) for x in diffs]
    n = len(vals)
    require(n > 0, "Empty bootstrap input")
    rng = random.Random(seed)

    boots = []
    for _ in range(B):
        s = 0.0
        for _j in range(n):
            s += vals[rng.randrange(n)]
        boots.append(s / n)

    boots.sort()
    return (
        percentile(boots, 0.025),
        percentile(boots, 0.975),
    )


def average_ranks(abs_vals):
    """
    Average ranks for ties, ranks start at 1.
    Returns ranks in original order.
    """
    indexed = sorted(
        enumerate(abs_vals),
        key=lambda t: t[1],
    )

    ranks = [0.0] * len(abs_vals)
    i = 0

    while i < len(indexed):
        j = i + 1
        while (
            j < len(indexed)
            and abs(indexed[j][1] - indexed[i][1]) <= 1e-15
        ):
            j += 1

        avg_rank = (
            (i + 1 + j) / 2.0
        )

        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank

        i = j

    return ranks


def rank_biserial(diffs):
    nz = [float(d) for d in diffs if abs(float(d)) > TOL]
    if not nz:
        return 0.0

    ranks = average_ranks([abs(d) for d in nz])
    w_pos = sum(r for r, d in zip(ranks, nz) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, nz) if d < 0)
    denom = w_pos + w_neg

    if denom == 0:
        return 0.0

    return float((w_pos - w_neg) / denom)


def exact_binom_two_sided(k, n):
    """
    Exact two-sided binomial p under p=0.5.
    Appropriate for sign test and McNemar discordant pairs.
    """
    if n <= 0:
        return 1.0

    k = int(k)
    n = int(n)
    lo = min(k, n - k)

    prob = sum(
        math.comb(n, i)
        for i in range(lo + 1)
    ) / (2 ** n)

    return min(1.0, 2.0 * prob)


def paired_pvalue(diffs):
    nz = [float(d) for d in diffs if abs(float(d)) > TOL]

    if not nz:
        return {
            "test": "all_ties",
            "p_value": 1.0,
        }

    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(
            nz,
            zero_method="wilcox",
            alternative="two-sided",
            correction=False,
            mode="auto",
        )

        return {
            "test": "wilcoxon_signed_rank_two_sided",
            "p_value": float(result.pvalue),
        }

    except Exception:
        pos = sum(d > 0 for d in nz)
        p = exact_binom_two_sided(pos, len(nz))
        return {
            "test": "exact_sign_test_fallback",
            "p_value": float(p),
        }


def continuous_pair(ds_vals, base_vals):
    require(len(ds_vals) == len(base_vals), "Pair length mismatch")

    diffs = [
        float(d) - float(x)
        for d, x in zip(ds_vals, base_vals)
    ]

    wins = sum(d > TOL for d in diffs)
    ties = sum(abs(d) <= TOL for d in diffs)
    losses = sum(d < -TOL for d in diffs)

    ci_lo, ci_hi = bootstrap_ci_mean(diffs)
    p = paired_pvalue(diffs)

    return {
        "N": len(diffs),
        "mean_ds": mean(ds_vals),
        "mean_baseline": mean(base_vals),
        "mean_delta_ds_minus_baseline": mean(diffs),
        "median_delta_ds_minus_baseline": median(diffs),
        "bootstrap95_mean_delta_lo": ci_lo,
        "bootstrap95_mean_delta_hi": ci_hi,
        "wins_ds": wins,
        "ties": ties,
        "losses_ds": losses,
        "win_rate_excluding_ties": (
            wins / (wins + losses)
            if (wins + losses) > 0
            else None
        ),
        "paired_test": p["test"],
        "p_value": p["p_value"],
        "rank_biserial_effect": rank_biserial(diffs),
    }


def binary_mcnemar(ds_vals, base_vals):
    require(len(ds_vals) == len(base_vals), "Pair length mismatch")

    b01 = 0  # DS=0, baseline=1
    b10 = 0  # DS=1, baseline=0
    agree11 = 0
    agree00 = 0

    for d, x in zip(ds_vals, base_vals):
        d = int(bool(d))
        x = int(bool(x))

        if d == 1 and x == 1:
            agree11 += 1
        elif d == 0 and x == 0:
            agree00 += 1
        elif d == 0 and x == 1:
            b01 += 1
        elif d == 1 and x == 0:
            b10 += 1

    discordant = b01 + b10
    p = exact_binom_two_sided(b10, discordant)

    return {
        "N": len(ds_vals),
        "ds_rate": mean(int(bool(v)) for v in ds_vals),
        "baseline_rate": mean(int(bool(v)) for v in base_vals),
        "delta_rate_ds_minus_baseline": (
            mean(int(bool(v)) for v in ds_vals)
            - mean(int(bool(v)) for v in base_vals)
        ),
        "agree_11": agree11,
        "agree_00": agree00,
        "ds0_baseline1": b01,
        "ds1_baseline0": b10,
        "discordant": discordant,
        "test": "exact_mcnemar_binomial_two_sided",
        "p_value": p,
    }


def holm_adjust(rows, p_field="p_value"):
    """
    Holm-Bonferroni adjusted p-values.
    Returns map baseline -> adjusted p.
    """
    pairs = sorted(
        [
            (r["baseline"], float(r[p_field]))
            for r in rows
        ],
        key=lambda x: x[1],
    )

    m = len(pairs)
    adjusted = {}
    running = 0.0

    for rank, (name, p) in enumerate(pairs, start=1):
        raw_adj = (m - rank + 1) * p
        running = max(running, raw_adj)
        adjusted[name] = min(1.0, running)

    return adjusted


def main():
    for p in (
        BASE_STAGE,
        BASE_RESULT,
        DS_STAGE3,
        DS_VALIDATION,
    ):
        require(p.is_file(), f"Missing required input: {p}")

    base_result = read_json(BASE_RESULT)
    ds_validation = read_json(DS_VALIDATION)

    require(base_result.get("status") == "passed", "Script 42 result not PASSED")
    require(ds_validation.get("status") == "passed", "D-S validation not PASSED")
    require(
        base_result.get("test_side_baseline_tuning") is False,
        "TEST-side baseline tuning detected",
    )

    ds_rows_all = read_csv(DS_STAGE3)
    ds_rows = [
        r for r in ds_rows_all
        if b(r["actually_fused"])
    ]
    require(len(ds_rows) == 529, f"Expected 529 D-S fused rows, found {len(ds_rows)}")

    ds_idx = {key(r): r for r in ds_rows}
    require(len(ds_idx) == 529, "Duplicate D-S Stage3 identity")

    base_all = read_csv(BASE_STAGE)

    results_cont = []
    results_bin = []
    fidelity_primary = []

    for baseline in BASELINES:
        rows = [
            r for r in base_all
            if r["baseline"] == baseline
            and r["stage"] == "Stage3"
            and b(r["actually_fused"])
        ]

        require(
            len(rows) == 529,
            f"{baseline}: expected 529 fused Stage3 rows, found {len(rows)}",
        )

        idx = {key(r): r for r in rows}
        require(len(idx) == 529, f"{baseline}: duplicate identity")
        require(set(idx) == set(ds_idx), f"{baseline}: identity set differs from D-S")

        ordered_keys = sorted(ds_idx)

        # Continuous metrics
        for metric, field in [
            ("rule_fidelity", "Fid_r"),
            ("model_level_validity", "model_cf_validity_rate"),
            ("reliability", "R_r"),
            ("coverage", "coverage_eval"),
            ("support", "n_r"),
        ]:
            ds_vals = [f(ds_idx[k][field]) for k in ordered_keys]
            base_vals = [f(idx[k][field]) for k in ordered_keys]

            require(
                all(v is not None for v in ds_vals + base_vals),
                f"{baseline}/{metric}: null values present",
            )

            z = continuous_pair(ds_vals, base_vals)
            row = {
                "baseline": baseline,
                "metric": metric,
                **z,
            }
            results_cont.append(row)

            if metric == "rule_fidelity":
                fidelity_primary.append(row)

        # Binary metrics
        for metric, field in [
            ("cf_validity", "cf_valid"),
            ("all_model_validity", "all_models_cf_valid"),
        ]:
            ds_vals = [b(ds_idx[k][field]) for k in ordered_keys]
            base_vals = [b(idx[k][field]) for k in ordered_keys]

            z = binary_mcnemar(ds_vals, base_vals)
            results_bin.append({
                "baseline": baseline,
                "metric": metric,
                **z,
            })

    holm = holm_adjust(fidelity_primary)

    for r in fidelity_primary:
        r["holm_adjusted_p_fidelity_family"] = holm[r["baseline"]]
        r["significant_after_holm_0_05"] = (
            holm[r["baseline"]] < 0.05
        )

    # Add adjusted p back into the full continuous table.
    for r in results_cont:
        if r["metric"] == "rule_fidelity":
            r["holm_adjusted_p_fidelity_family"] = holm[r["baseline"]]
            r["significant_after_holm_0_05"] = (
                holm[r["baseline"]] < 0.05
            )
        else:
            r["holm_adjusted_p_fidelity_family"] = None
            r["significant_after_holm_0_05"] = None

    OUT.mkdir(parents=True, exist_ok=True)

    write_csv(
        OUT / "formal_paired_continuous_statistics.csv",
        results_cont,
    )
    write_csv(
        OUT / "formal_paired_binary_mcnemar.csv",
        results_bin,
    )
    write_csv(
        OUT / "formal_primary_fidelity_pairwise.csv",
        fidelity_primary,
    )

    summary = {
        "scope": "ETCE Phase 4D-9 paired significance/effect-size comparison",
        "status": "passed",
        "formal_test": True,
        "paired_group_count": 529,
        "primary_metric": "Rule Fidelity",
        "primary_multiple_comparison_correction": "Holm-Bonferroni across six D-S-vs-baseline Fidelity tests",
        "bootstrap": {
            "replicates": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 95% CI of paired mean delta",
        },
        "continuous_results": results_cont,
        "binary_results": results_bin,
        "no_test_side_tuning": True,
        "next_step": (
            "Run support-aware BPA ablation: full R=rho*Fid versus "
            "fidelity-only R=Fid under the same frozen grouping/fusion protocol."
        ),
    }

    write_json(
        OUT / "formal_fusion_significance_summary.json",
        summary,
    )

    # Human report
    lines = [
        "# ETCE Formal Fusion Strategy Paired Significance",
        "",
        "Primary endpoint: paired Stage-3 Rule Fidelity on the same 529 compatible groups.",
        "",
        "| Baseline | DS mean | Baseline mean | Δ(DS-base) | 95% CI | W/T/L | p | Holm p | Rank-biserial |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in fidelity_primary:
        lines.append(
            f"| {r['baseline']} | "
            f"{r['mean_ds']:.6f} | "
            f"{r['mean_baseline']:.6f} | "
            f"{r['mean_delta_ds_minus_baseline']:+.6f} | "
            f"[{r['bootstrap95_mean_delta_lo']:+.6f}, "
            f"{r['bootstrap95_mean_delta_hi']:+.6f}] | "
            f"{r['wins_ds']}/{r['ties']}/{r['losses_ds']} | "
            f"{r['p_value']:.6g} | "
            f"{r['holm_adjusted_p_fidelity_family']:.6g} | "
            f"{r['rank_biserial_effect']:+.4f} |"
        )

    lines.extend([
        "",
        "Binary CF-validity and all-model-validity differences are evaluated "
        "with exact paired McNemar/binomial tests.",
        "",
        "No superiority statement should be made solely from raw means; "
        "interpret the paired effect size, CI, corrected p-value, and conflict-stratified behavior together.",
    ])

    (
        OUT / "PHASE4D_FUSION_SIGNIFICANCE_REPORT.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 118)
    print("ETCE PHASE 4D-9 PAIRED FUSION SIGNIFICANCE / EFFECT SIZE PASSED")
    print("=" * 118)
    print("Primary metric: Stage3 Rule Fidelity")
    print("Matched groups: 529")
    print("Delta convention: D-S minus baseline")
    print()

    for r in fidelity_primary:
        sig = (
            "YES"
            if r["significant_after_holm_0_05"]
            else "NO"
        )
        print(
            f"{r['baseline']:30s} | "
            f"DS={r['mean_ds']:.6f} | "
            f"Base={r['mean_baseline']:.6f} | "
            f"Delta={r['mean_delta_ds_minus_baseline']:+.6f} | "
            f"CI=[{r['bootstrap95_mean_delta_lo']:+.6f},"
            f"{r['bootstrap95_mean_delta_hi']:+.6f}] | "
            f"W/T/L={r['wins_ds']}/{r['ties']}/{r['losses_ds']} | "
            f"p={r['p_value']:.6g} | "
            f"Holm={r['holm_adjusted_p_fidelity_family']:.6g} | "
            f"r_rb={r['rank_biserial_effect']:+.4f} | "
            f"sig={sig}"
        )

    print()
    print("[Binary paired tests]")
    for r in results_bin:
        print(
            f"{r['baseline']:30s} | {r['metric']:18s} | "
            f"DS={r['ds_rate']:.6f} | "
            f"Base={r['baseline_rate']:.6f} | "
            f"Delta={r['delta_rate_ds_minus_baseline']:+.6f} | "
            f"discordant={r['discordant']} | "
            f"p={r['p_value']:.6g}"
        )

    print()
    print(f"Output: {OUT}")
    print("TEST-side tuning performed: False")
    print("NEXT: support-aware BPA ablation.")


if __name__ == "__main__":
    main()
