#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-7:
Descriptive diagnosis of Stage-3 formal TEST fidelity change.

READ-ONLY / INTERPRETATION ONLY.
This script must NOT be used to retune lambda, grouping, fusion order,
conflict handling, or representative-rule policy.

Inputs:
- Phase 4D-6 matched transition detail
- Phase 4D-4 stage outputs
- Phase 4D-4 conflict trace
- Phase 4D-5 independent validation

Outputs describe where Stage2->Stage3 fidelity changes occur:
- by prefix
- by number of black-box evidence units
- by conflict level
- by pre-fusion fidelity dispersion
- correlations with conflict and reliability

No model fitting or D-S recomputation is performed.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]

FORMAL = ROOT / "results/phase4/phase4d_formal_test"
COMP_DIR = FORMAL / "formal_test_stage_comparison"
FUSION_DIR = FORMAL / "formal_test_frozen_ds_fusion"
OUT = FORMAL / "formal_test_stage3_diagnostic"

DETAIL = COMP_DIR / "formal_matched_transition_detail.csv"
STAGE = FUSION_DIR / "phase4d_frozen_ds_stage_outputs.csv"
TRACE = FUSION_DIR / "phase4d_frozen_ds_conflict_trace.csv"
VALID = FUSION_DIR / "validation_phase4d_frozen_ds_fusion.json"


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
        w = csv.DictWriter(f, fieldnames=fields)
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


def i(v):
    return int(float(v))


def b(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "1", "yes"}


def mean(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def median(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(statistics.median(vals)) if vals else None


def stdev(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(statistics.pstdev(vals)) if len(vals) >= 2 else 0.0 if vals else None


def pearson(xs, ys):
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None and y is not None
        and math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return None

    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    mx = sum(x) / len(x)
    my = sum(y) / len(y)

    num = sum((a - mx) * (c - my) for a, c in pairs)
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((c - my) ** 2 for c in y))

    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def q(xs, p):
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    z = (len(vals) - 1) * p
    lo = int(math.floor(z))
    hi = int(math.ceil(z))
    if lo == hi:
        return vals[lo]
    frac = z - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def key_from_detail(r):
    return (
        r["case_id"],
        i(r["prefix"]),
        i(r["predicted_class"]),
        r["feature_signature"],
    )


def key_from_stage3(r):
    return (
        r["case_id"],
        i(r["prefix"]),
        i(r["predicted_class"]),
        json.dumps(json.loads(r["feature_signature"]), separators=(",", ":")),
    )


def key_from_trace(r):
    return (
        r["case_id"],
        i(r["prefix"]),
        i(r["predicted_class"]),
        json.dumps(json.loads(r["feature_signature"]), separators=(",", ":")),
    )


def summarize(rows, label_field=None):
    if not rows:
        return {}

    return {
        "N": len(rows),
        "mean_pre_fidelity": mean(r["pre_fidelity"] for r in rows),
        "mean_post_fidelity": mean(r["post_fidelity"] for r in rows),
        "mean_delta_fidelity": mean(r["delta_fidelity"] for r in rows),
        "median_delta_fidelity": median(r["delta_fidelity"] for r in rows),
        "improved_count": sum(r["direction"] == "improved" for r in rows),
        "tied_count": sum(r["direction"] == "tied" for r in rows),
        "worse_count": sum(r["direction"] == "worse" for r in rows),
        "mean_input_count": mean(r["input_count"] for r in rows),
        "mean_group_mean_K": mean(r["group_mean_K"] for r in rows),
        "mean_group_max_K": mean(r["group_max_K"] for r in rows),
        "mean_last_K": mean(r["last_K"] for r in rows),
        "mean_pre_reliability": mean(r["pre_reliability"] for r in rows),
        "mean_post_reliability": mean(r["post_reliability"] for r in rows),
        "mean_delta_reliability": mean(r["delta_reliability"] for r in rows),
        "mean_post_model_validity": mean(r["post_model_validity"] for r in rows),
        "post_all_model_valid_rate": mean(r["post_all_model_validity"] for r in rows),
    }


def main():
    for p in (DETAIL, STAGE, TRACE, VALID):
        require(p.exists(), f"Missing required input: {p}")

    valid = read_json(VALID)
    require(valid.get("status") == "passed", "Phase 4D-5 validation not PASSED")

    detail_all = read_csv(DETAIL)
    detail = [
        r for r in detail_all
        if r["transition"] == "Stage2->Stage3"
    ]
    require(len(detail) == 529, f"Expected 529 matched Stage3 groups, found {len(detail)}")

    stage_all = read_csv(STAGE)
    stage3 = [
        r for r in stage_all
        if r["stage"] == "Stage3" and b(r["actually_fused"])
    ]
    require(len(stage3) == 529, f"Expected 529 fused Stage3 rows, found {len(stage3)}")

    trace_all = read_csv(TRACE)
    trace3 = [r for r in trace_all if r["stage"] == "Stage3"]

    stage_idx = {key_from_stage3(r): r for r in stage3}
    require(len(stage_idx) == 529, "Duplicate Stage3 output key")

    trace_groups = defaultdict(list)
    for r in trace3:
        trace_groups[key_from_trace(r)].append(r)

    enriched = []

    for r in detail:
        key = key_from_detail(r)
        # Normalize signature serialization.
        key = (
            key[0],
            key[1],
            key[2],
            json.dumps(json.loads(key[3]), separators=(",", ":")),
        )

        require(key in stage_idx, f"Stage3 row missing for {key}")
        s = stage_idx[key]
        tr = trace_groups.get(key, [])

        delta = f(r["delta_fidelity"])
        if delta is None:
            direction = "NA"
        elif delta > 1e-12:
            direction = "improved"
        elif delta < -1e-12:
            direction = "worse"
        else:
            direction = "tied"

        ks = [f(x["K"]) for x in tr]

        enriched.append({
            "case_id": r["case_id"],
            "prefix": i(r["prefix"]),
            "predicted_class": i(r["predicted_class"]),
            "feature_signature": key[3],
            "input_count": i(r["input_count"]),
            "pre_fidelity": f(r["pre_mean_fidelity"]),
            "post_fidelity": f(r["post_fidelity"]),
            "delta_fidelity": delta,
            "direction": direction,
            "pre_reliability": f(r["pre_mean_reliability"]),
            "post_reliability": f(r["post_reliability"]),
            "delta_reliability": f(r["delta_reliability"]),
            "post_cf_validity": f(r["post_cf_validity"]),
            "post_model_validity": f(r["post_model_validity"]),
            "post_all_model_validity": f(r["post_all_model_validity"]),
            "group_pairwise_steps": len(tr),
            "group_mean_K": mean(ks),
            "group_max_K": max(ks) if ks else None,
            "last_K": f(s["last_K"]),
            "source_rule_count": i(s["source_rule_count"]),
            "source_model_count": i(s["source_model_count"]),
            "source_surrogate_count": i(s["source_surrogate_count"]),
            "post_support": i(s["n_r"]),
            "post_coverage": f(s["coverage_eval"]),
        })

    OUT.mkdir(parents=True, exist_ok=True)

    # By prefix.
    by_prefix = []
    for prefix in (5, 10, 15, 20):
        rows = [r for r in enriched if r["prefix"] == prefix]
        by_prefix.append({"prefix": prefix, **summarize(rows)})

    # By number of black-box inputs.
    by_count = []
    for cnt in sorted(set(r["input_count"] for r in enriched)):
        rows = [r for r in enriched if r["input_count"] == cnt]
        by_count.append({"blackbox_input_count": cnt, **summarize(rows)})

    # Conflict quartile bins defined descriptively from formal Stage3 groups.
    k_values = [r["group_mean_K"] for r in enriched]
    q25 = q(k_values, 0.25)
    q50 = q(k_values, 0.50)
    q75 = q(k_values, 0.75)

    def kbin(v):
        if v is None:
            return "NA"
        if v <= q25:
            return "Q1-low"
        if v <= q50:
            return "Q2"
        if v <= q75:
            return "Q3"
        return "Q4-high"

    by_conflict_groups = defaultdict(list)
    for r in enriched:
        by_conflict_groups[kbin(r["group_mean_K"])].append(r)

    by_conflict = []
    for name in ("Q1-low", "Q2", "Q3", "Q4-high", "NA"):
        if name in by_conflict_groups:
            by_conflict.append({
                "conflict_bin": name,
                "lower_context": {
                    "Q1-low": f"<= {q25:.6f}",
                    "Q2": f"({q25:.6f}, {q50:.6f}]",
                    "Q3": f"({q50:.6f}, {q75:.6f}]",
                    "Q4-high": f"> {q75:.6f}",
                    "NA": "NA",
                }[name],
                **summarize(by_conflict_groups[name]),
            })

    # Direction summary.
    by_direction = []
    for name in ("improved", "tied", "worse"):
        rows = [r for r in enriched if r["direction"] == name]
        by_direction.append({"direction": name, **summarize(rows)})

    correlations = {
        "delta_fidelity_vs_group_mean_K": pearson(
            [r["delta_fidelity"] for r in enriched],
            [r["group_mean_K"] for r in enriched],
        ),
        "delta_fidelity_vs_group_max_K": pearson(
            [r["delta_fidelity"] for r in enriched],
            [r["group_max_K"] for r in enriched],
        ),
        "delta_fidelity_vs_pre_fidelity": pearson(
            [r["delta_fidelity"] for r in enriched],
            [r["pre_fidelity"] for r in enriched],
        ),
        "delta_fidelity_vs_pre_reliability": pearson(
            [r["delta_fidelity"] for r in enriched],
            [r["pre_reliability"] for r in enriched],
        ),
        "delta_fidelity_vs_post_model_validity": pearson(
            [r["delta_fidelity"] for r in enriched],
            [r["post_model_validity"] for r in enriched],
        ),
    }

    overall = summarize(enriched)

    write_csv(OUT / "stage3_diagnostic_detail.csv", enriched)
    write_csv(OUT / "stage3_by_prefix.csv", by_prefix)
    write_csv(OUT / "stage3_by_blackbox_count.csv", by_count)
    write_csv(OUT / "stage3_by_conflict_quartile.csv", by_conflict)
    write_csv(OUT / "stage3_by_fidelity_direction.csv", by_direction)

    result = {
        "scope": "ETCE Phase 4D-7 descriptive Stage3 formal TEST diagnosis",
        "status": "passed",
        "interpretation_only": True,
        "protocol_retuning_performed": False,
        "lambda_selection_performed": False,
        "matched_stage3_group_count": len(enriched),
        "overall": overall,
        "conflict_quartiles": {
            "q25_group_mean_K": q25,
            "q50_group_mean_K": q50,
            "q75_group_mean_K": q75,
        },
        "correlations": correlations,
        "by_prefix": by_prefix,
        "by_blackbox_count": by_count,
        "by_conflict_quartile": by_conflict,
        "by_direction": by_direction,
        "warning": (
            "These are post-hoc descriptive formal TEST diagnostics. "
            "They must not be used to modify the frozen D-S protocol."
        ),
        "next_step": (
            "Proceed to predeclared fusion baselines / BPA ablations "
            "without changing the frozen ETCE protocol."
        ),
    }
    write_json(OUT / "stage3_diagnostic_summary.json", result)

    print("=" * 116)
    print("ETCE PHASE 4D-7 DESCRIPTIVE STAGE3 FORMAL TEST DIAGNOSTIC PASSED")
    print("=" * 116)
    print(f"Matched Stage3 groups: {len(enriched)}")
    print(
        f"Overall Fidelity: pre={overall['mean_pre_fidelity']:.6f} | "
        f"post={overall['mean_post_fidelity']:.6f} | "
        f"delta={overall['mean_delta_fidelity']:+.6f}"
    )
    print(
        f"Improved/tied/worse: "
        f"{overall['improved_count']}/"
        f"{overall['tied_count']}/"
        f"{overall['worse_count']}"
    )
    print()
    print("[By prefix]")
    for r in by_prefix:
        print(
            f"P{r['prefix']} | N={r['N']} | "
            f"pre={r['mean_pre_fidelity']:.6f} | "
            f"post={r['mean_post_fidelity']:.6f} | "
            f"delta={r['mean_delta_fidelity']:+.6f} | "
            f"I/T/W={r['improved_count']}/{r['tied_count']}/{r['worse_count']}"
        )

    print()
    print("[By black-box input count]")
    for r in by_count:
        print(
            f"{r['blackbox_input_count']} models | N={r['N']} | "
            f"pre={r['mean_pre_fidelity']:.6f} | "
            f"post={r['mean_post_fidelity']:.6f} | "
            f"delta={r['mean_delta_fidelity']:+.6f} | "
            f"meanK={r['mean_group_mean_K']:.6f}"
        )

    print()
    print("[By conflict quartile]")
    for r in by_conflict:
        print(
            f"{r['conflict_bin']} | N={r['N']} | "
            f"delta={r['mean_delta_fidelity']:+.6f} | "
            f"meanK={r['mean_group_mean_K']:.6f} | "
            f"I/T/W={r['improved_count']}/{r['tied_count']}/{r['worse_count']}"
        )

    print()
    print("[Correlations]")
    for k, v in correlations.items():
        print(f"{k}: {v if v is not None else 'NA'}")

    print()
    print(f"Output: {OUT}")
    print("Interpretation only: True")
    print("Protocol retuning performed: False")
    print("NEXT: predeclared fusion baselines / BPA ablations.")


if __name__ == "__main__":
    main()
