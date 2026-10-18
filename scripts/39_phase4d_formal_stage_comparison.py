#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-6:
Formal Original -> Stage1 -> Stage2 -> Stage3 comparison.

This script DOES NOT perform D-S fusion again.
It reads only independently validated Phase 4D-4/5 outputs and the formal
eligible source-rule bank, then builds comparison tables under two views:

A) Population overview
   - Original eligible CF candidates
   - Stage1 all outputs / genuinely fused outputs
   - Stage2 all outputs / genuinely fused outputs
   - Stage3 all outputs / genuinely fused outputs

B) Matched transition analysis
   - Original inputs -> Stage1 fused output, within each fusable Stage1 group
   - Stage1 inputs -> Stage2 fused output, within each fusable Stage2 set
   - Stage2 inputs -> Stage3 fused output, within each fusable Stage3 set

For each matched transition, the "pre" value is the arithmetic mean of the
actual evidence units entering that fusion group. The "post" value is the
single fused rule emitted by that group. This prevents changes in group counts
from being mistaken for fusion gains.

No TEST-side tuning, model selection, or protocol change is performed.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]

FORMAL_ROOT = ROOT / "results/phase4/phase4d_formal_test"
BANK_DIR = FORMAL_ROOT / "formal_test_multirule_bank"
FUSION_DIR = FORMAL_ROOT / "formal_test_frozen_ds_fusion"
OUT_DIR = FORMAL_ROOT / "formal_test_stage_comparison"

RULES = BANK_DIR / "canonical_rules_test.jsonl"
AUDIT = (
    FORMAL_ROOT
    / "formal_test_target_local_group_audit"
    / "formal_test_target_local_group_audit.json"
)
FUSION_RESULT = FUSION_DIR / "formal_test_frozen_ds_fusion.json"
FUSION_VALIDATION = FUSION_DIR / "validation_phase4d_frozen_ds_fusion.json"
STAGE_OUTPUTS = FUSION_DIR / "phase4d_frozen_ds_stage_outputs.csv"

FROZEN = ROOT / "configs/phase4c_ds_frozen_config.json"

LAMBDA = 5.0
MODELS = ("RF", "XGBoost", "MLP", "LSTM")
SURROGATES = ("LORE", "DT", "BN")
PREFIXES = (5, 10, 15, 20)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"Invalid JSONL at line {i}: {path}") from exc
    return rows


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()
    for row in rows:
        for k in row:
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


def to_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Cannot parse bool: {v!r}")


def to_opt_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"none", "nan", "null"}:
        return None
    return float(s)


def to_opt_int(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"none", "nan", "null"}:
        return None
    return int(float(s))


def mean(vals):
    x = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(sum(x) / len(x)) if x else None


def median(vals):
    x = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(statistics.median(x)) if x else None


def feature_signature(conditions):
    if not isinstance(conditions, list):
        return tuple()
    pos = []
    for c in conditions:
        if isinstance(c, dict) and c.get("position") is not None:
            pos.append(int(c["position"]))
    return tuple(sorted(set(pos)))


def signature_str(sig):
    return json.dumps(list(sig), separators=(",", ":"))


def parse_signature(s):
    x = json.loads(s)
    return tuple(int(v) for v in x)


def source_metrics(rule):
    n = int(rule["n_r"])
    fid = float(rule["Fid_r"])
    rho = float(n / (n + LAMBDA))
    R = float(rho * fid)

    return {
        "Fid_r": fid,
        "n_r": n,
        "coverage_eval": float(rule["coverage_eval"]),
        "rho_r": rho,
        "R_r": R,
        "ignorance": 1.0 - R,
        # Counterfactual source candidates already have a generated witness and
        # are represented as valid source CF evidence in the frozen Stage1 code.
        "cf_valid": 1.0,
    }


def stage_metrics(row):
    return {
        "Fid_r": to_opt_float(row["Fid_r"]),
        "n_r": to_opt_int(row["n_r"]),
        "coverage_eval": to_opt_float(row["coverage_eval"]),
        "rho_r": to_opt_float(row["rho_r"]),
        "R_r": to_opt_float(row["R_r"]),
        "ignorance": to_opt_float(row["last_ignorance_mass"]),
        "cf_valid": float(to_bool(row["cf_valid"])),
        "model_cf_validity_rate": to_opt_float(row["model_cf_validity_rate"]),
        "all_models_cf_valid": float(to_bool(row["all_models_cf_valid"])),
        "min_witness_edit": to_opt_float(row["min_witness_edit"]),
    }


def summarize_units(label, rows, source=False):
    if source:
        metrics = [source_metrics(r) for r in rows]
    else:
        metrics = [stage_metrics(r) for r in rows]

    return {
        "scope": label,
        "N": len(rows),
        "mean_fidelity": mean(m["Fid_r"] for m in metrics),
        "median_fidelity": median(m["Fid_r"] for m in metrics),
        "cf_validity_rate": mean(m["cf_valid"] for m in metrics),
        "mean_support": mean(m["n_r"] for m in metrics),
        "mean_coverage": mean(m["coverage_eval"] for m in metrics),
        "mean_rho": mean(m["rho_r"] for m in metrics),
        "mean_reliability": mean(m["R_r"] for m in metrics),
        "mean_ignorance": mean(m["ignorance"] for m in metrics),
        "mean_model_cf_validity_rate": (
            None if source else mean(m["model_cf_validity_rate"] for m in metrics)
        ),
        "all_models_cf_valid_rate": (
            None if source else mean(m["all_models_cf_valid"] for m in metrics)
        ),
        "mean_min_witness_edit": (
            mean(r.get("witness_edit_count") for r in rows)
            if source
            else mean(m["min_witness_edit"] for m in metrics)
        ),
    }


def stage1_key_from_source(r):
    return (
        r["case_id"],
        r["model"],
        int(r["prefix"]),
        r["surrogate"],
        int(r["predicted_class"]),
        feature_signature(r["conditions"]),
    )


def stage1_key_from_output(r):
    return (
        r["case_id"],
        r["model"],
        int(r["prefix"]),
        r["surrogate"],
        int(r["predicted_class"]),
        parse_signature(r["feature_signature"]),
    )


def stage2_key_from_s1(r):
    return (
        r["case_id"],
        r["model"],
        int(r["prefix"]),
        int(r["predicted_class"]),
        parse_signature(r["feature_signature"]),
    )


def stage2_key_from_output(r):
    return (
        r["case_id"],
        r["model"],
        int(r["prefix"]),
        int(r["predicted_class"]),
        parse_signature(r["feature_signature"]),
    )


def stage3_key_from_s2(r):
    return (
        r["case_id"],
        int(r["prefix"]),
        int(r["predicted_class"]),
        parse_signature(r["feature_signature"]),
    )


def stage3_key_from_output(r):
    return (
        r["case_id"],
        int(r["prefix"]),
        int(r["predicted_class"]),
        parse_signature(r["feature_signature"]),
    )


def unit_mean_metrics(rows, source=False):
    if source:
        ms = [source_metrics(r) for r in rows]
    else:
        ms = [stage_metrics(r) for r in rows]

    return {
        "fidelity": mean(m["Fid_r"] for m in ms),
        "support": mean(m["n_r"] for m in ms),
        "coverage": mean(m["coverage_eval"] for m in ms),
        "reliability": mean(m["R_r"] for m in ms),
        "ignorance": mean(m["ignorance"] for m in ms),
        "cf_validity": mean(m["cf_valid"] for m in ms),
        "model_validity": (
            None if source else mean(m["model_cf_validity_rate"] for m in ms)
        ),
        "all_model_validity": (
            None if source else mean(m["all_models_cf_valid"] for m in ms)
        ),
    }


def post_metrics(row):
    m = stage_metrics(row)
    return {
        "fidelity": m["Fid_r"],
        "support": m["n_r"],
        "coverage": m["coverage_eval"],
        "reliability": m["R_r"],
        "ignorance": m["ignorance"],
        "cf_validity": m["cf_valid"],
        "model_validity": m["model_cf_validity_rate"],
        "all_model_validity": m["all_models_cf_valid"],
    }


def delta(a, b):
    if a is None or b is None:
        return None
    return float(b - a)


def matched_transition(
    transition,
    pre_groups,
    post_index,
    source=False,
):
    detail = []

    for key, inputs in sorted(pre_groups.items(), key=lambda kv: str(kv[0])):
        if len(inputs) < 2:
            continue

        require(key in post_index, f"Matched post output missing for {transition}: {key}")

        pre = unit_mean_metrics(inputs, source=source)
        post = post_metrics(post_index[key])

        d_fid = delta(pre["fidelity"], post["fidelity"])

        if d_fid is None:
            direction = "NA"
        elif d_fid > 1e-12:
            direction = "improved"
        elif d_fid < -1e-12:
            direction = "worse"
        else:
            direction = "tied"

        detail.append({
            "transition": transition,
            "case_id": key[0],
            "model": (
                key[1]
                if transition in {"Original->Stage1", "Stage1->Stage2"}
                else None
            ),
            "prefix": (
                key[2]
                if transition in {"Original->Stage1", "Stage1->Stage2"}
                else key[1]
            ),
            "surrogate": (
                key[3] if transition == "Original->Stage1" else None
            ),
            "predicted_class": (
                key[4]
                if transition == "Original->Stage1"
                else key[3]
                if transition == "Stage1->Stage2"
                else key[2]
            ),
            "feature_signature": (
                signature_str(
                    key[5]
                    if transition == "Original->Stage1"
                    else key[4]
                    if transition == "Stage1->Stage2"
                    else key[3]
                )
            ),
            "input_count": len(inputs),
            "pre_mean_fidelity": pre["fidelity"],
            "post_fidelity": post["fidelity"],
            "delta_fidelity": d_fid,
            "fidelity_direction": direction,
            "pre_mean_support": pre["support"],
            "post_support": post["support"],
            "delta_support": delta(pre["support"], post["support"]),
            "pre_mean_coverage": pre["coverage"],
            "post_coverage": post["coverage"],
            "delta_coverage": delta(pre["coverage"], post["coverage"]),
            "pre_mean_reliability": pre["reliability"],
            "post_reliability": post["reliability"],
            "delta_reliability": delta(pre["reliability"], post["reliability"]),
            "pre_mean_ignorance": pre["ignorance"],
            "post_ignorance": post["ignorance"],
            "delta_ignorance": delta(pre["ignorance"], post["ignorance"]),
            "pre_mean_cf_validity": pre["cf_validity"],
            "post_cf_validity": post["cf_validity"],
            "delta_cf_validity": delta(pre["cf_validity"], post["cf_validity"]),
            "pre_mean_model_validity": pre["model_validity"],
            "post_model_validity": post["model_validity"],
            "delta_model_validity": delta(pre["model_validity"], post["model_validity"]),
            "pre_mean_all_model_validity": pre["all_model_validity"],
            "post_all_model_validity": post["all_model_validity"],
            "delta_all_model_validity": delta(
                pre["all_model_validity"],
                post["all_model_validity"],
            ),
        })

    return detail


def transition_summary(detail):
    by_transition = defaultdict(list)
    for r in detail:
        by_transition[r["transition"]].append(r)

    out = []
    for transition in (
        "Original->Stage1",
        "Stage1->Stage2",
        "Stage2->Stage3",
    ):
        rows = by_transition[transition]
        n = len(rows)
        improved = sum(r["fidelity_direction"] == "improved" for r in rows)
        tied = sum(r["fidelity_direction"] == "tied" for r in rows)
        worse = sum(r["fidelity_direction"] == "worse" for r in rows)

        out.append({
            "transition": transition,
            "matched_group_count": n,
            "mean_pre_fidelity": mean(r["pre_mean_fidelity"] for r in rows),
            "mean_post_fidelity": mean(r["post_fidelity"] for r in rows),
            "mean_delta_fidelity": mean(r["delta_fidelity"] for r in rows),
            "median_delta_fidelity": median(r["delta_fidelity"] for r in rows),
            "fidelity_improved_count": improved,
            "fidelity_tied_count": tied,
            "fidelity_worse_count": worse,
            "fidelity_improved_rate": float(improved / n) if n else None,
            "mean_pre_reliability": mean(r["pre_mean_reliability"] for r in rows),
            "mean_post_reliability": mean(r["post_reliability"] for r in rows),
            "mean_delta_reliability": mean(r["delta_reliability"] for r in rows),
            "mean_pre_coverage": mean(r["pre_mean_coverage"] for r in rows),
            "mean_post_coverage": mean(r["post_coverage"] for r in rows),
            "mean_delta_coverage": mean(r["delta_coverage"] for r in rows),
            "mean_pre_support": mean(r["pre_mean_support"] for r in rows),
            "mean_post_support": mean(r["post_support"] for r in rows),
            "mean_delta_support": mean(r["delta_support"] for r in rows),
            "mean_pre_cf_validity": mean(r["pre_mean_cf_validity"] for r in rows),
            "mean_post_cf_validity": mean(r["post_cf_validity"] for r in rows),
            "mean_delta_cf_validity": mean(r["delta_cf_validity"] for r in rows),
            "mean_pre_model_validity": mean(r["pre_mean_model_validity"] for r in rows),
            "mean_post_model_validity": mean(r["post_model_validity"] for r in rows),
            "mean_delta_model_validity": mean(r["delta_model_validity"] for r in rows),
            "mean_pre_all_model_validity": mean(
                r["pre_mean_all_model_validity"] for r in rows
            ),
            "mean_post_all_model_validity": mean(
                r["post_all_model_validity"] for r in rows
            ),
            "mean_delta_all_model_validity": mean(
                r["delta_all_model_validity"] for r in rows
            ),
        })

    return out


def prefix_summary(stage_rows):
    out = []

    for stage in ("Stage1", "Stage2", "Stage3"):
        rows_stage = [r for r in stage_rows if r["stage"] == stage]

        for prefix in PREFIXES:
            sub = [
                r for r in rows_stage
                if int(r["prefix"]) == prefix
                and to_bool(r["actually_fused"])
            ]
            if not sub:
                continue

            m = [stage_metrics(r) for r in sub]
            out.append({
                "stage": stage,
                "prefix": prefix,
                "fused_group_count": len(sub),
                "mean_fidelity": mean(x["Fid_r"] for x in m),
                "cf_validity_rate": mean(x["cf_valid"] for x in m),
                "mean_reliability": mean(x["R_r"] for x in m),
                "mean_coverage": mean(x["coverage_eval"] for x in m),
                "mean_support": mean(x["n_r"] for x in m),
                "mean_model_cf_validity_rate": mean(
                    x["model_cf_validity_rate"] for x in m
                ),
                "all_models_cf_valid_rate": mean(
                    x["all_models_cf_valid"] for x in m
                ),
            })

    return out


def original_breakdown(cf):
    groups = defaultdict(list)
    for r in cf:
        groups[
            (
                r["model"],
                int(r["prefix"]),
                r["surrogate"],
            )
        ].append(r)

    out = []
    for (model, prefix, surrogate), rows in sorted(
        groups.items(),
        key=lambda kv: (
            MODELS.index(kv[0][0]),
            kv[0][1],
            SURROGATES.index(kv[0][2]),
        ),
    ):
        s = summarize_units("Original", rows, source=True)
        out.append({
            "model": model,
            "prefix": prefix,
            "surrogate": surrogate,
            **{k: v for k, v in s.items() if k != "scope"},
        })
    return out


def fmt(v):
    return "NA" if v is None else f"{float(v):.4f}"


def main():
    required = [
        RULES,
        AUDIT,
        FUSION_RESULT,
        FUSION_VALIDATION,
        STAGE_OUTPUTS,
        FROZEN,
    ]
    missing = [str(p) for p in required if not p.is_file()]
    require(not missing, "Missing required artifacts:\n" + "\n".join(missing))

    audit = read_json(AUDIT)
    fusion_result = read_json(FUSION_RESULT)
    fusion_validation = read_json(FUSION_VALIDATION)
    frozen = read_json(FROZEN)

    # Hard gates: compare only independently validated, frozen outputs.
    require(audit.get("status") == "passed", "Phase 4D-3 audit not PASSED")
    require(fusion_result.get("status") == "passed", "Phase 4D-4 fusion not PASSED")
    require(
        fusion_validation.get("status") == "passed",
        "Phase 4D-5 independent fusion validation not PASSED",
    )
    require(float(frozen.get("lambda")) == LAMBDA, "Frozen lambda != 5")
    require(
        fusion_result.get("lambda_selection_performed") is False,
        "TEST-side lambda selection detected",
    )
    require(
        fusion_result.get("protocol_retuning_performed") is False,
        "TEST-side protocol retuning detected",
    )

    rules = read_jsonl(RULES)
    cf = [
        r for r in rules
        if r.get("rule_type") == "counterfactual"
        and bool(r.get("eligible_for_reliability"))
    ]
    require(len(cf) == 6172, f"Expected 6172 eligible CF candidates, found {len(cf)}")

    stage_rows = read_csv(STAGE_OUTPUTS)
    s1 = [r for r in stage_rows if r["stage"] == "Stage1"]
    s2 = [r for r in stage_rows if r["stage"] == "Stage2"]
    s3 = [r for r in stage_rows if r["stage"] == "Stage3"]

    require(len(s1) == 5056, f"Stage1 row count mismatch: {len(s1)}")
    require(len(s2) == 4287, f"Stage2 row count mismatch: {len(s2)}")
    require(len(s3) == 3496, f"Stage3 row count mismatch: {len(s3)}")
    require(sum(to_bool(r["actually_fused"]) for r in s1) == 919, "Stage1 fused count mismatch")
    require(sum(to_bool(r["actually_fused"]) for r in s2) == 614, "Stage2 fused count mismatch")
    require(sum(to_bool(r["actually_fused"]) for r in s3) == 529, "Stage3 fused count mismatch")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # A) Population overview
    # ------------------------------------------------------------------
    overview = [
        summarize_units("Original eligible CF candidates", cf, source=True),
        summarize_units("Stage1 all outputs", s1, source=False),
        summarize_units(
            "Stage1 genuinely fused outputs",
            [r for r in s1 if to_bool(r["actually_fused"])],
            source=False,
        ),
        summarize_units("Stage2 all outputs", s2, source=False),
        summarize_units(
            "Stage2 genuinely fused outputs",
            [r for r in s2 if to_bool(r["actually_fused"])],
            source=False,
        ),
        summarize_units("Stage3 all outputs", s3, source=False),
        summarize_units(
            "Stage3 genuinely fused outputs",
            [r for r in s3 if to_bool(r["actually_fused"])],
            source=False,
        ),
    ]

    # ------------------------------------------------------------------
    # B) Matched transition comparisons
    # ------------------------------------------------------------------
    original_s1_groups = defaultdict(list)
    for r in cf:
        original_s1_groups[stage1_key_from_source(r)].append(r)

    s1_index = {stage1_key_from_output(r): r for r in s1}
    require(len(s1_index) == len(s1), "Duplicate Stage1 output key")

    s1_to_s2_groups = defaultdict(list)
    for r in s1:
        s1_to_s2_groups[stage2_key_from_s1(r)].append(r)

    s2_index = {stage2_key_from_output(r): r for r in s2}
    require(len(s2_index) == len(s2), "Duplicate Stage2 output key")

    s2_to_s3_groups = defaultdict(list)
    for r in s2:
        s2_to_s3_groups[stage3_key_from_s2(r)].append(r)

    s3_index = {stage3_key_from_output(r): r for r in s3}
    require(len(s3_index) == len(s3), "Duplicate Stage3 output key")

    detail = []
    detail.extend(
        matched_transition(
            "Original->Stage1",
            original_s1_groups,
            s1_index,
            source=True,
        )
    )
    detail.extend(
        matched_transition(
            "Stage1->Stage2",
            s1_to_s2_groups,
            s2_index,
            source=False,
        )
    )
    detail.extend(
        matched_transition(
            "Stage2->Stage3",
            s2_to_s3_groups,
            s3_index,
            source=False,
        )
    )

    trans_summary = transition_summary(detail)

    # Hard-count checks against audited genuinely fusable groups.
    ts = {r["transition"]: r for r in trans_summary}
    require(ts["Original->Stage1"]["matched_group_count"] == 919, "Matched Stage1 count != 919")
    require(ts["Stage1->Stage2"]["matched_group_count"] == 614, "Matched Stage2 count != 614")
    require(ts["Stage2->Stage3"]["matched_group_count"] == 529, "Matched Stage3 count != 529")

    # ------------------------------------------------------------------
    # C) Prefix and original-source breakdowns
    # ------------------------------------------------------------------
    by_prefix = prefix_summary(stage_rows)
    orig_breakdown = original_breakdown(cf)

    write_csv(OUT_DIR / "formal_stage_population_overview.csv", overview)
    write_csv(OUT_DIR / "formal_matched_transition_detail.csv", detail)
    write_csv(OUT_DIR / "formal_matched_transition_summary.csv", trans_summary)
    write_csv(OUT_DIR / "formal_fused_stage_by_prefix.csv", by_prefix)
    write_csv(OUT_DIR / "formal_original_by_model_prefix_surrogate.csv", orig_breakdown)

    result = {
        "scope": "ETCE Phase 4D-6 formal Original -> Stage1 -> Stage2 -> Stage3 comparison",
        "status": "passed",
        "formal_test": True,
        "lambda": LAMBDA,
        "protocol_retuning_performed": False,
        "lambda_selection_performed": False,
        "phase4d5_independent_validation_required_and_passed": True,
        "comparison_policy": {
            "population_overview": (
                "Descriptive only; each row summarizes its own natural population. "
                "Do not interpret raw differences as causal fusion gains because N changes."
            ),
            "matched_transition": (
                "For each genuinely fusable compatibility group, pre = arithmetic mean "
                "of the actual evidence units entering that stage; post = the fused output "
                "for the same group."
            ),
            "primary_quality_metric": "Rule Fidelity on frozen held-out local evaluation data",
        },
        "population_overview": overview,
        "matched_transition_summary": trans_summary,
        "counts": {
            "original_eligible_cf_candidates": len(cf),
            "stage1_outputs": len(s1),
            "stage1_genuinely_fused": 919,
            "stage2_outputs": len(s2),
            "stage2_genuinely_fused": 614,
            "stage3_outputs": len(s3),
            "stage3_genuinely_fused": 529,
        },
        "next_step": (
            "Use the matched comparison to interpret stage-wise fusion effects; "
            "then run predeclared fusion baselines and BPA ablations."
        ),
    }
    write_json(OUT_DIR / "formal_stage_comparison.json", result)

    # Report
    lines = [
        "# ETCE Phase 4D-6 Formal Stage Comparison",
        "",
        "This report uses only independently validated formal TEST outputs.",
        "",
        "## Important comparison rule",
        "",
        "Population-level rows have different sample counts and are descriptive only. "
        "Fusion-effect conclusions should use the matched transition table.",
        "",
        "## Population overview",
        "",
        "| Scope | N | Fidelity | CF-valid | Support | Coverage | Reliability | Ignorance | Model-valid | All-model-valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in overview:
        lines.append(
            f"| {r['scope']} | {r['N']} | {fmt(r['mean_fidelity'])} | "
            f"{fmt(r['cf_validity_rate'])} | {fmt(r['mean_support'])} | "
            f"{fmt(r['mean_coverage'])} | {fmt(r['mean_reliability'])} | "
            f"{fmt(r['mean_ignorance'])} | {fmt(r['mean_model_cf_validity_rate'])} | "
            f"{fmt(r['all_models_cf_valid_rate'])} |"
        )

    lines.extend([
        "",
        "## Matched fusion transitions",
        "",
        "| Transition | N | Pre Fidelity | Post Fidelity | Δ Fidelity | Improved | Tied | Worse | Improved rate | Pre R | Post R | ΔR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for r in trans_summary:
        lines.append(
            f"| {r['transition']} | {r['matched_group_count']} | "
            f"{fmt(r['mean_pre_fidelity'])} | {fmt(r['mean_post_fidelity'])} | "
            f"{fmt(r['mean_delta_fidelity'])} | "
            f"{r['fidelity_improved_count']} | {r['fidelity_tied_count']} | "
            f"{r['fidelity_worse_count']} | {fmt(r['fidelity_improved_rate'])} | "
            f"{fmt(r['mean_pre_reliability'])} | {fmt(r['mean_post_reliability'])} | "
            f"{fmt(r['mean_delta_reliability'])} |"
        )

    lines.extend([
        "",
        "## Interpretation constraint",
        "",
        "Stage 3 re-evaluates the selected rule across multiple black-box-specific held-out "
        "contexts. Its Fidelity therefore answers a stricter cross-model question than "
        "Stage 1/2 single-model Fidelity. A lower Stage-3 Fidelity must not be described "
        "as a simple performance degradation without considering the simultaneous "
        "model-level and all-model counterfactual-validity results.",
    ])

    (OUT_DIR / "PHASE4D_FORMAL_STAGE_COMPARISON_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 116)
    print("ETCE PHASE 4D-6 FORMAL ORIGINAL -> STAGE1 -> STAGE2 -> STAGE3 COMPARISON PASSED")
    print("=" * 116)
    print(f"Original eligible CF candidates: {len(cf)}")
    print(f"Stage1 outputs/fused: {len(s1)}/919")
    print(f"Stage2 outputs/fused: {len(s2)}/614")
    print(f"Stage3 outputs/fused: {len(s3)}/529")
    print()
    print("[Matched fusion effects: Rule Fidelity]")
    for r in trans_summary:
        print(
            f"{r['transition']} | N={r['matched_group_count']} | "
            f"pre={r['mean_pre_fidelity']:.6f} | "
            f"post={r['mean_post_fidelity']:.6f} | "
            f"delta={r['mean_delta_fidelity']:+.6f} | "
            f"improved/tied/worse="
            f"{r['fidelity_improved_count']}/"
            f"{r['fidelity_tied_count']}/"
            f"{r['fidelity_worse_count']}"
        )
    print()
    print(f"Output: {OUT_DIR}")
    print("Protocol retuning performed: False")
    print("Lambda selection performed: False")
    print("NEXT: interpret matched results, then fusion baselines / BPA ablations.")


if __name__ == "__main__":
    main()
