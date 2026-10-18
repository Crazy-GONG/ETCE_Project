#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4C-BL0 / Script 41
Validation-only Fusion Baseline Protocol Freeze

Purpose
-------
Freeze the rule-level fusion baseline definitions and validation-derived
source weights BEFORE those baselines are applied to the formal TEST data.

Important chronology note
-------------------------
The primary frozen ETCE D-S formal TEST has already been executed. Therefore,
this file truthfully records the baseline protocol as a post-primary-test
definition. However, this script itself:
- reads validation artifacts only;
- does not read formal TEST files;
- does not use formal TEST outcomes to select weights or parameters;
- does not modify the already frozen ETCE D-S protocol.

Run
---
cd D:/ETCE
python -u ./scripts/41_phase4c_freeze_fusion_baseline_protocol.py
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]

RULES = (
    ROOT
    / "results/phase4/phase4b_v2_multirule_bank"
    / "canonical_rules_calibration.jsonl"
)
PHASE4B_VALIDATION = (
    ROOT
    / "results/phase4/phase4b_v2_multirule_bank"
    / "validation_phase4b_v2.json"
)
GROUP_AUDIT = (
    ROOT
    / "results/phase4/phase4c_v2_target_local_audit"
    / "target_local_group_audit_v2.json"
)
CACHE_VALIDATION = (
    ROOT
    / "results/phase4/phase4c_v2_fusion_eval_cache"
    / "validation_fusion_eval_cache.json"
)
DS_FROZEN = ROOT / "configs/phase4c_ds_frozen_config.json"

OUT_CONFIG = ROOT / "configs/phase4c_fusion_baselines_frozen_config.json"
OUT_REPORT = (
    ROOT
    / "results/phase4/phase4c_v2_fusion_baseline_protocol"
    / "PHASE4C_FUSION_BASELINES_FROZEN_PROTOCOL.md"
)
OUT_SUMMARY = (
    ROOT
    / "results/phase4/phase4c_v2_fusion_baseline_protocol"
    / "validation_source_weight_summary.json"
)

LAMBDA = 5.0
EXPECTED_ELIGIBLE_CF = 6997
SURROGATE_ORDER = ["LORE", "DT", "BN"]
MODEL_ORDER = ["RF", "XGBoost", "MLP", "LSTM"]
EPS = 1e-12


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid JSONL at line {line_no}: {path}"
                ) from exc
    return rows


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(xs):
    vals = [
        float(x)
        for x in xs
        if x is not None and math.isfinite(float(x))
    ]
    return sum(vals) / len(vals) if vals else None


def median(xs):
    vals = [
        float(x)
        for x in xs
        if x is not None and math.isfinite(float(x))
    ]
    return float(statistics.median(vals)) if vals else None


def reliability(rule):
    n = int(rule["n_r"])
    fid = float(rule["Fid_r"])
    rho = n / (n + LAMBDA)
    return float(rho * fid)


def normalize_weights(raw, ordered_keys):
    vals = {
        k: float(raw[k])
        for k in ordered_keys
    }
    total = sum(vals.values())
    require(total > 0, "Cannot normalize zero source weights")
    return {
        k: float(vals[k] / total)
        for k in ordered_keys
    }


def source_summary(rows, key_name, ordered_keys):
    grouped = defaultdict(list)

    for r in rows:
        grouped[str(r[key_name])].append(r)

    out = {}
    for key in ordered_keys:
        sub = grouped.get(key, [])
        require(sub, f"No eligible validation CF rules for {key_name}={key}")

        rels = [reliability(r) for r in sub]
        fids = [float(r["Fid_r"]) for r in sub]
        supports = [int(r["n_r"]) for r in sub]
        coverage = [float(r["coverage_eval"]) for r in sub]

        out[key] = {
            "eligible_cf_count": len(sub),
            "mean_fidelity": mean(fids),
            "median_fidelity": median(fids),
            "mean_support": mean(supports),
            "median_support": median(supports),
            "mean_coverage": mean(coverage),
            "mean_reliability_lambda5": mean(rels),
            "median_reliability_lambda5": median(rels),
        }

    return out


def select_best_source(summary, ordered_keys):
    """
    Validation-only deterministic selection:
    1) highest mean R at frozen lambda=5
    2) highest mean held-out Fidelity
    3) highest eligible-CF count
    4) frozen source order
    """
    rank = {k: i for i, k in enumerate(ordered_keys)}

    return sorted(
        ordered_keys,
        key=lambda k: (
            -float(summary[k]["mean_reliability_lambda5"]),
            -float(summary[k]["mean_fidelity"]),
            -int(summary[k]["eligible_cf_count"]),
            rank[k],
        ),
    )[0]


def find_probability_fields(rows):
    """
    Detect whether the canonical rule bank already contains a genuine
    surrogate class-probability field that could support probability averaging.

    We intentionally do NOT treat Fid_r / R_r / coverage as class probabilities.
    """
    names = set()
    for r in rows[: min(200, len(rows))]:
        names.update(map(str, r.keys()))

    candidates = []
    for name in sorted(names):
        low = name.lower()
        if (
            "prob" in low
            and name not in {
                "factual_probability",
            }
        ):
            candidates.append(name)
    return candidates


def main():
    required = [
        RULES,
        PHASE4B_VALIDATION,
        GROUP_AUDIT,
        CACHE_VALIDATION,
        DS_FROZEN,
    ]
    missing = [str(p) for p in required if not p.is_file()]
    require(
        not missing,
        "Missing required validation artifact(s):\n" + "\n".join(missing),
    )

    # Explicit guard: this script must not use a formal-test path as input.
    for p in required:
        low = str(p).replace("\\", "/").lower()
        require(
            "phase4d_formal_test" not in low
            and "formal_test_multirule_bank" not in low,
            f"Formal TEST path supplied to validation-only baseline freeze: {p}",
        )

    phase4b = read_json(PHASE4B_VALIDATION)
    audit = read_json(GROUP_AUDIT)
    cache_val = read_json(CACHE_VALIDATION)
    ds = read_json(DS_FROZEN)

    require(
        phase4b.get("status") == "passed",
        "Phase 4B-v2 validation is not PASSED",
    )
    require(
        phase4b.get("test_cases_used") is False,
        "Phase 4B-v2 validation reports test usage",
    )
    require(
        cache_val.get("status") == "passed",
        "Validation fusion cache is not PASSED",
    )
    require(
        cache_val.get("test_cases_used") is False,
        "Validation fusion cache reports test usage",
    )
    require(
        ds.get("status") == "frozen",
        "Primary D-S protocol is not frozen",
    )
    require(
        float(ds.get("lambda")) == LAMBDA,
        "Primary frozen lambda is not 5",
    )
    require(
        ds.get("selection_scope") == "validation_only",
        "Primary D-S lambda was not selected validation-only",
    )

    rules = read_jsonl(RULES)
    cf = [
        r for r in rules
        if r.get("rule_type") == "counterfactual"
        and bool(r.get("eligible_for_reliability"))
    ]

    require(
        len(cf) == EXPECTED_ELIGIBLE_CF,
        f"Expected {EXPECTED_ELIGIBLE_CF} eligible validation CF rules, found {len(cf)}",
    )

    surrogate_summary = source_summary(
        cf,
        "surrogate",
        SURROGATE_ORDER,
    )
    model_summary = source_summary(
        cf,
        "model",
        MODEL_ORDER,
    )

    best_surrogate = select_best_source(
        surrogate_summary,
        SURROGATE_ORDER,
    )
    best_model = select_best_source(
        model_summary,
        MODEL_ORDER,
    )

    surrogate_raw = {
        k: surrogate_summary[k]["mean_reliability_lambda5"]
        for k in SURROGATE_ORDER
    }
    model_raw = {
        k: model_summary[k]["mean_reliability_lambda5"]
        for k in MODEL_ORDER
    }

    surrogate_weights = normalize_weights(
        surrogate_raw,
        SURROGATE_ORDER,
    )
    model_weights = normalize_weights(
        model_raw,
        MODEL_ORDER,
    )

    probability_fields = find_probability_fields(cf)
    probability_averaging_supported = bool(probability_fields)

    protocol = {
        "schema_version": "etce-phase4c-fusion-baselines-frozen-v1",
        "status": "frozen",
        "scope": "validation-only parameterization of post-primary-test fusion baselines",
        "chronology": {
            "primary_etce_formal_test_already_executed": True,
            "baseline_protocol_defined_after_primary_test_execution": True,
            "formal_test_files_read_by_this_script": False,
            "formal_test_outcomes_used_for_parameter_selection": False,
            "reporting_requirement": (
                "Describe these fusion baselines as post-hoc comparison protocols "
                "parameterized only on validation data; do not claim they were "
                "pre-registered before the primary formal TEST."
            ),
        },
        "common_protocol": {
            "lambda_for_R": LAMBDA,
            "reliability": "R_i = n_i/(n_i+5) * Fid_i",
            "same_grouping_as_etce": ds["grouping"],
            "same_stage_order_as_etce": {
                "stage1": "candidate_id",
                "stage2": SURROGATE_ORDER,
                "stage3": MODEL_ORDER,
            },
            "same_fused_rule_re_evaluation_policy": (
                "After a baseline selects one interpretable rule hypothesis for a "
                "group, recompute n_r, Fid(r), coverage, and R(r) on the same frozen "
                "held-out cache before propagating it to the next hierarchy level."
            ),
            "candidate_hypothesis_policy": (
                "Non-D-S baselines select from hypotheses already present among the "
                "group inputs. They do not synthesize new intersection hypotheses. "
                "The ability of D-S to create a supported intersection is therefore "
                "part of the method difference, not silently granted to baselines."
            ),
            "global_tie_break": (
                "higher aggregation score -> higher input R -> more specific rule -> "
                "lexical canonical rule"
            ),
            "no_test_side_tuning": True,
        },
        "validation_selected_sources": {
            "best_single_surrogate": {
                "selected": best_surrogate,
                "selection_rule": (
                    "highest validation mean R; tie -> mean Fidelity -> "
                    "eligible-CF count -> frozen surrogate order"
                ),
            },
            "best_single_model": {
                "selected": best_model,
                "selection_rule": (
                    "highest validation mean R; tie -> mean Fidelity -> "
                    "eligible-CF count -> frozen model order"
                ),
            },
        },
        "validation_fixed_weights": {
            "surrogate_weights": surrogate_weights,
            "surrogate_weight_basis": (
                "normalized validation mean support-aware reliability R at lambda=5"
            ),
            "model_weights": model_weights,
            "model_weight_basis": (
                "normalized validation mean support-aware reliability R at lambda=5"
            ),
        },
        "baselines": {
            "best_single_evidence": {
                "status": "enabled",
                "definition": (
                    "Select the input rule with maximum R_i. "
                    "Tie -> more specific -> lexical canonical rule."
                ),
                "parameters": {},
            },
            "validation_best_source": {
                "status": "enabled",
                "definition": (
                    "At Stage2 prefer the validation-selected best surrogate when present; "
                    "at Stage3 prefer the validation-selected best black-box model when present. "
                    "If absent, fall back to maximum-R input with the global tie-break."
                ),
                "parameters": {
                    "best_surrogate": best_surrogate,
                    "best_model": best_model,
                },
            },
            "uniform_vote": {
                "status": "enabled",
                "definition": (
                    "Group inputs by exact canonical rule hypothesis; score(H)=number "
                    "of input evidence units with hypothesis H. Select maximum score."
                ),
                "parameters": {},
            },
            "validation_weighted_vote": {
                "status": "enabled",
                "definition": (
                    "Group inputs by exact canonical hypothesis. "
                    "Stage1 score(H)=sum R_i. "
                    "Stage2 score(H)=sum[R_i * w_surrogate(i)]. "
                    "Stage3 score(H)=sum[R_i * w_model(i)]."
                ),
                "parameters": {
                    "surrogate_weights": surrogate_weights,
                    "model_weights": model_weights,
                },
            },
            "fuzzy_probabilistic_or": {
                "status": "enabled",
                "definition": (
                    "For each exact candidate hypothesis H, each supporting input has "
                    "membership mu_i=clip(R_i,0,1). Aggregate repeated support with the "
                    "parameter-free probabilistic fuzzy OR: "
                    "mu(H)=1-product_i(1-mu_i). Select maximum mu(H)."
                ),
                "parameters": {},
            },
            "bayesian_categorical_noise": {
                "status": "enabled",
                "definition": (
                    "Candidate states are the unique input hypotheses. Use a uniform "
                    "prior. For M unique hypotheses and evidence i with reliability R_i: "
                    "P(e_i|H)=R_i if H equals the input hypothesis; otherwise "
                    "(1-R_i)/(M-1). Compute posterior scores in log space with epsilon "
                    "guard and select maximum posterior."
                ),
                "parameters": {
                    "prior": "uniform_over_unique_input_hypotheses",
                    "epsilon": EPS,
                },
            },
            "probability_averaging": {
                "status": (
                    "enabled"
                    if probability_averaging_supported
                    else "not_executable_from_current_canonical_rule_bank"
                ),
                "detected_probability_fields": probability_fields,
                "definition": (
                    "Only valid if comparable calibrated surrogate class probabilities "
                    "exist in the canonical evidence bank. Fid_r, R_r, coverage, and "
                    "factual black-box probability must NOT be relabeled as surrogate "
                    "class probabilities."
                ),
            },
            "ds_etce": {
                "status": "reference_method_frozen_elsewhere",
                "config": "configs/phase4c_ds_frozen_config.json",
                "definition": (
                    "Support-aware simple-support BPA with Dempster combination, "
                    "conflict K, ignorance Theta, deterministic representative-rule "
                    "selection, and held-out re-evaluation."
                ),
            },
        },
        "evaluation": {
            "primary_scope": (
                "genuinely fusable compatible groups at each hierarchy stage"
            ),
            "primary_metrics": [
                "Rule Fidelity",
                "CF Validity",
                "support",
                "coverage",
                "reliability",
                "min witness edit",
            ],
            "stage3_metrics": [
                "Rule Fidelity",
                "CF Validity",
                "model-level CF validity",
                "all-model CF validity",
            ],
            "comparison_requirement": (
                "Use the same validation/test compatibility groups and the same held-out "
                "cache for every operator; report counts and matched comparisons."
            ),
        },
        "provenance": {
            "canonical_rules_validation_sha256": sha256(RULES),
            "phase4b_validation_sha256": sha256(PHASE4B_VALIDATION),
            "group_audit_sha256": sha256(GROUP_AUDIT),
            "fusion_cache_validation_sha256": sha256(CACHE_VALIDATION),
            "primary_ds_frozen_config_sha256": sha256(DS_FROZEN),
        },
        "freeze_rule": (
            "After this config is written, baseline formulas, validation-selected "
            "source weights, source choices, tie-breaks, and epsilon must not be "
            "changed in response to formal TEST baseline results."
        ),
    }

    OUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT_CONFIG, protocol)

    summary = {
        "status": "passed",
        "formal_test_files_read": False,
        "eligible_validation_cf_count": len(cf),
        "lambda": LAMBDA,
        "surrogate_summary": surrogate_summary,
        "model_summary": model_summary,
        "best_single_surrogate": best_surrogate,
        "best_single_model": best_model,
        "surrogate_weights": surrogate_weights,
        "model_weights": model_weights,
        "probability_fields_detected": probability_fields,
        "probability_averaging_supported": probability_averaging_supported,
    }
    write_json(OUT_SUMMARY, summary)

    def fmt(v):
        return f"{float(v):.6f}"

    lines = [
        "# ETCE Validation-only Fusion Baseline Protocol Freeze",
        "",
        "> No formal TEST file is read by this script.",
        "",
        "## Chronology disclosure",
        "",
        "The primary ETCE formal TEST had already been executed before this "
        "baseline protocol was defined. Therefore these baselines are post-hoc "
        "comparison protocols, although all their parameters below are derived "
        "from validation data only.",
        "",
        "## Validation source summaries",
        "",
        "### Surrogates",
        "",
        "| Source | N | mean Fidelity | mean R(lambda=5) | normalized weight |",
        "|---|---:|---:|---:|---:|",
    ]

    for s in SURROGATE_ORDER:
        z = surrogate_summary[s]
        lines.append(
            f"| {s} | {z['eligible_cf_count']} | "
            f"{fmt(z['mean_fidelity'])} | "
            f"{fmt(z['mean_reliability_lambda5'])} | "
            f"{fmt(surrogate_weights[s])} |"
        )

    lines.extend([
        "",
        f"Validation-selected best surrogate: **{best_surrogate}**",
        "",
        "### Black-box models",
        "",
        "| Model | N | mean Fidelity | mean R(lambda=5) | normalized weight |",
        "|---|---:|---:|---:|---:|",
    ])

    for m in MODEL_ORDER:
        z = model_summary[m]
        lines.append(
            f"| {m} | {z['eligible_cf_count']} | "
            f"{fmt(z['mean_fidelity'])} | "
            f"{fmt(z['mean_reliability_lambda5'])} | "
            f"{fmt(model_weights[m])} |"
        )

    lines.extend([
        "",
        f"Validation-selected best black-box source: **{best_model}**",
        "",
        "## Frozen baseline operators",
        "",
        "- Best Single Evidence",
        "- Validation-best source",
        "- Uniform Vote",
        "- Validation-weighted Vote",
        "- Fuzzy probabilistic OR",
        "- Bayesian categorical-noise pooling",
        "- D-S ETCE reference",
        "",
        "Probability averaging is enabled only if genuine comparable surrogate "
        "class-probability fields are present. Reliability/fidelity values are "
        "not relabeled as probabilities.",
        "",
        "## Freeze",
        "",
        "No baseline definition or weight may be changed after formal TEST "
        "baseline results are inspected.",
    ])

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 116)
    print("ETCE PHASE 4C-BL0 VALIDATION-ONLY FUSION BASELINE PROTOCOL FREEZE PASSED")
    print("=" * 116)
    print(f"Eligible validation CF candidates: {len(cf)}")
    print(f"Frozen lambda inherited from ETCE: {LAMBDA:g}")
    print()

    print("[Validation surrogate source summary]")
    for s in SURROGATE_ORDER:
        z = surrogate_summary[s]
        print(
            f"{s:8s} | N={z['eligible_cf_count']:4d} | "
            f"Fid={z['mean_fidelity']:.6f} | "
            f"R={z['mean_reliability_lambda5']:.6f} | "
            f"weight={surrogate_weights[s]:.6f}"
        )
    print(f"Best single surrogate: {best_surrogate}")

    print()
    print("[Validation black-box source summary]")
    for m in MODEL_ORDER:
        z = model_summary[m]
        print(
            f"{m:8s} | N={z['eligible_cf_count']:4d} | "
            f"Fid={z['mean_fidelity']:.6f} | "
            f"R={z['mean_reliability_lambda5']:.6f} | "
            f"weight={model_weights[m]:.6f}"
        )
    print(f"Best single model source: {best_model}")

    print()
    print("[Probability averaging]")
    print(f"Detected probability fields: {probability_fields}")
    print(f"Executable from current canonical rule bank: {probability_averaging_supported}")

    print()
    print(f"Frozen config: {OUT_CONFIG}")
    print(f"Summary: {OUT_SUMMARY}")
    print(f"Report: {OUT_REPORT}")
    print("Formal TEST files read by this script: False")
    print("Formal TEST outcomes used for weights: False")
    print("Baseline protocol now frozen: True")
    print()
    print("NEXT: run the frozen baseline operators on the formal TEST compatible groups.")


if __name__ == "__main__":
    main()
