#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4C-v2:
Validation-only hierarchical Dempster-Shafer lambda sweep.

This script uses:
1) validated Phase 4B-v2 multi-rule canonical bank;
2) validated target-local grouping audit;
3) validated Phase 4C-v2 fusion evaluation cache.

It performs the three frozen target-local stages:
Stage 1:
  case × model × prefix × surrogate × predicted_class × feature_signature
Stage 2:
  case × model × prefix × predicted_class × feature_signature
Stage 3:
  case × prefix × predicted_class × feature_signature

Reliability:
  rho_r = n_r / (n_r + lambda)
  R(r)  = rho_r * Fid(r)

Simple-support BPA for a rule hypothesis H:
  m(H)     = R(r)
  m(Theta) = 1 - R(r)

Pairwise rule fusion:
- condition hypotheses are categorical position-wise sets;
- intersection is the rule-level conjunction;
- Dempster conflict K is the mass assigned to the empty intersection;
- if K is numerically total (1-K <= 1e-12), the pair is mapped to
  total ignorance rather than forcing an unstable normalization;
- otherwise Dempster normalization is applied;
- the highest-mass proper focal condition is selected deterministically
  as the interpretable fused rule;
- the selected fused rule is re-evaluated on the already-frozen
  held-out local evaluation cache, and its n_r / Fid(r) / R(r) are
  recomputed before the next hierarchy level.

Important:
- validation only;
- no formal test data;
- no lambda is automatically selected/frozen here;
- deterministic fusion order:
    Stage1 candidate_id;
    Stage2 LORE -> DT -> BN;
    Stage3 RF -> XGBoost -> MLP -> LSTM.

Run:
  python -u scripts/31_phase4c_v2_hierarchical_ds_lambda_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import statistics

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = (
    DEFAULT_ROOT
    / "results/phase4/phase4b_v2_multirule_bank/canonical_rules_calibration.jsonl"
)
DEFAULT_VALIDATION = (
    DEFAULT_ROOT
    / "results/phase4/phase4b_v2_multirule_bank/validation_phase4b_v2.json"
)
DEFAULT_AUDIT = (
    DEFAULT_ROOT
    / "results/phase4/phase4c_v2_target_local_audit/target_local_group_audit_v2.json"
)
DEFAULT_CACHE_DIR = (
    DEFAULT_ROOT
    / "results/phase4/phase4c_v2_fusion_eval_cache"
)
DEFAULT_CACHE_VALIDATION = (
    DEFAULT_CACHE_DIR / "validation_fusion_eval_cache.json"
)
DEFAULT_OUT = (
    DEFAULT_ROOT
    / "results/phase4/phase4c_v2_hierarchical_ds_sweep"
)
DEFAULT_LAMBDAS = [1, 5, 10, 20, 50]

SURROGATE_ORDER = {"LORE": 0, "DT": 1, "BN": 2}
MODEL_ORDER = {"RF": 0, "XGBoost": 1, "MLP": 2, "LSTM": 3}
K_EPS = 1e-12
THETA = None
EMPTY = object()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
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
                    f"Invalid JSONL at line {line_no}: {path}"
                ) from exc
    return rows


def read_gzip_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def q(values, p):
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    x = (len(vals) - 1) * float(p)
    lo = int(math.floor(x))
    hi = int(math.ceil(x))
    if lo == hi:
        return vals[lo]
    frac = x - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def mean(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def median(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.median(vals)) if vals else None


def canonical_values(values):
    return tuple(sorted(str(v) for v in values))


def conditions_to_hypothesis(conditions):
    """
    Symbolic rule hypothesis:
      ((position, (allowed_value_1, ...)), ...)
    THETA is represented separately by None.
    """
    by_pos = {}
    for cond in conditions:
        require(
            cond.get("operator") == "in",
            f"Unsupported canonical operator: {cond.get('operator')}",
        )
        pos = int(cond["position"])
        vals = canonical_values(cond["values"])
        require(vals, f"Empty condition values at position {pos}")
        require(pos not in by_pos, f"Duplicate position {pos} in one rule")
        by_pos[pos] = vals
    return tuple((p, by_pos[p]) for p in sorted(by_pos))


def hypothesis_to_conditions(hyp):
    if hyp is THETA:
        return []
    return [
        {
            "position": int(pos),
            "operator": "in",
            "values": list(vals),
        }
        for pos, vals in hyp
    ]


def feature_signature(hyp):
    if hyp is THETA:
        return tuple()
    return tuple(int(pos) for pos, _ in hyp)


def hyp_key(hyp):
    if hyp is THETA:
        return "__THETA__"
    return json.dumps(
        hypothesis_to_conditions(hyp),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def hypothesis_specificity(hyp):
    if hyp is THETA:
        return 10**9
    return sum(len(vals) for _, vals in hyp)


def intersect_hyp(a, b):
    if a is THETA:
        return b
    if b is THETA:
        return a

    da = {int(p): set(vals) for p, vals in a}
    db = {int(p): set(vals) for p, vals in b}
    out = {}

    for p in sorted(set(da) | set(db)):
        if p in da and p in db:
            vals = da[p] & db[p]
            if not vals:
                return EMPTY
            out[p] = tuple(sorted(vals))
        elif p in da:
            out[p] = tuple(sorted(da[p]))
        else:
            out[p] = tuple(sorted(db[p]))

    return tuple((p, out[p]) for p in sorted(out))


def initial_mass(hyp, reliability):
    r = min(1.0, max(0.0, float(reliability)))
    out = defaultdict(float)
    out[hyp] += r
    out[THETA] += 1.0 - r
    return dict(out)


def combine_mass(m1, m2):
    raw = defaultdict(float)
    conflict = 0.0

    for a, ma in m1.items():
        for b, mb in m2.items():
            prod = float(ma) * float(mb)
            inter = intersect_hyp(a, b)
            if inter is EMPTY:
                conflict += prod
            else:
                raw[inter] += prod

    if 1.0 - conflict <= K_EPS:
        return {THETA: 1.0}, float(conflict), True

    denom = 1.0 - conflict
    normalized = {
        h: float(v / denom)
        for h, v in raw.items()
        if v > 0.0
    }

    total = sum(normalized.values())
    if total > 0:
        normalized = {
            h: float(v / total)
            for h, v in normalized.items()
        }

    return normalized, float(conflict), False


def select_interpretable_hypothesis(mass):
    proper = [
        (h, float(v))
        for h, v in mass.items()
        if h is not THETA and float(v) > 0.0
    ]
    if not proper:
        return None, 0.0

    # Primary: highest normalized DS mass.
    # Tie-break: more specific condition, then lexical canonical form.
    proper.sort(
        key=lambda hv: (
            -hv[1],
            hypothesis_specificity(hv[0]),
            hyp_key(hv[0]),
        )
    )
    return proper[0]


def applies(sequence, hyp):
    if hyp is THETA:
        return True
    for pos, vals in hyp:
        idx = int(pos) - 1
        if idx < 0 or idx >= len(sequence):
            return False
        if str(sequence[idx]) not in set(vals):
            return False
    return True


def hamming(a, b):
    require(len(a) == len(b), "Hamming length mismatch")
    return sum(x != y for x, y in zip(a, b))


def load_cache_index(cache_dir: Path):
    target_dir = cache_dir / "targets"
    require(target_dir.is_dir(), f"Cache target dir missing: {target_dir}")

    index = {}
    for path in sorted(target_dir.glob("*.json.gz")):
        payload = read_gzip_json(path)
        key = (
            payload["case_id"],
            payload["model"],
            int(payload["prefix"]),
        )
        require(key not in index, f"Duplicate cache target: {key}")
        index[key] = payload
    return index


def evaluate_hypothesis(hyp, predicted_class, contexts):
    """
    contexts: iterable of cache payloads, possibly from multiple black-boxes.

    Returns pooled rule metrics plus per-model CF-validity diagnostics.
    """
    total_eval = 0
    total_support = 0
    total_correct = 0
    min_edit = None
    per_model = []

    for payload in contexts:
        sequences = payload["pool_sequences"]
        labels = payload["pool_labels"]
        eval_indices = payload["eval_indices"]
        factual = payload["factual_sequence"]

        m_support = 0
        m_correct = 0
        m_min_edit = None

        for i in eval_indices:
            seq = sequences[int(i)]
            total_eval += 1
            if not applies(seq, hyp):
                continue
            total_support += 1
            m_support += 1

            if int(labels[int(i)]) == int(predicted_class):
                total_correct += 1
                m_correct += 1
                ed = hamming(factual, seq)
                if min_edit is None or ed < min_edit:
                    min_edit = ed
                if m_min_edit is None or ed < m_min_edit:
                    m_min_edit = ed

        per_model.append(
            {
                "model": payload["model"],
                "eval_size": len(eval_indices),
                "support": m_support,
                "correct": m_correct,
                "fidelity": (
                    float(m_correct / m_support)
                    if m_support > 0
                    else None
                ),
                "cf_valid": bool(m_correct > 0),
                "min_witness_edit": m_min_edit,
            }
        )

    fidelity = (
        float(total_correct / total_support)
        if total_support > 0
        else None
    )
    coverage = (
        float(total_support / total_eval)
        if total_eval > 0
        else 0.0
    )
    model_validity_rate = (
        float(sum(int(x["cf_valid"]) for x in per_model) / len(per_model))
        if per_model
        else None
    )

    return {
        "n_r": int(total_support),
        "Fid_r": fidelity,
        "coverage_eval": coverage,
        "correct_count": int(total_correct),
        "cf_valid": bool(total_correct > 0),
        "model_cf_validity_rate": model_validity_rate,
        "all_models_cf_valid": bool(
            per_model and all(x["cf_valid"] for x in per_model)
        ),
        "min_witness_edit": min_edit,
        "per_model": per_model,
    }


def reliability_from_metrics(metrics, lam):
    n = int(metrics["n_r"])
    fid = metrics["Fid_r"]
    if n <= 0 or fid is None:
        return 0.0
    rho = float(n / (n + float(lam)))
    return float(rho * float(fid))


def source_rule(rule, lam):
    n = int(rule["n_r"])
    fid = float(rule["Fid_r"])
    rho = n / (n + float(lam))
    rel = rho * fid
    hyp = conditions_to_hypothesis(rule["conditions"])

    return {
        "hypothesis": hyp,
        "predicted_class": int(rule["predicted_class"]),
        "n_r": n,
        "Fid_r": fid,
        "coverage_eval": float(rule["coverage_eval"]),
        "rho_r": float(rho),
        "R_r": float(rel),
        "source_count": 1,
        "source_models": {rule["model"]},
        "source_surrogates": {rule["surrogate"]},
        "source_candidate_ids": [rule["candidate_id"]],
        "last_selected_mass": float(rel),
        "last_ignorance_mass": float(1.0 - rel),
        "last_K": None,
        "total_conflict_fallback_count": 0,
        "fusion_steps": 0,
        "min_witness_edit": rule.get("witness_edit_count"),
        "cf_valid": True,
        "model_cf_validity_rate": 1.0,
        "all_models_cf_valid": True,
    }


def evidence_mass(ev):
    return initial_mass(ev["hypothesis"], ev["R_r"])


def pairwise_rule_fusion(left, right, contexts, lam, stage, trace_meta):
    require(
        int(left["predicted_class"]) == int(right["predicted_class"]),
        "Attempted to fuse different predicted classes",
    )
    require(
        feature_signature(left["hypothesis"])
        == feature_signature(right["hypothesis"]),
        "Attempted to fuse different feature signatures",
    )

    mass, K, fallback = combine_mass(
        evidence_mass(left),
        evidence_mass(right),
    )

    selected_hyp, selected_mass = select_interpretable_hypothesis(mass)

    if selected_hyp is None:
        # Total ignorance: preserve the stronger source rule only as a
        # transparent fallback representation, but set output reliability
        # to zero so it carries no committed evidence into the next stage.
        candidates = [left, right]
        candidates.sort(
            key=lambda ev: (
                -float(ev["R_r"]),
                hypothesis_specificity(ev["hypothesis"]),
                hyp_key(ev["hypothesis"]),
            )
        )
        selected_hyp = candidates[0]["hypothesis"]
        selected_mass = 0.0
        eval_metrics = evaluate_hypothesis(
            selected_hyp,
            left["predicted_class"],
            contexts,
        )
        out_rel = 0.0
        ignorance = 1.0
    else:
        eval_metrics = evaluate_hypothesis(
            selected_hyp,
            left["predicted_class"],
            contexts,
        )
        out_rel = reliability_from_metrics(eval_metrics, lam)
        ignorance = float(mass.get(THETA, 0.0))

    out = {
        "hypothesis": selected_hyp,
        "predicted_class": int(left["predicted_class"]),
        "n_r": int(eval_metrics["n_r"]),
        "Fid_r": eval_metrics["Fid_r"],
        "coverage_eval": float(eval_metrics["coverage_eval"]),
        "rho_r": (
            float(eval_metrics["n_r"] / (eval_metrics["n_r"] + float(lam)))
            if eval_metrics["n_r"] > 0
            else 0.0
        ),
        "R_r": float(out_rel),
        "source_count": int(left["source_count"] + right["source_count"]),
        "source_models": set(left["source_models"]) | set(right["source_models"]),
        "source_surrogates": (
            set(left["source_surrogates"]) | set(right["source_surrogates"])
        ),
        "source_candidate_ids": (
            list(left["source_candidate_ids"])
            + list(right["source_candidate_ids"])
        ),
        "last_selected_mass": float(selected_mass),
        "last_ignorance_mass": float(ignorance),
        "last_K": float(K),
        "total_conflict_fallback_count": int(
            left["total_conflict_fallback_count"]
            + right["total_conflict_fallback_count"]
            + int(fallback)
        ),
        "fusion_steps": int(left["fusion_steps"] + right["fusion_steps"] + 1),
        "min_witness_edit": eval_metrics["min_witness_edit"],
        "cf_valid": bool(eval_metrics["cf_valid"]),
        "model_cf_validity_rate": eval_metrics["model_cf_validity_rate"],
        "all_models_cf_valid": bool(eval_metrics["all_models_cf_valid"]),
    }

    trace = {
        **trace_meta,
        "stage": stage,
        "K": float(K),
        "total_conflict_fallback": bool(fallback),
        "left_R": float(left["R_r"]),
        "right_R": float(right["R_r"]),
        "selected_focal_mass": float(selected_mass),
        "ignorance_mass": float(ignorance),
        "output_n_r": int(out["n_r"]),
        "output_Fid_r": out["Fid_r"],
        "output_R_r": float(out["R_r"]),
        "output_cf_valid": bool(out["cf_valid"]),
        "selected_conditions": json.dumps(
            hypothesis_to_conditions(selected_hyp),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    return out, trace


def group_key_stage1(rule):
    hyp = conditions_to_hypothesis(rule["conditions"])
    return (
        rule["case_id"],
        rule["model"],
        int(rule["prefix"]),
        rule["surrogate"],
        int(rule["predicted_class"]),
        feature_signature(hyp),
    )


def ev_to_row(ev, lam, stage, key, input_count):
    case_id = key[0]
    prefix = None
    model = None
    surrogate = None
    predicted_class = None
    signature = None

    if stage == "Stage1":
        _, model, prefix, surrogate, predicted_class, signature = key
    elif stage == "Stage2":
        _, model, prefix, predicted_class, signature = key
    elif stage == "Stage3":
        _, prefix, predicted_class, signature = key
    else:
        raise ValueError(stage)

    return {
        "lambda": float(lam),
        "stage": stage,
        "case_id": case_id,
        "model": model,
        "prefix": int(prefix),
        "surrogate": surrogate,
        "predicted_class": int(predicted_class),
        "feature_signature": json.dumps(list(signature)),
        "input_count": int(input_count),
        "actually_fused": bool(input_count >= 2),
        "source_rule_count": int(ev["source_count"]),
        "source_model_count": int(len(ev["source_models"])),
        "source_surrogate_count": int(len(ev["source_surrogates"])),
        "n_r": int(ev["n_r"]),
        "Fid_r": ev["Fid_r"],
        "coverage_eval": float(ev["coverage_eval"]),
        "rho_r": float(ev["rho_r"]),
        "R_r": float(ev["R_r"]),
        "cf_valid": bool(ev["cf_valid"]),
        "model_cf_validity_rate": ev["model_cf_validity_rate"],
        "all_models_cf_valid": bool(ev["all_models_cf_valid"]),
        "min_witness_edit": ev["min_witness_edit"],
        "fusion_steps": int(ev["fusion_steps"]),
        "total_conflict_fallback_count": int(
            ev["total_conflict_fallback_count"]
        ),
        "last_K": ev["last_K"],
        "last_selected_mass": float(ev["last_selected_mass"]),
        "last_ignorance_mass": float(ev["last_ignorance_mass"]),
        "selected_conditions": json.dumps(
            hypothesis_to_conditions(ev["hypothesis"]),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def summarize_stage(rows, conflict_rows, stage, lam):
    sub = [
        r for r in rows
        if r["stage"] == stage and float(r["lambda"]) == float(lam)
    ]
    fused = [r for r in sub if r["actually_fused"]]
    cr = [
        r for r in conflict_rows
        if r["stage"] == stage and float(r["lambda"]) == float(lam)
    ]

    def scope_stats(scope, name):
        return {
            f"{name}_count": len(scope),
            f"{name}_mean_fidelity": mean([r["Fid_r"] for r in scope]),
            f"{name}_median_fidelity": median([r["Fid_r"] for r in scope]),
            f"{name}_cf_validity_rate": mean(
                [int(r["cf_valid"]) for r in scope]
            ),
            f"{name}_mean_model_cf_validity_rate": mean(
                [r["model_cf_validity_rate"] for r in scope]
            ),
            f"{name}_all_models_cf_valid_rate": mean(
                [int(r["all_models_cf_valid"]) for r in scope]
            ),
            f"{name}_mean_support": mean([r["n_r"] for r in scope]),
            f"{name}_mean_coverage": mean([r["coverage_eval"] for r in scope]),
            f"{name}_mean_reliability": mean([r["R_r"] for r in scope]),
            f"{name}_mean_ignorance": mean(
                [r["last_ignorance_mass"] for r in scope]
            ),
            f"{name}_mean_min_witness_edit": mean(
                [r["min_witness_edit"] for r in scope]
            ),
        }

    out = {
        "lambda": float(lam),
        "stage": stage,
        "group_count": len(sub),
        "fusable_group_count": len(fused),
        "pairwise_fusion_count": len(cr),
        "mean_conflict_K": mean([r["K"] for r in cr]),
        "median_conflict_K": median([r["K"] for r in cr]),
        "p90_conflict_K": q([r["K"] for r in cr], 0.90),
        "max_conflict_K": (
            max([float(r["K"]) for r in cr])
            if cr else None
        ),
        "total_conflict_fallback_count": sum(
            int(r["total_conflict_fallback"]) for r in cr
        ),
    }
    out.update(scope_stats(sub, "all"))
    out.update(scope_stats(fused, "fused"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=str(DEFAULT_RULES))
    ap.add_argument("--validation", default=str(DEFAULT_VALIDATION))
    ap.add_argument("--audit", default=str(DEFAULT_AUDIT))
    ap.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument(
        "--cache_validation",
        default=str(DEFAULT_CACHE_VALIDATION),
    )
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=DEFAULT_LAMBDAS,
    )
    args = ap.parse_args()

    rules_path = Path(args.rules).expanduser().resolve()
    validation_path = Path(args.validation).expanduser().resolve()
    audit_path = Path(args.audit).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_validation_path = Path(
        args.cache_validation
    ).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in (
        rules_path,
        validation_path,
        audit_path,
        cache_validation_path,
    ):
        require(p.is_file(), f"Required input missing: {p}")

    validation = read_json(validation_path)
    audit = read_json(audit_path)
    cache_validation = read_json(cache_validation_path)

    require(validation.get("status") == "passed", "Phase 4B-v2 not PASSED")
    require(validation.get("test_cases_used") is False, "Phase 4B-v2 used test")
    require(
        audit.get("result", {}).get(
            "former_single_rule_degeneracy_resolved"
        ) is True,
        "Stage 1 is not non-degenerate",
    )
    require(audit.get("test_data_used") is False, "Grouping audit used test")
    require(
        cache_validation.get("status") == "passed",
        "Fusion eval cache not PASSED",
    )
    require(
        cache_validation.get("test_cases_used") is False,
        "Fusion cache used test",
    )
    require(
        cache_validation.get("checks", {}).get(
            "all_pool_digests_match"
        ) is True,
        "Fusion cache pool digest check failed",
    )
    require(
        cache_validation.get("checks", {}).get(
            "all_stratified_fit_eval_splits_match"
        ) is True,
        "Fusion cache split check failed",
    )

    lambdas = [float(v) for v in args.lambdas]
    require(all(v > 0 for v in lambdas), "Lambda must be positive")

    rules = read_jsonl(rules_path)
    cf = [
        r for r in rules
        if r.get("rule_type") == "counterfactual"
        and bool(r.get("eligible_for_reliability"))
    ]
    require(
        len(cf)
        == int(validation["eligible_counterfactual_candidate_count"]),
        "Eligible CF count disagrees with Phase 4B-v2 validator",
    )

    cache = load_cache_index(cache_dir)
    require(
        len(cache) == int(cache_validation["local_target_count"]),
        "Fusion cache target count mismatch",
    )

    # Build frozen Stage-1 source grouping once.
    s1_source_groups = defaultdict(list)
    for r in cf:
        s1_source_groups[group_key_stage1(r)].append(r)

    expected_s1 = int(audit["stage1"]["group_count"])
    expected_s1_fusable = int(
        audit["stage1"]["fusable_group_count_ge2_rules"]
    )
    require(
        len(s1_source_groups) == expected_s1,
        f"Stage1 group count mismatch: {len(s1_source_groups)} != {expected_s1}",
    )
    require(
        sum(len(v) >= 2 for v in s1_source_groups.values())
        == expected_s1_fusable,
        "Stage1 fusable count mismatch",
    )

    stage_rows = []
    conflict_rows = []
    stage3_rule_by_lambda = defaultdict(dict)

    print("=" * 110)
    print("ETCE PHASE 4C-v2 - HIERARCHICAL D-S VALIDATION SWEEP")
    print("=" * 110)
    print(f"Eligible CF candidates: {len(cf)}")
    print(f"Stage1 groups: {expected_s1} | fusable: {expected_s1_fusable}")
    print(f"Lambdas: {lambdas}")
    print("No formal test data are read.")
    print()

    for lam in lambdas:
        print(f"[lambda={lam:g}] Stage 1 ...", flush=True)

        # ----------------------------------------------------------
        # Stage 1: within target-local surrogate.
        # ----------------------------------------------------------
        s1_outputs = {}
        for key in sorted(
            s1_source_groups,
            key=lambda k: (
                k[0],
                MODEL_ORDER.get(k[1], 99),
                k[2],
                SURROGATE_ORDER.get(k[3], 99),
                k[4],
                k[5],
            ),
        ):
            source_rules = sorted(
                s1_source_groups[key],
                key=lambda r: str(r["candidate_id"]),
            )
            case_id, model, prefix, surrogate, pred, sig = key
            ctx = [cache[(case_id, model, int(prefix))]]

            current = source_rule(source_rules[0], lam)
            for step, r in enumerate(source_rules[1:], 1):
                right = source_rule(r, lam)
                current, trace = pairwise_rule_fusion(
                    current,
                    right,
                    ctx,
                    lam,
                    "Stage1",
                    {
                        "lambda": float(lam),
                        "case_id": case_id,
                        "model": model,
                        "prefix": int(prefix),
                        "surrogate": surrogate,
                        "predicted_class": int(pred),
                        "feature_signature": json.dumps(list(sig)),
                        "step": int(step),
                    },
                )
                conflict_rows.append(trace)

            s1_outputs[key] = current
            stage_rows.append(
                ev_to_row(
                    current,
                    lam,
                    "Stage1",
                    key,
                    len(source_rules),
                )
            )

        # ----------------------------------------------------------
        # Stage 2: across LORE / DT / BN within same black-box.
        # ----------------------------------------------------------
        s2_groups = defaultdict(list)
        for key, ev in s1_outputs.items():
            case_id, model, prefix, surrogate, pred, sig = key
            s2_key = (
                case_id,
                model,
                int(prefix),
                int(pred),
                sig,
            )
            s2_groups[s2_key].append((surrogate, ev))

        require(
            len(s2_groups) == int(audit["stage2"]["compatible_sets"]),
            "Stage2 compatible-set count mismatch",
        )
        require(
            sum(
                len({s for s, _ in items}) >= 2
                for items in s2_groups.values()
            )
            == int(audit["stage2"]["sets_with_ge2_surrogates"]),
            "Stage2 fusable-set count mismatch",
        )

        print(
            f"[lambda={lam:g}] Stage 2 ... "
            f"groups={len(s2_groups)}",
            flush=True,
        )

        s2_outputs = {}
        for key in sorted(
            s2_groups,
            key=lambda k: (
                k[0],
                MODEL_ORDER.get(k[1], 99),
                k[2],
                k[3],
                k[4],
            ),
        ):
            case_id, model, prefix, pred, sig = key
            items = sorted(
                s2_groups[key],
                key=lambda x: SURROGATE_ORDER.get(x[0], 99),
            )
            ctx = [cache[(case_id, model, int(prefix))]]

            current = items[0][1]
            for step, (surrogate, right) in enumerate(items[1:], 1):
                current, trace = pairwise_rule_fusion(
                    current,
                    right,
                    ctx,
                    lam,
                    "Stage2",
                    {
                        "lambda": float(lam),
                        "case_id": case_id,
                        "model": model,
                        "prefix": int(prefix),
                        "surrogate": f"cross-surrogate:{surrogate}",
                        "predicted_class": int(pred),
                        "feature_signature": json.dumps(list(sig)),
                        "step": int(step),
                    },
                )
                conflict_rows.append(trace)

            s2_outputs[key] = current
            stage_rows.append(
                ev_to_row(
                    current,
                    lam,
                    "Stage2",
                    key,
                    len(items),
                )
            )

        # ----------------------------------------------------------
        # Stage 3: across black-boxes for same case/prefix.
        # ----------------------------------------------------------
        s3_groups = defaultdict(list)
        for key, ev in s2_outputs.items():
            case_id, model, prefix, pred, sig = key
            s3_key = (
                case_id,
                int(prefix),
                int(pred),
                sig,
            )
            s3_groups[s3_key].append((model, ev))

        require(
            len(s3_groups) == int(audit["stage3"]["compatible_sets"]),
            "Stage3 compatible-set count mismatch",
        )
        require(
            sum(
                len({m for m, _ in items}) >= 2
                for items in s3_groups.values()
            )
            == int(audit["stage3"]["sets_with_ge2_blackboxes"]),
            "Stage3 fusable-set count mismatch",
        )

        print(
            f"[lambda={lam:g}] Stage 3 ... "
            f"groups={len(s3_groups)}",
            flush=True,
        )

        for key in sorted(
            s3_groups,
            key=lambda k: (k[0], k[1], k[2], k[3]),
        ):
            case_id, prefix, pred, sig = key
            items = sorted(
                s3_groups[key],
                key=lambda x: MODEL_ORDER.get(x[0], 99),
            )

            current = items[0][1]
            models_so_far = [items[0][0]]

            for step, (model, right) in enumerate(items[1:], 1):
                models_so_far.append(model)
                contexts = [
                    cache[(case_id, m, int(prefix))]
                    for m in models_so_far
                ]
                current, trace = pairwise_rule_fusion(
                    current,
                    right,
                    contexts,
                    lam,
                    "Stage3",
                    {
                        "lambda": float(lam),
                        "case_id": case_id,
                        "model": f"cross-model:{model}",
                        "prefix": int(prefix),
                        "surrogate": None,
                        "predicted_class": int(pred),
                        "feature_signature": json.dumps(list(sig)),
                        "step": int(step),
                    },
                )
                conflict_rows.append(trace)

            row = ev_to_row(
                current,
                lam,
                "Stage3",
                key,
                len(items),
            )
            stage_rows.append(row)

            if len(items) >= 2:
                stable_key = (
                    case_id,
                    int(prefix),
                    int(pred),
                    tuple(sig),
                )
                stage3_rule_by_lambda[stable_key][float(lam)] = (
                    row["selected_conditions"]
                )

        print(f"[lambda={lam:g}] complete.", flush=True)
        print()

    # --------------------------------------------------------------
    # Summary tables.
    # --------------------------------------------------------------
    summary_rows = []
    for lam in lambdas:
        for stage in ("Stage1", "Stage2", "Stage3"):
            summary_rows.append(
                summarize_stage(
                    stage_rows,
                    conflict_rows,
                    stage,
                    lam,
                )
            )

    # Lambda-induced Stage3 rule stability.
    stability_rows = []
    stable_all_count = 0
    for key, mapping in sorted(stage3_rule_by_lambda.items()):
        values = [mapping.get(float(lam)) for lam in lambdas]
        require(
            all(v is not None for v in values),
            f"Missing lambda result for Stage3 stability key {key}",
        )
        uniq = len(set(values))
        stable_all = uniq == 1
        stable_all_count += int(stable_all)
        case_id, prefix, pred, sig = key
        row = {
            "case_id": case_id,
            "prefix": int(prefix),
            "predicted_class": int(pred),
            "feature_signature": json.dumps(list(sig)),
            "unique_rules_across_lambdas": int(uniq),
            "stable_across_all_lambdas": bool(stable_all),
        }
        for lam in lambdas:
            row[f"rule_lambda_{lam:g}"] = mapping[float(lam)]
        stability_rows.append(row)

    stability_summary = {
        "stage3_fusable_group_count": len(stability_rows),
        "stable_across_all_lambdas_count": stable_all_count,
        "stable_across_all_lambdas_rate": (
            float(stable_all_count / len(stability_rows))
            if stability_rows
            else None
        ),
        "mean_unique_rules_across_lambdas": mean(
            [r["unique_rules_across_lambdas"] for r in stability_rows]
        ),
    }

    write_csv(
        out_dir / "phase4c_v2_ds_stage_outputs.csv",
        stage_rows,
    )
    write_csv(
        out_dir / "phase4c_v2_ds_conflict_trace.csv",
        conflict_rows,
    )
    write_csv(
        out_dir / "phase4c_v2_ds_lambda_summary.csv",
        summary_rows,
    )
    write_csv(
        out_dir / "phase4c_v2_stage3_lambda_stability.csv",
        stability_rows,
    )

    result = {
        "scope": "ETCE Phase 4C-v2 hierarchical D-S validation sweep",
        "status": "passed",
        "lambda_candidates": lambdas,
        "lambda_selected": None,
        "selection_status": "NOT_FROZEN_SWEEP_ONLY",
        "test_cases_used": False,
        "protocol": {
            "reliability": "R = n/(n+lambda) * Fid",
            "bpa": "m(H)=R; m(Theta)=1-R",
            "stage1_order": "candidate_id",
            "stage2_order": ["LORE", "DT", "BN"],
            "stage3_order": ["RF", "XGBoost", "MLP", "LSTM"],
            "K_total_conflict_guard": K_EPS,
            "total_conflict_policy": "map pair to total ignorance; representative fallback carries R=0",
            "representative_rule_policy": (
                "highest normalized proper focal mass; "
                "tie -> more specific -> lexical canonical rule"
            ),
            "fused_rule_re_evaluation": (
                "recompute n_r and Fid(r) on frozen held-out validation cache "
                "before propagating to next hierarchy level"
            ),
        },
        "preconditions": {
            "phase4b_v2_validation_passed": True,
            "stage1_non_degenerate": True,
            "fusion_eval_cache_passed": True,
            "formal_test_data_used": False,
        },
        "group_counts": {
            "stage1": int(audit["stage1"]["group_count"]),
            "stage1_fusable": int(
                audit["stage1"]["fusable_group_count_ge2_rules"]
            ),
            "stage2": int(audit["stage2"]["compatible_sets"]),
            "stage2_fusable": int(
                audit["stage2"]["sets_with_ge2_surrogates"]
            ),
            "stage3": int(audit["stage3"]["compatible_sets"]),
            "stage3_fusable": int(
                audit["stage3"]["sets_with_ge2_blackboxes"]
            ),
        },
        "stage3_lambda_stability": stability_summary,
        "next_step": (
            "Review validation-only Stage3 quality/conflict/stability across "
            "lambda candidates, predeclare the selection rule, then freeze one "
            "lambda before formal test is opened."
        ),
    }
    write_json(
        out_dir / "validation_ds_lambda_sweep.json",
        result,
    )

    # Compact human-readable report.
    s3 = [
        r for r in summary_rows
        if r["stage"] == "Stage3"
    ]
    lines = [
        "# ETCE Phase 4C-v2 Hierarchical D-S Lambda Sweep",
        "",
        "> Validation only. No formal test data were used. Lambda is not frozen in this script.",
        "",
        "## Stage-3 genuinely fused groups (>=2 black-boxes)",
        "",
        "| lambda | N | mean Fidelity | CF validity | model CF validity | all-model CF validity | mean R | mean ignorance | mean K | p90 K | max K |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in s3:
        def fv(x):
            return "NA" if x is None else f"{float(x):.4f}"
        lines.append(
            f"| {r['lambda']:g} | {r['fused_count']} | "
            f"{fv(r['fused_mean_fidelity'])} | "
            f"{fv(r['fused_cf_validity_rate'])} | "
            f"{fv(r['fused_mean_model_cf_validity_rate'])} | "
            f"{fv(r['fused_all_models_cf_valid_rate'])} | "
            f"{fv(r['fused_mean_reliability'])} | "
            f"{fv(r['fused_mean_ignorance'])} | "
            f"{fv(r['mean_conflict_K'])} | "
            f"{fv(r['p90_conflict_K'])} | "
            f"{fv(r['max_conflict_K'])} |"
        )
    lines.extend(
        [
            "",
            "## Lambda-induced rule stability",
            "",
            f"- Stage-3 fusable groups: {stability_summary['stage3_fusable_group_count']}",
            f"- Stable across all lambda values: {stability_summary['stable_across_all_lambdas_count']}",
            f"- Stability rate: {stability_summary['stable_across_all_lambdas_rate']:.4f}",
            f"- Mean number of distinct selected rules across lambdas: {stability_summary['mean_unique_rules_across_lambdas']:.4f}",
            "",
            "## Status",
            "",
            "No lambda is automatically selected here. Review this validation-only sweep and freeze the selection rule before opening formal test.",
        ]
    )
    (
        out_dir / "PHASE4C_V2_DS_SWEEP_REPORT.md"
    ).write_text("\n".join(lines), encoding="utf-8")

    print("=" * 110)
    print("PHASE 4C-v2 HIERARCHICAL D-S SWEEP PASSED")
    print("=" * 110)
    print("Stage-3 genuinely fused-group summary:")
    for r in s3:
        print(
            f"lambda={r['lambda']:g} | "
            f"N={r['fused_count']} | "
            f"Fid={r['fused_mean_fidelity']} | "
            f"CF-valid={r['fused_cf_validity_rate']} | "
            f"model-valid={r['fused_mean_model_cf_validity_rate']} | "
            f"all-model-valid={r['fused_all_models_cf_valid_rate']} | "
            f"meanK={r['mean_conflict_K']} | "
            f"p90K={r['p90_conflict_K']} | "
            f"maxK={r['max_conflict_K']}"
        )
    print()
    print(
        "Stage3 stable across all lambdas: "
        f"{stability_summary['stable_across_all_lambdas_count']}/"
        f"{stability_summary['stage3_fusable_group_count']} "
        f"({stability_summary['stable_across_all_lambdas_rate']:.2%})"
    )
    print(f"Outputs: {out_dir}")
    print("IMPORTANT: lambda_selected = null")
    print("No formal test data were read.")


if __name__ == "__main__":
    main()
