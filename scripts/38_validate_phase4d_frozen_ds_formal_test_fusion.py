#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-5:
Independent validator for Phase 4D-4 frozen formal TEST D-S fusion.

Purpose
-------
Independently reconstruct Stage1 -> Stage2 -> Stage3 from:
- formal TEST canonical rule bank;
- frozen formal TEST fusion-evaluation cache;
- Phase 4D-3 grouping audit;
- frozen validation-selected D-S config;

and compare the reproduced outputs against Phase 4D-4 artifacts.

Important
---------
- This validator does NOT import script 37.
- It does NOT select lambda.
- It does NOT modify any protocol.
- It does NOT fit or retune any surrogate.
- It does NOT use true outcome labels.
- It re-evaluates fused rules only on the already-frozen held-out local cache.

Run:
    cd D:\ETCE
    python -u .\scripts\38_validate_phase4d_frozen_ds_formal_test_fusion.py
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]

FORMAL_ROOT = ROOT / "results/phase4/phase4d_formal_test"
BANK = FORMAL_ROOT / "formal_test_multirule_bank"
FUSION_OUT = FORMAL_ROOT / "formal_test_frozen_ds_fusion"
AUDIT_DIR = FORMAL_ROOT / "formal_test_target_local_group_audit"

RULES = BANK / "canonical_rules_test.jsonl"
PHASE4D2_VALIDATION = BANK / "validation_phase4d_formal_test_multirule.json"
CACHE_MANIFEST = BANK / "fusion_eval_cache_manifest.json"
CACHE_TARGETS = BANK / "fusion_eval_cache" / "targets"

AUDIT = AUDIT_DIR / "formal_test_target_local_group_audit.json"
FROZEN = ROOT / "configs/phase4c_ds_frozen_config.json"

SAVED_STAGE = FUSION_OUT / "phase4d_frozen_ds_stage_outputs.csv"
SAVED_TRACE = FUSION_OUT / "phase4d_frozen_ds_conflict_trace.csv"
SAVED_SUMMARY = FUSION_OUT / "phase4d_frozen_ds_summary.csv"
SAVED_STAGE3 = FUSION_OUT / "phase4d_stage3_fused_rules.csv"
SAVED_RESULT = FUSION_OUT / "formal_test_frozen_ds_fusion.json"

OUT_VALIDATION = FUSION_OUT / "validation_phase4d_frozen_ds_fusion.json"

LAMBDA = 5.0
K_EPS = 1e-12
SURROGATE_ORDER = {"LORE": 0, "DT": 1, "BN": 2}
MODEL_ORDER = {"RF": 0, "XGBoost": 1, "MLP": 2, "LSTM": 3}
THETA = None
EMPTY = object()

FLOAT_TOL = 1e-12


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


def read_gz_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def canonical_values(values):
    return tuple(sorted(str(v) for v in values))


def conditions_to_hypothesis(conditions):
    by_pos = {}
    for cond in conditions:
        require(
            cond.get("operator") == "in",
            f"Unsupported operator: {cond.get('operator')}",
        )
        p = int(cond["position"])
        vals = canonical_values(cond["values"])
        require(vals, f"Empty values at position {p}")
        require(p not in by_pos, f"Duplicate position {p}")
        by_pos[p] = vals
    return tuple((p, by_pos[p]) for p in sorted(by_pos))


def hypothesis_to_conditions(hyp):
    if hyp is THETA:
        return []
    return [
        {"position": int(p), "operator": "in", "values": list(vals)}
        for p, vals in hyp
    ]


def feature_signature(hyp):
    if hyp is THETA:
        return tuple()
    return tuple(int(p) for p, _ in hyp)


def hyp_key(hyp):
    if hyp is THETA:
        return "__THETA__"
    return json.dumps(
        hypothesis_to_conditions(hyp),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def specificity(hyp):
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


def simple_support_mass(hyp, r):
    r = min(1.0, max(0.0, float(r)))
    out = defaultdict(float)
    out[hyp] += r
    out[THETA] += 1.0 - r
    return dict(out)


def combine_two(m1, m2):
    raw = defaultdict(float)
    K = 0.0

    for h1, v1 in m1.items():
        for h2, v2 in m2.items():
            product = float(v1) * float(v2)
            inter = intersect_hyp(h1, h2)
            if inter is EMPTY:
                K += product
            else:
                raw[inter] += product

    fallback = (1.0 - K) <= K_EPS
    if fallback:
        return {THETA: 1.0}, float(K), True

    denom = 1.0 - K
    out = {
        h: float(v / denom)
        for h, v in raw.items()
        if v > 0.0
    }

    s = sum(out.values())
    if s > 0:
        out = {h: float(v / s) for h, v in out.items()}

    return out, float(K), False


def choose_proper_focal(mass):
    candidates = [
        (h, float(v))
        for h, v in mass.items()
        if h is not THETA and float(v) > 0
    ]
    if not candidates:
        return None, 0.0

    candidates.sort(
        key=lambda hv: (
            -hv[1],
            specificity(hv[0]),
            hyp_key(hv[0]),
        )
    )
    return candidates[0]


def applies(sequence, hyp):
    if hyp is THETA:
        return True

    for p, vals in hyp:
        idx = int(p) - 1
        if idx < 0 or idx >= len(sequence):
            return False
        if str(sequence[idx]) not in set(vals):
            return False
    return True


def hamming(a, b):
    require(len(a) == len(b), "Hamming length mismatch")
    return sum(x != y for x, y in zip(a, b))


def evaluate(hyp, predicted_class, contexts):
    total_eval = 0
    total_support = 0
    total_correct = 0
    global_min_edit = None
    per_model = []

    for p in contexts:
        sequences = p["pool_sequences"]
        labels = p["pool_labels"]
        indices = p["eval_indices"]
        factual = p["factual_sequence"]

        support = 0
        correct = 0
        min_edit = None

        for raw_i in indices:
            i = int(raw_i)
            seq = sequences[i]
            total_eval += 1

            if not applies(seq, hyp):
                continue

            support += 1
            total_support += 1

            if int(labels[i]) == int(predicted_class):
                correct += 1
                total_correct += 1
                ed = hamming(factual, seq)

                if min_edit is None or ed < min_edit:
                    min_edit = ed
                if global_min_edit is None or ed < global_min_edit:
                    global_min_edit = ed

        per_model.append({
            "model": p["model"],
            "eval_size": len(indices),
            "support": support,
            "correct": correct,
            "fidelity": (
                float(correct / support)
                if support > 0 else None
            ),
            "cf_valid": bool(correct > 0),
            "min_witness_edit": min_edit,
        })

    fid = (
        float(total_correct / total_support)
        if total_support > 0 else None
    )
    coverage = (
        float(total_support / total_eval)
        if total_eval > 0 else 0.0
    )
    model_validity = (
        float(
            sum(int(x["cf_valid"]) for x in per_model)
            / len(per_model)
        )
        if per_model else None
    )

    return {
        "n_r": int(total_support),
        "Fid_r": fid,
        "coverage_eval": coverage,
        "cf_valid": bool(total_correct > 0),
        "model_cf_validity_rate": model_validity,
        "all_models_cf_valid": bool(
            per_model and all(x["cf_valid"] for x in per_model)
        ),
        "min_witness_edit": global_min_edit,
    }


def reliability(n, fid):
    n = int(n)
    if n <= 0 or fid is None:
        return 0.0
    rho = float(n / (n + LAMBDA))
    return float(rho * float(fid))


def from_source(rule):
    n = int(rule["n_r"])
    fid = float(rule["Fid_r"])
    rho = float(n / (n + LAMBDA))
    r = float(rho * fid)

    return {
        "hyp": conditions_to_hypothesis(rule["conditions"]),
        "pred": int(rule["predicted_class"]),
        "n": n,
        "fid": fid,
        "coverage": float(rule["coverage_eval"]),
        "rho": rho,
        "R": r,
        "source_count": 1,
        "models": {rule["model"]},
        "surrogates": {rule["surrogate"]},
        "candidate_ids": [rule["candidate_id"]],
        "last_mass": r,
        "last_ignorance": 1.0 - r,
        "last_K": None,
        "fallbacks": 0,
        "steps": 0,
        "min_edit": rule.get("witness_edit_count"),
        "cf_valid": True,
        "model_valid": 1.0,
        "all_models_valid": True,
    }


def fuse_pair(left, right, contexts):
    require(left["pred"] == right["pred"], "Predicted-class mismatch")
    require(
        feature_signature(left["hyp"]) == feature_signature(right["hyp"]),
        "Feature-signature mismatch",
    )

    mass, K, fallback = combine_two(
        simple_support_mass(left["hyp"], left["R"]),
        simple_support_mass(right["hyp"], right["R"]),
    )

    selected, selected_mass = choose_proper_focal(mass)

    if selected is None:
        reps = [left, right]
        reps.sort(
            key=lambda ev: (
                -float(ev["R"]),
                specificity(ev["hyp"]),
                hyp_key(ev["hyp"]),
            )
        )
        selected = reps[0]["hyp"]
        selected_mass = 0.0
        met = evaluate(selected, left["pred"], contexts)
        out_R = 0.0
        ignorance = 1.0
    else:
        met = evaluate(selected, left["pred"], contexts)
        out_R = reliability(met["n_r"], met["Fid_r"])
        ignorance = float(mass.get(THETA, 0.0))

    out = {
        "hyp": selected,
        "pred": int(left["pred"]),
        "n": int(met["n_r"]),
        "fid": met["Fid_r"],
        "coverage": float(met["coverage_eval"]),
        "rho": (
            float(met["n_r"] / (met["n_r"] + LAMBDA))
            if met["n_r"] > 0 else 0.0
        ),
        "R": float(out_R),
        "source_count": int(left["source_count"] + right["source_count"]),
        "models": set(left["models"]) | set(right["models"]),
        "surrogates": set(left["surrogates"]) | set(right["surrogates"]),
        "candidate_ids": list(left["candidate_ids"]) + list(right["candidate_ids"]),
        "last_mass": float(selected_mass),
        "last_ignorance": float(ignorance),
        "last_K": float(K),
        "fallbacks": int(left["fallbacks"] + right["fallbacks"] + int(fallback)),
        "steps": int(left["steps"] + right["steps"] + 1),
        "min_edit": met["min_witness_edit"],
        "cf_valid": bool(met["cf_valid"]),
        "model_valid": met["model_cf_validity_rate"],
        "all_models_valid": bool(met["all_models_cf_valid"]),
    }
    return out, float(K), bool(fallback), float(selected_mass), float(ignorance)


def stage1_key(rule):
    hyp = conditions_to_hypothesis(rule["conditions"])
    return (
        rule["case_id"],
        rule["model"],
        int(rule["prefix"]),
        rule["surrogate"],
        int(rule["predicted_class"]),
        feature_signature(hyp),
    )


def stage_row(ev, stage, key, input_count):
    if stage == "Stage1":
        case_id, model, prefix, surrogate, pred, sig = key
    elif stage == "Stage2":
        case_id, model, prefix, pred, sig = key
        surrogate = None
    elif stage == "Stage3":
        case_id, prefix, pred, sig = key
        model = None
        surrogate = None
    else:
        raise ValueError(stage)

    return {
        "lambda": float(LAMBDA),
        "stage": stage,
        "case_id": case_id,
        "model": model,
        "prefix": int(prefix),
        "surrogate": surrogate,
        "predicted_class": int(pred),
        "feature_signature": json.dumps(list(sig)),
        "input_count": int(input_count),
        "actually_fused": bool(input_count >= 2),
        "source_rule_count": int(ev["source_count"]),
        "source_model_count": int(len(ev["models"])),
        "source_surrogate_count": int(len(ev["surrogates"])),
        "n_r": int(ev["n"]),
        "Fid_r": ev["fid"],
        "coverage_eval": float(ev["coverage"]),
        "rho_r": float(ev["rho"]),
        "R_r": float(ev["R"]),
        "cf_valid": bool(ev["cf_valid"]),
        "model_cf_validity_rate": ev["model_valid"],
        "all_models_cf_valid": bool(ev["all_models_valid"]),
        "min_witness_edit": ev["min_edit"],
        "fusion_steps": int(ev["steps"]),
        "total_conflict_fallback_count": int(ev["fallbacks"]),
        "last_K": ev["last_K"],
        "last_selected_mass": float(ev["last_mass"]),
        "last_ignorance_mass": float(ev["last_ignorance"]),
        "selected_conditions": json.dumps(
            hypothesis_to_conditions(ev["hyp"]),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def mean(values):
    vals = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]
    return sum(vals) / len(vals) if vals else None


def median(values):
    vals = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]
    return float(statistics.median(vals)) if vals else None


def quantile(values, p):
    vals = sorted(
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    )
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    x = (len(vals) - 1) * p
    lo = int(math.floor(x))
    hi = int(math.ceil(x))
    if lo == hi:
        return vals[lo]
    f = x - lo
    return vals[lo] * (1 - f) + vals[hi] * f


def summarize(rows, traces, stage):
    sub = [r for r in rows if r["stage"] == stage]
    fused = [r for r in sub if r["actually_fused"]]
    cr = [r for r in traces if r["stage"] == stage]

    def scoped(scope, name):
        return {
            f"{name}_count": len(scope),
            f"{name}_mean_fidelity": mean(r["Fid_r"] for r in scope),
            f"{name}_median_fidelity": median(r["Fid_r"] for r in scope),
            f"{name}_cf_validity_rate": mean(int(r["cf_valid"]) for r in scope),
            f"{name}_mean_model_cf_validity_rate": mean(
                r["model_cf_validity_rate"] for r in scope
            ),
            f"{name}_all_models_cf_valid_rate": mean(
                int(r["all_models_cf_valid"]) for r in scope
            ),
            f"{name}_mean_support": mean(r["n_r"] for r in scope),
            f"{name}_mean_coverage": mean(r["coverage_eval"] for r in scope),
            f"{name}_mean_reliability": mean(r["R_r"] for r in scope),
            f"{name}_mean_ignorance": mean(
                r["last_ignorance_mass"] for r in scope
            ),
            f"{name}_mean_min_witness_edit": mean(
                r["min_witness_edit"] for r in scope
            ),
        }

    out = {
        "lambda": LAMBDA,
        "stage": stage,
        "group_count": len(sub),
        "fusable_group_count": len(fused),
        "pairwise_fusion_count": len(cr),
        "mean_conflict_K": mean(r["K"] for r in cr),
        "median_conflict_K": median(r["K"] for r in cr),
        "p90_conflict_K": quantile([r["K"] for r in cr], 0.90),
        "max_conflict_K": max((float(r["K"]) for r in cr), default=None),
        "total_conflict_fallback_count": sum(
            int(r["total_conflict_fallback"]) for r in cr
        ),
    }
    out.update(scoped(sub, "all"))
    out.update(scoped(fused, "fused"))
    return out


def load_cache():
    idx = {}
    for path in sorted(CACHE_TARGETS.glob("*.json.gz")):
        p = read_gz_json(path)
        key = (p["case_id"], p["model"], int(p["prefix"]))
        require(key not in idx, f"Duplicate cache target: {key}")
        idx[key] = p
    return idx


def norm_scalar(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    s = str(v).strip()
    if s == "":
        return None
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    if s.lower() in {"none", "nan", "null"}:
        return None
    try:
        if "." not in s and "e" not in s.lower():
            return int(s)
        return float(s)
    except Exception:
        return s


def json_norm(s):
    if s is None:
        return None
    if isinstance(s, (list, dict)):
        return json.dumps(s, ensure_ascii=False, sort_keys=True)
    try:
        return json.dumps(
            json.loads(str(s)),
            ensure_ascii=False,
            sort_keys=True,
        )
    except Exception:
        return str(s)


def close(a, b, tol=FLOAT_TOL):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def stage_identity(r):
    return (
        str(r["stage"]),
        str(r["case_id"]),
        "" if r.get("model") is None else str(r.get("model")),
        int(r["prefix"]),
        "" if r.get("surrogate") is None else str(r.get("surrogate")),
        int(r["predicted_class"]),
        json_norm(r["feature_signature"]),
    )


def trace_identity(r):
    return (
        str(r["stage"]),
        str(r["case_id"]),
        "" if r.get("model") is None else str(r.get("model")),
        int(r["prefix"]),
        "" if r.get("surrogate") is None else str(r.get("surrogate")),
        int(r["predicted_class"]),
        json_norm(r["feature_signature"]),
        int(r["step"]),
    )


def compare_row(expected, saved, numeric_fields, bool_fields, int_fields, json_fields, text_fields):
    for k in numeric_fields:
        a = expected.get(k)
        b = norm_scalar(saved.get(k))
        require(
            close(a, b),
            f"Numeric mismatch field={k}: expected={a!r}, saved={b!r}",
        )

    for k in bool_fields:
        a = bool(expected.get(k))
        b = bool(norm_scalar(saved.get(k)))
        require(a == b, f"Boolean mismatch field={k}: {a} != {b}")

    for k in int_fields:
        a = expected.get(k)
        b = norm_scalar(saved.get(k))
        if a is None:
            require(b is None, f"Expected null field={k}, saved={b!r}")
        else:
            require(int(a) == int(b), f"Integer mismatch field={k}: {a} != {b}")

    for k in json_fields:
        require(
            json_norm(expected.get(k)) == json_norm(saved.get(k)),
            f"JSON mismatch field={k}",
        )

    for k in text_fields:
        a = "" if expected.get(k) is None else str(expected.get(k))
        b = "" if saved.get(k) is None else str(saved.get(k))
        require(a == b, f"Text mismatch field={k}: {a!r} != {b!r}")


def main():
    required = [
        RULES,
        PHASE4D2_VALIDATION,
        CACHE_MANIFEST,
        AUDIT,
        FROZEN,
        SAVED_STAGE,
        SAVED_TRACE,
        SAVED_SUMMARY,
        SAVED_STAGE3,
        SAVED_RESULT,
    ]
    missing = [str(p) for p in required if not p.is_file()]
    require(not missing, "Missing required artifacts:\n" + "\n".join(missing))
    require(CACHE_TARGETS.is_dir(), f"Missing cache target dir: {CACHE_TARGETS}")

    phase4d2 = read_json(PHASE4D2_VALIDATION)
    audit = read_json(AUDIT)
    frozen = read_json(FROZEN)
    manifest = read_json(CACHE_MANIFEST)
    saved_result = read_json(SAVED_RESULT)

    print("=" * 116)
    print("ETCE PHASE 4D-5 INDEPENDENT FROZEN D-S FORMAL TEST VALIDATION")
    print("=" * 116)

    # Frozen protocol gates.
    require(phase4d2.get("status") == "passed", "Phase 4D-2 is not PASSED")
    require(audit.get("status") == "passed", "Phase 4D-3 is not PASSED")
    require(
        audit.get("stage1", {}).get("status") == "NON-DEGENERATE",
        "Stage1 non-degeneracy not established",
    )
    require(frozen.get("status") == "frozen", "D-S config not frozen")
    require(float(frozen.get("lambda")) == LAMBDA, "Frozen lambda != 5")
    require(frozen.get("selection_scope") == "validation_only", "Lambda not validation-only")
    require(
        frozen.get("formal_test_seen_before_freeze") is False,
        "Formal TEST seen before freeze",
    )

    fusion_cfg = frozen["fusion"]
    require(fusion_cfg["stage1_order"] == "candidate_id", "Stage1 order changed")
    require(fusion_cfg["stage2_order"] == ["LORE", "DT", "BN"], "Stage2 order changed")
    require(
        fusion_cfg["stage3_order"] == ["RF", "XGBoost", "MLP", "LSTM"],
        "Stage3 order changed",
    )
    require(
        float(fusion_cfg["K_total_conflict_guard"]) == K_EPS,
        "K guard changed",
    )
    require(
        fusion_cfg["total_conflict_policy"]
        == "map pair to total ignorance; representative fallback carries R=0",
        "Conflict policy changed",
    )
    require(
        fusion_cfg["representative_rule_policy"]
        == "highest normalized proper focal mass; tie -> more specific -> lexical canonical rule",
        "Representative rule policy changed",
    )

    require(manifest.get("status") == "built", "Cache manifest not built")
    require(int(manifest["target_count"]) == 1600, "Cache manifest target count != 1600")
    require(manifest.get("true_labels_used") is False, "True labels used in cache")

    require(saved_result.get("status") == "passed", "Saved Phase 4D-4 result not PASSED")
    require(float(saved_result.get("lambda")) == LAMBDA, "Saved lambda != 5")
    require(saved_result.get("lambda_selection_performed") is False, "TEST-side lambda selection detected")
    require(saved_result.get("protocol_retuning_performed") is False, "TEST-side retuning detected")

    # Read source evidence and cache.
    rules = read_jsonl(RULES)
    cf = [
        r for r in rules
        if r.get("rule_type") == "counterfactual"
        and bool(r.get("eligible_for_reliability"))
    ]
    require(len(rules) == 9846, f"Canonical rule count != 9846: {len(rules)}")
    require(len(cf) == 6172, f"Eligible CF count != 6172: {len(cf)}")

    cache = load_cache()
    require(len(cache) == 1600, f"Cache target count != 1600: {len(cache)}")
    require(
        all(p.get("formal_test_target") is True for p in cache.values()),
        "Non-formal cache payload detected",
    )
    require(
        all(p.get("true_label_used") is False for p in cache.values()),
        "true_label_used=True detected",
    )

    # Independent Stage1 grouping.
    s1_groups = defaultdict(list)
    for r in cf:
        s1_groups[stage1_key(r)].append(r)

    require(len(s1_groups) == int(audit["stage1"]["group_count"]), "Stage1 group count mismatch")
    require(
        sum(len(v) >= 2 for v in s1_groups.values())
        == int(audit["stage1"]["fusable_group_count_ge2_rules"]),
        "Stage1 fusable count mismatch",
    )

    reproduced_stage = []
    reproduced_trace = []

    print(f"[Input] canonical rules={len(rules)} | eligible CF={len(cf)} | cache={len(cache)}")
    print("[Recompute] Stage 1 ...")

    s1_out = {}
    for key in sorted(
        s1_groups,
        key=lambda k: (
            k[0], MODEL_ORDER.get(k[1], 99), k[2],
            SURROGATE_ORDER.get(k[3], 99), k[4], k[5],
        ),
    ):
        case_id, model, prefix, surrogate, pred, sig = key
        ordered = sorted(s1_groups[key], key=lambda r: str(r["candidate_id"]))
        ctx = [cache[(case_id, model, int(prefix))]]
        current = from_source(ordered[0])

        for step, r in enumerate(ordered[1:], 1):
            right = from_source(r)
            current, K, fallback, sel_mass, ign = fuse_pair(current, right, ctx)
            reproduced_trace.append({
                "lambda": LAMBDA,
                "case_id": case_id,
                "model": model,
                "prefix": int(prefix),
                "surrogate": surrogate,
                "predicted_class": int(pred),
                "feature_signature": json.dumps(list(sig)),
                "step": int(step),
                "stage": "Stage1",
                "K": K,
                "total_conflict_fallback": fallback,
                "left_R": None,   # checked structurally/numerically below except unavailable pre-state fields
                "right_R": None,
                "selected_focal_mass": sel_mass,
                "ignorance_mass": ign,
                "output_n_r": int(current["n"]),
                "output_Fid_r": current["fid"],
                "output_R_r": float(current["R"]),
                "output_cf_valid": bool(current["cf_valid"]),
                "selected_conditions": json.dumps(
                    hypothesis_to_conditions(current["hyp"]),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            })

        s1_out[key] = current
        reproduced_stage.append(stage_row(current, "Stage1", key, len(ordered)))

    print(f"[Recompute] Stage 1 complete | groups={len(s1_out)}")

    # Stage2.
    s2_groups = defaultdict(list)
    for key, ev in s1_out.items():
        case_id, model, prefix, surrogate, pred, sig = key
        s2_groups[(case_id, model, int(prefix), int(pred), sig)].append((surrogate, ev))

    require(len(s2_groups) == int(audit["stage2"]["compatible_sets"]), "Stage2 group count mismatch")
    require(
        sum(len({s for s, _ in items}) >= 2 for items in s2_groups.values())
        == int(audit["stage2"]["sets_with_ge2_surrogates"]),
        "Stage2 fusable count mismatch",
    )

    print("[Recompute] Stage 2 ...")
    s2_out = {}

    for key in sorted(
        s2_groups,
        key=lambda k: (k[0], MODEL_ORDER.get(k[1], 99), k[2], k[3], k[4]),
    ):
        case_id, model, prefix, pred, sig = key
        items = sorted(s2_groups[key], key=lambda x: SURROGATE_ORDER.get(x[0], 99))
        ctx = [cache[(case_id, model, int(prefix))]]
        current = items[0][1]

        for step, (surrogate, right) in enumerate(items[1:], 1):
            current, K, fallback, sel_mass, ign = fuse_pair(current, right, ctx)
            reproduced_trace.append({
                "lambda": LAMBDA,
                "case_id": case_id,
                "model": model,
                "prefix": int(prefix),
                "surrogate": f"cross-surrogate:{surrogate}",
                "predicted_class": int(pred),
                "feature_signature": json.dumps(list(sig)),
                "step": int(step),
                "stage": "Stage2",
                "K": K,
                "total_conflict_fallback": fallback,
                "left_R": None,
                "right_R": None,
                "selected_focal_mass": sel_mass,
                "ignorance_mass": ign,
                "output_n_r": int(current["n"]),
                "output_Fid_r": current["fid"],
                "output_R_r": float(current["R"]),
                "output_cf_valid": bool(current["cf_valid"]),
                "selected_conditions": json.dumps(
                    hypothesis_to_conditions(current["hyp"]),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            })

        s2_out[key] = current
        reproduced_stage.append(stage_row(current, "Stage2", key, len(items)))

    print(f"[Recompute] Stage 2 complete | groups={len(s2_out)}")

    # Stage3.
    s3_groups = defaultdict(list)
    for key, ev in s2_out.items():
        case_id, model, prefix, pred, sig = key
        s3_groups[(case_id, int(prefix), int(pred), sig)].append((model, ev))

    require(len(s3_groups) == int(audit["stage3"]["compatible_sets"]), "Stage3 group count mismatch")
    require(
        sum(len({m for m, _ in items}) >= 2 for items in s3_groups.values())
        == int(audit["stage3"]["sets_with_ge2_blackboxes"]),
        "Stage3 fusable count mismatch",
    )

    print("[Recompute] Stage 3 ...")
    s3_out = {}

    for key in sorted(s3_groups, key=lambda k: (k[0], k[1], k[2], k[3])):
        case_id, prefix, pred, sig = key
        items = sorted(s3_groups[key], key=lambda x: MODEL_ORDER.get(x[0], 99))

        current = items[0][1]
        models_so_far = [items[0][0]]

        for step, (model, right) in enumerate(items[1:], 1):
            models_so_far.append(model)
            contexts = [
                cache[(case_id, m, int(prefix))]
                for m in models_so_far
            ]
            current, K, fallback, sel_mass, ign = fuse_pair(current, right, contexts)
            reproduced_trace.append({
                "lambda": LAMBDA,
                "case_id": case_id,
                "model": f"cross-model:{model}",
                "prefix": int(prefix),
                "surrogate": None,
                "predicted_class": int(pred),
                "feature_signature": json.dumps(list(sig)),
                "step": int(step),
                "stage": "Stage3",
                "K": K,
                "total_conflict_fallback": fallback,
                "left_R": None,
                "right_R": None,
                "selected_focal_mass": sel_mass,
                "ignorance_mass": ign,
                "output_n_r": int(current["n"]),
                "output_Fid_r": current["fid"],
                "output_R_r": float(current["R"]),
                "output_cf_valid": bool(current["cf_valid"]),
                "selected_conditions": json.dumps(
                    hypothesis_to_conditions(current["hyp"]),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            })

        s3_out[key] = current
        reproduced_stage.append(stage_row(current, "Stage3", key, len(items)))

    print(f"[Recompute] Stage 3 complete | groups={len(s3_out)}")

    # Compare Stage output table.
    saved_stage = read_csv(SAVED_STAGE)
    require(len(saved_stage) == len(reproduced_stage), "Stage-output row count mismatch")

    saved_stage_idx = {stage_identity(r): r for r in saved_stage}
    require(len(saved_stage_idx) == len(saved_stage), "Duplicate saved stage identity")

    numeric_fields = [
        "lambda", "Fid_r", "coverage_eval", "rho_r", "R_r",
        "model_cf_validity_rate", "last_K", "last_selected_mass",
        "last_ignorance_mass",
    ]
    int_fields = [
        "input_count", "source_rule_count", "source_model_count",
        "source_surrogate_count", "n_r", "min_witness_edit",
        "fusion_steps", "total_conflict_fallback_count",
    ]
    bool_fields = ["actually_fused", "cf_valid", "all_models_cf_valid"]
    json_fields = ["feature_signature", "selected_conditions"]

    for expected in reproduced_stage:
        key = stage_identity(expected)
        require(key in saved_stage_idx, f"Missing saved stage row: {key}")
        compare_row(
            expected,
            saved_stage_idx[key],
            numeric_fields=numeric_fields,
            bool_fields=bool_fields,
            int_fields=int_fields,
            json_fields=json_fields,
            text_fields=[],
        )

    print(f"[Compare] Stage outputs: {len(reproduced_stage)} / {len(saved_stage)} exact identities matched")

    # Compare conflict trace. We independently validate all fields that are
    # derivable from the final pair result without reusing script 37's transient
    # left/right pre-state serialization.
    saved_trace = read_csv(SAVED_TRACE)
    require(len(saved_trace) == len(reproduced_trace), "Conflict-trace row count mismatch")

    saved_trace_idx = {trace_identity(r): r for r in saved_trace}
    require(len(saved_trace_idx) == len(saved_trace), "Duplicate saved trace identity")

    trace_numeric = [
        "lambda", "K", "selected_focal_mass", "ignorance_mass",
        "output_Fid_r", "output_R_r",
    ]
    trace_int = ["step", "output_n_r"]
    trace_bool = ["total_conflict_fallback", "output_cf_valid"]
    trace_json = ["feature_signature", "selected_conditions"]

    for expected in reproduced_trace:
        key = trace_identity(expected)
        require(key in saved_trace_idx, f"Missing saved trace row: {key}")
        compare_row(
            expected,
            saved_trace_idx[key],
            numeric_fields=trace_numeric,
            bool_fields=trace_bool,
            int_fields=trace_int,
            json_fields=trace_json,
            text_fields=[],
        )

    print(f"[Compare] Conflict trace: {len(reproduced_trace)} / {len(saved_trace)} exact identities matched")

    # Recompute summary and compare.
    reproduced_summary = [
        summarize(reproduced_stage, reproduced_trace, stage)
        for stage in ("Stage1", "Stage2", "Stage3")
    ]
    saved_summary = read_csv(SAVED_SUMMARY)
    require(len(saved_summary) == 3, "Saved summary must have exactly 3 rows")
    saved_summary_idx = {r["stage"]: r for r in saved_summary}

    for expected in reproduced_summary:
        saved = saved_summary_idx[expected["stage"]]
        for k, v in expected.items():
            if k == "stage":
                require(saved["stage"] == v, f"Stage summary name mismatch: {v}")
            elif isinstance(v, bool):
                require(bool(norm_scalar(saved.get(k))) == v, f"Summary bool mismatch: {k}")
            elif isinstance(v, int):
                require(int(norm_scalar(saved.get(k))) == v, f"Summary int mismatch: {k}")
            elif v is None:
                require(norm_scalar(saved.get(k)) is None, f"Summary null mismatch: {k}")
            else:
                require(
                    close(float(v), norm_scalar(saved.get(k))),
                    f"Summary numeric mismatch {expected['stage']} {k}: {v} != {saved.get(k)}",
                )

    print("[Compare] Stage summaries: PASSED")

    # Stage3 exported subset must exactly match Stage3 rows.
    saved_stage3 = read_csv(SAVED_STAGE3)
    reproduced_stage3 = [r for r in reproduced_stage if r["stage"] == "Stage3"]
    require(len(saved_stage3) == len(reproduced_stage3), "Stage3 export row count mismatch")
    saved_stage3_idx = {stage_identity(r): r for r in saved_stage3}
    for expected in reproduced_stage3:
        key = stage_identity(expected)
        require(key in saved_stage3_idx, f"Missing Stage3 export row: {key}")
        compare_row(
            expected,
            saved_stage3_idx[key],
            numeric_fields=numeric_fields,
            bool_fields=bool_fields,
            int_fields=int_fields,
            json_fields=json_fields,
            text_fields=[],
        )

    print("[Compare] Stage3 fused-rule export: PASSED")

    total_fallbacks = sum(
        int(r["total_conflict_fallback"])
        for r in reproduced_trace
    )
    require(
        int(saved_result["total_conflict_fallback_count"]) == total_fallbacks,
        "Saved result fallback total mismatch",
    )

    # Result JSON summary consistency.
    for expected in reproduced_summary:
        js = saved_result["summary"][expected["stage"]]
        for k, v in expected.items():
            if isinstance(v, str):
                require(js[k] == v, f"Result JSON text mismatch: {expected['stage']} {k}")
            elif isinstance(v, int):
                require(int(js[k]) == v, f"Result JSON int mismatch: {expected['stage']} {k}")
            elif v is None:
                require(js[k] is None, f"Result JSON null mismatch: {expected['stage']} {k}")
            else:
                require(
                    close(float(js[k]), float(v)),
                    f"Result JSON numeric mismatch: {expected['stage']} {k}",
                )

    validation = {
        "scope": "ETCE Phase 4D-5 independent validation of frozen formal TEST D-S fusion",
        "status": "passed",
        "lambda": LAMBDA,
        "lambda_selection_performed": False,
        "protocol_retuning_performed": False,
        "formal_test": True,
        "true_outcome_labels_used": False,
        "reproduced": {
            "canonical_rules": len(rules),
            "eligible_cf_candidates": len(cf),
            "cache_targets": len(cache),
            "stage1_groups": len(s1_out),
            "stage1_fusable": sum(len(v) >= 2 for v in s1_groups.values()),
            "stage2_groups": len(s2_out),
            "stage2_fusable": sum(
                len({s for s, _ in items}) >= 2
                for items in s2_groups.values()
            ),
            "stage3_groups": len(s3_out),
            "stage3_fusable": sum(
                len({m for m, _ in items}) >= 2
                for items in s3_groups.values()
            ),
            "stage_output_rows": len(reproduced_stage),
            "conflict_trace_rows": len(reproduced_trace),
            "total_conflict_fallback_count": total_fallbacks,
        },
        "checks": {
            "frozen_config_verified": True,
            "phase4d2_passed": True,
            "phase4d3_passed": True,
            "formal_cache_verified": True,
            "all_stage_output_identities_match": True,
            "all_checked_stage_values_match": True,
            "all_conflict_trace_identities_match": True,
            "all_checked_conflict_values_match": True,
            "all_stage_summaries_match": True,
            "stage3_export_matches": True,
            "result_json_matches": True,
        },
        "stage_summary": {
            r["stage"]: r for r in reproduced_summary
        },
        "next_step": (
            "Formal-test D-S outputs are independently validated. "
            "Proceed to Original -> Stage1 -> Stage2 -> Stage3 comparison "
            "and then predeclared fusion baselines/ablations."
        ),
    }

    write_json(OUT_VALIDATION, validation)

    s3 = validation["stage_summary"]["Stage3"]

    print()
    print("=" * 116)
    print("ETCE PHASE 4D-5 INDEPENDENT FROZEN D-S FORMAL TEST VALIDATION PASSED")
    print("=" * 116)
    print(f"Canonical rules reproduced: {len(rules)}")
    print(f"Eligible CF candidates: {len(cf)}")
    print(f"Formal TEST cache targets: {len(cache)}")
    print(
        "Groups / fusable: "
        f"Stage1={len(s1_out)}/{validation['reproduced']['stage1_fusable']} | "
        f"Stage2={len(s2_out)}/{validation['reproduced']['stage2_fusable']} | "
        f"Stage3={len(s3_out)}/{validation['reproduced']['stage3_fusable']}"
    )
    print(f"Stage output rows matched: {len(reproduced_stage)}")
    print(f"Conflict trace rows matched: {len(reproduced_trace)}")
    print(f"Total-conflict fallback count: {total_fallbacks}")
    print()
    print("Stage3 independently reproduced:")
    print(f"  Rule Fidelity        : {s3['fused_mean_fidelity']:.12f}")
    print(f"  CF Validity          : {s3['fused_cf_validity_rate']:.12f}")
    print(f"  Model-level Validity : {s3['fused_mean_model_cf_validity_rate']:.12f}")
    print(f"  All-model Validity   : {s3['fused_all_models_cf_valid_rate']:.12f}")
    print(f"  Mean Conflict K      : {s3['mean_conflict_K']:.12f}")
    print(f"  P90 Conflict K       : {s3['p90_conflict_K']:.12f}")
    print(f"  Max Conflict K       : {s3['max_conflict_K']:.12f}")
    print()
    print(f"Validation JSON: {OUT_VALIDATION}")
    print("Lambda selection performed: False")
    print("Protocol retuning performed: False")
    print("NEXT: Original -> Stage1 -> Stage2 -> Stage3 formal comparison.")


if __name__ == "__main__":
    main()
