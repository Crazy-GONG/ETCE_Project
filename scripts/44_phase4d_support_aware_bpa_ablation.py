#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-10 / Script 44
Support-aware BPA ablation on formal TEST.

Ablation:
    Full ETCE:
        rho = n/(n+5)
        R = rho * Fid
    w/o Support Adjustment:
        rho = 1
        R = Fid

Everything else is kept identical:
- same eligible formal TEST CF bank
- same Stage1/2/3 grouping
- same deterministic fusion order
- same Dempster combination
- same K -> 1 guard
- same representative-rule tie break
- same held-out cache
- same fused-rule re-evaluation policy
- no TEST-side tuning

The ablation definition was predeclared in the experiment plan and has
no tunable parameter.

Run:
    cd D:/ETCE
    python -u ./scripts/44_phase4d_support_aware_bpa_ablation.py
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]

FORMAL = ROOT / "results/phase4/phase4d_formal_test"
BANK = FORMAL / "formal_test_multirule_bank"
AUDIT_DIR = FORMAL / "formal_test_target_local_group_audit"
FULL_DIR = FORMAL / "formal_test_frozen_ds_fusion"
OUT = FORMAL / "formal_test_support_aware_bpa_ablation"

RULES = BANK / "canonical_rules_test.jsonl"
CACHE_DIR = BANK / "fusion_eval_cache" / "targets"
CACHE_MANIFEST = BANK / "fusion_eval_cache_manifest.json"
BANK_VALIDATION = BANK / "validation_phase4d_formal_test_multirule.json"
AUDIT = AUDIT_DIR / "formal_test_target_local_group_audit.json"

FROZEN_CONFIG = ROOT / "configs/phase4c_ds_frozen_config.json"
FULL_STAGE = FULL_DIR / "phase4d_frozen_ds_stage_outputs.csv"
FULL_VALIDATION = FULL_DIR / "validation_phase4d_frozen_ds_fusion.json"

LAMBDA = 5.0
K_EPS = 1e-12
THETA = None
EMPTY = object()

SURROGATE_ORDER = {"LORE": 0, "DT": 1, "BN": 2}
MODEL_ORDER = {"RF": 0, "XGBoost": 1, "MLP": 2, "LSTM": 3}

BOOTSTRAP_B = 20000
BOOTSTRAP_SEED = 20260930
TOL = 1e-12


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
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


def mean(xs):
    vals = [
        float(x)
        for x in xs
        if x is not None and math.isfinite(float(x))
    ]
    return float(sum(vals) / len(vals)) if vals else None


def median(xs):
    vals = [
        float(x)
        for x in xs
        if x is not None and math.isfinite(float(x))
    ]
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
        sorted_vals[lo] * (1.0 - frac)
        + sorted_vals[hi] * frac
    )


def bootstrap_ci_mean(
    diffs,
    B=BOOTSTRAP_B,
    seed=BOOTSTRAP_SEED,
):
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

        avg_rank = (i + 1 + j) / 2.0

        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank

        i = j

    return ranks


def rank_biserial(diffs):
    nz = [
        float(d)
        for d in diffs
        if abs(float(d)) > TOL
    ]

    if not nz:
        return 0.0

    ranks = average_ranks([abs(d) for d in nz])

    w_pos = sum(
        r for r, d in zip(ranks, nz)
        if d > 0
    )
    w_neg = sum(
        r for r, d in zip(ranks, nz)
        if d < 0
    )

    denom = w_pos + w_neg

    if denom == 0:
        return 0.0

    return float((w_pos - w_neg) / denom)


def exact_binom_two_sided(k, n):
    if n <= 0:
        return 1.0

    lo = min(int(k), int(n) - int(k))

    prob = sum(
        math.comb(int(n), i)
        for i in range(lo + 1)
    ) / (2 ** int(n))

    return min(1.0, 2.0 * prob)


def wilcoxon_or_sign(diffs):
    nz = [
        float(d)
        for d in diffs
        if abs(float(d)) > TOL
    ]

    if not nz:
        return {
            "test": "all_ties",
            "p_value": 1.0,
        }

    try:
        from scipy.stats import wilcoxon

        z = wilcoxon(
            nz,
            zero_method="wilcox",
            alternative="two-sided",
            correction=False,
            mode="auto",
        )

        return {
            "test": "wilcoxon_signed_rank_two_sided",
            "p_value": float(z.pvalue),
        }

    except Exception:
        positive = sum(d > 0 for d in nz)

        return {
            "test": "exact_sign_test_fallback",
            "p_value": exact_binom_two_sided(
                positive,
                len(nz),
            ),
        }


def conditions_to_hypothesis(conditions):
    by_pos = {}

    for c in conditions:
        require(
            c.get("operator") == "in",
            f"Unsupported operator: {c.get('operator')}",
        )

        p = int(c["position"])
        vals = tuple(
            sorted(str(v) for v in c["values"])
        )

        require(vals, f"Empty values at position {p}")
        require(p not in by_pos, f"Duplicate position {p}")

        by_pos[p] = vals

    return tuple(
        (p, by_pos[p])
        for p in sorted(by_pos)
    )


def hypothesis_to_conditions(hyp):
    if hyp is THETA:
        return []

    return [
        {
            "position": int(p),
            "operator": "in",
            "values": list(vals),
        }
        for p, vals in hyp
    ]


def feature_signature(hyp):
    if hyp is THETA:
        return tuple()

    return tuple(
        int(p)
        for p, _ in hyp
    )


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

    return sum(
        len(vals)
        for _, vals in hyp
    )


def intersect_hyp(a, b):
    if a is THETA:
        return b

    if b is THETA:
        return a

    da = {
        int(p): set(vals)
        for p, vals in a
    }
    db = {
        int(p): set(vals)
        for p, vals in b
    }

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

    return tuple(
        (p, out[p])
        for p in sorted(out)
    )


def simple_support_mass(hyp, R):
    R = min(
        1.0,
        max(0.0, float(R)),
    )

    out = defaultdict(float)
    out[hyp] += R
    out[THETA] += 1.0 - R

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
        return {
            THETA: 1.0
        }, float(K), True

    denom = 1.0 - K

    out = {
        h: float(v / denom)
        for h, v in raw.items()
        if v > 0.0
    }

    total = sum(out.values())

    if total > 0:
        out = {
            h: float(v / total)
            for h, v in out.items()
        }

    return out, float(K), False


def choose_proper_focal(mass):
    candidates = [
        (h, float(v))
        for h, v in mass.items()
        if h is not THETA
        and float(v) > 0.0
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
    require(
        len(a) == len(b),
        "Hamming length mismatch",
    )

    return sum(
        x != y
        for x, y in zip(a, b)
    )


def evaluate(
    hyp,
    predicted_class,
    contexts,
):
    total_eval = 0
    total_support = 0
    total_correct = 0
    global_min_edit = None
    per_model = []

    for payload in contexts:
        sequences = payload["pool_sequences"]
        labels = payload["pool_labels"]
        indices = payload["eval_indices"]
        factual = payload["factual_sequence"]

        support = 0
        correct = 0
        min_edit = None

        for raw_i in indices:
            idx = int(raw_i)
            seq = sequences[idx]

            total_eval += 1

            if not applies(seq, hyp):
                continue

            support += 1
            total_support += 1

            if int(labels[idx]) == int(predicted_class):
                correct += 1
                total_correct += 1

                ed = hamming(
                    factual,
                    seq,
                )

                if (
                    min_edit is None
                    or ed < min_edit
                ):
                    min_edit = ed

                if (
                    global_min_edit is None
                    or ed < global_min_edit
                ):
                    global_min_edit = ed

        per_model.append({
            "model": payload["model"],
            "support": support,
            "correct": correct,
            "cf_valid": bool(correct > 0),
            "fidelity": (
                float(correct / support)
                if support > 0
                else None
            ),
        })

    fid = (
        float(total_correct / total_support)
        if total_support > 0
        else None
    )

    coverage = (
        float(total_support / total_eval)
        if total_eval > 0
        else 0.0
    )

    model_validity = (
        float(
            sum(
                int(x["cf_valid"])
                for x in per_model
            )
            / len(per_model)
        )
        if per_model
        else None
    )

    return {
        "n_r": int(total_support),
        "Fid_r": fid,
        "coverage_eval": coverage,
        "cf_valid": bool(total_correct > 0),
        "model_cf_validity_rate": model_validity,
        "all_models_cf_valid": bool(
            per_model
            and all(
                x["cf_valid"]
                for x in per_model
            )
        ),
        "min_witness_edit": global_min_edit,
    }


def ablated_reliability(
    n,
    fid,
):
    """
    w/o Support Adjustment:
        rho := 1
        R := Fid
    """
    if int(n) <= 0 or fid is None:
        return 0.0

    return float(fid)


def source_evidence(rule):
    n = int(rule["n_r"])
    fid = float(rule["Fid_r"])

    return {
        "hyp": conditions_to_hypothesis(
            rule["conditions"]
        ),
        "pred": int(rule["predicted_class"]),
        "n": n,
        "fid": fid,
        "coverage": float(
            rule["coverage_eval"]
        ),
        "rho": 1.0,
        "R": ablated_reliability(
            n,
            fid,
        ),
        "source_count": 1,
        "models": {rule["model"]},
        "surrogates": {rule["surrogate"]},
        "candidate_ids": [
            str(rule["candidate_id"])
        ],
        "last_mass": ablated_reliability(
            n,
            fid,
        ),
        "last_ignorance": (
            1.0
            - ablated_reliability(
                n,
                fid,
            )
        ),
        "last_K": None,
        "fallbacks": 0,
        "steps": 0,
        "min_edit": rule.get(
            "witness_edit_count"
        ),
        "cf_valid": True,
        "model_valid": 1.0,
        "all_models_valid": True,
    }


def fuse_pair(
    left,
    right,
    contexts,
):
    require(
        left["pred"] == right["pred"],
        "Predicted-class mismatch",
    )

    require(
        feature_signature(left["hyp"])
        == feature_signature(right["hyp"]),
        "Feature-signature mismatch",
    )

    mass, K, fallback = combine_two(
        simple_support_mass(
            left["hyp"],
            left["R"],
        ),
        simple_support_mass(
            right["hyp"],
            right["R"],
        ),
    )

    selected, selected_mass = (
        choose_proper_focal(mass)
    )

    if selected is None:
        reps = [
            left,
            right,
        ]

        reps.sort(
            key=lambda ev: (
                -float(ev["R"]),
                specificity(ev["hyp"]),
                hyp_key(ev["hyp"]),
            )
        )

        selected = reps[0]["hyp"]
        selected_mass = 0.0

        met = evaluate(
            selected,
            left["pred"],
            contexts,
        )

        out_R = 0.0
        ignorance = 1.0

    else:
        met = evaluate(
            selected,
            left["pred"],
            contexts,
        )

        out_R = ablated_reliability(
            met["n_r"],
            met["Fid_r"],
        )

        ignorance = float(
            mass.get(
                THETA,
                0.0,
            )
        )

    return {
        "hyp": selected,
        "pred": int(left["pred"]),
        "n": int(met["n_r"]),
        "fid": met["Fid_r"],
        "coverage": float(
            met["coverage_eval"]
        ),
        "rho": (
            1.0
            if int(met["n_r"]) > 0
            else 0.0
        ),
        "R": float(out_R),
        "source_count": int(
            left["source_count"]
            + right["source_count"]
        ),
        "models": (
            set(left["models"])
            | set(right["models"])
        ),
        "surrogates": (
            set(left["surrogates"])
            | set(right["surrogates"])
        ),
        "candidate_ids": (
            list(left["candidate_ids"])
            + list(right["candidate_ids"])
        ),
        "last_mass": float(
            selected_mass
        ),
        "last_ignorance": float(
            ignorance
        ),
        "last_K": float(K),
        "fallbacks": int(
            left["fallbacks"]
            + right["fallbacks"]
            + int(fallback)
        ),
        "steps": int(
            left["steps"]
            + right["steps"]
            + 1
        ),
        "min_edit": met[
            "min_witness_edit"
        ],
        "cf_valid": bool(
            met["cf_valid"]
        ),
        "model_valid": met[
            "model_cf_validity_rate"
        ],
        "all_models_valid": bool(
            met["all_models_cf_valid"]
        ),
    }, float(K), bool(fallback)


def s1_key(rule):
    hyp = conditions_to_hypothesis(
        rule["conditions"]
    )

    return (
        rule["case_id"],
        rule["model"],
        int(rule["prefix"]),
        rule["surrogate"],
        int(rule["predicted_class"]),
        feature_signature(hyp),
    )


def stage_row(
    ev,
    stage,
    key,
    input_count,
):
    if stage == "Stage1":
        (
            case_id,
            model,
            prefix,
            surrogate,
            pred,
            sig,
        ) = key

    elif stage == "Stage2":
        (
            case_id,
            model,
            prefix,
            pred,
            sig,
        ) = key
        surrogate = None

    elif stage == "Stage3":
        (
            case_id,
            prefix,
            pred,
            sig,
        ) = key
        model = None
        surrogate = None

    else:
        raise ValueError(stage)

    return {
        "variant": "w/o_support_adjustment",
        "stage": stage,
        "case_id": case_id,
        "model": model,
        "prefix": int(prefix),
        "surrogate": surrogate,
        "predicted_class": int(pred),
        "feature_signature": json.dumps(
            list(sig)
        ),
        "input_count": int(
            input_count
        ),
        "actually_fused": bool(
            input_count >= 2
        ),
        "source_rule_count": int(
            ev["source_count"]
        ),
        "source_model_count": int(
            len(ev["models"])
        ),
        "source_surrogate_count": int(
            len(ev["surrogates"])
        ),
        "n_r": int(ev["n"]),
        "Fid_r": ev["fid"],
        "coverage_eval": float(
            ev["coverage"]
        ),
        "rho_r": float(
            ev["rho"]
        ),
        "R_r": float(
            ev["R"]
        ),
        "cf_valid": bool(
            ev["cf_valid"]
        ),
        "model_cf_validity_rate": (
            ev["model_valid"]
        ),
        "all_models_cf_valid": bool(
            ev["all_models_valid"]
        ),
        "min_witness_edit": (
            ev["min_edit"]
        ),
        "fusion_steps": int(
            ev["steps"]
        ),
        "total_conflict_fallback_count": int(
            ev["fallbacks"]
        ),
        "last_K": ev["last_K"],
        "last_selected_mass": float(
            ev["last_mass"]
        ),
        "last_ignorance_mass": float(
            ev["last_ignorance"]
        ),
        "selected_conditions": json.dumps(
            hypothesis_to_conditions(
                ev["hyp"]
            ),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def load_cache():
    idx = {}

    for p in sorted(
        CACHE_DIR.glob("*.json.gz")
    ):
        z = read_gz(p)

        key = (
            z["case_id"],
            z["model"],
            int(z["prefix"]),
        )

        require(
            key not in idx,
            f"Duplicate cache target: {key}",
        )

        idx[key] = z

    return idx


def summarize(
    rows,
    stage,
):
    sub = [
        r for r in rows
        if r["stage"] == stage
    ]

    fused = [
        r for r in sub
        if r["actually_fused"]
    ]

    return {
        "stage": stage,
        "group_count": len(sub),
        "fused_count": len(fused),
        "fused_mean_fidelity": mean(
            r["Fid_r"]
            for r in fused
        ),
        "fused_median_fidelity": median(
            r["Fid_r"]
            for r in fused
        ),
        "fused_cf_validity_rate": mean(
            int(r["cf_valid"])
            for r in fused
        ),
        "fused_mean_model_cf_validity_rate": mean(
            r["model_cf_validity_rate"]
            for r in fused
        ),
        "fused_all_models_cf_valid_rate": mean(
            int(r["all_models_cf_valid"])
            for r in fused
        ),
        "fused_mean_support": mean(
            r["n_r"]
            for r in fused
        ),
        "fused_mean_coverage": mean(
            r["coverage_eval"]
            for r in fused
        ),
        "fused_mean_reliability": mean(
            r["R_r"]
            for r in fused
        ),
        "fused_mean_ignorance": mean(
            r["last_ignorance_mass"]
            for r in fused
        ),
        "fused_mean_min_witness_edit": mean(
            r["min_witness_edit"]
            for r in fused
        ),
        "mean_conflict_K": mean(
            r["last_K"]
            for r in fused
        ),
        "p90_conflict_K": (
            percentile(
                sorted(
                    float(r["last_K"])
                    for r in fused
                    if r["last_K"] is not None
                ),
                0.90,
            )
            if fused
            else None
        ),
        "max_conflict_K": max(
            (
                float(r["last_K"])
                for r in fused
                if r["last_K"] is not None
            ),
            default=None,
        ),
        "total_conflict_fallback_count": sum(
            int(r[
                "total_conflict_fallback_count"
            ])
            for r in fused
        ),
    }


def parse_bool(v):
    if isinstance(v, bool):
        return v

    return str(v).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def parse_float(v):
    if v is None:
        return None

    s = str(v).strip()

    if s == "" or s.lower() in {
        "none",
        "nan",
        "null",
    }:
        return None

    return float(s)


def identity(row):
    return (
        row["case_id"],
        int(float(row["prefix"])),
        int(float(row["predicted_class"])),
        json.dumps(
            json.loads(
                row["feature_signature"]
            ),
            separators=(",", ":"),
        ),
    )


def paired_continuous(
    full_vals,
    ablated_vals,
):
    diffs = [
        float(a) - float(b)
        for a, b in zip(
            full_vals,
            ablated_vals,
        )
    ]

    wins = sum(
        d > TOL
        for d in diffs
    )
    ties = sum(
        abs(d) <= TOL
        for d in diffs
    )
    losses = sum(
        d < -TOL
        for d in diffs
    )

    ci_lo, ci_hi = bootstrap_ci_mean(
        diffs
    )

    p = wilcoxon_or_sign(
        diffs
    )

    return {
        "N": len(diffs),
        "full_mean": mean(
            full_vals
        ),
        "ablated_mean": mean(
            ablated_vals
        ),
        "delta_full_minus_ablated": mean(
            diffs
        ),
        "median_delta_full_minus_ablated": median(
            diffs
        ),
        "bootstrap95_delta_lo": ci_lo,
        "bootstrap95_delta_hi": ci_hi,
        "full_wins": wins,
        "ties": ties,
        "full_losses": losses,
        "paired_test": p["test"],
        "p_value": p["p_value"],
        "rank_biserial_effect": rank_biserial(
            diffs
        ),
    }


def paired_binary(
    full_vals,
    ablated_vals,
):
    f0a1 = 0
    f1a0 = 0
    same11 = 0
    same00 = 0

    for full, abl in zip(
        full_vals,
        ablated_vals,
    ):
        full = int(bool(full))
        abl = int(bool(abl))

        if full == 1 and abl == 1:
            same11 += 1

        elif full == 0 and abl == 0:
            same00 += 1

        elif full == 0 and abl == 1:
            f0a1 += 1

        else:
            f1a0 += 1

    discordant = f0a1 + f1a0

    return {
        "N": len(full_vals),
        "full_rate": mean(
            int(bool(v))
            for v in full_vals
        ),
        "ablated_rate": mean(
            int(bool(v))
            for v in ablated_vals
        ),
        "delta_full_minus_ablated": (
            mean(
                int(bool(v))
                for v in full_vals
            )
            - mean(
                int(bool(v))
                for v in ablated_vals
            )
        ),
        "agree_11": same11,
        "agree_00": same00,
        "full0_ablated1": f0a1,
        "full1_ablated0": f1a0,
        "discordant": discordant,
        "test": (
            "exact_mcnemar_binomial_two_sided"
        ),
        "p_value": exact_binom_two_sided(
            f1a0,
            discordant,
        ),
    }


def main():
    required = [
        RULES,
        CACHE_MANIFEST,
        BANK_VALIDATION,
        AUDIT,
        FROZEN_CONFIG,
        FULL_STAGE,
        FULL_VALIDATION,
    ]

    missing = [
        str(p)
        for p in required
        if not p.is_file()
    ]

    require(
        not missing,
        "Missing required artifact(s):\n"
        + "\n".join(missing),
    )

    require(
        CACHE_DIR.is_dir(),
        f"Missing cache dir: {CACHE_DIR}",
    )

    frozen = read_json(
        FROZEN_CONFIG
    )
    audit = read_json(
        AUDIT
    )
    bank_val = read_json(
        BANK_VALIDATION
    )
    cache_manifest = read_json(
        CACHE_MANIFEST
    )
    full_validation = read_json(
        FULL_VALIDATION
    )

    require(
        frozen.get("status") == "frozen",
        "D-S protocol not frozen",
    )
    require(
        float(frozen.get("lambda"))
        == LAMBDA,
        "Frozen lambda != 5",
    )
    require(
        frozen.get("selection_scope")
        == "validation_only",
        "Lambda not validation-only",
    )
    require(
        audit.get("status") == "passed",
        "Formal grouping audit not PASSED",
    )
    require(
        bank_val.get("status") == "passed",
        "Formal rule-bank validation not PASSED",
    )
    require(
        cache_manifest.get("status")
        == "built",
        "Formal cache not built",
    )
    require(
        full_validation.get("status")
        == "passed",
        "Full ETCE independent validation not PASSED",
    )

    rules = read_jsonl(
        RULES
    )

    cf = [
        r for r in rules
        if r.get("rule_type")
        == "counterfactual"
        and bool(
            r.get(
                "eligible_for_reliability"
            )
        )
    ]

    require(
        len(rules) == 9846,
        f"Canonical rule count mismatch: {len(rules)}",
    )
    require(
        len(cf) == 6172,
        f"Eligible CF count mismatch: {len(cf)}",
    )

    cache = load_cache()

    require(
        len(cache) == 1600,
        f"Cache target count mismatch: {len(cache)}",
    )

    print("=" * 118)
    print("ETCE PHASE 4D-10 SUPPORT-AWARE BPA ABLATION")
    print("=" * 118)
    print("Full ETCE             : R = n/(n+5) * Fid")
    print("w/o Support Adjustment: R = Fid")
    print("Same grouping/order/D-S/tie-break/cache: True")
    print("TEST-side tuning: False")
    print()

    # ==========================================================
    # Stage 1
    # ==========================================================
    g1 = defaultdict(list)

    for r in cf:
        g1[
            s1_key(r)
        ].append(r)

    require(
        len(g1)
        == int(
            audit["stage1"][
                "group_count"
            ]
        ),
        "Stage1 group count mismatch",
    )

    s1_out = {}
    stage_rows = []

    print("[Ablation] Stage 1 ...")

    for key in sorted(
        g1,
        key=lambda k: (
            k[0],
            MODEL_ORDER.get(
                k[1],
                99,
            ),
            k[2],
            SURROGATE_ORDER.get(
                k[3],
                99,
            ),
            k[4],
            k[5],
        ),
    ):
        (
            case_id,
            model,
            prefix,
            surrogate,
            pred,
            sig,
        ) = key

        ordered = sorted(
            g1[key],
            key=lambda r: str(
                r["candidate_id"]
            ),
        )

        ctx = [
            cache[
                (
                    case_id,
                    model,
                    int(prefix),
                )
            ]
        ]

        current = source_evidence(
            ordered[0]
        )

        for r in ordered[1:]:
            current, _, _ = fuse_pair(
                current,
                source_evidence(r),
                ctx,
            )

        s1_out[key] = current

        stage_rows.append(
            stage_row(
                current,
                "Stage1",
                key,
                len(ordered),
            )
        )

    print(
        f"[Ablation] Stage 1 complete | groups={len(s1_out)}"
    )

    # ==========================================================
    # Stage 2
    # ==========================================================
    g2 = defaultdict(list)

    for key, ev in s1_out.items():
        (
            case_id,
            model,
            prefix,
            surrogate,
            pred,
            sig,
        ) = key

        g2[
            (
                case_id,
                model,
                prefix,
                pred,
                sig,
            )
        ].append(
            (
                surrogate,
                ev,
            )
        )

    require(
        len(g2)
        == int(
            audit["stage2"][
                "compatible_sets"
            ]
        ),
        "Stage2 group count mismatch",
    )

    s2_out = {}

    print("[Ablation] Stage 2 ...")

    for key in sorted(
        g2,
        key=lambda k: (
            k[0],
            MODEL_ORDER.get(
                k[1],
                99,
            ),
            k[2],
            k[3],
            k[4],
        ),
    ):
        (
            case_id,
            model,
            prefix,
            pred,
            sig,
        ) = key

        items = sorted(
            g2[key],
            key=lambda x: (
                SURROGATE_ORDER.get(
                    x[0],
                    99,
                )
            ),
        )

        ctx = [
            cache[
                (
                    case_id,
                    model,
                    int(prefix),
                )
            ]
        ]

        current = items[0][1]

        for _, right in items[1:]:
            current, _, _ = fuse_pair(
                current,
                right,
                ctx,
            )

        s2_out[key] = current

        stage_rows.append(
            stage_row(
                current,
                "Stage2",
                key,
                len(items),
            )
        )

    print(
        f"[Ablation] Stage 2 complete | groups={len(s2_out)}"
    )

    # ==========================================================
    # Stage 3
    # ==========================================================
    g3 = defaultdict(list)

    for key, ev in s2_out.items():
        (
            case_id,
            model,
            prefix,
            pred,
            sig,
        ) = key

        g3[
            (
                case_id,
                prefix,
                pred,
                sig,
            )
        ].append(
            (
                model,
                ev,
            )
        )

    require(
        len(g3)
        == int(
            audit["stage3"][
                "compatible_sets"
            ]
        ),
        "Stage3 group count mismatch",
    )

    s3_out = {}

    print("[Ablation] Stage 3 ...")

    for key in sorted(
        g3,
        key=lambda k: (
            k[0],
            k[1],
            k[2],
            k[3],
        ),
    ):
        (
            case_id,
            prefix,
            pred,
            sig,
        ) = key

        items = sorted(
            g3[key],
            key=lambda x: (
                MODEL_ORDER.get(
                    x[0],
                    99,
                )
            ),
        )

        current = items[0][1]
        models_so_far = [
            items[0][0]
        ]

        for model, right in items[1:]:
            models_so_far.append(
                model
            )

            contexts = [
                cache[
                    (
                        case_id,
                        m,
                        int(prefix),
                    )
                ]
                for m in models_so_far
            ]

            current, _, _ = fuse_pair(
                current,
                right,
                contexts,
            )

        s3_out[key] = current

        stage_rows.append(
            stage_row(
                current,
                "Stage3",
                key,
                len(items),
            )
        )

    print(
        f"[Ablation] Stage 3 complete | groups={len(s3_out)}"
    )

    # ==========================================================
    # Summaries
    # ==========================================================
    summaries = [
        summarize(
            stage_rows,
            stage,
        )
        for stage in (
            "Stage1",
            "Stage2",
            "Stage3",
        )
    ]

    # Full ETCE Stage3 matched rows.
    full_rows_all = read_csv(
        FULL_STAGE
    )

    full_s3 = [
        r for r in full_rows_all
        if r["stage"] == "Stage3"
        and parse_bool(
            r["actually_fused"]
        )
    ]

    abl_s3 = [
        r for r in stage_rows
        if r["stage"] == "Stage3"
        and r["actually_fused"]
    ]

    require(
        len(full_s3) == 529,
        f"Full Stage3 fused rows !=529: {len(full_s3)}",
    )
    require(
        len(abl_s3) == 529,
        f"Ablated Stage3 fused rows !=529: {len(abl_s3)}",
    )

    full_idx = {
        identity(r): r
        for r in full_s3
    }
    abl_idx = {
        identity(r): r
        for r in abl_s3
    }

    require(
        set(full_idx)
        == set(abl_idx),
        "Full/ablated Stage3 identity sets differ",
    )

    keys = sorted(
        full_idx
    )

    paired_rows = []

    for metric, field in [
        (
            "rule_fidelity",
            "Fid_r",
        ),
        (
            "model_level_validity",
            "model_cf_validity_rate",
        ),
        (
            "reliability",
            "R_r",
        ),
        (
            "coverage",
            "coverage_eval",
        ),
        (
            "support",
            "n_r",
        ),
    ]:
        full_vals = [
            parse_float(
                full_idx[k][field]
            )
            for k in keys
        ]

        abl_vals = [
            parse_float(
                abl_idx[k][field]
            )
            for k in keys
        ]

        require(
            all(
                v is not None
                for v in (
                    full_vals
                    + abl_vals
                )
            ),
            f"Null in paired metric: {metric}",
        )

        paired_rows.append({
            "metric": metric,
            **paired_continuous(
                full_vals,
                abl_vals,
            ),
        })

    binary_rows = []

    for metric, field in [
        (
            "cf_validity",
            "cf_valid",
        ),
        (
            "all_model_validity",
            "all_models_cf_valid",
        ),
    ]:
        full_vals = [
            parse_bool(
                full_idx[k][field]
            )
            for k in keys
        ]

        abl_vals = [
            bool(
                abl_idx[k][field]
            )
            for k in keys
        ]

        binary_rows.append({
            "metric": metric,
            **paired_binary(
                full_vals,
                abl_vals,
            ),
        })

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        OUT
        / "support_ablation_stage_outputs.csv",
        stage_rows,
    )

    write_csv(
        OUT
        / "support_ablation_stage_summary.csv",
        summaries,
    )

    write_csv(
        OUT
        / "support_ablation_paired_continuous.csv",
        paired_rows,
    )

    write_csv(
        OUT
        / "support_ablation_paired_binary.csv",
        binary_rows,
    )

    result = {
        "scope": (
            "ETCE Phase 4D-10 support-aware BPA ablation"
        ),
        "status": "passed",
        "formal_test": True,
        "full_variant": {
            "rho": "n/(n+5)",
            "R": "rho*Fid",
        },
        "ablated_variant": {
            "name": "w/o Support Adjustment",
            "rho": "1",
            "R": "Fid",
        },
        "same_grouping": True,
        "same_fusion_order": True,
        "same_ds_combination": True,
        "same_tie_break": True,
        "same_eval_cache": True,
        "test_side_tuning": False,
        "stage_summary": {
            r["stage"]: r
            for r in summaries
        },
        "paired_stage3_continuous": paired_rows,
        "paired_stage3_binary": binary_rows,
        "next_step": (
            "Interpret support-aware reliability contribution; "
            "then run agreement/dependence/conflict analysis."
        ),
    }

    write_json(
        OUT
        / "support_aware_bpa_ablation.json",
        result,
    )

    # Human report
    lines = [
        "# ETCE Support-aware BPA Ablation",
        "",
        "Full: R = n/(n+5) * Fid",
        "",
        "Ablated: R = Fid",
        "",
        "Everything else is held fixed.",
        "",
        "## Stage summary",
        "",
        "| Stage | N fused | Fidelity | CF-valid | Model-valid | All-model-valid | Mean R | Mean ignorance | Mean K(last) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in summaries:
        lines.append(
            f"| {r['stage']} | "
            f"{r['fused_count']} | "
            f"{r['fused_mean_fidelity']:.6f} | "
            f"{r['fused_cf_validity_rate']:.6f} | "
            f"{r['fused_mean_model_cf_validity_rate']:.6f} | "
            f"{r['fused_all_models_cf_valid_rate']:.6f} | "
            f"{r['fused_mean_reliability']:.6f} | "
            f"{r['fused_mean_ignorance']:.6f} | "
            f"{r['mean_conflict_K']:.6f} |"
        )

    lines.extend([
        "",
        "## Stage3 paired Full minus Ablated",
        "",
        "| Metric | Full | Ablated | Delta | 95% CI | W/T/L | p | Rank-biserial |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for r in paired_rows:
        lines.append(
            f"| {r['metric']} | "
            f"{r['full_mean']:.6f} | "
            f"{r['ablated_mean']:.6f} | "
            f"{r['delta_full_minus_ablated']:+.6f} | "
            f"[{r['bootstrap95_delta_lo']:+.6f}, "
            f"{r['bootstrap95_delta_hi']:+.6f}] | "
            f"{r['full_wins']}/{r['ties']}/{r['full_losses']} | "
            f"{r['p_value']:.6g} | "
            f"{r['rank_biserial_effect']:+.4f} |"
        )

    (
        OUT
        / "PHASE4D_SUPPORT_AWARE_BPA_ABLATION_REPORT.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 118)
    print("ETCE PHASE 4D-10 SUPPORT-AWARE BPA ABLATION PASSED")
    print("=" * 118)

    for r in summaries:
        print(
            f"{r['stage']} | "
            f"fused={r['fused_count']} | "
            f"Fid={r['fused_mean_fidelity']:.6f} | "
            f"CF={r['fused_cf_validity_rate']:.6f} | "
            f"ModelValid={r['fused_mean_model_cf_validity_rate']:.6f} | "
            f"AllModel={r['fused_all_models_cf_valid_rate']:.6f} | "
            f"R={r['fused_mean_reliability']:.6f} | "
            f"meanK={r['mean_conflict_K']:.6f}"
        )

    print()
    print("[Stage3 paired Full ETCE minus w/o Support Adjustment]")

    for r in paired_rows:
        print(
            f"{r['metric']:24s} | "
            f"Full={r['full_mean']:.6f} | "
            f"Abl={r['ablated_mean']:.6f} | "
            f"Delta={r['delta_full_minus_ablated']:+.6f} | "
            f"CI=[{r['bootstrap95_delta_lo']:+.6f},"
            f"{r['bootstrap95_delta_hi']:+.6f}] | "
            f"W/T/L={r['full_wins']}/{r['ties']}/{r['full_losses']} | "
            f"p={r['p_value']:.6g} | "
            f"r_rb={r['rank_biserial_effect']:+.4f}"
        )

    print()
    print("[Stage3 binary paired Full ETCE minus Ablation]")

    for r in binary_rows:
        print(
            f"{r['metric']:24s} | "
            f"Full={r['full_rate']:.6f} | "
            f"Abl={r['ablated_rate']:.6f} | "
            f"Delta={r['delta_full_minus_ablated']:+.6f} | "
            f"discordant={r['discordant']} | "
            f"p={r['p_value']:.6g}"
        )

    print()
    print(f"Output: {OUT}")
    print("TEST-side tuning performed: False")
    print("NEXT: agreement / dependence / conflict analysis.")


if __name__ == "__main__":
    main()
