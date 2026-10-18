#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-12 / Script 46
Controlled correlated-source stress test.

Predeclared experiment-plan idea:
    LORE + DT + BN
vs
    LORE + DT + DT'

Concrete stress definition used here:
    DT' is an EXACT DUPLICATE of the Stage-1 DT evidence unit.

This is intentionally an upper-bound / worst-case dependence stress:
- DT' is NOT claimed to be an independent surrogate.
- It is NOT a newly trained model.
- It represents perfect source duplication / maximal dependence.
- No tunable parameter is introduced.
- No ETCE protocol is retuned.

Scope:
- Formal TEST Stage-2 groups containing all three original surrogates
  LORE, DT, BN.
- Expected count from the validated grouping audit: 155.
- Compare original frozen Stage-2 ETCE output against the stress output
  obtained from LORE -> DT -> DT' using the SAME D-S rule and SAME held-out
  cache.

Primary questions:
1) Does duplicated dependent evidence change the selected fused rule?
2) Does it change Fidelity / CF validity / coverage / support?
3) Does it artificially increase selected focal mass or reduce ignorance?
4) How does conflict K change?

This is a post-hoc sensitivity/stress analysis, not a new primary endpoint.

Run:
    cd D:/ETCE
    python -u ./scripts/46_phase4d_correlated_source_stress.py
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
DS_DIR = FORMAL / "formal_test_frozen_ds_fusion"
AUDIT_DIR = FORMAL / "formal_test_target_local_group_audit"
OUT = FORMAL / "formal_test_correlated_source_stress"

CACHE_DIR = BANK / "fusion_eval_cache" / "targets"
STAGE_CSV = DS_DIR / "phase4d_frozen_ds_stage_outputs.csv"
TRACE_CSV = DS_DIR / "phase4d_frozen_ds_conflict_trace.csv"
DS_VALIDATION = DS_DIR / "validation_phase4d_frozen_ds_fusion.json"
AUDIT = AUDIT_DIR / "formal_test_target_local_group_audit.json"
FROZEN = ROOT / "configs/phase4c_ds_frozen_config.json"

LAMBDA = 5.0
K_EPS = 1e-12
THETA = None
EMPTY = object()
TOL = 1e-12

SURROGATE_ORDER = {
    "LORE": 0,
    "DT": 1,
    "BN": 2,
}

BOOTSTRAP_B = 20000
BOOTSTRAP_SEED = 20260930


# ---------------------------------------------------------------------
# IO / helpers
# ---------------------------------------------------------------------

def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


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


def parse_bool(v):
    if isinstance(v, bool):
        return v

    return str(v).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def mean(xs):
    vals = [
        float(x)
        for x in xs
        if x is not None
        and math.isfinite(float(x))
    ]

    return (
        float(sum(vals) / len(vals))
        if vals
        else None
    )


def median(xs):
    vals = [
        float(x)
        for x in xs
        if x is not None
        and math.isfinite(float(x))
    ]

    return (
        float(statistics.median(vals))
        if vals
        else None
    )


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None

    if len(sorted_vals) == 1:
        return float(sorted_vals[0])

    z = (
        (len(sorted_vals) - 1)
        * float(p)
    )

    lo = int(math.floor(z))
    hi = int(math.ceil(z))

    if lo == hi:
        return float(
            sorted_vals[lo]
        )

    frac = z - lo

    return float(
        sorted_vals[lo]
        * (1 - frac)
        + sorted_vals[hi]
        * frac
    )


def bootstrap_ci_mean(
    diffs,
    B=BOOTSTRAP_B,
    seed=BOOTSTRAP_SEED,
):
    vals = [
        float(x)
        for x in diffs
    ]

    require(
        vals,
        "Empty bootstrap input",
    )

    rng = random.Random(seed)
    n = len(vals)
    boots = []

    for _ in range(B):
        s = 0.0

        for _j in range(n):
            s += vals[
                rng.randrange(n)
            ]

        boots.append(
            s / n
        )

    boots.sort()

    return (
        percentile(
            boots,
            0.025,
        ),
        percentile(
            boots,
            0.975,
        ),
    )


def exact_binom_two_sided(k, n):
    if n <= 0:
        return 1.0

    k = int(k)
    n = int(n)
    lo = min(
        k,
        n - k,
    )

    prob = sum(
        math.comb(n, i)
        for i in range(lo + 1)
    ) / (2 ** n)

    return min(
        1.0,
        2.0 * prob,
    )


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
            "p_value": float(
                z.pvalue
            ),
        }

    except Exception:
        positive = sum(
            d > 0
            for d in nz
        )

        return {
            "test": "exact_sign_test_fallback",
            "p_value": exact_binom_two_sided(
                positive,
                len(nz),
            ),
        }


# ---------------------------------------------------------------------
# D-S canonical hypothesis logic
# ---------------------------------------------------------------------

def conditions_to_hypothesis(conditions):
    by_pos = {}

    for c in conditions:
        require(
            c.get("operator") == "in",
            (
                "Unsupported canonical "
                f"operator: {c.get('operator')}"
            ),
        )

        p = int(
            c["position"]
        )

        vals = tuple(
            sorted(
                str(v)
                for v in c["values"]
            )
        )

        require(
            vals,
            f"Empty values at position {p}",
        )

        require(
            p not in by_pos,
            f"Duplicate position {p}",
        )

        by_pos[p] = vals

    return tuple(
        (
            p,
            by_pos[p],
        )
        for p in sorted(
            by_pos
        )
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
        hypothesis_to_conditions(
            hyp
        ),
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

    for p in sorted(
        set(da)
        | set(db)
    ):
        if p in da and p in db:
            vals = (
                da[p]
                & db[p]
            )

            if not vals:
                return EMPTY

            out[p] = tuple(
                sorted(vals)
            )

        elif p in da:
            out[p] = tuple(
                sorted(
                    da[p]
                )
            )

        else:
            out[p] = tuple(
                sorted(
                    db[p]
                )
            )

    return tuple(
        (
            p,
            out[p],
        )
        for p in sorted(
            out
        )
    )


def initial_mass(
    hyp,
    reliability,
):
    r = min(
        1.0,
        max(
            0.0,
            float(reliability),
        ),
    )

    out = defaultdict(float)
    out[hyp] += r
    out[THETA] += (
        1.0 - r
    )

    return dict(out)


def combine_mass(m1, m2):
    raw = defaultdict(float)
    conflict = 0.0

    for a, ma in m1.items():
        for b, mb in m2.items():
            prod = (
                float(ma)
                * float(mb)
            )

            inter = intersect_hyp(
                a,
                b,
            )

            if inter is EMPTY:
                conflict += prod

            else:
                raw[inter] += prod

    if (
        1.0 - conflict
        <= K_EPS
    ):
        return (
            {
                THETA: 1.0
            },
            float(conflict),
            True,
        )

    denom = (
        1.0
        - conflict
    )

    normalized = {
        h: float(
            v / denom
        )
        for h, v in raw.items()
        if v > 0.0
    }

    total = sum(
        normalized.values()
    )

    if total > 0:
        normalized = {
            h: float(
                v / total
            )
            for h, v in normalized.items()
        }

    return (
        normalized,
        float(conflict),
        False,
    )


def select_interpretable_hypothesis(
    mass,
):
    proper = [
        (
            h,
            float(v),
        )
        for h, v in mass.items()
        if h is not THETA
        and float(v) > 0.0
    ]

    if not proper:
        return (
            None,
            0.0,
        )

    proper.sort(
        key=lambda hv: (
            -hv[1],
            specificity(
                hv[0]
            ),
            hyp_key(
                hv[0]
            ),
        )
    )

    return proper[0]


def applies(sequence, hyp):
    if hyp is THETA:
        return True

    for p, vals in hyp:
        idx = int(p) - 1

        if (
            idx < 0
            or idx
            >= len(sequence)
        ):
            return False

        if (
            str(
                sequence[idx]
            )
            not in set(vals)
        ):
            return False

    return True


def hamming(a, b):
    require(
        len(a) == len(b),
        "Hamming length mismatch",
    )

    return sum(
        x != y
        for x, y in zip(
            a,
            b,
        )
    )


# ---------------------------------------------------------------------
# Formal cache / evaluation
# ---------------------------------------------------------------------

def load_cache():
    idx = {}

    require(
        CACHE_DIR.is_dir(),
        f"Missing cache: {CACHE_DIR}",
    )

    for p in sorted(
        CACHE_DIR.glob(
            "*.json.gz"
        )
    ):
        z = read_gz(p)

        key = (
            z["case_id"],
            z["model"],
            int(
                z["prefix"]
            ),
        )

        require(
            key not in idx,
            f"Duplicate cache key: {key}",
        )

        idx[key] = z

    return idx


def evaluate_hypothesis(
    hyp,
    predicted_class,
    contexts,
):
    total_eval = 0
    total_support = 0
    total_correct = 0
    min_edit = None
    per_model = []

    for payload in contexts:
        sequences = payload[
            "pool_sequences"
        ]

        labels = payload[
            "pool_labels"
        ]

        eval_indices = payload[
            "eval_indices"
        ]

        factual = payload[
            "factual_sequence"
        ]

        m_support = 0
        m_correct = 0
        m_min_edit = None

        for raw_i in eval_indices:
            i = int(raw_i)
            seq = sequences[i]
            total_eval += 1

            if not applies(
                seq,
                hyp,
            ):
                continue

            total_support += 1
            m_support += 1

            if (
                int(
                    labels[i]
                )
                == int(
                    predicted_class
                )
            ):
                total_correct += 1
                m_correct += 1

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
                    m_min_edit is None
                    or ed < m_min_edit
                ):
                    m_min_edit = ed

        per_model.append({
            "model": payload[
                "model"
            ],
            "support": m_support,
            "correct": m_correct,
            "cf_valid": bool(
                m_correct > 0
            ),
        })

    fidelity = (
        float(
            total_correct
            / total_support
        )
        if total_support > 0
        else None
    )

    coverage = (
        float(
            total_support
            / total_eval
        )
        if total_eval > 0
        else 0.0
    )

    model_validity = (
        float(
            sum(
                int(
                    x[
                        "cf_valid"
                    ]
                )
                for x in per_model
            )
            / len(per_model)
        )
        if per_model
        else None
    )

    return {
        "n_r": int(
            total_support
        ),
        "Fid_r": fidelity,
        "coverage_eval": coverage,
        "cf_valid": bool(
            total_correct > 0
        ),
        "model_cf_validity_rate": (
            model_validity
        ),
        "all_models_cf_valid": bool(
            per_model
            and all(
                x[
                    "cf_valid"
                ]
                for x in per_model
            )
        ),
        "min_witness_edit": min_edit,
    }


def reliability(
    n,
    fid,
):
    n = int(n)

    if (
        n <= 0
        or fid is None
    ):
        return 0.0

    rho = (
        n
        / (
            n
            + LAMBDA
        )
    )

    return float(
        rho
        * float(fid)
    )


# ---------------------------------------------------------------------
# Evidence and fusion
# ---------------------------------------------------------------------

def evidence_from_stage1_row(
    r,
):
    hyp = conditions_to_hypothesis(
        json.loads(
            r[
                "selected_conditions"
            ]
        )
    )

    return {
        "hypothesis": hyp,
        "predicted_class": int(
            float(
                r[
                    "predicted_class"
                ]
            )
        ),
        "n_r": int(
            float(
                r["n_r"]
            )
        ),
        "Fid_r": parse_float(
            r["Fid_r"]
        ),
        "coverage_eval": parse_float(
            r[
                "coverage_eval"
            ]
        ),
        "rho_r": parse_float(
            r["rho_r"]
        ),
        "R_r": parse_float(
            r["R_r"]
        ),
        "source_count": int(
            float(
                r[
                    "source_rule_count"
                ]
            )
        ),
        "source_models": {
            r["model"]
        },
        "source_surrogates": {
            r["surrogate"]
        },
        "min_witness_edit": (
            parse_float(
                r[
                    "min_witness_edit"
                ]
            )
        ),
        "cf_valid": parse_bool(
            r["cf_valid"]
        ),
        "model_cf_validity_rate": parse_float(
            r[
                "model_cf_validity_rate"
            ]
        ),
        "all_models_cf_valid": parse_bool(
            r[
                "all_models_cf_valid"
            ]
        ),
        "fusion_steps": int(
            float(
                r[
                    "fusion_steps"
                ]
            )
        ),
        "total_conflict_fallback_count": int(
            float(
                r[
                    "total_conflict_fallback_count"
                ]
            )
        ),
        "last_K": parse_float(
            r["last_K"]
        ),
        "last_selected_mass": parse_float(
            r[
                "last_selected_mass"
            ]
        ),
        "last_ignorance_mass": parse_float(
            r[
                "last_ignorance_mass"
            ]
        ),
    }


def clone_as_dt_prime(ev):
    """
    Exact evidence duplicate.

    This is deliberately NOT independent evidence.
    """
    return {
        **ev,
        "source_models": set(
            ev[
                "source_models"
            ]
        ),
        "source_surrogates": {
            "DT_prime"
        },
    }


def fuse_pair(
    left,
    right,
    contexts,
):
    require(
        int(
            left[
                "predicted_class"
            ]
        )
        == int(
            right[
                "predicted_class"
            ]
        ),
        "Predicted-class mismatch",
    )

    require(
        feature_signature(
            left[
                "hypothesis"
            ]
        )
        == feature_signature(
            right[
                "hypothesis"
            ]
        ),
        "Feature-signature mismatch",
    )

    mass, K, fallback = (
        combine_mass(
            initial_mass(
                left[
                    "hypothesis"
                ],
                left["R_r"],
            ),
            initial_mass(
                right[
                    "hypothesis"
                ],
                right["R_r"],
            ),
        )
    )

    selected_hyp, selected_mass = (
        select_interpretable_hypothesis(
            mass
        )
    )

    if selected_hyp is None:
        reps = [
            left,
            right,
        ]

        reps.sort(
            key=lambda ev: (
                -float(
                    ev["R_r"]
                ),
                specificity(
                    ev[
                        "hypothesis"
                    ]
                ),
                hyp_key(
                    ev[
                        "hypothesis"
                    ]
                ),
            )
        )

        selected_hyp = reps[
            0
        ][
            "hypothesis"
        ]

        selected_mass = 0.0

        met = evaluate_hypothesis(
            selected_hyp,
            left[
                "predicted_class"
            ],
            contexts,
        )

        out_R = 0.0
        ignorance = 1.0

    else:
        met = evaluate_hypothesis(
            selected_hyp,
            left[
                "predicted_class"
            ],
            contexts,
        )

        out_R = reliability(
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
        "hypothesis": selected_hyp,
        "predicted_class": int(
            left[
                "predicted_class"
            ]
        ),
        "n_r": int(
            met["n_r"]
        ),
        "Fid_r": met[
            "Fid_r"
        ],
        "coverage_eval": float(
            met[
                "coverage_eval"
            ]
        ),
        "rho_r": (
            float(
                met["n_r"]
                / (
                    met["n_r"]
                    + LAMBDA
                )
            )
            if met["n_r"] > 0
            else 0.0
        ),
        "R_r": float(
            out_R
        ),
        "source_count": int(
            left[
                "source_count"
            ]
            + right[
                "source_count"
            ]
        ),
        "source_models": (
            set(
                left[
                    "source_models"
                ]
            )
            | set(
                right[
                    "source_models"
                ]
            )
        ),
        "source_surrogates": (
            set(
                left[
                    "source_surrogates"
                ]
            )
            | set(
                right[
                    "source_surrogates"
                ]
            )
        ),
        "min_witness_edit": met[
            "min_witness_edit"
        ],
        "cf_valid": bool(
            met[
                "cf_valid"
            ]
        ),
        "model_cf_validity_rate": (
            met[
                "model_cf_validity_rate"
            ]
        ),
        "all_models_cf_valid": bool(
            met[
                "all_models_cf_valid"
            ]
        ),
        "fusion_steps": int(
            left[
                "fusion_steps"
            ]
            + right[
                "fusion_steps"
            ]
            + 1
        ),
        "total_conflict_fallback_count": int(
            left[
                "total_conflict_fallback_count"
            ]
            + right[
                "total_conflict_fallback_count"
            ]
            + int(
                fallback
            )
        ),
        "last_K": float(K),
        "last_selected_mass": float(
            selected_mass
        ),
        "last_ignorance_mass": float(
            ignorance
        ),
    }, float(K), bool(fallback)


# ---------------------------------------------------------------------
# Group identities
# ---------------------------------------------------------------------

def normalized_signature(s):
    return json.dumps(
        json.loads(
            str(s)
        ),
        separators=(",", ":"),
    )


def stage2_key(r):
    return (
        r[
            "case_id"
        ],
        r[
            "model"
        ],
        int(
            float(
                r[
                    "prefix"
                ]
            )
        ),
        int(
            float(
                r[
                    "predicted_class"
                ]
            )
        ),
        normalized_signature(
            r[
                "feature_signature"
            ]
        ),
    )


def stage1_to_stage2_key(r):
    return (
        r[
            "case_id"
        ],
        r[
            "model"
        ],
        int(
            float(
                r[
                    "prefix"
                ]
            )
        ),
        int(
            float(
                r[
                    "predicted_class"
                ]
            )
        ),
        normalized_signature(
            r[
                "feature_signature"
            ]
        ),
    )


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def paired_continuous(
    original,
    stress,
):
    require(
        len(original)
        == len(stress),
        "Paired length mismatch",
    )

    diffs = [
        float(o)
        - float(s)
        for o, s in zip(
            original,
            stress,
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

    lo, hi = bootstrap_ci_mean(
        diffs
    )

    p = wilcoxon_or_sign(
        diffs
    )

    return {
        "N": len(
            diffs
        ),
        "original_mean": mean(
            original
        ),
        "stress_mean": mean(
            stress
        ),
        "delta_original_minus_stress": mean(
            diffs
        ),
        "median_delta": median(
            diffs
        ),
        "bootstrap95_lo": lo,
        "bootstrap95_hi": hi,
        "original_wins": wins,
        "ties": ties,
        "stress_wins": losses,
        "paired_test": p[
            "test"
        ],
        "p_value": p[
            "p_value"
        ],
    }


def paired_binary(
    original,
    stress,
):
    o0s1 = 0
    o1s0 = 0
    same11 = 0
    same00 = 0

    for o, s in zip(
        original,
        stress,
    ):
        o = int(
            bool(o)
        )
        s = int(
            bool(s)
        )

        if o == 1 and s == 1:
            same11 += 1

        elif o == 0 and s == 0:
            same00 += 1

        elif o == 0 and s == 1:
            o0s1 += 1

        else:
            o1s0 += 1

    discordant = (
        o0s1
        + o1s0
    )

    return {
        "N": len(
            original
        ),
        "original_rate": mean(
            int(
                bool(v)
            )
            for v in original
        ),
        "stress_rate": mean(
            int(
                bool(v)
            )
            for v in stress
        ),
        "delta_original_minus_stress": (
            mean(
                int(
                    bool(v)
                )
                for v in original
            )
            - mean(
                int(
                    bool(v)
                )
                for v in stress
            )
        ),
        "original0_stress1": o0s1,
        "original1_stress0": o1s0,
        "discordant": discordant,
        "p_value": exact_binom_two_sided(
            o1s0,
            discordant,
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    required = [
        STAGE_CSV,
        TRACE_CSV,
        DS_VALIDATION,
        AUDIT,
        FROZEN,
    ]

    missing = [
        str(p)
        for p in required
        if not p.is_file()
    ]

    require(
        not missing,
        "Missing required file(s):\n"
        + "\n".join(
            missing
        ),
    )

    ds_validation = read_json(
        DS_VALIDATION
    )

    audit = read_json(
        AUDIT
    )

    frozen = read_json(
        FROZEN
    )

    require(
        ds_validation.get(
            "status"
        ) == "passed",
        "Formal D-S validation not PASSED",
    )

    require(
        audit.get(
            "status"
        ) == "passed",
        "Grouping audit not PASSED",
    )

    require(
        frozen.get(
            "status"
        ) == "frozen",
        "D-S config not frozen",
    )

    require(
        float(
            frozen[
                "lambda"
            ]
        )
        == LAMBDA,
        "Frozen lambda != 5",
    )

    rows = read_csv(
        STAGE_CSV
    )

    trace_rows = read_csv(
        TRACE_CSV
    )

    stage1 = [
        r
        for r in rows
        if r[
            "stage"
        ] == "Stage1"
    ]

    stage2 = [
        r
        for r in rows
        if r[
            "stage"
        ] == "Stage2"
    ]

    require(
        len(stage1) == 5056,
        (
            "Expected 5056 Stage1 outputs, "
            f"found {len(stage1)}"
        ),
    )

    require(
        len(stage2) == 4287,
        (
            "Expected 4287 Stage2 outputs, "
            f"found {len(stage2)}"
        ),
    )

    # Build Stage1 source inventory for Stage2.
    g = defaultdict(
        dict
    )

    for r in stage1:
        key = (
            stage1_to_stage2_key(
                r
            )
        )

        surrogate = r[
            "surrogate"
        ]

        require(
            surrogate
            in SURROGATE_ORDER,
            (
                "Unexpected surrogate "
                f"{surrogate}"
            ),
        )

        require(
            surrogate
            not in g[key],
            (
                "Duplicate Stage1 "
                f"{surrogate} source: {key}"
            ),
        )

        g[key][
            surrogate
        ] = r

    all3_keys = [
        key
        for key, d in g.items()
        if set(
            d
        ) == {
            "LORE",
            "DT",
            "BN",
        }
    ]

    require(
        len(all3_keys) == 155,
        (
            "Expected 155 Stage2 sets with "
            f"all 3 surrogates, found {len(all3_keys)}"
        ),
    )

    original_s2_idx = {
        stage2_key(
            r
        ): r
        for r in stage2
    }

    require(
        len(
            original_s2_idx
        ) == 4287,
        "Duplicate Stage2 output identity",
    )

    # Original trace groups for mean K comparison.
    trace_by_key = defaultdict(
        list
    )

    for r in trace_rows:
        if r[
            "stage"
        ] != "Stage2":
            continue

        trace_by_key[
            stage2_key(
                r
            )
        ].append(
            r
        )

    cache = load_cache()

    require(
        len(cache) == 1600,
        (
            "Formal cache target count "
            f"!=1600: {len(cache)}"
        ),
    )

    print("=" * 118)
    print("ETCE PHASE 4D-12 CONTROLLED CORRELATED-SOURCE STRESS TEST")
    print("=" * 118)
    print("Original evidence set : LORE + DT + BN")
    print("Stress evidence set   : LORE + DT + DT'")
    print("DT' definition        : exact duplicate of DT Stage-1 evidence")
    print("Dependence level      : perfect correlation / worst-case duplicate")
    print("DT' claimed independent: False")
    print("Tunable stress parameter: None")
    print("Protocol retuning: False")
    print(f"All-three-surrogate matched groups: {len(all3_keys)}")
    print()

    detail = []
    stress_fallback_total = 0

    for key in sorted(
        all3_keys
    ):
        (
            case_id,
            model,
            prefix,
            pred,
            sig_json,
        ) = key

        d = g[key]

        lore = evidence_from_stage1_row(
            d[
                "LORE"
            ]
        )

        dt = evidence_from_stage1_row(
            d[
                "DT"
            ]
        )

        bn = evidence_from_stage1_row(
            d[
                "BN"
            ]
        )

        dt_prime = clone_as_dt_prime(
            dt
        )

        require(
            hyp_key(
                dt_prime[
                    "hypothesis"
                ]
            )
            == hyp_key(
                dt[
                    "hypothesis"
                ]
            ),
            "DT' is not an exact hypothesis clone",
        )

        require(
            abs(
                float(
                    dt_prime[
                        "R_r"
                    ]
                )
                - float(
                    dt[
                        "R_r"
                    ]
                )
            )
            <= TOL,
            "DT' reliability changed",
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

        # Frozen order:
        # LORE -> DT -> DT'
        current, k1, fb1 = fuse_pair(
            lore,
            dt,
            ctx,
        )

        current, k2, fb2 = fuse_pair(
            current,
            dt_prime,
            ctx,
        )

        stress_fallback_total += (
            int(fb1)
            + int(fb2)
        )

        original = (
            original_s2_idx[
                key
            ]
        )

        tr = trace_by_key.get(
            key,
            [],
        )

        require(
            len(tr) == 2,
            (
                "Expected 2 original Stage2 "
                f"fusion steps for all-3 group: "
                f"{key}, got {len(tr)}"
            ),
        )

        original_ks = [
            parse_float(
                r[
                    "K"
                ]
            )
            for r in tr
        ]

        original_hyp = (
            conditions_to_hypothesis(
                json.loads(
                    original[
                        "selected_conditions"
                    ]
                )
            )
        )

        stress_hyp = current[
            "hypothesis"
        ]

        detail.append({
            "case_id": case_id,
            "model": model,
            "prefix": int(
                prefix
            ),
            "predicted_class": int(
                pred
            ),
            "feature_signature": (
                sig_json
            ),
            "DT_equals_BN_hypothesis": bool(
                hyp_key(
                    dt[
                        "hypothesis"
                    ]
                )
                == hyp_key(
                    bn[
                        "hypothesis"
                    ]
                )
            ),
            "original_selected_conditions": (
                original[
                    "selected_conditions"
                ]
            ),
            "stress_selected_conditions": json.dumps(
                hypothesis_to_conditions(
                    stress_hyp
                ),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "selected_rule_changed": bool(
                hyp_key(
                    original_hyp
                )
                != hyp_key(
                    stress_hyp
                )
            ),
            "original_Fid": parse_float(
                original[
                    "Fid_r"
                ]
            ),
            "stress_Fid": current[
                "Fid_r"
            ],
            "delta_original_minus_stress_Fid": (
                (
                    parse_float(
                        original[
                            "Fid_r"
                        ]
                    )
                    - current[
                        "Fid_r"
                    ]
                )
                if (
                    parse_float(
                        original[
                            "Fid_r"
                        ]
                    )
                    is not None
                    and current[
                        "Fid_r"
                    ]
                    is not None
                )
                else None
            ),
            "original_CF": int(
                parse_bool(
                    original[
                        "cf_valid"
                    ]
                )
            ),
            "stress_CF": int(
                current[
                    "cf_valid"
                ]
            ),
            "original_R": parse_float(
                original[
                    "R_r"
                ]
            ),
            "stress_R": current[
                "R_r"
            ],
            "original_support": parse_float(
                original[
                    "n_r"
                ]
            ),
            "stress_support": current[
                "n_r"
            ],
            "original_coverage": parse_float(
                original[
                    "coverage_eval"
                ]
            ),
            "stress_coverage": current[
                "coverage_eval"
            ],
            "original_min_edit": parse_float(
                original[
                    "min_witness_edit"
                ]
            ),
            "stress_min_edit": current[
                "min_witness_edit"
            ],
            "original_last_selected_mass": parse_float(
                original[
                    "last_selected_mass"
                ]
            ),
            "stress_last_selected_mass": current[
                "last_selected_mass"
            ],
            "original_last_ignorance": parse_float(
                original[
                    "last_ignorance_mass"
                ]
            ),
            "stress_last_ignorance": current[
                "last_ignorance_mass"
            ],
            "original_mean_K": mean(
                original_ks
            ),
            "stress_mean_K": mean(
                [
                    k1,
                    k2,
                ]
            ),
            "original_max_K": max(
                original_ks
            ),
            "stress_max_K": max(
                k1,
                k2,
            ),
            "original_last_K": parse_float(
                original[
                    "last_K"
                ]
            ),
            "stress_last_K": current[
                "last_K"
            ],
            "stress_fallback_count": int(
                fb1
            )
            + int(
                fb2
            ),
        })

    # --------------------------------------------------------------
    # Paired summaries
    # --------------------------------------------------------------
    continuous_specs = [
        (
            "rule_fidelity",
            "original_Fid",
            "stress_Fid",
        ),
        (
            "reliability",
            "original_R",
            "stress_R",
        ),
        (
            "support",
            "original_support",
            "stress_support",
        ),
        (
            "coverage",
            "original_coverage",
            "stress_coverage",
        ),
        (
            "selected_focal_mass",
            "original_last_selected_mass",
            "stress_last_selected_mass",
        ),
        (
            "ignorance_mass",
            "original_last_ignorance",
            "stress_last_ignorance",
        ),
        (
            "mean_conflict_K",
            "original_mean_K",
            "stress_mean_K",
        ),
        (
            "max_conflict_K",
            "original_max_K",
            "stress_max_K",
        ),
        (
            "last_conflict_K",
            "original_last_K",
            "stress_last_K",
        ),
    ]

    paired = []

    for (
        metric,
        ofield,
        sfield,
    ) in continuous_specs:
        usable = [
            r
            for r in detail
            if r[
                ofield
            ]
            is not None
            and r[
                sfield
            ]
            is not None
        ]

        paired.append({
            "metric": metric,
            **paired_continuous(
                [
                    r[
                        ofield
                    ]
                    for r in usable
                ],
                [
                    r[
                        sfield
                    ]
                    for r in usable
                ],
            ),
        })

    binary_cf = paired_binary(
        [
            r[
                "original_CF"
            ]
            for r in detail
        ],
        [
            r[
                "stress_CF"
            ]
            for r in detail
        ],
    )

    changed = sum(
        int(
            r[
                "selected_rule_changed"
            ]
        )
        for r in detail
    )

    dt_bn_same = sum(
        int(
            r[
                "DT_equals_BN_hypothesis"
            ]
        )
        for r in detail
    )

    # Stratify by whether original BN rule differed from DT.
    strata = []

    for label, flag in [
        (
            "DT_equals_BN",
            True,
        ),
        (
            "DT_differs_from_BN",
            False,
        ),
    ]:
        rows_s = [
            r
            for r in detail
            if r[
                "DT_equals_BN_hypothesis"
            ]
            is flag
        ]

        strata.append({
            "stratum": label,
            "N": len(
                rows_s
            ),
            "selected_rule_change_rate": mean(
                int(
                    r[
                        "selected_rule_changed"
                    ]
                )
                for r in rows_s
            ),
            "mean_original_fidelity": mean(
                r[
                    "original_Fid"
                ]
                for r in rows_s
            ),
            "mean_stress_fidelity": mean(
                r[
                    "stress_Fid"
                ]
                for r in rows_s
            ),
            "mean_delta_original_minus_stress_fidelity": mean(
                r[
                    "delta_original_minus_stress_Fid"
                ]
                for r in rows_s
            ),
            "mean_original_ignorance": mean(
                r[
                    "original_last_ignorance"
                ]
                for r in rows_s
            ),
            "mean_stress_ignorance": mean(
                r[
                    "stress_last_ignorance"
                ]
                for r in rows_s
            ),
            "mean_original_selected_mass": mean(
                r[
                    "original_last_selected_mass"
                ]
                for r in rows_s
            ),
            "mean_stress_selected_mass": mean(
                r[
                    "stress_last_selected_mass"
                ]
                for r in rows_s
            ),
        })

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        OUT
        / "correlated_source_stress_detail.csv",
        detail,
    )

    write_csv(
        OUT
        / "correlated_source_stress_paired_metrics.csv",
        paired,
    )

    write_csv(
        OUT
        / "correlated_source_stress_strata.csv",
        strata,
    )

    result = {
        "scope": (
            "ETCE Phase 4D-12 controlled correlated-source stress"
        ),
        "status": "passed",
        "formal_test": True,
        "stress_design": {
            "original": [
                "LORE",
                "DT",
                "BN",
            ],
            "stress": [
                "LORE",
                "DT",
                "DT_prime",
            ],
            "DT_prime_definition": (
                "exact duplicate of Stage1 DT evidence"
            ),
            "DT_prime_independent": False,
            "interpretation": (
                "perfect-correlation upper-bound / source-duplication sensitivity test"
            ),
            "tunable_parameter": None,
        },
        "chronology_disclosure": {
            "stress_family_predeclared_in_experiment_plan": True,
            "exact_DT_prime_as_perfect_clone_concretized_post_primary_test": True,
            "analysis_role": "post-hoc sensitivity / stress test",
        },
        "matched_all3_groups": len(
            detail
        ),
        "DT_equals_BN_hypothesis_count": (
            dt_bn_same
        ),
        "DT_differs_from_BN_hypothesis_count": (
            len(detail)
            - dt_bn_same
        ),
        "selected_rule_changed_count": (
            changed
        ),
        "selected_rule_changed_rate": (
            changed
            / len(detail)
        ),
        "stress_total_conflict_fallback_count": (
            stress_fallback_total
        ),
        "paired_continuous": paired,
        "paired_cf_validity": (
            binary_cf
        ),
        "strata": strata,
        "protocol_retuning": False,
        "next_step": (
            "Interpret dependence sensitivity, then proceed to process/temporal validity."
        ),
    }

    write_json(
        OUT
        / "correlated_source_stress_summary.json",
        result,
    )

    # Human report
    def fmt(v):
        return (
            "NA"
            if v is None
            else f"{float(v):.6f}"
        )

    lines = [
        "# ETCE Controlled Correlated-source Stress Test",
        "",
        "Original: LORE + DT + BN",
        "",
        "Stress: LORE + DT + DT′",
        "",
        "DT′ is an exact duplicate of DT and is explicitly treated as perfectly dependent evidence, not as an independent surrogate.",
        "",
        f"- Matched all-three-surrogate groups: {len(detail)}",
        f"- Selected fused rule changed: {changed}/{len(detail)}",
        f"- DT and BN originally had identical hypotheses: {dt_bn_same}/{len(detail)}",
        "",
        "## Paired results",
        "",
        "| Metric | Original | Stress | Delta original-stress | 95% CI | W/T/L | p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in paired:
        lines.append(
            f"| {r['metric']} | "
            f"{fmt(r['original_mean'])} | "
            f"{fmt(r['stress_mean'])} | "
            f"{fmt(r['delta_original_minus_stress'])} | "
            f"[{fmt(r['bootstrap95_lo'])}, {fmt(r['bootstrap95_hi'])}] | "
            f"{r['original_wins']}/{r['ties']}/{r['stress_wins']} | "
            f"{r['p_value']:.6g} |"
        )

    lines.extend([
        "",
        "This stress test evaluates sensitivity to source duplication / maximal dependence. "
        "It is not evidence that DT′ constitutes an additional independent source.",
    ])

    (
        OUT
        / "PHASE4D_CORRELATED_SOURCE_STRESS_REPORT.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 118)
    print("ETCE PHASE 4D-12 CONTROLLED CORRELATED-SOURCE STRESS TEST PASSED")
    print("=" * 118)
    print(f"Matched all-3 Stage2 groups: {len(detail)}")
    print(
        f"DT == BN original hypothesis: "
        f"{dt_bn_same}/{len(detail)}"
    )
    print(
        f"Selected fused rule changed under DT duplication: "
        f"{changed}/{len(detail)} "
        f"({changed/len(detail):.4%})"
    )
    print(
        f"Stress total-conflict fallback count: "
        f"{stress_fallback_total}"
    )
    print()

    print("[Paired Original (LORE+DT+BN) minus Stress (LORE+DT+DT')]")
    for r in paired:
        print(
            f"{r['metric']:24s} | "
            f"Orig={r['original_mean']:.6f} | "
            f"Stress={r['stress_mean']:.6f} | "
            f"Delta={r['delta_original_minus_stress']:+.6f} | "
            f"CI=[{r['bootstrap95_lo']:+.6f},"
            f"{r['bootstrap95_hi']:+.6f}] | "
            f"W/T/L={r['original_wins']}/{r['ties']}/{r['stress_wins']} | "
            f"p={r['p_value']:.6g}"
        )

    print()
    print("[CF validity]")
    print(
        f"Original={binary_cf['original_rate']:.6f} | "
        f"Stress={binary_cf['stress_rate']:.6f} | "
        f"Delta={binary_cf['delta_original_minus_stress']:+.6f} | "
        f"discordant={binary_cf['discordant']} | "
        f"p={binary_cf['p_value']:.6g}"
    )

    print()
    print("[Stratified by original DT/BN hypothesis relation]")
    for r in strata:
        print(
            f"{r['stratum']:20s} | "
            f"N={r['N']} | "
            f"change={r['selected_rule_change_rate']:.6f} | "
            f"OrigFid={r['mean_original_fidelity']:.6f} | "
            f"StressFid={r['mean_stress_fidelity']:.6f} | "
            f"Delta={r['mean_delta_original_minus_stress_fidelity']:+.6f} | "
            f"OrigIgn={r['mean_original_ignorance']:.6f} | "
            f"StressIgn={r['mean_stress_ignorance']:.6f}"
        )

    print()
    print(f"Output: {OUT}")
    print("DT' treated as independent evidence: False")
    print("Protocol retuning performed: False")
    print("NEXT: process / temporal validity experiment.")


if __name__ == "__main__":
    main()
