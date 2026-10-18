#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4F-4:
Frozen hierarchical Dempster-Shafer fusion on the untouched confirmation set.

This script is a confirmation-set adaptation of the validated Phase 4C-v2
hierarchical D-S implementation. It MUST NOT tune or modify the protocol.

Frozen protocol
---------------
lambda = 5

Reliability:
    rho_r = n_r / (n_r + lambda)
    R(r)  = rho_r * Fid(r)

BPA:
    m(H)     = R(r)
    m(Theta) = 1 - R(r)

Grouping:
Stage 1:
    case_id × model × prefix × surrogate × predicted_class × feature_signature
Stage 2:
    case_id × model × prefix × predicted_class × feature_signature
Stage 3:
    case_id × prefix × predicted_class × feature_signature

Fusion order:
    Stage 1: candidate_id
    Stage 2: LORE -> DT -> BN
    Stage 3: RF -> XGBoost -> MLP -> LSTM

Conflict:
    if 1-K <= 1e-12:
        map pair to total ignorance;
        representative fallback carries R=0

Representative rule:
    highest normalized proper focal mass;
    tie -> more specific -> lexical canonical rule

Re-evaluation:
    recompute n_r and Fid(r) on the frozen held-out confirmation TEST
    fusion-evaluation cache before propagating to the next hierarchy level.

No protocol retuning is permitted.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics


DEFAULT_ROOT = Path(__file__).resolve().parents[1]

CONFIRMATION_ROOT = (
    DEFAULT_ROOT / "results/phase4/phase4f_confirmation"
)
BANK_DIR = CONFIRMATION_ROOT / "confirmation_multirule_bank"
DEFAULT_RULES = BANK_DIR / "canonical_rules_confirmation.jsonl"
DEFAULT_VALIDATION = BANK_DIR / "validation_phase4f_confirmation_multirule.json"
DEFAULT_CACHE_DIR = BANK_DIR / "fusion_eval_cache" / "targets"
DEFAULT_CACHE_MANIFEST = BANK_DIR / "fusion_eval_cache_manifest.json"

DEFAULT_AUDIT = (
    CONFIRMATION_ROOT
    / "confirmation_target_local_group_audit"
    / "confirmation_target_local_group_audit.json"
)

DEFAULT_SELECTION = CONFIRMATION_ROOT / "confirmation_case_selection.json"

DEFAULT_ADAPTIVE_CONFIG = (
    DEFAULT_ROOT
    / "results/phase4/phase4e_selective_stage3"
    / "selective_stage3_frozen_config.json"
)

DEFAULT_FROZEN_CONFIG = (
    DEFAULT_ROOT / "configs/phase4c_ds_frozen_config.json"
)

DEFAULT_OUT = (
    CONFIRMATION_ROOT / "confirmation_frozen_ds_fusion"
)

FROZEN_LAMBDA = 5.0
FROZEN_SELECTIVE_THRESHOLD = 0.50
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


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


def mean(values):
    vals = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]
    return float(sum(vals) / len(vals)) if vals else None


def median(values):
    vals = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]
    return float(statistics.median(vals)) if vals else None


def q(values, p):
    vals = sorted(
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    )
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


def canonical_values(values):
    return tuple(sorted(str(v) for v in values))


def conditions_to_hypothesis(conditions):
    """
    Symbolic categorical rule hypothesis:
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
        require(
            pos not in by_pos,
            f"Duplicate position {pos} in one rule",
        )
        by_pos[pos] = vals

    return tuple(
        (p, by_pos[p])
        for p in sorted(by_pos)
    )


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


def initial_mass(hyp, reliability):
    r = min(
        1.0,
        max(0.0, float(reliability)),
    )
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
        return {
            THETA: 1.0
        }, float(conflict), True

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
    require(
        len(a) == len(b),
        "Hamming length mismatch",
    )
    return sum(
        x != y
        for x, y in zip(a, b)
    )


def load_cache_index(cache_dir: Path):
    require(
        cache_dir.is_dir(),
        f"Cache target dir missing: {cache_dir}",
    )

    index = {}

    for path in sorted(
        cache_dir.glob("*.json.gz")
    ):
        payload = read_gzip_json(path)
        key = (
            payload["case_id"],
            payload["model"],
            int(payload["prefix"]),
        )

        require(
            key not in index,
            f"Duplicate cache target: {key}",
        )
        index[key] = payload

    return index


def evaluate_hypothesis(
    hyp,
    predicted_class,
    contexts,
):
    """
    Re-evaluate one selected fused rule on the frozen held-out
    confirmation TEST cache.

    `contexts` may contain one or multiple black-box targets.
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

            if (
                int(labels[int(i)])
                == int(predicted_class)
            ):
                total_correct += 1
                m_correct += 1

                ed = hamming(factual, seq)

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
                "cf_valid": bool(
                    m_correct > 0
                ),
                "min_witness_edit": m_min_edit,
            }
        )

    fidelity = (
        float(
            total_correct / total_support
        )
        if total_support > 0
        else None
    )

    coverage = (
        float(
            total_support / total_eval
        )
        if total_eval > 0
        else 0.0
    )

    model_validity_rate = (
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
        "Fid_r": fidelity,
        "coverage_eval": coverage,
        "correct_count": int(total_correct),
        "cf_valid": bool(
            total_correct > 0
        ),
        "model_cf_validity_rate": (
            model_validity_rate
        ),
        "all_models_cf_valid": bool(
            per_model
            and all(
                x["cf_valid"]
                for x in per_model
            )
        ),
        "min_witness_edit": min_edit,
        "per_model": per_model,
    }


def reliability_from_metrics(
    metrics,
    lam,
):
    n = int(metrics["n_r"])
    fid = metrics["Fid_r"]

    if n <= 0 or fid is None:
        return 0.0

    rho = float(
        n / (n + float(lam))
    )
    return float(
        rho * float(fid)
    )


def source_rule(rule, lam):
    n = int(rule["n_r"])
    fid = float(rule["Fid_r"])
    rho = n / (n + float(lam))
    rel = rho * fid

    hyp = conditions_to_hypothesis(
        rule["conditions"]
    )

    return {
        "hypothesis": hyp,
        "predicted_class": int(
            rule["predicted_class"]
        ),
        "n_r": n,
        "Fid_r": fid,
        "coverage_eval": float(
            rule["coverage_eval"]
        ),
        "rho_r": float(rho),
        "R_r": float(rel),
        "source_count": 1,
        "source_models": {
            rule["model"]
        },
        "source_surrogates": {
            rule["surrogate"]
        },
        "source_candidate_ids": [
            rule["candidate_id"]
        ],
        "last_selected_mass": float(rel),
        "last_ignorance_mass": float(
            1.0 - rel
        ),
        "last_K": None,
        "total_conflict_fallback_count": 0,
        "fusion_steps": 0,
        "min_witness_edit": rule.get(
            "witness_edit_count"
        ),
        "cf_valid": True,
        "model_cf_validity_rate": 1.0,
        "all_models_cf_valid": True,
    }


def evidence_mass(ev):
    return initial_mass(
        ev["hypothesis"],
        ev["R_r"],
    )


def pairwise_rule_fusion(
    left,
    right,
    contexts,
    lam,
    stage,
    trace_meta,
):
    require(
        int(left["predicted_class"])
        == int(right["predicted_class"]),
        "Attempted to fuse different predicted classes",
    )

    require(
        feature_signature(
            left["hypothesis"]
        )
        == feature_signature(
            right["hypothesis"]
        ),
        "Attempted to fuse different feature signatures",
    )

    mass, K, fallback = combine_mass(
        evidence_mass(left),
        evidence_mass(right),
    )

    (
        selected_hyp,
        selected_mass,
    ) = select_interpretable_hypothesis(
        mass
    )

    if selected_hyp is None:
        # Frozen total-conflict policy:
        # preserve stronger rule as transparent representation,
        # but output R=0.
        candidates = [left, right]

        candidates.sort(
            key=lambda ev: (
                -float(ev["R_r"]),
                hypothesis_specificity(
                    ev["hypothesis"]
                ),
                hyp_key(
                    ev["hypothesis"]
                ),
            )
        )

        selected_hyp = (
            candidates[0]["hypothesis"]
        )
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

        out_rel = reliability_from_metrics(
            eval_metrics,
            lam,
        )

        ignorance = float(
            mass.get(
                THETA,
                0.0,
            )
        )

    out = {
        "hypothesis": selected_hyp,
        "predicted_class": int(
            left["predicted_class"]
        ),
        "n_r": int(
            eval_metrics["n_r"]
        ),
        "Fid_r": (
            eval_metrics["Fid_r"]
        ),
        "coverage_eval": float(
            eval_metrics["coverage_eval"]
        ),
        "rho_r": (
            float(
                eval_metrics["n_r"]
                / (
                    eval_metrics["n_r"]
                    + float(lam)
                )
            )
            if eval_metrics["n_r"] > 0
            else 0.0
        ),
        "R_r": float(out_rel),
        "source_count": int(
            left["source_count"]
            + right["source_count"]
        ),
        "source_models": (
            set(left["source_models"])
            | set(right["source_models"])
        ),
        "source_surrogates": (
            set(left["source_surrogates"])
            | set(right["source_surrogates"])
        ),
        "source_candidate_ids": (
            list(
                left[
                    "source_candidate_ids"
                ]
            )
            + list(
                right[
                    "source_candidate_ids"
                ]
            )
        ),
        "last_selected_mass": float(
            selected_mass
        ),
        "last_ignorance_mass": float(
            ignorance
        ),
        "last_K": float(K),
        "total_conflict_fallback_count": int(
            left[
                "total_conflict_fallback_count"
            ]
            + right[
                "total_conflict_fallback_count"
            ]
            + int(fallback)
        ),
        "fusion_steps": int(
            left["fusion_steps"]
            + right["fusion_steps"]
            + 1
        ),
        "min_witness_edit": (
            eval_metrics[
                "min_witness_edit"
            ]
        ),
        "cf_valid": bool(
            eval_metrics["cf_valid"]
        ),
        "model_cf_validity_rate": (
            eval_metrics[
                "model_cf_validity_rate"
            ]
        ),
        "all_models_cf_valid": bool(
            eval_metrics[
                "all_models_cf_valid"
            ]
        ),
    }

    trace = {
        **trace_meta,
        "stage": stage,
        "K": float(K),
        "total_conflict_fallback": bool(
            fallback
        ),
        "left_R": float(
            left["R_r"]
        ),
        "right_R": float(
            right["R_r"]
        ),
        "selected_focal_mass": float(
            selected_mass
        ),
        "ignorance_mass": float(
            ignorance
        ),
        "output_n_r": int(
            out["n_r"]
        ),
        "output_Fid_r": (
            out["Fid_r"]
        ),
        "output_R_r": float(
            out["R_r"]
        ),
        "output_cf_valid": bool(
            out["cf_valid"]
        ),
        "selected_conditions": (
            json.dumps(
                hypothesis_to_conditions(
                    selected_hyp
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        ),
    }

    return out, trace


def group_key_stage1(rule):
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


def ev_to_row(
    ev,
    lam,
    stage,
    key,
    input_count,
):
    case_id = key[0]
    prefix = None
    model = None
    surrogate = None
    predicted_class = None
    signature = None

    if stage == "Stage1":
        (
            _,
            model,
            prefix,
            surrogate,
            predicted_class,
            signature,
        ) = key

    elif stage == "Stage2":
        (
            _,
            model,
            prefix,
            predicted_class,
            signature,
        ) = key

    elif stage == "Stage3":
        (
            _,
            prefix,
            predicted_class,
            signature,
        ) = key

    else:
        raise ValueError(stage)

    return {
        "lambda": float(lam),
        "stage": stage,
        "case_id": case_id,
        "model": model,
        "prefix": int(prefix),
        "surrogate": surrogate,
        "predicted_class": int(
            predicted_class
        ),
        "feature_signature": (
            json.dumps(
                list(signature)
            )
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
            len(
                ev["source_models"]
            )
        ),
        "source_surrogate_count": int(
            len(
                ev[
                    "source_surrogates"
                ]
            )
        ),
        "n_r": int(
            ev["n_r"]
        ),
        "Fid_r": (
            ev["Fid_r"]
        ),
        "coverage_eval": float(
            ev["coverage_eval"]
        ),
        "rho_r": float(
            ev["rho_r"]
        ),
        "R_r": float(
            ev["R_r"]
        ),
        "cf_valid": bool(
            ev["cf_valid"]
        ),
        "model_cf_validity_rate": (
            ev[
                "model_cf_validity_rate"
            ]
        ),
        "all_models_cf_valid": bool(
            ev[
                "all_models_cf_valid"
            ]
        ),
        "min_witness_edit": (
            ev[
                "min_witness_edit"
            ]
        ),
        "fusion_steps": int(
            ev["fusion_steps"]
        ),
        "total_conflict_fallback_count": int(
            ev[
                "total_conflict_fallback_count"
            ]
        ),
        "last_K": (
            ev["last_K"]
        ),
        "last_selected_mass": float(
            ev[
                "last_selected_mass"
            ]
        ),
        "last_ignorance_mass": float(
            ev[
                "last_ignorance_mass"
            ]
        ),
        "selected_conditions": (
            json.dumps(
                hypothesis_to_conditions(
                    ev["hypothesis"]
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        ),
    }


def summarize_stage(
    rows,
    conflict_rows,
    stage,
    lam,
):
    sub = [
        r
        for r in rows
        if (
            r["stage"] == stage
            and float(
                r["lambda"]
            )
            == float(lam)
        )
    ]

    fused = [
        r
        for r in sub
        if r["actually_fused"]
    ]

    cr = [
        r
        for r in conflict_rows
        if (
            r["stage"] == stage
            and float(
                r["lambda"]
            )
            == float(lam)
        )
    ]

    def scope_stats(
        scope,
        name,
    ):
        return {
            f"{name}_count": len(scope),
            f"{name}_mean_fidelity": mean(
                [
                    r["Fid_r"]
                    for r in scope
                ]
            ),
            f"{name}_median_fidelity": median(
                [
                    r["Fid_r"]
                    for r in scope
                ]
            ),
            f"{name}_cf_validity_rate": mean(
                [
                    int(
                        r["cf_valid"]
                    )
                    for r in scope
                ]
            ),
            f"{name}_mean_model_cf_validity_rate": mean(
                [
                    r[
                        "model_cf_validity_rate"
                    ]
                    for r in scope
                ]
            ),
            f"{name}_all_models_cf_valid_rate": mean(
                [
                    int(
                        r[
                            "all_models_cf_valid"
                        ]
                    )
                    for r in scope
                ]
            ),
            f"{name}_mean_support": mean(
                [
                    r["n_r"]
                    for r in scope
                ]
            ),
            f"{name}_mean_coverage": mean(
                [
                    r["coverage_eval"]
                    for r in scope
                ]
            ),
            f"{name}_mean_reliability": mean(
                [
                    r["R_r"]
                    for r in scope
                ]
            ),
            f"{name}_mean_ignorance": mean(
                [
                    r[
                        "last_ignorance_mass"
                    ]
                    for r in scope
                ]
            ),
            f"{name}_mean_min_witness_edit": mean(
                [
                    r[
                        "min_witness_edit"
                    ]
                    for r in scope
                ]
            ),
        }

    out = {
        "lambda": float(lam),
        "stage": stage,
        "group_count": len(sub),
        "fusable_group_count": len(fused),
        "pairwise_fusion_count": len(cr),
        "mean_conflict_K": mean(
            [
                r["K"]
                for r in cr
            ]
        ),
        "median_conflict_K": median(
            [
                r["K"]
                for r in cr
            ]
        ),
        "p90_conflict_K": q(
            [
                r["K"]
                for r in cr
            ],
            0.90,
        ),
        "max_conflict_K": (
            max(
                float(r["K"])
                for r in cr
            )
            if cr
            else None
        ),
        "total_conflict_fallback_count": sum(
            int(
                r[
                    "total_conflict_fallback"
                ]
            )
            for r in cr
        ),
    }

    out.update(
        scope_stats(
            sub,
            "all",
        )
    )

    out.update(
        scope_stats(
            fused,
            "fused",
        )
    )

    return out


def validate_frozen_config(cfg):
    require(
        cfg.get("status") == "frozen",
        "D-S config is not frozen",
    )

    require(
        cfg.get(
            "selection_scope"
        )
        == "validation_only",
        "Frozen config was not selected on validation only",
    )

    require(
        cfg.get(
            "formal_test_seen_before_freeze"
        )
        is False,
        "Formal TEST was seen before protocol freeze",
    )

    require(
        abs(
            float(
                cfg.get("lambda")
            )
            - FROZEN_LAMBDA
        )
        <= 1e-12,
        "Frozen lambda is not 5",
    )

    grouping = cfg.get(
        "grouping",
        {},
    )

    require(
        grouping.get(
            "stage1"
        )
        == (
            "case_id × model × prefix × surrogate × "
            "predicted_class × feature_signature"
        ),
        "Frozen Stage1 grouping mismatch",
    )

    require(
        grouping.get(
            "stage2"
        )
        == (
            "case_id × model × prefix × "
            "predicted_class × feature_signature"
        ),
        "Frozen Stage2 grouping mismatch",
    )

    require(
        grouping.get(
            "stage3"
        )
        == (
            "case_id × prefix × "
            "predicted_class × feature_signature"
        ),
        "Frozen Stage3 grouping mismatch",
    )

    fusion = cfg.get(
        "fusion",
        {},
    )

    require(
        fusion.get(
            "stage1_order"
        )
        == "candidate_id",
        "Frozen Stage1 order mismatch",
    )

    require(
        fusion.get(
            "stage2_order"
        )
        == [
            "LORE",
            "DT",
            "BN",
        ],
        "Frozen Stage2 order mismatch",
    )

    require(
        fusion.get(
            "stage3_order"
        )
        == [
            "RF",
            "XGBoost",
            "MLP",
            "LSTM",
        ],
        "Frozen Stage3 order mismatch",
    )

    require(
        math.isclose(
            float(
                fusion.get(
                    "K_total_conflict_guard"
                )
            ),
            K_EPS,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "Frozen K total-conflict guard mismatch",
    )

    require(
        fusion.get(
            "total_conflict_policy"
        )
        == (
            "map pair to total ignorance; "
            "representative fallback carries R=0"
        ),
        "Frozen total-conflict policy mismatch",
    )

    require(
        fusion.get(
            "representative_rule_policy"
        )
        == (
            "highest normalized proper focal mass; "
            "tie -> more specific -> lexical canonical rule"
        ),
        "Frozen representative-rule policy mismatch",
    )

    require(
        fusion.get(
            "fused_rule_re_evaluation"
        )
        == (
            "recompute n_r and Fid(r) on frozen held-out "
            "validation cache before propagating to next hierarchy level"
        ),
        (
            "Frozen fused-rule re-evaluation policy mismatch. "
            "The literal config wording must remain unchanged even though "
            "the confirmation TEST stage applies that already-frozen policy to "
            "the confirmation TEST held-out cache."
        ),
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--rules",
        default=str(
            DEFAULT_RULES
        ),
    )

    ap.add_argument(
        "--validation",
        default=str(
            DEFAULT_VALIDATION
        ),
    )

    ap.add_argument(
        "--audit",
        default=str(
            DEFAULT_AUDIT
        ),
    )

    ap.add_argument(
        "--cache_dir",
        default=str(
            DEFAULT_CACHE_DIR
        ),
    )

    ap.add_argument(
        "--cache_manifest",
        default=str(
            DEFAULT_CACHE_MANIFEST
        ),
    )

    ap.add_argument(
        "--frozen_config",
        default=str(
            DEFAULT_FROZEN_CONFIG
        ),
    )

    ap.add_argument(
        "--selection",
        default=str(
            DEFAULT_SELECTION
        ),
    )

    ap.add_argument(
        "--adaptive_config",
        default=str(
            DEFAULT_ADAPTIVE_CONFIG
        ),
    )

    ap.add_argument(
        "--out_dir",
        default=str(
            DEFAULT_OUT
        ),
    )

    args = ap.parse_args()

    rules_path = (
        Path(
            args.rules
        )
        .expanduser()
        .resolve()
    )

    validation_path = (
        Path(
            args.validation
        )
        .expanduser()
        .resolve()
    )

    audit_path = (
        Path(
            args.audit
        )
        .expanduser()
        .resolve()
    )

    cache_dir = (
        Path(
            args.cache_dir
        )
        .expanduser()
        .resolve()
    )

    cache_manifest_path = (
        Path(
            args.cache_manifest
        )
        .expanduser()
        .resolve()
    )

    frozen_config_path = (
        Path(
            args.frozen_config
        )
        .expanduser()
        .resolve()
    )

    selection_path = (
        Path(args.selection).expanduser().resolve()
    )

    adaptive_config_path = (
        Path(args.adaptive_config).expanduser().resolve()
    )

    out_dir = (
        Path(
            args.out_dir
        )
        .expanduser()
        .resolve()
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for p in (
        rules_path,
        validation_path,
        audit_path,
        cache_manifest_path,
        frozen_config_path,
        selection_path,
        adaptive_config_path,
    ):
        require(
            p.is_file(),
            f"Required input missing: {p}",
        )

    require(
        cache_dir.is_dir(),
        f"Required cache directory missing: {cache_dir}",
    )

    validation = read_json(
        validation_path
    )
    audit = read_json(
        audit_path
    )
    cache_manifest = read_json(
        cache_manifest_path
    )
    frozen = read_json(
        frozen_config_path
    )
    selection = read_json(selection_path)
    adaptive = read_json(adaptive_config_path)

    # --------------------------------------------------------------
    # Frozen protocol / confirmation gates.
    # --------------------------------------------------------------
    validate_frozen_config(
        frozen
    )

    require(
        selection.get("schema_version")
        == "etce-phase4f-confirmation-selection-v1"
        and selection.get("status") == "frozen",
        "Phase 4F confirmation selection is not frozen",
    )
    require(
        int(selection.get("selected_case_count", -1)) == 200
        and int(selection.get("confirmation_target_count", -1)) == 3200,
        "Phase 4F confirmation selection size mismatch",
    )
    require(
        int(selection.get("global_sha256_rank_start", -1)) == 101
        and int(selection.get("global_sha256_rank_end", -1)) == 300
        and int(selection.get("overlap_with_original_phase4d", -1)) == 0,
        "Phase 4F confirmation rank/overlap protocol mismatch",
    )
    require(
        selection.get("selection_uses_phase4d_or_phase4e_performance")
        is False,
        "Observed Phase-4D/4E performance influenced confirmation selection",
    )
    require(
        selection.get("frozen_ds_config_sha256")
        == sha256_file(frozen_config_path),
        "Frozen D-S config changed after confirmation selection",
    )
    require(
        selection.get("adaptive_config_sha256")
        == sha256_file(adaptive_config_path),
        "Selective Stage-3 config changed after confirmation selection",
    )
    require(
        adaptive.get("status") == "validation_frozen_formal_exploratory"
        and abs(
            float(
                adaptive.get("gate", {}).get(
                    "pre_mean_fidelity_threshold"
                )
            )
            - FROZEN_SELECTIVE_THRESHOLD
        )
        <= 1e-12,
        "Selective Stage-3 protocol is not frozen at 0.50",
    )

    require(
        validation.get(
            "status"
        )
        == "passed",
        "Phase 4F-2 confirmation rule-bank validation is not PASSED",
    )
    require(
        int(validation.get("confirmation_target_count", -1)) == 3200
        and int(validation.get("selected_confirmation_case_count", -1))
        == 200
        and int(validation.get("eligible_counterfactual_candidate_count", -1))
        == 11294,
        "Phase 4F-2 validation inventory mismatch",
    )
    require(
        abs(float(validation.get("frozen_lambda", -1)) - FROZEN_LAMBDA)
        <= 1e-12
        and abs(
            float(validation.get("frozen_selective_stage3_threshold", -1))
            - FROZEN_SELECTIVE_THRESHOLD
        )
        <= 1e-12,
        "Phase 4F-2 validation did not preserve frozen parameters",
    )

    require(
        audit.get(
            "status"
        )
        == "passed",
        "Phase 4F-3 grouping audit is not PASSED",
    )

    require(
        audit.get(
            "stage1",
            {},
        ).get(
            "status"
        )
        == "NON-DEGENERATE",
        "Phase 4F-3 did not establish Stage1 non-degeneracy",
    )
    require(
        int(audit.get("frozen_selected_case_count", -1)) == 200
        and int(audit.get("confirmation_target_count", -1)) == 3200
        and int(audit.get("target_surrogate_group_count", -1)) == 9600
        and int(audit.get("canonical_rule_count", -1)) == 18273
        and int(audit.get("eligible_cf_candidate_count", -1)) == 11294,
        "Phase 4F-3 audit inventory mismatch",
    )
    require(
        abs(float(audit.get("frozen_lambda", -1)) - FROZEN_LAMBDA)
        <= 1e-12
        and abs(
            float(audit.get("frozen_selective_stage3_threshold", -1))
            - FROZEN_SELECTIVE_THRESHOLD
        )
        <= 1e-12
        and audit.get("protocol_retuning_performed") is False
        and audit.get("ds_fusion_performed") is False
        and audit.get("selective_gate_applied") is False,
        "Phase 4F-3 audit indicates protocol drift or premature fusion",
    )

    require(
        cache_manifest.get(
            "status"
        )
        == "built",
        "Confirmation fusion cache manifest is not built",
    )

    require(
        cache_manifest.get("schema_version")
        == "phase4f-confirmation-fusion-eval-target-v1",
        "Confirmation fusion cache schema mismatch",
    )

    require(
        cache_manifest.get(
            "same_targets_as_confirmation_rule_bank"
        )
        is True,
        "Confirmation fusion cache target identity not confirmed",
    )

    require(
        int(
            cache_manifest.get(
                "target_count",
                -1,
            )
        )
        == 3200,
        "Confirmation TEST cache target count is not 3200",
    )

    require(
        cache_manifest.get(
            "true_labels_used"
        )
        is False,
        "Confirmation fusion cache used true labels",
    )

    # --------------------------------------------------------------
    # Load eligible confirmation TEST CF rules.
    # --------------------------------------------------------------
    rules = read_jsonl(
        rules_path
    )

    require(
        len(rules)
        == int(
            audit[
                "canonical_rule_count"
            ]
        ),
        "Canonical rule count disagrees with Phase 4F-3 audit",
    )

    cf = [
        r
        for r in rules
        if (
            r.get(
                "rule_type"
            )
            == "counterfactual"
            and bool(
                r.get(
                    "eligible_for_reliability"
                )
            )
        )
    ]

    require(
        len(cf)
        == int(
            audit[
                "eligible_cf_candidate_count"
            ]
        ),
        "Eligible CF count disagrees with Phase 4F-3 audit",
    )

    require(
        len(cf) == 11294,
        f"Expected 11294 eligible confirmation TEST CF candidates, found {len(cf)}",
    )

    # --------------------------------------------------------------
    # Load confirmation TEST held-out cache.
    # --------------------------------------------------------------
    cache = load_cache_index(
        cache_dir
    )

    require(
        len(cache) == 3200,
        f"Confirmation fusion cache target count mismatch: {len(cache)}",
    )

    require(
        all(
            p.get("schema_version")
            == "phase4f-confirmation-fusion-eval-target-v1"
            for p in cache.values()
        ),
        "A confirmation fusion cache payload has the wrong schema",
    )

    require(
        all(
            p.get(
                "confirmation_target"
            )
            is True
            for p in cache.values()
        ),
        "A fusion cache payload is not marked confirmation_target=True",
    )

    require(
        all(
            p.get(
                "true_label_used"
            )
            is False
            for p in cache.values()
        ),
        "A fusion cache payload indicates true_label_used=True",
    )

    lam = FROZEN_LAMBDA

    # --------------------------------------------------------------
    # Build Stage-1 source grouping.
    # --------------------------------------------------------------
    s1_source_groups = defaultdict(
        list
    )

    for r in cf:
        s1_source_groups[
            group_key_stage1(r)
        ].append(r)

    expected_s1 = int(
        audit[
            "stage1"
        ][
            "group_count"
        ]
    )

    expected_s1_fusable = int(
        audit[
            "stage1"
        ][
            "fusable_group_count_ge2_rules"
        ]
    )

    require(
        len(
            s1_source_groups
        )
        == expected_s1,
        (
            "Stage1 group count mismatch: "
            f"{len(s1_source_groups)} != {expected_s1}"
        ),
    )

    require(
        sum(
            len(v) >= 2
            for v in s1_source_groups.values()
        )
        == expected_s1_fusable,
        "Stage1 fusable group count mismatch",
    )

    stage_rows = []
    conflict_rows = []

    print(
        "=" * 116
    )
    print(
        "ETCE PHASE 4F-4 FROZEN D-S CONFIRMATION FUSION"
    )
    print(
        "=" * 116
    )
    print(
        f"Frozen lambda: {lam:g}"
    )
    print(
        f"Frozen selective Stage-3 threshold: "
        f"{FROZEN_SELECTIVE_THRESHOLD:.2f} (not applied here)"
    )
    print(
        f"Eligible CF candidates: {len(cf)}"
    )
    print(
        f"Confirmation TEST cache targets: {len(cache)}"
    )
    print(
        f"Stage1 groups: {expected_s1} | fusable: {expected_s1_fusable}"
    )
    print(
        "Protocol source: configs/phase4c_ds_frozen_config.json"
    )
    print(
        "Protocol retuning: False"
    )
    print()

    # ==============================================================
    # Stage 1
    # ==============================================================
    print(
        "[Stage 1] within target-local surrogate ..."
    )

    s1_outputs = {}

    for key in sorted(
        s1_source_groups,
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
        source_rules = sorted(
            s1_source_groups[key],
            key=lambda r: str(
                r["candidate_id"]
            ),
        )

        (
            case_id,
            model,
            prefix,
            surrogate,
            pred,
            sig,
        ) = key

        ctx = [
            cache[
                (
                    case_id,
                    model,
                    int(prefix),
                )
            ]
        ]

        current = source_rule(
            source_rules[0],
            lam,
        )

        for step, r in enumerate(
            source_rules[1:],
            1,
        ):
            right = source_rule(
                r,
                lam,
            )

            current, trace = (
                pairwise_rule_fusion(
                    current,
                    right,
                    ctx,
                    lam,
                    "Stage1",
                    {
                        "lambda": float(
                            lam
                        ),
                        "case_id": case_id,
                        "model": model,
                        "prefix": int(
                            prefix
                        ),
                        "surrogate": surrogate,
                        "predicted_class": int(
                            pred
                        ),
                        "feature_signature": (
                            json.dumps(
                                list(sig)
                            )
                        ),
                        "step": int(
                            step
                        ),
                    },
                )
            )

            conflict_rows.append(
                trace
            )

        s1_outputs[
            key
        ] = current

        stage_rows.append(
            ev_to_row(
                current,
                lam,
                "Stage1",
                key,
                len(
                    source_rules
                ),
            )
        )

    print(
        f"[Stage 1] complete | groups={len(s1_outputs)}"
    )

    # ==============================================================
    # Stage 2
    # ==============================================================
    s2_groups = defaultdict(
        list
    )

    for key, ev in s1_outputs.items():
        (
            case_id,
            model,
            prefix,
            surrogate,
            pred,
            sig,
        ) = key

        s2_key = (
            case_id,
            model,
            int(prefix),
            int(pred),
            sig,
        )

        s2_groups[
            s2_key
        ].append(
            (
                surrogate,
                ev,
            )
        )

    require(
        len(
            s2_groups
        )
        == int(
            audit[
                "stage2"
            ][
                "compatible_sets"
            ]
        ),
        "Stage2 compatible-set count mismatch",
    )

    require(
        sum(
            len(
                {
                    s
                    for s, _ in items
                }
            )
            >= 2
            for items in s2_groups.values()
        )
        == int(
            audit[
                "stage2"
            ][
                "sets_with_ge2_surrogates"
            ]
        ),
        "Stage2 fusable-set count mismatch",
    )

    print(
        "[Stage 2] across LORE -> DT -> BN "
        f"| groups={len(s2_groups)} ..."
    )

    s2_outputs = {}

    for key in sorted(
        s2_groups,
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
            s2_groups[key],
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

        for step, (
            surrogate,
            right,
        ) in enumerate(
            items[1:],
            1,
        ):
            current, trace = (
                pairwise_rule_fusion(
                    current,
                    right,
                    ctx,
                    lam,
                    "Stage2",
                    {
                        "lambda": float(
                            lam
                        ),
                        "case_id": case_id,
                        "model": model,
                        "prefix": int(
                            prefix
                        ),
                        "surrogate": (
                            f"cross-surrogate:{surrogate}"
                        ),
                        "predicted_class": int(
                            pred
                        ),
                        "feature_signature": (
                            json.dumps(
                                list(sig)
                            )
                        ),
                        "step": int(
                            step
                        ),
                    },
                )
            )

            conflict_rows.append(
                trace
            )

        s2_outputs[
            key
        ] = current

        stage_rows.append(
            ev_to_row(
                current,
                lam,
                "Stage2",
                key,
                len(
                    items
                ),
            )
        )

    print(
        f"[Stage 2] complete | groups={len(s2_outputs)}"
    )

    # ==============================================================
    # Stage 3
    # ==============================================================
    s3_groups = defaultdict(
        list
    )

    for key, ev in s2_outputs.items():
        (
            case_id,
            model,
            prefix,
            pred,
            sig,
        ) = key

        s3_key = (
            case_id,
            int(prefix),
            int(pred),
            sig,
        )

        s3_groups[
            s3_key
        ].append(
            (
                model,
                ev,
            )
        )

    require(
        len(
            s3_groups
        )
        == int(
            audit[
                "stage3"
            ][
                "compatible_sets"
            ]
        ),
        "Stage3 compatible-set count mismatch",
    )

    require(
        sum(
            len(
                {
                    m
                    for m, _ in items
                }
            )
            >= 2
            for items in s3_groups.values()
        )
        == int(
            audit[
                "stage3"
            ][
                "sets_with_ge2_blackboxes"
            ]
        ),
        "Stage3 fusable-set count mismatch",
    )

    print(
        "[Stage 3] across RF -> XGBoost -> MLP -> LSTM "
        f"| groups={len(s3_groups)} ..."
    )

    s3_outputs = {}

    for key in sorted(
        s3_groups,
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
            s3_groups[key],
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

        for step, (
            model,
            right,
        ) in enumerate(
            items[1:],
            1,
        ):
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

            current, trace = (
                pairwise_rule_fusion(
                    current,
                    right,
                    contexts,
                    lam,
                    "Stage3",
                    {
                        "lambda": float(
                            lam
                        ),
                        "case_id": case_id,
                        "model": (
                            f"cross-model:{model}"
                        ),
                        "prefix": int(
                            prefix
                        ),
                        "surrogate": None,
                        "predicted_class": int(
                            pred
                        ),
                        "feature_signature": (
                            json.dumps(
                                list(sig)
                            )
                        ),
                        "step": int(
                            step
                        ),
                    },
                )
            )

            conflict_rows.append(
                trace
            )

        s3_outputs[
            key
        ] = current

        stage_rows.append(
            ev_to_row(
                current,
                lam,
                "Stage3",
                key,
                len(
                    items
                ),
            )
        )

    print(
        f"[Stage 3] complete | groups={len(s3_outputs)}"
    )
    print()

    # ==============================================================
    # Confirmation summaries
    # ==============================================================
    summary_rows = [
        summarize_stage(
            stage_rows,
            conflict_rows,
            stage,
            lam,
        )
        for stage in (
            "Stage1",
            "Stage2",
            "Stage3",
        )
    ]

    stage3_summary = next(
        r
        for r in summary_rows
        if r["stage"] == "Stage3"
    )

    total_fallbacks = sum(
        int(
            r[
                "total_conflict_fallback"
            ]
        )
        for r in conflict_rows
    )

    # ==============================================================
    # Persist confirmation outputs
    # ==============================================================
    write_csv(
        out_dir
        / "phase4f_confirmation_frozen_ds_stage_outputs.csv",
        stage_rows,
    )

    write_csv(
        out_dir
        / "phase4f_confirmation_frozen_ds_conflict_trace.csv",
        conflict_rows,
    )

    write_csv(
        out_dir
        / "phase4f_confirmation_frozen_ds_summary.csv",
        summary_rows,
    )

    # Stage-3 rules only, convenient for later baseline/ablation comparison.
    stage3_rows = [
        r
        for r in stage_rows
        if r["stage"] == "Stage3"
    ]

    write_csv(
        out_dir
        / "phase4f_confirmation_stage3_fused_rules.csv",
        stage3_rows,
    )

    result = {
        "scope": (
            "ETCE Phase 4F-4 frozen hierarchical "
            "D-S confirmation TEST fusion"
        ),
        "status": "passed",
        "confirmation_test": True,
        "lambda": FROZEN_LAMBDA,
        "frozen_selective_stage3_threshold": FROZEN_SELECTIVE_THRESHOLD,
        "lambda_selection_performed": False,
        "selective_threshold_selection_performed": False,
        "selective_gate_applied": False,
        "protocol_retuning_performed": False,
        "protocol_source": (
            str(
                frozen_config_path
            )
        ),
        "frozen_protocol_verified": True,
        "phase4f_2_validation_passed": True,
        "phase4f_3_group_audit_passed": True,
        "cache_target_count": len(
            cache
        ),
        "eligible_cf_candidate_count": len(
            cf
        ),
        "group_counts": {
            "stage1": int(
                audit[
                    "stage1"
                ][
                    "group_count"
                ]
            ),
            "stage1_fusable": int(
                audit[
                    "stage1"
                ][
                    "fusable_group_count_ge2_rules"
                ]
            ),
            "stage2": int(
                audit[
                    "stage2"
                ][
                    "compatible_sets"
                ]
            ),
            "stage2_fusable": int(
                audit[
                    "stage2"
                ][
                    "sets_with_ge2_surrogates"
                ]
            ),
            "stage3": int(
                audit[
                    "stage3"
                ][
                    "compatible_sets"
                ]
            ),
            "stage3_fusable": int(
                audit[
                    "stage3"
                ][
                    "sets_with_ge2_blackboxes"
                ]
            ),
        },
        "summary": {
            r["stage"]: r
            for r in summary_rows
        },
        "total_conflict_fallback_count": int(
            total_fallbacks
        ),
        "next_step": (
            "Independent validation of frozen confirmation D-S outputs, "
            "then Original -> Stage1 -> Stage2 -> Stage3 comparison "
            "and predeclared fusion baselines/ablations."
        ),
    }

    result_path = (
        out_dir
        / "confirmation_frozen_ds_fusion.json"
    )

    write_json(
        result_path,
        result,
    )

    # Human-readable report.
    def fv(x):
        if x is None:
            return "NA"
        return f"{float(x):.4f}"

    lines = [
        "# ETCE Phase 4F-4 Frozen D-S Confirmation Fusion",
        "",
        "- Status: **PASSED**",
        "- Frozen lambda: **5**",
        "- Frozen selective Stage-3 threshold: **0.50** (not applied here)",
        "- Lambda selection on confirmation TEST: **False**",
        "- Selective-threshold selection on confirmation TEST: **False**",
        "- Selective Stage-3 gate applied in this script: **False**",
        "- Protocol retuning on confirmation TEST: **False**",
        f"- Total-conflict fallback count: **{total_fallbacks}**",
        "",
        "## Stage summary",
        "",
        "| Stage | groups | genuinely fused | Fidelity | CF-valid | model-valid | all-model-valid | mean R | mean ignorance | mean K | p90 K | max K |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in summary_rows:
        lines.append(
            f"| {r['stage']} | "
            f"{r['group_count']} | "
            f"{r['fused_count']} | "
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
            "## Stage-3 confirmation result",
            "",
            f"- Genuinely fused groups: {stage3_summary['fused_count']}",
            f"- Mean Rule Fidelity: {fv(stage3_summary['fused_mean_fidelity'])}",
            f"- CF validity: {fv(stage3_summary['fused_cf_validity_rate'])}",
            f"- Mean model CF validity: {fv(stage3_summary['fused_mean_model_cf_validity_rate'])}",
            f"- All-model CF validity: {fv(stage3_summary['fused_all_models_cf_valid_rate'])}",
            f"- Mean reliability: {fv(stage3_summary['fused_mean_reliability'])}",
            f"- Mean ignorance: {fv(stage3_summary['fused_mean_ignorance'])}",
            f"- Mean conflict K: {fv(stage3_summary['mean_conflict_K'])}",
            f"- P90 conflict K: {fv(stage3_summary['p90_conflict_K'])}",
            f"- Max conflict K: {fv(stage3_summary['max_conflict_K'])}",
            "",
            "The frozen validation-selected D-S protocol was applied without confirmation-side retuning.",
        ]
    )

    (
        out_dir
        / "PHASE4F_FROZEN_DS_CONFIRMATION_REPORT.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # ==============================================================
    # Terminal summary
    # ==============================================================
    print(
        "=" * 116
    )
    print(
        "ETCE PHASE 4F-4 FROZEN D-S CONFIRMATION FUSION PASSED"
    )
    print(
        "=" * 116
    )

    for r in summary_rows:
        print(
            f"{r['stage']} | "
            f"groups={r['group_count']} | "
            f"fused={r['fused_count']} | "
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
        f"Total-conflict fallback count: {total_fallbacks}"
    )
    print(
        "Frozen lambda: 5"
    )
    print(
        "Frozen selective Stage-3 threshold: 0.50 (not applied here)"
    )
    print(
        "Lambda selection performed: False"
    )
    print(
        "Selective threshold selection performed: False"
    )
    print(
        "Selective Stage-3 gate applied: False"
    )
    print(
        "Protocol retuning performed: False"
    )
    print(
        f"Output: {out_dir}"
    )
    print(
        f"Result JSON: {result_path}"
    )
    print()
    print(
        "NEXT: independently validate the confirmation D-S outputs before applying the pre-frozen selective Stage-3 rule."
    )


if __name__ == "__main__":
    main()
