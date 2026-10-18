#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4F-9 / Script 56
Independent validation of validation-frozen fusion baselines on the untouched
confirmation TEST.

Uses the already frozen validation-only baseline protocol from Script 41.
No baseline formula, source weight, source choice, tie-break, or epsilon may
be changed in response to confirmation TEST results.

Baselines:
- Best Single Evidence
- Validation-best Source
- Uniform Vote
- Validation-weighted Vote
- Fuzzy Probabilistic OR
- Bayesian Categorical-noise Pooling
- D-S ETCE (reference only; loaded from independently validated Phase 4F-4/5)
- Selective Stage-3 ETCE (reference only; loaded from independently validated
  Phase 4F-6/7)

Probability Averaging is intentionally omitted because Script 41 established
that the current canonical rule bank contains no comparable surrogate class
probability field.

All non-D-S baselines:
- use the same Stage1/2/3 compatible groups;
- select only from exact hypotheses present in their inputs;
- use frozen deterministic tie-breaking;
- re-evaluate the selected rule on the same confirmation held-out cache;
- propagate the re-evaluated evidence to the next stage.

Run:
    cd D:/ETCE
    python -u ./scripts/56_phase4f_validate_confirmation_frozen_baselines.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]

CONFIRMATION = ROOT / "results/phase4/phase4f_confirmation"
BANK = CONFIRMATION / "confirmation_multirule_bank"
AUDIT_DIR = CONFIRMATION / "confirmation_target_local_group_audit"
DS_DIR = CONFIRMATION / "confirmation_frozen_ds_fusion"
SELECTIVE_DIR = CONFIRMATION / "confirmation_selective_stage3"

RULES = BANK / "canonical_rules_confirmation.jsonl"
CACHE_DIR = BANK / "fusion_eval_cache" / "targets"
CACHE_MANIFEST = BANK / "fusion_eval_cache_manifest.json"
BANK_VALIDATION = BANK / "validation_phase4f_confirmation_multirule.json"

AUDIT = AUDIT_DIR / "confirmation_target_local_group_audit.json"
BASELINE_CONFIG = ROOT / "configs/phase4c_fusion_baselines_frozen_config.json"
DS_CONFIG = ROOT / "configs/phase4c_ds_frozen_config.json"
ADAPTIVE_CONFIG = (
    ROOT
    / "results/phase4/phase4e_selective_stage3"
    / "selective_stage3_frozen_config.json"
)
SELECTION = CONFIRMATION / "confirmation_case_selection.json"
DS_RESULT = DS_DIR / "confirmation_frozen_ds_fusion.json"
DS_VALIDATION = DS_DIR / "validation_phase4f_confirmation_frozen_ds_fusion.json"
SELECTIVE_RESULT = SELECTIVE_DIR / "confirmation_selective_stage3.json"
SELECTIVE_VALIDATION = (
    SELECTIVE_DIR / "validation_phase4f_confirmation_selective_stage3.json"
)

OUT = CONFIRMATION / "confirmation_fusion_baselines"

SAVED_STAGE = OUT / "confirmation_fusion_baseline_stage_outputs.csv"
SAVED_SUMMARY = OUT / "confirmation_fusion_baseline_stage_summary.csv"
SAVED_COMPARISON = OUT / "confirmation_fusion_strategy_comparison_stage3.csv"
SAVED_PREFIX = OUT / "confirmation_fusion_baselines_stage3_by_prefix.csv"
SAVED_RESULT = OUT / "confirmation_fusion_strategy_comparison.json"
OUT_VALIDATION = OUT / "validation_phase4f_confirmation_frozen_baselines.json"

LAMBDA = 5.0
SURROGATE_ORDER = {"LORE": 0, "DT": 1, "BN": 2}
MODEL_ORDER = {"RF": 0, "XGBoost": 1, "MLP": 2, "LSTM": 3}
EPS = 1e-12

BASELINES = [
    "best_single_evidence",
    "validation_best_source",
    "uniform_vote",
    "validation_weighted_vote",
    "fuzzy_probabilistic_or",
    "bayesian_categorical_noise",
]


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
                    f"Invalid JSONL line {line_no}: {path}"
                ) from exc
    return rows


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


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


def conditions_to_hypothesis(conditions):
    by_pos = {}
    for c in conditions:
        require(c.get("operator") == "in", "Unsupported operator")
        p = int(c["position"])
        vals = tuple(sorted(str(v) for v in c["values"]))
        require(vals, f"Empty values at position {p}")
        require(p not in by_pos, f"Duplicate position {p}")
        by_pos[p] = vals
    return tuple((p, by_pos[p]) for p in sorted(by_pos))


def hypothesis_to_conditions(hyp):
    return [
        {
            "position": int(p),
            "operator": "in",
            "values": list(vals),
        }
        for p, vals in hyp
    ]


def feature_signature(hyp):
    return tuple(int(p) for p, _ in hyp)


def hyp_key(hyp):
    return json.dumps(
        hypothesis_to_conditions(hyp),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def specificity(hyp):
    return sum(len(vals) for _, vals in hyp)


def applies(seq, hyp):
    for p, vals in hyp:
        idx = int(p) - 1
        if idx < 0 or idx >= len(seq):
            return False
        if str(seq[idx]) not in set(vals):
            return False
    return True


def hamming(a, b):
    require(len(a) == len(b), "Hamming length mismatch")
    return sum(x != y for x, y in zip(a, b))


def reliability(n, fid):
    n = int(n)
    if n <= 0 or fid is None:
        return 0.0
    return float((n / (n + LAMBDA)) * float(fid))


def evaluate(hyp, predicted_class, contexts):
    total_eval = 0
    total_support = 0
    total_correct = 0
    global_min_edit = None
    per_model = []

    for payload in contexts:
        seqs = payload["pool_sequences"]
        labels = payload["pool_labels"]
        eval_idx = payload["eval_indices"]
        factual = payload["factual_sequence"]

        support = 0
        correct = 0
        min_edit = None

        for raw_i in eval_idx:
            i = int(raw_i)
            seq = seqs[i]
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
            "model": payload["model"],
            "support": support,
            "correct": correct,
            "cf_valid": bool(correct > 0),
            "fidelity": (
                float(correct / support)
                if support > 0 else None
            ),
        })

    fid = (
        float(total_correct / total_support)
        if total_support > 0 else None
    )

    coverage = (
        float(total_support / total_eval)
        if total_eval > 0 else 0.0
    )

    model_valid = (
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
        "rho_r": (
            float(total_support / (total_support + LAMBDA))
            if total_support > 0 else 0.0
        ),
        "R_r": reliability(total_support, fid),
        "cf_valid": bool(total_correct > 0),
        "model_cf_validity_rate": model_valid,
        "all_models_cf_valid": bool(
            per_model and all(x["cf_valid"] for x in per_model)
        ),
        "min_witness_edit": global_min_edit,
    }


def source_evidence(rule):
    n = int(rule["n_r"])
    fid = float(rule["Fid_r"])
    hyp = conditions_to_hypothesis(rule["conditions"])
    return {
        "hyp": hyp,
        "pred": int(rule["predicted_class"]),
        "n_r": n,
        "Fid_r": fid,
        "coverage_eval": float(rule["coverage_eval"]),
        "rho_r": float(n / (n + LAMBDA)),
        "R_r": reliability(n, fid),
        "cf_valid": True,
        "model_cf_validity_rate": 1.0,
        "all_models_cf_valid": True,
        "min_witness_edit": rule.get("witness_edit_count"),
        "candidate_id": str(rule["candidate_id"]),
        "source_models": {rule["model"]},
        "source_surrogates": {rule["surrogate"]},
        "source_rule_count": 1,
    }


def reeval_evidence(
    hyp,
    pred,
    contexts,
    inputs,
):
    m = evaluate(hyp, pred, contexts)
    return {
        "hyp": hyp,
        "pred": int(pred),
        **m,
        "candidate_id": None,
        "source_models": set().union(*(x["source_models"] for x in inputs)),
        "source_surrogates": set().union(*(x["source_surrogates"] for x in inputs)),
        "source_rule_count": sum(int(x["source_rule_count"]) for x in inputs),
    }


def load_cache():
    idx = {}
    for p in sorted(CACHE_DIR.glob("*.json.gz")):
        z = read_gz(p)
        key = (z["case_id"], z["model"], int(z["prefix"]))
        require(key not in idx, f"Duplicate cache key: {key}")
        idx[key] = z
    return idx


def best_by_R(inputs):
    return sorted(
        inputs,
        key=lambda x: (
            -float(x["R_r"]),
            specificity(x["hyp"]),
            hyp_key(x["hyp"]),
        ),
    )[0]


def candidate_groups(inputs):
    by_h = defaultdict(list)
    for x in inputs:
        by_h[x["hyp"]].append(x)
    return by_h


def choose_scored_hypothesis(score_by_h, inputs_by_h):
    """
    Frozen global tie break:
      higher aggregation score
      -> higher input R
      -> more specific
      -> lexical canonical rule
    """
    ranked = []
    for hyp, score in score_by_h.items():
        max_input_R = max(float(x["R_r"]) for x in inputs_by_h[hyp])
        ranked.append(
            (
                hyp,
                float(score),
                max_input_R,
            )
        )

    ranked.sort(
        key=lambda z: (
            -z[1],
            -z[2],
            specificity(z[0]),
            hyp_key(z[0]),
        )
    )
    return ranked[0][0], ranked[0][1]


def select_baseline(
    name,
    items,
    stage,
    cfg,
):
    """
    items: list of (source_label, evidence)
    stage source_label:
      Stage1 -> candidate_id
      Stage2 -> surrogate
      Stage3 -> model
    """
    inputs = [ev for _, ev in items]
    require(inputs, "Empty baseline input group")

    if len(inputs) == 1:
        return inputs[0]["hyp"], {
            "selection_score": float(inputs[0]["R_r"]),
            "unique_hypothesis_count": 1,
        }

    if name == "best_single_evidence":
        ev = best_by_R(inputs)
        return ev["hyp"], {
            "selection_score": float(ev["R_r"]),
            "unique_hypothesis_count": len(candidate_groups(inputs)),
        }

    if name == "validation_best_source":
        preferred = None
        if stage == "Stage2":
            preferred = cfg["validation_selected_sources"][
                "best_single_surrogate"
            ]["selected"]
        elif stage == "Stage3":
            preferred = cfg["validation_selected_sources"][
                "best_single_model"
            ]["selected"]

        if preferred is not None:
            hits = [
                ev
                for label, ev in items
                if str(label) == str(preferred)
            ]
            if hits:
                require(
                    len(hits) == 1,
                    f"Multiple preferred source hits at {stage}: {preferred}",
                )
                return hits[0]["hyp"], {
                    "selection_score": float(hits[0]["R_r"]),
                    "unique_hypothesis_count": len(candidate_groups(inputs)),
                }

        ev = best_by_R(inputs)
        return ev["hyp"], {
            "selection_score": float(ev["R_r"]),
            "unique_hypothesis_count": len(candidate_groups(inputs)),
        }

    by_h = candidate_groups(inputs)

    if name == "uniform_vote":
        scores = {
            h: float(len(v))
            for h, v in by_h.items()
        }
        h, score = choose_scored_hypothesis(scores, by_h)
        return h, {
            "selection_score": score,
            "unique_hypothesis_count": len(by_h),
        }

    if name == "validation_weighted_vote":
        scores = defaultdict(float)

        if stage == "Stage1":
            for _, ev in items:
                scores[ev["hyp"]] += float(ev["R_r"])

        elif stage == "Stage2":
            weights = cfg["validation_fixed_weights"]["surrogate_weights"]
            for label, ev in items:
                scores[ev["hyp"]] += (
                    float(ev["R_r"]) * float(weights[str(label)])
                )

        elif stage == "Stage3":
            weights = cfg["validation_fixed_weights"]["model_weights"]
            for label, ev in items:
                scores[ev["hyp"]] += (
                    float(ev["R_r"]) * float(weights[str(label)])
                )
        else:
            raise ValueError(stage)

        h, score = choose_scored_hypothesis(scores, by_h)
        return h, {
            "selection_score": score,
            "unique_hypothesis_count": len(by_h),
        }

    if name == "fuzzy_probabilistic_or":
        scores = {}
        for h, evs in by_h.items():
            product = 1.0
            for ev in evs:
                mu = min(1.0, max(0.0, float(ev["R_r"])))
                product *= (1.0 - mu)
            scores[h] = float(1.0 - product)

        h, score = choose_scored_hypothesis(scores, by_h)
        return h, {
            "selection_score": score,
            "unique_hypothesis_count": len(by_h),
        }

    if name == "bayesian_categorical_noise":
        hyps = list(by_h.keys())
        M = len(hyps)

        if M == 1:
            return hyps[0], {
                "selection_score": 1.0,
                "unique_hypothesis_count": 1,
            }

        log_scores = {}
        log_prior = -math.log(M)

        for H in hyps:
            lp = log_prior

            for _, ev in items:
                r = min(
                    1.0 - EPS,
                    max(EPS, float(ev["R_r"])),
                )

                if ev["hyp"] == H:
                    likelihood = r
                else:
                    likelihood = max(
                        EPS,
                        (1.0 - r) / (M - 1),
                    )

                lp += math.log(likelihood)

            log_scores[H] = lp

        # Convert only for interpretable reporting; ranking by log-score is exact.
        max_lp = max(log_scores.values())
        exp_scores = {
            h: math.exp(v - max_lp)
            for h, v in log_scores.items()
        }
        denom = sum(exp_scores.values())
        post = {
            h: float(v / denom)
            for h, v in exp_scores.items()
        }

        h, score = choose_scored_hypothesis(post, by_h)
        return h, {
            "selection_score": score,
            "unique_hypothesis_count": M,
        }

    raise ValueError(f"Unknown baseline: {name}")


def s1_key(rule):
    hyp = conditions_to_hypothesis(rule["conditions"])
    return (
        rule["case_id"],
        rule["model"],
        int(rule["prefix"]),
        rule["surrogate"],
        int(rule["predicted_class"]),
        feature_signature(hyp),
    )


def stage_row(
    baseline,
    stage,
    key,
    ev,
    input_count,
    selection_meta,
):
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
        "baseline": baseline,
        "stage": stage,
        "case_id": case_id,
        "model": model,
        "prefix": int(prefix),
        "surrogate": surrogate,
        "predicted_class": int(pred),
        "feature_signature": json.dumps(list(sig)),
        "input_count": int(input_count),
        "actually_fused": bool(input_count >= 2),
        "selected_conditions": json.dumps(
            hypothesis_to_conditions(ev["hyp"]),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "selection_score": selection_meta.get("selection_score"),
        "unique_hypothesis_count": selection_meta.get(
            "unique_hypothesis_count"
        ),
        "n_r": int(ev["n_r"]),
        "Fid_r": ev["Fid_r"],
        "coverage_eval": float(ev["coverage_eval"]),
        "rho_r": float(ev["rho_r"]),
        "R_r": float(ev["R_r"]),
        "cf_valid": bool(ev["cf_valid"]),
        "model_cf_validity_rate": ev["model_cf_validity_rate"],
        "all_models_cf_valid": bool(ev["all_models_cf_valid"]),
        "min_witness_edit": ev["min_witness_edit"],
        "source_rule_count": int(ev["source_rule_count"]),
        "source_model_count": len(ev["source_models"]),
        "source_surrogate_count": len(ev["source_surrogates"]),
    }


def summarize_stage(rows, baseline, stage):
    sub = [
        r for r in rows
        if r["baseline"] == baseline
        and r["stage"] == stage
    ]
    fused = [r for r in sub if r["actually_fused"]]

    def sm(scope, prefix):
        return {
            f"{prefix}_count": len(scope),
            f"{prefix}_mean_fidelity": mean(
                r["Fid_r"] for r in scope
            ),
            f"{prefix}_median_fidelity": median(
                r["Fid_r"] for r in scope
            ),
            f"{prefix}_cf_validity_rate": mean(
                int(r["cf_valid"]) for r in scope
            ),
            f"{prefix}_mean_model_cf_validity_rate": mean(
                r["model_cf_validity_rate"] for r in scope
            ),
            f"{prefix}_all_models_cf_valid_rate": mean(
                int(r["all_models_cf_valid"]) for r in scope
            ),
            f"{prefix}_mean_support": mean(
                r["n_r"] for r in scope
            ),
            f"{prefix}_mean_coverage": mean(
                r["coverage_eval"] for r in scope
            ),
            f"{prefix}_mean_reliability": mean(
                r["R_r"] for r in scope
            ),
            f"{prefix}_mean_min_witness_edit": mean(
                r["min_witness_edit"] for r in scope
            ),
        }

    out = {
        "baseline": baseline,
        "stage": stage,
        "group_count": len(sub),
        "fusable_group_count": len(fused),
    }
    out.update(sm(sub, "all"))
    out.update(sm(fused, "fused"))
    return out


def run_one_baseline(name, cf, cache, cfg, audit):
    stage_rows = []

    # ---------------- Stage 1 ----------------
    g1 = defaultdict(list)
    for r in cf:
        g1[s1_key(r)].append(r)

    require(
        len(g1) == int(audit["stage1"]["group_count"]),
        f"{name}: Stage1 group count mismatch",
    )

    s1_out = {}

    for key in sorted(
        g1,
        key=lambda k: (
            k[0],
            MODEL_ORDER.get(k[1], 99),
            k[2],
            SURROGATE_ORDER.get(k[3], 99),
            k[4],
            k[5],
        ),
    ):
        case_id, model, prefix, surrogate, pred, sig = key

        source_rules = sorted(
            g1[key],
            key=lambda r: str(r["candidate_id"]),
        )
        items = [
            (str(r["candidate_id"]), source_evidence(r))
            for r in source_rules
        ]

        if len(items) == 1:
            ev = items[0][1]
            meta = {
                "selection_score": float(ev["R_r"]),
                "unique_hypothesis_count": 1,
            }
        else:
            selected_hyp, meta = select_baseline(
                name,
                items,
                "Stage1",
                cfg,
            )
            ctx = [cache[(case_id, model, int(prefix))]]
            ev = reeval_evidence(
                selected_hyp,
                pred,
                ctx,
                [x[1] for x in items],
            )

        s1_out[key] = ev
        stage_rows.append(
            stage_row(
                name,
                "Stage1",
                key,
                ev,
                len(items),
                meta,
            )
        )

    require(
        sum(len(v) >= 2 for v in g1.values())
        == int(audit["stage1"]["fusable_group_count_ge2_rules"]),
        f"{name}: Stage1 fusable count mismatch",
    )

    # ---------------- Stage 2 ----------------
    g2 = defaultdict(list)
    for key, ev in s1_out.items():
        case_id, model, prefix, surrogate, pred, sig = key
        g2[(case_id, model, prefix, pred, sig)].append(
            (surrogate, ev)
        )

    require(
        len(g2) == int(audit["stage2"]["compatible_sets"]),
        f"{name}: Stage2 group count mismatch",
    )

    s2_out = {}

    for key in sorted(
        g2,
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
            g2[key],
            key=lambda x: SURROGATE_ORDER.get(x[0], 99),
        )

        if len(items) == 1:
            ev = items[0][1]
            meta = {
                "selection_score": float(ev["R_r"]),
                "unique_hypothesis_count": 1,
            }
        else:
            selected_hyp, meta = select_baseline(
                name,
                items,
                "Stage2",
                cfg,
            )
            ctx = [cache[(case_id, model, int(prefix))]]
            ev = reeval_evidence(
                selected_hyp,
                pred,
                ctx,
                [x[1] for x in items],
            )

        s2_out[key] = ev
        stage_rows.append(
            stage_row(
                name,
                "Stage2",
                key,
                ev,
                len(items),
                meta,
            )
        )

    require(
        sum(len({s for s, _ in v}) >= 2 for v in g2.values())
        == int(audit["stage2"]["sets_with_ge2_surrogates"]),
        f"{name}: Stage2 fusable count mismatch",
    )

    # ---------------- Stage 3 ----------------
    g3 = defaultdict(list)
    for key, ev in s2_out.items():
        case_id, model, prefix, pred, sig = key
        g3[(case_id, prefix, pred, sig)].append(
            (model, ev)
        )

    require(
        len(g3) == int(audit["stage3"]["compatible_sets"]),
        f"{name}: Stage3 group count mismatch",
    )

    s3_out = {}

    for key in sorted(
        g3,
        key=lambda k: (
            k[0],
            k[1],
            k[2],
            k[3],
        ),
    ):
        case_id, prefix, pred, sig = key

        items = sorted(
            g3[key],
            key=lambda x: MODEL_ORDER.get(x[0], 99),
        )

        if len(items) == 1:
            ev = items[0][1]
            meta = {
                "selection_score": float(ev["R_r"]),
                "unique_hypothesis_count": 1,
            }
        else:
            selected_hyp, meta = select_baseline(
                name,
                items,
                "Stage3",
                cfg,
            )

            contexts = [
                cache[(case_id, model, int(prefix))]
                for model, _ in items
            ]

            ev = reeval_evidence(
                selected_hyp,
                pred,
                contexts,
                [x[1] for x in items],
            )

        s3_out[key] = ev
        stage_rows.append(
            stage_row(
                name,
                "Stage3",
                key,
                ev,
                len(items),
                meta,
            )
        )

    require(
        sum(len({m for m, _ in v}) >= 2 for v in g3.values())
        == int(audit["stage3"]["sets_with_ge2_blackboxes"]),
        f"{name}: Stage3 fusable count mismatch",
    )

    return stage_rows


def prefix_summary(stage_rows):
    out = []
    for baseline in BASELINES:
        s3 = [
            r for r in stage_rows
            if r["baseline"] == baseline
            and r["stage"] == "Stage3"
            and r["actually_fused"]
        ]

        for p in (5, 10, 15, 20):
            sub = [r for r in s3 if int(r["prefix"]) == p]
            if not sub:
                continue
            out.append({
                "baseline": baseline,
                "prefix": p,
                "N": len(sub),
                "mean_fidelity": mean(r["Fid_r"] for r in sub),
                "cf_validity_rate": mean(int(r["cf_valid"]) for r in sub),
                "mean_model_cf_validity_rate": mean(
                    r["model_cf_validity_rate"] for r in sub
                ),
                "all_models_cf_valid_rate": mean(
                    int(r["all_models_cf_valid"]) for r in sub
                ),
                "mean_reliability": mean(r["R_r"] for r in sub),
                "mean_coverage": mean(r["coverage_eval"] for r in sub),
                "mean_support": mean(r["n_r"] for r in sub),
                "mean_min_witness_edit": mean(
                    r["min_witness_edit"] for r in sub
                ),
            })
    return out


def fmt(x):
    return "NA" if x is None else f"{float(x):.6f}"


def producer_main_unused():
    required_files = [
        RULES,
        CACHE_MANIFEST,
        BANK_VALIDATION,
        AUDIT,
        BASELINE_CONFIG,
        DS_CONFIG,
        ADAPTIVE_CONFIG,
        SELECTION,
        DS_RESULT,
        DS_VALIDATION,
        SELECTIVE_RESULT,
        SELECTIVE_VALIDATION,
    ]
    missing = [str(p) for p in required_files if not p.is_file()]
    require(
        not missing,
        "Missing required file(s):\n" + "\n".join(missing),
    )
    require(CACHE_DIR.is_dir(), f"Missing cache dir: {CACHE_DIR}")

    baseline_cfg = read_json(BASELINE_CONFIG)
    ds_cfg = read_json(DS_CONFIG)
    adaptive_cfg = read_json(ADAPTIVE_CONFIG)
    selection = read_json(SELECTION)
    ds_result = read_json(DS_RESULT)
    ds_validation = read_json(DS_VALIDATION)
    selective_result = read_json(SELECTIVE_RESULT)
    selective_validation = read_json(SELECTIVE_VALIDATION)
    audit = read_json(AUDIT)
    cache_manifest = read_json(CACHE_MANIFEST)
    bank_val = read_json(BANK_VALIDATION)

    require(
        baseline_cfg.get("status") == "frozen",
        "Fusion baseline config is not frozen",
    )
    require(
        baseline_cfg["chronology"]["formal_test_files_read_by_this_script"]
        is False,
        "Script 41 reports formal TEST use",
    )
    require(
        baseline_cfg["chronology"]["formal_test_outcomes_used_for_parameter_selection"]
        is False,
        "Script 41 reports TEST-side parameter selection",
    )
    require(
        baseline_cfg["baselines"]["probability_averaging"]["status"]
        == "not_executable_from_current_canonical_rule_bank",
        "Probability averaging status changed unexpectedly",
    )

    require(ds_cfg.get("status") == "frozen", "D-S config not frozen")
    require(float(ds_cfg.get("lambda")) == LAMBDA, "D-S lambda != 5")
    require(
        selection.get("schema_version")
        == "etce-phase4f-confirmation-selection-v1"
        and selection.get("status") == "frozen",
        "Confirmation selection is not frozen",
    )
    require(
        int(selection.get("selected_case_count", -1)) == 200
        and int(selection.get("confirmation_target_count", -1)) == 3200
        and int(selection.get("global_sha256_rank_start", -1)) == 101
        and int(selection.get("global_sha256_rank_end", -1)) == 300
        and int(selection.get("overlap_with_original_phase4d", -1)) == 0,
        "Confirmation selection inventory/rank protocol mismatch",
    )
    require(
        selection.get("selection_uses_phase4d_or_phase4e_performance") is False,
        "Observed Phase-4D/4E performance influenced confirmation selection",
    )
    require(
        selection.get("frozen_ds_config_sha256") == sha256_file(DS_CONFIG)
        and selection.get("adaptive_config_sha256")
        == sha256_file(ADAPTIVE_CONFIG),
        "Frozen configuration hash mismatch",
    )
    require(
        adaptive_cfg.get("status") == "validation_frozen_formal_exploratory"
        and float(
            adaptive_cfg.get("gate", {}).get(
                "pre_mean_fidelity_threshold",
                -1,
            )
        )
        == 0.50,
        "Frozen selective Stage-3 config mismatch",
    )
    require(
        ds_result.get("status") == "passed"
        and ds_result.get("confirmation_test") is True,
        "D-S confirmation result not PASSED",
    )
    require(
        ds_validation.get("status") == "passed"
        and ds_validation.get("confirmation_test") is True,
        "D-S confirmation independent validation not PASSED",
    )
    require(
        selective_result.get("status") == "passed"
        and selective_result.get("confirmatory_evaluation") is True
        and selective_result.get("selective_threshold_selection_performed")
        is False
        and selective_result.get("protocol_retuning_performed") is False,
        "Selective Stage-3 confirmation result not frozen/PASSED",
    )
    require(
        selective_validation.get("status") == "passed"
        and selective_validation.get("confirmatory_evaluation") is True,
        "Selective Stage-3 confirmation independent validation not PASSED",
    )
    require(audit.get("status") == "passed", "Confirmation grouping audit not PASSED")
    require(bank_val.get("status") == "passed", "Confirmation rule bank validation not PASSED")
    require(cache_manifest.get("status") == "built", "Confirmation cache not built")
    require(
        cache_manifest.get("schema_version")
        == "phase4f-confirmation-fusion-eval-target-v1"
        and cache_manifest.get("same_targets_as_confirmation_rule_bank") is True,
        "Confirmation cache schema/identity mismatch",
    )
    require(int(cache_manifest["target_count"]) == 3200, "Confirmation cache target count !=3200")
    require(cache_manifest.get("true_labels_used") is False, "Confirmation cache used true labels")

    rules = read_jsonl(RULES)
    cf = [
        r for r in rules
        if r.get("rule_type") == "counterfactual"
        and bool(r.get("eligible_for_reliability"))
    ]

    require(len(rules) == 18273, f"Expected 18273 canonical rules, found {len(rules)}")
    require(len(cf) == 11294, f"Expected 11294 eligible CF candidates, found {len(cf)}")

    cache = load_cache()
    require(len(cache) == 3200, f"Expected 3200 cache targets, found {len(cache)}")
    require(
        all(
            p.get("schema_version")
            == "phase4f-confirmation-fusion-eval-target-v1"
            and p.get("confirmation_target") is True
            and p.get("true_label_used") is False
            for p in cache.values()
        ),
        "Invalid confirmation cache payload detected",
    )

    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 118)
    print("ETCE PHASE 4F-8 VALIDATION-FROZEN BASELINES ON CONFIRMATION TEST")
    print("=" * 118)
    print(f"Eligible CF candidates: {len(cf)}")
    print(f"Confirmation cache targets: {len(cache)}")
    print(f"Frozen baseline config: {BASELINE_CONFIG}")
    print("Probability Averaging: OMITTED (no comparable surrogate probability field)")
    print("Confirmation-side baseline tuning: False")
    print()

    all_stage_rows = []

    for name in BASELINES:
        print(f"[Run] {name} ...")
        rows = run_one_baseline(
            name,
            cf,
            cache,
            baseline_cfg,
            audit,
        )
        all_stage_rows.extend(rows)

        s3 = [
            r for r in rows
            if r["stage"] == "Stage3"
            and r["actually_fused"]
        ]

        print(
            f"      Stage3 N={len(s3)} | "
            f"Fid={mean(r['Fid_r'] for r in s3):.6f} | "
            f"CF-valid={mean(int(r['cf_valid']) for r in s3):.6f} | "
            f"model-valid={mean(r['model_cf_validity_rate'] for r in s3):.6f} | "
            f"all-model-valid={mean(int(r['all_models_cf_valid']) for r in s3):.6f}"
        )

    summary_rows = []
    for name in BASELINES:
        for stage in ("Stage1", "Stage2", "Stage3"):
            summary_rows.append(
                summarize_stage(
                    all_stage_rows,
                    name,
                    stage,
                )
            )

    # Add D-S ETCE reference from already validated output.
    ds_stage3 = ds_result["summary"]["Stage3"]

    comparison = []

    for name in BASELINES:
        row = next(
            r for r in summary_rows
            if r["baseline"] == name
            and r["stage"] == "Stage3"
        )
        comparison.append({
            "method": name,
            "role": "frozen_non_ds_baseline",
            "N": row["fused_count"],
            "rule_fidelity": row["fused_mean_fidelity"],
            "cf_validity": row["fused_cf_validity_rate"],
            "model_level_validity": row[
                "fused_mean_model_cf_validity_rate"
            ],
            "all_model_validity": row[
                "fused_all_models_cf_valid_rate"
            ],
            "mean_support": row["fused_mean_support"],
            "mean_coverage": row["fused_mean_coverage"],
            "mean_reliability": row["fused_mean_reliability"],
            "mean_min_witness_edit": row[
                "fused_mean_min_witness_edit"
            ],
        })

    comparison.append({
        "method": "ds_etce",
        "role": "independently_validated_reference",
        "N": ds_stage3["fused_count"],
        "rule_fidelity": ds_stage3["fused_mean_fidelity"],
        "cf_validity": ds_stage3["fused_cf_validity_rate"],
        "model_level_validity": ds_stage3[
            "fused_mean_model_cf_validity_rate"
        ],
        "all_model_validity": ds_stage3[
            "fused_all_models_cf_valid_rate"
        ],
        "mean_support": ds_stage3["fused_mean_support"],
        "mean_coverage": ds_stage3["fused_mean_coverage"],
        "mean_reliability": ds_stage3["fused_mean_reliability"],
        "mean_min_witness_edit": ds_stage3[
            "fused_mean_min_witness_edit"
        ],
    })

    # Add the independently validated selective Stage-3 confirmation result.
    # Only its primary Rule Fidelity is comparable here: when the frozen gate
    # rejects Stage 3 it retains multiple Stage-2 evidence objects rather than
    # emitting one representative fused rule, so the other rule-level fields
    # are intentionally left undefined.
    selective_fidelity = selective_result["fidelity"]["selective_stage3"]
    selective_n = selective_result["inventory"][
        "matched_fused_stage3_groups"
    ]
    require(int(selective_n) == 1041, "Selective Stage3 N != 1041")
    comparison.append({
        "method": "selective_stage3_etce",
        "role": "independently_validated_confirmatory_reference",
        "N": int(selective_n),
        "rule_fidelity": float(selective_fidelity),
        "cf_validity": None,
        "model_level_validity": None,
        "all_model_validity": None,
        "mean_support": None,
        "mean_coverage": None,
        "mean_reliability": None,
        "mean_min_witness_edit": None,
    })

    by_prefix = prefix_summary(all_stage_rows)

    write_csv(
        OUT / "confirmation_fusion_baseline_stage_outputs.csv",
        all_stage_rows,
    )
    write_csv(
        OUT / "confirmation_fusion_baseline_stage_summary.csv",
        summary_rows,
    )
    write_csv(
        OUT / "confirmation_fusion_strategy_comparison_stage3.csv",
        comparison,
    )
    write_csv(
        OUT / "confirmation_fusion_baselines_stage3_by_prefix.csv",
        by_prefix,
    )

    result = {
        "scope": "ETCE Phase 4F-8 validation-frozen fusion baselines on untouched confirmation TEST",
        "status": "passed",
        "confirmation_test": True,
        "confirmation_side_baseline_tuning": False,
        "baseline_config_frozen_before_this_run": True,
        "probability_averaging_executed": False,
        "probability_averaging_reason": (
            "No comparable surrogate class-probability field exists "
            "in the canonical rule bank."
        ),
        "eligible_cf_candidates": len(cf),
        "cache_targets": len(cache),
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
        "methods": BASELINES + ["ds_etce", "selective_stage3_etce"],
        "stage3_comparison": comparison,
        "chronology_disclosure": baseline_cfg["chronology"],
        "next_step": (
            "Independently validate the confirmation baseline outputs, then "
            "run paired case-clustered significance/effect-size comparisons "
            "on the 1041 matched Stage3 groups."
        ),
    }
    write_json(
        OUT / "confirmation_fusion_strategy_comparison.json",
        result,
    )

    # Markdown report
    lines = [
        "# ETCE Confirmation Fusion Strategy Comparison",
        "",
        "All non-D-S baseline definitions and weights were frozen in Script 41 "
        "using validation data only. Although the original Phase-4D formal TEST "
        "had already been run before those definitions were frozen, the present "
        "Phase-4F confirmation cohort was selected afterward and remained "
        "untouched until all protocols were fixed.",
        "",
        "Probability Averaging is not executed because the canonical rule bank "
        "does not contain comparable surrogate class probabilities.",
        "The selective Stage-3 row reports primary Rule Fidelity only because "
        "rejected groups retain their Stage-2 evidence set rather than one "
        "representative fused rule.",
        "",
        "## Stage-3 genuinely fused groups",
        "",
        "| Method | N | Fidelity | CF-valid | Model-valid | All-model-valid | Support | Coverage | Reliability | Min edit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in comparison:
        lines.append(
            f"| {r['method']} | {r['N']} | "
            f"{fmt(r['rule_fidelity'])} | "
            f"{fmt(r['cf_validity'])} | "
            f"{fmt(r['model_level_validity'])} | "
            f"{fmt(r['all_model_validity'])} | "
            f"{fmt(r['mean_support'])} | "
            f"{fmt(r['mean_coverage'])} | "
            f"{fmt(r['mean_reliability'])} | "
            f"{fmt(r['mean_min_witness_edit'])} |"
        )

    lines.extend([
        "",
        "No ranking or superiority claim is generated automatically. "
        "Interpretation should follow paired statistical analysis and "
        "conflict-stratified results.",
    ])

    (OUT / "PHASE4F_CONFIRMATION_FUSION_STRATEGY_COMPARISON_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 118)
    print("ETCE PHASE 4F-8 VALIDATION-FROZEN CONFIRMATION BASELINES PASSED")
    print("=" * 118)
    print("[Stage3 matched genuinely fused groups: N=1041]")
    for r in comparison:
        print(
            f"{r['method']:30s} | "
            f"Fid={fmt(r['rule_fidelity'])} | "
            f"CF={fmt(r['cf_validity'])} | "
            f"ModelValid={fmt(r['model_level_validity'])} | "
            f"AllModel={fmt(r['all_model_validity'])} | "
            f"R={fmt(r['mean_reliability'])}"
        )

    print()
    print(f"Output: {OUT}")
    print("Confirmation-side baseline tuning performed: False")
    print("Probability Averaging executed: False")
    print("NEXT: independently validate baseline outputs, then run paired significance/effect-size comparisons.")


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def blank(value):
    return value is None or str(value).strip().lower() in {"", "none", "nan"}


def bool_value(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def close_value(a, b, tol=1e-12):
    if a is None:
        return blank(b)
    if blank(b):
        return False
    return abs(float(a) - float(b)) <= tol


def json_norm(value):
    if blank(value):
        return None
    return json.dumps(
        json.loads(str(value)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stage_identity(row):
    return (
        str(row["baseline"]),
        str(row["stage"]),
        str(row["case_id"]),
        "" if blank(row.get("model")) else str(row.get("model")),
        int(row["prefix"]),
        "" if blank(row.get("surrogate")) else str(row.get("surrogate")),
        int(row["predicted_class"]),
        json_norm(row["feature_signature"]),
    )


def compare_stage_rows(expected, saved):
    require(len(expected) == len(saved), "Stage-output row count mismatch")
    exp_idx = {stage_identity(r): r for r in expected}
    got_idx = {stage_identity(r): r for r in saved}
    require(len(exp_idx) == len(expected), "Duplicate reproduced stage identity")
    require(len(got_idx) == len(saved), "Duplicate saved stage identity")
    require(set(exp_idx) == set(got_idx), "Stage-output identities differ")

    numeric = [
        "selection_score",
        "n_r",
        "Fid_r",
        "coverage_eval",
        "rho_r",
        "R_r",
        "model_cf_validity_rate",
        "min_witness_edit",
    ]
    integers = [
        "input_count",
        "unique_hypothesis_count",
        "source_rule_count",
        "source_model_count",
        "source_surrogate_count",
    ]
    booleans = ["actually_fused", "cf_valid", "all_models_cf_valid"]

    for identity, exp in exp_idx.items():
        got = got_idx[identity]
        require(
            json_norm(exp["selected_conditions"])
            == json_norm(got["selected_conditions"]),
            f"Selected-rule mismatch: {identity}",
        )
        for field in numeric:
            require(
                close_value(exp.get(field), got.get(field)),
                f"Stage numeric mismatch {identity} {field}: "
                f"{exp.get(field)} != {got.get(field)}",
            )
        for field in integers:
            require(
                int(exp[field]) == int(got[field]),
                f"Stage integer mismatch {identity} {field}",
            )
        for field in booleans:
            require(
                bool(exp[field]) == bool_value(got[field]),
                f"Stage boolean mismatch {identity} {field}",
            )


def table_identity(row, fields):
    return tuple(str(row[f]) for f in fields)


def compare_dict_rows(expected, saved, identity_fields):
    exp_idx = {table_identity(r, identity_fields): r for r in expected}
    got_idx = {table_identity(r, identity_fields): r for r in saved}
    require(len(exp_idx) == len(expected), "Duplicate reproduced table identity")
    require(len(got_idx) == len(saved), "Duplicate saved table identity")
    require(set(exp_idx) == set(got_idx), "Table identities differ")

    for identity, exp in exp_idx.items():
        got = got_idx[identity]
        for field, value in exp.items():
            if field in identity_fields:
                continue
            if isinstance(value, bool):
                require(
                    value == bool_value(got[field]),
                    f"Table boolean mismatch {identity} {field}",
                )
            elif isinstance(value, int):
                require(
                    int(got[field]) == value,
                    f"Table integer mismatch {identity} {field}",
                )
            elif isinstance(value, float) or value is None:
                require(
                    close_value(value, got.get(field)),
                    f"Table numeric mismatch {identity} {field}: "
                    f"{value} != {got.get(field)}",
                )
            else:
                require(
                    str(value) == str(got[field]),
                    f"Table text mismatch {identity} {field}",
                )


def main():
    required_files = [
        RULES,
        CACHE_MANIFEST,
        BANK_VALIDATION,
        AUDIT,
        BASELINE_CONFIG,
        DS_CONFIG,
        ADAPTIVE_CONFIG,
        SELECTION,
        DS_RESULT,
        DS_VALIDATION,
        SELECTIVE_RESULT,
        SELECTIVE_VALIDATION,
        SAVED_STAGE,
        SAVED_SUMMARY,
        SAVED_COMPARISON,
        SAVED_PREFIX,
        SAVED_RESULT,
    ]
    missing = [str(p) for p in required_files if not p.is_file()]
    require(not missing, "Missing required file(s):\n" + "\n".join(missing))
    require(CACHE_DIR.is_dir(), f"Missing cache dir: {CACHE_DIR}")

    baseline_cfg = read_json(BASELINE_CONFIG)
    ds_cfg = read_json(DS_CONFIG)
    adaptive_cfg = read_json(ADAPTIVE_CONFIG)
    selection = read_json(SELECTION)
    ds_result = read_json(DS_RESULT)
    ds_validation = read_json(DS_VALIDATION)
    selective_result = read_json(SELECTIVE_RESULT)
    selective_validation = read_json(SELECTIVE_VALIDATION)
    audit = read_json(AUDIT)
    cache_manifest = read_json(CACHE_MANIFEST)
    bank_validation = read_json(BANK_VALIDATION)
    saved_result = read_json(SAVED_RESULT)

    print("=" * 118)
    print("ETCE PHASE 4F-9 INDEPENDENT CONFIRMATION BASELINE VALIDATION")
    print("=" * 118)

    # Frozen-chain checks.
    require(baseline_cfg.get("status") == "frozen", "Baseline config not frozen")
    require(
        baseline_cfg["chronology"]["formal_test_files_read_by_this_script"]
        is False
        and baseline_cfg["chronology"][
            "formal_test_outcomes_used_for_parameter_selection"
        ]
        is False,
        "Baseline protocol chronology invalid",
    )
    require(
        baseline_cfg["baselines"]["probability_averaging"]["status"]
        == "not_executable_from_current_canonical_rule_bank",
        "Probability-averaging status changed",
    )
    require(
        ds_cfg.get("status") == "frozen"
        and float(ds_cfg.get("lambda", -1)) == LAMBDA,
        "D-S protocol mismatch",
    )
    require(
        selection.get("status") == "frozen"
        and selection.get("schema_version")
        == "etce-phase4f-confirmation-selection-v1"
        and int(selection.get("selected_case_count", -1)) == 200
        and int(selection.get("confirmation_target_count", -1)) == 3200
        and int(selection.get("global_sha256_rank_start", -1)) == 101
        and int(selection.get("global_sha256_rank_end", -1)) == 300
        and int(selection.get("overlap_with_original_phase4d", -1)) == 0
        and selection.get("selection_uses_phase4d_or_phase4e_performance")
        is False,
        "Confirmation selection protocol mismatch",
    )
    require(
        selection.get("frozen_ds_config_sha256") == sha256_file(DS_CONFIG)
        and selection.get("adaptive_config_sha256")
        == sha256_file(ADAPTIVE_CONFIG),
        "Frozen configuration hash mismatch",
    )
    require(
        adaptive_cfg.get("status") == "validation_frozen_formal_exploratory"
        and float(
            adaptive_cfg.get("gate", {}).get(
                "pre_mean_fidelity_threshold",
                -1,
            )
        )
        == 0.50,
        "Selective protocol mismatch",
    )
    require(
        ds_result.get("status") == "passed"
        and ds_validation.get("status") == "passed"
        and selective_result.get("status") == "passed"
        and selective_validation.get("status") == "passed",
        "Upstream confirmation validation not PASSED",
    )
    require(
        audit.get("status") == "passed"
        and bank_validation.get("status") == "passed",
        "Confirmation bank/audit not PASSED",
    )
    require(
        cache_manifest.get("status") == "built"
        and cache_manifest.get("schema_version")
        == "phase4f-confirmation-fusion-eval-target-v1"
        and cache_manifest.get("same_targets_as_confirmation_rule_bank") is True
        and int(cache_manifest.get("target_count", -1)) == 3200
        and cache_manifest.get("true_labels_used") is False,
        "Confirmation cache manifest mismatch",
    )
    require(
        saved_result.get("status") == "passed"
        and saved_result.get("confirmation_test") is True
        and saved_result.get("confirmation_side_baseline_tuning") is False
        and saved_result.get("probability_averaging_executed") is False,
        "Saved Phase 4F-8 result flags invalid",
    )

    rules = read_jsonl(RULES)
    cf = [
        r
        for r in rules
        if r.get("rule_type") == "counterfactual"
        and bool(r.get("eligible_for_reliability"))
    ]
    require(len(rules) == 18273, "Canonical rule count != 18273")
    require(len(cf) == 11294, "Eligible CF count != 11294")
    cache = load_cache()
    require(len(cache) == 3200, "Cache target count != 3200")
    require(
        all(
            p.get("schema_version")
            == "phase4f-confirmation-fusion-eval-target-v1"
            and p.get("confirmation_target") is True
            and p.get("true_label_used") is False
            for p in cache.values()
        ),
        "Invalid confirmation cache payload",
    )

    # Independent recomputation of all baseline stages.
    reproduced_stage = []
    for name in BASELINES:
        print(f"[Recompute] {name} ...")
        rows = run_one_baseline(name, cf, cache, baseline_cfg, audit)
        reproduced_stage.extend(rows)
        s3 = [
            r
            for r in rows
            if r["stage"] == "Stage3" and r["actually_fused"]
        ]
        require(len(s3) == 1041, f"{name}: fused Stage3 N != 1041")
        print(
            f"            Stage3 N={len(s3)} | "
            f"Fid={mean(r['Fid_r'] for r in s3):.12f}"
        )

    expected_stage_rows = len(BASELINES) * (9130 + 7539 + 5969)
    require(
        len(reproduced_stage) == expected_stage_rows,
        "Reproduced total stage-row count mismatch",
    )
    saved_stage = read_csv_rows(SAVED_STAGE)
    compare_stage_rows(reproduced_stage, saved_stage)
    print(
        f"[Compare] Stage outputs: {len(reproduced_stage)} / "
        f"{len(saved_stage)} identities and values matched"
    )

    reproduced_summary = [
        summarize_stage(reproduced_stage, name, stage)
        for name in BASELINES
        for stage in ("Stage1", "Stage2", "Stage3")
    ]
    saved_summary = read_csv_rows(SAVED_SUMMARY)
    compare_dict_rows(
        reproduced_summary,
        saved_summary,
        ["baseline", "stage"],
    )
    print("[Compare] Stage summaries: PASSED")

    reproduced_prefix = prefix_summary(reproduced_stage)
    saved_prefix = read_csv_rows(SAVED_PREFIX)
    compare_dict_rows(
        reproduced_prefix,
        saved_prefix,
        ["baseline", "prefix"],
    )
    print("[Compare] Prefix summaries: PASSED")

    # Reconstruct the Stage3 comparison table, including validated references.
    reproduced_comparison = []
    for name in BASELINES:
        row = next(
            r
            for r in reproduced_summary
            if r["baseline"] == name and r["stage"] == "Stage3"
        )
        reproduced_comparison.append(
            {
                "method": name,
                "role": "frozen_non_ds_baseline",
                "N": row["fused_count"],
                "rule_fidelity": row["fused_mean_fidelity"],
                "cf_validity": row["fused_cf_validity_rate"],
                "model_level_validity": row[
                    "fused_mean_model_cf_validity_rate"
                ],
                "all_model_validity": row[
                    "fused_all_models_cf_valid_rate"
                ],
                "mean_support": row["fused_mean_support"],
                "mean_coverage": row["fused_mean_coverage"],
                "mean_reliability": row["fused_mean_reliability"],
                "mean_min_witness_edit": row[
                    "fused_mean_min_witness_edit"
                ],
            }
        )

    ds_stage3 = ds_result["summary"]["Stage3"]
    reproduced_comparison.append(
        {
            "method": "ds_etce",
            "role": "independently_validated_reference",
            "N": ds_stage3["fused_count"],
            "rule_fidelity": ds_stage3["fused_mean_fidelity"],
            "cf_validity": ds_stage3["fused_cf_validity_rate"],
            "model_level_validity": ds_stage3[
                "fused_mean_model_cf_validity_rate"
            ],
            "all_model_validity": ds_stage3[
                "fused_all_models_cf_valid_rate"
            ],
            "mean_support": ds_stage3["fused_mean_support"],
            "mean_coverage": ds_stage3["fused_mean_coverage"],
            "mean_reliability": ds_stage3["fused_mean_reliability"],
            "mean_min_witness_edit": ds_stage3[
                "fused_mean_min_witness_edit"
            ],
        }
    )
    reproduced_comparison.append(
        {
            "method": "selective_stage3_etce",
            "role": "independently_validated_confirmatory_reference",
            "N": int(
                selective_result["inventory"][
                    "matched_fused_stage3_groups"
                ]
            ),
            "rule_fidelity": float(
                selective_result["fidelity"]["selective_stage3"]
            ),
            "cf_validity": None,
            "model_level_validity": None,
            "all_model_validity": None,
            "mean_support": None,
            "mean_coverage": None,
            "mean_reliability": None,
            "mean_min_witness_edit": None,
        }
    )
    saved_comparison = read_csv_rows(SAVED_COMPARISON)
    compare_dict_rows(
        reproduced_comparison,
        saved_comparison,
        ["method"],
    )
    print("[Compare] Stage3 strategy comparison: PASSED")

    require(
        saved_result["methods"]
        == BASELINES + ["ds_etce", "selective_stage3_etce"],
        "Saved result method order mismatch",
    )
    require(
        saved_result["group_counts"]
        == {
            "stage1": 9130,
            "stage1_fusable": 1746,
            "stage2": 7539,
            "stage2_fusable": 1209,
            "stage3": 5969,
            "stage3_fusable": 1041,
        },
        "Saved result group counts mismatch",
    )
    require(
        len(saved_result["stage3_comparison"])
        == len(reproduced_comparison),
        "Saved result comparison length mismatch",
    )
    for expected, saved in zip(
        reproduced_comparison,
        saved_result["stage3_comparison"],
    ):
        require(expected["method"] == saved["method"], "JSON method mismatch")
        require(expected["role"] == saved["role"], "JSON role mismatch")
        for field, value in expected.items():
            if field in {"method", "role"}:
                continue
            if isinstance(value, int):
                require(int(saved[field]) == value, f"JSON integer mismatch: {field}")
            else:
                require(
                    close_value(value, saved.get(field)),
                    f"JSON numeric mismatch: {expected['method']} {field}",
                )
    print("[Compare] Result JSON: PASSED")

    validation = {
        "scope": (
            "ETCE Phase 4F-9 independent validation of validation-frozen "
            "fusion baselines on untouched confirmation TEST"
        ),
        "status": "passed",
        "confirmation_test": True,
        "confirmation_side_baseline_tuning": False,
        "probability_averaging_executed": False,
        "reproduced": {
            "canonical_rules": len(rules),
            "eligible_cf_candidates": len(cf),
            "cache_targets": len(cache),
            "baseline_count": len(BASELINES),
            "stage_output_rows": len(reproduced_stage),
            "stage_summary_rows": len(reproduced_summary),
            "prefix_summary_rows": len(reproduced_prefix),
            "stage3_groups_per_method": 1041,
        },
        "checks": {
            "frozen_protocol_chain_verified": True,
            "all_stage_output_identities_match": True,
            "all_checked_stage_output_values_match": True,
            "all_stage_summaries_match": True,
            "all_prefix_summaries_match": True,
            "stage3_strategy_comparison_matches": True,
            "result_json_matches": True,
        },
        "stage3_comparison": reproduced_comparison,
        "next_step": (
            "Run paired case-clustered significance and effect-size "
            "comparisons for selective Stage3 against every frozen baseline."
        ),
    }
    write_json(OUT_VALIDATION, validation)

    print()
    print("=" * 118)
    print("ETCE PHASE 4F-9 INDEPENDENT CONFIRMATION BASELINE VALIDATION PASSED")
    print("=" * 118)
    print(f"Canonical rules reproduced : {len(rules)}")
    print(f"Eligible CF reproduced     : {len(cf)}")
    print(f"Stage rows matched         : {len(reproduced_stage)}")
    print(f"Stage3 groups/method       : 1041")
    for row in reproduced_comparison:
        print(
            f"{row['method']:30s} | "
            f"Fid={fmt(row['rule_fidelity'])}"
        )
    print("Confirmation-side baseline tuning performed: False")
    print(f"Validation JSON: {OUT_VALIDATION}")
    print("NEXT: paired significance/effect-size comparisons on 1041 groups.")


if __name__ == "__main__":
    main()
