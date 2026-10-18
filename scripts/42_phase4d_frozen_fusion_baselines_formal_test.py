#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-8 / Script 42
Frozen Fusion Baselines on Formal TEST

Uses the already frozen validation-only baseline protocol from Script 41.
No baseline formula, source weight, source choice, tie-break, or epsilon may
be changed in response to formal TEST results.

Baselines:
- Best Single Evidence
- Validation-best Source
- Uniform Vote
- Validation-weighted Vote
- Fuzzy Probabilistic OR
- Bayesian Categorical-noise Pooling
- D-S ETCE (reference only; loaded from independently validated Phase 4D-4/5)

Probability Averaging is intentionally omitted because Script 41 established
that the current canonical rule bank contains no comparable surrogate class
probability field.

All non-D-S baselines:
- use the same Stage1/2/3 compatible groups;
- select only from exact hypotheses present in their inputs;
- use frozen deterministic tie-breaking;
- re-evaluate the selected rule on the same formal held-out cache;
- propagate the re-evaluated evidence to the next stage.

Run:
    cd D:/ETCE
    python -u ./scripts/42_phase4d_frozen_fusion_baselines_formal_test.py
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

FORMAL = ROOT / "results/phase4/phase4d_formal_test"
BANK = FORMAL / "formal_test_multirule_bank"
AUDIT_DIR = FORMAL / "formal_test_target_local_group_audit"
DS_DIR = FORMAL / "formal_test_frozen_ds_fusion"

RULES = BANK / "canonical_rules_test.jsonl"
CACHE_DIR = BANK / "fusion_eval_cache" / "targets"
CACHE_MANIFEST = BANK / "fusion_eval_cache_manifest.json"
BANK_VALIDATION = BANK / "validation_phase4d_formal_test_multirule.json"

AUDIT = AUDIT_DIR / "formal_test_target_local_group_audit.json"
BASELINE_CONFIG = ROOT / "configs/phase4c_fusion_baselines_frozen_config.json"
DS_CONFIG = ROOT / "configs/phase4c_ds_frozen_config.json"
DS_RESULT = DS_DIR / "formal_test_frozen_ds_fusion.json"
DS_VALIDATION = DS_DIR / "validation_phase4d_frozen_ds_fusion.json"

OUT = FORMAL / "formal_test_fusion_baselines"

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


def main():
    required_files = [
        RULES,
        CACHE_MANIFEST,
        BANK_VALIDATION,
        AUDIT,
        BASELINE_CONFIG,
        DS_CONFIG,
        DS_RESULT,
        DS_VALIDATION,
    ]
    missing = [str(p) for p in required_files if not p.is_file()]
    require(
        not missing,
        "Missing required file(s):\n" + "\n".join(missing),
    )
    require(CACHE_DIR.is_dir(), f"Missing cache dir: {CACHE_DIR}")

    baseline_cfg = read_json(BASELINE_CONFIG)
    ds_cfg = read_json(DS_CONFIG)
    ds_result = read_json(DS_RESULT)
    ds_validation = read_json(DS_VALIDATION)
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
    require(ds_result.get("status") == "passed", "D-S formal result not PASSED")
    require(ds_validation.get("status") == "passed", "D-S independent validation not PASSED")
    require(audit.get("status") == "passed", "Formal grouping audit not PASSED")
    require(bank_val.get("status") == "passed", "Formal rule bank validation not PASSED")
    require(cache_manifest.get("status") == "built", "Formal cache not built")
    require(int(cache_manifest["target_count"]) == 1600, "Formal cache target count !=1600")

    rules = read_jsonl(RULES)
    cf = [
        r for r in rules
        if r.get("rule_type") == "counterfactual"
        and bool(r.get("eligible_for_reliability"))
    ]

    require(len(rules) == 9846, f"Expected 9846 canonical rules, found {len(rules)}")
    require(len(cf) == 6172, f"Expected 6172 eligible CF candidates, found {len(cf)}")

    cache = load_cache()
    require(len(cache) == 1600, f"Expected 1600 cache targets, found {len(cache)}")

    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 118)
    print("ETCE PHASE 4D-8 FROZEN FUSION BASELINES ON FORMAL TEST")
    print("=" * 118)
    print(f"Eligible CF candidates: {len(cf)}")
    print(f"Formal cache targets: {len(cache)}")
    print(f"Frozen baseline config: {BASELINE_CONFIG}")
    print("Probability Averaging: OMITTED (no comparable surrogate probability field)")
    print("TEST-side baseline tuning: False")
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

    by_prefix = prefix_summary(all_stage_rows)

    write_csv(
        OUT / "formal_fusion_baseline_stage_outputs.csv",
        all_stage_rows,
    )
    write_csv(
        OUT / "formal_fusion_baseline_stage_summary.csv",
        summary_rows,
    )
    write_csv(
        OUT / "formal_fusion_strategy_comparison_stage3.csv",
        comparison,
    )
    write_csv(
        OUT / "formal_fusion_baselines_stage3_by_prefix.csv",
        by_prefix,
    )

    result = {
        "scope": "ETCE Phase 4D-8 frozen fusion baselines on formal TEST",
        "status": "passed",
        "formal_test": True,
        "test_side_baseline_tuning": False,
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
        "methods": BASELINES + ["ds_etce"],
        "stage3_comparison": comparison,
        "chronology_disclosure": baseline_cfg["chronology"],
        "next_step": (
            "Run paired statistical comparison on the 529 matched Stage3 groups, "
            "then perform support-aware BPA ablation."
        ),
    }
    write_json(
        OUT / "formal_fusion_strategy_comparison.json",
        result,
    )

    # Markdown report
    lines = [
        "# ETCE Formal Fusion Strategy Comparison",
        "",
        "All non-D-S baseline definitions and weights were frozen in Script 41 "
        "using validation data only. The primary ETCE TEST had already been run "
        "before those baseline definitions were frozen; this chronology is "
        "reported explicitly.",
        "",
        "Probability Averaging is not executed because the canonical rule bank "
        "does not contain comparable surrogate class probabilities.",
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

    (OUT / "PHASE4D_FUSION_STRATEGY_COMPARISON_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 118)
    print("ETCE PHASE 4D-8 FROZEN FUSION BASELINES FORMAL TEST PASSED")
    print("=" * 118)
    print("[Stage3 genuinely fused groups: N=529]")
    for r in comparison:
        print(
            f"{r['method']:30s} | "
            f"Fid={r['rule_fidelity']:.6f} | "
            f"CF={r['cf_validity']:.6f} | "
            f"ModelValid={r['model_level_validity']:.6f} | "
            f"AllModel={r['all_model_validity']:.6f} | "
            f"R={r['mean_reliability']:.6f}"
        )

    print()
    print(f"Output: {OUT}")
    print("TEST-side baseline tuning performed: False")
    print("Probability Averaging executed: False")
    print("NEXT: paired significance/effect-size comparison, then support-aware BPA ablation.")


if __name__ == "__main__":
    main()
