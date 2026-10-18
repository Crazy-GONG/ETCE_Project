#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-13B / Script 47B
Process-aware proposal constraint ablation on frozen formal TEST targets.

Full ETCE:
    Existing validated process-aware formal local pool.

w/o Process-aware Proposal:
    Same factual prefix / same target / same black-box / same local seed /
    same candidate-count budget / same max edits=3 / same immutable historical
    outcome restriction, but candidate replacements come from the TRAIN-only
    global activity vocabulary WITHOUT same-position support, event-family
    preservation, DFG filtering, or process-aware mutation/crossover.

flow_valid(...) is used ONLY AFTER relaxed candidates are generated, as an
evaluation metric. No lambda / BPA / D-S / surrogate tuning occurs.

Run:
    cd D:/ETCE
    python -u ./scripts/47b_phase4d_process_aware_proposal_ablation.py
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".deps"
if sys.platform == "win32" and LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from etce.phase2.data import PREFIXES, load_frozen_common_cohort
from etce.phase3.core import (
    CanonicalPredictor,
    DIRECT_PROXIES,
    TERMINAL_OFFER_ACTIVITIES,
    FORBIDDEN_MUTATION_TARGETS,
    flow_valid,
    load_process_constraints,
    sequence_edit_count,
)
from etce.utils.io import read_json

RESULTS3 = ROOT / "results/phase3"
FORMAL = ROOT / "results/phase4/phase4d_formal_test"
SELECTION_CSV = FORMAL / "formal_test_case_selection.csv"
BANK = FORMAL / "formal_test_multirule_bank"
CACHE_DIR = BANK / "fusion_eval_cache" / "targets"
BANK_VALIDATION = BANK / "validation_phase4d_formal_test_multirule.json"
OUT = FORMAL / "formal_test_process_aware_proposal_ablation"

MODELS = ("RF", "XGBoost", "MLP", "LSTM")
MAX_EDITS = 3
MAX_OPTIONS = 8
POOL_SEED_BASE = 0
MAX_ATTEMPTS = 50000
TOL = 1e-12

IMMUTABLE = TERMINAL_OFFER_ACTIVITIES | DIRECT_PROXIES
FORBIDDEN = (
    set(FORBIDDEN_MUTATION_TARGETS)
    | TERMINAL_OFFER_ACTIVITIES
    | DIRECT_PROXIES
)


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def mean(xs):
    vals = [
        float(x) for x in xs
        if x is not None and math.isfinite(float(x))
    ]
    return float(sum(vals) / len(vals)) if vals else None


def median(xs):
    vals = [
        float(x) for x in xs
        if x is not None and math.isfinite(float(x))
    ]
    return float(statistics.median(vals)) if vals else None


def exact_binom_two_sided(k, n):
    if n <= 0:
        return 1.0
    k, n = int(k), int(n)
    lo = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2.0 * p)


def wilcoxon_or_sign(diffs):
    nz = [float(d) for d in diffs if abs(float(d)) > TOL]
    if not nz:
        return "all_ties", 1.0
    try:
        from scipy.stats import wilcoxon
        z = wilcoxon(
            nz,
            zero_method="wilcox",
            alternative="two-sided",
            correction=False,
            mode="auto",
        )
        return "wilcoxon_signed_rank_two_sided", float(z.pvalue)
    except Exception:
        pos = sum(d > 0 for d in nz)
        return "exact_sign_test_fallback", exact_binom_two_sided(pos, len(nz))


def paired_continuous(full_vals, abl_vals):
    require(len(full_vals) == len(abl_vals), "Paired continuous length mismatch")
    diffs = [float(f) - float(a) for f, a in zip(full_vals, abl_vals)]
    test, p = wilcoxon_or_sign(diffs)
    return {
        "N": len(diffs),
        "full_mean": mean(full_vals),
        "ablated_mean": mean(abl_vals),
        "delta_full_minus_ablated": mean(diffs),
        "median_delta": median(diffs),
        "full_wins": sum(d > TOL for d in diffs),
        "ties": sum(abs(d) <= TOL for d in diffs),
        "ablated_wins": sum(d < -TOL for d in diffs),
        "paired_test": test,
        "p_value": p,
    }


def paired_binary(full_vals, abl_vals):
    require(len(full_vals) == len(abl_vals), "Paired binary length mismatch")
    f0a1 = f1a0 = same11 = same00 = 0
    for f, a in zip(full_vals, abl_vals):
        f, a = int(bool(f)), int(bool(a))
        if f == 1 and a == 1:
            same11 += 1
        elif f == 0 and a == 0:
            same00 += 1
        elif f == 0 and a == 1:
            f0a1 += 1
        else:
            f1a0 += 1
    d = f0a1 + f1a0
    return {
        "N": len(full_vals),
        "full_rate": mean(int(bool(v)) for v in full_vals),
        "ablated_rate": mean(int(bool(v)) for v in abl_vals),
        "delta_full_minus_ablated": (
            mean(int(bool(v)) for v in full_vals)
            - mean(int(bool(v)) for v in abl_vals)
        ),
        "full0_ablated1": f0a1,
        "full1_ablated0": f1a0,
        "discordant": d,
        "test": "exact_mcnemar_binomial_two_sided",
        "p_value": exact_binom_two_sided(f1a0, d),
    }


def read_selection():
    with SELECTION_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    require(rows and "case_id" in rows[0], "Selection CSV has no case_id column")
    ids = [str(r["case_id"]).strip() for r in rows]
    require(len(ids) == 100, f"Expected 100 formal TEST rows, found {len(ids)}")
    require(len(set(ids)) == 100, "Formal TEST case IDs are not unique")
    return ids


def load_full_cache():
    idx = {}
    require(CACHE_DIR.is_dir(), f"Missing formal cache: {CACHE_DIR}")
    for p in sorted(CACHE_DIR.glob("*.json.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            z = json.load(f)
        key = (z["case_id"], z["model"], int(z["prefix"]))
        require(key not in idx, f"Duplicate formal cache target: {key}")
        idx[key] = z
    return idx


def local_seed(model, prefix, case_id):
    raw = f"{POOL_SEED_BASE}|{model}|{prefix}|{case_id}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def build_train_vocab(cohort):
    all_acts = set()
    for case in cohort:
        if case.split == "train":
            all_acts.update(str(a) for a in case.activities)
    require(all_acts, "Empty TRAIN activity vocabulary")
    mutable = sorted(a for a in all_acts if a not in FORBIDDEN)
    require(mutable, "No mutable TRAIN activities")
    return sorted(all_acts), mutable


def ranked_options(old, pos, mutable_vocab, seed, max_options=MAX_OPTIONS):
    vals = [a for a in mutable_vocab if a != old]
    vals.sort(
        key=lambda a: (
            hashlib.sha256(f"{seed}|{pos}|{a}".encode("utf-8")).hexdigest(),
            a,
        )
    )
    return vals[:max_options]


def generate_relaxed_pool(
    factual,
    mutable_vocab,
    rng_seed,
    target_pool_size,
):
    """
    Matched-budget relaxed proposal:
    - no same-position support
    - no event-family preservation
    - no DFG acceptance filter
    - no process-aware crossover
    - immutable historical outcomes remain unchanged
    - no forbidden outcome/proxy activity is introduced
    """
    factual = tuple(str(x) for x in factual)
    rng = random.Random(int(rng_seed))
    seen = {factual}
    pool = [factual]

    mutable_positions = [
        i for i, old in enumerate(factual)
        if old not in IMMUTABLE
    ]
    if not mutable_positions or target_pool_size <= 1:
        return pool

    options = {
        p: ranked_options(factual[p], p, mutable_vocab, rng_seed)
        for p in mutable_positions
    }
    active = [p for p in mutable_positions if options[p]]

    # Deterministic one-edit shell.
    for p in active:
        for replacement in options[p]:
            child = list(factual)
            child[p] = replacement
            child = tuple(child)
            if child in seen:
                continue
            require(sequence_edit_count(factual, child) == 1, "Invalid one-edit shell")
            seen.add(child)
            pool.append(child)
            if len(pool) >= target_pool_size:
                return pool

    attempts = 0
    while len(pool) < target_pool_size and attempts < MAX_ATTEMPTS:
        attempts += 1
        if not active:
            break
        k = rng.randint(1, min(MAX_EDITS, len(active)))
        positions = rng.sample(active, k)
        child = list(factual)
        for p in positions:
            child[p] = rng.choice(options[p])
        child = tuple(child)

        if child in seen:
            continue
        edits = sequence_edit_count(factual, child)
        if edits <= 0 or edits > MAX_EDITS:
            continue

        require(
            all(new == old if old in IMMUTABLE else True
                for old, new in zip(factual, child)),
            "Relaxed generator changed immutable historical outcome",
        )
        require(
            all(
                (new not in FORBIDDEN) if new != old else True
                for old, new in zip(factual, child)
            ),
            "Relaxed generator introduced forbidden outcome/proxy activity",
        )

        seen.add(child)
        pool.append(child)

    return pool


def temporal_monotone(timestamps):
    return all(b >= a for a, b in zip(timestamps, timestamps[1:]))


def evaluate_pool(pool, labels, factual, factual_label, constraints):
    recs = []
    for seq, label in zip(pool, labels):
        seq = tuple(seq)
        if seq == tuple(factual):
            continue
        recs.append({
            "flow_valid": bool(flow_valid(seq, constraints["edge_count"])),
            "flip": bool(int(label) != int(factual_label)),
            "edit_count": int(sequence_edit_count(factual, seq)),
        })

    valid = [r for r in recs if r["flow_valid"]]
    invalid = [r for r in recs if not r["flow_valid"]]
    flips = [r for r in recs if r["flip"]]
    valid_flips = [r for r in flips if r["flow_valid"]]
    invalid_flips = [r for r in flips if not r["flow_valid"]]

    return {
        "candidate_count": len(recs),
        "flow_valid_candidate_count": len(valid),
        "flow_invalid_candidate_count": len(invalid),
        "flow_valid_candidate_rate": len(valid) / len(recs) if recs else None,
        "flip_candidate_count": len(flips),
        "valid_flip_count": len(valid_flips),
        "invalid_flip_count": len(invalid_flips),
        "has_any_flip": bool(flips),
        "has_process_valid_flip": bool(valid_flips),
        "has_process_invalid_flip": bool(invalid_flips),
        "process_valid_flip_rate_among_flips": (
            len(valid_flips) / len(flips) if flips else None
        ),
        "process_invalid_flip_rate_among_flips": (
            len(invalid_flips) / len(flips) if flips else None
        ),
        "valid_flip_rate_over_all_candidates": (
            len(valid_flips) / len(recs) if recs else None
        ),
        "invalid_flip_rate_over_all_candidates": (
            len(invalid_flips) / len(recs) if recs else None
        ),
        "min_edit_any_flip": (
            min(r["edit_count"] for r in flips) if flips else None
        ),
        "min_edit_process_valid_flip": (
            min(r["edit_count"] for r in valid_flips) if valid_flips else None
        ),
    }


def summarize_variant(rows, variant):
    sub = [r for r in rows if r["variant"] == variant]
    cand = sum(int(r["candidate_count"]) for r in sub)
    valid_cand = sum(int(r["flow_valid_candidate_count"]) for r in sub)
    flips = sum(int(r["flip_candidate_count"]) for r in sub)
    valid_flips = sum(int(r["valid_flip_count"]) for r in sub)
    invalid_flips = sum(int(r["invalid_flip_count"]) for r in sub)

    return {
        "variant": variant,
        "target_count": len(sub),
        "candidate_count": cand,
        "flow_valid_candidate_count": valid_cand,
        "flow_invalid_candidate_count": cand - valid_cand,
        "flow_valid_candidate_rate": valid_cand / cand if cand else None,
        "GSR_any_flip": mean(int(r["has_any_flip"]) for r in sub),
        "process_valid_target_rate": mean(
            int(r["has_process_valid_flip"]) for r in sub
        ),
        "process_invalid_flip_target_rate": mean(
            int(r["has_process_invalid_flip"]) for r in sub
        ),
        "flip_candidate_count": flips,
        "valid_flip_count": valid_flips,
        "invalid_flip_count": invalid_flips,
        "process_valid_flip_rate_among_flips": (
            valid_flips / flips if flips else None
        ),
        "process_invalid_flip_rate_among_flips": (
            invalid_flips / flips if flips else None
        ),
        "valid_flip_rate_over_all_candidates": (
            valid_flips / cand if cand else None
        ),
        "invalid_flip_rate_over_all_candidates": (
            invalid_flips / cand if cand else None
        ),
        "temporal_valid_target_rate": mean(
            int(r["temporal_valid"]) for r in sub
        ),
        "mean_min_edit_any_flip": mean(
            r["min_edit_any_flip"] for r in sub
        ),
        "mean_min_edit_process_valid_flip": mean(
            r["min_edit_process_valid_flip"] for r in sub
        ),
        "mean_pool_fill_ratio": mean(
            r["pool_fill_ratio"] for r in sub
        ),
    }


def main():
    require(BANK_VALIDATION.is_file(), f"Missing: {BANK_VALIDATION}")
    require(
        read_json(BANK_VALIDATION).get("status") == "passed",
        "Formal multi-rule bank validation not PASSED",
    )

    selected_ids = read_selection()
    full_cache = load_full_cache()
    require(len(full_cache) == 1600, f"Expected 1600 cache targets, got {len(full_cache)}")
    require(
        {k[0] for k in full_cache} == set(selected_ids),
        "Cache case IDs do not match frozen selection",
    )

    cohort, _ = load_frozen_common_cohort(ROOT)
    case_map = {c.case_id: c for c in cohort}
    all_train_vocab, mutable_vocab = build_train_vocab(cohort)
    constraints = load_process_constraints(RESULTS3)
    canonical = read_json(
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUT / "relaxed_proposal_vocabulary.json",
        {
            "source": "frozen TRAIN split only",
            "all_training_activity_count": len(all_train_vocab),
            "mutable_activity_count": len(mutable_vocab),
            "test_used_to_define_vocabulary": False,
            "mutable_activities": mutable_vocab,
        },
    )

    print("=" * 118)
    print("ETCE PHASE 4D-13B PROCESS-AWARE PROPOSAL CONSTRAINT ABLATION")
    print("=" * 118)
    print("Full ETCE                  : validated process-aware local proposal")
    print("w/o Process-aware Proposal : TRAIN-global activity replacement")
    print("Same formal targets         : True")
    print("Same target pool-size budget: True")
    print("Same local seed             : True")
    print("Same max edits              : 3")
    print("Immutable outcomes preserved: True")
    print("Family preservation ablation: OFF")
    print("Same-position support       : OFF")
    print("DFG filtering in ablation   : OFF")
    print("flow_valid evaluation only  : True")
    print("Protocol retuning           : False")
    print(f"TRAIN activities            : {len(all_train_vocab)}")
    print(f"Relaxed mutable activities  : {len(mutable_vocab)}")
    print()

    detail = []
    counter = 0

    for model in MODELS:
        for prefix in PREFIXES:
            predictor = CanonicalPredictor(
                ROOT,
                model,
                prefix,
                canonical["items"][model][str(prefix)],
            )

            for case_id in selected_ids:
                counter += 1
                key = (case_id, model, int(prefix))
                fc = full_cache[key]
                case = case_map[case_id]
                factual = tuple(case.activities[:prefix])
                timestamps = tuple(case.timestamps[:prefix])
                temporal_ok = temporal_monotone(timestamps)

                full_pool = [tuple(x) for x in fc["pool_sequences"]]
                full_labels = np.asarray(fc["pool_labels"], dtype=int)
                factual_idx = full_pool.index(factual)
                factual_label = int(full_labels[factual_idx])

                full_m = evaluate_pool(
                    full_pool,
                    full_labels,
                    factual,
                    factual_label,
                    constraints,
                )
                require(
                    full_m["flow_invalid_candidate_count"] == 0,
                    f"Full formal cache contains DFG-invalid candidate: {key}",
                )

                budget = len(full_pool)
                seed = local_seed(model, prefix, case_id)

                abl_pool = generate_relaxed_pool(
                    factual,
                    mutable_vocab,
                    seed,
                    budget,
                )

                abl_probs = np.asarray(
                    predictor.predict_proba(
                        abl_pool,
                        [timestamps] * len(abl_pool),
                    ),
                    dtype=float,
                )
                abl_labels = (abl_probs >= 0.5).astype(int)
                abl_factual_idx = abl_pool.index(factual)
                require(
                    int(abl_labels[abl_factual_idx]) == factual_label,
                    f"Factual black-box label changed: {key}",
                )

                abl_m = evaluate_pool(
                    abl_pool,
                    abl_labels,
                    factual,
                    factual_label,
                    constraints,
                )

                common = {
                    "case_id": case_id,
                    "model": model,
                    "prefix": int(prefix),
                    "factual_label": factual_label,
                    "temporal_valid": bool(temporal_ok),
                    "target_pool_size_budget": budget,
                }

                detail.append({
                    **common,
                    "variant": "Full_ETCE",
                    "pool_size": len(full_pool),
                    "pool_fill_ratio": 1.0,
                    **full_m,
                })
                detail.append({
                    **common,
                    "variant": "w/o_Process_Aware_Proposal",
                    "pool_size": len(abl_pool),
                    "pool_fill_ratio": len(abl_pool) / budget,
                    **abl_m,
                })

                if counter % 100 == 0 or counter == 1600:
                    print(
                        f"[{counter:04d}/1600] {model} P{prefix} {case_id} | "
                        f"Budget={budget} | AblFill={len(abl_pool)} | "
                        f"FullFlow={full_m['flow_valid_candidate_rate']:.4f} | "
                        f"AblFlow={abl_m['flow_valid_candidate_rate']:.4f} | "
                        f"FullFlip={int(full_m['has_any_flip'])} | "
                        f"AblFlip={int(abl_m['has_any_flip'])} | "
                        f"AblInvalidFlip={int(abl_m['has_process_invalid_flip'])}",
                        flush=True,
                    )

    require(len(detail) == 3200, f"Expected 3200 rows, got {len(detail)}")

    full_rows = [r for r in detail if r["variant"] == "Full_ETCE"]
    abl_rows = [r for r in detail if r["variant"] == "w/o_Process_Aware_Proposal"]

    def k(r):
        return (r["case_id"], r["model"], int(r["prefix"]))

    full_idx = {k(r): r for r in full_rows}
    abl_idx = {k(r): r for r in abl_rows}
    require(set(full_idx) == set(abl_idx), "Full/Ablation identities differ")
    keys = sorted(full_idx)

    overall = [
        summarize_variant(detail, "Full_ETCE"),
        summarize_variant(detail, "w/o_Process_Aware_Proposal"),
    ]

    continuous = []
    for metric in (
        "flow_valid_candidate_rate",
        "valid_flip_rate_over_all_candidates",
        "invalid_flip_rate_over_all_candidates",
        "process_valid_flip_rate_among_flips",
        "min_edit_any_flip",
        "min_edit_process_valid_flip",
        "pool_fill_ratio",
    ):
        usable = [
            key for key in keys
            if full_idx[key][metric] is not None
            and abl_idx[key][metric] is not None
        ]
        if usable:
            continuous.append({
                "metric": metric,
                **paired_continuous(
                    [full_idx[key][metric] for key in usable],
                    [abl_idx[key][metric] for key in usable],
                ),
            })

    binary = []
    for metric in (
        "has_any_flip",
        "has_process_valid_flip",
        "has_process_invalid_flip",
        "temporal_valid",
    ):
        binary.append({
            "metric": metric,
            **paired_binary(
                [full_idx[key][metric] for key in keys],
                [abl_idx[key][metric] for key in keys],
            ),
        })

    # Model/prefix summaries.
    grouped = []
    for variant in ("Full_ETCE", "w/o_Process_Aware_Proposal"):
        for model in MODELS:
            for prefix in PREFIXES:
                sub = [
                    r for r in detail
                    if r["variant"] == variant
                    and r["model"] == model
                    and int(r["prefix"]) == int(prefix)
                ]
                if sub:
                    s = summarize_variant(sub, variant)
                    grouped.append({
                        "variant": variant,
                        "model": model,
                        "prefix": int(prefix),
                        **{kk: vv for kk, vv in s.items() if kk != "variant"},
                    })

    write_csv(OUT / "proposal_ablation_target_detail.csv", detail)
    write_csv(OUT / "proposal_ablation_overall_summary.csv", overall)
    write_csv(OUT / "proposal_ablation_by_model_prefix.csv", grouped)
    write_csv(OUT / "proposal_ablation_paired_continuous.csv", continuous)
    write_csv(OUT / "proposal_ablation_paired_binary.csv", binary)

    result = {
        "scope": "ETCE Phase 4D-13B process-aware proposal constraint ablation",
        "status": "passed",
        "formal_test": True,
        "formal_target_count": 1600,
        "full_variant": "validated process-aware local proposal",
        "ablated_variant": (
            "matched-budget TRAIN-global activity replacement without "
            "same-position/event-family/DFG proposal constraints"
        ),
        "train_activity_domain": {
            "all_activity_count": len(all_train_vocab),
            "mutable_activity_count": len(mutable_vocab),
            "test_used_to_define_domain": False,
        },
        "controlled_factors": {
            "same_formal_targets": True,
            "same_factual_prefix": True,
            "same_blackbox": True,
            "same_local_seed": True,
            "same_target_pool_size_budget": True,
            "same_max_edits": MAX_EDITS,
            "immutable_outcomes_preserved": True,
            "forbidden_outcomes_not_introduced": True,
            "timestamps_preserved": True,
        },
        "removed_process_knowledge": {
            "same_position_support": True,
            "event_family_preservation": True,
            "DFG_filter": True,
            "process_aware_mutation_crossover": True,
        },
        "overall": {r["variant"]: r for r in overall},
        "paired_continuous": continuous,
        "paired_binary": binary,
        "temporal_recourse_claimed": False,
        "pn_conformance_evaluated": False,
        "test_side_tuning": False,
        "protocol_retuning": False,
        "interpretation_guard": (
            "This is a proposal-constraint ablation, broader than the redundant "
            "single-line terminal flow_valid check evaluated in 47A."
        ),
    }
    write_json(OUT / "proposal_ablation_summary.json", result)

    def fmt(v):
        return "NA" if v is None else f"{float(v):.6f}"

    print()
    print("=" * 118)
    print("ETCE PHASE 4D-13B PROCESS-AWARE PROPOSAL CONSTRAINT ABLATION PASSED")
    print("=" * 118)
    for r in overall:
        print(
            f"{r['variant']:30s} | "
            f"Targets={r['target_count']} | "
            f"Candidates={r['candidate_count']} | "
            f"FlowValid={r['flow_valid_candidate_rate']:.6f} | "
            f"GSR={r['GSR_any_flip']:.6f} | "
            f"ValidTarget={r['process_valid_target_rate']:.6f} | "
            f"InvalidCFTarget={r['process_invalid_flip_target_rate']:.6f} | "
            f"ValidFlipRate={fmt(r['process_valid_flip_rate_among_flips'])} | "
            f"InvalidFlipRate={fmt(r['process_invalid_flip_rate_among_flips'])} | "
            f"MinEdit={fmt(r['mean_min_edit_any_flip'])} | "
            f"Fill={r['mean_pool_fill_ratio']:.6f}"
        )

    print()
    print("[Paired continuous Full minus Ablation]")
    for r in continuous:
        print(
            f"{r['metric']:40s} | "
            f"Full={r['full_mean']:.6f} | "
            f"Abl={r['ablated_mean']:.6f} | "
            f"Delta={r['delta_full_minus_ablated']:+.6f} | "
            f"W/T/L={r['full_wins']}/{r['ties']}/{r['ablated_wins']} | "
            f"p={r['p_value']:.6g}"
        )

    print()
    print("[Paired binary Full minus Ablation]")
    for r in binary:
        print(
            f"{r['metric']:30s} | "
            f"Full={r['full_rate']:.6f} | "
            f"Abl={r['ablated_rate']:.6f} | "
            f"Delta={r['delta_full_minus_ablated']:+.6f} | "
            f"discordant={r['discordant']} | "
            f"p={r['p_value']:.6g}"
        )

    print()
    print(f"Output: {OUT}")
    print("flow_valid used to generate ablation candidates: False")
    print("TRAIN-only activity vocabulary used: True")
    print("Temporal recourse claimed: False")
    print("Protocol retuning performed: False")
    print("NEXT: independent Petri-net conformance evaluation.")


if __name__ == "__main__":
    main()
