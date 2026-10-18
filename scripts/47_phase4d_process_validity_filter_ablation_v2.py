#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-13A / Script 47
Process-validity filter ablation on the frozen formal TEST targets.

Predeclared ablation:
    Full ETCE                  : process-validity filter ON
    w/o Process-validity Filter: same local search, but remove the DFG flow-validity rejection

This script evaluates the GENERATION / LOCAL-NEIGHBORHOOD effect of that filter.
It does NOT retune ETCE, lambda, D-S, surrogate hyperparameters, or case selection.

Important scope:
- Full ETCE neighborhood is read from the already validated formal fusion cache.
- Ablation neighborhood is regenerated deterministically from the same factual prefix,
  same local seed, same pool-size target, same max edits, same semantic mutation options,
  same immutable/outcome restrictions, same crossover rate, and same attempt budget.
- The ONLY intended difference is removal of:
      flow_valid(child, training_DFG)
  rejection during local pool generation.
- Black-box predictions are recomputed with the frozen canonical model using the
  factual timestamps exactly as in the full method.
- Because timestamps are not changed by this activity-substitution generator,
  temporal validity is audited but is expected to remain unchanged. This experiment
  must NOT be described as demonstrating temporal-recourse capability.
- PN conformance is not evaluated here; it should be handled separately with a
  training-derived process model.

Outputs:
- target-level Full vs Ablation generation statistics
- candidate-level flow-validity statistics
- paired tests for target-level GSR / process-valid CF availability
- by-prefix / by-model summaries

Run:
    cd D:/ETCE
    python -u ./scripts/47_phase4d_process_validity_filter_ablation.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
import statistics

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
from etce.phase3.semantic_core import (
    family_preserved,
    semantic_crossover,
    semantic_mutate_once,
    semantic_mutation_options,
)
from etce.utils.io import read_json

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

RESULTS3 = ROOT / "results/phase3"

FORMAL = ROOT / "results/phase4/phase4d_formal_test"
SELECTION = FORMAL / "formal_test_case_selection.csv"
BANK = FORMAL / "formal_test_multirule_bank"
CACHE_DIR = BANK / "fusion_eval_cache" / "targets"
BANK_VALIDATION = BANK / "validation_phase4d_formal_test_multirule.json"

OUT = FORMAL / "formal_test_process_validity_filter_ablation"

MODELS = ("RF", "XGBoost", "MLP", "LSTM")
POOL_SIZE = 320
MAX_EDITS = 3
MAX_OPTIONS = 8
POOL_SEED_BASE = 0
CROSSOVER_RATE = 0.35
MAX_ATTEMPTS = 20000

TOL = 1e-12
BOOTSTRAP_B = 20000
BOOTSTRAP_SEED = 20260930


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


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


def local_seed(model, prefix, case_id):
    raw = f"{POOL_SEED_BASE}|{model}|{prefix}|{case_id}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def mean(xs):
    vals = [
        float(x)
        for x in xs
        if x is not None
        and math.isfinite(float(x))
    ]
    return float(sum(vals) / len(vals)) if vals else None


def median(xs):
    vals = [
        float(x)
        for x in xs
        if x is not None
        and math.isfinite(float(x))
    ]
    return float(statistics.median(vals)) if vals else None


def percentile(sorted_vals, p):
    vals = list(sorted_vals)

    if not vals:
        return None

    if len(vals) == 1:
        return float(vals[0])

    z = (len(vals) - 1) * float(p)
    lo = int(math.floor(z))
    hi = int(math.ceil(z))

    if lo == hi:
        return float(vals[lo])

    frac = z - lo
    return float(
        vals[lo] * (1 - frac)
        + vals[hi] * frac
    )


def bootstrap_ci_mean(
    diffs,
    B=BOOTSTRAP_B,
    seed=BOOTSTRAP_SEED,
):
    vals = [float(v) for v in diffs]
    require(vals, "Empty bootstrap input")

    rng = random.Random(seed)
    n = len(vals)
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


def exact_binom_two_sided(k, n):
    if n <= 0:
        return 1.0

    k = int(k)
    n = int(n)
    lo = min(k, n - k)

    p = sum(
        math.comb(n, i)
        for i in range(lo + 1)
    ) / (2 ** n)

    return min(1.0, 2.0 * p)


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


def paired_continuous(full_vals, abl_vals):
    require(
        len(full_vals) == len(abl_vals),
        "Paired continuous length mismatch",
    )

    diffs = [
        float(f) - float(a)
        for f, a in zip(
            full_vals,
            abl_vals,
        )
    ]

    lo, hi = bootstrap_ci_mean(diffs)
    p = wilcoxon_or_sign(diffs)

    return {
        "N": len(diffs),
        "full_mean": mean(full_vals),
        "ablated_mean": mean(abl_vals),
        "delta_full_minus_ablated": mean(diffs),
        "bootstrap95_lo": lo,
        "bootstrap95_hi": hi,
        "full_wins": sum(d > TOL for d in diffs),
        "ties": sum(abs(d) <= TOL for d in diffs),
        "ablated_wins": sum(d < -TOL for d in diffs),
        "paired_test": p["test"],
        "p_value": p["p_value"],
    }


def paired_binary(full_vals, abl_vals):
    require(
        len(full_vals) == len(abl_vals),
        "Paired binary length mismatch",
    )

    f0a1 = 0
    f1a0 = 0
    same11 = 0
    same00 = 0

    for f, a in zip(full_vals, abl_vals):
        f = int(bool(f))
        a = int(bool(a))

        if f == 1 and a == 1:
            same11 += 1
        elif f == 0 and a == 0:
            same00 += 1
        elif f == 0 and a == 1:
            f0a1 += 1
        else:
            f1a0 += 1

    discordant = f0a1 + f1a0

    return {
        "N": len(full_vals),
        "full_rate": mean(int(bool(v)) for v in full_vals),
        "ablated_rate": mean(int(bool(v)) for v in abl_vals),
        "delta_full_minus_ablated": (
            mean(int(bool(v)) for v in full_vals)
            - mean(int(bool(v)) for v in abl_vals)
        ),
        "agree_11": same11,
        "agree_00": same00,
        "full0_ablated1": f0a1,
        "full1_ablated0": f1a0,
        "discordant": discordant,
        "test": "exact_mcnemar_binomial_two_sided",
        "p_value": exact_binom_two_sided(
            f1a0,
            discordant,
        ),
    }


def temporal_monotone(timestamps):
    """
    Only audits factual timestamps because activity-substitution candidates
    reuse them exactly.
    """
    if not timestamps:
        return True

    for a, b in zip(
        timestamps,
        timestamps[1:],
    ):
        if b < a:
            return False

    return True


# ---------------------------------------------------------------------
# Full formal cache
# ---------------------------------------------------------------------

def load_full_cache():
    import gzip

    idx = {}

    require(
        CACHE_DIR.is_dir(),
        f"Missing formal cache directory: {CACHE_DIR}",
    )

    for p in sorted(CACHE_DIR.glob("*.json.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            z = json.load(f)

        key = (
            z["case_id"],
            z["model"],
            int(z["prefix"]),
        )

        require(
            key not in idx,
            f"Duplicate formal cache target: {key}",
        )

        idx[key] = z

    return idx


# ---------------------------------------------------------------------
# Ablated generator: identical except NO DFG flow-validity rejection
# ---------------------------------------------------------------------

def generate_without_process_validity_filter(
    factual,
    constraints,
    rng_seed,
    pool_size=POOL_SIZE,
    max_edits=MAX_EDITS,
    max_options_per_position=MAX_OPTIONS,
    crossover_rate=CROSSOVER_RATE,
    max_attempts=MAX_ATTEMPTS,
):
    factual = tuple(factual)
    rng = random.Random(int(rng_seed))

    seen = {factual}
    pool = [factual]

    # Deterministic one-edit shell:
    # same mutation options as Full, only no flow_valid(...) rejection.
    for pos in range(len(factual)):
        for replacement in semantic_mutation_options(
            factual,
            pos,
            constraints,
            max_options=max_options_per_position,
        ):
            child = list(factual)
            child[pos] = replacement
            child = tuple(child)

            if child in seen:
                continue

            edits = sequence_edit_count(
                factual,
                child,
            )

            if edits <= 0 or edits > max_edits:
                continue

            # INTENTIONAL ABLATION:
            # no flow_valid(child, constraints["edge_count"]) filter here.

            valid = True

            for old, new in zip(
                factual,
                child,
            ):
                if old in (
                    TERMINAL_OFFER_ACTIVITIES
                    | DIRECT_PROXIES
                ):
                    if new != old:
                        valid = False
                        break

                elif new != old:
                    if new in FORBIDDEN_MUTATION_TARGETS:
                        valid = False
                        break

                    if not family_preserved(
                        old,
                        new,
                    ):
                        valid = False
                        break

            if not valid:
                continue

            seen.add(child)
            pool.append(child)

            if len(pool) >= pool_size:
                return pool

    attempts = 0

    while (
        len(pool) < pool_size
        and attempts < max_attempts
    ):
        attempts += 1

        parent = rng.choice(pool)
        child = None

        if (
            len(pool) >= 2
            and rng.random() < crossover_rate
        ):
            parent_b = rng.choice(pool)

            child = semantic_crossover(
                parent,
                parent_b,
                factual,
                constraints,
                rng,
            )
        else:
            child = tuple(parent)

        if child is None:
            continue

        mutated = semantic_mutate_once(
            child,
            constraints,
            rng,
            max_options=max_options_per_position,
        )

        if mutated is not None:
            child = mutated

        if child in seen:
            continue

        edits = sequence_edit_count(
            factual,
            child,
        )

        if edits <= 0 or edits > max_edits:
            continue

        # INTENTIONAL ABLATION:
        # no flow_valid(child, constraints["edge_count"]) filter here.

        valid = True

        for old, new in zip(
            factual,
            child,
        ):
            if old in (
                TERMINAL_OFFER_ACTIVITIES
                | DIRECT_PROXIES
            ):
                if new != old:
                    valid = False
                    break

            elif new != old:
                if new in FORBIDDEN_MUTATION_TARGETS:
                    valid = False
                    break

                if not family_preserved(
                    old,
                    new,
                ):
                    valid = False
                    break

        if not valid:
            continue

        seen.add(child)
        pool.append(child)

    return pool


# ---------------------------------------------------------------------
# Target metrics
# ---------------------------------------------------------------------

def candidate_metrics(
    pool,
    labels,
    factual,
    factual_label,
    constraints,
):
    require(
        len(pool) == len(labels),
        "Pool / label length mismatch",
    )

    records = []

    for seq, label in zip(
        pool,
        labels,
    ):
        seq = tuple(seq)

        # Exclude the factual point from candidate statistics.
        if seq == tuple(factual):
            continue

        flip = (
            int(label)
            != int(factual_label)
        )

        fv = flow_valid(
            seq,
            constraints["edge_count"],
        )

        records.append({
            "flip": bool(flip),
            "flow_valid": bool(fv),
            "edit_count": int(
                sequence_edit_count(
                    factual,
                    seq,
                )
            ),
        })

    flips = [
        r
        for r in records
        if r["flip"]
    ]

    valid_flips = [
        r
        for r in flips
        if r["flow_valid"]
    ]

    return {
        "candidate_count": len(records),
        "flow_valid_candidate_count": sum(
            int(r["flow_valid"])
            for r in records
        ),
        "flow_valid_candidate_rate": (
            mean(
                int(r["flow_valid"])
                for r in records
            )
            if records
            else None
        ),
        "flip_candidate_count": len(flips),
        "has_any_flip": bool(flips),
        "process_valid_flip_count": len(
            valid_flips
        ),
        "process_valid_flip_rate_among_flips": (
            len(valid_flips) / len(flips)
            if flips
            else None
        ),
        "has_process_valid_flip": bool(
            valid_flips
        ),
        "min_edit_any_flip": (
            min(
                r["edit_count"]
                for r in flips
            )
            if flips
            else None
        ),
        "min_edit_process_valid_flip": (
            min(
                r["edit_count"]
                for r in valid_flips
            )
            if valid_flips
            else None
        ),
    }


def summarize_rows(rows, variant):
    sub = [
        r
        for r in rows
        if r["variant"] == variant
    ]

    candidate_total = sum(
        int(r["candidate_count"])
        for r in sub
    )

    candidate_valid = sum(
        int(r["flow_valid_candidate_count"])
        for r in sub
    )

    flip_total = sum(
        int(r["flip_candidate_count"])
        for r in sub
    )

    process_valid_flip_total = sum(
        int(r["process_valid_flip_count"])
        for r in sub
    )

    return {
        "variant": variant,
        "target_count": len(sub),
        "candidate_count": candidate_total,
        "flow_valid_candidate_count": candidate_valid,
        "flow_valid_candidate_rate": (
            candidate_valid / candidate_total
            if candidate_total > 0
            else None
        ),
        "targets_with_any_flip": sum(
            int(r["has_any_flip"])
            for r in sub
        ),
        "GSR_any_flip": mean(
            int(r["has_any_flip"])
            for r in sub
        ),
        "targets_with_process_valid_flip": sum(
            int(r["has_process_valid_flip"])
            for r in sub
        ),
        "process_valid_target_rate": mean(
            int(r["has_process_valid_flip"])
            for r in sub
        ),
        "flip_candidate_count": flip_total,
        "process_valid_flip_count": process_valid_flip_total,
        "process_valid_flip_rate_among_flips": (
            process_valid_flip_total / flip_total
            if flip_total > 0
            else None
        ),
        "temporal_valid_target_rate": mean(
            int(r["temporal_valid"])
            for r in sub
        ),
        "mean_min_edit_any_flip": mean(
            r["min_edit_any_flip"]
            for r in sub
        ),
        "mean_min_edit_process_valid_flip": mean(
            r["min_edit_process_valid_flip"]
            for r in sub
        ),
        "mean_pool_size": mean(
            r["pool_size"]
            for r in sub
        ),
    }


def grouped_summary(rows):
    out = []

    for variant in (
        "Full_ETCE",
        "w/o_Process_Validity_Filter",
    ):
        for model in MODELS:
            for prefix in PREFIXES:
                sub = [
                    r
                    for r in rows
                    if r["variant"] == variant
                    and r["model"] == model
                    and int(r["prefix"]) == int(prefix)
                ]

                if not sub:
                    continue

                s = summarize_rows(
                    sub,
                    variant,
                )

                out.append({
                    "variant": variant,
                    "model": model,
                    "prefix": int(prefix),
                    **{
                        k: v
                        for k, v in s.items()
                        if k != "variant"
                    },
                })

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    required = [
        SELECTION,
        BANK_VALIDATION,
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json",
        RESULTS3 / "constraints/process_constraints.json",
    ]

    missing = [
        str(p)
        for p in required
        if not p.is_file()
    ]

    require(
        not missing,
        "Missing required input(s):\n"
        + "\n".join(missing),
    )

    bank_validation = read_json(
        BANK_VALIDATION
    )

    require(
        bank_validation.get("status") == "passed",
        "Formal multi-rule bank validation not PASSED",
    )

    with SELECTION.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        selection_rows = list(
            csv.DictReader(f)
        )

    require(
        selection_rows,
        f"Formal TEST selection CSV is empty: {SELECTION}",
    )

    require(
        "case_id"
        in selection_rows[0],
        (
            "Formal TEST selection CSV has no "
            f"'case_id' column: {SELECTION}"
        ),
    )

    case_ids = [
        str(r["case_id"]).strip()
        for r in selection_rows
    ]

    require(
        len(case_ids) == 100,
        (
            "Expected 100 rows in formal TEST "
            f"selection CSV, found {len(case_ids)}"
        ),
    )

    require(
        len(set(case_ids)) == 100,
        (
            "Formal TEST selection CSV must contain "
            "100 unique case_id values"
        ),
    )

    full_cache = load_full_cache()

    require(
        len(full_cache) == 1600,
        f"Expected 1600 formal cache targets, found {len(full_cache)}",
    )

    cache_case_ids = {
        key[0]
        for key in full_cache
    }

    require(
        cache_case_ids == set(case_ids),
        (
            "Frozen selection CSV case IDs do not "
            "match the 1600-target formal cache"
        ),
    )

    cases, _ = load_frozen_common_cohort(
        ROOT
    )

    case_map = {
        c.case_id: c
        for c in cases
    }

    require(
        all(
            cid in case_map
            for cid in case_ids
        ),
        "A formal selected case is missing from frozen cohort",
    )

    constraints = load_process_constraints(
        RESULTS3
    )

    canonical = read_json(
        RESULTS3
        / "canonical_blackboxes/canonical_manifest.json"
    )

    print("=" * 118)
    print("ETCE PHASE 4D-13A PROCESS-VALIDITY FILTER ABLATION")
    print("=" * 118)
    print("Full ETCE                  : training-DFG flow-validity filter ON")
    print("w/o Process-validity Filter: same generator, DFG rejection removed")
    print("Same case/model/prefix set : True")
    print("Same local seed             : True")
    print("Same semantic restrictions  : True")
    print("Same max edits              : 3")
    print("Same timestamps             : factual timestamps preserved")
    print("Temporal recourse claimed   : False")
    print("PN conformance evaluated    : False (separate experiment)")
    print("Protocol retuning           : False")
    print()

    detail = []
    target_no = 0

    for model in MODELS:
        for prefix in PREFIXES:
            predictor = CanonicalPredictor(
                ROOT,
                model,
                prefix,
                canonical["items"][model][str(prefix)],
            )

            for case_id in case_ids:
                target_no += 1
                key = (
                    case_id,
                    model,
                    int(prefix),
                )

                require(
                    key in full_cache,
                    f"Full formal cache missing target: {key}",
                )

                case = case_map[
                    case_id
                ]

                factual = tuple(
                    case.activities[
                        :prefix
                    ]
                )

                timestamps = tuple(
                    case.timestamps[
                        :prefix
                    ]
                )

                temporal_ok = temporal_monotone(
                    timestamps
                )

                # --------------------------------------------------
                # Full
                # --------------------------------------------------
                fc = full_cache[key]

                full_pool = [
                    tuple(x)
                    for x in fc[
                        "pool_sequences"
                    ]
                ]

                full_labels = np.asarray(
                    fc[
                        "pool_labels"
                    ],
                    dtype=int,
                )

                factual_idx = full_pool.index(
                    factual
                )

                full_factual_label = int(
                    full_labels[
                        factual_idx
                    ]
                )

                full_m = candidate_metrics(
                    full_pool,
                    full_labels,
                    factual,
                    full_factual_label,
                    constraints,
                )

                require(
                    full_m[
                        "flow_valid_candidate_rate"
                    ] is None
                    or abs(
                        full_m[
                            "flow_valid_candidate_rate"
                        ]
                        - 1.0
                    )
                    <= TOL,
                    (
                        "Full ETCE formal cache contains "
                        f"DFG-invalid candidate: {key}"
                    ),
                )

                # --------------------------------------------------
                # Ablated generator
                # --------------------------------------------------
                seed = local_seed(
                    model,
                    prefix,
                    case_id,
                )

                abl_pool = (
                    generate_without_process_validity_filter(
                        factual,
                        constraints,
                        rng_seed=seed,
                        pool_size=POOL_SIZE,
                        max_edits=MAX_EDITS,
                        max_options_per_position=MAX_OPTIONS,
                        crossover_rate=CROSSOVER_RATE,
                        max_attempts=MAX_ATTEMPTS,
                    )
                )

                abl_probs = np.asarray(
                    predictor.predict_proba(
                        abl_pool,
                        [
                            timestamps
                        ] * len(
                            abl_pool
                        ),
                    ),
                    dtype=float,
                )

                abl_labels = (
                    abl_probs >= 0.5
                ).astype(int)

                abl_fact_idx = (
                    abl_pool.index(
                        factual
                    )
                )

                abl_factual_label = int(
                    abl_labels[
                        abl_fact_idx
                    ]
                )

                require(
                    abl_factual_label
                    == full_factual_label,
                    (
                        "Factual black-box label changed "
                        f"between Full/Ablation: {key}"
                    ),
                )

                abl_m = candidate_metrics(
                    abl_pool,
                    abl_labels,
                    factual,
                    abl_factual_label,
                    constraints,
                )

                common = {
                    "case_id": case_id,
                    "model": model,
                    "prefix": int(
                        prefix
                    ),
                    "factual_label": int(
                        full_factual_label
                    ),
                    "temporal_valid": bool(
                        temporal_ok
                    ),
                }

                detail.append({
                    **common,
                    "variant": "Full_ETCE",
                    "pool_size": len(
                        full_pool
                    ),
                    **full_m,
                })

                detail.append({
                    **common,
                    "variant": "w/o_Process_Validity_Filter",
                    "pool_size": len(
                        abl_pool
                    ),
                    **abl_m,
                })

                if (
                    target_no % 100 == 0
                    or target_no == 1600
                ):
                    print(
                        f"[{target_no:04d}/1600] "
                        f"{model} P{prefix} {case_id} | "
                        f"FullFlow={full_m['flow_valid_candidate_rate']:.4f} | "
                        f"AblFlow={abl_m['flow_valid_candidate_rate']:.4f} | "
                        f"FullFlip={int(full_m['has_any_flip'])} | "
                        f"AblFlip={int(abl_m['has_any_flip'])}",
                        flush=True,
                    )

    require(
        len(detail) == 3200,
        f"Expected 3200 variant-target rows, found {len(detail)}",
    )

    full_rows = [
        r
        for r in detail
        if r["variant"] == "Full_ETCE"
    ]

    abl_rows = [
        r
        for r in detail
        if r["variant"] == "w/o_Process_Validity_Filter"
    ]

    require(
        len(full_rows)
        == len(abl_rows)
        == 1600,
        "Full/Ablation target counts differ",
    )

    def key(r):
        return (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
        )

    full_idx = {
        key(r): r
        for r in full_rows
    }

    abl_idx = {
        key(r): r
        for r in abl_rows
    }

    require(
        set(full_idx)
        == set(abl_idx),
        "Full/Ablation identities differ",
    )

    keys = sorted(
        full_idx
    )

    summary = [
        summarize_rows(
            detail,
            "Full_ETCE",
        ),
        summarize_rows(
            detail,
            "w/o_Process_Validity_Filter",
        ),
    ]

    by_model_prefix = grouped_summary(
        detail
    )

    continuous = []

    for metric in (
        "flow_valid_candidate_rate",
        "process_valid_flip_rate_among_flips",
        "min_edit_any_flip",
        "min_edit_process_valid_flip",
    ):
        usable = [
            k
            for k in keys
            if full_idx[k][metric] is not None
            and abl_idx[k][metric] is not None
        ]

        if not usable:
            continue

        continuous.append({
            "metric": metric,
            **paired_continuous(
                [
                    full_idx[k][metric]
                    for k in usable
                ],
                [
                    abl_idx[k][metric]
                    for k in usable
                ],
            ),
        })

    binary = []

    for metric in (
        "has_any_flip",
        "has_process_valid_flip",
        "temporal_valid",
    ):
        binary.append({
            "metric": metric,
            **paired_binary(
                [
                    full_idx[k][metric]
                    for k in keys
                ],
                [
                    abl_idx[k][metric]
                    for k in keys
                ],
            ),
        })

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        OUT
        / "process_filter_ablation_target_detail.csv",
        detail,
    )

    write_csv(
        OUT
        / "process_filter_ablation_summary.csv",
        summary,
    )

    write_csv(
        OUT
        / "process_filter_ablation_by_model_prefix.csv",
        by_model_prefix,
    )

    write_csv(
        OUT
        / "process_filter_ablation_paired_continuous.csv",
        continuous,
    )

    write_csv(
        OUT
        / "process_filter_ablation_paired_binary.csv",
        binary,
    )

    result = {
        "scope": (
            "ETCE Phase 4D-13A process-validity filter ablation"
        ),
        "status": "passed",
        "formal_test": True,
        "formal_targets": 1600,
        "ablation": {
            "full": (
                "training-derived DFG flow-validity filter ON"
            ),
            "without_process_validity_filter": (
                "same deterministic local generator with only the DFG rejection removed"
            ),
            "same_semantic_restrictions": True,
            "same_immutable_restrictions": True,
            "same_position_domain_logic": True,
            "same_max_edits": MAX_EDITS,
            "same_local_seed": True,
            "same_canonical_blackbox": True,
        },
        "temporal_scope": {
            "candidate_timestamps": (
                "factual timestamps reused exactly"
            ),
            "temporal_validity_audited": True,
            "temporal_recourse_claimed": False,
        },
        "pn_conformance_evaluated": False,
        "overall_summary": {
            r["variant"]: r
            for r in summary
        },
        "paired_continuous": continuous,
        "paired_binary": binary,
        "test_side_tuning": False,
        "protocol_retuning": False,
        "next_step": (
            "Run independent Petri-net conformance evaluation on Full vs "
            "w/o Process-validity Filter candidates, then integrate RQ5."
        ),
    }

    write_json(
        OUT
        / "process_filter_ablation_summary.json",
        result,
    )

    def fmt(v):
        return (
            "NA"
            if v is None
            else f"{float(v):.6f}"
        )

    lines = [
        "# ETCE Process-validity Filter Ablation",
        "",
        "Full ETCE keeps the training-derived DFG flow-validity filter.",
        "",
        "The ablation removes only that DFG rejection while retaining the same "
        "semantic/immutable restrictions, local seed, edit budget and black box.",
        "",
        "Temporal validity here is an audit only: activity-substitution candidates "
        "reuse factual timestamps exactly, so this does not establish temporal recourse.",
        "",
        "Petri-net conformance is deferred to an independent evaluation.",
        "",
        "## Overall",
        "",
        "| Variant | Targets | Candidates | Flow-valid candidate rate | GSR(any flip) | Process-valid target rate | Process-valid flip rate | Temporal-valid target rate | Mean min edit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in summary:
        lines.append(
            f"| {r['variant']} | "
            f"{r['target_count']} | "
            f"{r['candidate_count']} | "
            f"{fmt(r['flow_valid_candidate_rate'])} | "
            f"{fmt(r['GSR_any_flip'])} | "
            f"{fmt(r['process_valid_target_rate'])} | "
            f"{fmt(r['process_valid_flip_rate_among_flips'])} | "
            f"{fmt(r['temporal_valid_target_rate'])} | "
            f"{fmt(r['mean_min_edit_any_flip'])} |"
        )

    (
        OUT
        / "PHASE4D_PROCESS_VALIDITY_FILTER_ABLATION_REPORT.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 118)
    print("ETCE PHASE 4D-13A PROCESS-VALIDITY FILTER ABLATION PASSED")
    print("=" * 118)

    for r in summary:
        print(
            f"{r['variant']:30s} | "
            f"Targets={r['target_count']} | "
            f"Candidates={r['candidate_count']} | "
            f"FlowValid={r['flow_valid_candidate_rate']:.6f} | "
            f"GSR={r['GSR_any_flip']:.6f} | "
            f"ProcessValidTarget={r['process_valid_target_rate']:.6f} | "
            f"ProcessValidFlip={fmt(r['process_valid_flip_rate_among_flips'])} | "
            f"TemporalValid={r['temporal_valid_target_rate']:.6f} | "
            f"MinEdit={fmt(r['mean_min_edit_any_flip'])}"
        )

    print()
    print("[Paired continuous Full minus Ablation]")

    for r in continuous:
        print(
            f"{r['metric']:38s} | "
            f"Full={r['full_mean']:.6f} | "
            f"Abl={r['ablated_mean']:.6f} | "
            f"Delta={r['delta_full_minus_ablated']:+.6f} | "
            f"CI=[{r['bootstrap95_lo']:+.6f},"
            f"{r['bootstrap95_hi']:+.6f}] | "
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
    print("PN conformance evaluated: False")
    print("Temporal recourse claimed: False")
    print("Protocol retuning performed: False")
    print("NEXT: independent Petri-net conformance evaluation.")


if __name__ == "__main__":
    main()
