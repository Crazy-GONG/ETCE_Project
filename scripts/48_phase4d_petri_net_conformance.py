#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-13C / Script 48
Independent training-derived Petri-net conformance evaluation.

Purpose
-------
Complete RQ5 process-validity evaluation after:
- 47A: terminal flow_valid check redundancy audit
- 47B: process-aware proposal constraint ablation

This script independently evaluates actual prediction-flipping candidate traces
against Petri nets discovered ONLY from TRAIN cases.

Design
------
For each frozen prefix endpoint k in {5,10,15,20}:
1) take TRAIN cases only;
2) truncate every eligible TRAIN trace to its first k activities;
3) discover a prefix-specific Petri net with standard Inductive Miner,
   noise_threshold = 0.0 (fixed; no validation/test tuning);
4) evaluate Full ETCE and the 47B relaxed proposal on that same PN.

Why prefix-specific PN?
-----------------------
Formal explanations are prefix-specific. Discovering a PN from TRAIN prefixes of
the same length avoids penalizing a valid observed prefix merely because it does
not reach the final marking of a full-trace model.

Primary PN metrics
------------------
- token-replay trace fitness (continuous)
- perfectly fitting CF-candidate rate (trace_is_fit)
- target rate with >=1 perfectly fitting prediction-flipping candidate
- mean / best PN fitness per target among prediction-flipping candidates

Important boundaries
--------------------
- PN discovery uses TRAIN only.
- No formal TEST data define the PN.
- No noise threshold or conformance threshold is tuned.
- Perfect PN fit uses PM4Py token-replay trace_is_fit directly.
- Candidate generation is not changed.
- Temporal recourse is NOT claimed.
- Actionability is NOT claimed.

Dependency
----------
PM4Py is required. If missing:
    python -m pip install pm4py

Run
---
    cd D:/ETCE
    python -u ./scripts/48_phase4d_petri_net_conformance.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".deps"

if sys.platform == "win32" and LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

sys.path.insert(0, str(ROOT / "src"))

import numpy as np

try:
    import pm4py
    from pm4py.objects.log.obj import EventLog, Trace, Event
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
except Exception as exc:
    raise SystemExit(
        "\nPM4Py is required for Script 48.\n"
        "Install it with:\n\n"
        "    python -m pip install pm4py\n\n"
        f"Original import error: {exc}\n"
    )

from etce.phase2.data import PREFIXES, load_frozen_common_cohort
from etce.phase3.core import (
    CanonicalPredictor,
    flow_valid,
    load_process_constraints,
    sequence_edit_count,
)
from etce.utils.io import read_json


# =====================================================================
# Reuse EXACT 47B relaxed generator
# =====================================================================

SCRIPT47B = (
    ROOT
    / "scripts"
    / "47b_phase4d_process_aware_proposal_ablation.py"
)

if not SCRIPT47B.is_file():
    raise SystemExit(
        f"Required Script 47B not found: {SCRIPT47B}"
    )

_spec = importlib.util.spec_from_file_location(
    "etce_script47b",
    SCRIPT47B,
)
m47 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(m47)


# =====================================================================
# Paths / constants
# =====================================================================

RESULTS3 = ROOT / "results/phase3"

FORMAL = ROOT / "results/phase4/phase4d_formal_test"
BANK = FORMAL / "formal_test_multirule_bank"

OUT47B = (
    FORMAL
    / "formal_test_process_aware_proposal_ablation"
)

SUMMARY47B = (
    OUT47B
    / "proposal_ablation_summary.json"
)

OUT = (
    FORMAL
    / "formal_test_petri_net_conformance"
)

PN_DIR = OUT / "pn_models"

MODELS = (
    "RF",
    "XGBoost",
    "MLP",
    "LSTM",
)

NOISE_THRESHOLD = 0.0
TOKEN_REPLAY_BATCH = 2000

TOL = 1e-12


# =====================================================================
# Generic helpers
# =====================================================================

def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def write_json(path, obj):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_csv(path, rows):
    rows = list(rows)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    fields = []
    seen = set()

    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


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


def exact_binom_two_sided(k, n):
    if n <= 0:
        return 1.0

    k = int(k)
    n = int(n)

    lo = min(
        k,
        n - k,
    )

    p = sum(
        math.comb(n, i)
        for i in range(lo + 1)
    ) / (2 ** n)

    return min(
        1.0,
        2.0 * p,
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


def paired_continuous(
    full_vals,
    abl_vals,
):
    require(
        len(full_vals)
        == len(abl_vals),
        "Paired continuous length mismatch",
    )

    diffs = [
        float(f) - float(a)
        for f, a in zip(
            full_vals,
            abl_vals,
        )
    ]

    test = wilcoxon_or_sign(
        diffs
    )

    return {
        "N": len(diffs),
        "full_mean": mean(
            full_vals
        ),
        "ablated_mean": mean(
            abl_vals
        ),
        "delta_full_minus_ablated": mean(
            diffs
        ),
        "median_delta": median(
            diffs
        ),
        "full_wins": sum(
            d > TOL
            for d in diffs
        ),
        "ties": sum(
            abs(d) <= TOL
            for d in diffs
        ),
        "ablated_wins": sum(
            d < -TOL
            for d in diffs
        ),
        "paired_test": test[
            "test"
        ],
        "p_value": test[
            "p_value"
        ],
    }


def paired_binary(
    full_vals,
    abl_vals,
):
    require(
        len(full_vals)
        == len(abl_vals),
        "Paired binary length mismatch",
    )

    f0a1 = 0
    f1a0 = 0
    agree11 = 0
    agree00 = 0

    for f, a in zip(
        full_vals,
        abl_vals,
    ):
        f = int(bool(f))
        a = int(bool(a))

        if f == 1 and a == 1:
            agree11 += 1

        elif f == 0 and a == 0:
            agree00 += 1

        elif f == 0 and a == 1:
            f0a1 += 1

        else:
            f1a0 += 1

    discordant = (
        f0a1
        + f1a0
    )

    return {
        "N": len(full_vals),
        "full_rate": mean(
            int(bool(v))
            for v in full_vals
        ),
        "ablated_rate": mean(
            int(bool(v))
            for v in abl_vals
        ),
        "delta_full_minus_ablated": (
            mean(
                int(bool(v))
                for v in full_vals
            )
            - mean(
                int(bool(v))
                for v in abl_vals
            )
        ),
        "agree_11": agree11,
        "agree_00": agree00,
        "full0_ablated1": f0a1,
        "full1_ablated0": f1a0,
        "discordant": discordant,
        "test": "exact_mcnemar_binomial_two_sided",
        "p_value": exact_binom_two_sided(
            f1a0,
            discordant,
        ),
    }


# =====================================================================
# Event-log / PN helpers
# =====================================================================

def sequence_to_trace(
    seq,
    case_id=None,
):
    tr = Trace()

    if case_id is not None:
        tr.attributes[
            "concept:name"
        ] = str(
            case_id
        )

    for act in seq:
        tr.append(
            Event({
                "concept:name": str(
                    act
                )
            })
        )

    return tr


def build_train_prefix_log(
    cohort,
    prefix,
):
    """
    TRAIN only.

    Each first-k prefix is treated as a complete trace in the prefix-specific
    evaluation model.
    """
    log = EventLog()
    n = 0

    for case in cohort:
        if case.split != "train":
            continue

        if len(
            case.activities
        ) < int(prefix):
            continue

        seq = tuple(
            case.activities[
                :int(prefix)
            ]
        )

        log.append(
            sequence_to_trace(
                seq,
                case.case_id,
            )
        )

        n += 1

    require(
        n > 0,
        f"No TRAIN traces for prefix {prefix}",
    )

    return log


def discover_prefix_pn(
    train_log,
    prefix,
):
    """
    Prefer PM4Py high-level Inductive Miner API.
    Fallback to low-level IM + process-tree converter for compatibility.
    """
    try:
        net, im, fm = (
            pm4py.discover_petri_net_inductive(
                train_log,
                noise_threshold=NOISE_THRESHOLD,
            )
        )

    except Exception:
        from pm4py.algo.discovery.inductive import algorithm as inductive_miner
        from pm4py.objects.conversion.process_tree import converter as pt_converter

        tree = inductive_miner.apply(
            train_log
        )

        net, im, fm = (
            pt_converter.apply(
                tree
            )
        )

    require(
        net is not None,
        f"PN discovery failed for P{prefix}",
    )

    PN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pnml_path = (
        PN_DIR
        / f"training_prefix_{prefix}.pnml"
    )

    try:
        pm4py.write_pnml(
            net,
            im,
            fm,
            str(pnml_path),
        )
    except Exception:
        # Export failure must not invalidate conformance itself.
        pnml_path = None

    return (
        net,
        im,
        fm,
        pnml_path,
    )


def token_replay_unique_sequences(
    seqs,
    net,
    im,
    fm,
    label,
):
    """
    Returns:
        {tuple(sequence): diagnostics}
    """
    unique = sorted(
        set(
            tuple(
                str(a)
                for a in seq
            )
            for seq in seqs
        )
    )

    out = {}

    total = len(
        unique
    )

    for start in range(
        0,
        total,
        TOKEN_REPLAY_BATCH,
    ):
        batch = unique[
            start:
            start + TOKEN_REPLAY_BATCH
        ]

        log = EventLog(
            [
                sequence_to_trace(
                    seq
                )
                for seq in batch
            ]
        )

        replay = token_replay.apply(
            log,
            net,
            im,
            fm,
        )

        require(
            len(replay)
            == len(batch),
            "Token-replay output length mismatch",
        )

        for seq, diag in zip(
            batch,
            replay,
        ):
            fitness = diag.get(
                "trace_fitness"
            )

            is_fit = diag.get(
                "trace_is_fit"
            )

            require(
                fitness is not None,
                "Token replay missing trace_fitness",
            )

            if is_fit is None:
                # Defensive exact criterion.
                is_fit = (
                    abs(
                        float(fitness)
                        - 1.0
                    )
                    <= TOL
                )

            out[
                tuple(seq)
            ] = {
                "trace_fitness": float(
                    fitness
                ),
                "trace_is_fit": bool(
                    is_fit
                ),
                "missing_tokens": diag.get(
                    "missing_tokens"
                ),
                "remaining_tokens": diag.get(
                    "remaining_tokens"
                ),
                "consumed_tokens": diag.get(
                    "consumed_tokens"
                ),
                "produced_tokens": diag.get(
                    "produced_tokens"
                ),
            }

        done = min(
            start
            + TOKEN_REPLAY_BATCH,
            total,
        )

        print(
            f"[PN replay {label}] "
            f"{done}/{total} unique traces",
            flush=True,
        )

    return out


# =====================================================================
# Main
# =====================================================================

def main():
    require(
        SUMMARY47B.is_file(),
        (
            "47B summary missing. Run Script 47B first: "
            f"{SUMMARY47B}"
        ),
    )

    s47 = read_json(
        SUMMARY47B
    )

    require(
        s47.get(
            "status"
        ) == "passed",
        "Script 47B did not PASS",
    )

    require(
        s47.get(
            "protocol_retuning"
        ) is False,
        "47B reports protocol retuning",
    )

    selected_ids = (
        m47.read_selection()
    )

    full_cache = (
        m47.load_full_cache()
    )

    require(
        len(
            full_cache
        ) == 1600,
        "Formal cache target count != 1600",
    )

    cohort, _ = (
        load_frozen_common_cohort(
            ROOT
        )
    )

    case_map = {
        c.case_id: c
        for c in cohort
    }

    _, mutable_vocab = (
        m47.build_train_vocab(
            cohort
        )
    )

    canonical = read_json(
        RESULTS3
        / "canonical_blackboxes/canonical_manifest.json"
    )

    constraints = (
        load_process_constraints(
            RESULTS3
        )
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    PN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 118)
    print("ETCE PHASE 4D-13C TRAINING-DERIVED PETRI-NET CONFORMANCE")
    print("=" * 118)
    print("PN discovery          : Inductive Miner")
    print("PN discovery data     : TRAIN only")
    print("PN type               : prefix-specific P5/P10/P15/P20")
    print("Inductive noise       : 0.0 (fixed)")
    print("Conformance           : PM4Py token replay")
    print("Primary traces        : prediction-flipping candidates")
    print("Perfect-fit criterion : PM4Py trace_is_fit")
    print("Same 47B generator    : imported directly from Script 47B")
    print("TEST-side PN tuning   : False")
    print("Protocol retuning     : False")
    print()

    # -----------------------------------------------------------------
    # 1. Discover prefix-specific TRAIN-only PN models.
    # -----------------------------------------------------------------
    pn_models = {}
    discovery_rows = []

    for prefix in PREFIXES:
        print(
            f"[PN discovery] Prefix {prefix} ...",
            flush=True,
        )

        train_log = (
            build_train_prefix_log(
                cohort,
                prefix,
            )
        )

        net, im, fm, pnml = (
            discover_prefix_pn(
                train_log,
                prefix,
            )
        )

        pn_models[
            int(prefix)
        ] = (
            net,
            im,
            fm,
        )

        discovery_rows.append({
            "prefix": int(
                prefix
            ),
            "train_trace_count": len(
                train_log
            ),
            "places": len(
                net.places
            ),
            "transitions": len(
                net.transitions
            ),
            "arcs": len(
                net.arcs
            ),
            "noise_threshold": (
                NOISE_THRESHOLD
            ),
            "pnml_path": (
                ""
                if pnml is None
                else str(pnml)
            ),
        })

        print(
            f"[PN discovery] P{prefix} | "
            f"train={len(train_log)} | "
            f"places={len(net.places)} | "
            f"transitions={len(net.transitions)} | "
            f"arcs={len(net.arcs)}",
            flush=True,
        )

    # -----------------------------------------------------------------
    # 2. Regenerate Full + 47B relaxed flipping candidates.
    # -----------------------------------------------------------------
    target_raw = []
    unique_by_prefix = defaultdict(
        set
    )

    target_no = 0

    for model in MODELS:
        for prefix in PREFIXES:
            predictor = (
                CanonicalPredictor(
                    ROOT,
                    model,
                    prefix,
                    canonical[
                        "items"
                    ][model][
                        str(prefix)
                    ],
                )
            )

            for case_id in selected_ids:
                target_no += 1

                key = (
                    case_id,
                    model,
                    int(prefix),
                )

                fc = full_cache[
                    key
                ]

                case = case_map[
                    case_id
                ]

                factual = tuple(
                    str(a)
                    for a in case.activities[
                        :prefix
                    ]
                )

                timestamps = tuple(
                    case.timestamps[
                        :prefix
                    ]
                )

                full_pool = [
                    tuple(
                        str(a)
                        for a in seq
                    )
                    for seq in fc[
                        "pool_sequences"
                    ]
                ]

                full_labels = np.asarray(
                    fc[
                        "pool_labels"
                    ],
                    dtype=int,
                )

                require(
                    factual in full_pool,
                    f"Factual missing: {key}",
                )

                fact_idx = (
                    full_pool.index(
                        factual
                    )
                )

                factual_label = int(
                    full_labels[
                        fact_idx
                    ]
                )

                full_flips = [
                    seq
                    for seq, label
                    in zip(
                        full_pool,
                        full_labels,
                    )
                    if seq != factual
                    and int(label)
                    != factual_label
                ]

                seed = m47.local_seed(
                    model,
                    prefix,
                    case_id,
                )

                relaxed_pool = (
                    m47.generate_relaxed_pool(
                        factual,
                        mutable_vocab,
                        rng_seed=seed,
                        target_pool_size=len(
                            full_pool
                        ),
                    )
                )

                relaxed_probs = np.asarray(
                    predictor.predict_proba(
                        relaxed_pool,
                        [
                            timestamps
                        ] * len(
                            relaxed_pool
                        ),
                    ),
                    dtype=float,
                )

                relaxed_labels = (
                    relaxed_probs
                    >= 0.5
                ).astype(int)

                relaxed_fact_idx = (
                    relaxed_pool.index(
                        factual
                    )
                )

                relaxed_factual_label = int(
                    relaxed_labels[
                        relaxed_fact_idx
                    ]
                )

                require(
                    relaxed_factual_label
                    == factual_label,
                    (
                        "Factual black-box prediction changed "
                        f"for {key}"
                    ),
                )

                relaxed_flips = [
                    tuple(
                        str(a)
                        for a in seq
                    )
                    for seq, label
                    in zip(
                        relaxed_pool,
                        relaxed_labels,
                    )
                    if tuple(seq) != factual
                    and int(label)
                    != factual_label
                ]

                # DFG diagnostics retained for interpretation.
                full_dfg_valid = [
                    bool(
                        flow_valid(
                            seq,
                            constraints[
                                "edge_count"
                            ],
                        )
                    )
                    for seq in full_flips
                ]

                relaxed_dfg_valid = [
                    bool(
                        flow_valid(
                            seq,
                            constraints[
                                "edge_count"
                            ],
                        )
                    )
                    for seq in relaxed_flips
                ]

                require(
                    all(
                        full_dfg_valid
                    ),
                    (
                        "Full ETCE flip candidate is "
                        f"DFG-invalid: {key}"
                    ),
                )

                unique_by_prefix[
                    int(prefix)
                ].update(
                    full_flips
                )

                unique_by_prefix[
                    int(prefix)
                ].update(
                    relaxed_flips
                )

                # Also score factual prefix as a reference.
                unique_by_prefix[
                    int(prefix)
                ].add(
                    factual
                )

                target_raw.append({
                    "case_id": case_id,
                    "model": model,
                    "prefix": int(
                        prefix
                    ),
                    "factual": factual,
                    "factual_label": int(
                        factual_label
                    ),
                    "Full_ETCE": {
                        "flips": full_flips,
                        "dfg_valid": full_dfg_valid,
                    },
                    "w/o_Process_Aware_Proposal": {
                        "flips": relaxed_flips,
                        "dfg_valid": relaxed_dfg_valid,
                    },
                })

                if (
                    target_no % 100
                    == 0
                    or target_no
                    == 1600
                ):
                    print(
                        f"[CF inventory {target_no:04d}/1600] "
                        f"{model} P{prefix} {case_id} | "
                        f"FullFlip={len(full_flips)} | "
                        f"AblFlip={len(relaxed_flips)}",
                        flush=True,
                    )

    # -----------------------------------------------------------------
    # 3. Conformance cache for unique traces.
    # -----------------------------------------------------------------
    fitness_cache = {}

    for prefix in PREFIXES:
        net, im, fm = (
            pn_models[
                int(prefix)
            ]
        )

        seqs = (
            unique_by_prefix[
                int(prefix)
            ]
        )

        print()
        print(
            f"[PN replay] P{prefix}: "
            f"{len(seqs)} unique factual/CF traces",
            flush=True,
        )

        fit = (
            token_replay_unique_sequences(
                seqs,
                net,
                im,
                fm,
                label=f"P{prefix}",
            )
        )

        for seq, diag in fit.items():
            fitness_cache[
                (
                    int(prefix),
                    tuple(seq),
                )
            ] = diag

    # -----------------------------------------------------------------
    # 4. Target-level metrics.
    # -----------------------------------------------------------------
    detail = []

    for row in target_raw:
        prefix = int(
            row[
                "prefix"
            ]
        )

        factual_diag = (
            fitness_cache[
                (
                    prefix,
                    tuple(
                        row[
                            "factual"
                        ]
                    ),
                )
            ]
        )

        for variant in (
            "Full_ETCE",
            "w/o_Process_Aware_Proposal",
        ):
            flips = row[
                variant
            ][
                "flips"
            ]

            dfg_valid = row[
                variant
            ][
                "dfg_valid"
            ]

            diags = [
                fitness_cache[
                    (
                        prefix,
                        tuple(seq),
                    )
                ]
                for seq in flips
            ]

            fitnesses = [
                float(
                    d[
                        "trace_fitness"
                    ]
                )
                for d in diags
            ]

            pn_fit_flags = [
                bool(
                    d[
                        "trace_is_fit"
                    ]
                )
                for d in diags
            ]

            pn_fit_dfg_valid = [
                bool(pnfit)
                and bool(dfg)
                for pnfit, dfg
                in zip(
                    pn_fit_flags,
                    dfg_valid,
                )
            ]

            detail.append({
                "case_id": row[
                    "case_id"
                ],
                "model": row[
                    "model"
                ],
                "prefix": prefix,
                "variant": variant,
                "factual_label": row[
                    "factual_label"
                ],
                "factual_pn_fitness": float(
                    factual_diag[
                        "trace_fitness"
                    ]
                ),
                "factual_pn_fit": bool(
                    factual_diag[
                        "trace_is_fit"
                    ]
                ),
                "flip_candidate_count": len(
                    flips
                ),
                "has_any_flip": bool(
                    flips
                ),
                "mean_pn_fitness_flips": (
                    mean(
                        fitnesses
                    )
                    if fitnesses
                    else None
                ),
                "median_pn_fitness_flips": (
                    median(
                        fitnesses
                    )
                    if fitnesses
                    else None
                ),
                "best_pn_fitness_flips": (
                    max(
                        fitnesses
                    )
                    if fitnesses
                    else None
                ),
                "pn_fit_flip_count": sum(
                    int(v)
                    for v in pn_fit_flags
                ),
                "pn_fit_flip_rate": (
                    mean(
                        int(v)
                        for v in pn_fit_flags
                    )
                    if pn_fit_flags
                    else None
                ),
                "has_pn_fit_flip": bool(
                    any(
                        pn_fit_flags
                    )
                ),
                "dfg_valid_flip_count": sum(
                    int(v)
                    for v in dfg_valid
                ),
                "dfg_valid_flip_rate": (
                    mean(
                        int(v)
                        for v in dfg_valid
                    )
                    if dfg_valid
                    else None
                ),
                "pn_and_dfg_fit_flip_count": sum(
                    int(v)
                    for v in pn_fit_dfg_valid
                ),
                "has_pn_and_dfg_fit_flip": bool(
                    any(
                        pn_fit_dfg_valid
                    )
                ),
            })

    # -----------------------------------------------------------------
    # 5. Summaries.
    # -----------------------------------------------------------------
    def summarize_variant(
        variant,
        rows,
    ):
        sub = [
            r
            for r in rows
            if r[
                "variant"
            ] == variant
        ]

        with_flips = [
            r
            for r in sub
            if r[
                "has_any_flip"
            ]
        ]

        total_flips = sum(
            int(
                r[
                    "flip_candidate_count"
                ]
            )
            for r in sub
        )

        total_pn_fit = sum(
            int(
                r[
                    "pn_fit_flip_count"
                ]
            )
            for r in sub
        )

        all_fitness_weighted = []

        # Reconstruct candidate-weighted fitness without a huge candidate CSV.
        for raw in target_raw:
            if (
                raw["model"],
                int(raw["prefix"]),
                raw["case_id"],
            ) not in {
                (
                    r["model"],
                    int(r["prefix"]),
                    r["case_id"],
                )
                for r in sub
            }:
                continue

            for seq in raw[
                variant
            ][
                "flips"
            ]:
                all_fitness_weighted.append(
                    fitness_cache[
                        (
                            int(
                                raw[
                                    "prefix"
                                ]
                            ),
                            tuple(seq),
                        )
                    ][
                        "trace_fitness"
                    ]
                )

        return {
            "variant": variant,
            "target_count": len(
                sub
            ),
            "targets_with_flip": len(
                with_flips
            ),
            "GSR": mean(
                int(
                    r[
                        "has_any_flip"
                    ]
                )
                for r in sub
            ),
            "flip_candidate_count": total_flips,
            "candidate_weighted_mean_pn_fitness": (
                mean(
                    all_fitness_weighted
                )
            ),
            "candidate_weighted_pn_fit_rate": (
                total_pn_fit
                / total_flips
                if total_flips
                > 0
                else None
            ),
            "target_mean_of_mean_pn_fitness": (
                mean(
                    r[
                        "mean_pn_fitness_flips"
                    ]
                    for r in with_flips
                )
            ),
            "target_mean_best_pn_fitness": (
                mean(
                    r[
                        "best_pn_fitness_flips"
                    ]
                    for r in with_flips
                )
            ),
            "target_rate_with_pn_fit_flip": mean(
                int(
                    r[
                        "has_pn_fit_flip"
                    ]
                )
                for r in sub
            ),
            "target_rate_with_pn_and_dfg_fit_flip": mean(
                int(
                    r[
                        "has_pn_and_dfg_fit_flip"
                    ]
                )
                for r in sub
            ),
            "mean_factual_pn_fitness": mean(
                r[
                    "factual_pn_fitness"
                ]
                for r in sub
            ),
            "factual_pn_fit_rate": mean(
                int(
                    r[
                        "factual_pn_fit"
                    ]
                )
                for r in sub
            ),
        }

    overall = [
        summarize_variant(
            "Full_ETCE",
            detail,
        ),
        summarize_variant(
            "w/o_Process_Aware_Proposal",
            detail,
        ),
    ]

    by_model_prefix = []

    for variant in (
        "Full_ETCE",
        "w/o_Process_Aware_Proposal",
    ):
        for model in MODELS:
            for prefix in PREFIXES:
                sub = [
                    r
                    for r in detail
                    if r[
                        "variant"
                    ] == variant
                    and r[
                        "model"
                    ] == model
                    and int(
                        r[
                            "prefix"
                        ]
                    ) == int(
                        prefix
                    )
                ]

                if not sub:
                    continue

                flips = [
                    r
                    for r in sub
                    if r[
                        "has_any_flip"
                    ]
                ]

                by_model_prefix.append({
                    "variant": variant,
                    "model": model,
                    "prefix": int(
                        prefix
                    ),
                    "target_count": len(
                        sub
                    ),
                    "GSR": mean(
                        int(
                            r[
                                "has_any_flip"
                            ]
                        )
                        for r in sub
                    ),
                    "target_rate_with_pn_fit_flip": mean(
                        int(
                            r[
                                "has_pn_fit_flip"
                            ]
                        )
                        for r in sub
                    ),
                    "mean_pn_fitness_flips": mean(
                        r[
                            "mean_pn_fitness_flips"
                        ]
                        for r in flips
                    ),
                    "mean_best_pn_fitness": mean(
                        r[
                            "best_pn_fitness_flips"
                        ]
                        for r in flips
                    ),
                })

    # -----------------------------------------------------------------
    # 6. Paired target-level stats.
    # -----------------------------------------------------------------
    def identity(r):
        return (
            r[
                "case_id"
            ],
            r[
                "model"
            ],
            int(
                r[
                    "prefix"
                ]
            ),
        )

    full_idx = {
        identity(r): r
        for r in detail
        if r[
            "variant"
        ] == "Full_ETCE"
    }

    abl_idx = {
        identity(r): r
        for r in detail
        if r[
            "variant"
        ] == "w/o_Process_Aware_Proposal"
    }

    require(
        set(full_idx)
        == set(abl_idx),
        "Full/ablation identities differ",
    )

    keys = sorted(
        full_idx
    )

    paired_cont = []

    for metric in (
        "mean_pn_fitness_flips",
        "best_pn_fitness_flips",
        "pn_fit_flip_rate",
    ):
        usable = [
            k
            for k in keys
            if full_idx[k][metric]
            is not None
            and abl_idx[k][metric]
            is not None
        ]

        paired_cont.append({
            "metric": metric,
            **paired_continuous(
                [
                    full_idx[k][
                        metric
                    ]
                    for k in usable
                ],
                [
                    abl_idx[k][
                        metric
                    ]
                    for k in usable
                ],
            ),
        })

    paired_bin = []

    for metric in (
        "has_any_flip",
        "has_pn_fit_flip",
        "has_pn_and_dfg_fit_flip",
    ):
        paired_bin.append({
            "metric": metric,
            **paired_binary(
                [
                    full_idx[k][
                        metric
                    ]
                    for k in keys
                ],
                [
                    abl_idx[k][
                        metric
                    ]
                    for k in keys
                ],
            ),
        })

    # -----------------------------------------------------------------
    # 7. Save outputs.
    # -----------------------------------------------------------------
    write_csv(
        OUT
        / "pn_discovery_summary.csv",
        discovery_rows,
    )

    write_csv(
        OUT
        / "pn_conformance_target_detail.csv",
        detail,
    )

    write_csv(
        OUT
        / "pn_conformance_overall_summary.csv",
        overall,
    )

    write_csv(
        OUT
        / "pn_conformance_by_model_prefix.csv",
        by_model_prefix,
    )

    write_csv(
        OUT
        / "pn_conformance_paired_continuous.csv",
        paired_cont,
    )

    write_csv(
        OUT
        / "pn_conformance_paired_binary.csv",
        paired_bin,
    )

    result = {
        "scope": (
            "ETCE Phase 4D-13C independent training-derived PN conformance"
        ),
        "status": "passed",
        "formal_test": True,
        "pn_discovery": {
            "algorithm": (
                "PM4Py Inductive Miner"
            ),
            "data_scope": (
                "TRAIN split only"
            ),
            "prefix_specific_models": True,
            "prefixes": [
                int(x)
                for x in PREFIXES
            ],
            "noise_threshold": (
                NOISE_THRESHOLD
            ),
            "test_used_for_discovery": False,
        },
        "conformance": {
            "algorithm": (
                "PM4Py token-based replay"
            ),
            "primary_population": (
                "prediction-flipping local candidate traces"
            ),
            "continuous_metric": (
                "trace_fitness"
            ),
            "perfect_fit_metric": (
                "trace_is_fit"
            ),
            "threshold_tuning": False,
        },
        "overall": {
            r[
                "variant"
            ]: r
            for r in overall
        },
        "paired_continuous": (
            paired_cont
        ),
        "paired_binary": (
            paired_bin
        ),
        "temporal_recourse_claimed": False,
        "actionability_claimed": False,
        "test_side_tuning": False,
        "protocol_retuning": False,
        "interpretation_guard": (
            "PN fitness is an independent process-model consistency metric. "
            "It does not by itself establish real-world actionability."
        ),
        "next_step": (
            "Integrate RQ5 DFG + temporal + PN findings, then proceed to "
            "sparsity/edit/proximity and runtime/stability experiments."
        ),
    }

    write_json(
        OUT
        / "pn_conformance_summary.json",
        result,
    )

    def fmt(v):
        if v is None:
            return "NA"

        return (
            f"{float(v):.6f}"
        )

    lines = [
        "# ETCE Training-derived Petri-net Conformance",
        "",
        "Petri nets are discovered independently from TRAIN prefixes using "
        "Inductive Miner with fixed noise threshold 0.0.",
        "",
        "Formal TEST data do not define or tune the process model.",
        "",
        "## Overall",
        "",
        "| Variant | GSR | CF candidates | Mean PN fitness | Perfect PN-fit candidate rate | Target with PN-fit CF | Target with PN+DFG-fit CF |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in overall:
        lines.append(
            f"| {r['variant']} | "
            f"{fmt(r['GSR'])} | "
            f"{r['flip_candidate_count']} | "
            f"{fmt(r['candidate_weighted_mean_pn_fitness'])} | "
            f"{fmt(r['candidate_weighted_pn_fit_rate'])} | "
            f"{fmt(r['target_rate_with_pn_fit_flip'])} | "
            f"{fmt(r['target_rate_with_pn_and_dfg_fit_flip'])} |"
        )

    (
        OUT
        / "PHASE4D_PETRI_NET_CONFORMANCE_REPORT.md"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Console.
    # -----------------------------------------------------------------
    print()
    print("=" * 118)
    print("ETCE PHASE 4D-13C TRAINING-DERIVED PETRI-NET CONFORMANCE PASSED")
    print("=" * 118)

    for r in overall:
        print(
            f"{r['variant']:30s} | "
            f"GSR={r['GSR']:.6f} | "
            f"CF={r['flip_candidate_count']} | "
            f"PNFitMean={r['candidate_weighted_mean_pn_fitness']:.6f} | "
            f"PerfectCF={r['candidate_weighted_pn_fit_rate']:.6f} | "
            f"PNValidTarget={r['target_rate_with_pn_fit_flip']:.6f} | "
            f"PN+DFGTarget={r['target_rate_with_pn_and_dfg_fit_flip']:.6f}"
        )

    print()
    print("[Paired continuous Full minus Ablation]")

    for r in paired_cont:
        print(
            f"{r['metric']:28s} | "
            f"N={r['N']} | "
            f"Full={r['full_mean']:.6f} | "
            f"Abl={r['ablated_mean']:.6f} | "
            f"Delta={r['delta_full_minus_ablated']:+.6f} | "
            f"W/T/L={r['full_wins']}/{r['ties']}/{r['ablated_wins']} | "
            f"p={r['p_value']:.6g}"
        )

    print()
    print("[Paired binary Full minus Ablation]")

    for r in paired_bin:
        print(
            f"{r['metric']:28s} | "
            f"Full={r['full_rate']:.6f} | "
            f"Abl={r['ablated_rate']:.6f} | "
            f"Delta={r['delta_full_minus_ablated']:+.6f} | "
            f"discordant={r['discordant']} | "
            f"p={r['p_value']:.6g}"
        )

    print()
    print(f"Output: {OUT}")
    print("PN discovery used TEST: False")
    print("PN threshold tuning: False")
    print("Actionability claimed: False")
    print("Protocol retuning performed: False")
    print(
        "NEXT: integrate RQ5, then sparsity/edit/proximity."
    )


if __name__ == "__main__":
    main()
