#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-15 / Script 50
Formal runtime / scalability benchmark for the CURRENT frozen ETCE protocol.

Why this benchmark
------------------
The old manuscript contained runtime numbers for an obsolete experimental
pipeline. This benchmark instead replays the exact frozen Phase4D hierarchical
D-S implementation on the current formal BPIC2017 protocol.

Measured scopes
---------------
1) Input loading time (rule bank + formal fusion cache).
2) Exact in-memory frozen D-S hierarchy on all formal TEST inputs:
       Stage1 -> Stage2 -> Stage3
   repeated several times after a validation/warm-up run.
3) Prefix-stratified in-memory runtime for P5/P10/P15/P20.
4) Incremental Python peak allocation during one exact full hierarchy replay
   using tracemalloc.

Important interpretation
------------------------
- Input loading and core fusion are reported separately.
- The core benchmark excludes CSV/JSON output serialization.
- It includes grouping, BPA combination, representative-rule selection, and
  held-out fused-rule re-evaluation, i.e. the substantive Phase4D fusion core.
- It does NOT include upstream black-box training or local-neighborhood /
  surrogate-bank construction.
- No TEST tuning or protocol modification is performed.
- Hardware/runtime metadata are saved so reported seconds are reproducible.

Correctness gate
----------------
Before timing, the script performs one full exact replay and checks every one
of the 12,839 Stage1/2/3 output rows against the validated frozen formal D-S
CSV (selected rule + main numeric metrics). Timing is only reported if this
gate passes.

Run
---
    cd D:/ETCE
    python -u ./scripts/50_phase4d_runtime_scalability.py
"""

from __future__ import annotations

import csv
import gc
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT37 = (
    ROOT
    / "scripts"
    / "37_phase4d_frozen_ds_formal_test_fusion_v2.py"
)

FORMAL = ROOT / "results/phase4/phase4d_formal_test"

RULES_PATH = (
    FORMAL
    / "formal_test_multirule_bank"
    / "canonical_rules_test.jsonl"
)

CACHE_DIR = (
    FORMAL
    / "formal_test_multirule_bank"
    / "fusion_eval_cache"
    / "targets"
)

AUDIT_PATH = (
    FORMAL
    / "formal_test_target_local_group_audit"
    / "formal_test_target_local_group_audit.json"
)

FROZEN_OUTPUT = (
    FORMAL
    / "formal_test_frozen_ds_fusion"
    / "phase4d_frozen_ds_stage_outputs.csv"
)

FROZEN_VALIDATION = (
    FORMAL
    / "formal_test_frozen_ds_fusion"
    / "validation_phase4d_frozen_ds_fusion.json"
)

OUT = (
    FORMAL
    / "formal_test_runtime_scalability"
)

REPEATS_FULL = 5
REPEATS_PREFIX = 5
TOL = 1e-12


# =====================================================================
# Basic utility
# =====================================================================

def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def read_json(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def read_csv(path):
    with Path(path).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
        )


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
        and math.isfinite(
            float(x)
        )
    ]

    return (
        float(
            sum(vals)
            / len(vals)
        )
        if vals
        else None
    )


def median(xs):
    vals = [
        float(x)
        for x in xs
        if x is not None
        and math.isfinite(
            float(x)
        )
    ]

    return (
        float(
            statistics.median(
                vals
            )
        )
        if vals
        else None
    )


def stdev(xs):
    vals = [
        float(x)
        for x in xs
        if x is not None
        and math.isfinite(
            float(x)
        )
    ]

    return (
        float(
            statistics.stdev(
                vals
            )
        )
        if len(vals) >= 2
        else 0.0
    )


def percentile(xs, p):
    vals = sorted(
        float(x)
        for x in xs
        if x is not None
        and math.isfinite(
            float(x)
        )
    )

    if not vals:
        return None

    if len(vals) == 1:
        return vals[0]

    z = (
        (len(vals) - 1)
        * float(p)
    )

    lo = int(
        math.floor(z)
    )

    hi = int(
        math.ceil(z)
    )

    if lo == hi:
        return vals[lo]

    frac = (
        z - lo
    )

    return float(
        vals[lo]
        * (1.0 - frac)
        + vals[hi]
        * frac
    )


def parse_bool(v):
    if isinstance(v, bool):
        return v

    return (
        str(v)
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
            "y",
        }
    )


def parse_float(v):
    if v is None:
        return None

    s = str(v).strip()

    if (
        s == ""
        or s.lower()
        in {
            "none",
            "null",
            "nan",
        }
    ):
        return None

    return float(s)


# =====================================================================
# Load exact Script 37 implementation
# =====================================================================

def load_script37():
    require(
        SCRIPT37.is_file(),
        (
            "Exact frozen implementation missing: "
            f"{SCRIPT37}"
        ),
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            "etce_phase4d_frozen_ds_runtime",
            SCRIPT37,
        )
    )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    require(
        spec.loader
        is not None,
        "Could not load Script 37",
    )

    spec.loader.exec_module(
        module
    )

    return module


m37 = load_script37()


# =====================================================================
# Exact hierarchy replay
# =====================================================================

def eligible_rules(
    rules,
    prefix=None,
):
    out = [
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
            and (
                prefix is None
                or int(
                    r["prefix"]
                )
                == int(prefix)
            )
        )
    ]

    return out


def replay_hierarchy(
    cf,
    cache,
    collect_rows=False,
):
    """
    Exact substantive hierarchy from Script 37.
    Returns output dictionaries + per-stage wall-clock time.
    """
    lam = (
        m37.FROZEN_LAMBDA
    )

    t_total_0 = (
        time.perf_counter()
    )

    # --------------------------------------------------------------
    # Stage 1 grouping + fusion
    # --------------------------------------------------------------
    t0 = time.perf_counter()

    s1_groups = defaultdict(
        list
    )

    for r in cf:
        s1_groups[
            m37.group_key_stage1(
                r
            )
        ].append(
            r
        )

    s1_outputs = {}
    stage_rows = []

    s1_pairwise = 0

    for key in sorted(
        s1_groups,
        key=lambda k: (
            k[0],
            m37.MODEL_ORDER.get(
                k[1],
                99,
            ),
            k[2],
            m37.SURROGATE_ORDER.get(
                k[3],
                99,
            ),
            k[4],
            k[5],
        ),
    ):
        source_rules = sorted(
            s1_groups[
                key
            ],
            key=lambda r: str(
                r[
                    "candidate_id"
                ]
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

        current = (
            m37.source_rule(
                source_rules[0],
                lam,
            )
        )

        for step, r in enumerate(
            source_rules[1:],
            1,
        ):
            right = (
                m37.source_rule(
                    r,
                    lam,
                )
            )

            current, _trace = (
                m37.pairwise_rule_fusion(
                    current,
                    right,
                    ctx,
                    lam,
                    "Stage1",
                    {
                        "lambda": float(
                            lam
                        ),
                        "case_id": (
                            case_id
                        ),
                        "model": model,
                        "prefix": int(
                            prefix
                        ),
                        "surrogate": (
                            surrogate
                        ),
                        "predicted_class": int(
                            pred
                        ),
                        "feature_signature": json.dumps(
                            list(sig)
                        ),
                        "step": int(
                            step
                        ),
                    },
                )
            )

            s1_pairwise += 1

        s1_outputs[
            key
        ] = current

        if collect_rows:
            stage_rows.append(
                m37.ev_to_row(
                    current,
                    lam,
                    "Stage1",
                    key,
                    len(
                        source_rules
                    ),
                )
            )

    t1 = time.perf_counter()

    # --------------------------------------------------------------
    # Stage 2 grouping + fusion
    # --------------------------------------------------------------
    s2_groups = defaultdict(
        list
    )

    for key, ev in (
        s1_outputs.items()
    ):
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

    s2_outputs = {}
    s2_pairwise = 0

    for key in sorted(
        s2_groups,
        key=lambda k: (
            k[0],
            m37.MODEL_ORDER.get(
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
            s2_groups[
                key
            ],
            key=lambda x: (
                m37.SURROGATE_ORDER.get(
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

        current = (
            items[0][1]
        )

        for step, (
            surrogate,
            right,
        ) in enumerate(
            items[1:],
            1,
        ):
            current, _trace = (
                m37.pairwise_rule_fusion(
                    current,
                    right,
                    ctx,
                    lam,
                    "Stage2",
                    {
                        "lambda": float(
                            lam
                        ),
                        "case_id": (
                            case_id
                        ),
                        "model": model,
                        "prefix": int(
                            prefix
                        ),
                        "surrogate": (
                            "cross-surrogate:"
                            f"{surrogate}"
                        ),
                        "predicted_class": int(
                            pred
                        ),
                        "feature_signature": json.dumps(
                            list(sig)
                        ),
                        "step": int(
                            step
                        ),
                    },
                )
            )

            s2_pairwise += 1

        s2_outputs[
            key
        ] = current

        if collect_rows:
            stage_rows.append(
                m37.ev_to_row(
                    current,
                    lam,
                    "Stage2",
                    key,
                    len(
                        items
                    ),
                )
            )

    t2 = time.perf_counter()

    # --------------------------------------------------------------
    # Stage 3 grouping + fusion
    # --------------------------------------------------------------
    s3_groups = defaultdict(
        list
    )

    for key, ev in (
        s2_outputs.items()
    ):
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

    s3_outputs = {}
    s3_pairwise = 0

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
            s3_groups[
                key
            ],
            key=lambda x: (
                m37.MODEL_ORDER.get(
                    x[0],
                    99,
                )
            ),
        )

        current = (
            items[0][1]
        )

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
                for m
                in models_so_far
            ]

            current, _trace = (
                m37.pairwise_rule_fusion(
                    current,
                    right,
                    contexts,
                    lam,
                    "Stage3",
                    {
                        "lambda": float(
                            lam
                        ),
                        "case_id": (
                            case_id
                        ),
                        "model": (
                            "cross-model:"
                            f"{model}"
                        ),
                        "prefix": int(
                            prefix
                        ),
                        "surrogate": None,
                        "predicted_class": int(
                            pred
                        ),
                        "feature_signature": json.dumps(
                            list(sig)
                        ),
                        "step": int(
                            step
                        ),
                    },
                )
            )

            s3_pairwise += 1

        s3_outputs[
            key
        ] = current

        if collect_rows:
            stage_rows.append(
                m37.ev_to_row(
                    current,
                    lam,
                    "Stage3",
                    key,
                    len(
                        items
                    ),
                )
            )

    t3 = time.perf_counter()

    total_seconds = (
        t3
        - t_total_0
    )

    return {
        "s1_outputs": (
            s1_outputs
        ),
        "s2_outputs": (
            s2_outputs
        ),
        "s3_outputs": (
            s3_outputs
        ),
        "stage_rows": (
            stage_rows
        ),
        "runtime": {
            "stage1_seconds": (
                t1 - t0
            ),
            "stage2_seconds": (
                t2 - t1
            ),
            "stage3_seconds": (
                t3 - t2
            ),
            "total_seconds": (
                total_seconds
            ),
        },
        "inventory": {
            "eligible_cf": len(
                cf
            ),
            "stage1_groups": len(
                s1_outputs
            ),
            "stage2_groups": len(
                s2_outputs
            ),
            "stage3_groups": len(
                s3_outputs
            ),
            "stage1_pairwise": (
                s1_pairwise
            ),
            "stage2_pairwise": (
                s2_pairwise
            ),
            "stage3_pairwise": (
                s3_pairwise
            ),
            "total_pairwise": (
                s1_pairwise
                + s2_pairwise
                + s3_pairwise
            ),
        },
    }


# =====================================================================
# Correctness gate against frozen output
# =====================================================================

def stage_row_key(r):
    stage = str(
        r["stage"]
    )

    sig = tuple(
        json.loads(
            r[
                "feature_signature"
            ]
        )
    )

    if stage == "Stage1":
        return (
            stage,
            str(
                r[
                    "case_id"
                ]
            ),
            str(
                r[
                    "model"
                ]
            ),
            int(
                float(
                    r[
                        "prefix"
                    ]
                )
            ),
            str(
                r[
                    "surrogate"
                ]
            ),
            int(
                float(
                    r[
                        "predicted_class"
                    ]
                )
            ),
            sig,
        )

    if stage == "Stage2":
        return (
            stage,
            str(
                r[
                    "case_id"
                ]
            ),
            str(
                r[
                    "model"
                ]
            ),
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
            sig,
        )

    if stage == "Stage3":
        return (
            stage,
            str(
                r[
                    "case_id"
                ]
            ),
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
            sig,
        )

    raise ValueError(
        stage
    )


def canonical_conditions(cell):
    if isinstance(
        cell,
        list,
    ):
        obj = cell
    else:
        obj = json.loads(
            str(cell)
        )

    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def validate_replay(
    reproduced_rows,
    frozen_rows,
):
    require(
        len(
            reproduced_rows
        )
        == 12839,
        (
            "Expected 12839 replay rows, "
            f"found {len(reproduced_rows)}"
        ),
    )

    require(
        len(
            frozen_rows
        )
        == 12839,
        (
            "Expected 12839 frozen rows, "
            f"found {len(frozen_rows)}"
        ),
    )

    rep = {
        stage_row_key(
            r
        ): r
        for r
        in reproduced_rows
    }

    fro = {
        stage_row_key(
            r
        ): r
        for r
        in frozen_rows
    }

    require(
        len(rep)
        == len(
            reproduced_rows
        ),
        "Duplicate replay row key",
    )

    require(
        len(fro)
        == len(
            frozen_rows
        ),
        "Duplicate frozen row key",
    )

    require(
        set(rep)
        == set(fro),
        "Replay/frozen stage identities differ",
    )

    max_numeric_diff = 0.0
    condition_mismatch = 0

    numeric_fields = (
        "n_r",
        "Fid_r",
        "coverage_eval",
        "rho_r",
        "R_r",
        "model_cf_validity_rate",
        "min_witness_edit",
        "last_K",
        "last_selected_mass",
        "last_ignorance_mass",
    )

    boolean_fields = (
        "cf_valid",
        "all_models_cf_valid",
    )

    for key in rep:
        a = rep[
            key
        ]
        b = fro[
            key
        ]

        if (
            canonical_conditions(
                a[
                    "selected_conditions"
                ]
            )
            != canonical_conditions(
                b[
                    "selected_conditions"
                ]
            )
        ):
            condition_mismatch += 1

        for field in numeric_fields:
            av = parse_float(
                a.get(
                    field
                )
            )

            bv = parse_float(
                b.get(
                    field
                )
            )

            if (
                av is None
                and bv is None
            ):
                continue

            require(
                av is not None
                and bv is not None,
                (
                    f"Null mismatch "
                    f"{field}: {key}"
                ),
            )

            d = abs(
                av - bv
            )

            max_numeric_diff = max(
                max_numeric_diff,
                d,
            )

            require(
                d <= 1e-12,
                (
                    f"Numeric mismatch "
                    f"{field} at {key}: "
                    f"{av} vs {bv}"
                ),
            )

        for field in boolean_fields:
            require(
                parse_bool(
                    a.get(
                        field
                    )
                )
                == parse_bool(
                    b.get(
                        field
                    )
                ),
                (
                    f"Boolean mismatch "
                    f"{field}: {key}"
                ),
            )

    require(
        condition_mismatch
        == 0,
        (
            "Selected-condition mismatch count: "
            f"{condition_mismatch}"
        ),
    )

    return {
        "matched_rows": len(
            rep
        ),
        "selected_condition_mismatch": (
            condition_mismatch
        ),
        "max_numeric_abs_diff": (
            max_numeric_diff
        ),
    }


# =====================================================================
# Benchmark helpers
# =====================================================================

def summarize_repetitions(
    label,
    prefix,
    inventory,
    runs,
):
    fields = (
        "stage1_seconds",
        "stage2_seconds",
        "stage3_seconds",
        "total_seconds",
    )

    row = {
        "scope": label,
        "prefix": (
            ""
            if prefix is None
            else int(prefix)
        ),
        "repeats": len(
            runs
        ),
        **inventory,
    }

    for field in fields:
        vals = [
            r[
                field
            ]
            for r in runs
        ]

        short = (
            field.replace(
                "_seconds",
                ""
            )
        )

        row[
            f"{short}_mean_seconds"
        ] = mean(
            vals
        )

        row[
            f"{short}_std_seconds"
        ] = stdev(
            vals
        )

        row[
            f"{short}_median_seconds"
        ] = median(
            vals
        )

        row[
            f"{short}_p90_seconds"
        ] = percentile(
            vals,
            0.90,
        )

    total_mean = row[
        "total_mean_seconds"
    ]

    row[
        "mean_ms_per_stage3_group"
    ] = (
        1000.0
        * total_mean
        / inventory[
            "stage3_groups"
        ]
        if inventory[
            "stage3_groups"
        ]
        > 0
        else None
    )

    row[
        "mean_ms_per_pairwise_fusion"
    ] = (
        1000.0
        * total_mean
        / inventory[
            "total_pairwise"
        ]
        if inventory[
            "total_pairwise"
        ]
        > 0
        else None
    )

    return row


def benchmark_scope(
    label,
    prefix,
    cf,
    cache,
    repeats,
):
    # One unrecorded warm-up.
    replay_hierarchy(
        cf,
        cache,
        collect_rows=False,
    )

    runs = []

    inventory = None

    for i in range(
        1,
        repeats + 1,
    ):
        gc.collect()

        result = (
            replay_hierarchy(
                cf,
                cache,
                collect_rows=False,
            )
        )

        runs.append(
            result[
                "runtime"
            ]
        )

        inventory = (
            result[
                "inventory"
            ]
        )

        print(
            f"[Benchmark {label}] "
            f"run {i}/{repeats} | "
            f"S1={result['runtime']['stage1_seconds']:.6f}s | "
            f"S2={result['runtime']['stage2_seconds']:.6f}s | "
            f"S3={result['runtime']['stage3_seconds']:.6f}s | "
            f"Total={result['runtime']['total_seconds']:.6f}s",
            flush=True,
        )

    return (
        summarize_repetitions(
            label,
            prefix,
            inventory,
            runs,
        ),
        runs,
    )


# =====================================================================
# Main
# =====================================================================

def main():
    required = (
        SCRIPT37,
        RULES_PATH,
        CACHE_DIR,
        AUDIT_PATH,
        FROZEN_OUTPUT,
        FROZEN_VALIDATION,
    )

    missing = [
        str(p)
        for p in required
        if not Path(p).exists()
    ]

    require(
        not missing,
        (
            "Missing required input(s):\n"
            + "\n".join(
                missing
            )
        ),
    )

    frozen_validation = (
        read_json(
            FROZEN_VALIDATION
        )
    )

    require(
        frozen_validation.get(
            "status"
        )
        == "passed",
        "Formal frozen D-S validation is not PASSED",
    )

    audit = read_json(
        AUDIT_PATH
    )

    require(
        audit.get(
            "status"
        )
        == "passed",
        "Formal grouping audit is not PASSED",
    )

    print("=" * 118)
    print("ETCE PHASE 4D-15 FORMAL RUNTIME / SCALABILITY BENCHMARK")
    print("=" * 118)
    print("Implementation source : 37_phase4d_frozen_ds_formal_test_fusion_v2.py")
    print("Benchmark core        : grouping + D-S fusion + held-out fused-rule re-evaluation")
    print("Disk output writing   : excluded from core timing")
    print("Black-box training    : excluded")
    print("Rule-bank generation  : excluded")
    print("Formal TEST tuning    : False")
    print("Protocol retuning     : False")
    print()

    # --------------------------------------------------------------
    # Input-load runtime.
    # --------------------------------------------------------------
    gc.collect()

    t0 = time.perf_counter()

    rules = (
        m37.read_jsonl(
            RULES_PATH
        )
    )

    t1 = time.perf_counter()

    cache = (
        m37.load_cache_index(
            CACHE_DIR
        )
    )

    t2 = time.perf_counter()

    cf = eligible_rules(
        rules
    )

    require(
        len(cf) == 6172,
        (
            "Expected 6172 eligible CF, "
            f"found {len(cf)}"
        ),
    )

    require(
        len(cache) == 1600,
        (
            "Expected 1600 cache targets, "
            f"found {len(cache)}"
        ),
    )

    load_row = {
        "rule_jsonl_load_seconds": (
            t1 - t0
        ),
        "cache_load_seconds": (
            t2 - t1
        ),
        "total_input_load_seconds": (
            t2 - t0
        ),
        "canonical_rule_count": (
            len(rules)
        ),
        "eligible_cf_count": (
            len(cf)
        ),
        "cache_target_count": (
            len(cache)
        ),
    }

    print(
        "[Input load] "
        f"rules={load_row['rule_jsonl_load_seconds']:.6f}s | "
        f"cache={load_row['cache_load_seconds']:.6f}s | "
        f"total={load_row['total_input_load_seconds']:.6f}s"
    )
    print()

    # --------------------------------------------------------------
    # Exact correctness replay (also warm-up).
    # --------------------------------------------------------------
    print(
        "[Correctness gate] replaying exact full hierarchy ..."
    )

    validation_replay = (
        replay_hierarchy(
            cf,
            cache,
            collect_rows=True,
        )
    )

    frozen_rows = (
        read_csv(
            FROZEN_OUTPUT
        )
    )

    correctness = (
        validate_replay(
            validation_replay[
                "stage_rows"
            ],
            frozen_rows,
        )
    )

    require(
        validation_replay[
            "inventory"
        ][
            "stage1_groups"
        ] == 5056,
        "Stage1 group count mismatch",
    )

    require(
        validation_replay[
            "inventory"
        ][
            "stage2_groups"
        ] == 4287,
        "Stage2 group count mismatch",
    )

    require(
        validation_replay[
            "inventory"
        ][
            "stage3_groups"
        ] == 3496,
        "Stage3 group count mismatch",
    )

    print(
        "[Correctness gate] PASSED | "
        f"matched={correctness['matched_rows']} | "
        f"max_abs_diff={correctness['max_numeric_abs_diff']:.3e}"
    )
    print()

    # --------------------------------------------------------------
    # Full benchmark.
    # --------------------------------------------------------------
    full_summary, full_runs = (
        benchmark_scope(
            "Formal_All_Prefixes",
            None,
            cf,
            cache,
            REPEATS_FULL,
        )
    )

    # --------------------------------------------------------------
    # Prefix-stratified benchmarks.
    # --------------------------------------------------------------
    prefix_summaries = []
    raw_runs = []

    for prefix in (
        5,
        10,
        15,
        20,
    ):
        cf_p = eligible_rules(
            rules,
            prefix=prefix,
        )

        cache_p = {
            k: v
            for k, v
            in cache.items()
            if int(
                k[2]
            ) == int(
                prefix
            )
        }

        require(
            len(cache_p)
            == 400,
            (
                f"Expected 400 cache targets for P{prefix}, "
                f"found {len(cache_p)}"
            ),
        )

        summary, runs = (
            benchmark_scope(
                f"P{prefix}",
                prefix,
                cf_p,
                cache_p,
                REPEATS_PREFIX,
            )
        )

        prefix_summaries.append(
            summary
        )

        for i, r in enumerate(
            runs,
            1,
        ):
            raw_runs.append({
                "scope": (
                    f"P{prefix}"
                ),
                "prefix": prefix,
                "run": i,
                **r,
            })

    for i, r in enumerate(
        full_runs,
        1,
    ):
        raw_runs.append({
            "scope": (
                "Formal_All_Prefixes"
            ),
            "prefix": "",
            "run": i,
            **r,
        })

    # --------------------------------------------------------------
    # Incremental Python memory peak.
    # --------------------------------------------------------------
    print()
    print(
        "[Memory] tracemalloc exact full hierarchy replay ..."
    )

    gc.collect()

    tracemalloc.start()

    mem_t0 = (
        time.perf_counter()
    )

    mem_result = (
        replay_hierarchy(
            cf,
            cache,
            collect_rows=False,
        )
    )

    mem_t1 = (
        time.perf_counter()
    )

    current_bytes, peak_bytes = (
        tracemalloc.get_traced_memory()
    )

    tracemalloc.stop()

    memory_row = {
        "measurement": (
            "incremental_python_allocations_during_fusion_core"
        ),
        "current_mb": (
            current_bytes
            / (1024 ** 2)
        ),
        "peak_mb": (
            peak_bytes
            / (1024 ** 2)
        ),
        "traced_run_seconds": (
            mem_t1 - mem_t0
        ),
        "note": (
            "tracemalloc reports Python-managed allocations created "
            "after tracing starts; input rule/cache memory loaded earlier "
            "is intentionally excluded."
        ),
    }

    print(
        f"[Memory] peak incremental Python allocation = "
        f"{memory_row['peak_mb']:.3f} MB"
    )

    # --------------------------------------------------------------
    # Runtime scaling descriptors.
    # --------------------------------------------------------------
    scaling = []

    base = (
        prefix_summaries[0]
    )

    for row in (
        prefix_summaries
    ):
        scaling.append({
            "prefix": row[
                "prefix"
            ],
            "eligible_cf": row[
                "eligible_cf"
            ],
            "stage1_groups": row[
                "stage1_groups"
            ],
            "stage2_groups": row[
                "stage2_groups"
            ],
            "stage3_groups": row[
                "stage3_groups"
            ],
            "total_pairwise": row[
                "total_pairwise"
            ],
            "mean_total_seconds": row[
                "total_mean_seconds"
            ],
            "runtime_ratio_vs_P5": (
                row[
                    "total_mean_seconds"
                ]
                / base[
                    "total_mean_seconds"
                ]
            ),
            "pairwise_ratio_vs_P5": (
                row[
                    "total_pairwise"
                ]
                / base[
                    "total_pairwise"
                ]
                if base[
                    "total_pairwise"
                ] > 0
                else None
            ),
            "eligible_cf_ratio_vs_P5": (
                row[
                    "eligible_cf"
                ]
                / base[
                    "eligible_cf"
                ]
                if base[
                    "eligible_cf"
                ] > 0
                else None
            ),
        })

    # --------------------------------------------------------------
    # Metadata and outputs.
    # --------------------------------------------------------------
    hardware = {
        "platform": (
            platform.platform()
        ),
        "system": (
            platform.system()
        ),
        "release": (
            platform.release()
        ),
        "machine": (
            platform.machine()
        ),
        "processor": (
            platform.processor()
        ),
        "logical_cpu_count": (
            os.cpu_count()
        ),
        "python_version": (
            platform.python_version()
        ),
        "python_implementation": (
            platform.python_implementation()
        ),
    }

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = [
        full_summary,
        *prefix_summaries,
    ]

    write_csv(
        OUT
        / "runtime_summary.csv",
        summary_rows,
    )

    write_csv(
        OUT
        / "runtime_raw_repetitions.csv",
        raw_runs,
    )

    write_csv(
        OUT
        / "runtime_prefix_scaling.csv",
        scaling,
    )

    write_csv(
        OUT
        / "runtime_input_load.csv",
        [
            load_row
        ],
    )

    write_csv(
        OUT
        / "runtime_memory.csv",
        [
            memory_row
        ],
    )

    result = {
        "scope": (
            "ETCE Phase 4D-15 current-protocol runtime/scalability"
        ),
        "status": "passed",
        "formal_test": True,
        "benchmark_definition": {
            "core_includes": [
                "Stage1/2/3 grouping",
                "support-aware BPA construction",
                "pairwise D-S combination",
                "representative fused-rule selection",
                "held-out fused-rule re-evaluation",
            ],
            "core_excludes": [
                "black-box training",
                "local-neighborhood generation",
                "surrogate fitting / rule-bank construction",
                "disk output serialization",
            ],
            "input_loading_reported_separately": True,
            "full_repeats": (
                REPEATS_FULL
            ),
            "prefix_repeats": (
                REPEATS_PREFIX
            ),
        },
        "correctness_gate": (
            correctness
        ),
        "input_load": (
            load_row
        ),
        "full_runtime": (
            full_summary
        ),
        "prefix_runtime": (
            prefix_summaries
        ),
        "prefix_scaling": (
            scaling
        ),
        "memory": (
            memory_row
        ),
        "hardware": (
            hardware
        ),
        "old_manuscript_runtime_reused": False,
        "test_side_tuning": False,
        "protocol_retuning": False,
        "interpretation_guard": (
            "Measured seconds characterize the current implementation "
            "on the recorded hardware. Prefix scaling is empirical over "
            "P5/P10/P15/P20 and should not be extrapolated as a formal "
            "asymptotic complexity proof."
        ),
        "next_step": (
            "Stability/diversity and consolidated hypothesis-driven "
            "ablation/statistical table."
        ),
    }

    write_json(
        OUT
        / "runtime_scalability_summary.json",
        result,
    )

    # Markdown report.
    lines = [
        "# ETCE Current-protocol Runtime / Scalability",
        "",
        "This benchmark replays the validated frozen Phase4D D-S core.",
        "",
        "The old manuscript runtime measurements are not reused.",
        "",
        "## Full formal runtime",
        "",
        "| Scope | CF | S1 groups | S2 groups | S3 groups | Pairwise fusion | S1 (s) | S2 (s) | S3 (s) | Total (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(v):
        if v is None:
            return "NA"
        return f"{float(v):.6f}"

    for r in summary_rows:
        lines.append(
            f"| {r['scope']} | "
            f"{r['eligible_cf']} | "
            f"{r['stage1_groups']} | "
            f"{r['stage2_groups']} | "
            f"{r['stage3_groups']} | "
            f"{r['total_pairwise']} | "
            f"{fmt(r['stage1_mean_seconds'])} | "
            f"{fmt(r['stage2_mean_seconds'])} | "
            f"{fmt(r['stage3_mean_seconds'])} | "
            f"{fmt(r['total_mean_seconds'])} |"
        )

    lines.extend([
        "",
        f"Peak incremental Python allocation during one full fusion replay: "
        f"{memory_row['peak_mb']:.3f} MB.",
        "",
        "Timing excludes black-box training and upstream rule-bank construction.",
    ])

    (
        OUT
        / "PHASE4D_RUNTIME_SCALABILITY_REPORT.md"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # Console summary.
    # --------------------------------------------------------------
    print()
    print("=" * 118)
    print("ETCE PHASE 4D-15 FORMAL RUNTIME / SCALABILITY PASSED")
    print("=" * 118)

    print(
        "[Correctness] "
        f"matched={correctness['matched_rows']} | "
        f"max_numeric_diff={correctness['max_numeric_abs_diff']:.3e} | "
        f"condition_mismatch={correctness['selected_condition_mismatch']}"
    )

    print()
    print("[Runtime summary]")

    for r in summary_rows:
        print(
            f"{r['scope']:20s} | "
            f"CF={r['eligible_cf']:4d} | "
            f"S1/S2/S3={r['stage1_groups']}/"
            f"{r['stage2_groups']}/"
            f"{r['stage3_groups']} | "
            f"Pairs={r['total_pairwise']:4d} | "
            f"S1={r['stage1_mean_seconds']:.6f}s | "
            f"S2={r['stage2_mean_seconds']:.6f}s | "
            f"S3={r['stage3_mean_seconds']:.6f}s | "
            f"Total={r['total_mean_seconds']:.6f}"
            f"±{r['total_std_seconds']:.6f}s | "
            f"ms/pair={r['mean_ms_per_pairwise_fusion']:.4f}"
        )

    print()
    print("[Prefix scaling]")

    for r in scaling:
        print(
            f"P{r['prefix']:2d} | "
            f"CF={r['eligible_cf']:4d} | "
            f"Pairs={r['total_pairwise']:4d} | "
            f"Time={r['mean_total_seconds']:.6f}s | "
            f"Time/P5={r['runtime_ratio_vs_P5']:.3f}x | "
            f"Pairs/P5={r['pairwise_ratio_vs_P5']:.3f}x"
        )

    print()
    print(
        f"[Input load] total={load_row['total_input_load_seconds']:.6f}s"
    )
    print(
        f"[Memory] incremental Python peak={memory_row['peak_mb']:.3f} MB"
    )
    print(
        f"[Hardware] {hardware['machine']} | "
        f"CPU={hardware['processor']} | "
        f"logical CPUs={hardware['logical_cpu_count']} | "
        f"Python={hardware['python_version']}"
    )

    print()
    print(f"Output: {OUT}")
    print("Old manuscript runtime reused: False")
    print("Formal TEST tuning performed: False")
    print("Protocol retuning performed: False")
    print("NEXT: stability / diversity.")


if __name__ == "__main__":
    main()
