#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-17 / Script 52
Three-seed explanation stability under neighborhood-generation randomness.

Design
------
POOL_SEED_BASES = [0, 1, 2]

Seed-base 0:
    Reuses the already validated primary formal TEST multi-rule bank.

Seed-bases 1 and 2:
    Regenerate the SAME 100 formal TEST cases × 4 models × P5/P10/P15/P20
    with the exact Phase4D-1 protocol, changing ONLY POOL_SEED_BASE.

Frozen across all seeds:
- formal TEST cases
- black-box checkpoints
- prefixes
- pool size = 320
- max edits = 3
- max options = 8
- process-valid proposal
- deterministic stratified local split logic
- LORE / DT / BN definitions and hyperparameters
- lambda = 5
- BPA mapping
- Stage1/2/3 grouping
- fusion order
- conflict policy
- representative-rule tie breaking
- fused-rule held-out re-evaluation

Important chronology
--------------------
The experiment plan predeclared multi-seed stability analysis, but the concrete
additional seed bases [1,2] are fixed here AFTER the primary formal TEST was
already executed. Therefore this is a post-primary-test robustness analysis,
not a preregistered primary endpoint.

Stability outputs
-----------------
A) Per-seed Stage3 genuinely fused metrics:
   Fidelity / CF validity / model validity / all-model validity /
   Reliability / ignorance / min witness edit / conflict K.

B) Pairwise seed stability on common Stage3 fused group identities:
   - exact selected-rule agreement
   - condition-position footprint Jaccard similarity

C) Target-level footprint stability:
   For each case_id × prefix, union all activity positions appearing in
   genuinely fused Stage3 selected rules, then compare seed pairs by Jaccard.

This target-level metric is explicitly an "explanation-footprint union"
stability measure. It is not claimed to be stability of one unique final rule,
because the current formal Stage3 export may contain multiple compatible fused
groups for the same case × prefix.

Run
---
    cd D:/ETCE
    python -u ./scripts/52_phase4d_multiseed_explanation_stability.py

The script is resumable because Phase4D-1 target caches are kept separately
for seed bases 1 and 2.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT34 = (
    ROOT
    / "scripts"
    / "34_phase4d_formal_test_multirule_bank.py"
)

SCRIPT37 = (
    ROOT
    / "scripts"
    / "37_phase4d_frozen_ds_formal_test_fusion_v2.py"
)

FORMAL = (
    ROOT
    / "results/phase4/phase4d_formal_test"
)

PRIMARY_BANK = (
    FORMAL
    / "formal_test_multirule_bank"
)

PRIMARY_STAGE = (
    FORMAL
    / "formal_test_frozen_ds_fusion"
    / "phase4d_frozen_ds_stage_outputs.csv"
)

PRIMARY_VALIDATION = (
    FORMAL
    / "formal_test_frozen_ds_fusion"
    / "validation_phase4d_frozen_ds_fusion.json"
)

OUT = (
    FORMAL
    / "formal_test_multiseed_stability"
)

SEED_ROOT = (
    OUT
    / "seed_banks"
)

SEED_BASES = (
    0,
    1,
    2,
)

FROZEN_LAMBDA = 5.0

EXPECTED_PRIMARY_STAGE_ROWS = 12839
TOL = 1e-12


# =====================================================================
# Utility
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


def read_jsonl(path):
    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(
            f,
            1,
        ):
            s = line.strip()

            if not s:
                continue

            try:
                rows.append(
                    json.loads(s)
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid JSONL at {path}:{line_no}"
                ) from exc

    return rows


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


def parse_signature(raw):
    if isinstance(
        raw,
        (list, tuple),
    ):
        vals = raw

    else:
        s = str(raw).strip()

        try:
            vals = json.loads(
                s
            )
        except Exception:
            vals = ast.literal_eval(
                s
            )

    return tuple(
        sorted(
            int(v)
            for v in vals
        )
    )


def conditions_obj(raw):
    if isinstance(
        raw,
        list,
    ):
        obj = raw
    else:
        s = str(raw).strip()

        try:
            obj = json.loads(
                s
            )
        except Exception:
            obj = ast.literal_eval(
                s
            )

    require(
        isinstance(
            obj,
            list,
        ),
        "selected_conditions is not a list",
    )

    return obj


def canonical_conditions(raw):
    return json.dumps(
        conditions_obj(
            raw
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def condition_positions(raw):
    return frozenset(
        int(
            c[
                "position"
            ]
        )
        for c in conditions_obj(
            raw
        )
        if isinstance(
            c,
            dict,
        )
        and c.get(
            "position"
        )
        is not None
    )


def jaccard_similarity(
    a,
    b,
):
    a = set(a)
    b = set(b)

    union = (
        a
        | b
    )

    if not union:
        return 1.0

    return float(
        len(
            a
            & b
        )
        / len(
            union
        )
    )


# =====================================================================
# Import exact Phase4D-1 and Phase4D-4 implementations
# =====================================================================

def load_module(
    path,
    name,
):
    require(
        path.is_file(),
        f"Missing required script: {path}",
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            path,
        )
    )

    require(
        spec.loader
        is not None,
        f"Could not load {path}",
    )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


m34 = load_module(
    SCRIPT34,
    "etce_phase4d_bank_multiseed",
)

m37 = load_module(
    SCRIPT37,
    "etce_phase4d_ds_multiseed",
)


# =====================================================================
# Bank generation
# =====================================================================

def configure_seed_bank(
    seed_base,
):
    """
    Redirect Script34 globals into an isolated per-seed directory.
    """
    bank_dir = (
        SEED_ROOT
        / f"seed_{seed_base}"
        / "formal_test_multirule_bank"
    )

    m34.POOL_SEED_BASE = int(
        seed_base
    )

    m34.OUT = bank_dir
    m34.CACHE = (
        bank_dir
        / "target_cache"
    )

    m34.EVAL_CACHE = (
        bank_dir
        / "fusion_eval_cache"
        / "targets"
    )

    # Script34 sets this event at the end of each main() call.
    m34._STOP_HEARTBEAT.clear()

    return bank_dir


def ensure_seed_bank(
    seed_base,
):
    if int(
        seed_base
    ) == 0:
        require(
            (
                PRIMARY_BANK
                / "canonical_rules_test.jsonl"
            ).is_file(),
            "Primary seed-0 formal bank missing",
        )

        require(
            (
                PRIMARY_BANK
                / "fusion_eval_cache"
                / "targets"
            ).is_dir(),
            "Primary seed-0 eval cache missing",
        )

        return (
            PRIMARY_BANK
        )

    bank_dir = (
        configure_seed_bank(
            seed_base
        )
    )

    print()
    print("=" * 118)
    print(
        f"[Seed base {seed_base}] "
        "building/resuming exact Phase4D-1 bank"
    )
    print("=" * 118)

    m34.main()

    require(
        (
            bank_dir
            / "canonical_rules_test.jsonl"
        ).is_file(),
        (
            f"Seed {seed_base} bank did not produce canonical_rules_test.jsonl"
        ),
    )

    require(
        (
            bank_dir
            / "fusion_eval_cache"
            / "targets"
        ).is_dir(),
        (
            f"Seed {seed_base} bank did not produce fusion eval cache"
        ),
    )

    cfg = read_json(
        bank_dir
        / "formal_test_run_config.json"
    )

    require(
        int(
            cfg[
                "pool_seed_base"
            ]
        )
        == int(
            seed_base
        ),
        (
            f"Seed-bank config mismatch: expected {seed_base}, "
            f"found {cfg.get('pool_seed_base')}"
        ),
    )

    return bank_dir


# =====================================================================
# Dynamic exact frozen D-S replay
# =====================================================================

def load_cache_index(
    cache_dir,
):
    return (
        m37.load_cache_index(
            cache_dir
        )
    )


def eligible_cf(
    rules,
):
    return [
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


def fuse_seed_bank(
    seed_base,
    bank_dir,
):
    """
    Same hierarchy as frozen Script37, but with dynamic group counts so that
    stochastic robustness runs are not forced to match seed-0 inventory.
    """
    rules = read_jsonl(
        bank_dir
        / "canonical_rules_test.jsonl"
    )

    cf = eligible_cf(
        rules
    )

    cache = load_cache_index(
        bank_dir
        / "fusion_eval_cache"
        / "targets"
    )

    require(
        len(
            cache
        )
        == 1600,
        (
            f"Seed {seed_base}: expected 1600 cache targets, "
            f"found {len(cache)}"
        ),
    )

    lam = (
        FROZEN_LAMBDA
    )

    # --------------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------------
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
    conflict_rows = []

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
            source_rules[
                1:
            ],
            1,
        ):
            right = (
                m37.source_rule(
                    r,
                    lam,
                )
            )

            current, trace = (
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
                        "case_id": case_id,
                        "model": model,
                        "prefix": int(
                            prefix
                        ),
                        "surrogate": surrogate,
                        "predicted_class": int(
                            pred
                        ),
                        "feature_signature": json.dumps(
                            list(
                                sig
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

    # --------------------------------------------------------------
    # Stage 2
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
            int(
                prefix
            ),
            int(
                pred
            ),
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
            items[
                1:
            ],
            1,
        ):
            current, trace = (
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
                        "feature_signature": json.dumps(
                            list(
                                sig
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

    # --------------------------------------------------------------
    # Stage 3
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
            int(
                prefix
            ),
            int(
                pred
            ),
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
            items[
                1:
            ],
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
                        int(
                            prefix
                        ),
                    )
                ]
                for m
                in models_so_far
            ]

            current, trace = (
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
                        "feature_signature": json.dumps(
                            list(
                                sig
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

    return {
        "seed_base": int(
            seed_base
        ),
        "eligible_cf_count": len(
            cf
        ),
        "stage_rows": stage_rows,
        "conflict_rows": conflict_rows,
        "stage1_group_count": len(
            s1_outputs
        ),
        "stage2_group_count": len(
            s2_outputs
        ),
        "stage3_group_count": len(
            s3_outputs
        ),
    }


# =====================================================================
# Seed-0 correctness gate
# =====================================================================

def row_identity(
    r,
):
    stage = str(
        r[
            "stage"
        ]
    )

    sig = tuple(
        parse_signature(
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


def validate_seed0(
    reproduced,
):
    saved = read_csv(
        PRIMARY_STAGE
    )

    require(
        len(
            saved
        )
        == EXPECTED_PRIMARY_STAGE_ROWS,
        "Primary frozen Stage output count mismatch",
    )

    require(
        len(
            reproduced
        )
        == EXPECTED_PRIMARY_STAGE_ROWS,
        (
            "Seed-0 replay count mismatch: "
            f"{len(reproduced)}"
        ),
    )

    a = {
        row_identity(
            r
        ): r
        for r
        in reproduced
    }

    b = {
        row_identity(
            r
        ): r
        for r
        in saved
    }

    require(
        set(a)
        == set(b),
        "Seed-0 replay identities differ from frozen primary output",
    )

    max_diff = 0.0

    numeric_fields = (
        "Fid_r",
        "coverage_eval",
        "R_r",
        "model_cf_validity_rate",
        "min_witness_edit",
        "last_K",
        "last_selected_mass",
        "last_ignorance_mass",
    )

    for key in a:
        ra = a[
            key
        ]
        rb = b[
            key
        ]

        require(
            canonical_conditions(
                ra[
                    "selected_conditions"
                ]
            )
            == canonical_conditions(
                rb[
                    "selected_conditions"
                ]
            ),
            (
                f"Seed-0 selected rule mismatch: {key}"
            ),
        )

        for field in numeric_fields:
            x = parse_float(
                ra.get(
                    field
                )
            )

            y = parse_float(
                rb.get(
                    field
                )
            )

            if (
                x is None
                and y is None
            ):
                continue

            require(
                x is not None
                and y is not None,
                (
                    f"Seed-0 null mismatch {field}: {key}"
                ),
            )

            d = abs(
                x - y
            )

            max_diff = max(
                max_diff,
                d,
            )

            require(
                d <= TOL,
                (
                    f"Seed-0 numeric mismatch {field}: "
                    f"{key}: {x} vs {y}"
                ),
            )

    return {
        "matched_rows": len(
            a
        ),
        "max_numeric_abs_diff": (
            max_diff
        ),
        "selected_rule_mismatch": 0,
    }


# =====================================================================
# Summaries and stability
# =====================================================================

def stage3_identity(
    r,
):
    return (
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
        tuple(
            parse_signature(
                r[
                    "feature_signature"
                ]
            )
        ),
    )


def fused_stage3(
    rows,
):
    return [
        r
        for r in rows
        if r[
            "stage"
        ] == "Stage3"
        and parse_bool(
            r[
                "actually_fused"
            ]
        )
    ]


def seed_stage3_summary(
    seed_base,
    result,
):
    rows = fused_stage3(
        result[
            "stage_rows"
        ]
    )

    stage3_conflict = [
        r
        for r in result[
            "conflict_rows"
        ]
        if r[
            "stage"
        ] == "Stage3"
    ]

    valid_edit = [
        parse_float(
            r[
                "min_witness_edit"
            ]
        )
        for r in rows
        if parse_float(
            r[
                "min_witness_edit"
            ]
        )
        is not None
    ]

    return {
        "seed_base": int(
            seed_base
        ),
        "eligible_cf_count": result[
            "eligible_cf_count"
        ],
        "stage1_group_count": result[
            "stage1_group_count"
        ],
        "stage2_group_count": result[
            "stage2_group_count"
        ],
        "stage3_group_count": result[
            "stage3_group_count"
        ],
        "stage3_fused_count": len(
            rows
        ),
        "stage3_mean_fidelity": mean(
            parse_float(
                r[
                    "Fid_r"
                ]
            )
            for r in rows
        ),
        "stage3_cf_validity": mean(
            int(
                parse_bool(
                    r[
                        "cf_valid"
                    ]
                )
            )
            for r in rows
        ),
        "stage3_model_validity": mean(
            parse_float(
                r[
                    "model_cf_validity_rate"
                ]
            )
            for r in rows
        ),
        "stage3_all_model_validity": mean(
            int(
                parse_bool(
                    r[
                        "all_models_cf_valid"
                    ]
                )
            )
            for r in rows
        ),
        "stage3_mean_reliability": mean(
            parse_float(
                r[
                    "R_r"
                ]
            )
            for r in rows
        ),
        "stage3_mean_ignorance": mean(
            parse_float(
                r[
                    "last_ignorance_mass"
                ]
            )
            for r in rows
        ),
        "stage3_mean_min_witness_edit": mean(
            valid_edit
        ),
        "stage3_pairwise_fusion_count": len(
            stage3_conflict
        ),
        "stage3_mean_conflict_K": mean(
            parse_float(
                r[
                    "K"
                ]
            )
            for r in stage3_conflict
        ),
    }


def aggregate_seed_metrics(
    seed_rows,
):
    metrics = [
        "eligible_cf_count",
        "stage1_group_count",
        "stage2_group_count",
        "stage3_group_count",
        "stage3_fused_count",
        "stage3_mean_fidelity",
        "stage3_cf_validity",
        "stage3_model_validity",
        "stage3_all_model_validity",
        "stage3_mean_reliability",
        "stage3_mean_ignorance",
        "stage3_mean_min_witness_edit",
        "stage3_mean_conflict_K",
    ]

    out = []

    for metric in metrics:
        vals = [
            r[
                metric
            ]
            for r in seed_rows
        ]

        out.append({
            "metric": metric,
            "mean": mean(
                vals
            ),
            "std": stdev(
                vals
            ),
            "min": min(
                float(v)
                for v in vals
            ),
            "max": max(
                float(v)
                for v in vals
            ),
            "seed_count": len(
                vals
            ),
        })

    return out


def pairwise_group_stability(
    seed_a,
    rows_a,
    seed_b,
    rows_b,
):
    a = {
        stage3_identity(
            r
        ): r
        for r in fused_stage3(
            rows_a
        )
    }

    b = {
        stage3_identity(
            r
        ): r
        for r in fused_stage3(
            rows_b
        )
    }

    keys_a = set(
        a
    )

    keys_b = set(
        b
    )

    common = sorted(
        keys_a
        & keys_b
    )

    union = (
        keys_a
        | keys_b
    )

    exact = []
    jacc = []

    for key in common:
        ca = canonical_conditions(
            a[
                key
            ][
                "selected_conditions"
            ]
        )

        cb = canonical_conditions(
            b[
                key
            ][
                "selected_conditions"
            ]
        )

        exact.append(
            int(
                ca == cb
            )
        )

        pa = condition_positions(
            a[
                key
            ][
                "selected_conditions"
            ]
        )

        pb = condition_positions(
            b[
                key
            ][
                "selected_conditions"
            ]
        )

        jacc.append(
            jaccard_similarity(
                pa,
                pb,
            )
        )

    return {
        "seed_a": int(
            seed_a
        ),
        "seed_b": int(
            seed_b
        ),
        "stage3_fused_a": len(
            a
        ),
        "stage3_fused_b": len(
            b
        ),
        "common_group_count": len(
            common
        ),
        "group_identity_jaccard": (
            len(
                common
            )
            / len(
                union
            )
            if union
            else 1.0
        ),
        "exact_selected_rule_agreement": (
            mean(
                exact
            )
            if exact
            else None
        ),
        "mean_condition_position_jaccard": (
            mean(
                jacc
            )
            if jacc
            else None
        ),
        "median_condition_position_jaccard": (
            median(
                jacc
            )
            if jacc
            else None
        ),
    }


def target_footprint_map(
    rows,
):
    """
    case_id × prefix -> union of selected condition positions over genuinely
    fused Stage3 groups.
    """
    out = defaultdict(
        set
    )

    for r in fused_stage3(
        rows
    ):
        key = (
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
        )

        out[
            key
        ].update(
            condition_positions(
                r[
                    "selected_conditions"
                ]
            )
        )

    return {
        key: frozenset(
            vals
        )
        for key, vals
        in out.items()
    }


def pairwise_target_footprint_stability(
    seed_a,
    rows_a,
    seed_b,
    rows_b,
):
    a = target_footprint_map(
        rows_a
    )

    b = target_footprint_map(
        rows_b
    )

    ka = set(
        a
    )

    kb = set(
        b
    )

    common = sorted(
        ka
        & kb
    )

    union = (
        ka
        | kb
    )

    sims = [
        jaccard_similarity(
            a[
                k
            ],
            b[
                k
            ],
        )
        for k in common
    ]

    exact = [
        int(
            a[
                k
            ]
            == b[
                k
            ]
        )
        for k in common
    ]

    return {
        "seed_a": int(
            seed_a
        ),
        "seed_b": int(
            seed_b
        ),
        "targets_with_stage3_fused_a": len(
            a
        ),
        "targets_with_stage3_fused_b": len(
            b
        ),
        "common_target_count": len(
            common
        ),
        "target_availability_jaccard": (
            len(
                common
            )
            / len(
                union
            )
            if union
            else 1.0
        ),
        "mean_target_footprint_jaccard": (
            mean(
                sims
            )
            if sims
            else None
        ),
        "median_target_footprint_jaccard": (
            median(
                sims
            )
            if sims
            else None
        ),
        "exact_target_footprint_agreement": (
            mean(
                exact
            )
            if exact
            else None
        ),
    }


# =====================================================================
# Main
# =====================================================================

def main():
    require(
        PRIMARY_VALIDATION.is_file(),
        (
            f"Missing primary formal validation: "
            f"{PRIMARY_VALIDATION}"
        ),
    )

    primary_validation = read_json(
        PRIMARY_VALIDATION
    )

    require(
        primary_validation.get(
            "status"
        )
        == "passed",
        "Primary formal D-S output is not validated",
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 118)
    print("ETCE PHASE 4D-17 THREE-SEED EXPLANATION STABILITY")
    print("=" * 118)
    print(f"Seed bases: {list(SEED_BASES)}")
    print("Seed 0: existing validated primary formal run")
    print("Seeds 1/2: exact Phase4D-1 regeneration; ONLY pool seed base changes")
    print("Frozen lambda/BPA/grouping/fusion order/conflict/tie-break: True")
    print("Used for TEST retuning: False")
    print("Chronology: post-primary-test robustness analysis")
    print()

    seed_results = {}
    seed_summaries = []

    for seed_base in SEED_BASES:
        bank_dir = (
            ensure_seed_bank(
                seed_base
            )
        )

        print()
        print(
            f"[Seed {seed_base}] exact frozen D-S replay ..."
        )

        result = fuse_seed_bank(
            seed_base,
            bank_dir,
        )

        seed_results[
            int(
                seed_base
            )
        ] = result

        s = seed_stage3_summary(
            seed_base,
            result,
        )

        seed_summaries.append(
            s
        )

        print(
            f"[Seed {seed_base}] "
            f"EligibleCF={s['eligible_cf_count']} | "
            f"S1/S2/S3={s['stage1_group_count']}/"
            f"{s['stage2_group_count']}/"
            f"{s['stage3_group_count']} | "
            f"Fused3={s['stage3_fused_count']} | "
            f"Fid={s['stage3_mean_fidelity']:.6f} | "
            f"CF={s['stage3_cf_validity']:.6f} | "
            f"ModelValid={s['stage3_model_validity']:.6f} | "
            f"AllModel={s['stage3_all_model_validity']:.6f}"
        )

    # --------------------------------------------------------------
    # Seed-0 exact primary-result gate.
    # --------------------------------------------------------------
    seed0_gate = validate_seed0(
        seed_results[
            0
        ][
            "stage_rows"
        ]
    )

    print()
    print(
        "[Seed-0 correctness] PASSED | "
        f"matched={seed0_gate['matched_rows']} | "
        f"maxdiff={seed0_gate['max_numeric_abs_diff']:.3e}"
    )

    # --------------------------------------------------------------
    # Mean ± std across seeds.
    # --------------------------------------------------------------
    aggregate = (
        aggregate_seed_metrics(
            seed_summaries
        )
    )

    # --------------------------------------------------------------
    # Pairwise group + target footprint stability.
    # --------------------------------------------------------------
    group_pairwise = []
    target_pairwise = []

    for a, b in combinations(
        SEED_BASES,
        2,
    ):
        group_pairwise.append(
            pairwise_group_stability(
                a,
                seed_results[
                    a
                ][
                    "stage_rows"
                ],
                b,
                seed_results[
                    b
                ][
                    "stage_rows"
                ],
            )
        )

        target_pairwise.append(
            pairwise_target_footprint_stability(
                a,
                seed_results[
                    a
                ][
                    "stage_rows"
                ],
                b,
                seed_results[
                    b
                ][
                    "stage_rows"
                ],
            )
        )

    # --------------------------------------------------------------
    # Persist per-seed Stage3 output for auditability.
    # --------------------------------------------------------------
    for seed_base in SEED_BASES:
        write_csv(
            OUT
            / f"seed_{seed_base}_stage3_outputs.csv",
            [
                r
                for r in seed_results[
                    seed_base
                ][
                    "stage_rows"
                ]
                if r[
                    "stage"
                ]
                == "Stage3"
            ],
        )

    write_csv(
        OUT
        / "multiseed_stage3_per_seed_summary.csv",
        seed_summaries,
    )

    write_csv(
        OUT
        / "multiseed_stage3_mean_std.csv",
        aggregate,
    )

    write_csv(
        OUT
        / "multiseed_group_stability_pairwise.csv",
        group_pairwise,
    )

    write_csv(
        OUT
        / "multiseed_target_footprint_stability_pairwise.csv",
        target_pairwise,
    )

    result = {
        "scope": (
            "ETCE Phase 4D-17 three-seed explanation stability"
        ),
        "status": "passed",
        "formal_test": True,
        "seed_bases": [
            int(x)
            for x in SEED_BASES
        ],
        "primary_seed_base": 0,
        "additional_seed_bases": [
            1,
            2,
        ],
        "chronology_disclosure": {
            "multi_seed_stability_family_predeclared_in_experiment_plan": True,
            "exact_additional_seed_bases_fixed_after_primary_formal_test": True,
            "analysis_role": (
                "post-primary-test robustness analysis"
            ),
        },
        "frozen_across_seeds": {
            "formal_case_selection": True,
            "blackbox_checkpoints": True,
            "prefixes": True,
            "pool_size": True,
            "max_edits": True,
            "max_options": True,
            "process_valid_proposal": True,
            "surrogate_definitions": True,
            "surrogate_hyperparameters": True,
            "lambda": FROZEN_LAMBDA,
            "bpa_mapping": True,
            "stage_grouping": True,
            "fusion_order": True,
            "conflict_policy": True,
            "tie_breaking": True,
            "fused_rule_re_evaluation": True,
        },
        "changed_across_seeds": (
            "POOL_SEED_BASE only"
        ),
        "seed0_correctness_gate": (
            seed0_gate
        ),
        "per_seed": (
            seed_summaries
        ),
        "mean_std": (
            aggregate
        ),
        "pairwise_common_group_stability": (
            group_pairwise
        ),
        "pairwise_target_footprint_stability": (
            target_pairwise
        ),
        "stability_metric_guard": (
            "Target-level footprint stability compares unions of activity "
            "positions implicated by genuinely fused Stage3 rules; it is not "
            "claimed to be one-unique-final-rule stability."
        ),
        "test_side_tuning": False,
        "protocol_retuning": False,
        "next_step": (
            "Consolidate RQ6 hypothesis-driven ablation and family-wise "
            "statistical/effect-size table, then external-dataset validation."
        ),
    }

    write_json(
        OUT
        / "multiseed_stability_summary.json",
        result,
    )

    # --------------------------------------------------------------
    # Console
    # --------------------------------------------------------------
    print()
    print("=" * 118)
    print("ETCE PHASE 4D-17 THREE-SEED EXPLANATION STABILITY PASSED")
    print("=" * 118)

    print("[Per-seed Stage3 summary]")

    for s in seed_summaries:
        print(
            f"seed={s['seed_base']} | "
            f"EligibleCF={s['eligible_cf_count']} | "
            f"Fused3={s['stage3_fused_count']} | "
            f"Fid={s['stage3_mean_fidelity']:.6f} | "
            f"CF={s['stage3_cf_validity']:.6f} | "
            f"ModelValid={s['stage3_model_validity']:.6f} | "
            f"AllModel={s['stage3_all_model_validity']:.6f} | "
            f"R={s['stage3_mean_reliability']:.6f} | "
            f"Ign={s['stage3_mean_ignorance']:.6f} | "
            f"Edit={s['stage3_mean_min_witness_edit']:.6f} | "
            f"K={s['stage3_mean_conflict_K']:.6f}"
        )

    print()
    print("[Mean ± std across seeds]")

    for r in aggregate:
        print(
            f"{r['metric']:34s} | "
            f"{r['mean']:.6f} ± {r['std']:.6f}"
        )

    print()
    print("[Pairwise common Stage3-group stability]")

    for r in group_pairwise:
        print(
            f"{r['seed_a']} vs {r['seed_b']} | "
            f"Common={r['common_group_count']} | "
            f"GroupJacc={r['group_identity_jaccard']:.6f} | "
            f"ExactRule={r['exact_selected_rule_agreement']:.6f} | "
            f"PosJacc={r['mean_condition_position_jaccard']:.6f}"
        )

    print()
    print("[Pairwise case×prefix footprint stability]")

    for r in target_pairwise:
        print(
            f"{r['seed_a']} vs {r['seed_b']} | "
            f"CommonTargets={r['common_target_count']} | "
            f"AvailJacc={r['target_availability_jaccard']:.6f} | "
            f"FootJacc={r['mean_target_footprint_jaccard']:.6f} | "
            f"ExactFoot={r['exact_target_footprint_agreement']:.6f}"
        )

    print()
    print(f"Output: {OUT}")
    print("Seed count used for primary tuning: False")
    print("Protocol retuning performed: False")
    print(
        "Exact additional seed bases preregistered before primary TEST: False"
    )
    print(
        "NEXT: consolidated RQ6 ablation/significance table."
    )


if __name__ == "__main__":
    main()
