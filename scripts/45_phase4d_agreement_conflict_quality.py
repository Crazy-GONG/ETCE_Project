#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-11 / Script 45
Agreement -> Conflict -> Quality analysis.

Predeclared agreement proxy
---------------------------
For each local black-box target x, on the SAME held-out local evaluation set Z_x:

    A_ij(x) = mean_z I[g_i(z) == g_j(z)]

for the three internal surrogate branches:
    LORE, DT, BN

and

    A(x) = (A_LORE,DT + A_LORE,BN + A_DT,BN) / 3

Important:
- A(x) is an empirical surrogate-output agreement / dependence proxy.
- It is NOT a proof of statistical independence/dependence.
- Low / Medium / High agreement thresholds are the validation-set 33rd and
  67th percentiles, as predeclared in ETCE_EXPERIMENT_PLAN.md.
- Numeric thresholds are computed from validation only.
- Formal TEST is not used to choose thresholds.
- No ETCE protocol or lambda is changed.

This script independently reconstructs surrogate predictions from the frozen
validation/test fusion caches and checks reconstructed surrogate fidelity
against the saved surrogate-metric files.

Primary formal TEST analysis:
    Agreement A(x) -> Stage2 D-S conflict K
    Agreement A(x) -> Stage2 fused Rule Fidelity / CF Validity
    Conflict K      -> Stage2 fused Rule Fidelity / CF Validity

Both group-level and target-aggregated analyses are reported to avoid relying
only on repeated compatible groups from the same local target.

Run:
    cd D:/ETCE
    python -u ./scripts/45_phase4d_agreement_conflict_quality.py
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".deps"
if sys.platform == "win32" and LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from etce.phase4.surrogate_core import (
    NaiveBayesBN,
    make_position_domains,
    make_onehot_schema,
    onehot_transform,
    sequence_edit_count,
    sequence_hash,
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

VAL_BANK = ROOT / "results/phase4/phase4b_v2_multirule_bank"
VAL_CACHE_ROOT = ROOT / "results/phase4/phase4c_v2_fusion_eval_cache"
VAL_CACHE_DIR = VAL_CACHE_ROOT / "targets"
VAL_CACHE_VALIDATION = VAL_CACHE_ROOT / "validation_fusion_eval_cache.json"
VAL_METRICS = VAL_BANK / "surrogate_metrics_calibration.csv"
VAL_BANK_VALIDATION = VAL_BANK / "validation_phase4b_v2.json"

FORMAL = ROOT / "results/phase4/phase4d_formal_test"
TEST_BANK = FORMAL / "formal_test_multirule_bank"
TEST_CACHE_DIR = TEST_BANK / "fusion_eval_cache" / "targets"
TEST_METRICS = TEST_BANK / "surrogate_metrics_test.csv"
TEST_BANK_VALIDATION = TEST_BANK / "validation_phase4d_formal_test_multirule.json"

DS_DIR = FORMAL / "formal_test_frozen_ds_fusion"
DS_STAGE = DS_DIR / "phase4d_frozen_ds_stage_outputs.csv"
DS_TRACE = DS_DIR / "phase4d_frozen_ds_conflict_trace.csv"
DS_VALIDATION = DS_DIR / "validation_phase4d_frozen_ds_fusion.json"

FROZEN_DS = ROOT / "configs/phase4c_ds_frozen_config.json"
THRESHOLD_CONFIG = ROOT / "configs/phase4c_agreement_thresholds_frozen_config.json"

OUT = FORMAL / "formal_test_agreement_conflict_quality"

SURROGATES = ("LORE", "DT", "BN")
FID_TOL = 1e-12
TOL = 1e-12


# ---------------------------------------------------------------------
# IO / utility
# ---------------------------------------------------------------------

def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


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


def quantile(xs, p):
    vals = sorted(
        float(x)
        for x in xs
        if x is not None
        and math.isfinite(float(x))
    )

    if not vals:
        return None

    if len(vals) == 1:
        return vals[0]

    z = (len(vals) - 1) * float(p)
    lo = int(math.floor(z))
    hi = int(math.ceil(z))

    if lo == hi:
        return vals[lo]

    frac = z - lo
    return (
        vals[lo] * (1 - frac)
        + vals[hi] * frac
    )


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


def normalized_signature(s):
    return json.dumps(
        json.loads(str(s)),
        separators=(",", ":"),
    )


def target_key_from_payload(p):
    return (
        p["case_id"],
        p["model"],
        int(p["prefix"]),
    )


def target_key_from_metric(r):
    return (
        r["case_id"],
        r["model"],
        int(r["prefix"]),
    )


def target_key_from_stage(r):
    return (
        r["case_id"],
        r["model"],
        int(float(r["prefix"])),
    )


def stage2_group_key(r):
    return (
        r["case_id"],
        r["model"],
        int(float(r["prefix"])),
        int(float(r["predicted_class"])),
        normalized_signature(
            r["feature_signature"]
        ),
    )


# ---------------------------------------------------------------------
# Cache load
# ---------------------------------------------------------------------

def load_cache_dir(path):
    require(
        path.is_dir(),
        f"Cache directory missing: {path}",
    )

    idx = {}

    for fp in sorted(path.glob("*.json.gz")):
        p = read_gz(fp)
        key = target_key_from_payload(p)

        require(
            key not in idx,
            f"Duplicate cache target: {key}",
        )

        idx[key] = p

    return idx


# ---------------------------------------------------------------------
# Exact surrogate prediction reconstruction
# ---------------------------------------------------------------------

def tree_predictions(
    branch_name,
    fit_sequences,
    fit_labels,
    eval_sequences,
    factual,
):
    fit_sequences = [
        tuple(seq)
        for seq in fit_sequences
    ]
    eval_sequences = [
        tuple(seq)
        for seq in eval_sequences
    ]
    factual = tuple(factual)

    fit_labels = np.asarray(
        fit_labels,
        dtype=int,
    )

    if len(set(fit_labels.tolist())) < 2:
        return None

    if branch_name == "LORE":
        max_depth = None
        min_samples_leaf = 2
        balance = True

    elif branch_name == "DT":
        max_depth = 4
        min_samples_leaf = 5
        balance = False

    else:
        raise ValueError(branch_name)

    if balance:
        by_class = {
            cls: [
                i
                for i, value in enumerate(fit_labels)
                if int(value) == cls
            ]
            for cls in (0, 1)
        }

        n = min(
            len(by_class[0]),
            len(by_class[1]),
        )

        if n == 0:
            return None

        selected = []

        for cls in (0, 1):
            ranked = sorted(
                by_class[cls],
                key=lambda i: (
                    sequence_edit_count(
                        factual,
                        fit_sequences[i],
                    ),
                    sequence_hash(
                        fit_sequences[i]
                    ),
                ),
            )
            selected.extend(
                ranked[:n]
            )

        selected = sorted(selected)

        fit_sequences = [
            fit_sequences[i]
            for i in selected
        ]

        fit_labels = fit_labels[
            selected
        ]

    all_sequences = (
        list(fit_sequences)
        + list(eval_sequences)
        + [tuple(factual)]
    )

    position_domains = (
        make_position_domains(
            all_sequences
        )
    )

    feature_meta = (
        make_onehot_schema(
            position_domains
        )
    )

    X_fit = onehot_transform(
        fit_sequences,
        feature_meta,
    )

    X_eval = onehot_transform(
        eval_sequences,
        feature_meta,
    )

    tree = DecisionTreeClassifier(
        random_state=0,
        max_depth=max_depth,
        min_samples_leaf=int(
            min_samples_leaf
        ),
        criterion="gini",
    )

    tree.fit(
        X_fit,
        fit_labels,
    )

    return tree.predict(
        X_eval
    ).astype(int)


def bn_predictions(
    fit_sequences,
    fit_labels,
    eval_sequences,
):
    fit_labels = np.asarray(
        fit_labels,
        dtype=int,
    )

    if len(set(fit_labels.tolist())) < 2:
        return None

    bn = NaiveBayesBN.fit(
        [
            tuple(seq)
            for seq in fit_sequences
        ],
        fit_labels,
        alpha=1.0,
    )

    return bn.predict(
        [
            tuple(seq)
            for seq in eval_sequences
        ]
    ).astype(int)


def reconstruct_predictions(payload):
    sequences = [
        tuple(seq)
        for seq in payload[
            "pool_sequences"
        ]
    ]

    labels = np.asarray(
        payload["pool_labels"],
        dtype=int,
    )

    fit_idx = [
        int(i)
        for i in payload[
            "fit_indices"
        ]
    ]

    eval_idx = [
        int(i)
        for i in payload[
            "eval_indices"
        ]
    ]

    factual = tuple(
        payload[
            "factual_sequence"
        ]
    )

    fit_sequences = [
        sequences[i]
        for i in fit_idx
    ]

    fit_labels = labels[
        fit_idx
    ]

    eval_sequences = [
        sequences[i]
        for i in eval_idx
    ]

    eval_labels = labels[
        eval_idx
    ]

    pred = {
        "LORE": tree_predictions(
            "LORE",
            fit_sequences,
            fit_labels,
            eval_sequences,
            factual,
        ),
        "DT": tree_predictions(
            "DT",
            fit_sequences,
            fit_labels,
            eval_sequences,
            factual,
        ),
        "BN": bn_predictions(
            fit_sequences,
            fit_labels,
            eval_sequences,
        ),
    }

    return pred, eval_labels


def metric_index(rows):
    idx = {}

    for r in rows:
        key = (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
            r["surrogate"],
        )

        require(
            key not in idx,
            f"Duplicate surrogate metric: {key}",
        )

        idx[key] = r

    return idx


def verify_fidelity(
    target_key,
    predictions,
    eval_labels,
    saved_metric_index,
):
    diffs = []

    for surrogate in SURROGATES:
        rec = saved_metric_index[
            (
                target_key[0],
                target_key[1],
                target_key[2],
                surrogate,
            )
        ]

        saved = parse_float(
            rec.get(
                "surrogate_fidelity_eval"
            )
        )

        p = predictions[
            surrogate
        ]

        if p is None:
            require(
                saved is None,
                (
                    "Reconstructed branch unavailable "
                    f"but saved fidelity is present: "
                    f"{target_key}/{surrogate}"
                ),
            )
            continue

        calc = float(
            np.mean(
                p
                == np.asarray(
                    eval_labels,
                    dtype=int,
                )
            )
        )

        require(
            saved is not None,
            (
                "Reconstructed branch available "
                f"but saved fidelity missing: "
                f"{target_key}/{surrogate}"
            ),
        )

        diff = abs(
            calc
            - float(saved)
        )

        require(
            diff <= FID_TOL,
            (
                "Surrogate fidelity mismatch "
                f"{target_key}/{surrogate}: "
                f"{calc} != {saved}"
            ),
        )

        diffs.append(diff)

    return max(diffs) if diffs else 0.0


def agreement_row(
    target_key,
    predictions,
    eval_labels,
):
    if len(eval_labels) == 0:
        return {
            "case_id": target_key[0],
            "model": target_key[1],
            "prefix": target_key[2],
            "eval_size": 0,
            "agreement_available": False,
            "availability_reason": "empty_eval",
        }

    if any(
        predictions[s] is None
        for s in SURROGATES
    ):
        missing = [
            s
            for s in SURROGATES
            if predictions[s] is None
        ]

        return {
            "case_id": target_key[0],
            "model": target_key[1],
            "prefix": target_key[2],
            "eval_size": len(eval_labels),
            "agreement_available": False,
            "availability_reason": (
                "unavailable_surrogate:"
                + ",".join(missing)
            ),
        }

    lore = predictions["LORE"]
    dt = predictions["DT"]
    bn = predictions["BN"]

    require(
        len(lore)
        == len(dt)
        == len(bn)
        == len(eval_labels),
        "Surrogate prediction length mismatch",
    )

    a_ld = float(
        np.mean(
            lore == dt
        )
    )

    a_lb = float(
        np.mean(
            lore == bn
        )
    )

    a_db = float(
        np.mean(
            dt == bn
        )
    )

    a = float(
        (
            a_ld
            + a_lb
            + a_db
        )
        / 3.0
    )

    return {
        "case_id": target_key[0],
        "model": target_key[1],
        "prefix": target_key[2],
        "eval_size": len(
            eval_labels
        ),
        "agreement_available": True,
        "availability_reason": "ok",
        "A_LORE_DT": a_ld,
        "A_LORE_BN": a_lb,
        "A_DT_BN": a_db,
        "A_mean": a,
        "LORE_fidelity": float(
            np.mean(
                lore
                == eval_labels
            )
        ),
        "DT_fidelity": float(
            np.mean(
                dt
                == eval_labels
            )
        ),
        "BN_fidelity": float(
            np.mean(
                bn
                == eval_labels
            )
        ),
    }


def build_agreement_table(
    cache,
    saved_metric_rows,
    label,
):
    saved_idx = metric_index(
        saved_metric_rows
    )

    require(
        len(saved_idx)
        == len(cache) * 3,
        (
            f"{label}: expected exactly "
            "3 surrogate metric rows per target"
        ),
    )

    out = []
    max_fid_diff = 0.0

    for n, key in enumerate(
        sorted(cache),
        start=1,
    ):
        p = cache[key]

        pred, eval_labels = (
            reconstruct_predictions(
                p
            )
        )

        max_fid_diff = max(
            max_fid_diff,
            verify_fidelity(
                key,
                pred,
                eval_labels,
                saved_idx,
            ),
        )

        out.append(
            agreement_row(
                key,
                pred,
                eval_labels,
            )
        )

        if (
            n % 200 == 0
            or n == len(cache)
        ):
            print(
                f"[{label}] reconstructed "
                f"{n}/{len(cache)} targets",
                flush=True,
            )

    return out, max_fid_diff


# ---------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------

def pearson(xs, ys):
    pairs = [
        (
            float(x),
            float(y),
        )
        for x, y in zip(
            xs,
            ys,
        )
        if x is not None
        and y is not None
        and math.isfinite(
            float(x)
        )
        and math.isfinite(
            float(y)
        )
    ]

    if len(pairs) < 2:
        return None

    x = [
        p[0]
        for p in pairs
    ]
    y = [
        p[1]
        for p in pairs
    ]

    mx = mean(x)
    my = mean(y)

    num = sum(
        (a - mx)
        * (b - my)
        for a, b in zip(
            x,
            y,
        )
    )

    dx = math.sqrt(
        sum(
            (a - mx) ** 2
            for a in x
        )
    )

    dy = math.sqrt(
        sum(
            (b - my) ** 2
            for b in y
        )
    )

    if dx == 0 or dy == 0:
        return None

    return float(
        num
        / (dx * dy)
    )


def average_ranks(vals):
    indexed = sorted(
        enumerate(
            float(v)
            for v in vals
        ),
        key=lambda x: x[1],
    )

    ranks = [
        0.0
    ] * len(indexed)

    i = 0

    while i < len(indexed):
        j = i + 1

        while (
            j < len(indexed)
            and abs(
                indexed[j][1]
                - indexed[i][1]
            )
            <= 1e-15
        ):
            j += 1

        rank = (
            (i + 1 + j)
            / 2.0
        )

        for k in range(i, j):
            ranks[
                indexed[k][0]
            ] = rank

        i = j

    return ranks


def spearman(xs, ys):
    pairs = [
        (
            float(x),
            float(y),
        )
        for x, y in zip(
            xs,
            ys,
        )
        if x is not None
        and y is not None
        and math.isfinite(
            float(x)
        )
        and math.isfinite(
            float(y)
        )
    ]

    if len(pairs) < 2:
        return None

    x = [
        p[0]
        for p in pairs
    ]

    y = [
        p[1]
        for p in pairs
    ]

    return pearson(
        average_ranks(x),
        average_ranks(y),
    )


def corr_record(
    scope,
    x_name,
    y_name,
    rows,
):
    x = [
        r.get(x_name)
        for r in rows
    ]

    y = [
        r.get(y_name)
        for r in rows
    ]

    n = sum(
        a is not None
        and b is not None
        for a, b in zip(
            x,
            y,
        )
    )

    return {
        "scope": scope,
        "x": x_name,
        "y": y_name,
        "N": n,
        "pearson_r": pearson(
            x,
            y,
        ),
        "spearman_rho": spearman(
            x,
            y,
        ),
    }


# ---------------------------------------------------------------------
# Formal D-S joins
# ---------------------------------------------------------------------

def build_stage2_analysis(
    agreement_rows,
    stage_rows,
    trace_rows,
    q33,
    q67,
):
    agreement_idx = {
        (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
        ): r
        for r in agreement_rows
        if r[
            "agreement_available"
        ]
    }

    s2 = [
        r
        for r in stage_rows
        if r["stage"]
        == "Stage2"
        and parse_bool(
            r["actually_fused"]
        )
    ]

    require(
        len(s2) == 614,
        (
            "Expected 614 genuinely fused "
            f"Stage2 groups, found {len(s2)}"
        ),
    )

    trace_groups = defaultdict(
        list
    )

    for r in trace_rows:
        if r["stage"] != "Stage2":
            continue

        trace_groups[
            stage2_group_key(r)
        ].append(r)

    def tier(a):
        if a <= q33:
            return "Low"
        if a <= q67:
            return "Medium"
        return "High"

    detail = []

    for r in s2:
        tkey = target_key_from_stage(
            r
        )

        if tkey not in agreement_idx:
            continue

        a = agreement_idx[tkey]
        gkey = stage2_group_key(
            r
        )

        tr = trace_groups.get(
            gkey,
            [],
        )

        require(
            tr,
            f"Stage2 trace missing: {gkey}",
        )

        ks = [
            parse_float(x["K"])
            for x in tr
        ]

        detail.append({
            "case_id": r[
                "case_id"
            ],
            "model": r[
                "model"
            ],
            "prefix": int(
                float(
                    r["prefix"]
                )
            ),
            "predicted_class": int(
                float(
                    r[
                        "predicted_class"
                    ]
                )
            ),
            "feature_signature": (
                normalized_signature(
                    r[
                        "feature_signature"
                    ]
                )
            ),
            "A_LORE_DT": a[
                "A_LORE_DT"
            ],
            "A_LORE_BN": a[
                "A_LORE_BN"
            ],
            "A_DT_BN": a[
                "A_DT_BN"
            ],
            "A_mean": a[
                "A_mean"
            ],
            "agreement_tier": tier(
                a["A_mean"]
            ),
            "pairwise_steps": len(
                tr
            ),
            "mean_K": mean(
                ks
            ),
            "max_K": max(
                ks
            ),
            "last_K": parse_float(
                r["last_K"]
            ),
            "Fid_r": parse_float(
                r["Fid_r"]
            ),
            "cf_valid": int(
                parse_bool(
                    r[
                        "cf_valid"
                    ]
                )
            ),
            "R_r": parse_float(
                r["R_r"]
            ),
            "coverage_eval": (
                parse_float(
                    r[
                        "coverage_eval"
                    ]
                )
            ),
            "n_r": parse_float(
                r["n_r"]
            ),
            "ignorance": (
                parse_float(
                    r[
                        "last_ignorance_mass"
                    ]
                )
            ),
        })

    return detail


def tier_summary(
    detail,
):
    out = []

    for tier in (
        "Low",
        "Medium",
        "High",
    ):
        rows = [
            r
            for r in detail
            if r[
                "agreement_tier"
            ] == tier
        ]

        if not rows:
            out.append({
                "agreement_tier": tier,
                "N": 0,
            })
            continue

        out.append({
            "agreement_tier": tier,
            "N": len(
                rows
            ),
            "mean_agreement": mean(
                r["A_mean"]
                for r in rows
            ),
            "mean_K": mean(
                r["mean_K"]
                for r in rows
            ),
            "median_K": median(
                r["mean_K"]
                for r in rows
            ),
            "max_K": max(
                r["max_K"]
                for r in rows
            ),
            "mean_rule_fidelity": mean(
                r["Fid_r"]
                for r in rows
            ),
            "cf_validity_rate": mean(
                r["cf_valid"]
                for r in rows
            ),
            "mean_reliability": mean(
                r["R_r"]
                for r in rows
            ),
            "mean_coverage": mean(
                r[
                    "coverage_eval"
                ]
                for r in rows
            ),
            "mean_support": mean(
                r["n_r"]
                for r in rows
            ),
            "mean_ignorance": mean(
                r["ignorance"]
                for r in rows
            ),
        })

    return out


def aggregate_by_target(
    detail,
):
    groups = defaultdict(
        list
    )

    for r in detail:
        groups[
            (
                r["case_id"],
                r["model"],
                r["prefix"],
            )
        ].append(r)

    out = []

    for key, rows in sorted(
        groups.items()
    ):
        agreements = {
            round(
                float(
                    r["A_mean"]
                ),
                15,
            )
            for r in rows
        }

        require(
            len(
                agreements
            ) == 1,
            (
                "Agreement should be target-level "
                f"constant: {key}"
            ),
        )

        out.append({
            "case_id": key[0],
            "model": key[1],
            "prefix": key[2],
            "group_count": len(
                rows
            ),
            "A_mean": rows[0][
                "A_mean"
            ],
            "agreement_tier": rows[
                0
            ][
                "agreement_tier"
            ],
            "mean_group_K": mean(
                r["mean_K"]
                for r in rows
            ),
            "max_group_K": max(
                r["max_K"]
                for r in rows
            ),
            "mean_rule_fidelity": mean(
                r["Fid_r"]
                for r in rows
            ),
            "cf_validity_rate": mean(
                r["cf_valid"]
                for r in rows
            ),
            "mean_reliability": mean(
                r["R_r"]
                for r in rows
            ),
            "mean_coverage": mean(
                r[
                    "coverage_eval"
                ]
                for r in rows
            ),
            "mean_support": mean(
                r["n_r"]
                for r in rows
            ),
        })

    return out


def stage_conflict_summary(
    trace_rows,
):
    out = []

    for stage in (
        "Stage1",
        "Stage2",
        "Stage3",
    ):
        rows = [
            r
            for r in trace_rows
            if r["stage"]
            == stage
        ]

        ks = [
            parse_float(
                r["K"]
            )
            for r in rows
        ]

        out.append({
            "stage": stage,
            "pairwise_fusion_count": len(
                rows
            ),
            "mean_K": mean(
                ks
            ),
            "median_K": median(
                ks
            ),
            "p90_K": quantile(
                ks,
                0.90,
            ),
            "max_K": max(
                ks
            )
            if ks
            else None,
            "zero_K_rate": mean(
                int(
                    abs(float(k))
                    <= TOL
                )
                for k in ks
            ),
        })

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    required = [
        VAL_CACHE_VALIDATION,
        VAL_METRICS,
        VAL_BANK_VALIDATION,
        TEST_METRICS,
        TEST_BANK_VALIDATION,
        DS_STAGE,
        DS_TRACE,
        DS_VALIDATION,
        FROZEN_DS,
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

    val_cache_check = read_json(
        VAL_CACHE_VALIDATION
    )
    val_bank_check = read_json(
        VAL_BANK_VALIDATION
    )
    test_bank_check = read_json(
        TEST_BANK_VALIDATION
    )
    ds_validation = read_json(
        DS_VALIDATION
    )
    frozen_ds = read_json(
        FROZEN_DS
    )

    require(
        val_cache_check.get(
            "status"
        ) == "passed",
        "Validation cache not PASSED",
    )

    require(
        val_cache_check.get(
            "test_cases_used"
        ) is False,
        "Validation cache used TEST",
    )

    require(
        val_bank_check.get(
            "status"
        ) == "passed",
        "Validation rule bank not PASSED",
    )

    require(
        val_bank_check.get(
            "test_cases_used"
        ) is False,
        "Validation rule bank used TEST",
    )

    require(
        test_bank_check.get(
            "status"
        ) == "passed",
        "Formal TEST rule bank not PASSED",
    )

    require(
        ds_validation.get(
            "status"
        ) == "passed",
        "Formal D-S validation not PASSED",
    )

    require(
        frozen_ds.get(
            "status"
        ) == "frozen",
        "D-S config not frozen",
    )

    print("=" * 118)
    print("ETCE PHASE 4D-11 AGREEMENT -> CONFLICT -> QUALITY ANALYSIS")
    print("=" * 118)
    print(
        "Agreement proxy: empirical LORE/DT/BN output agreement on frozen held-out Z_x"
    )
    print(
        "Agreement thresholds: validation 33rd / 67th percentiles"
    )
    print(
        "Formal TEST used to choose thresholds: False"
    )
    print(
        "Protocol retuning: False"
    )
    print()

    # --------------------------------------------------------------
    # A) Validation agreement reconstruction and threshold freeze.
    # --------------------------------------------------------------
    val_cache = load_cache_dir(
        VAL_CACHE_DIR
    )

    require(
        len(val_cache) == 1600,
        (
            "Expected 1600 validation cache targets, "
            f"found {len(val_cache)}"
        ),
    )

    print(
        "[Validation] reconstructing surrogate predictions / agreement ..."
    )

    val_agreement, val_max_diff = (
        build_agreement_table(
            val_cache,
            read_csv(
                VAL_METRICS
            ),
            "VALID",
        )
    )

    val_available = [
        r
        for r in val_agreement
        if r[
            "agreement_available"
        ]
    ]

    require(
        val_available,
        "No validation targets have 3-surrogate agreement",
    )

    q33 = quantile(
        [
            r["A_mean"]
            for r in val_available
        ],
        0.33,
    )

    q67 = quantile(
        [
            r["A_mean"]
            for r in val_available
        ],
        0.67,
    )

    require(
        q33 is not None
        and q67 is not None,
        "Agreement thresholds unavailable",
    )

    require(
        q33 <= q67,
        "Invalid agreement thresholds",
    )

    threshold_payload = {
        "schema_version": (
            "etce-phase4c-agreement-thresholds-v1"
        ),
        "status": "frozen",
        "definition": {
            "A_LORE_DT": (
                "mean I[g_LORE(z)=g_DT(z)] on the same held-out Z_x"
            ),
            "A_LORE_BN": (
                "mean I[g_LORE(z)=g_BN(z)] on the same held-out Z_x"
            ),
            "A_DT_BN": (
                "mean I[g_DT(z)=g_BN(z)] on the same held-out Z_x"
            ),
            "A_mean": (
                "(A_LORE_DT + A_LORE_BN + A_DT_BN)/3"
            ),
            "interpretation": (
                "empirical surrogate-output agreement/dependence proxy; "
                "not proof of statistical independence"
            ),
        },
        "threshold_source": (
            "validation_only"
        ),
        "validation_target_count": len(
            val_agreement
        ),
        "validation_agreement_available_count": len(
            val_available
        ),
        "q33": q33,
        "q67": q67,
        "tier_rule": {
            "Low": "A <= q33",
            "Medium": "q33 < A <= q67",
            "High": "A > q67",
        },
        "formal_test_used_to_compute_thresholds": False,
        "chronology_disclosure": {
            "numeric_thresholds_computed_after_primary_formal_test_execution": True,
            "threshold_rule_predeclared_in_experiment_plan": True,
            "formal_test_values_not_used_to_choose_numeric_thresholds": True,
        },
        "max_reconstructed_surrogate_fidelity_abs_diff": (
            val_max_diff
        ),
        "freeze_rule": (
            "Do not change q33/q67 after formal TEST agreement results are inspected."
        ),
    }

    write_json(
        THRESHOLD_CONFIG,
        threshold_payload,
    )

    print(
        f"[Validation] agreement available: "
        f"{len(val_available)}/{len(val_agreement)}"
    )
    print(
        f"[Validation] q33={q33:.6f} | q67={q67:.6f}"
    )
    print(
        "[Validation] max reconstructed surrogate-fidelity abs diff: "
        f"{val_max_diff:.3e}"
    )
    print(
        f"[Validation] thresholds frozen: {THRESHOLD_CONFIG}"
    )
    print()

    # --------------------------------------------------------------
    # B) Formal TEST agreement reconstruction.
    # --------------------------------------------------------------
    test_cache = load_cache_dir(
        TEST_CACHE_DIR
    )

    require(
        len(test_cache) == 1600,
        (
            "Expected 1600 formal TEST cache targets, "
            f"found {len(test_cache)}"
        ),
    )

    print(
        "[Formal TEST] reconstructing surrogate predictions / agreement ..."
    )

    test_agreement, test_max_diff = (
        build_agreement_table(
            test_cache,
            read_csv(
                TEST_METRICS
            ),
            "TEST",
        )
    )

    test_available = [
        r
        for r in test_agreement
        if r[
            "agreement_available"
        ]
    ]

    print(
        f"[Formal TEST] agreement available: "
        f"{len(test_available)}/{len(test_agreement)}"
    )
    print(
        "[Formal TEST] max reconstructed surrogate-fidelity abs diff: "
        f"{test_max_diff:.3e}"
    )
    print()

    # --------------------------------------------------------------
    # C) Join formal TEST Stage2 D-S conflict / quality.
    # --------------------------------------------------------------
    stage_rows = read_csv(
        DS_STAGE
    )
    trace_rows = read_csv(
        DS_TRACE
    )

    detail = build_stage2_analysis(
        test_agreement,
        stage_rows,
        trace_rows,
        q33,
        q67,
    )

    target_agg = aggregate_by_target(
        detail
    )

    tiers = tier_summary(
        detail
    )

    conflict_stage = (
        stage_conflict_summary(
            trace_rows
        )
    )

    # Group-level correlations.
    group_corr = []

    for x, y in [
        (
            "A_mean",
            "mean_K",
        ),
        (
            "A_mean",
            "max_K",
        ),
        (
            "A_mean",
            "Fid_r",
        ),
        (
            "A_mean",
            "cf_valid",
        ),
        (
            "A_mean",
            "R_r",
        ),
        (
            "mean_K",
            "Fid_r",
        ),
        (
            "mean_K",
            "cf_valid",
        ),
        (
            "mean_K",
            "R_r",
        ),
        (
            "mean_K",
            "coverage_eval",
        ),
        (
            "mean_K",
            "n_r",
        ),
    ]:
        group_corr.append(
            corr_record(
                "Stage2_group_level",
                x,
                y,
                detail,
            )
        )

    # Target-level aggregate correlations.
    target_corr = []

    for x, y in [
        (
            "A_mean",
            "mean_group_K",
        ),
        (
            "A_mean",
            "max_group_K",
        ),
        (
            "A_mean",
            "mean_rule_fidelity",
        ),
        (
            "A_mean",
            "cf_validity_rate",
        ),
        (
            "A_mean",
            "mean_reliability",
        ),
        (
            "mean_group_K",
            "mean_rule_fidelity",
        ),
        (
            "mean_group_K",
            "cf_validity_rate",
        ),
        (
            "mean_group_K",
            "mean_coverage",
        ),
        (
            "mean_group_K",
            "mean_support",
        ),
    ]:
        target_corr.append(
            corr_record(
                "Stage2_target_aggregate",
                x,
                y,
                target_agg,
            )
        )

    # Agreement distribution by validation-fixed tier.
    target_tier_counts = []

    for tier in (
        "Low",
        "Medium",
        "High",
    ):
        if tier == "Low":
            rows = [
                r
                for r in test_available
                if r[
                    "A_mean"
                ] <= q33
            ]
        elif tier == "Medium":
            rows = [
                r
                for r in test_available
                if q33
                < r["A_mean"]
                <= q67
            ]
        else:
            rows = [
                r
                for r in test_available
                if r[
                    "A_mean"
                ] > q67
            ]

        target_tier_counts.append({
            "agreement_tier": tier,
            "target_count": len(
                rows
            ),
            "mean_A": mean(
                r["A_mean"]
                for r in rows
            ),
            "median_A": median(
                r["A_mean"]
                for r in rows
            ),
        })

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        OUT
        / "validation_target_agreement.csv",
        val_agreement,
    )

    write_csv(
        OUT
        / "formal_target_agreement.csv",
        test_agreement,
    )

    write_csv(
        OUT
        / "formal_stage2_agreement_conflict_quality_detail.csv",
        detail,
    )

    write_csv(
        OUT
        / "formal_stage2_agreement_tier_summary.csv",
        tiers,
    )

    write_csv(
        OUT
        / "formal_stage2_target_aggregate.csv",
        target_agg,
    )

    write_csv(
        OUT
        / "formal_agreement_conflict_correlations_group.csv",
        group_corr,
    )

    write_csv(
        OUT
        / "formal_agreement_conflict_correlations_target.csv",
        target_corr,
    )

    write_csv(
        OUT
        / "formal_agreement_target_tier_counts.csv",
        target_tier_counts,
    )

    write_csv(
        OUT
        / "formal_conflict_summary_by_stage.csv",
        conflict_stage,
    )

    result = {
        "scope": (
            "ETCE Phase 4D-11 Agreement -> Conflict -> Quality"
        ),
        "status": "passed",
        "formal_test": True,
        "agreement_definition": (
            threshold_payload[
                "definition"
            ]
        ),
        "thresholds": {
            "source": "validation_only",
            "q33": q33,
            "q67": q67,
        },
        "validation_reconstruction": {
            "target_count": len(
                val_agreement
            ),
            "agreement_available": len(
                val_available
            ),
            "max_surrogate_fidelity_abs_diff": (
                val_max_diff
            ),
        },
        "formal_reconstruction": {
            "target_count": len(
                test_agreement
            ),
            "agreement_available": len(
                test_available
            ),
            "max_surrogate_fidelity_abs_diff": (
                test_max_diff
            ),
        },
        "stage2_analysis": {
            "fused_group_rows_with_agreement": len(
                detail
            ),
            "target_aggregate_count": len(
                target_agg
            ),
            "tier_summary": tiers,
            "group_level_correlations": (
                group_corr
            ),
            "target_level_correlations": (
                target_corr
            ),
        },
        "stage_conflict_summary": (
            conflict_stage
        ),
        "interpretation_guard": (
            "A(x) is an empirical output-agreement/dependence proxy. "
            "Do not interpret it as proof of statistical source independence."
        ),
        "test_side_tuning": False,
        "protocol_retuning": False,
        "next_step": (
            "Run correlated-source stress test (LORE+DT+BN vs LORE+DT+DT') "
            "or proceed to process/temporal validity experiment."
        ),
    }

    write_json(
        OUT
        / "agreement_conflict_quality_summary.json",
        result,
    )

    # Human-readable report.
    lines = [
        "# ETCE Agreement–Conflict–Quality Analysis",
        "",
        "Agreement is an empirical surrogate-output agreement/dependence proxy, "
        "not a proof of statistical independence.",
        "",
        f"- Validation q33: {q33:.6f}",
        f"- Validation q67: {q67:.6f}",
        f"- Formal TEST agreement available: {len(test_available)}/{len(test_agreement)}",
        f"- Stage2 genuinely fused groups joined: {len(detail)}",
        f"- Stage2 target aggregates: {len(target_agg)}",
        "",
        "## Agreement-tier summary",
        "",
        "| Tier | N groups | mean A | mean K | Fidelity | CF-valid | R | Coverage | Support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(v):
        return (
            "NA"
            if v is None
            else f"{float(v):.6f}"
        )

    for r in tiers:
        lines.append(
            f"| {r['agreement_tier']} | "
            f"{r.get('N', 0)} | "
            f"{fmt(r.get('mean_agreement'))} | "
            f"{fmt(r.get('mean_K'))} | "
            f"{fmt(r.get('mean_rule_fidelity'))} | "
            f"{fmt(r.get('cf_validity_rate'))} | "
            f"{fmt(r.get('mean_reliability'))} | "
            f"{fmt(r.get('mean_coverage'))} | "
            f"{fmt(r.get('mean_support'))} |"
        )

    lines.extend([
        "",
        "## Group-level correlations",
        "",
        "| X | Y | N | Pearson | Spearman |",
        "|---|---|---:|---:|---:|",
    ])

    for r in group_corr:
        lines.append(
            f"| {r['x']} | {r['y']} | "
            f"{r['N']} | "
            f"{fmt(r['pearson_r'])} | "
            f"{fmt(r['spearman_rho'])} |"
        )

    lines.extend([
        "",
        "## Target-aggregate correlations",
        "",
        "| X | Y | N | Pearson | Spearman |",
        "|---|---|---:|---:|---:|",
    ])

    for r in target_corr:
        lines.append(
            f"| {r['x']} | {r['y']} | "
            f"{r['N']} | "
            f"{fmt(r['pearson_r'])} | "
            f"{fmt(r['spearman_rho'])} |"
        )

    (
        OUT
        / "PHASE4D_AGREEMENT_CONFLICT_QUALITY_REPORT.md"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print("=" * 118)
    print("ETCE PHASE 4D-11 AGREEMENT -> CONFLICT -> QUALITY PASSED")
    print("=" * 118)
    print(
        f"Validation agreement available: "
        f"{len(val_available)}/{len(val_agreement)}"
    )
    print(
        f"Frozen thresholds: q33={q33:.6f} | q67={q67:.6f}"
    )
    print(
        f"Formal agreement available: "
        f"{len(test_available)}/{len(test_agreement)}"
    )
    print(
        f"Stage2 fused groups joined: {len(detail)}"
    )
    print(
        f"Stage2 target aggregates: {len(target_agg)}"
    )
    print()

    print("[Agreement tiers: Stage2 group level]")
    for r in tiers:
        if r.get(
            "N",
            0,
        ) == 0:
            print(
                f"{r['agreement_tier']:8s} | N=0"
            )
            continue

        print(
            f"{r['agreement_tier']:8s} | "
            f"N={r['N']:4d} | "
            f"A={r['mean_agreement']:.6f} | "
            f"K={r['mean_K']:.6f} | "
            f"Fid={r['mean_rule_fidelity']:.6f} | "
            f"CF={r['cf_validity_rate']:.6f} | "
            f"R={r['mean_reliability']:.6f}"
        )

    print()
    print("[Primary group-level correlations]")
    for r in group_corr:
        if (
            r["x"],
            r["y"],
        ) in {
            (
                "A_mean",
                "mean_K",
            ),
            (
                "A_mean",
                "Fid_r",
            ),
            (
                "mean_K",
                "Fid_r",
            ),
            (
                "mean_K",
                "cf_valid",
            ),
        }:
            print(
                f"{r['x']} -> {r['y']} | "
                f"N={r['N']} | "
                f"Pearson={r['pearson_r']} | "
                f"Spearman={r['spearman_rho']}"
            )

    print()
    print("[Primary target-aggregate correlations]")
    for r in target_corr:
        if (
            r["x"],
            r["y"],
        ) in {
            (
                "A_mean",
                "mean_group_K",
            ),
            (
                "A_mean",
                "mean_rule_fidelity",
            ),
            (
                "mean_group_K",
                "mean_rule_fidelity",
            ),
            (
                "mean_group_K",
                "cf_validity_rate",
            ),
        }:
            print(
                f"{r['x']} -> {r['y']} | "
                f"N={r['N']} | "
                f"Pearson={r['pearson_r']} | "
                f"Spearman={r['spearman_rho']}"
            )

    print()
    print("[Stage conflict summary]")
    for r in conflict_stage:
        print(
            f"{r['stage']} | "
            f"pairs={r['pairwise_fusion_count']} | "
            f"meanK={r['mean_K']:.6f} | "
            f"p90K={r['p90_K']:.6f} | "
            f"maxK={r['max_K']:.6f} | "
            f"zeroK={r['zero_K_rate']:.6f}"
        )

    print()
    print(
        f"Output: {OUT}"
    )
    print(
        "Agreement proxy is descriptive, not proof of independence: True"
    )
    print(
        "Formal TEST used to select thresholds: False"
    )
    print(
        "Protocol retuning performed: False"
    )
    print(
        "NEXT: correlated-source stress test or process/temporal validity."
    )


if __name__ == "__main__":
    main()
