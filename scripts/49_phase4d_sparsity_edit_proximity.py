#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-14 / Script 49
Sparsity / Sequence Edit / Proximity evaluation.

Scope
-----
Formal BPIC2017 TEST only. No tuning.

Metrics
-------
The current ETCE formal explanation space is an activity-prefix sequence, so
minimality is measured at activity-position level:

    Sequence Edit Count  = Hamming(x, x_cf)
    Sequence Sparsity    = Hamming(x, x_cf) / prefix_length
    Sequence Proximity   = 1 - Sequence Sparsity

Lower edit / sparsity is better; higher proximity is better.

This script deliberately calls the metric "sequence-position sparsity" rather
than generic attribute sparsity because the current formal representation
modifies activity positions, not arbitrary continuous/tabular attributes.

Populations
-----------
A) Original eligible CF rules (candidate-level descriptive reference)
B) Frozen ETCE D-S Stage1 / Stage2 / Stage3 genuinely fused outputs
C) Stage3 fusion baselines on the SAME 529 genuinely fused Stage3 groups:
   best single, validation-best source, uniform vote, validation-weighted vote,
   fuzzy probabilistic OR, Bayesian categorical noise, and D-S ETCE.

For baseline Stage3 rows, minimal witness edit is independently recomputed on
the frozen held-out formal fusion-evaluation cache using selected_conditions.

Important
---------
- Quality metrics are aggregated only where a valid counterfactual witness
  exists; N_total and N_valid are always reported.
- Original candidate-level results are descriptive and are NOT treated as
  paired observations against later fusion stages.
- Stage3 method comparisons are paired on the exact same compatible groups.
- No lambda / BPA / grouping / fusion strategy is tuned here.
"""

from __future__ import annotations

import ast
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORMAL = ROOT / "results/phase4/phase4d_formal_test"
BANK = FORMAL / "formal_test_multirule_bank"
RULES = BANK / "canonical_rules_test.jsonl"
CACHE_DIR = BANK / "fusion_eval_cache" / "targets"

DS_DIR = FORMAL / "formal_test_frozen_ds_fusion"
DS_STAGE = DS_DIR / "phase4d_frozen_ds_stage_outputs.csv"
DS_VALIDATION = DS_DIR / "validation_phase4d_frozen_ds_fusion.json"

BASELINE_DIR = FORMAL / "formal_test_fusion_baselines"
BASELINE_STAGE = BASELINE_DIR / "formal_fusion_baseline_stage_outputs.csv"

OUT = FORMAL / "formal_test_minimality_proximity"

EXPECTED_ELIGIBLE_CF = 6172
EXPECTED_STAGE1_FUSED = 919
EXPECTED_STAGE2_FUSED = 614
EXPECTED_STAGE3_FUSED = 529

TOL = 1e-12


# =====================================================================
# IO / parsing
# =====================================================================

def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid JSONL at {path}:{line_no}"
                ) from exc
    return rows


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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


def parse_bool(v):
    if isinstance(v, bool):
        return v

    return str(v).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def parse_float(v):
    if v is None:
        return None

    s = str(v).strip()

    if s == "" or s.lower() in {
        "none",
        "null",
        "nan",
    }:
        return None

    return float(s)


def parse_int(v):
    x = parse_float(v)
    return None if x is None else int(round(x))


def first_existing_key(row, names):
    for name in names:
        if name in row:
            return name
    return None


def parse_signature(raw):
    if isinstance(raw, (list, tuple)):
        vals = raw

    else:
        s = str(raw).strip()

        try:
            vals = json.loads(s)
        except Exception:
            vals = ast.literal_eval(s)

    return tuple(
        sorted(
            int(x)
            for x in vals
        )
    )


def signature_key(raw):
    return json.dumps(
        list(
            parse_signature(raw)
        ),
        separators=(",", ":"),
    )


def mean(xs):
    vals = [
        float(x)
        for x in xs
        if x is not None
        and math.isfinite(float(x))
    ]
    return float(sum(vals) / len(vals)) if vals else None


def median(xs):
    vals = sorted(
        float(x)
        for x in xs
        if x is not None
        and math.isfinite(float(x))
    )

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
        vals[lo] * (1.0 - frac)
        + vals[hi] * frac
    )


# =====================================================================
# Counterfactual conditions / cache evaluation
# =====================================================================

def conditions_from_cell(cell):
    if isinstance(cell, list):
        return cell

    s = str(cell).strip()

    if s == "":
        return []

    try:
        obj = json.loads(s)
    except Exception:
        obj = ast.literal_eval(s)

    require(
        isinstance(obj, list),
        f"Conditions must be a list, got: {type(obj)}",
    )

    return obj


def applies(sequence, conditions):
    for cond in conditions:
        require(
            cond.get("operator") == "in",
            f"Unsupported operator: {cond.get('operator')}",
        )

        pos = int(cond["position"]) - 1

        if pos < 0 or pos >= len(sequence):
            return False

        allowed = {
            str(v)
            for v in cond["values"]
        }

        if str(sequence[pos]) not in allowed:
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


def load_cache():
    require(
        CACHE_DIR.is_dir(),
        f"Missing cache directory: {CACHE_DIR}",
    )

    idx = {}

    for fp in sorted(CACHE_DIR.glob("*.json.gz")):
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            p = json.load(f)

        key = (
            str(p["case_id"]),
            str(p["model"]),
            int(p["prefix"]),
        )

        require(
            key not in idx,
            f"Duplicate cache target: {key}",
        )

        idx[key] = p

    require(
        len(idx) == 1600,
        f"Expected 1600 formal cache targets, found {len(idx)}",
    )

    return idx


def minimal_valid_witness_edit(
    conditions,
    predicted_class,
    contexts,
):
    best = None

    for p in contexts:
        factual = tuple(
            p["factual_sequence"]
        )

        sequences = p[
            "pool_sequences"
        ]

        labels = p[
            "pool_labels"
        ]

        for raw_i in p[
            "eval_indices"
        ]:
            i = int(raw_i)
            seq = tuple(
                sequences[i]
            )

            if not applies(
                seq,
                conditions,
            ):
                continue

            if int(labels[i]) != int(predicted_class):
                continue

            d = hamming(
                factual,
                seq,
            )

            if best is None or d < best:
                best = d

    return best


# =====================================================================
# Minimality records
# =====================================================================

def metric_record(
    source,
    method,
    stage,
    case_id,
    prefix,
    edit_count,
    cf_valid,
    model=None,
    surrogate=None,
    predicted_class=None,
    feature_signature=None,
    paired_stage3_key=None,
):
    prefix = int(prefix)

    edit = (
        None
        if edit_count is None
        else float(edit_count)
    )

    valid = bool(
        cf_valid
        and edit is not None
    )

    sparsity = (
        edit / prefix
        if valid
        else None
    )

    proximity = (
        1.0 - sparsity
        if valid
        else None
    )

    return {
        "source": source,
        "method": method,
        "stage": stage,
        "case_id": case_id,
        "model": model,
        "surrogate": surrogate,
        "prefix": prefix,
        "predicted_class": predicted_class,
        "feature_signature": feature_signature,
        "cf_valid": int(bool(cf_valid)),
        "witness_available": int(valid),
        "sequence_edit_count": edit,
        "sequence_position_sparsity": sparsity,
        "normalized_hamming_proximity": proximity,
        "paired_stage3_key": paired_stage3_key,
    }


def summarize(records, label_fields=("method", "stage")):
    groups = defaultdict(list)

    for r in records:
        key = tuple(
            r.get(f)
            for f in label_fields
        )
        groups[key].append(r)

    out = []

    for key, rows in sorted(
        groups.items(),
        key=lambda kv: str(kv[0]),
    ):
        valid = [
            r
            for r in rows
            if int(
                r[
                    "witness_available"
                ]
            ) == 1
        ]

        edits = [
            r[
                "sequence_edit_count"
            ]
            for r in valid
        ]

        sparsity = [
            r[
                "sequence_position_sparsity"
            ]
            for r in valid
        ]

        proximity = [
            r[
                "normalized_hamming_proximity"
            ]
            for r in valid
        ]

        row = {
            field: value
            for field, value in zip(
                label_fields,
                key,
            )
        }

        row.update({
            "N_total": len(rows),
            "N_valid_witness": len(valid),
            "valid_witness_rate": (
                len(valid) / len(rows)
                if rows
                else None
            ),
            "mean_edit": mean(edits),
            "median_edit": median(edits),
            "p90_edit": quantile(
                edits,
                0.90,
            ),
            "edit_eq_1_rate": (
                mean(
                    int(
                        float(x)
                        <= 1.0 + TOL
                    )
                    for x in edits
                )
                if edits
                else None
            ),
            "edit_le_2_rate": (
                mean(
                    int(
                        float(x)
                        <= 2.0 + TOL
                    )
                    for x in edits
                )
                if edits
                else None
            ),
            "mean_sequence_position_sparsity": mean(
                sparsity
            ),
            "median_sequence_position_sparsity": median(
                sparsity
            ),
            "mean_normalized_hamming_proximity": mean(
                proximity
            ),
            "median_normalized_hamming_proximity": median(
                proximity
            ),
        })

        out.append(row)

    return out


# =====================================================================
# Stage3 identity / baseline reconstruction
# =====================================================================

def stage3_key_from_row(r):
    return (
        str(r["case_id"]),
        int(float(r["prefix"])),
        int(float(r["predicted_class"])),
        signature_key(
            r["feature_signature"]
        ),
    )


def stage2_key_from_row(r):
    return (
        str(r["case_id"]),
        str(r["model"]),
        int(float(r["prefix"])),
        int(float(r["predicted_class"])),
        signature_key(
            r["feature_signature"]
        ),
    )


def build_stage3_context_models(ds_rows):
    """
    Map each Stage3 group to the exact Stage2 model contexts that feed it.
    """
    out = defaultdict(list)

    for r in ds_rows:
        if r["stage"] != "Stage2":
            continue

        key = (
            str(r["case_id"]),
            int(float(r["prefix"])),
            int(float(r["predicted_class"])),
            signature_key(
                r["feature_signature"]
            ),
        )

        out[key].append(
            str(r["model"])
        )

    return {
        k: sorted(set(v))
        for k, v in out.items()
    }


def locate_baseline_columns(rows):
    require(
        rows,
        f"Baseline stage file is empty: {BASELINE_STAGE}",
    )

    sample = rows[0]

    method_col = first_existing_key(
        sample,
        (
            "strategy",
            "method",
            "fusion_strategy",
            "baseline",
        ),
    )

    stage_col = first_existing_key(
        sample,
        (
            "stage",
            "Stage",
        ),
    )

    cond_col = first_existing_key(
        sample,
        (
            "selected_conditions",
            "conditions",
            "rule_conditions",
        ),
    )

    require(
        method_col is not None,
        (
            "Could not identify baseline strategy column. "
            f"Available columns: {list(sample)}"
        ),
    )

    require(
        stage_col is not None,
        (
            "Could not identify baseline stage column. "
            f"Available columns: {list(sample)}"
        ),
    )

    require(
        cond_col is not None,
        (
            "Could not identify baseline selected-condition column. "
            f"Available columns: {list(sample)}"
        ),
    )

    return (
        method_col,
        stage_col,
        cond_col,
    )


# =====================================================================
# Paired Stage3 comparisons
# =====================================================================

def wilcoxon_or_sign(diffs):
    nz = [
        float(x)
        for x in diffs
        if abs(float(x)) > TOL
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
        # Deterministic two-sided sign-test fallback.
        pos = sum(
            x > 0
            for x in nz
        )

        n = len(nz)
        lo = min(
            pos,
            n - pos,
        )

        p = (
            2.0
            * sum(
                math.comb(n, i)
                for i in range(lo + 1)
            )
            / (2 ** n)
        )

        return {
            "test": "exact_sign_test_fallback",
            "p_value": min(
                1.0,
                p,
            ),
        }


def paired_compare(
    ds_map,
    baseline_map,
    metric,
):
    common = sorted(
        set(ds_map)
        & set(baseline_map)
    )

    usable = [
        k
        for k in common
        if ds_map[k].get(metric)
        is not None
        and baseline_map[k].get(metric)
        is not None
    ]

    dsv = [
        float(
            ds_map[k][
                metric
            ]
        )
        for k in usable
    ]

    bv = [
        float(
            baseline_map[k][
                metric
            ]
        )
        for k in usable
    ]

    diffs = [
        a - b
        for a, b in zip(
            dsv,
            bv,
        )
    ]

    test = wilcoxon_or_sign(
        diffs
    )

    return {
        "N_matched_groups": len(
            common
        ),
        "N_usable": len(
            usable
        ),
        "ds_mean": mean(
            dsv
        ),
        "baseline_mean": mean(
            bv
        ),
        "delta_ds_minus_baseline": mean(
            diffs
        ),
        "ds_wins_lower_is_better": (
            sum(
                d < -TOL
                for d in diffs
            )
            if metric
            != "normalized_hamming_proximity"
            else sum(
                d > TOL
                for d in diffs
            )
        ),
        "ties": sum(
            abs(d) <= TOL
            for d in diffs
        ),
        "baseline_wins": (
            sum(
                d > TOL
                for d in diffs
            )
            if metric
            != "normalized_hamming_proximity"
            else sum(
                d < -TOL
                for d in diffs
            )
        ),
        "paired_test": test[
            "test"
        ],
        "p_value": test[
            "p_value"
        ],
    }


# =====================================================================
# Main
# =====================================================================

def main():
    required = [
        RULES,
        DS_STAGE,
        DS_VALIDATION,
        BASELINE_STAGE,
    ]

    missing = [
        str(p)
        for p in required
        if not p.is_file()
    ]

    require(
        not missing,
        "Missing required input(s):\n"
        + "\n".join(
            missing
        ),
    )

    ds_validation = read_json(
        DS_VALIDATION
    )

    require(
        ds_validation.get(
            "status"
        ) == "passed",
        "Frozen formal D-S validation not PASSED",
    )

    cache = load_cache()
    rules = read_jsonl(
        RULES
    )
    ds_rows = read_csv(
        DS_STAGE
    )
    baseline_rows = read_csv(
        BASELINE_STAGE
    )

    print("=" * 118)
    print("ETCE PHASE 4D-14 SPARSITY / SEQUENCE EDIT / PROXIMITY")
    print("=" * 118)
    print("Metric space             : activity positions in fixed-length prefixes")
    print("Sequence edit            : Hamming(x, x_cf)")
    print("Sequence-position sparsity: edit / prefix_length")
    print("Normalized proximity     : 1 - edit / prefix_length")
    print("Quality aggregation      : valid witness rows only; N_total and N_valid reported")
    print("Formal TEST tuning       : False")
    print("Protocol retuning        : False")
    print()

    detail = []

    # --------------------------------------------------------------
    # A) Original eligible CF candidates.
    # --------------------------------------------------------------
    eligible = [
        r
        for r in rules
        if str(
            r.get(
                "rule_type"
            )
        ) == "counterfactual"
        and parse_bool(
            r.get(
                "eligible_for_reliability"
            )
        )
    ]

    require(
        len(eligible)
        == EXPECTED_ELIGIBLE_CF,
        (
            f"Expected {EXPECTED_ELIGIBLE_CF} eligible CF candidates, "
            f"found {len(eligible)}"
        ),
    )

    for r in eligible:
        edit = parse_float(
            r.get(
                "witness_edit_count"
            )
        )

        detail.append(
            metric_record(
                source=(
                    "canonical_rule_bank"
                ),
                method=(
                    "Original eligible CF"
                ),
                stage=(
                    "Original"
                ),
                case_id=str(
                    r["case_id"]
                ),
                model=str(
                    r["model"]
                ),
                surrogate=str(
                    r["surrogate"]
                ),
                prefix=int(
                    r["prefix"]
                ),
                predicted_class=int(
                    r[
                        "predicted_class"
                    ]
                ),
                feature_signature=(
                    json.dumps(
                        sorted({
                            int(c["position"])
                            for c in r[
                                "conditions"
                            ]
                        })
                    )
                ),
                edit_count=edit,
                cf_valid=(
                    edit is not None
                ),
            )
        )

    # --------------------------------------------------------------
    # B) D-S Stage1 / Stage2 / Stage3 genuinely fused.
    # --------------------------------------------------------------
    expected_fused = {
        "Stage1": (
            EXPECTED_STAGE1_FUSED
        ),
        "Stage2": (
            EXPECTED_STAGE2_FUSED
        ),
        "Stage3": (
            EXPECTED_STAGE3_FUSED
        ),
    }

    ds_stage3_map = {}

    for stage in (
        "Stage1",
        "Stage2",
        "Stage3",
    ):
        rows = [
            r
            for r in ds_rows
            if r[
                "stage"
            ] == stage
            and parse_bool(
                r[
                    "actually_fused"
                ]
            )
        ]

        require(
            len(rows)
            == expected_fused[
                stage
            ],
            (
                f"{stage} fused count mismatch: "
                f"expected {expected_fused[stage]}, found {len(rows)}"
            ),
        )

        for r in rows:
            edit = parse_float(
                r.get(
                    "min_witness_edit"
                )
            )

            pair_key = None

            if stage == "Stage3":
                pair_key = json.dumps(
                    stage3_key_from_row(
                        r
                    ),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

            rec = metric_record(
                source=(
                    "frozen_ds_stage_output"
                ),
                method=(
                    "D-S ETCE"
                ),
                stage=stage,
                case_id=str(
                    r["case_id"]
                ),
                model=(
                    r.get(
                        "model"
                    )
                    or None
                ),
                surrogate=(
                    r.get(
                        "surrogate"
                    )
                    or None
                ),
                prefix=int(
                    float(
                        r[
                            "prefix"
                        ]
                    )
                ),
                predicted_class=int(
                    float(
                        r[
                            "predicted_class"
                        ]
                    )
                ),
                feature_signature=(
                    signature_key(
                        r[
                            "feature_signature"
                        ]
                    )
                ),
                edit_count=edit,
                cf_valid=parse_bool(
                    r[
                        "cf_valid"
                    ]
                ),
                paired_stage3_key=(
                    pair_key
                ),
            )

            detail.append(
                rec
            )

            if stage == "Stage3":
                key = stage3_key_from_row(
                    r
                )

                require(
                    key
                    not in ds_stage3_map,
                    f"Duplicate DS Stage3 key: {key}",
                )

                ds_stage3_map[
                    key
                ] = rec

    require(
        len(
            ds_stage3_map
        ) == EXPECTED_STAGE3_FUSED,
        "D-S Stage3 matched-group inventory mismatch",
    )

    # --------------------------------------------------------------
    # C) Stage3 baseline methods on exact 529 fused groups.
    # --------------------------------------------------------------
    method_col, stage_col, cond_col = (
        locate_baseline_columns(
            baseline_rows
        )
    )

    stage3_context_models = (
        build_stage3_context_models(
            ds_rows
        )
    )

    baseline_stage3 = [
        r
        for r in baseline_rows
        if str(
            r[
                stage_col
            ]
        ).strip().lower()
        == "stage3"
    ]

    require(
        baseline_stage3,
        "No Stage3 rows found in baseline output",
    )

    # Normalize methods and collect.
    baseline_maps = defaultdict(
        dict
    )

    baseline_missing_groups = defaultdict(
        int
    )

    for idx, r in enumerate(
        baseline_stage3,
        start=1,
    ):
        method = str(
            r[
                method_col
            ]
        ).strip()

        # Avoid duplicating D-S if baseline output happens to include it.
        if method.lower() in {
            "ds_etce",
            "d-s etce",
            "d_s_etce",
            "ds",
        }:
            continue

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
            int(
                float(
                    r[
                        "predicted_class"
                    ]
                )
            ),
            signature_key(
                r[
                    "feature_signature"
                ]
            ),
        )

        # We only compare the exact formal 529 genuinely fused Stage3 groups.
        if key not in ds_stage3_map:
            continue

        models = (
            stage3_context_models.get(
                key,
                []
            )
        )

        require(
            len(models) >= 2,
            (
                "Fused Stage3 baseline group has "
                f"<2 model contexts: {key}"
            ),
        )

        contexts = [
            cache[
                (
                    key[0],
                    model,
                    key[1],
                )
            ]
            for model in models
        ]

        conditions = (
            conditions_from_cell(
                r[
                    cond_col
                ]
            )
        )

        edit = minimal_valid_witness_edit(
            conditions,
            key[2],
            contexts,
        )

        # If the baseline export has an explicit validity flag, honor it.
        cf_col = first_existing_key(
            r,
            (
                "cf_valid",
                "cf_validity",
                "output_cf_valid",
            ),
        )

        cf_valid = (
            edit is not None
            if cf_col is None
            else parse_bool(
                r[
                    cf_col
                ]
            )
        )

        rec = metric_record(
            source=(
                "formal_fusion_baseline_stage_output"
            ),
            method=method,
            stage="Stage3",
            case_id=key[0],
            model=None,
            surrogate=None,
            prefix=key[1],
            predicted_class=key[2],
            feature_signature=key[3],
            edit_count=edit,
            cf_valid=cf_valid,
            paired_stage3_key=json.dumps(
                key,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

        require(
            key
            not in baseline_maps[
                method
            ],
            (
                f"Duplicate baseline Stage3 group "
                f"for method={method}: {key}"
            ),
        )

        baseline_maps[
            method
        ][
            key
        ] = rec

        detail.append(
            rec
        )

    require(
        baseline_maps,
        "No usable non-D-S Stage3 baseline methods were reconstructed",
    )

    print(
        "[Baseline Stage3 inventories]"
    )

    for method in sorted(
        baseline_maps
    ):
        n = len(
            baseline_maps[
                method
            ]
        )

        print(
            f"{method:32s} | "
            f"matched={n}/{EXPECTED_STAGE3_FUSED}"
        )

        require(
            n == EXPECTED_STAGE3_FUSED,
            (
                f"Baseline method {method} does not cover "
                f"all {EXPECTED_STAGE3_FUSED} Stage3 groups"
            ),
        )

    # --------------------------------------------------------------
    # Summary tables.
    # --------------------------------------------------------------
    main_summary = summarize(
        detail,
        label_fields=(
            "method",
            "stage",
        ),
    )

    by_prefix = summarize(
        detail,
        label_fields=(
            "method",
            "stage",
            "prefix",
        ),
    )

    # Paired comparisons: D-S vs every non-D-S Stage3 strategy.
    paired = []

    for method in sorted(
        baseline_maps
    ):
        bmap = baseline_maps[
            method
        ]

        for metric in (
            "sequence_edit_count",
            "sequence_position_sparsity",
            "normalized_hamming_proximity",
        ):
            paired.append({
                "baseline_method": method,
                "metric": metric,
                **paired_compare(
                    ds_stage3_map,
                    bmap,
                    metric,
                ),
            })

    # --------------------------------------------------------------
    # Save.
    # --------------------------------------------------------------
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        OUT
        / "minimality_proximity_detail.csv",
        detail,
    )

    write_csv(
        OUT
        / "minimality_proximity_summary.csv",
        main_summary,
    )

    write_csv(
        OUT
        / "minimality_proximity_by_prefix.csv",
        by_prefix,
    )

    write_csv(
        OUT
        / "stage3_minimality_paired_comparison.csv",
        paired,
    )

    result = {
        "scope": (
            "ETCE Phase 4D-14 sparsity / sequence edit / proximity"
        ),
        "status": "passed",
        "formal_test": True,
        "metric_definition": {
            "sequence_edit_count": (
                "minimum Hamming distance between factual prefix "
                "and a held-out valid counterfactual witness"
            ),
            "sequence_position_sparsity": (
                "sequence_edit_count / prefix_length"
            ),
            "normalized_hamming_proximity": (
                "1 - sequence_position_sparsity"
            ),
            "terminology_guard": (
                "This is sequence-position sparsity, not generic "
                "continuous/tabular attribute sparsity."
            ),
        },
        "quality_denominator": (
            "valid counterfactual witnesses only; N_total and N_valid are reported"
        ),
        "inventories": {
            "eligible_original_cf": len(
                eligible
            ),
            "stage1_fused": (
                EXPECTED_STAGE1_FUSED
            ),
            "stage2_fused": (
                EXPECTED_STAGE2_FUSED
            ),
            "stage3_fused": (
                EXPECTED_STAGE3_FUSED
            ),
            "baseline_stage3_methods": {
                method: len(rows)
                for method, rows
                in baseline_maps.items()
            },
        },
        "main_summary": (
            main_summary
        ),
        "paired_stage3": (
            paired
        ),
        "test_side_tuning": False,
        "protocol_retuning": False,
        "interpretation_guard": (
            "Original candidate-level rows are descriptive and are not "
            "paired directly against fused-stage rows. Stage3 method "
            "comparisons are paired on the same 529 compatible groups."
        ),
        "next_step": (
            "Runtime and scalability, then stability/diversity and "
            "external-dataset validation."
        ),
    }

    write_json(
        OUT
        / "minimality_proximity_summary.json",
        result,
    )

    # Paper-readable markdown.
    def fmt(v):
        if v is None:
            return "NA"
        return f"{float(v):.6f}"

    lines = [
        "# ETCE Formal Minimality / Sparsity / Proximity",
        "",
        "Metrics are defined in the activity-position explanation space.",
        "",
        "- Sequence Edit = Hamming(x, x_cf)",
        "- Sequence-position Sparsity = Edit / prefix length",
        "- Normalized Hamming Proximity = 1 - Sparsity",
        "",
        "Quality metrics use only valid counterfactual witnesses; total and valid counts are both reported.",
        "",
        "## Main summary",
        "",
        "| Method | Stage | N total | N valid | Mean edit | Median edit | Mean sparsity | Mean proximity | Edit=1 | Edit<=2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in main_summary:
        lines.append(
            f"| {r['method']} | {r['stage']} | "
            f"{r['N_total']} | "
            f"{r['N_valid_witness']} | "
            f"{fmt(r['mean_edit'])} | "
            f"{fmt(r['median_edit'])} | "
            f"{fmt(r['mean_sequence_position_sparsity'])} | "
            f"{fmt(r['mean_normalized_hamming_proximity'])} | "
            f"{fmt(r['edit_eq_1_rate'])} | "
            f"{fmt(r['edit_le_2_rate'])} |"
        )

    (
        OUT
        / "PHASE4D_MINIMALITY_PROXIMITY_REPORT.md"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # Console report.
    # --------------------------------------------------------------
    print()
    print("=" * 118)
    print("ETCE PHASE 4D-14 SPARSITY / SEQUENCE EDIT / PROXIMITY PASSED")
    print("=" * 118)

    print("[Main summary]")

    preferred_order = [
        (
            "Original eligible CF",
            "Original",
        ),
        (
            "D-S ETCE",
            "Stage1",
        ),
        (
            "D-S ETCE",
            "Stage2",
        ),
        (
            "D-S ETCE",
            "Stage3",
        ),
    ]

    summary_idx = {
        (
            r[
                "method"
            ],
            r[
                "stage"
            ],
        ): r
        for r in main_summary
    }

    printed = set()

    for key in preferred_order:
        r = summary_idx.get(
            key
        )

        if r is None:
            continue

        printed.add(
            key
        )

        print(
            f"{r['method']:32s} | "
            f"{r['stage']:8s} | "
            f"N={r['N_total']:4d} | "
            f"Valid={r['N_valid_witness']:4d} | "
            f"Edit={r['mean_edit']:.6f} | "
            f"Sparsity={r['mean_sequence_position_sparsity']:.6f} | "
            f"Proximity={r['mean_normalized_hamming_proximity']:.6f} | "
            f"Edit1={r['edit_eq_1_rate']:.6f} | "
            f"Edit<=2={r['edit_le_2_rate']:.6f}"
        )

    for key, r in sorted(
        summary_idx.items(),
        key=lambda kv: str(kv[0]),
    ):
        if key in printed:
            continue

        if r["stage"] != "Stage3":
            continue

        print(
            f"{r['method']:32s} | "
            f"{r['stage']:8s} | "
            f"N={r['N_total']:4d} | "
            f"Valid={r['N_valid_witness']:4d} | "
            f"Edit={r['mean_edit']:.6f} | "
            f"Sparsity={r['mean_sequence_position_sparsity']:.6f} | "
            f"Proximity={r['mean_normalized_hamming_proximity']:.6f} | "
            f"Edit1={r['edit_eq_1_rate']:.6f} | "
            f"Edit<=2={r['edit_le_2_rate']:.6f}"
        )

    print()
    print("[Stage3 paired D-S minus baseline]")

    for r in paired:
        print(
            f"{r['baseline_method']:32s} | "
            f"{r['metric']:34s} | "
            f"N={r['N_usable']:3d} | "
            f"DS={r['ds_mean']:.6f} | "
            f"Base={r['baseline_mean']:.6f} | "
            f"Delta={r['delta_ds_minus_baseline']:+.6f} | "
            f"W/T/L={r['ds_wins_lower_is_better']}/"
            f"{r['ties']}/"
            f"{r['baseline_wins']} | "
            f"p={r['p_value']:.6g}"
        )

    print()
    print(f"Output: {OUT}")
    print("Metric called generic attribute sparsity: False")
    print("Stage3 methods paired on same 529 groups: True")
    print("Formal TEST tuning performed: False")
    print("Protocol retuning performed: False")
    print("NEXT: runtime / scalability.")


if __name__ == "__main__":
    main()
