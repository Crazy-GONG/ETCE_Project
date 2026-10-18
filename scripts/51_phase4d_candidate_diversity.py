#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-16 / Script 51
Formal candidate-diversity analysis.

Purpose
-------
Complete the diversity part of the revised evaluation using the CURRENT
formal BPIC2017 rule bank. No new model fitting or TEST tuning is performed.

Population
----------
All 6,172 eligible counterfactual candidates in the frozen formal TEST
canonical rule bank.

Target-local grouping
---------------------
case_id × model × prefix

For every target with >= 2 unique counterfactual witnesses, report:

1) Candidate count
2) Unique witness rate
3) Unique changed-position-set rate
4) Mean pairwise normalized Hamming distance between unique witnesses
5) Mean pairwise Jaccard distance between changed-position sets
6) Number of surrogate sources represented

Definitions
-----------
For factual prefix x and witness x_cf:

    ChangedPos(x_cf) = {j : x_j != x_cf_j}

For two witnesses a,b of the same prefix length k:

    WitnessDiversity(a,b) = Hamming(a,b) / k

For changed-position sets Sa,Sb:

    ChangeSetDiversity(a,b) =
        1 - |Sa ∩ Sb| / |Sa ∪ Sb|

When both sets are empty (should not happen for eligible CF witnesses),
distance is defined as 0.

Important
---------
- Diversity is descriptive. Higher diversity is NOT automatically better.
- The analysis describes the candidate set available before final fusion /
  representative-rule selection.
- No claim is made that these 1,600 targets are statistically independent
  because the same case appears across models and prefixes.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORMAL = ROOT / "results/phase4/phase4d_formal_test"
BANK = FORMAL / "formal_test_multirule_bank"

RULES = BANK / "canonical_rules_test.jsonl"
SELECTION = FORMAL / "formal_test_case_selection.csv"

OUT = FORMAL / "formal_test_candidate_diversity"

EXPECTED_ELIGIBLE = 6172
EXPECTED_TARGETS = 1600

MODELS = (
    "RF",
    "XGBoost",
    "MLP",
    "LSTM",
)

PREFIXES = (
    5,
    10,
    15,
    20,
)


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


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
                    f"Invalid JSONL at line {line_no}: {path}"
                ) from exc

    return rows


def read_csv(path):
    with Path(path).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(
            csv.DictReader(f)
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


def q(xs, p):
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
        return vals[
            lo
        ]

    frac = z - lo

    return float(
        vals[lo]
        * (1.0 - frac)
        + vals[hi]
        * frac
    )


def witness_tuple(raw):
    require(
        raw is not None,
        "Eligible CF candidate missing witness_sequence",
    )

    if isinstance(
        raw,
        list,
    ):
        seq = raw

    elif isinstance(
        raw,
        tuple,
    ):
        seq = list(raw)

    elif isinstance(
        raw,
        str,
    ):
        s = raw.strip()

        require(
            s != "",
            "Empty witness_sequence",
        )

        try:
            seq = json.loads(
                s
            )
        except Exception:
            # Canonical bank normally stores JSON arrays directly.
            # Keep a narrow compatibility fallback.
            import ast

            seq = ast.literal_eval(
                s
            )

    else:
        raise TypeError(
            f"Unsupported witness_sequence type: {type(raw)}"
        )

    require(
        isinstance(
            seq,
            (list, tuple),
        ),
        "Parsed witness_sequence is not sequence-like",
    )

    return tuple(
        str(x)
        for x in seq
    )


def factual_from_rule(rule):
    """
    No factual sequence is needed from another file.

    For a canonical CF candidate, changed positions can be reconstructed
    from the saved conditions only if they specify full equality constraints,
    which is not generally guaranteed. Therefore we use witness-to-witness
    diversity directly, while changed-position sets are taken from the
    candidate's canonical condition positions as the rule-change footprint.

    This is intentionally named rule-position footprint diversity rather than
    factual-edit-set diversity.
    """
    return None


def footprint(rule):
    conds = rule.get(
        "conditions",
        []
    )

    require(
        isinstance(
            conds,
            list,
        ),
        "Canonical conditions must be a list",
    )

    return frozenset(
        int(
            c[
                "position"
            ]
        )
        for c in conds
        if isinstance(
            c,
            dict,
        )
        and c.get(
            "position"
        )
        is not None
    )


def norm_hamming(a, b):
    require(
        len(a) == len(b),
        "Witness length mismatch",
    )

    if not a:
        return 0.0

    d = sum(
        x != y
        for x, y in zip(
            a,
            b,
        )
    )

    return float(
        d / len(a)
    )


def jaccard_distance(
    a,
    b,
):
    union = (
        set(a)
        | set(b)
    )

    if not union:
        return 0.0

    inter = (
        set(a)
        & set(b)
    )

    return float(
        1.0
        - len(inter)
        / len(union)
    )


def target_key(rule):
    return (
        str(
            rule[
                "case_id"
            ]
        ),
        str(
            rule[
                "model"
            ]
        ),
        int(
            rule[
                "prefix"
            ]
        ),
    )


def summarize_target(
    case_id,
    model,
    prefix,
    candidates,
):
    prefix = int(
        prefix
    )

    witness_list = [
        witness_tuple(
            r[
                "witness_sequence"
            ]
        )
        for r in candidates
    ]

    for w in witness_list:
        require(
            len(w) == prefix,
            (
                f"Witness length {len(w)} != prefix {prefix} "
                f"for {(case_id, model, prefix)}"
            ),
        )

    footprints = [
        footprint(r)
        for r in candidates
    ]

    unique_witnesses = sorted(
        set(
            witness_list
        )
    )

    unique_footprints = sorted(
        set(
            footprints
        ),
        key=lambda x: (
            len(x),
            tuple(
                sorted(x)
            ),
        ),
    )

    witness_pair_dist = [
        norm_hamming(
            a,
            b,
        )
        for a, b
        in combinations(
            unique_witnesses,
            2,
        )
    ]

    footprint_pair_dist = [
        jaccard_distance(
            a,
            b,
        )
        for a, b
        in combinations(
            unique_footprints,
            2,
        )
    ]

    surrogates = sorted({
        str(
            r[
                "surrogate"
            ]
        )
        for r in candidates
    })

    predicted_classes = sorted({
        int(
            r[
                "predicted_class"
            ]
        )
        for r in candidates
    })

    edit_counts = [
        float(
            r[
                "witness_edit_count"
            ]
        )
        for r in candidates
        if r.get(
            "witness_edit_count"
        )
        is not None
    ]

    return {
        "case_id": (
            case_id
        ),
        "model": model,
        "prefix": prefix,
        "candidate_count": len(
            candidates
        ),
        "unique_witness_count": len(
            unique_witnesses
        ),
        "unique_witness_rate": (
            len(
                unique_witnesses
            )
            / len(
                candidates
            )
        ),
        "unique_rule_footprint_count": len(
            unique_footprints
        ),
        "unique_rule_footprint_rate": (
            len(
                unique_footprints
            )
            / len(
                candidates
            )
        ),
        "surrogate_source_count": len(
            surrogates
        ),
        "surrogate_sources": (
            "|".join(
                surrogates
            )
        ),
        "predicted_class_count": len(
            predicted_classes
        ),
        "mean_witness_edit_count": mean(
            edit_counts
        ),
        "mean_pairwise_witness_hamming": (
            mean(
                witness_pair_dist
            )
        ),
        "median_pairwise_witness_hamming": (
            median(
                witness_pair_dist
            )
        ),
        "p90_pairwise_witness_hamming": (
            q(
                witness_pair_dist,
                0.90,
            )
        ),
        "mean_pairwise_rule_footprint_jaccard": (
            mean(
                footprint_pair_dist
            )
        ),
        "median_pairwise_rule_footprint_jaccard": (
            median(
                footprint_pair_dist
            )
        ),
        "diversity_defined_witness": bool(
            len(
                unique_witnesses
            )
            >= 2
        ),
        "diversity_defined_footprint": bool(
            len(
                unique_footprints
            )
            >= 2
        ),
    }


def aggregate(
    rows,
    label,
    model=None,
    prefix=None,
):
    sub = [
        r
        for r in rows
        if (
            model is None
            or r[
                "model"
            ] == model
        )
        and (
            prefix is None
            or int(
                r[
                    "prefix"
                ]
            )
            == int(
                prefix
            )
        )
    ]

    witness_def = [
        r
        for r in sub
        if r[
            "diversity_defined_witness"
        ]
    ]

    footprint_def = [
        r
        for r in sub
        if r[
            "diversity_defined_footprint"
        ]
    ]

    return {
        "scope": label,
        "model": (
            ""
            if model is None
            else model
        ),
        "prefix": (
            ""
            if prefix is None
            else int(
                prefix
            )
        ),
        "target_count": len(
            sub
        ),
        "targets_with_ge2_unique_witnesses": len(
            witness_def
        ),
        "targets_with_ge2_unique_footprints": len(
            footprint_def
        ),
        "mean_candidate_count": mean(
            r[
                "candidate_count"
            ]
            for r in sub
        ),
        "median_candidate_count": median(
            r[
                "candidate_count"
            ]
            for r in sub
        ),
        "mean_unique_witness_rate": mean(
            r[
                "unique_witness_rate"
            ]
            for r in sub
        ),
        "mean_unique_rule_footprint_rate": mean(
            r[
                "unique_rule_footprint_rate"
            ]
            for r in sub
        ),
        "mean_surrogate_source_count": mean(
            r[
                "surrogate_source_count"
            ]
            for r in sub
        ),
        "mean_pairwise_witness_hamming": mean(
            r[
                "mean_pairwise_witness_hamming"
            ]
            for r in witness_def
        ),
        "median_target_pairwise_witness_hamming": median(
            r[
                "mean_pairwise_witness_hamming"
            ]
            for r in witness_def
        ),
        "mean_pairwise_rule_footprint_jaccard": mean(
            r[
                "mean_pairwise_rule_footprint_jaccard"
            ]
            for r in footprint_def
        ),
        "mean_witness_edit_count": mean(
            r[
                "mean_witness_edit_count"
            ]
            for r in sub
        ),
    }


def main():
    require(
        RULES.is_file(),
        f"Missing rule bank: {RULES}",
    )

    require(
        SELECTION.is_file(),
        f"Missing formal selection: {SELECTION}",
    )

    rules = (
        read_jsonl(
            RULES
        )
    )

    eligible = [
        r
        for r in rules
        if (
            r.get(
                "rule_type"
            )
            == "counterfactual"
            and parse_bool(
                r.get(
                    "eligible_for_reliability"
                )
            )
        )
    ]

    require(
        len(
            eligible
        )
        == EXPECTED_ELIGIBLE,
        (
            f"Expected {EXPECTED_ELIGIBLE} eligible CF, "
            f"found {len(eligible)}"
        ),
    )

    selected_rows = (
        read_csv(
            SELECTION
        )
    )

    case_ids = [
        str(
            r[
                "case_id"
            ]
        ).strip()
        for r in selected_rows
    ]

    require(
        len(
            case_ids
        ) == 100,
        "Formal selection must contain 100 cases",
    )

    all_target_keys = [
        (
            case_id,
            model,
            prefix,
        )
        for case_id in case_ids
        for model in MODELS
        for prefix in PREFIXES
    ]

    require(
        len(
            all_target_keys
        )
        == EXPECTED_TARGETS,
        "Expected 1600 target identities",
    )

    grouped = defaultdict(
        list
    )

    for r in eligible:
        grouped[
            target_key(
                r
            )
        ].append(
            r
        )

    target_rows = []

    for key in all_target_keys:
        (
            case_id,
            model,
            prefix,
        ) = key

        candidates = grouped.get(
            key,
            [],
        )

        if not candidates:
            target_rows.append({
                "case_id": case_id,
                "model": model,
                "prefix": int(
                    prefix
                ),
                "candidate_count": 0,
                "unique_witness_count": 0,
                "unique_witness_rate": None,
                "unique_rule_footprint_count": 0,
                "unique_rule_footprint_rate": None,
                "surrogate_source_count": 0,
                "surrogate_sources": "",
                "predicted_class_count": 0,
                "mean_witness_edit_count": None,
                "mean_pairwise_witness_hamming": None,
                "median_pairwise_witness_hamming": None,
                "p90_pairwise_witness_hamming": None,
                "mean_pairwise_rule_footprint_jaccard": None,
                "median_pairwise_rule_footprint_jaccard": None,
                "diversity_defined_witness": False,
                "diversity_defined_footprint": False,
            })

            continue

        target_rows.append(
            summarize_target(
                case_id,
                model,
                prefix,
                candidates,
            )
        )

    require(
        len(
            target_rows
        )
        == EXPECTED_TARGETS,
        (
            f"Expected {EXPECTED_TARGETS} target rows, "
            f"found {len(target_rows)}"
        ),
    )

    summary = [
        aggregate(
            target_rows,
            "Overall",
        )
    ]

    for prefix in PREFIXES:
        summary.append(
            aggregate(
                target_rows,
                f"P{prefix}",
                prefix=prefix,
            )
        )

    for model in MODELS:
        summary.append(
            aggregate(
                target_rows,
                model,
                model=model,
            )
        )

    by_model_prefix = []

    for model in MODELS:
        for prefix in PREFIXES:
            by_model_prefix.append(
                aggregate(
                    target_rows,
                    f"{model}_P{prefix}",
                    model=model,
                    prefix=prefix,
                )
            )

    candidate_targets = [
        r
        for r in target_rows
        if int(
            r[
                "candidate_count"
            ]
        ) > 0
    ]

    no_candidate_targets = (
        EXPECTED_TARGETS
        - len(
            candidate_targets
        )
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        OUT
        / "candidate_diversity_target_detail.csv",
        target_rows,
    )

    write_csv(
        OUT
        / "candidate_diversity_summary.csv",
        summary,
    )

    write_csv(
        OUT
        / "candidate_diversity_by_model_prefix.csv",
        by_model_prefix,
    )

    result = {
        "scope": (
            "ETCE Phase 4D-16 formal candidate-diversity analysis"
        ),
        "status": "passed",
        "formal_test": True,
        "eligible_cf_candidate_count": len(
            eligible
        ),
        "formal_target_count": (
            EXPECTED_TARGETS
        ),
        "targets_with_candidates": len(
            candidate_targets
        ),
        "targets_without_candidates": (
            no_candidate_targets
        ),
        "definitions": {
            "witness_diversity": (
                "mean pairwise normalized Hamming distance "
                "between unique counterfactual witness sequences"
            ),
            "rule_footprint_diversity": (
                "mean pairwise Jaccard distance between sets "
                "of constrained activity positions in canonical rules"
            ),
            "unique_witness_rate": (
                "number of unique witness sequences / candidate count"
            ),
            "unique_rule_footprint_rate": (
                "number of unique canonical rule-position footprints / candidate count"
            ),
        },
        "terminology_guard": (
            "Rule-position footprint diversity is not asserted to be the exact "
            "factual edit-set diversity because canonical rule conditions can "
            "represent sets of allowed activity values."
        ),
        "overall": summary[
            0
        ],
        "by_prefix": [
            r
            for r in summary
            if str(
                r[
                    "scope"
                ]
            ).startswith(
                "P"
            )
        ],
        "by_model": [
            r
            for r in summary
            if r[
                "scope"
            ] in MODELS
        ],
        "test_side_tuning": False,
        "protocol_retuning": False,
        "interpretation_guard": (
            "Candidate diversity is descriptive. Higher diversity is not "
            "treated as inherently superior; it quantifies the breadth of "
            "valid alternatives available before final fusion/selection."
        ),
        "next_step": (
            "Multi-seed explanation stability analysis."
        ),
    }

    write_json(
        OUT
        / "candidate_diversity_summary.json",
        result,
    )

    print("=" * 118)
    print("ETCE PHASE 4D-16 FORMAL CANDIDATE DIVERSITY PASSED")
    print("=" * 118)
    print(
        f"Eligible CF candidates: {len(eligible)}"
    )
    print(
        f"Formal targets: {EXPECTED_TARGETS}"
    )
    print(
        f"Targets with >=1 candidate: {len(candidate_targets)}"
    )
    print(
        f"Targets without candidate: {no_candidate_targets}"
    )
    print()

    print("[Diversity summary]")

    for r in summary:
        print(
            f"{r['scope']:12s} | "
            f"N={r['target_count']:4d} | "
            f">=2Wit={r['targets_with_ge2_unique_witnesses']:4d} | "
            f"Cand={r['mean_candidate_count'] if r['mean_candidate_count'] is not None else float('nan'):.4f} | "
            f"UniqueWit={r['mean_unique_witness_rate'] if r['mean_unique_witness_rate'] is not None else float('nan'):.4f} | "
            f"WitHam={r['mean_pairwise_witness_hamming'] if r['mean_pairwise_witness_hamming'] is not None else float('nan'):.4f} | "
            f"FootJacc={r['mean_pairwise_rule_footprint_jaccard'] if r['mean_pairwise_rule_footprint_jaccard'] is not None else float('nan'):.4f} | "
            f"Sources={r['mean_surrogate_source_count'] if r['mean_surrogate_source_count'] is not None else float('nan'):.4f}"
        )

    print()
    print(f"Output: {OUT}")
    print("Candidate diversity treated as automatically better: False")
    print("Formal TEST tuning performed: False")
    print("Protocol retuning performed: False")
    print("NEXT: multi-seed explanation stability.")


if __name__ == "__main__":
    main()
