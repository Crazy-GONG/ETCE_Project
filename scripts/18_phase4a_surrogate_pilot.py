"""ETCE Phase 4A local surrogate-rule extraction pilot.

Phase 4A is a validation-split software/method pilot. It does NOT use test cases
and is not final paper statistics.

Run:
    python scripts/18_phase4a_surrogate_pilot.py
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from etce.phase2.data import PREFIXES, load_frozen_common_cohort
from etce.phase3.core import (
    CanonicalPredictor,
    DIRECT_PROXIES,
    TERMINAL_OFFER_ACTIVITIES,
    flow_valid,
    load_process_constraints,
    process_support_score,
    sequence_edit_count,
)
from etce.phase3.semantic_core import family_preserved
from etce.phase4.surrogate_core import (
    generate_process_valid_pool,
    run_all_surrogates,
    sequence_hash,
)
from etce.utils.io import read_json, write_csv, write_json

RESULTS3 = ROOT / "results/phase3"
OUT = ROOT / "results/phase4/phase4a"

MODELS = ("RF", "XGBoost", "MLP", "LSTM")
PILOT_CASE_COUNT = 8
POOL_SIZE = 320
MAX_EDITS = 3
MAX_OPTIONS = 8
POOL_SEED_BASE = 0


def stable_rank(case_id):
    return hashlib.sha256(
        case_id.encode("utf-8")
    ).hexdigest()


def local_seed(model, prefix, case_id):
    text = (
        f"{POOL_SEED_BASE}|{model}|{prefix}|{case_id}"
    ).encode("utf-8")
    return int(
        hashlib.sha256(text).hexdigest()[:16],
        16,
    )


def write_gzip_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with gzip.open(
        path,
        "wt",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    phase3b = read_json(
        RESULTS3 / "phase3b/validation_phase3b.json"
    )
    if phase3b.get("status") != "passed":
        raise RuntimeError(
            "Validated Phase 3B is required before Phase 4A"
        )

    canonical = read_json(
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json"
    )
    constraints = load_process_constraints(
        RESULTS3
    )

    cases, _ = load_frozen_common_cohort(
        ROOT
    )
    val_cases = sorted(
        (
            case
            for case in cases
            if case.split == "val"
        ),
        key=lambda case: (
            stable_rank(
                case.case_id
            ),
            case.case_id,
        ),
    )
    pilot_cases = val_cases[
        :PILOT_CASE_COUNT
    ]
    pilot_ids = [
        case.case_id
        for case in pilot_cases
    ]

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        OUT / "pilot_case_selection.json",
        {
            "split": "validation",
            "pilot_case_count": PILOT_CASE_COUNT,
            "case_ids": pilot_ids,
            "selection_rule": (
                "First 8 frozen validation cases after sorting by SHA256(case_id)."
            ),
            "true_labels_used_for_selection": False,
            "blackbox_outputs_used_for_selection": False,
            "test_cases_used": False,
        },
    )

    write_json(
        OUT / "phase4a_run_config.json",
        {
            "scope": "software/method pilot",
            "split": "validation",
            "models": list(MODELS),
            "prefixes": list(PREFIXES),
            "pool_size": POOL_SIZE,
            "max_edits": MAX_EDITS,
            "max_options_per_position": MAX_OPTIONS,
            "neighborhood_generation_uses_blackbox_labels": False,
            "neighborhood_constraints": [
                "train-only DFG",
                "same-position support",
                "event-family preservation",
                "immutable historical terminal outcomes",
                "forbidden outcome mutation targets",
                "fixed factual timestamps",
            ],
            "surrogates": {
                "LORE": (
                    "ETCE process-aware adaptation of LORE workflow: "
                    "class-balanced local fit subset + local decision tree"
                ),
                "DT": (
                    "natural local fit subset + shallow CART "
                    "(max_depth=4, min_samples_leaf=5)"
                ),
                "BN": (
                    "Naive-Bayes-structured Bayesian Network Y->X_i, "
                    "Laplace alpha=1"
                ),
            },
            "rule_evaluation_split": (
                "deterministic 70/30 sequence-hash fit/eval split"
            ),
            "paper_statistics": False,
        },
    )

    neighborhood_rows = []
    summary_rows = []
    rule_rows = []

    total = (
        len(pilot_cases)
        * len(MODELS)
        * len(PREFIXES)
    )
    done = 0

    for model in MODELS:
        for prefix in PREFIXES:
            predictor = CanonicalPredictor(
                ROOT,
                model,
                prefix,
                canonical["items"][model][str(prefix)],
            )

            for case in pilot_cases:
                done += 1
                factual = tuple(
                    case.activities[:prefix]
                )
                timestamps = tuple(
                    case.timestamps[:prefix]
                )
                seed = local_seed(
                    model,
                    prefix,
                    case.case_id,
                )

                pool = generate_process_valid_pool(
                    factual,
                    constraints,
                    rng_seed=seed,
                    pool_size=POOL_SIZE,
                    max_edits=MAX_EDITS,
                    max_options_per_position=MAX_OPTIONS,
                )
                probs = predictor.predict_proba(
                    pool,
                    [timestamps] * len(pool),
                )
                labels = (
                    np.asarray(probs)
                    >= 0.5
                ).astype(int)

                factual_probability = float(
                    predictor.predict_proba(
                        [factual],
                        [timestamps],
                    )[0]
                )
                factual_label = int(
                    factual_probability
                    >= 0.5
                )

                result = run_all_surrogates(
                    pool,
                    labels,
                    factual,
                    factual_label,
                )

                fit_set = {
                    sequence_hash(seq)
                    for seq in result[
                        "fit_sequences"
                    ]
                }
                eval_set = {
                    sequence_hash(seq)
                    for seq in result[
                        "eval_sequences"
                    ]
                }

                for seq, prob, label in zip(
                    pool,
                    probs,
                    labels,
                ):
                    h = sequence_hash(seq)
                    changed = [
                        i + 1
                        for i, (
                            old,
                            new,
                        ) in enumerate(
                            zip(
                                factual,
                                seq,
                            )
                        )
                        if old != new
                    ]
                    neighborhood_rows.append(
                        {
                            "case_id": case.case_id,
                            "model": model,
                            "prefix": int(prefix),
                            "canonical_seed": int(
                                predictor.seed
                            ),
                            "pool_seed": int(seed),
                            "sequence_hash": h,
                            "local_split": (
                                "fit"
                                if h in fit_set
                                else "eval"
                            ),
                            "bb_probability": float(
                                prob
                            ),
                            "bb_label": int(
                                label
                            ),
                            "edit_count": int(
                                sequence_edit_count(
                                    factual,
                                    seq,
                                )
                            ),
                            "process_support": float(
                                process_support_score(
                                    seq,
                                    constraints[
                                        "edge_count"
                                    ],
                                )
                            ),
                            "changed_positions": json.dumps(
                                changed
                            ),
                            "activities": json.dumps(
                                seq,
                                ensure_ascii=False,
                            ),
                        }
                    )

                fit_labels = np.asarray(
                    result["fit_labels"],
                    dtype=int,
                )
                eval_labels = np.asarray(
                    result["eval_labels"],
                    dtype=int,
                )

                summary_base = {
                    "case_id": case.case_id,
                    "true_label_audit_only": int(
                        case.label
                    ),
                    "model": model,
                    "prefix": int(prefix),
                    "canonical_seed": int(
                        predictor.seed
                    ),
                    "pool_seed": int(seed),
                    "pool_size_requested": POOL_SIZE,
                    "pool_size_generated": len(
                        pool
                    ),
                    "pool_class0": int(
                        np.sum(
                            labels == 0
                        )
                    ),
                    "pool_class1": int(
                        np.sum(
                            labels == 1
                        )
                    ),
                    "fit_size": len(
                        result["fit_sequences"]
                    ),
                    "eval_size": len(
                        result["eval_sequences"]
                    ),
                    "fit_class0": int(
                        np.sum(
                            fit_labels == 0
                        )
                    ),
                    "fit_class1": int(
                        np.sum(
                            fit_labels == 1
                        )
                    ),
                    "eval_class0": int(
                        np.sum(
                            eval_labels == 0
                        )
                    ),
                    "eval_class1": int(
                        np.sum(
                            eval_labels == 1
                        )
                    ),
                    "factual_prediction": factual_label,
                    "factual_probability": factual_probability,
                    "opposite_class_present_pool": bool(
                        np.any(
                            labels
                            != factual_label
                        )
                    ),
                }

                for branch in result[
                    "branches"
                ]:
                    surrogate = branch[
                        "branch"
                    ]
                    factual_rule = branch[
                        "factual_rule"
                    ]
                    cf_rule = branch[
                        "counterfactual_rule"
                    ]

                    row = dict(
                        summary_base
                    )
                    row.update(
                        {
                            "surrogate": surrogate,
                            "status": branch[
                                "status"
                            ],
                            "fit_size_used": branch[
                                "fit_size_used"
                            ],
                            "surrogate_fidelity_eval": (
                                ""
                                if branch[
                                    "surrogate_fidelity_eval"
                                ] is None
                                else float(
                                    branch[
                                        "surrogate_fidelity_eval"
                                    ]
                                )
                            ),
                            "factual_rule_available": bool(
                                factual_rule
                                is not None
                            ),
                            "factual_rule_support_eval": (
                                ""
                                if factual_rule is None
                                else int(
                                    factual_rule[
                                        "support"
                                    ]
                                )
                            ),
                            "factual_rule_fidelity_eval": (
                                ""
                                if (
                                    factual_rule is None
                                    or factual_rule[
                                        "fidelity"
                                    ] is None
                                )
                                else float(
                                    factual_rule[
                                        "fidelity"
                                    ]
                                )
                            ),
                            "factual_rule_length": (
                                ""
                                if factual_rule is None
                                else int(
                                    factual_rule[
                                        "rule_length"
                                    ]
                                )
                            ),
                            "cf_rule_available": bool(
                                cf_rule
                                is not None
                            ),
                            "cf_rule_support_eval": (
                                ""
                                if cf_rule is None
                                else int(
                                    cf_rule[
                                        "support"
                                    ]
                                )
                            ),
                            "cf_rule_fidelity_eval": (
                                ""
                                if (
                                    cf_rule is None
                                    or cf_rule[
                                        "fidelity"
                                    ] is None
                                )
                                else float(
                                    cf_rule[
                                        "fidelity"
                                    ]
                                )
                            ),
                            "cf_rule_length": (
                                ""
                                if cf_rule is None
                                else int(
                                    cf_rule[
                                        "rule_length"
                                    ]
                                )
                            ),
                            "cf_witness_edit_count": (
                                ""
                                if (
                                    cf_rule is None
                                    or "witness_edit_count"
                                    not in cf_rule
                                )
                                else int(
                                    cf_rule[
                                        "witness_edit_count"
                                    ]
                                )
                            ),
                        }
                    )
                    if surrogate in (
                        "LORE",
                        "DT",
                    ):
                        row[
                            "model_complexity"
                        ] = (
                            f"depth={branch.get('tree_depth')};"
                            f"leaves={branch.get('tree_leaves')}"
                        )
                    else:
                        row[
                            "model_complexity"
                        ] = (
                            f"alpha={branch.get('bn_alpha')}"
                        )

                    summary_rows.append(
                        row
                    )

                    for rule_type, rule in (
                        (
                            "factual",
                            factual_rule,
                        ),
                        (
                            "counterfactual",
                            cf_rule,
                        ),
                    ):
                        if rule is None:
                            continue
                        rule_rows.append(
                            {
                                "case_id": case.case_id,
                                "model": model,
                                "prefix": int(
                                    prefix
                                ),
                                "surrogate": surrogate,
                                "rule_type": rule_type,
                                "predicted_class": int(
                                    rule[
                                        "predicted_class"
                                    ]
                                ),
                                "conditions": rule[
                                    "conditions"
                                ],
                                "support_eval": int(
                                    rule[
                                        "support"
                                    ]
                                ),
                                "fidelity_eval": rule[
                                    "fidelity"
                                ],
                                "coverage_eval": float(
                                    rule[
                                        "coverage"
                                    ]
                                ),
                                "rule_length": int(
                                    rule[
                                        "rule_length"
                                    ]
                                ),
                                "witness_sequence": rule.get(
                                    "witness_sequence"
                                ),
                                "witness_edit_count": rule.get(
                                    "witness_edit_count"
                                ),
                            }
                        )

                print(
                    f"[{done:03d}/{total}] "
                    f"{model} P{prefix} "
                    f"{case.case_id}: "
                    f"pool={len(pool)} "
                    f"class=({int(np.sum(labels==0))},"
                    f"{int(np.sum(labels==1))})",
                    flush=True,
                )

    neighborhood_fields = [
        "case_id",
        "model",
        "prefix",
        "canonical_seed",
        "pool_seed",
        "sequence_hash",
        "local_split",
        "bb_probability",
        "bb_label",
        "edit_count",
        "process_support",
        "changed_positions",
        "activities",
    ]
    write_gzip_csv(
        OUT / "local_neighborhoods.csv.gz",
        neighborhood_rows,
        neighborhood_fields,
    )

    write_csv(
        OUT / "surrogate_metrics.csv",
        summary_rows,
    )

    with (
        OUT / "canonical_rules.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as stream:
        for rule in rule_rows:
            stream.write(
                json.dumps(
                    rule,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    # Aggregate diagnostics.
    aggregates = []
    for surrogate in (
        "LORE",
        "DT",
        "BN",
    ):
        group = [
            r
            for r in summary_rows
            if r["surrogate"]
            == surrogate
        ]
        fidelity = [
            float(
                r[
                    "surrogate_fidelity_eval"
                ]
            )
            for r in group
            if r[
                "surrogate_fidelity_eval"
            ] != ""
        ]
        factual_available = [
            r[
                "factual_rule_available"
            ]
            for r in group
        ]
        cf_available = [
            r["cf_rule_available"]
            for r in group
        ]
        fact_fid = [
            float(
                r[
                    "factual_rule_fidelity_eval"
                ]
            )
            for r in group
            if r[
                "factual_rule_fidelity_eval"
            ] != ""
        ]
        cf_fid = [
            float(
                r[
                    "cf_rule_fidelity_eval"
                ]
            )
            for r in group
            if r[
                "cf_rule_fidelity_eval"
            ] != ""
        ]
        aggregates.append(
            {
                "surrogate": surrogate,
                "targets": len(group),
                "mean_surrogate_fidelity_eval": (
                    ""
                    if not fidelity
                    else float(
                        np.mean(fidelity)
                    )
                ),
                "factual_rule_availability": (
                    float(
                        np.mean(
                            factual_available
                        )
                    )
                ),
                "cf_rule_availability": (
                    float(
                        np.mean(
                            cf_available
                        )
                    )
                ),
                "mean_factual_rule_fidelity_eval": (
                    ""
                    if not fact_fid
                    else float(
                        np.mean(
                            fact_fid
                        )
                    )
                ),
                "mean_cf_rule_fidelity_eval": (
                    ""
                    if not cf_fid
                    else float(
                        np.mean(
                            cf_fid
                        )
                    )
                ),
            }
        )
    write_csv(
        OUT / "surrogate_aggregate.csv",
        aggregates,
    )

    report = [
        "# ETCE Phase 4A Surrogate Rule Extraction Pilot",
        "",
        "> Validation-split method/software pilot. Not final paper statistics.",
        "",
        "## Internal evidence sources",
        "",
        "- **LORE branch**: ETCE process-aware adaptation of the LORE workflow; class-balanced local fit subset, local decision tree, factual/counterfactual path rules.",
        "- **DT branch**: natural local fit subset with shallow CART.",
        "- **BN branch**: Naive-Bayes-structured Bayesian Network `Y -> X_i` with Laplace smoothing.",
        "",
        "The internal LORE branch is not presented as the external official LORE/LORE_sa baseline; that comparison is reserved for a later baseline experiment.",
        "",
        "## Shared local evidence protocol",
        "",
        f"- Pilot cases: {PILOT_CASE_COUNT} frozen validation cases selected independently of labels/model outputs.",
        f"- Maximum local pool size: {POOL_SIZE}; max edits: {MAX_EDITS}.",
        "- Pool generation is black-box-label independent.",
        "- Every generated sequence must preserve the Phase 3A.1 train-only DFG, same-position support, event family, immutable historical outcomes and fixed timestamps.",
        "- Surrogate fitting/evaluation uses a deterministic sequence-hash 70/30 local split.",
        "- Rule support and rule fidelity are measured on the local evaluation subset, not the surrogate fitting subset.",
        "",
        "## Aggregate pilot diagnostics",
        "",
        "| Surrogate | Targets | Surrogate fidelity | Factual rule availability | CF rule availability | Factual rule fidelity | CF rule fidelity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value):
        if value == "":
            return "-"
        return f"{float(value):.4f}"

    for r in aggregates:
        report.append(
            f"| {r['surrogate']} | {r['targets']} | "
            f"{fmt(r['mean_surrogate_fidelity_eval'])} | "
            f"{r['factual_rule_availability']:.4f} | "
            f"{r['cf_rule_availability']:.4f} | "
            f"{fmt(r['mean_factual_rule_fidelity_eval'])} | "
            f"{fmt(r['mean_cf_rule_fidelity_eval'])} |"
        )

    report += [
        "",
        "## Stopping point",
        "",
        "Phase 4A only establishes local surrogate/rule extraction and measures `Fid(r)` and support `n_r` on held-out local evidence.",
        "Support-aware reliability `rho_r`, `R(r)`, BPA assignment and D-S fusion are intentionally not implemented yet.",
        "",
        "<!-- VALIDATION_START -->",
        "Awaiting Phase 4A validation.",
        "<!-- VALIDATION_END -->",
        "",
    ]

    (
        OUT / "PHASE4A_REPORT.md"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("Phase 4A complete.")
    print(OUT / "PHASE4A_REPORT.md")
    print("Run scripts/19_validate_phase4a.py next.")


if __name__ == "__main__":
    main()
