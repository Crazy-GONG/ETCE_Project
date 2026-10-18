#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4F-11 / Script 58
Independent validation of the Phase 4F-10 confirmation significance and
effect-size analysis.

This validator does not import Script 57. It independently:

1. verifies the frozen confirmation protocol and all upstream validators;
2. reconstructs the common 1,041-group identity universe;
3. reconstructs comparison-specific finite Rule-Fidelity pairs without
   imputing zero-support/undefined fidelities;
4. recomputes case-clustered 50,000-replicate bootstrap intervals;
5. recomputes case-level Wilcoxon tests, exact sign tests, paired
   rank-biserial effects, standardized paired effects, and W/T/L counts;
6. reapplies Holm-Bonferroni correction within the two declared families;
7. compares the detail, case, pairwise, and JSON outputs from Script 57.

The bootstrap uses a different batching layout from Script 57 while consuming
the same frozen NumPy random stream.

Run:
    cd D:\\ETCE
    python -u .\\scripts\\58_phase4f_validate_confirmation_significance_effect_size.py
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = ROOT / "results/phase4/phase4f_confirmation"
SELECTIVE_DIR = CONFIRMATION / "confirmation_selective_stage3"
BASELINE_DIR = CONFIRMATION / "confirmation_fusion_baselines"
DS_DIR = CONFIRMATION / "confirmation_frozen_ds_fusion"
SIGNIFICANCE_DIR = CONFIRMATION / "confirmation_fusion_significance"

SELECTIVE_DETAIL = (
    SELECTIVE_DIR / "phase4f_confirmation_selective_stage3_detail.csv"
)
SELECTIVE_RESULT = SELECTIVE_DIR / "confirmation_selective_stage3.json"
SELECTIVE_VALIDATION = (
    SELECTIVE_DIR / "validation_phase4f_confirmation_selective_stage3.json"
)
BASELINE_STAGE = (
    BASELINE_DIR / "confirmation_fusion_baseline_stage_outputs.csv"
)
BASELINE_RESULT = (
    BASELINE_DIR / "confirmation_fusion_strategy_comparison.json"
)
BASELINE_VALIDATION = (
    BASELINE_DIR / "validation_phase4f_confirmation_frozen_baselines.json"
)
DS_RESULT = DS_DIR / "confirmation_frozen_ds_fusion.json"
DS_VALIDATION = DS_DIR / "validation_phase4f_confirmation_frozen_ds_fusion.json"
SELECTION = CONFIRMATION / "confirmation_case_selection.json"
DS_CONFIG = ROOT / "configs/phase4c_ds_frozen_config.json"
BASELINE_CONFIG = ROOT / "configs/phase4c_fusion_baselines_frozen_config.json"
SELECTIVE_CONFIG = (
    ROOT
    / "results/phase4/phase4e_selective_stage3"
    / "selective_stage3_frozen_config.json"
)

SAVED_DETAIL = (
    SIGNIFICANCE_DIR / "phase4f_confirmation_selective_paired_detail.csv"
)
SAVED_PAIRWISE = (
    SIGNIFICANCE_DIR
    / "phase4f_confirmation_selective_pairwise_statistics.csv"
)
SAVED_CASE = (
    SIGNIFICANCE_DIR / "phase4f_confirmation_selective_case_statistics.csv"
)
SAVED_RESULT = (
    SIGNIFICANCE_DIR / "confirmation_selective_significance_effect_size.json"
)
OUT_VALIDATION = (
    SIGNIFICANCE_DIR
    / "validation_phase4f_confirmation_significance_effect_size.json"
)

BASELINES = [
    "best_single_evidence",
    "validation_best_source",
    "uniform_vote",
    "validation_weighted_vote",
    "fuzzy_probabilistic_or",
    "bayesian_categorical_noise",
]
COMPARISONS = [
    ("core_confirmatory", "fixed_stage3_ds_etce", "fixed_stage3_fidelity"),
    ("core_confirmatory", "stop_stage2", "pre_mean_fidelity"),
] + [
    ("frozen_non_ds_baselines", name, f"fidelity__{name}")
    for name in BASELINES
]

KEY = ["case_id", "prefix", "predicted_class", "signature"]
EXPECTED_GROUPS = 1041
FROZEN_LAMBDA = 5.0
FROZEN_THRESHOLD = 0.50
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_SEED = 20261008
ALPHA = 0.05
TOL = 1e-12


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_signature(value):
    try:
        return json.dumps(
            json.loads(str(value)),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except Exception:
        return str(value)


def booleans(series):
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def close_value(a, b, tol=TOL):
    if missing(a) or missing(b):
        return missing(a) and missing(b)
    af = float(a)
    bf = float(b)
    return abs(af - bf) <= tol + tol * max(abs(af), abs(bf))


def direction(values):
    values = np.asarray(values, dtype=float)
    return {
        "wins": int(np.sum(values > TOL)),
        "ties": int(np.sum(np.abs(values) <= TOL)),
        "losses": int(np.sum(values < -TOL)),
    }


def exact_sign_p(positive, nonzero):
    positive = int(positive)
    nonzero = int(nonzero)
    if nonzero == 0:
        return 1.0
    lower = min(positive, nonzero - positive)
    tail = sum(math.comb(nonzero, k) for k in range(lower + 1))
    return float(min(1.0, 2.0 * tail / (2 ** nonzero)))


def tied_ranks(values):
    values = np.asarray(values, dtype=float)
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    result = np.zeros(len(values), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while (
            stop < len(order)
            and abs(values[order[stop]] - values[order[start]]) <= 1e-15
        ):
            stop += 1
        average = (start + 1 + stop) / 2.0
        for position in range(start, stop):
            result[order[position]] = average
        start = stop
    return result


def paired_rank_biserial(values):
    values = np.asarray(values, dtype=float)
    values = values[np.abs(values) > TOL]
    if len(values) == 0:
        return 0.0
    ranks = tied_ranks(np.abs(values))
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    total = positive + negative
    return 0.0 if total == 0 else float((positive - negative) / total)


def wilcoxon_case_test(values):
    values = np.asarray(values, dtype=float)
    values = values[np.abs(values) > TOL]
    if len(values) == 0:
        return "all_ties", 1.0, 0
    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:
        raise RuntimeError("SciPy is required for the frozen Wilcoxon test") from exc
    try:
        result = wilcoxon(
            values,
            zero_method="wilcox",
            alternative="two-sided",
            correction=False,
            method="auto",
        )
    except TypeError:
        result = wilcoxon(
            values,
            zero_method="wilcox",
            alternative="two-sided",
            correction=False,
            mode="auto",
        )
    return (
        "case_level_wilcoxon_signed_rank_two_sided",
        float(result.pvalue),
        int(len(values)),
    )


def independent_cluster_bootstrap(paired, baseline_col, rng):
    work = paired[["case_id", "selective_fidelity", baseline_col]].copy()
    work["delta"] = (
        work["selective_fidelity"].astype(float)
        - work[baseline_col].astype(float)
    )
    clusters = work.groupby("case_id", sort=True)["delta"].agg(
        case_sum="sum",
        case_count="count",
        case_mean="mean",
    )
    sums = clusters["case_sum"].to_numpy(float)
    counts = clusters["case_count"].to_numpy(float)
    means = clusters["case_mean"].to_numpy(float)
    n_cases = len(clusters)
    require(n_cases > 0, "Bootstrap has no case clusters")

    pooled = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    equal_case = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    # Intentionally differs from Script 57's 2,000-draw batches.
    chunk = 1_250
    for begin in range(0, BOOTSTRAP_REPLICATES, chunk):
        end = min(begin + chunk, BOOTSTRAP_REPLICATES)
        indices = rng.integers(
            low=0,
            high=n_cases,
            size=(end - begin, n_cases),
            endpoint=False,
        )
        pooled[begin:end] = (
            np.sum(sums[indices], axis=1)
            / np.sum(counts[indices], axis=1)
        )
        equal_case[begin:end] = np.mean(means[indices], axis=1)

    return {
        "case_clustered_95_lo": float(np.quantile(pooled, 0.025)),
        "case_clustered_95_hi": float(np.quantile(pooled, 0.975)),
        "equal_case_95_lo": float(np.quantile(equal_case, 0.025)),
        "equal_case_95_hi": float(np.quantile(equal_case, 0.975)),
    }


def case_table(paired, family, baseline, baseline_col):
    work = paired[["case_id", "selective_fidelity", baseline_col]].copy()
    work["delta"] = (
        work["selective_fidelity"].astype(float)
        - work[baseline_col].astype(float)
    )
    grouped = work.groupby("case_id", sort=True).agg(
        group_count=("delta", "size"),
        selective_mean_fidelity=("selective_fidelity", "mean"),
        baseline_mean_fidelity=(baseline_col, "mean"),
        case_mean_delta=("delta", "mean"),
        case_sum_delta=("delta", "sum"),
    ).reset_index()
    grouped.insert(0, "family", family)
    grouped.insert(1, "baseline", baseline)
    return grouped


def recompute_one(detail, family, baseline, baseline_col, rng):
    method_values = pd.to_numeric(
        detail["selective_fidelity"], errors="coerce"
    )
    baseline_values = pd.to_numeric(detail[baseline_col], errors="coerce")
    finite = np.isfinite(method_values) & np.isfinite(baseline_values)
    paired = detail.loc[finite].copy()
    require(len(paired) > 0, f"{baseline}: no finite pairs")

    method = paired["selective_fidelity"].astype(float).to_numpy()
    comparator = paired[baseline_col].astype(float).to_numpy()
    differences = method - comparator
    cases = case_table(paired, family, baseline, baseline_col)
    case_differences = cases["case_mean_delta"].astype(float).to_numpy()
    group_d = direction(differences)
    case_d = direction(case_differences)
    interval = independent_cluster_bootstrap(paired, baseline_col, rng)
    test_name, p_value, nonzero = wilcoxon_case_test(case_differences)
    nonzero_signs = case_d["wins"] + case_d["losses"]
    case_sd = (
        float(np.std(case_differences, ddof=1))
        if len(case_differences) > 1
        else 0.0
    )

    result = {
        "family": family,
        "method": "selective_stage3_etce",
        "baseline": baseline,
        "N_identity_universe": int(len(detail)),
        "N_groups": int(len(paired)),
        "undefined_baseline_fidelity_groups": int(len(detail) - len(paired)),
        "N_cases": int(len(cases)),
        "mean_selective": float(np.mean(method)),
        "mean_baseline": float(np.mean(comparator)),
        "group_weighted_mean_delta": float(np.mean(differences)),
        "group_weighted_median_delta": float(np.median(differences)),
        **interval,
        "equal_case_mean_delta": float(np.mean(case_differences)),
        "group_wins": group_d["wins"],
        "group_ties": group_d["ties"],
        "group_losses": group_d["losses"],
        "case_wins": case_d["wins"],
        "case_ties": case_d["ties"],
        "case_losses": case_d["losses"],
        "case_win_rate_excluding_ties": (
            case_d["wins"] / nonzero_signs if nonzero_signs else None
        ),
        "paired_test": test_name,
        "p_value": p_value,
        "nonzero_case_pairs": nonzero,
        "case_level_exact_sign_p_value": exact_sign_p(
            case_d["wins"], nonzero_signs
        ),
        "case_level_rank_biserial": paired_rank_biserial(case_differences),
        "case_level_paired_standardized_mean_effect_dz": (
            float(np.mean(case_differences) / case_sd)
            if case_sd > TOL
            else 0.0
        ),
    }
    return result, cases


def holm(rows):
    order = sorted(range(len(rows)), key=lambda i: float(rows[i]["p_value"]))
    adjusted = [None] * len(rows)
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order, start=1):
        candidate = (total - rank + 1) * float(rows[index]["p_value"])
        running = max(running, candidate)
        adjusted[index] = float(min(1.0, running))
    return adjusted


def add_multiple_testing(rows):
    for family in sorted({row["family"] for row in rows}):
        members = [row for row in rows if row["family"] == family]
        adjusted = holm(members)
        for row, p_adjusted in zip(members, adjusted):
            row["holm_adjusted_p_within_family"] = p_adjusted
            row["ci_excludes_zero"] = bool(
                row["case_clustered_95_lo"] > 0
                or row["case_clustered_95_hi"] < 0
            )
            row["significant_after_holm_0_05"] = bool(p_adjusted < ALPHA)
            row["selective_superiority_supported"] = bool(
                row["group_weighted_mean_delta"] > TOL
                and row["case_clustered_95_lo"] > 0
                and p_adjusted < ALPHA
            )


def verify_protocol():
    required = [
        SELECTIVE_DETAIL,
        SELECTIVE_RESULT,
        SELECTIVE_VALIDATION,
        BASELINE_STAGE,
        BASELINE_RESULT,
        BASELINE_VALIDATION,
        DS_RESULT,
        DS_VALIDATION,
        SELECTION,
        DS_CONFIG,
        BASELINE_CONFIG,
        SELECTIVE_CONFIG,
        SAVED_DETAIL,
        SAVED_PAIRWISE,
        SAVED_CASE,
        SAVED_RESULT,
    ]
    absent = [str(path) for path in required if not path.is_file()]
    require(not absent, "Missing required file(s):\n" + "\n".join(absent))

    selective_result = read_json(SELECTIVE_RESULT)
    selective_validation = read_json(SELECTIVE_VALIDATION)
    baseline_result = read_json(BASELINE_RESULT)
    baseline_validation = read_json(BASELINE_VALIDATION)
    ds_result = read_json(DS_RESULT)
    ds_validation = read_json(DS_VALIDATION)
    selection = read_json(SELECTION)
    ds_config = read_json(DS_CONFIG)
    baseline_config = read_json(BASELINE_CONFIG)
    selective_config = read_json(SELECTIVE_CONFIG)
    saved_result = read_json(SAVED_RESULT)

    require(
        all(
            item.get("status") == "passed"
            for item in [
                selective_result,
                selective_validation,
                baseline_result,
                baseline_validation,
                ds_result,
                ds_validation,
                saved_result,
            ]
        ),
        "An upstream artifact is not PASSED",
    )
    require(
        all(
            item.get("confirmation_test") is True
            for item in [
                selective_result,
                selective_validation,
                baseline_result,
                baseline_validation,
                saved_result,
            ]
        ),
        "A result is not marked confirmation TEST",
    )
    require(
        int(selective_result["inventory"]["matched_fused_stage3_groups"])
        == EXPECTED_GROUPS
        and int(
            baseline_validation["reproduced"]["stage3_groups_per_method"]
        )
        == EXPECTED_GROUPS
        and int(saved_result["paired_group_count"]) == EXPECTED_GROUPS,
        "Frozen identity-universe count mismatch",
    )
    require(
        selective_result.get("lambda_selection_performed") is False
        and selective_result.get("selective_threshold_selection_performed")
        is False
        and selective_result.get("protocol_retuning_performed") is False
        and baseline_result.get("confirmation_side_baseline_tuning") is False
        and baseline_validation.get("confirmation_side_baseline_tuning")
        is False,
        "Confirmation-side selection or retuning detected",
    )
    require(
        ds_config.get("status") == "frozen"
        and close_value(ds_config.get("lambda"), FROZEN_LAMBDA),
        "Frozen D-S config mismatch",
    )
    require(
        baseline_config.get("status") == "frozen"
        and baseline_config["chronology"][
            "formal_test_files_read_by_this_script"
        ]
        is False
        and baseline_config["chronology"][
            "formal_test_outcomes_used_for_parameter_selection"
        ]
        is False,
        "Frozen baseline chronology mismatch",
    )
    gate = selective_config.get("gate", {})
    bootstrap = selective_config.get("bootstrap", {})
    require(
        selective_config.get("status")
        == "validation_frozen_formal_exploratory"
        and close_value(
            gate.get("pre_mean_fidelity_threshold"), FROZEN_THRESHOLD
        )
        and gate.get("accept_stage3_if")
        == "pre_mean_fidelity <= threshold"
        and gate.get("otherwise") == "retain_stage2"
        and bootstrap.get("unit") == "case_id"
        and int(bootstrap.get("replicates", -1)) == BOOTSTRAP_REPLICATES
        and int(bootstrap.get("seed", -1)) == BOOTSTRAP_SEED,
        "Frozen selective/bootstrap protocol mismatch",
    )
    require(
        selection.get("status") == "frozen"
        and selection.get("schema_version")
        == "etce-phase4f-confirmation-selection-v1"
        and int(selection.get("selected_case_count", -1)) == 200
        and int(selection.get("confirmation_target_count", -1)) == 3200
        and int(selection.get("global_sha256_rank_start", -1)) == 101
        and int(selection.get("global_sha256_rank_end", -1)) == 300
        and int(selection.get("overlap_with_original_phase4d", -1)) == 0
        and selection.get(
            "selection_uses_phase4d_or_phase4e_performance"
        )
        is False,
        "Frozen confirmation selection mismatch",
    )
    require(
        selection.get("frozen_ds_config_sha256") == sha256_file(DS_CONFIG)
        and selection.get("adaptive_config_sha256")
        == sha256_file(SELECTIVE_CONFIG),
        "Frozen configuration hash mismatch",
    )
    return {
        "selective_result": selective_result,
        "baseline_result": baseline_result,
        "saved_result": saved_result,
    }


def reconstruct_detail(upstream):
    detail = pd.read_csv(
        SELECTIVE_DETAIL,
        dtype={"case_id": str},
        low_memory=False,
    )
    require(len(detail) == EXPECTED_GROUPS, "Selective detail rows != 1041")
    detail["case_id"] = detail["case_id"].astype(str)
    detail["prefix"] = detail["prefix"].astype(int)
    detail["predicted_class"] = detail["predicted_class"].astype(int)
    detail["signature"] = detail["feature_signature"].map(
        canonical_signature
    )
    require(not detail.duplicated(KEY).any(), "Duplicate selective identity")
    detail = detail.sort_values(KEY).reset_index(drop=True)

    baseline_stage = pd.read_csv(
        BASELINE_STAGE,
        dtype={"case_id": str},
        low_memory=False,
    )
    stage3 = baseline_stage[
        baseline_stage["stage"].eq("Stage3")
        & booleans(baseline_stage["actually_fused"])
    ].copy()
    require(
        len(stage3) == EXPECTED_GROUPS * len(BASELINES),
        "Baseline Stage3 fused inventory mismatch",
    )
    stage3["case_id"] = stage3["case_id"].astype(str)
    stage3["prefix"] = stage3["prefix"].astype(int)
    stage3["predicted_class"] = stage3["predicted_class"].astype(int)
    stage3["signature"] = stage3["feature_signature"].map(
        canonical_signature
    )

    identity = set(detail[KEY].itertuples(index=False, name=None))
    for baseline in BASELINES:
        rows = stage3[stage3["baseline"].eq(baseline)].copy()
        require(len(rows) == EXPECTED_GROUPS, f"{baseline}: rows != 1041")
        require(not rows.duplicated(KEY).any(), f"{baseline}: duplicate key")
        require(
            set(rows[KEY].itertuples(index=False, name=None)) == identity,
            f"{baseline}: identity mismatch",
        )
        lookup = rows.set_index(KEY)["Fid_r"].astype(float).to_dict()
        detail[f"fidelity__{baseline}"] = [
            float(lookup[key])
            for key in detail[KEY].itertuples(index=False, name=None)
        ]

    expected = upstream["selective_result"]["fidelity"]
    for column, saved_mean in [
        ("fixed_stage3_fidelity", expected["fixed_stage3"]),
        ("pre_mean_fidelity", expected["stop_stage2"]),
        ("selective_fidelity", expected["selective_stage3"]),
    ]:
        require(
            close_value(detail[column].astype(float).mean(), saved_mean),
            f"{column}: marginal mean mismatch",
        )
    baseline_means = {
        row["method"]: row["rule_fidelity"]
        for row in upstream["baseline_result"]["stage3_comparison"]
    }
    for baseline in BASELINES:
        require(
            close_value(
                detail[f"fidelity__{baseline}"].mean(),
                baseline_means[baseline],
            ),
            f"{baseline}: marginal mean mismatch",
        )
    return detail


def compare_detail(reproduced):
    saved = pd.read_csv(SAVED_DETAIL, dtype={"case_id": str}, low_memory=False)
    saved["case_id"] = saved["case_id"].astype(str)
    saved["prefix"] = saved["prefix"].astype(int)
    saved["predicted_class"] = saved["predicted_class"].astype(int)
    saved["signature"] = saved["feature_signature"].map(canonical_signature)
    saved = saved.sort_values(KEY).reset_index(drop=True)
    require(len(saved) == len(reproduced), "Saved detail row-count mismatch")
    require(
        list(saved[KEY].itertuples(index=False, name=None))
        == list(reproduced[KEY].itertuples(index=False, name=None)),
        "Saved detail identity mismatch",
    )
    fields = [
        "pre_mean_fidelity",
        "fixed_stage3_fidelity",
        "selective_fidelity",
    ] + [f"fidelity__{name}" for name in BASELINES]
    for field in fields:
        require(field in saved.columns, f"Saved detail missing {field}")
        for a, b in zip(reproduced[field], saved[field]):
            require(close_value(a, b), f"Saved detail mismatch: {field}")


def compare_rows(reproduced, saved, key_fields):
    require(len(reproduced) == len(saved), "Row-count mismatch")
    saved_index = {
        tuple(str(row[field]) for field in key_fields): row
        for _, row in saved.iterrows()
    }
    require(len(saved_index) == len(saved), "Duplicate saved comparison key")
    for expected in reproduced:
        key = tuple(str(expected[field]) for field in key_fields)
        require(key in saved_index, f"Missing saved row: {key}")
        actual = saved_index[key]
        for field, value in expected.items():
            require(field in actual.index, f"Saved row missing field {field}")
            if isinstance(value, (bool, np.bool_)):
                require(
                    str(actual[field]).strip().lower()
                    == str(bool(value)).lower(),
                    f"Boolean mismatch: {key} / {field}",
                )
            elif isinstance(value, str):
                require(str(actual[field]) == value, f"Text mismatch: {key} / {field}")
            elif isinstance(value, (int, np.integer)):
                require(int(float(actual[field])) == int(value), f"Integer mismatch: {key} / {field}")
            else:
                require(close_value(value, actual[field]), f"Numeric mismatch: {key} / {field}")


def compare_case_frames(reproduced):
    saved = pd.read_csv(SAVED_CASE, dtype={"case_id": str}, low_memory=False)
    expected = pd.concat(reproduced, ignore_index=True)
    require(len(saved) == len(expected), "Saved case-stat row count mismatch")
    keys = ["family", "baseline", "case_id"]
    saved_index = saved.set_index(keys).sort_index()
    expected_index = expected.set_index(keys).sort_index()
    require(
        list(saved_index.index) == list(expected_index.index),
        "Saved case-stat identity mismatch",
    )
    for field in [
        "group_count",
        "selective_mean_fidelity",
        "baseline_mean_fidelity",
        "case_mean_delta",
        "case_sum_delta",
    ]:
        for expected_value, saved_value in zip(
            expected_index[field], saved_index[field]
        ):
            require(
                close_value(expected_value, saved_value),
                f"Saved case-stat mismatch: {field}",
            )


def compare_json(reproduced, upstream):
    saved = upstream["saved_result"]
    require(saved["method"] == "selective_stage3_etce", "JSON method mismatch")
    require(saved["primary_endpoint"] == "Rule Fidelity", "JSON endpoint mismatch")
    require(int(saved["paired_group_count"]) == EXPECTED_GROUPS, "JSON N mismatch")
    require(
        int(saved["bootstrap"]["replicates"]) == BOOTSTRAP_REPLICATES
        and int(saved["bootstrap"]["seed"]) == BOOTSTRAP_SEED
        and saved["bootstrap"]["cluster"] == "case_id",
        "JSON bootstrap mismatch",
    )
    require(
        saved["multiple_testing"]["procedure"] == "Holm-Bonferroni"
        and saved["multiple_testing"]["families"]
        == {"core_confirmatory": 2, "frozen_non_ds_baselines": 6},
        "JSON multiplicity mismatch",
    )
    require(len(saved["comparisons"]) == len(reproduced), "JSON comparison count")
    for expected, actual in zip(reproduced, saved["comparisons"]):
        require(expected["baseline"] == actual["baseline"], "JSON order mismatch")
        for field, value in expected.items():
            if isinstance(value, bool):
                require(actual[field] is value, f"JSON bool mismatch: {field}")
            elif isinstance(value, str):
                require(actual[field] == value, f"JSON text mismatch: {field}")
            elif isinstance(value, int):
                require(int(actual[field]) == value, f"JSON int mismatch: {field}")
            else:
                require(close_value(value, actual[field]), f"JSON numeric mismatch: {field}")


def fmt(value):
    return f"{float(value):.6g}"


def main():
    upstream = verify_protocol()
    print("=" * 118)
    print("ETCE PHASE 4F-11 INDEPENDENT CONFIRMATION SIGNIFICANCE VALIDATION")
    print("=" * 118)
    print("[Recompute] 1041-group identity universe and finite paired contrasts")

    detail = reconstruct_detail(upstream)
    compare_detail(detail)
    print(f"[Compare] Paired detail identities/values: {len(detail)} / {len(detail)} matched")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = []
    case_frames = []
    for family, baseline, column in COMPARISONS:
        row, cases = recompute_one(detail, family, baseline, column, rng)
        results.append(row)
        case_frames.append(cases)
        print(
            f"[Recompute] {baseline:30s} | "
            f"N={row['N_groups']} | undef={row['undefined_baseline_fidelity_groups']} | "
            f"Delta={row['group_weighted_mean_delta']:+.12f}"
        )

    add_multiple_testing(results)
    saved_pairwise = pd.read_csv(SAVED_PAIRWISE, low_memory=False)
    compare_rows(results, saved_pairwise, ["family", "baseline"])
    print("[Compare] Pairwise statistics: PASSED")
    compare_case_frames(case_frames)
    print("[Compare] Case-level statistics: PASSED")
    compare_json(results, upstream)
    print("[Compare] Result JSON: PASSED")

    validation = {
        "scope": (
            "ETCE Phase 4F-11 independent validation of paired "
            "confirmation significance and effect sizes"
        ),
        "status": "passed",
        "confirmation_test": True,
        "paired_identity_universe": int(len(detail)),
        "cases_in_identity_universe": int(detail["case_id"].nunique()),
        "bootstrap": {
            "cluster": "case_id",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "multiple_testing": {
            "procedure": "Holm-Bonferroni",
            "families": {
                "core_confirmatory": 2,
                "frozen_non_ds_baselines": 6,
            },
        },
        "reproduced_comparisons": results,
        "checks": {
            "frozen_protocol_chain_verified": True,
            "paired_detail_matches": True,
            "comparison_specific_complete_cases_reproduced": True,
            "undefined_fidelity_not_imputed": True,
            "case_clustered_bootstrap_matches": True,
            "case_level_wilcoxon_matches": True,
            "case_level_effect_sizes_match": True,
            "holm_adjustments_match": True,
            "case_statistics_match": True,
            "result_json_matches": True,
        },
        "protocol_flags": {
            "lambda_selection_performed": False,
            "selective_threshold_selection_performed": False,
            "confirmation_side_baseline_tuning_performed": False,
            "protocol_retuning_performed": False,
        },
        "next_step": (
            "Update the paper's confirmation claims and tables using the "
            "strict paired effects and validated multiplicity-adjusted results."
        ),
    }
    write_json(OUT_VALIDATION, validation)

    print()
    print("=" * 118)
    print("ETCE PHASE 4F-11 INDEPENDENT CONFIRMATION SIGNIFICANCE VALIDATION PASSED")
    print("=" * 118)
    print(
        f"Identity universe: {len(detail)} groups | "
        f"{detail['case_id'].nunique()} cases"
    )
    for row in results:
        print(
            f"{row['baseline']:30s} | "
            f"N={row['N_groups']:4d} | "
            f"undef={row['undefined_baseline_fidelity_groups']:3d} | "
            f"Delta={row['group_weighted_mean_delta']:+.6f} | "
            f"CI=[{row['case_clustered_95_lo']:+.6f},"
            f"{row['case_clustered_95_hi']:+.6f}] | "
            f"Holm={fmt(row['holm_adjusted_p_within_family'])} | "
            f"r_rb={row['case_level_rank_biserial']:+.4f} | "
            f"supported={'YES' if row['selective_superiority_supported'] else 'NO'}"
        )
    print("Lambda selection performed: False")
    print("Selective threshold selection performed: False")
    print("Confirmation-side baseline tuning performed: False")
    print("Protocol retuning performed: False")
    print(f"Validation JSON: {OUT_VALIDATION}")
    print("NEXT: update the paper's confirmation-result claims and tables.")


if __name__ == "__main__":
    main()
