#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4F-10 / Script 57
Paired, case-clustered significance and effect-size analysis on the untouched
confirmation TEST cohort.

Primary endpoint
----------------
Rule Fidelity in the same universe of 1,041 genuinely fused Stage-3
compatible groups. A method can yield zero held-out support after
re-evaluation; its fidelity is then undefined. Such a group remains in the
identity audit and is excluded only from the affected complete-case paired
contrast. The effective N is reported for every comparison; no value is
imputed.

Method under evaluation
-----------------------
The validation-frozen selective Stage-3 policy:
    accept Stage 3 iff pre-mean-fidelity <= 0.50;
    otherwise retain the Stage-2 evidence.

Comparison families
-------------------
1. Core confirmatory family (Holm correction over 2 comparisons):
   - selective Stage 3 vs fixed Stage-3 D-S ETCE
   - selective Stage 3 vs stopping at Stage 2

2. Frozen non-D-S baseline family (Holm correction over 6 comparisons):
   - best single evidence
   - validation-best source
   - uniform vote
   - validation-weighted vote
   - fuzzy probabilistic OR
   - Bayesian categorical-noise pooling

Dependence handling
-------------------
- Confidence intervals use case_id-clustered bootstrap resampling with the
  pre-frozen 50,000-replicate, seed-20261008 protocol.
- Wilcoxon signed-rank tests and paired rank-biserial effects operate on one
  mean fidelity difference per case, rather than treating multiple groups
  from the same case as independent.
- Group-level and case-level wins/ties/losses are both reported.

No lambda selection, threshold selection, baseline tuning, or protocol
retuning is performed here.

Run:
    cd D:\\ETCE
    python -u .\\scripts\\57_phase4f_confirmation_significance_effect_size.py
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
OUT_DIR = CONFIRMATION / "confirmation_fusion_significance"

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

DETAIL_OUT = OUT_DIR / "phase4f_confirmation_selective_paired_detail.csv"
PAIRWISE_OUT = (
    OUT_DIR / "phase4f_confirmation_selective_pairwise_statistics.csv"
)
CASE_OUT = OUT_DIR / "phase4f_confirmation_selective_case_statistics.csv"
RESULT_OUT = OUT_DIR / "confirmation_selective_significance_effect_size.json"
REPORT_OUT = OUT_DIR / "PHASE4F_CONFIRMATION_SIGNIFICANCE_EFFECT_SIZE_REPORT.md"

BASELINES = [
    "best_single_evidence",
    "validation_best_source",
    "uniform_vote",
    "validation_weighted_vote",
    "fuzzy_probabilistic_or",
    "bayesian_categorical_noise",
]

COMPARISONS = [
    {
        "family": "core_confirmatory",
        "baseline": "fixed_stage3_ds_etce",
        "column": "fixed_stage3_fidelity",
    },
    {
        "family": "core_confirmatory",
        "baseline": "stop_stage2",
        "column": "pre_mean_fidelity",
    },
] + [
    {
        "family": "frozen_non_ds_baselines",
        "baseline": name,
        "column": f"fidelity__{name}",
    }
    for name in BASELINES
]

EXPECTED_GROUPS = 1041
FROZEN_LAMBDA = 5.0
FROZEN_THRESHOLD = 0.50
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_SEED = 20261008
ALPHA = 0.05
TOL = 1e-12
KEY = ["case_id", "prefix", "predicted_class", "signature"]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_signature(value):
    try:
        return json.dumps(
            json.loads(str(value)),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except Exception:
        return str(value)


def bool_series(series: pd.Series):
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def close(a, b, tol=TOL):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def direction_counts(values):
    x = np.asarray(values, dtype=float)
    return {
        "wins": int((x > TOL).sum()),
        "ties": int((np.abs(x) <= TOL).sum()),
        "losses": int((x < -TOL).sum()),
    }


def exact_binomial_two_sided(successes, trials):
    """Exact two-sided sign-test p-value under p=0.5."""
    trials = int(trials)
    successes = int(successes)
    if trials <= 0:
        return 1.0
    tail = min(successes, trials - successes)
    probability = sum(
        math.comb(trials, i) for i in range(tail + 1)
    ) / (2 ** trials)
    return float(min(1.0, 2.0 * probability))


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while (
            j < len(order)
            and abs(values[order[j]] - values[order[i]]) <= 1e-15
        ):
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def rank_biserial(differences):
    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[np.abs(diffs) > TOL]
    if len(diffs) == 0:
        return 0.0
    ranks = average_ranks(np.abs(diffs))
    positive = float(ranks[diffs > 0].sum())
    negative = float(ranks[diffs < 0].sum())
    denominator = positive + negative
    return 0.0 if denominator == 0 else (positive - negative) / denominator


def paired_wilcoxon(case_differences):
    diffs = np.asarray(case_differences, dtype=float)
    nonzero = diffs[np.abs(diffs) > TOL]
    if len(nonzero) == 0:
        return {
            "paired_test": "all_ties",
            "p_value": 1.0,
            "nonzero_case_pairs": 0,
        }

    try:
        from scipy.stats import wilcoxon

        try:
            result = wilcoxon(
                nonzero,
                zero_method="wilcox",
                alternative="two-sided",
                correction=False,
                method="auto",
            )
        except TypeError:
            # SciPy < 1.9 used the equivalent keyword name ``mode``.
            result = wilcoxon(
                nonzero,
                zero_method="wilcox",
                alternative="two-sided",
                correction=False,
                mode="auto",
            )
        return {
            "paired_test": "case_level_wilcoxon_signed_rank_two_sided",
            "p_value": float(result.pvalue),
            "nonzero_case_pairs": int(len(nonzero)),
        }
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required for the pre-specified Wilcoxon analysis. "
            "Install scipy and rerun; no alternative test is substituted."
        ) from exc


def clustered_bootstrap(detail, method_col, baseline_col, rng):
    delta = (
        detail[method_col].astype(float)
        - detail[baseline_col].astype(float)
    )
    clustered = pd.DataFrame(
        {"case_id": detail["case_id"], "delta": delta}
    ).groupby("case_id", sort=True)["delta"].agg(["sum", "count", "mean"])

    sums = clustered["sum"].to_numpy(float)
    counts = clustered["count"].to_numpy(float)
    means = clustered["mean"].to_numpy(float)
    n_cases = len(clustered)
    require(n_cases > 0, "No case clusters available")

    pooled_draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    equal_case_draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    batch = 2_000
    for start in range(0, BOOTSTRAP_REPLICATES, batch):
        stop = min(start + batch, BOOTSTRAP_REPLICATES)
        sampled = rng.integers(
            0,
            n_cases,
            size=(stop - start, n_cases),
            endpoint=False,
        )
        pooled_draws[start:stop] = (
            sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
        )
        equal_case_draws[start:stop] = means[sampled].mean(axis=1)

    return {
        "case_clustered_95_lo": float(
            np.quantile(pooled_draws, 0.025)
        ),
        "case_clustered_95_hi": float(
            np.quantile(pooled_draws, 0.975)
        ),
        "equal_case_95_lo": float(
            np.quantile(equal_case_draws, 0.025)
        ),
        "equal_case_95_hi": float(
            np.quantile(equal_case_draws, 0.975)
        ),
    }


def holm_adjust(rows):
    """Holm adjusted p-values within one explicitly supplied family."""
    ordered = sorted(
        [(i, float(row["p_value"])) for i, row in enumerate(rows)],
        key=lambda pair: pair[1],
    )
    adjusted = [None] * len(rows)
    running = 0.0
    m = len(ordered)
    for rank, (original_index, p_value) in enumerate(ordered, start=1):
        candidate = (m - rank + 1) * p_value
        running = max(running, candidate)
        adjusted[original_index] = float(min(1.0, running))
    return adjusted


def case_rows_for_comparison(detail, baseline_name, baseline_col, family):
    frame = detail[
        ["case_id", "selective_fidelity", baseline_col]
    ].copy()
    frame["delta"] = (
        frame["selective_fidelity"].astype(float)
        - frame[baseline_col].astype(float)
    )
    grouped = frame.groupby("case_id", sort=True).agg(
        group_count=("delta", "size"),
        selective_mean_fidelity=("selective_fidelity", "mean"),
        baseline_mean_fidelity=(baseline_col, "mean"),
        case_mean_delta=("delta", "mean"),
        case_sum_delta=("delta", "sum"),
    ).reset_index()
    grouped.insert(0, "family", family)
    grouped.insert(1, "baseline", baseline_name)
    return grouped


def compare_one(
    detail,
    baseline_name,
    baseline_col,
    family,
    rng,
):
    eligible = np.isfinite(
        pd.to_numeric(detail["selective_fidelity"], errors="coerce")
    ) & np.isfinite(
        pd.to_numeric(detail[baseline_col], errors="coerce")
    )
    paired = detail.loc[eligible].copy()
    require(
        len(paired) > 0,
        f"{baseline_name}: no complete fidelity pairs",
    )

    selective = paired["selective_fidelity"].astype(float).to_numpy()
    baseline = paired[baseline_col].astype(float).to_numpy()
    delta = selective - baseline

    cases = case_rows_for_comparison(
        paired,
        baseline_name,
        baseline_col,
        family,
    )
    case_delta = cases["case_mean_delta"].astype(float).to_numpy()
    group_direction = direction_counts(delta)
    case_direction = direction_counts(case_delta)
    bootstrap = clustered_bootstrap(
        paired,
        "selective_fidelity",
        baseline_col,
        rng,
    )
    test = paired_wilcoxon(case_delta)

    nonzero_cases = case_direction["wins"] + case_direction["losses"]
    sign_p = exact_binomial_two_sided(
        case_direction["wins"],
        nonzero_cases,
    )
    case_sd = float(np.std(case_delta, ddof=1)) if len(case_delta) > 1 else 0.0

    row = {
        "family": family,
        "method": "selective_stage3_etce",
        "baseline": baseline_name,
        "N_identity_universe": int(len(detail)),
        "N_groups": int(len(paired)),
        "undefined_baseline_fidelity_groups": int(len(detail) - len(paired)),
        "N_cases": int(len(cases)),
        "mean_selective": float(np.mean(selective)),
        "mean_baseline": float(np.mean(baseline)),
        "group_weighted_mean_delta": float(np.mean(delta)),
        "group_weighted_median_delta": float(np.median(delta)),
        **bootstrap,
        "equal_case_mean_delta": float(np.mean(case_delta)),
        "group_wins": group_direction["wins"],
        "group_ties": group_direction["ties"],
        "group_losses": group_direction["losses"],
        "case_wins": case_direction["wins"],
        "case_ties": case_direction["ties"],
        "case_losses": case_direction["losses"],
        "case_win_rate_excluding_ties": (
            case_direction["wins"] / nonzero_cases
            if nonzero_cases > 0
            else None
        ),
        **test,
        "case_level_exact_sign_p_value": sign_p,
        "case_level_rank_biserial": float(rank_biserial(case_delta)),
        "case_level_paired_standardized_mean_effect_dz": (
            float(np.mean(case_delta) / case_sd)
            if case_sd > TOL
            else 0.0
        ),
    }
    return row, cases


def verify_frozen_chain():
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
    ]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, "Missing required file(s):\n" + "\n".join(missing))

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

    require(
        all(
            payload.get("status") == "passed"
            for payload in [
                selective_result,
                selective_validation,
                baseline_result,
                baseline_validation,
                ds_result,
                ds_validation,
            ]
        ),
        "An upstream result or independent validation is not PASSED",
    )
    require(
        selective_result.get("confirmation_test") is True
        and selective_validation.get("confirmation_test") is True
        and baseline_result.get("confirmation_test") is True
        and baseline_validation.get("confirmation_test") is True,
        "An upstream artifact is not marked as confirmation TEST",
    )
    require(
        int(
            selective_result["inventory"]["matched_fused_stage3_groups"]
        ) == EXPECTED_GROUPS
        and int(
            baseline_validation["reproduced"]["stage3_groups_per_method"]
        ) == EXPECTED_GROUPS,
        "Upstream Stage3 matched-group inventory mismatch",
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
        and abs(float(ds_config.get("lambda", -1)) - FROZEN_LAMBDA) <= TOL,
        "Frozen D-S lambda/config mismatch",
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
        "Frozen baseline chronology/config mismatch",
    )
    gate = selective_config.get("gate", {})
    bootstrap = selective_config.get("bootstrap", {})
    require(
        selective_config.get("status")
        == "validation_frozen_formal_exploratory"
        and abs(
            float(gate.get("pre_mean_fidelity_threshold", -1))
            - FROZEN_THRESHOLD
        )
        <= TOL
        and gate.get("accept_stage3_if")
        == "pre_mean_fidelity <= threshold"
        and gate.get("otherwise") == "retain_stage2"
        and bootstrap.get("unit") == "case_id"
        and int(bootstrap.get("replicates", -1)) == BOOTSTRAP_REPLICATES
        and int(bootstrap.get("seed", -1)) == BOOTSTRAP_SEED,
        "Frozen selective/bootstrap protocol mismatch",
    )
    require(
        selection.get("schema_version")
        == "etce-phase4f-confirmation-selection-v1"
        and selection.get("status") == "frozen"
        and int(selection.get("selected_case_count", -1)) == 200
        and int(selection.get("confirmation_target_count", -1)) == 3200
        and int(selection.get("global_sha256_rank_start", -1)) == 101
        and int(selection.get("global_sha256_rank_end", -1)) == 300
        and int(selection.get("overlap_with_original_phase4d", -1)) == 0
        and selection.get(
            "selection_uses_phase4d_or_phase4e_performance"
        )
        is False,
        "Frozen confirmation-case selection mismatch",
    )
    require(
        selection.get("frozen_ds_config_sha256") == sha256_file(DS_CONFIG)
        and selection.get("adaptive_config_sha256")
        == sha256_file(SELECTIVE_CONFIG),
        "Frozen configuration hash mismatch",
    )

    return {
        "selective_result": selective_result,
        "selective_validation": selective_validation,
        "baseline_result": baseline_result,
        "baseline_validation": baseline_validation,
        "ds_result": ds_result,
        "ds_validation": ds_validation,
    }


def build_paired_detail(upstream):
    detail = pd.read_csv(
        SELECTIVE_DETAIL,
        dtype={"case_id": str},
        low_memory=False,
    )
    require(len(detail) == EXPECTED_GROUPS, "Selective detail row count != 1041")
    required_columns = {
        "case_id",
        "prefix",
        "predicted_class",
        "feature_signature",
        "pre_mean_fidelity",
        "fixed_stage3_fidelity",
        "selective_fidelity",
        "accept_stage3",
    }
    require(
        required_columns.issubset(detail.columns),
        "Selective detail is missing required columns",
    )
    detail["case_id"] = detail["case_id"].astype(str)
    detail["prefix"] = detail["prefix"].astype(int)
    detail["predicted_class"] = detail["predicted_class"].astype(int)
    detail["signature"] = detail["feature_signature"].map(
        normalize_signature
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
        & bool_series(baseline_stage["actually_fused"])
    ].copy()
    require(
        len(stage3) == EXPECTED_GROUPS * len(BASELINES),
        "Fused baseline Stage3 inventory mismatch",
    )
    stage3["case_id"] = stage3["case_id"].astype(str)
    stage3["prefix"] = stage3["prefix"].astype(int)
    stage3["predicted_class"] = stage3["predicted_class"].astype(int)
    stage3["signature"] = stage3["feature_signature"].map(
        normalize_signature
    )

    selective_keys = set(map(tuple, detail[KEY].itertuples(index=False, name=None)))
    for baseline in BASELINES:
        subset = stage3[stage3["baseline"].eq(baseline)].copy()
        require(
            len(subset) == EXPECTED_GROUPS,
            f"{baseline}: fused Stage3 rows != 1041",
        )
        require(
            not subset.duplicated(KEY).any(),
            f"{baseline}: duplicate Stage3 identity",
        )
        baseline_keys = set(
            map(tuple, subset[KEY].itertuples(index=False, name=None))
        )
        require(
            baseline_keys == selective_keys,
            f"{baseline}: identity set differs from selective detail",
        )
        value_map = subset.set_index(KEY)["Fid_r"].astype(float)
        detail[f"fidelity__{baseline}"] = [
            float(value_map.loc[key])
            for key in detail[KEY].itertuples(index=False, name=None)
        ]

    core_columns = [
        "pre_mean_fidelity",
        "fixed_stage3_fidelity",
        "selective_fidelity",
    ]
    checked_columns = core_columns + [
        f"fidelity__{baseline}" for baseline in BASELINES
    ]
    for column in checked_columns:
        values = pd.to_numeric(detail[column], errors="coerce")
        require(
            not np.isinf(values.to_numpy(float)).any(),
            f"{column}: infinite fidelity detected",
        )
        require(
            int(values.notna().sum()) > 0,
            f"{column}: every fidelity value is undefined",
        )

    selective_result = upstream["selective_result"]
    expected_means = {
        "fixed_stage3_fidelity": selective_result["fidelity"][
            "fixed_stage3"
        ],
        "pre_mean_fidelity": selective_result["fidelity"]["stop_stage2"],
        "selective_fidelity": selective_result["fidelity"][
            "selective_stage3"
        ],
    }
    for column, expected in expected_means.items():
        # Phase 4F-6 used pandas Series.mean(), which excludes undefined
        # values. Reproduce that marginal mean; inferential contrasts below
        # separately report their finite-pair N.
        observed = float(detail[column].astype(float).mean(skipna=True))
        require(
            close(observed, expected),
            f"Saved {column} mean mismatch: {observed} != {expected}",
        )

    saved_comparison = {
        row["method"]: row
        for row in upstream["baseline_result"]["stage3_comparison"]
    }
    for baseline in BASELINES:
        # Script 55's frozen summary deliberately ignores undefined fidelity
        # values produced when the selected rule has zero pooled held-out
        # support.  Reproduce that exact marginal summary here, while the
        # inferential comparison below uses complete pairs and reports its N.
        observed = float(detail[f"fidelity__{baseline}"].mean(skipna=True))
        expected = float(saved_comparison[baseline]["rule_fidelity"])
        require(
            close(observed, expected),
            f"{baseline}: mean fidelity mismatch",
        )

    return detail


def validate_core_bootstrap_against_phase4f6(rows, upstream):
    saved = {
        row["comparison"]: row
        for row in upstream["selective_result"][
            "case_clustered_comparisons"
        ]
    }
    mapping = {
        "fixed_stage3_ds_etce": (
            "selective_fidelity_vs_fixed_stage3_fidelity"
        ),
        "stop_stage2": "selective_fidelity_vs_pre_mean_fidelity",
    }
    for row in rows:
        if row["baseline"] not in mapping:
            continue
        prior = saved[mapping[row["baseline"]]]
        for field in [
            "case_clustered_95_lo",
            "case_clustered_95_hi",
            "equal_case_95_lo",
            "equal_case_95_hi",
        ]:
            require(
                close(row[field], prior[field]),
                f"Core bootstrap mismatch for {row['baseline']} / {field}",
            )


def apply_family_adjustments(rows):
    families = sorted({row["family"] for row in rows})
    for family in families:
        members = [row for row in rows if row["family"] == family]
        adjusted = holm_adjust(members)
        for row, adjusted_p in zip(members, adjusted):
            row["holm_adjusted_p_within_family"] = adjusted_p
            row["ci_excludes_zero"] = bool(
                row["case_clustered_95_lo"] > 0
                or row["case_clustered_95_hi"] < 0
            )
            row["significant_after_holm_0_05"] = bool(
                adjusted_p < ALPHA
            )
            row["selective_superiority_supported"] = bool(
                row["group_weighted_mean_delta"] > TOL
                and row["case_clustered_95_lo"] > 0
                and adjusted_p < ALPHA
            )


def fmt(value, digits=6):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def main():
    upstream = verify_frozen_chain()

    print("=" * 118)
    print("ETCE PHASE 4F-10 CONFIRMATION SIGNIFICANCE / EFFECT SIZE")
    print("=" * 118)
    print("Endpoint: paired Rule Fidelity in 1041 matched fused Stage3 groups")
    print("Undefined fidelity: comparison-specific complete cases; no imputation")
    print("Dependence unit: case_id")
    print("Bootstrap: case_id clusters | 50000 replicates | seed=20261008")
    print("Multiplicity: Holm within 2 predeclared comparison families")
    print("Confirmation-side tuning: False")
    print()

    detail = build_paired_detail(upstream)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pairwise = []
    case_frames = []
    for comparison in COMPARISONS:
        print(f"[Compare] selective_stage3_etce vs {comparison['baseline']} ...")
        row, cases = compare_one(
            detail,
            comparison["baseline"],
            comparison["column"],
            comparison["family"],
            rng,
        )
        pairwise.append(row)
        case_frames.append(cases)

    validate_core_bootstrap_against_phase4f6(pairwise, upstream)
    apply_family_adjustments(pairwise)

    pairwise_df = pd.DataFrame(pairwise)
    case_df = pd.concat(case_frames, ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_detail = detail.drop(columns=["signature"])
    export_detail.to_csv(DETAIL_OUT, index=False, encoding="utf-8-sig")
    pairwise_df.to_csv(PAIRWISE_OUT, index=False, encoding="utf-8-sig")
    case_df.to_csv(CASE_OUT, index=False, encoding="utf-8-sig")

    family_sizes = {
        family: sum(row["family"] == family for row in pairwise)
        for family in sorted({row["family"] for row in pairwise})
    }
    result = {
        "scope": (
            "ETCE Phase 4F-10 paired case-clustered significance and "
            "effect-size analysis on untouched confirmation TEST"
        ),
        "status": "passed",
        "confirmation_test": True,
        "confirmatory_evaluation": True,
        "primary_endpoint": "Rule Fidelity",
        "method": "selective_stage3_etce",
        "paired_group_count": int(len(detail)),
        "paired_case_count": int(detail["case_id"].nunique()),
        "frozen_lambda": FROZEN_LAMBDA,
        "frozen_selective_threshold": FROZEN_THRESHOLD,
        "bootstrap": {
            "cluster": "case_id",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 95% CI",
            "primary_estimand": (
                "group-weighted paired mean fidelity difference with "
                "case-cluster resampling"
            ),
        },
        "paired_test": {
            "unit": "case_id mean paired fidelity difference",
            "test": "two-sided Wilcoxon signed-rank",
            "effect_size": "paired rank-biserial correlation",
        },
        "multiple_testing": {
            "procedure": "Holm-Bonferroni",
            "alpha": ALPHA,
            "families": family_sizes,
            "family_rationale": (
                "The two predeclared core confirmation contrasts and the "
                "six frozen non-D-S benchmark contrasts are controlled as "
                "separate, explicitly reported families."
            ),
        },
        "comparisons": pairwise,
        "confirmatory_summary": {
            row["baseline"]: {
                "mean_delta": row["group_weighted_mean_delta"],
                "case_clustered_95ci": [
                    row["case_clustered_95_lo"],
                    row["case_clustered_95_hi"],
                ],
                "raw_p": row["p_value"],
                "holm_p": row["holm_adjusted_p_within_family"],
                "case_level_rank_biserial": row[
                    "case_level_rank_biserial"
                ],
                "selective_superiority_supported": row[
                    "selective_superiority_supported"
                ],
            }
            for row in pairwise
        },
        "protocol_flags": {
            "lambda_selection_performed": False,
            "selective_threshold_selection_performed": False,
            "confirmation_side_baseline_tuning_performed": False,
            "protocol_retuning_performed": False,
            "probability_averaging_executed": False,
        },
        "checks": {
            "upstream_independent_validations_passed": True,
            "frozen_config_hashes_verified": True,
            "all_methods_have_identical_1041_group_keys": True,
            "all_saved_method_means_reproduced": True,
            "phase4f6_core_bootstrap_intervals_reproduced": True,
            "undefined_fidelity_not_imputed": True,
        },
        "missing_fidelity_policy": (
            "A method may yield zero held-out support after re-evaluation, "
            "making Fid_r undefined. Identity matching still covers all "
            "1041 groups. Each contrast uses only finite paired fidelities, "
            "reports its effective N and undefined count, and performs no "
            "imputation. This makes the implicit null exclusion in the "
            "frozen Phase 4F-6 pandas aggregation explicit and auditable."
        ),
        "metric_scope_note": (
            "Only Rule Fidelity is compared because rejected selective "
            "Stage-3 groups retain multiple Stage-2 evidence objects rather "
            "than emitting one representative rule; single-rule CF/model "
            "validity, support, coverage, and reliability are therefore not "
            "defined for the selective policy."
        ),
        "outputs": {
            "paired_detail_csv": str(DETAIL_OUT),
            "pairwise_statistics_csv": str(PAIRWISE_OUT),
            "case_statistics_csv": str(CASE_OUT),
            "report_md": str(REPORT_OUT),
        },
        "next_step": (
            "Independently validate Phase 4F-10 statistics, then update the "
            "paper's confirmation claims and tables."
        ),
    }
    write_json(RESULT_OUT, result)

    lines = [
        "# ETCE Phase 4F-10 Confirmation Significance and Effect Size",
        "",
        "Primary endpoint: paired Rule Fidelity in the same universe of "
        "1,041 genuinely fused Stage-3 compatible groups. Confidence "
        "intervals resample "
        "`case_id` clusters; Wilcoxon tests and rank-biserial effects use one "
        "mean paired difference per case.",
        "",
        "| Family | Baseline | Paired N | Undefined | Selective | Baseline | Delta | Clustered 95% CI | Case W/T/L | Raw p | Holm p | Rank-biserial | Supported |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in pairwise:
        lines.append(
            f"| {row['family']} | {row['baseline']} | "
            f"{row['N_groups']} | "
            f"{row['undefined_baseline_fidelity_groups']} | "
            f"{row['mean_selective']:.6f} | "
            f"{row['mean_baseline']:.6f} | "
            f"{row['group_weighted_mean_delta']:+.6f} | "
            f"[{row['case_clustered_95_lo']:+.6f}, "
            f"{row['case_clustered_95_hi']:+.6f}] | "
            f"{row['case_wins']}/{row['case_ties']}/{row['case_losses']} | "
            f"{row['p_value']:.6g} | "
            f"{row['holm_adjusted_p_within_family']:.6g} | "
            f"{row['case_level_rank_biserial']:+.4f} | "
            f"{'Yes' if row['selective_superiority_supported'] else 'No'} |"
        )
    lines.extend(
        [
            "",
            "Holm correction is applied separately to the two core "
            "confirmatory contrasts and the six validation-frozen non-D-S "
            "baseline contrasts. A superiority indicator requires a positive "
            "mean difference, a case-clustered 95% CI wholly above zero, and "
            "a within-family Holm-adjusted p-value below 0.05.",
            "",
            "Only Rule Fidelity is tested. Other rule-level metrics are "
            "undefined when the selective policy retains multiple Stage-2 "
            "evidence objects.",
            "",
            "Zero held-out support makes fidelity undefined for any affected "
            "method. Those values are not imputed; each contrast reports its "
            "complete-case paired N. Identity matching still covers the full "
            "1,041-group universe.",
            "",
            "No confirmation-side tuning or protocol retuning was performed.",
        ]
    )
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 118)
    print("ETCE PHASE 4F-10 CONFIRMATION SIGNIFICANCE / EFFECT SIZE PASSED")
    print("=" * 118)
    print(
        f"Matched groups: {len(detail)} | "
        f"Cases: {detail['case_id'].nunique()}"
    )
    print("Delta convention: selective_stage3_etce minus comparator")
    print()
    for row in pairwise:
        supported = (
            "YES" if row["selective_superiority_supported"] else "NO"
        )
        print(
            f"{row['baseline']:30s} | "
            f"N={row['N_groups']:4d} | "
            f"undef={row['undefined_baseline_fidelity_groups']:3d} | "
            f"Sel={row['mean_selective']:.6f} | "
            f"Base={row['mean_baseline']:.6f} | "
            f"Delta={row['group_weighted_mean_delta']:+.6f} | "
            f"CI=[{row['case_clustered_95_lo']:+.6f},"
            f"{row['case_clustered_95_hi']:+.6f}] | "
            f"case W/T/L={row['case_wins']}/"
            f"{row['case_ties']}/{row['case_losses']} | "
            f"p={row['p_value']:.6g} | "
            f"Holm={row['holm_adjusted_p_within_family']:.6g} | "
            f"r_rb={row['case_level_rank_biserial']:+.4f} | "
            f"supported={supported}"
        )
    print()
    print("Lambda selection performed: False")
    print("Selective threshold selection performed: False")
    print("Confirmation-side baseline tuning performed: False")
    print("Protocol retuning performed: False")
    print(f"Result JSON: {RESULT_OUT}")
    print("NEXT: independently validate Phase 4F-10 statistics.")


if __name__ == "__main__":
    main()
