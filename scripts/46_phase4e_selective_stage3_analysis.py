#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4E: validation-frozen selective Stage-3 analysis.

The script does not retune the existing Phase-4D formal test. It:
1. reconstructs validation Stage2->Stage3 matched groups at frozen lambda=5;
2. evaluates the transparent validation-frozen gate
      accept Stage 3 iff mean Stage-2 held-out fidelity <= 0.50;
3. applies that gate to the already-observed Phase-4D formal test only as an
   exploratory transfer check;
4. reports case-clustered bootstrap confidence intervals.

A new untouched confirmation cohort is required before the selective gate can
be reported as a confirmatory improvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None


FROZEN_LAMBDA = 5.0
PRE_FIDELITY_THRESHOLD = 0.50
BOOTSTRAP_SEED = 20261008
BOOTSTRAP_REPLICATES = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Default: <project_root>/results/phase4/phase4e_selective_stage3",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return path


def normalize_signature(value: object) -> str:
    try:
        return json.dumps(json.loads(str(value)), separators=(",", ":"))
    except Exception:
        return str(value)


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def direction_counts(delta: pd.Series) -> dict[str, int]:
    values = delta.to_numpy(float)
    return {
        "improved": int((values > 1e-12).sum()),
        "tied": int((np.abs(values) <= 1e-12).sum()),
        "worse": int((values < -1e-12).sum()),
    }


def reconstruct_validation(stage_path: Path) -> pd.DataFrame:
    stage = pd.read_csv(stage_path, low_memory=False)
    stage["sig"] = stage["feature_signature"].map(normalize_signature)
    frozen = stage[stage["lambda"].astype(float).eq(FROZEN_LAMBDA)].copy()

    stage2 = frozen[frozen["stage"].eq("Stage2")].copy()
    stage3 = frozen[
        frozen["stage"].eq("Stage3") & bool_series(frozen["actually_fused"])
    ].copy()

    key = ["case_id", "prefix", "predicted_class", "sig"]
    stage3_index = stage3.set_index(key)
    if stage3_index.index.has_duplicates:
        raise AssertionError("Duplicate validation Stage-3 output key")

    rows: list[dict[str, object]] = []
    for group_key, inputs in stage2.groupby(key, dropna=False):
        if len(inputs) < 2 or group_key not in stage3_index.index:
            continue

        post = stage3_index.loc[group_key]
        fids = inputs["Fid_r"].astype(float).to_numpy()
        reliabilities = inputs["R_r"].astype(float).to_numpy()
        rows.append(
            {
                "case_id": group_key[0],
                "prefix": int(group_key[1]),
                "predicted_class": int(group_key[2]),
                "feature_signature": group_key[3],
                "input_count": int(len(inputs)),
                "source_model_count": int(inputs["model"].nunique()),
                "pre_fidelity": float(fids.mean()),
                "pre_fidelity_sd": float(fids.std()),
                "pre_reliability": float(reliabilities.mean()),
                "post_fidelity": float(post["Fid_r"]),
                "post_reliability": float(post["R_r"]),
                "last_K": float(post["last_K"]),
            }
        )

    detail = pd.DataFrame(rows)
    detail["delta_fidelity"] = detail["post_fidelity"] - detail["pre_fidelity"]
    detail["accept_stage3"] = detail["pre_fidelity"].le(PRE_FIDELITY_THRESHOLD)
    detail["selective_fidelity"] = np.where(
        detail["accept_stage3"], detail["post_fidelity"], detail["pre_fidelity"]
    )
    detail["selective_delta_vs_stage2"] = (
        detail["selective_fidelity"] - detail["pre_fidelity"]
    )
    if len(detail) != 629:
        raise AssertionError(f"Expected 629 validation Stage-3 groups, found {len(detail)}")
    return detail


def load_formal_detail(stage3_path: Path) -> pd.DataFrame:
    detail = pd.read_csv(stage3_path, low_memory=False)
    if len(detail) != 529:
        raise AssertionError(f"Expected 529 formal Stage-3 groups, found {len(detail)}")
    detail["sig"] = detail["feature_signature"].map(normalize_signature)
    detail["fixed_ds"] = detail["post_fidelity"].astype(float)
    detail["stop_stage2"] = detail["pre_fidelity"].astype(float)
    detail["accept_stage3"] = detail["pre_fidelity"].le(PRE_FIDELITY_THRESHOLD)
    detail["selective_stage3"] = np.where(
        detail["accept_stage3"], detail["fixed_ds"], detail["stop_stage2"]
    )
    detail["selective_delta_vs_stage2"] = (
        detail["selective_stage3"] - detail["stop_stage2"]
    )
    return detail


def attach_formal_baselines(detail: pd.DataFrame, baseline_path: Path) -> pd.DataFrame:
    usecols = [
        "baseline",
        "stage",
        "case_id",
        "prefix",
        "predicted_class",
        "feature_signature",
        "actually_fused",
        "Fid_r",
    ]
    baselines = pd.read_csv(baseline_path, usecols=usecols, low_memory=False)
    baselines = baselines[
        baselines["stage"].eq("Stage3") & bool_series(baselines["actually_fused"])
    ].copy()
    baselines["sig"] = baselines["feature_signature"].map(normalize_signature)
    key = ["case_id", "prefix", "predicted_class", "sig"]
    wide = baselines.pivot(index=key, columns="baseline", values="Fid_r").reset_index()
    merged = detail.merge(wide, on=key, how="left", validate="one_to_one")
    expected = ["best_single_evidence", "validation_best_source", "uniform_vote"]
    if merged[expected].isna().any().any():
        raise AssertionError("Missing aligned formal-test baseline values")
    return merged


def summarize_scope(scope: str, frame: pd.DataFrame) -> list[dict[str, object]]:
    if scope == "validation":
        methods = {
            "fixed_stage3": "post_fidelity",
            "stop_stage2": "pre_fidelity",
            "selective_stage3": "selective_fidelity",
        }
    else:
        methods = {
            "fixed_stage3": "fixed_ds",
            "stop_stage2": "stop_stage2",
            "selective_stage3": "selective_stage3",
            "best_single_evidence": "best_single_evidence",
            "validation_best_source": "validation_best_source",
            "uniform_vote": "uniform_vote",
        }

    rows: list[dict[str, object]] = []
    for method, column in methods.items():
        rows.append(
            {
                "scope": scope,
                "method": method,
                "N_groups": int(len(frame)),
                "N_cases": int(frame["case_id"].nunique()),
                "mean_fidelity": float(frame[column].mean()),
                "stage3_accept_count": (
                    int(frame["accept_stage3"].sum()) if method == "selective_stage3" else None
                ),
                "stage3_accept_rate": (
                    float(frame["accept_stage3"].mean()) if method == "selective_stage3" else None
                ),
            }
        )
    return rows


def case_clustered_comparison(
    frame: pd.DataFrame,
    method_col: str,
    baseline_col: str,
    rng: np.random.Generator,
) -> dict[str, object]:
    delta = frame[method_col].astype(float) - frame[baseline_col].astype(float)
    cases = pd.DataFrame({"case_id": frame["case_id"], "delta": delta}).groupby(
        "case_id"
    )["delta"].agg(["sum", "count", "mean"])

    sums = cases["sum"].to_numpy(float)
    counts = cases["count"].to_numpy(float)
    means = cases["mean"].to_numpy(float)
    n_cases = len(cases)
    indices = rng.integers(
        0, n_cases, size=(BOOTSTRAP_REPLICATES, n_cases), endpoint=False
    )
    pooled = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    equal_case = means[indices].mean(axis=1)

    p_value = None
    if wilcoxon is not None:
        p_value = float(
            wilcoxon(means, zero_method="pratt", alternative="two-sided", method="approx").pvalue
        )

    return {
        "comparison": f"{method_col} vs {baseline_col}",
        "N_groups": int(len(frame)),
        "N_cases": int(n_cases),
        "mean_method": float(frame[method_col].mean()),
        "mean_baseline": float(frame[baseline_col].mean()),
        "group_weighted_mean_delta": float(delta.mean()),
        "case_clustered_95_lo": float(np.quantile(pooled, 0.025)),
        "case_clustered_95_hi": float(np.quantile(pooled, 0.975)),
        "equal_case_mean_delta": float(means.mean()),
        "equal_case_95_lo": float(np.quantile(equal_case, 0.025)),
        "equal_case_95_hi": float(np.quantile(equal_case, 0.975)),
        "case_wins": int((means > 1e-12).sum()),
        "case_ties": int((np.abs(means) <= 1e-12).sum()),
        "case_losses": int((means < -1e-12).sum()),
        "case_wilcoxon_p": p_value,
    }


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    phase4 = root / "results" / "phase4"
    formal = phase4 / "phase4d_formal_test"
    out = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else phase4 / "phase4e_selective_stage3"
    )
    out.mkdir(parents=True, exist_ok=True)

    validation_stage_path = require_file(
        phase4
        / "phase4c_v2_hierarchical_ds_sweep"
        / "phase4c_v2_ds_stage_outputs.csv"
    )
    formal_stage3_path = require_file(
        formal / "formal_test_stage3_diagnostic" / "stage3_diagnostic_detail.csv"
    )
    formal_baseline_path = require_file(
        formal
        / "formal_test_fusion_baselines"
        / "formal_fusion_baseline_stage_outputs.csv"
    )

    validation = reconstruct_validation(validation_stage_path)
    formal_detail = load_formal_detail(formal_stage3_path)
    formal_detail = attach_formal_baselines(formal_detail, formal_baseline_path)

    summary_rows = summarize_scope("validation", validation)
    summary_rows.extend(summarize_scope("formal_exploratory", formal_detail))
    summary = pd.DataFrame(summary_rows)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = pd.DataFrame(
        [
            case_clustered_comparison(formal_detail, "selective_stage3", baseline, rng)
            for baseline in [
                "fixed_ds",
                "stop_stage2",
                "best_single_evidence",
                "validation_best_source",
                "uniform_vote",
            ]
        ]
    )

    validation.to_csv(out / "validation_selective_stage3_detail.csv", index=False)
    formal_detail.to_csv(out / "formal_selective_stage3_exploratory_detail.csv", index=False)
    summary.to_csv(out / "selective_stage3_summary.csv", index=False)
    comparisons.to_csv(out / "selective_stage3_case_clustered.csv", index=False)

    config = {
        "status": "validation_frozen_formal_exploratory",
        "lambda": FROZEN_LAMBDA,
        "gate": {
            "accept_stage3_if": "pre_mean_fidelity <= threshold",
            "pre_mean_fidelity_threshold": PRE_FIDELITY_THRESHOLD,
            "otherwise": "retain_stage2",
        },
        "selection_basis": (
            "Validation Stage2->Stage3 matched groups only; threshold 0.50 is a "
            "transparent binary-fidelity boundary supported by grouped validation analysis."
        ),
        "formal_test_use": (
            "Exploratory transfer check only because the Phase-4D formal-test results "
            "were already observed before this gate was proposed."
        ),
        "confirmation_requirement": (
            "Evaluate exactly once on a newly frozen untouched confirmation cohort "
            "before making confirmatory performance claims."
        ),
        "bootstrap": {
            "unit": "case_id",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
    }
    (out / "selective_stage3_frozen_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    v_fixed = float(validation["post_fidelity"].mean())
    v_stop = float(validation["pre_fidelity"].mean())
    v_select = float(validation["selective_fidelity"].mean())
    f_fixed = float(formal_detail["fixed_ds"].mean())
    f_stop = float(formal_detail["stop_stage2"].mean())
    f_select = float(formal_detail["selective_stage3"].mean())

    report = f"""# ETCE Phase 4E - Selective Stage-3 Analysis

## Frozen gate

Accept the Stage-3 output only when the mean held-out fidelity of its Stage-2
inputs is at most **{PRE_FIDELITY_THRESHOLD:.2f}**. Otherwise, retain the Stage-2
evidence. The rule is determined without using formal-test outcomes.

## Validation

- Groups: {len(validation)} from {validation['case_id'].nunique()} cases
- Fixed Stage 3 fidelity: {v_fixed:.6f}
- Stop at Stage 2 fidelity: {v_stop:.6f}
- Selective Stage 3 fidelity: {v_select:.6f}
- Accepted Stage-3 groups: {int(validation['accept_stage3'].sum())}/{len(validation)}
- Selective gain over fixed Stage 3: {v_select - v_fixed:+.6f}
- Selective gain over stopping at Stage 2: {v_select - v_stop:+.6f}

## Existing formal test - exploratory transfer check

- Groups: {len(formal_detail)} from {formal_detail['case_id'].nunique()} cases
- Fixed Stage 3 fidelity: {f_fixed:.6f}
- Stop at Stage 2 fidelity: {f_stop:.6f}
- Selective Stage 3 fidelity: {f_select:.6f}
- Accepted Stage-3 groups: {int(formal_detail['accept_stage3'].sum())}/{len(formal_detail)}
- Selective gain over fixed Stage 3: {f_select - f_fixed:+.6f}
- Selective gain over stopping at Stage 2: {f_select - f_stop:+.6f}

The formal-test check is exploratory, not confirmatory. A newly frozen untouched
confirmation cohort is required before revising the paper's headline results.
"""
    (out / "PHASE4E_SELECTIVE_STAGE3_REPORT.md").write_text(report, encoding="utf-8")

    print("=" * 100)
    print("ETCE PHASE 4E - SELECTIVE STAGE-3 ANALYSIS COMPLETED")
    print("=" * 100)
    print(f"Validation fixed/stop/selective: {v_fixed:.6f} / {v_stop:.6f} / {v_select:.6f}")
    print(f"Formal exploratory fixed/stop/selective: {f_fixed:.6f} / {f_stop:.6f} / {f_select:.6f}")
    print(f"Output: {out}")
    print("NEXT: freeze a new untouched confirmation cohort before confirmatory claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
