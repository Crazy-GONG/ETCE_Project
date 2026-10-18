#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4F-7: independent validation of the one-shot confirmation-set
application of the validation-frozen selective Stage-3 gate.

This validator does not import script 53.  It independently reconstructs the
Stage2-to-Stage3 matches, gate decisions, fidelity summaries, and case-clustered
bootstrap comparisons, then checks the saved Phase 4F-6 artifacts.

Run:
    cd D:\\ETCE
    python -u .\\scripts\\54_phase4f_validate_frozen_selective_stage3_confirmation.py
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE4F_ROOT = ROOT / "results/phase4/phase4f_confirmation"
FUSION_DIR = PHASE4F_ROOT / "confirmation_frozen_ds_fusion"
SELECTIVE_DIR = PHASE4F_ROOT / "confirmation_selective_stage3"

STAGE_OUTPUTS = FUSION_DIR / "phase4f_confirmation_frozen_ds_stage_outputs.csv"
FUSION_VALIDATION = (
    FUSION_DIR / "validation_phase4f_confirmation_frozen_ds_fusion.json"
)
SELECTION = PHASE4F_ROOT / "confirmation_case_selection.json"
FROZEN_DS = ROOT / "configs/phase4c_ds_frozen_config.json"
FROZEN_SELECTIVE = (
    ROOT
    / "results/phase4/phase4e_selective_stage3"
    / "selective_stage3_frozen_config.json"
)

SAVED_DETAIL = (
    SELECTIVE_DIR / "phase4f_confirmation_selective_stage3_detail.csv"
)
SAVED_SUMMARY = (
    SELECTIVE_DIR / "phase4f_confirmation_selective_stage3_summary.csv"
)
SAVED_COMPARISONS = (
    SELECTIVE_DIR / "phase4f_confirmation_selective_stage3_case_clustered.csv"
)
SAVED_RESULT = SELECTIVE_DIR / "confirmation_selective_stage3.json"
OUT_VALIDATION = (
    SELECTIVE_DIR / "validation_phase4f_confirmation_selective_stage3.json"
)

FROZEN_LAMBDA = 5.0
FROZEN_THRESHOLD = 0.50
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_SEED = 20261008
TOL = 1e-12


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


def normalize_json(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return json.dumps(
            json.loads(str(value)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except Exception:
        return str(value)


def bool_series(series: pd.Series):
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def bool_value(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def close(a, b, tol=TOL):
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return b is None or (isinstance(b, float) and math.isnan(b))
    if b is None or (isinstance(b, float) and math.isnan(b)):
        return False
    return abs(float(a) - float(b)) <= tol


def direction_counts(values):
    x = np.asarray(values, dtype=float)
    return {
        "wins": int((x > TOL).sum()),
        "ties": int((np.abs(x) <= TOL).sum()),
        "losses": int((x < -TOL).sum()),
    }


def clustered_bootstrap(
    detail: pd.DataFrame,
    method_col: str,
    baseline_col: str,
    rng: np.random.Generator,
):
    delta = detail[method_col].astype(float) - detail[baseline_col].astype(float)
    grouped = pd.DataFrame(
        {"case_id": detail["case_id"], "delta": delta}
    ).groupby("case_id", sort=True)["delta"].agg(["sum", "count", "mean"])

    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    means = grouped["mean"].to_numpy(float)
    n_cases = len(grouped)

    pooled_draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    equal_case_draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    # Deliberately use a different chunk size from script 53.  NumPy consumes
    # the same deterministic integer stream, while this remains a separate
    # implementation of the resampling loop.
    chunk = 1_250
    for start in range(0, BOOTSTRAP_REPLICATES, chunk):
        stop = min(start + chunk, BOOTSTRAP_REPLICATES)
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

    gd = direction_counts(delta)
    cd = direction_counts(means)
    return {
        "comparison": f"{method_col}_vs_{baseline_col}",
        "N_groups": int(len(detail)),
        "N_cases": int(n_cases),
        "mean_method": float(detail[method_col].mean()),
        "mean_baseline": float(detail[baseline_col].mean()),
        "group_weighted_mean_delta": float(delta.mean()),
        "case_clustered_95_lo": float(np.quantile(pooled_draws, 0.025)),
        "case_clustered_95_hi": float(np.quantile(pooled_draws, 0.975)),
        "equal_case_mean_delta": float(means.mean()),
        "equal_case_95_lo": float(np.quantile(equal_case_draws, 0.025)),
        "equal_case_95_hi": float(np.quantile(equal_case_draws, 0.975)),
        "group_wins": gd["wins"],
        "group_ties": gd["ties"],
        "group_losses": gd["losses"],
        "case_wins": cd["wins"],
        "case_ties": cd["ties"],
        "case_losses": cd["losses"],
    }


def identity_from_row(row):
    return (
        str(row["case_id"]),
        int(row["prefix"]),
        int(row["predicted_class"]),
        normalize_signature(row["feature_signature"]),
    )


def compare_detail(expected: pd.DataFrame, saved: pd.DataFrame):
    require(len(expected) == len(saved), "Selective detail row count mismatch")
    expected_idx = {
        identity_from_row(row): row for _, row in expected.iterrows()
    }
    saved_idx = {identity_from_row(row): row for _, row in saved.iterrows()}
    require(
        len(expected_idx) == len(expected)
        and len(saved_idx) == len(saved),
        "Duplicate detail identity",
    )
    require(set(expected_idx) == set(saved_idx), "Detail identities differ")

    numeric = [
        "pre_mean_fidelity",
        "pre_fidelity_sd",
        "pre_min_fidelity",
        "pre_max_fidelity",
        "pre_mean_reliability",
        "fixed_stage3_fidelity",
        "fixed_stage3_reliability",
        "fixed_stage3_last_K",
        "frozen_threshold",
        "selective_fidelity",
        "fixed_delta_vs_stage2",
        "selective_delta_vs_fixed_stage3",
        "selective_delta_vs_stage2",
    ]
    integers = ["stage2_input_count", "stage2_model_count"]

    for key, exp in expected_idx.items():
        got = saved_idx[key]
        for field in numeric:
            require(
                close(exp[field], got[field]),
                f"Detail numeric mismatch {key} {field}: "
                f"{exp[field]} != {got[field]}",
            )
        for field in integers:
            require(
                int(exp[field]) == int(got[field]),
                f"Detail integer mismatch {key} {field}",
            )
        require(
            bool(exp["accept_stage3"]) == bool_value(got["accept_stage3"]),
            f"Gate decision mismatch: {key}",
        )
        require(
            str(exp["decision"]) == str(got["decision"]),
            f"Decision label mismatch: {key}",
        )
        require(
            normalize_json(exp["stage2_models"])
            == normalize_json(got["stage2_models"]),
            f"Stage2 source models mismatch: {key}",
        )


def compare_table(expected_rows, saved: pd.DataFrame, key_field: str):
    expected_idx = {str(r[key_field]): r for r in expected_rows}
    saved_idx = {str(r[key_field]): r for _, r in saved.iterrows()}
    require(set(expected_idx) == set(saved_idx), f"{key_field} identities differ")
    for key, exp in expected_idx.items():
        got = saved_idx[key]
        for field, value in exp.items():
            if field == key_field:
                continue
            if value is None:
                require(
                    pd.isna(got[field]) or str(got[field]).strip() == "",
                    f"Expected blank {key} {field}",
                )
            elif isinstance(value, (int, np.integer)):
                require(int(got[field]) == int(value), f"Integer mismatch {key} {field}")
            else:
                require(close(value, got[field]), f"Numeric mismatch {key} {field}")


def main():
    required = [
        STAGE_OUTPUTS,
        FUSION_VALIDATION,
        SELECTION,
        FROZEN_DS,
        FROZEN_SELECTIVE,
        SAVED_DETAIL,
        SAVED_SUMMARY,
        SAVED_COMPARISONS,
        SAVED_RESULT,
    ]
    missing = [str(p) for p in required if not p.is_file()]
    require(not missing, "Missing required artifacts:\n" + "\n".join(missing))

    fusion_validation = read_json(FUSION_VALIDATION)
    selection = read_json(SELECTION)
    frozen_ds = read_json(FROZEN_DS)
    frozen_selective = read_json(FROZEN_SELECTIVE)
    saved_result = read_json(SAVED_RESULT)

    print("=" * 116)
    print("ETCE PHASE 4F-7 INDEPENDENT SELECTIVE STAGE-3 CONFIRMATION VALIDATION")
    print("=" * 116)

    # Recheck the complete frozen chain independently.
    require(frozen_ds.get("status") == "frozen", "D-S config is not frozen")
    require(float(frozen_ds.get("lambda", -1)) == FROZEN_LAMBDA, "lambda != 5")
    require(
        frozen_ds.get("selection_scope") == "validation_only",
        "Lambda was not selected on validation only",
    )
    require(
        selection.get("status") == "frozen"
        and selection.get("schema_version")
        == "etce-phase4f-confirmation-selection-v1",
        "Confirmation selection is not frozen",
    )
    require(
        int(selection.get("selected_case_count", -1)) == 200
        and int(selection.get("confirmation_target_count", -1)) == 3200
        and int(selection.get("global_sha256_rank_start", -1)) == 101
        and int(selection.get("global_sha256_rank_end", -1)) == 300
        and int(selection.get("overlap_with_original_phase4d", -1)) == 0,
        "Confirmation selection inventory/rank mismatch",
    )
    require(
        selection.get("selection_uses_phase4d_or_phase4e_performance") is False,
        "Observed results influenced confirmation selection",
    )
    require(
        selection.get("frozen_ds_config_sha256") == sha256_file(FROZEN_DS)
        and selection.get("adaptive_config_sha256")
        == sha256_file(FROZEN_SELECTIVE),
        "Frozen configuration hash mismatch",
    )

    gate = frozen_selective.get("gate", {})
    bootstrap = frozen_selective.get("bootstrap", {})
    require(
        frozen_selective.get("status")
        == "validation_frozen_formal_exploratory"
        and float(frozen_selective.get("lambda", -1)) == FROZEN_LAMBDA
        and gate.get("accept_stage3_if")
        == "pre_mean_fidelity <= threshold"
        and float(gate.get("pre_mean_fidelity_threshold", -1))
        == FROZEN_THRESHOLD
        and gate.get("otherwise") == "retain_stage2",
        "Frozen selective gate mismatch",
    )
    require(
        bootstrap.get("unit") == "case_id"
        and int(bootstrap.get("replicates", -1)) == BOOTSTRAP_REPLICATES
        and int(bootstrap.get("seed", -1)) == BOOTSTRAP_SEED,
        "Frozen bootstrap protocol mismatch",
    )
    require(
        fusion_validation.get("status") == "passed"
        and fusion_validation.get("confirmation_test") is True,
        "Phase 4F-5 is not PASSED",
    )
    require(
        saved_result.get("status") == "passed"
        and saved_result.get("confirmation_test") is True
        and saved_result.get("confirmatory_evaluation") is True,
        "Saved Phase 4F-6 result is not PASSED confirmatory output",
    )
    require(
        saved_result.get("lambda_selection_performed") is False
        and saved_result.get("selective_threshold_selection_performed") is False
        and saved_result.get("protocol_retuning_performed") is False
        and saved_result.get("selective_gate_applied") is True,
        "Saved Phase 4F-6 protocol flags are invalid",
    )

    # Independent reconstruction from Phase 4F-4 stage rows.
    stage = pd.read_csv(STAGE_OUTPUTS, low_memory=False)
    require(len(stage) == 22638, "Stage-output row count != 22638")
    stage["sig"] = stage["feature_signature"].map(normalize_signature)
    stage2 = stage[stage["stage"].eq("Stage2")].copy()
    stage3_all = stage[stage["stage"].eq("Stage3")].copy()
    stage3 = stage3_all[bool_series(stage3_all["actually_fused"])].copy()
    require(len(stage2) == 7539, "Stage2 row count != 7539")
    require(len(stage3_all) == 5969, "Stage3 row count != 5969")
    require(len(stage3) == 1041, "Fused Stage3 row count != 1041")

    group_key = ["case_id", "prefix", "predicted_class", "sig"]
    stage2_groups = {
        k: v.copy()
        for k, v in stage2.groupby(group_key, sort=False, dropna=False)
    }
    reconstructed = []
    for _, post in stage3.iterrows():
        k = (
            post["case_id"],
            post["prefix"],
            post["predicted_class"],
            post["sig"],
        )
        require(k in stage2_groups, f"Missing Stage2 input group: {k}")
        inputs = stage2_groups[k]
        model_count = int(inputs["model"].nunique())
        require(model_count >= 2, f"Non-fusable matched group: {k}")
        require(
            model_count == int(post["source_model_count"]),
            f"Source-model count mismatch: {k}",
        )
        fids = inputs["Fid_r"].astype(float).to_numpy()
        reliabilities = inputs["R_r"].astype(float).to_numpy()
        pre = float(np.mean(fids))
        fixed = float(post["Fid_r"])
        accept = pre <= FROZEN_THRESHOLD
        selective = fixed if accept else pre
        reconstructed.append(
            {
                "case_id": str(post["case_id"]),
                "prefix": int(post["prefix"]),
                "predicted_class": int(post["predicted_class"]),
                "feature_signature": str(post["sig"]),
                "stage2_input_count": int(len(inputs)),
                "stage2_model_count": model_count,
                "stage2_models": json.dumps(
                    sorted(str(x) for x in inputs["model"].unique()),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "pre_mean_fidelity": pre,
                "pre_fidelity_sd": float(np.std(fids, ddof=0)),
                "pre_min_fidelity": float(np.min(fids)),
                "pre_max_fidelity": float(np.max(fids)),
                "pre_mean_reliability": float(np.mean(reliabilities)),
                "fixed_stage3_fidelity": fixed,
                "fixed_stage3_reliability": float(post["R_r"]),
                "fixed_stage3_last_K": (
                    np.nan if pd.isna(post["last_K"]) else float(post["last_K"])
                ),
                "frozen_threshold": FROZEN_THRESHOLD,
                "accept_stage3": bool(accept),
                "decision": "accept_stage3" if accept else "retain_stage2",
                "selective_fidelity": selective,
                "fixed_delta_vs_stage2": fixed - pre,
                "selective_delta_vs_fixed_stage3": selective - fixed,
                "selective_delta_vs_stage2": selective - pre,
            }
        )

    detail = pd.DataFrame(reconstructed)
    require(len(detail) == 1041, "Reconstructed detail count != 1041")
    accept_count = int(detail["accept_stage3"].sum())
    require(accept_count == 229, f"Accepted Stage3 count != 229: {accept_count}")

    saved_detail = pd.read_csv(SAVED_DETAIL, low_memory=False)
    compare_detail(detail, saved_detail)
    print(f"[Compare] Gate detail: {len(detail)} / {len(saved_detail)} identities and values matched")

    fixed_mean = float(detail["fixed_stage3_fidelity"].mean())
    stop_mean = float(detail["pre_mean_fidelity"].mean())
    selective_mean = float(detail["selective_fidelity"].mean())
    accept_rate = float(detail["accept_stage3"].mean())

    summary_rows = [
        {
            "method": "fixed_stage3",
            "N_groups": len(detail),
            "N_cases": int(detail["case_id"].nunique()),
            "mean_fidelity": fixed_mean,
            "stage3_accept_count": None,
            "stage3_accept_rate": None,
        },
        {
            "method": "stop_stage2",
            "N_groups": len(detail),
            "N_cases": int(detail["case_id"].nunique()),
            "mean_fidelity": stop_mean,
            "stage3_accept_count": None,
            "stage3_accept_rate": None,
        },
        {
            "method": "selective_stage3",
            "N_groups": len(detail),
            "N_cases": int(detail["case_id"].nunique()),
            "mean_fidelity": selective_mean,
            "stage3_accept_count": accept_count,
            "stage3_accept_rate": accept_rate,
        },
    ]
    saved_summary = pd.read_csv(SAVED_SUMMARY, low_memory=False)
    compare_table(summary_rows, saved_summary, "method")
    print("[Compare] Selective summaries: PASSED")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = [
        clustered_bootstrap(
            detail,
            "selective_fidelity",
            "fixed_stage3_fidelity",
            rng,
        ),
        clustered_bootstrap(
            detail,
            "selective_fidelity",
            "pre_mean_fidelity",
            rng,
        ),
        clustered_bootstrap(
            detail,
            "fixed_stage3_fidelity",
            "pre_mean_fidelity",
            rng,
        ),
    ]
    saved_comparisons = pd.read_csv(SAVED_COMPARISONS, low_memory=False)
    compare_table(comparisons, saved_comparisons, "comparison")
    print("[Compare] Case-clustered bootstrap: PASSED")

    saved_inventory = saved_result["inventory"]
    saved_fidelity = saved_result["fidelity"]
    require(
        int(saved_inventory["matched_fused_stage3_groups"]) == len(detail)
        and int(saved_inventory["accepted_stage3_groups"]) == accept_count
        and int(saved_inventory["retained_stage2_groups"])
        == len(detail) - accept_count
        and close(saved_inventory["stage3_accept_rate"], accept_rate),
        "Saved result inventory mismatch",
    )
    expected_fidelity = {
        "fixed_stage3": fixed_mean,
        "stop_stage2": stop_mean,
        "selective_stage3": selective_mean,
        "selective_gain_vs_fixed_stage3": selective_mean - fixed_mean,
        "selective_gain_vs_stop_stage2": selective_mean - stop_mean,
    }
    for field, value in expected_fidelity.items():
        require(
            close(saved_fidelity[field], value),
            f"Saved result fidelity mismatch: {field}",
        )
    require(
        len(saved_result["case_clustered_comparisons"]) == len(comparisons),
        "Saved result comparison count mismatch",
    )
    for expected, saved in zip(
        comparisons,
        saved_result["case_clustered_comparisons"],
    ):
        require(
            expected["comparison"] == saved["comparison"],
            "Saved JSON comparison identity mismatch",
        )
        for field, value in expected.items():
            if field == "comparison":
                continue
            if isinstance(value, int):
                require(int(saved[field]) == value, f"Saved JSON integer mismatch: {field}")
            else:
                require(close(saved[field], value), f"Saved JSON numeric mismatch: {field}")
    print("[Compare] Result JSON: PASSED")

    sel_fixed = comparisons[0]
    sel_stop = comparisons[1]
    validation = {
        "scope": (
            "ETCE Phase 4F-7 independent validation of the validation-frozen "
            "selective Stage-3 gate on untouched confirmation TEST"
        ),
        "status": "passed",
        "confirmation_test": True,
        "confirmatory_evaluation": True,
        "true_outcome_labels_used": False,
        "lambda": FROZEN_LAMBDA,
        "frozen_threshold": FROZEN_THRESHOLD,
        "lambda_selection_performed": False,
        "selective_threshold_selection_performed": False,
        "protocol_retuning_performed": False,
        "selective_gate_applied_in_validator": True,
        "reproduced": {
            "matched_fused_stage3_groups": int(len(detail)),
            "cases_represented": int(detail["case_id"].nunique()),
            "accepted_stage3_groups": accept_count,
            "retained_stage2_groups": int(len(detail) - accept_count),
            "fixed_stage3_fidelity": fixed_mean,
            "stop_stage2_fidelity": stop_mean,
            "selective_stage3_fidelity": selective_mean,
            "selective_gain_vs_fixed_stage3": selective_mean - fixed_mean,
            "selective_gain_vs_stop_stage2": selective_mean - stop_mean,
            "selective_vs_fixed_case_clustered_95ci": [
                sel_fixed["case_clustered_95_lo"],
                sel_fixed["case_clustered_95_hi"],
            ],
            "selective_vs_stop_case_clustered_95ci": [
                sel_stop["case_clustered_95_lo"],
                sel_stop["case_clustered_95_hi"],
            ],
        },
        "checks": {
            "frozen_config_hashes_verified": True,
            "phase4f5_independent_fusion_validation_passed": True,
            "all_gate_detail_identities_match": True,
            "all_gate_detail_values_match": True,
            "all_gate_decisions_match": True,
            "summary_matches": True,
            "case_clustered_bootstrap_matches": True,
            "result_json_matches": True,
        },
        "next_step": (
            "Use the independently validated confirmation result to update the "
            "ETCE empirical claims, while retaining the non-universal fusion narrative."
        ),
    }
    write_json(OUT_VALIDATION, validation)

    print()
    print("=" * 116)
    print("ETCE PHASE 4F-7 INDEPENDENT SELECTIVE STAGE-3 CONFIRMATION VALIDATION PASSED")
    print("=" * 116)
    print(f"Matched groups reproduced : {len(detail)}")
    print(f"Accepted Stage3 reproduced: {accept_count}/{len(detail)} ({accept_rate:.2%})")
    print(f"Fixed Stage3 fidelity     : {fixed_mean:.12f}")
    print(f"Stop at Stage2 fidelity   : {stop_mean:.12f}")
    print(f"Selective fidelity        : {selective_mean:.12f}")
    print(f"Gain vs fixed Stage3      : {selective_mean - fixed_mean:+.12f}")
    print(f"Gain vs stop Stage2       : {selective_mean - stop_mean:+.12f}")
    print(
        "95% CI vs fixed          : "
        f"[{sel_fixed['case_clustered_95_lo']:+.12f}, "
        f"{sel_fixed['case_clustered_95_hi']:+.12f}]"
    )
    print(
        "95% CI vs stop           : "
        f"[{sel_stop['case_clustered_95_lo']:+.12f}, "
        f"{sel_stop['case_clustered_95_hi']:+.12f}]"
    )
    print("Threshold selection performed: False")
    print("Protocol retuning performed: False")
    print(f"Validation JSON: {OUT_VALIDATION}")
    print("NEXT: update the paper's confirmation-result claims and tables.")


if __name__ == "__main__":
    main()
