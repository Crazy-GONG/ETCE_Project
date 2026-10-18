#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4F-6: apply the pre-frozen selective Stage-3 gate once to the
untouched confirmation cohort.

Frozen gate
-----------
Accept the Stage-3 output iff the mean held-out fidelity of its matched
Stage-2 inputs is <= 0.50; otherwise retain the Stage-2 evidence.

This script does not select lambda, select a threshold, retune the protocol,
fit a surrogate, or use true outcome labels.  It consumes only the already
independently validated Phase 4F-4/4F-5 outputs.

Run:
    cd D:\\ETCE
    python -u .\\scripts\\53_phase4f_apply_frozen_selective_stage3_confirmation.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE4F_ROOT = ROOT / "results/phase4/phase4f_confirmation"
FUSION_DIR = PHASE4F_ROOT / "confirmation_frozen_ds_fusion"
OUT_DIR = PHASE4F_ROOT / "confirmation_selective_stage3"

STAGE_OUTPUTS = FUSION_DIR / "phase4f_confirmation_frozen_ds_stage_outputs.csv"
FUSION_RESULT = FUSION_DIR / "confirmation_frozen_ds_fusion.json"
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

DETAIL_OUT = OUT_DIR / "phase4f_confirmation_selective_stage3_detail.csv"
SUMMARY_OUT = OUT_DIR / "phase4f_confirmation_selective_stage3_summary.csv"
COMPARISON_OUT = (
    OUT_DIR / "phase4f_confirmation_selective_stage3_case_clustered.csv"
)
RESULT_OUT = OUT_DIR / "confirmation_selective_stage3.json"
REPORT_OUT = OUT_DIR / "PHASE4F_SELECTIVE_STAGE3_CONFIRMATION_REPORT.md"

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


def bool_series(series: pd.Series):
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


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
    batch = 2_000
    for start in range(0, BOOTSTRAP_REPLICATES, batch):
        stop = min(start + batch, BOOTSTRAP_REPLICATES)
        idx = rng.integers(
            0,
            n_cases,
            size=(stop - start, n_cases),
            endpoint=False,
        )
        pooled_draws[start:stop] = (
            sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        )
        equal_case_draws[start:stop] = means[idx].mean(axis=1)

    group_dir = direction_counts(delta)
    case_dir = direction_counts(means)
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
        "group_wins": group_dir["wins"],
        "group_ties": group_dir["ties"],
        "group_losses": group_dir["losses"],
        "case_wins": case_dir["wins"],
        "case_ties": case_dir["ties"],
        "case_losses": case_dir["losses"],
    }


def fv(value):
    return f"{float(value):.6f}"


def main():
    required = [
        STAGE_OUTPUTS,
        FUSION_RESULT,
        FUSION_VALIDATION,
        SELECTION,
        FROZEN_DS,
        FROZEN_SELECTIVE,
    ]
    missing = [str(p) for p in required if not p.is_file()]
    require(not missing, "Missing required artifacts:\n" + "\n".join(missing))

    fusion_result = read_json(FUSION_RESULT)
    fusion_validation = read_json(FUSION_VALIDATION)
    selection = read_json(SELECTION)
    frozen_ds = read_json(FROZEN_DS)
    frozen_selective = read_json(FROZEN_SELECTIVE)

    # Frozen protocol gates.
    require(frozen_ds.get("status") == "frozen", "D-S protocol is not frozen")
    require(
        abs(float(frozen_ds.get("lambda", -1)) - FROZEN_LAMBDA) <= TOL,
        "Frozen lambda is not 5",
    )
    require(
        frozen_ds.get("selection_scope") == "validation_only",
        "Lambda was not selected on validation only",
    )
    require(
        selection.get("schema_version")
        == "etce-phase4f-confirmation-selection-v1"
        and selection.get("status") == "frozen",
        "Confirmation selection is not frozen",
    )
    require(
        int(selection.get("selected_case_count", -1)) == 200
        and int(selection.get("confirmation_target_count", -1)) == 3200
        and int(selection.get("global_sha256_rank_start", -1)) == 101
        and int(selection.get("global_sha256_rank_end", -1)) == 300
        and int(selection.get("overlap_with_original_phase4d", -1)) == 0,
        "Confirmation selection inventory/rank protocol mismatch",
    )
    require(
        selection.get("selection_uses_phase4d_or_phase4e_performance") is False,
        "Observed Phase-4D/4E performance influenced confirmation selection",
    )
    require(
        selection.get("frozen_ds_config_sha256") == sha256_file(FROZEN_DS),
        "Frozen D-S config hash changed",
    )
    require(
        selection.get("adaptive_config_sha256")
        == sha256_file(FROZEN_SELECTIVE),
        "Frozen selective Stage-3 config hash changed",
    )

    gate = frozen_selective.get("gate", {})
    require(
        frozen_selective.get("status")
        == "validation_frozen_formal_exploratory",
        "Selective Stage-3 config status changed",
    )
    require(
        float(frozen_selective.get("lambda", -1)) == FROZEN_LAMBDA
        and gate.get("accept_stage3_if")
        == "pre_mean_fidelity <= threshold"
        and abs(
            float(gate.get("pre_mean_fidelity_threshold", -1))
            - FROZEN_THRESHOLD
        )
        <= TOL
        and gate.get("otherwise") == "retain_stage2",
        "Selective Stage-3 gate differs from the frozen protocol",
    )
    bootstrap_cfg = frozen_selective.get("bootstrap", {})
    require(
        bootstrap_cfg.get("unit") == "case_id"
        and int(bootstrap_cfg.get("replicates", -1))
        == BOOTSTRAP_REPLICATES
        and int(bootstrap_cfg.get("seed", -1)) == BOOTSTRAP_SEED,
        "Frozen case-clustered bootstrap protocol changed",
    )

    require(
        fusion_result.get("status") == "passed"
        and fusion_result.get("confirmation_test") is True,
        "Phase 4F-4 fusion result is not PASSED confirmation output",
    )
    require(
        fusion_result.get("lambda_selection_performed") is False
        and fusion_result.get("selective_threshold_selection_performed")
        is False
        and fusion_result.get("selective_gate_applied") is False
        and fusion_result.get("protocol_retuning_performed") is False,
        "Phase 4F-4 indicates selection, retuning, or premature gating",
    )
    require(
        fusion_validation.get("status") == "passed"
        and fusion_validation.get("confirmation_test") is True,
        "Phase 4F-5 independent validation is not PASSED",
    )
    require(
        fusion_validation.get("lambda_selection_performed") is False
        and fusion_validation.get("selective_threshold_selection_performed")
        is False
        and fusion_validation.get("selective_gate_applied") is False
        and fusion_validation.get("protocol_retuning_performed") is False,
        "Phase 4F-5 indicates selection, retuning, or premature gating",
    )
    reproduced = fusion_validation.get("reproduced", {})
    require(
        int(reproduced.get("stage_output_rows", -1)) == 22638
        and int(reproduced.get("stage3_groups", -1)) == 5969
        and int(reproduced.get("stage3_fusable", -1)) == 1041,
        "Phase 4F-5 independently reproduced inventory mismatch",
    )

    print("=" * 116)
    print("ETCE PHASE 4F-6 PRE-FROZEN SELECTIVE STAGE-3 CONFIRMATION")
    print("=" * 116)
    print("Frozen gate: accept Stage3 iff pre-mean-fidelity <= 0.50")
    print("Otherwise: retain Stage2 evidence")
    print("Bootstrap: case_id clusters | 50000 replicates | seed=20261008")
    print("Protocol retuning: False")
    print()

    stage = pd.read_csv(STAGE_OUTPUTS, low_memory=False)
    require(len(stage) == 22638, f"Stage-output row count != 22638: {len(stage)}")
    require(
        np.allclose(stage["lambda"].astype(float), FROZEN_LAMBDA),
        "Non-frozen lambda found in stage outputs",
    )
    stage["sig"] = stage["feature_signature"].map(normalize_signature)

    stage2 = stage[stage["stage"].eq("Stage2")].copy()
    stage3_all = stage[stage["stage"].eq("Stage3")].copy()
    stage3 = stage3_all[bool_series(stage3_all["actually_fused"])].copy()
    require(len(stage2) == 7539, f"Stage2 rows != 7539: {len(stage2)}")
    require(len(stage3_all) == 5969, f"Stage3 rows != 5969: {len(stage3_all)}")
    require(len(stage3) == 1041, f"Fused Stage3 rows != 1041: {len(stage3)}")

    key = ["case_id", "prefix", "predicted_class", "sig"]
    require(
        not stage3.duplicated(key).any(),
        "Duplicate fused Stage3 identity",
    )
    stage2_groups = {
        group_key: frame.copy()
        for group_key, frame in stage2.groupby(key, sort=False, dropna=False)
    }

    rows = []
    for _, post in stage3.sort_values(key).iterrows():
        group_key = (
            post["case_id"],
            post["prefix"],
            post["predicted_class"],
            post["sig"],
        )
        require(group_key in stage2_groups, f"Missing Stage2 inputs: {group_key}")
        inputs = stage2_groups[group_key]
        model_count = int(inputs["model"].nunique())
        require(model_count >= 2, f"Stage3 group has <2 Stage2 models: {group_key}")
        require(
            model_count == int(post["source_model_count"]),
            f"Stage3 source-model count mismatch: {group_key}",
        )

        pre_fids = inputs["Fid_r"].astype(float).to_numpy()
        pre_rel = inputs["R_r"].astype(float).to_numpy()
        pre_fidelity = float(pre_fids.mean())
        post_fidelity = float(post["Fid_r"])
        accept = bool(pre_fidelity <= FROZEN_THRESHOLD)
        selective = post_fidelity if accept else pre_fidelity

        rows.append(
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
                "pre_mean_fidelity": pre_fidelity,
                "pre_fidelity_sd": float(pre_fids.std(ddof=0)),
                "pre_min_fidelity": float(pre_fids.min()),
                "pre_max_fidelity": float(pre_fids.max()),
                "pre_mean_reliability": float(pre_rel.mean()),
                "fixed_stage3_fidelity": post_fidelity,
                "fixed_stage3_reliability": float(post["R_r"]),
                "fixed_stage3_last_K": (
                    None if pd.isna(post["last_K"]) else float(post["last_K"])
                ),
                "frozen_threshold": FROZEN_THRESHOLD,
                "accept_stage3": accept,
                "decision": "accept_stage3" if accept else "retain_stage2",
                "selective_fidelity": selective,
                "fixed_delta_vs_stage2": post_fidelity - pre_fidelity,
                "selective_delta_vs_fixed_stage3": selective - post_fidelity,
                "selective_delta_vs_stage2": selective - pre_fidelity,
            }
        )

    detail = pd.DataFrame(rows)
    require(len(detail) == 1041, f"Selective detail groups != 1041: {len(detail)}")
    require(
        detail[key[:-1] + ["feature_signature"]].duplicated().sum() == 0,
        "Duplicate selective-detail identity",
    )

    fixed_mean = float(detail["fixed_stage3_fidelity"].mean())
    saved_fixed = float(
        fusion_result["summary"]["Stage3"]["fused_mean_fidelity"]
    )
    require(
        abs(fixed_mean - saved_fixed) <= TOL,
        f"Reconstructed fixed Stage3 mean mismatch: {fixed_mean} != {saved_fixed}",
    )

    stop_mean = float(detail["pre_mean_fidelity"].mean())
    selective_mean = float(detail["selective_fidelity"].mean())
    accept_count = int(detail["accept_stage3"].sum())
    accept_rate = float(detail["accept_stage3"].mean())

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
    comparisons_df = pd.DataFrame(comparisons)

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
    summary = pd.DataFrame(summary_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_OUT, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
    comparisons_df.to_csv(COMPARISON_OUT, index=False, encoding="utf-8-sig")

    sel_vs_fixed = comparisons[0]
    sel_vs_stop = comparisons[1]
    result = {
        "scope": (
            "ETCE Phase 4F-6 one-shot application of the validation-frozen "
            "selective Stage-3 gate to untouched confirmation TEST"
        ),
        "status": "passed",
        "confirmation_test": True,
        "confirmatory_evaluation": True,
        "true_outcome_labels_used": False,
        "lambda": FROZEN_LAMBDA,
        "frozen_threshold": FROZEN_THRESHOLD,
        "gate": {
            "accept_stage3_if": "pre_mean_fidelity <= threshold",
            "otherwise": "retain_stage2",
        },
        "lambda_selection_performed": False,
        "selective_threshold_selection_performed": False,
        "protocol_retuning_performed": False,
        "selective_gate_applied": True,
        "inventory": {
            "confirmation_cases_with_fused_stage3": int(
                detail["case_id"].nunique()
            ),
            "matched_fused_stage3_groups": int(len(detail)),
            "accepted_stage3_groups": accept_count,
            "retained_stage2_groups": int(len(detail) - accept_count),
            "stage3_accept_rate": accept_rate,
        },
        "fidelity": {
            "fixed_stage3": fixed_mean,
            "stop_stage2": stop_mean,
            "selective_stage3": selective_mean,
            "selective_gain_vs_fixed_stage3": selective_mean - fixed_mean,
            "selective_gain_vs_stop_stage2": selective_mean - stop_mean,
        },
        "case_clustered_comparisons": comparisons,
        "confirmatory_indicators": {
            "selective_gain_vs_fixed_positive": (
                selective_mean - fixed_mean > TOL
            ),
            "selective_vs_fixed_95ci_excludes_zero": (
                sel_vs_fixed["case_clustered_95_lo"] > 0
                or sel_vs_fixed["case_clustered_95_hi"] < 0
            ),
            "selective_gain_vs_stop_positive": (
                selective_mean - stop_mean > TOL
            ),
            "selective_vs_stop_95ci_excludes_zero": (
                sel_vs_stop["case_clustered_95_lo"] > 0
                or sel_vs_stop["case_clustered_95_hi"] < 0
            ),
        },
        "checks": {
            "confirmation_selection_hashes_verified": True,
            "phase4f4_fusion_passed": True,
            "phase4f5_independent_validation_passed": True,
            "frozen_gate_verified": True,
            "bootstrap_protocol_verified": True,
            "all_1041_fused_stage3_groups_matched_to_stage2_inputs": True,
            "fixed_stage3_mean_reproduced": True,
        },
        "outputs": {
            "detail_csv": str(DETAIL_OUT),
            "summary_csv": str(SUMMARY_OUT),
            "case_clustered_csv": str(COMPARISON_OUT),
            "report_md": str(REPORT_OUT),
        },
        "next_step": (
            "Independently validate the frozen selective Stage-3 confirmation "
            "outputs before using them in the paper."
        ),
    }
    write_json(RESULT_OUT, result)

    report = f"""# ETCE Phase 4F-6 Selective Stage-3 Confirmation

## Protocol

- Status: **PASSED**
- Untouched confirmation evaluation: **True**
- Frozen lambda: **5**
- Frozen gate: accept Stage 3 iff pre-mean fidelity <= **0.50**
- Otherwise: retain Stage-2 evidence
- Threshold selection on confirmation TEST: **False**
- Protocol retuning on confirmation TEST: **False**
- Bootstrap: case_id clusters, 50,000 replicates, seed 20261008

## Confirmation result

- Matched fused Stage-3 groups: **{len(detail)}**
- Cases represented: **{detail['case_id'].nunique()}**
- Accepted Stage-3 groups: **{accept_count}/{len(detail)}** ({accept_rate:.2%})
- Fixed Stage-3 fidelity: **{fixed_mean:.6f}**
- Stop-at-Stage-2 fidelity: **{stop_mean:.6f}**
- Selective Stage-3 fidelity: **{selective_mean:.6f}**
- Selective gain vs fixed Stage 3: **{selective_mean - fixed_mean:+.6f}**
- Selective gain vs stopping at Stage 2: **{selective_mean - stop_mean:+.6f}**

## Case-clustered 95% confidence intervals

- Selective vs fixed Stage 3: [{sel_vs_fixed['case_clustered_95_lo']:+.6f}, {sel_vs_fixed['case_clustered_95_hi']:+.6f}]
- Selective vs stop at Stage 2: [{sel_vs_stop['case_clustered_95_lo']:+.6f}, {sel_vs_stop['case_clustered_95_hi']:+.6f}]

This is the one-shot confirmatory transfer of the validation-frozen gate.  The
gate and threshold were not selected or modified using confirmation outcomes.
"""
    REPORT_OUT.write_text(report, encoding="utf-8")

    print("[Gate] matched fused Stage3 groups:", len(detail))
    print(
        f"[Gate] accepted Stage3: {accept_count}/{len(detail)} "
        f"({accept_rate:.2%})"
    )
    print()
    print("=" * 116)
    print("ETCE PHASE 4F-6 PRE-FROZEN SELECTIVE STAGE-3 CONFIRMATION PASSED")
    print("=" * 116)
    print(f"Fixed Stage3 fidelity   : {fv(fixed_mean)}")
    print(f"Stop at Stage2 fidelity : {fv(stop_mean)}")
    print(f"Selective fidelity      : {fv(selective_mean)}")
    print(f"Gain vs fixed Stage3    : {selective_mean - fixed_mean:+.6f}")
    print(f"Gain vs stop Stage2     : {selective_mean - stop_mean:+.6f}")
    print(
        "Case-clustered 95% CI vs fixed: "
        f"[{sel_vs_fixed['case_clustered_95_lo']:+.6f}, "
        f"{sel_vs_fixed['case_clustered_95_hi']:+.6f}]"
    )
    print(
        "Case-clustered 95% CI vs stop : "
        f"[{sel_vs_stop['case_clustered_95_lo']:+.6f}, "
        f"{sel_vs_stop['case_clustered_95_hi']:+.6f}]"
    )
    print("Frozen threshold: 0.50")
    print("Threshold selection performed: False")
    print("Protocol retuning performed: False")
    print(f"Result JSON: {RESULT_OUT}")
    print("NEXT: independently validate the selective confirmation outputs.")


if __name__ == "__main__":
    main()
