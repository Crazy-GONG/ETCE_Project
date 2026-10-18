#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 68E: summarize original-vs-masked Stage 1--3 fidelity.

The comparison is restricted to the common P10/P15/P20 scope. It also applies
the already frozen selective Stage-3 gate (accept Stage 3 iff mean Stage-2
fidelity <= 0.50) without tuning the threshold.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    os.environ.get("ETCE_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
PREFIXES = (10, 15, 20)
STAGES = ("Stage1", "Stage2", "Stage3")
FROZEN_LAMBDA = 5.0
FROZEN_TAU = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--original-stage-outputs", type=Path, default=None)
    parser.add_argument("--masked-stage-outputs", type=Path, default=None)
    parser.add_argument("--blackbox-comparison", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def normalize_signature(value: object) -> str:
    try:
        return json.dumps(json.loads(str(value)), separators=(",", ":"))
    except Exception:
        return str(value)


def load_stage_outputs(path: Path) -> pd.DataFrame:
    require(path.is_file(), f"Missing stage output: {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "lambda", "stage", "case_id", "prefix", "predicted_class",
        "feature_signature", "actually_fused", "Fid_r",
    }
    require(required.issubset(frame.columns), f"Unexpected stage-output schema: {path}")
    frame = frame[frame["lambda"].astype(float).eq(FROZEN_LAMBDA)].copy()
    frame["prefix"] = frame["prefix"].astype(int)
    frame = frame[frame["prefix"].isin(PREFIXES)].copy()
    frame["actually_fused_bool"] = bool_series(frame["actually_fused"])
    frame["Fid_r"] = pd.to_numeric(frame["Fid_r"], errors="coerce")
    require(frame["Fid_r"].notna().all(), f"Missing fidelity values: {path}")
    frame["sig"] = frame["feature_signature"].map(normalize_signature)
    return frame


def stage_summary(scope: str, frame: pd.DataFrame) -> list[dict]:
    rows = []
    for stage in STAGES:
        staged = frame[frame["stage"].eq(stage)]
        fused = staged[staged["actually_fused_bool"]]
        require(len(fused) > 0, f"No genuinely fused {stage} rows for {scope}")
        rows.append({
            "scope": scope,
            "prefix": "P10-P20",
            "stage": stage,
            "group_count": int(len(staged)),
            "fused_count": int(len(fused)),
            "fused_mean_fidelity": float(fused["Fid_r"].mean()),
        })
        for prefix in PREFIXES:
            staged_p = staged[staged["prefix"].eq(prefix)]
            fused_p = staged_p[staged_p["actually_fused_bool"]]
            rows.append({
                "scope": scope,
                "prefix": f"P{prefix}",
                "stage": stage,
                "group_count": int(len(staged_p)),
                "fused_count": int(len(fused_p)),
                "fused_mean_fidelity": (
                    float(fused_p["Fid_r"].mean()) if len(fused_p) else np.nan
                ),
            })
    return rows


def selective_stage3(scope: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    key = ["case_id", "prefix", "predicted_class", "sig"]
    stage2 = frame[frame["stage"].eq("Stage2")].copy()
    stage3 = frame[
        frame["stage"].eq("Stage3") & frame["actually_fused_bool"]
    ].copy()
    require(not stage3.duplicated(key).any(), f"Duplicate Stage-3 keys: {scope}")
    stage3_idx = stage3.set_index(key)

    rows = []
    for group_key, inputs in stage2.groupby(key, dropna=False):
        if len(inputs) < 2 or group_key not in stage3_idx.index:
            continue
        post = stage3_idx.loc[group_key]
        pre = float(inputs["Fid_r"].mean())
        fixed = float(post["Fid_r"])
        accept = pre <= FROZEN_TAU
        rows.append({
            "scope": scope,
            "case_id": group_key[0],
            "prefix": int(group_key[1]),
            "predicted_class": int(group_key[2]),
            "feature_signature": group_key[3],
            "stage2_input_count": int(len(inputs)),
            "stop_stage2_fidelity": pre,
            "fixed_stage3_fidelity": fixed,
            "accept_stage3": bool(accept),
            "selective_stage3_fidelity": fixed if accept else pre,
        })

    detail = pd.DataFrame(rows)
    require(len(detail) == len(stage3), f"Unmatched Stage2->Stage3 groups: {scope}")
    summary = {
        "scope": scope,
        "group_count": int(len(detail)),
        "case_count": int(detail["case_id"].nunique()),
        "tau": FROZEN_TAU,
        "tau_selected_here": False,
        "stage3_accept_count": int(detail["accept_stage3"].sum()),
        "stage3_accept_rate": float(detail["accept_stage3"].mean()),
        "stop_stage2_mean_fidelity": float(detail["stop_stage2_fidelity"].mean()),
        "fixed_stage3_mean_fidelity": float(detail["fixed_stage3_fidelity"].mean()),
        "selective_stage3_mean_fidelity": float(
            detail["selective_stage3_fidelity"].mean()
        ),
    }
    return detail, summary


def comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    original = summary[summary["scope"].eq("original")].drop(columns="scope")
    masked = summary[summary["scope"].eq("masked")].drop(columns="scope")
    merged = original.merge(
        masked,
        on=["prefix", "stage"],
        suffixes=("_original", "_masked"),
        validate="one_to_one",
    )
    merged["delta_fused_mean_fidelity"] = (
        merged["fused_mean_fidelity_masked"]
        - merged["fused_mean_fidelity_original"]
    )
    return merged


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    original_path = (args.original_stage_outputs or (
        root / "results/phase4/phase4d_formal_test/formal_test_frozen_ds_fusion"
        / "phase4d_frozen_ds_stage_outputs.csv"
    )).expanduser().resolve()
    masked_path = (args.masked_stage_outputs or (
        root / "results/stage68_outcome_proxy_masking/frozen_ds_fusion"
        / "masked_frozen_ds_stage_outputs.csv"
    )).expanduser().resolve()
    blackbox_path = (args.blackbox_comparison or (
        root / "results/stage68_outcome_proxy_masking/blackbox/metrics"
        / "masked_vs_original_macro_f1.csv"
    )).expanduser().resolve()
    out = (args.out_dir or (
        root / "results/stage68_outcome_proxy_masking/comparison"
    )).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    original = load_stage_outputs(original_path)
    masked = load_stage_outputs(masked_path)
    stage_rows = stage_summary("original", original) + stage_summary("masked", masked)
    stage_frame = pd.DataFrame(stage_rows)
    comparison = comparison_table(stage_frame)

    original_sel, original_sel_summary = selective_stage3("original", original)
    masked_sel, masked_sel_summary = selective_stage3("masked", masked)
    selective_summary = pd.DataFrame([original_sel_summary, masked_sel_summary])
    selective_comparison = {
        "tau": FROZEN_TAU,
        "tau_selected_here": False,
        "delta_stop_stage2_mean_fidelity": (
            masked_sel_summary["stop_stage2_mean_fidelity"]
            - original_sel_summary["stop_stage2_mean_fidelity"]
        ),
        "delta_fixed_stage3_mean_fidelity": (
            masked_sel_summary["fixed_stage3_mean_fidelity"]
            - original_sel_summary["fixed_stage3_mean_fidelity"]
        ),
        "delta_selective_stage3_mean_fidelity": (
            masked_sel_summary["selective_stage3_mean_fidelity"]
            - original_sel_summary["selective_stage3_mean_fidelity"]
        ),
    }

    require(blackbox_path.is_file(), f"Missing Stage 68A comparison: {blackbox_path}")
    blackbox = pd.read_csv(blackbox_path)
    require("delta_macro_f1" in blackbox.columns, "Stage 68A comparison schema mismatch")
    max_abs_blackbox_delta = float(blackbox["delta_macro_f1"].abs().max())
    max_abs_stage_delta = float(comparison["delta_fused_mean_fidelity"].abs().max())

    stage_frame.to_csv(out / "stage_fidelity_original_and_masked.csv", index=False)
    comparison.to_csv(out / "stage_fidelity_comparison.csv", index=False)
    pd.concat([original_sel, masked_sel], ignore_index=True).to_csv(
        out / "selective_stage3_detail.csv", index=False
    )
    selective_summary.to_csv(out / "selective_stage3_summary.csv", index=False)

    result = {
        "stage": "68E",
        "status": "passed",
        "prefix_scope": list(PREFIXES),
        "frozen_lambda": FROZEN_LAMBDA,
        "frozen_tau": FROZEN_TAU,
        "lambda_selected_here": False,
        "tau_selected_here": False,
        "blackbox_max_absolute_macro_f1_delta": max_abs_blackbox_delta,
        "stage_fidelity_max_absolute_delta": max_abs_stage_delta,
        "selective_stage3": selective_comparison,
        "interpretation_guardrail": (
            "Descriptive sensitivity comparison on a changed representation; "
            "group populations may differ and no masked result was used for tuning."
        ),
    }
    (out / "stage68_masking_sensitivity_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    overall = comparison[comparison["prefix"].eq("P10-P20")]
    lines = [
        "# ETCE Stage 68 Outcome-Proxy Masking Sensitivity",
        "",
        "- Common scope: P10, P15 and P20.",
        "- Frozen lambda: 5; selected on the original validation protocol.",
        "- Frozen selective Stage-3 threshold: 0.50; not retuned.",
        "- TEST outcome labels were not used.",
        f"- Maximum absolute black-box Macro-F1 change: {max_abs_blackbox_delta:.6f}.",
        f"- Maximum absolute Stage-fidelity change (all reported scopes): {max_abs_stage_delta:.6f}.",
        "",
        "## Stage 1--3 genuinely fused fidelity",
        "",
        "| Stage | Original | Masked | Delta | Original N | Masked N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        lines.append(
            f"| {row.stage} | {row.fused_mean_fidelity_original:.6f} | "
            f"{row.fused_mean_fidelity_masked:.6f} | "
            f"{row.delta_fused_mean_fidelity:+.6f} | "
            f"{row.fused_count_original} | {row.fused_count_masked} |"
        )
    lines += [
        "",
        "## Frozen selective Stage 3",
        "",
        "| Scope | Groups | Stop Stage 2 | Fixed Stage 3 | Selective Stage 3 | Accept rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in selective_summary.itertuples(index=False):
        lines.append(
            f"| {row.scope} | {row.group_count} | "
            f"{row.stop_stage2_mean_fidelity:.6f} | "
            f"{row.fixed_stage3_mean_fidelity:.6f} | "
            f"{row.selective_stage3_mean_fidelity:.6f} | "
            f"{row.stage3_accept_rate:.6f} |"
        )
    lines += [
        "",
        "This is a descriptive robustness check. Because masking can change rule availability and compatible groups, group counts are reported explicitly.",
        "",
    ]
    (out / "STAGE68_MASKING_SENSITIVITY_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("=" * 100)
    print("ETCE STAGE 68E OUTCOME-PROXY MASKING COMPARISON")
    print("=" * 100)
    print(comparison.to_string(index=False))
    print()
    print(selective_summary.to_string(index=False))
    print()
    print(f"Black-box max |delta Macro-F1|: {max_abs_blackbox_delta:.6f}")
    print(f"Stage fidelity max |delta|: {max_abs_stage_delta:.6f}")
    print(f"Outputs: {out}")
    print("STAGE 68E PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
