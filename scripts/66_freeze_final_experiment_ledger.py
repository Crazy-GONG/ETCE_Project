#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage66
Final experiment freeze + manuscript/reviewer evidence ledger.

This stage performs NO training, NO model selection, NO retuning and NO test
inference. It freezes the already validated BPIC2017 + BPIC2018 conclusions
into manuscript-facing artifacts.

Outputs
-------
results/final_freeze_stage66/
  ETCE_FINAL_EXPERIMENT_LEDGER.json
  manuscript_core_results.csv
  reviewer_evidence_map.csv
  claim_guardrails.csv
  source_artifact_manifest.csv
  FINAL_EXPERIMENT_SUMMARY.md
  SHA256_MANIFEST.json

Run
---
cd /root/autodl-tmp/ETCE
./.venv/bin/python -u scripts/66_freeze_final_experiment_ledger.py \
  --project_root /root/autodl-tmp/ETCE
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def first_existing(paths: List[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def collect_sources(root: Path) -> List[dict]:
    candidates = [
        root / "configs/phase4c_ds_frozen_config.json",

        root / "results/phase4/phase4d_formal_test/formal_test_multirule_bank/"
               "validation_phase4d_formal_test_multirule.json",

        root / "results/phase4/phase4d_formal_test/formal_test_frozen_ds_fusion/"
               "validation_phase4d_frozen_ds_fusion.json",

        root / "results/phase4/phase4d_formal_test/formal_test_frozen_ds_fusion/"
               "formal_test_frozen_ds_fusion.json",

        root / "results/phase4/phase4e_bpic2018_external/"
               "stage63_external_multirule_validation/"
               "validation_stage62_external_multirule.json",

        root / "results/phase4/phase4e_bpic2018_external/"
               "stage64_external_frozen_ds_fusion/"
               "stage64_external_frozen_ds_fusion.json",

        root / "results/phase4/phase4e_bpic2018_external/"
               "stage65_validate_external_frozen_ds/"
               "validation_stage64_external_frozen_ds.json",
    ]

    rows = []
    for p in candidates:
        rows.append({
            "path": str(p.relative_to(root)) if p.exists() else str(p),
            "exists": p.exists(),
            "sha256": sha256_file(p) if p.exists() and p.is_file() else "",
            "bytes": p.stat().st_size if p.exists() and p.is_file() else "",
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=Path, default=Path("/root/autodl-tmp/ETCE"))
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.project_root.resolve()
    out = (
        args.out_dir.resolve()
        if args.out_dir
        else root / "results/final_freeze_stage66"
    )

    if out.exists() and any(out.iterdir()) and args.overwrite:
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Hard protocol gates
    # ------------------------------------------------------------------
    frozen_path = root / "configs/phase4c_ds_frozen_config.json"
    require(frozen_path.is_file(), f"Missing frozen D-S config: {frozen_path}")
    frozen = read_json(frozen_path)

    require(frozen.get("status") == "frozen", "D-S config status is not frozen")
    require(float(frozen.get("lambda")) == 5.0, "Frozen lambda is not 5")
    require(frozen.get("selection_scope") == "validation_only",
            "Lambda selection scope is not validation_only")

    stage65_path = (
        root / "results/phase4/phase4e_bpic2018_external/"
        "stage65_validate_external_frozen_ds/"
        "validation_stage64_external_frozen_ds.json"
    )
    require(stage65_path.is_file(), f"Missing Stage65 validation: {stage65_path}")
    stage65 = read_json(stage65_path)
    require(stage65.get("status") == "PASSED", "Stage65 is not PASSED")

    # ------------------------------------------------------------------
    # Frozen result ledger
    #
    # These values are the validated final results from the completed ETCE
    # experiment chain. Stage66 does not re-estimate them; it records them
    # as the manuscript-facing ledger.
    # ------------------------------------------------------------------
    ledger = {
        "schema_version": "etce-final-experiment-ledger-v1",
        "status": "FROZEN",
        "protocol": {
            "dataset_primary": "BPIC2017",
            "dataset_external": "BPIC2018",
            "prefixes": [5, 10, 15, 20],
            "black_boxes": ["RF", "XGBoost", "MLP", "LSTM"],
            "surrogates": ["LORE", "DT", "BN"],
            "formal_test_cases": 100,
            "formal_targets": 1600,
            "lambda": 5,
            "lambda_selected_on": "validation_only",
            "formal_test_retuning": False,
        },

        "bpic2017_rule_bank": {
            "canonical_rules": 9846,
            "cf_candidates": 7029,
            "eligible_cf_candidates": 6172,
            "target_surrogate_groups": 4800,
            "groups_with_ge2_cf_candidates": 1311,
            "max_cf_candidates_per_group": 26,
        },

        "bpic2017_grouping": {
            "stage1_groups": 5056,
            "stage1_fusable": 919,
            "stage2_groups": 4287,
            "stage2_fusable": 614,
            "stage3_groups": 3496,
            "stage3_fusable": 529,
        },

        "bpic2017_stage_comparison": {
            "original_to_stage1": {
                "N": 919,
                "delta_fidelity": 0.101551,
                "improved": 534,
                "tied": 212,
                "worsened": 173,
            },
            "stage1_to_stage2": {
                "N": 614,
                "delta_fidelity": 0.035433,
                "improved": 136,
                "tied": 443,
                "worsened": 35,
            },
            "stage2_to_stage3": {
                "N": 529,
                "delta_fidelity": -0.044821,
                "improved": 110,
                "tied": 196,
                "worsened": 223,
            },
        },

        "bpic2017_stage3": {
            "pre_stage3_fidelity": 0.713038,
            "post_stage3_fidelity": 0.668218,
            "prefix_deltas": {
                "P5": 0.0,
                "P10": -0.036395,
                "P15": -0.058500,
                "P20": -0.050085,
            },
        },

        "bpic2017_baselines_N529": {
            "DS_stage3_fidelity": 0.668218,
            "best_single_evidence": 0.665629,
            "validation_best_source": 0.681075,
            "uniform_vote": 0.668035,
        },

        "bpic2017_significance_N529": {
            "DS_vs_best_single_evidence": {
                "delta": 0.002589,
                "ci95": [-0.000842, 0.006824],
                "W_T_L": [15, 503, 11],
                "p": 0.567567,
                "rank_biserial_r": 0.1282,
                "significant": False,
            },
            "DS_vs_validation_best_source": {
                "delta": -0.012858,
                "ci95": [-0.023446, -0.002582],
                "p": 0.0353868,
                "holm_adjusted_p": 0.212321,
                "significant_after_correction": False,
            },
            "DS_vs_uniform_vote": {
                "delta": 0.000182,
                "ci95": [-0.005074, 0.005295],
                "significant": False,
            },
        },

        "bpic2017_agreement_conflict": {
            "validation_usable_targets": 993,
            "formal_usable_targets": 939,
            "agreement_thresholds": {
                "q33": 0.666667,
                "q67": 0.870968,
            },
            "stage2_fidelity_by_agreement": {
                "low": {"N": 52, "fidelity": 0.627839},
                "medium": {"N": 320, "fidelity": 0.844535},
                "high": {"N": 242, "fidelity": 0.900389},
            },
            "agreement_to_fidelity": {
                "N": 614,
                "pearson": 0.3116,
                "spearman": 0.3109,
            },
            "mean_group_agreement_to_mean_group_fidelity": {
                "N": 445,
                "pearson": 0.3798,
            },
            "mean_conflict_K": {
                "stage1": 0.1516,
                "stage2": 0.0189,
                "stage3": 0.0643,
            },
        },

        "support_aware_ablation": {
            "without_n_over_n_plus_lambda": {
                "stage1_fidelity": 0.861038,
                "stage2_fidelity": 0.866837,
                "stage3_fidelity": 0.736019,
            },
            "interpretation": (
                "Removing support-aware discounting raises fidelity on the formal test; "
                "this must be reported as an ablation/limitation rather than used to "
                "silently retune the frozen protocol."
            ),
        },

        "bpic2018_external": {
            "formal_targets": 1600,
            "two_class_contexts": 1,
            "two_class_context_rate": 0.000625,
            "total_cf_rules": 2,
            "eligible_cf_rules": 1,
            "stage1": {"groups": 1, "fusable": 0},
            "stage2": {"groups": 1, "fusable": 0},
            "stage3": {"groups": 1, "fusable": 0},
            "pairwise_ds_fusions": 0,
            "total_conflict_fallbacks": 0,
            "test_retuning": False,
            "true_2017_labels_used_for_rule_or_fusion_tuning": False,
            "stage65_validation": "PASSED",
            "interpretation": (
                "External applicability/limitation validation. Evidence availability "
                "is too sparse for genuine hierarchical D-S fusion; this is not an "
                "external performance-improvement result."
            ),
        },
    }

    # ------------------------------------------------------------------
    # Manuscript-facing result table
    # ------------------------------------------------------------------
    core_rows = [
        {
            "section": "Formal multi-rule bank",
            "result": "Eligible CF candidates",
            "value": 6172,
            "N_or_scope": "BPIC2017 formal test",
            "manuscript_role": "main/supporting",
        },
        {
            "section": "Stage hierarchy",
            "result": "Original → Stage1 ΔFidelity",
            "value": "+0.101551",
            "N_or_scope": "N=919",
            "manuscript_role": "main",
        },
        {
            "section": "Stage hierarchy",
            "result": "Stage1 → Stage2 ΔFidelity",
            "value": "+0.035433",
            "N_or_scope": "N=614",
            "manuscript_role": "main",
        },
        {
            "section": "Stage hierarchy",
            "result": "Stage2 → Stage3 ΔFidelity",
            "value": "-0.044821",
            "N_or_scope": "N=529",
            "manuscript_role": "main + limitation",
        },
        {
            "section": "Final Stage3",
            "result": "D-S Stage3 Fidelity",
            "value": "0.668218",
            "N_or_scope": "N=529",
            "manuscript_role": "main",
        },
        {
            "section": "Baselines",
            "result": "Best single evidence Fidelity",
            "value": "0.665629",
            "N_or_scope": "N=529",
            "manuscript_role": "main",
        },
        {
            "section": "Baselines",
            "result": "Validation-best source Fidelity",
            "value": "0.681075",
            "N_or_scope": "N=529",
            "manuscript_role": "main",
        },
        {
            "section": "Baselines",
            "result": "Uniform vote Fidelity",
            "value": "0.668035",
            "N_or_scope": "N=529",
            "manuscript_role": "main",
        },
        {
            "section": "Significance",
            "result": "D-S vs best-single Δ",
            "value": "+0.002589; p=0.567567; non-significant",
            "N_or_scope": "N=529",
            "manuscript_role": "main",
        },
        {
            "section": "Significance",
            "result": "D-S vs validation-best Δ",
            "value": "-0.012858; Holm p=0.212321; non-significant",
            "N_or_scope": "N=529",
            "manuscript_role": "main",
        },
        {
            "section": "Significance",
            "result": "D-S vs uniform vote Δ",
            "value": "+0.000182; non-significant",
            "N_or_scope": "N=529",
            "manuscript_role": "main",
        },
        {
            "section": "Agreement",
            "result": "Stage2 high-agreement Fidelity",
            "value": "0.900389",
            "N_or_scope": "N=242",
            "manuscript_role": "diagnostic/supporting",
        },
        {
            "section": "Conflict",
            "result": "Mean conflict K Stage1 / Stage2 / Stage3",
            "value": "0.1516 / 0.0189 / 0.0643",
            "N_or_scope": "formal fusion trace",
            "manuscript_role": "diagnostic/supporting",
        },
        {
            "section": "External validation",
            "result": "BPIC2018 two-class context availability",
            "value": "1/1600 = 0.0625%",
            "N_or_scope": "formal external targets",
            "manuscript_role": "limitation/external-validity",
        },
        {
            "section": "External validation",
            "result": "BPIC2018 genuine D-S fusion operations",
            "value": "0",
            "N_or_scope": "1 eligible CF rule",
            "manuscript_role": "limitation/external-validity",
        },
    ]

    reviewer_rows = [
        {
            "reviewer": "Reviewer 1",
            "concern": "Experimental protocol / leakage / prefix vs full-trace",
            "evidence": (
                "Case-level temporal split; fixed true prefixes; formal 100-case cohort "
                "selected without labels/black-box/explanation outputs; λ=5 frozen on "
                "validation before formal test."
            ),
            "where_to_use": "Experimental setup + response letter",
        },
        {
            "reviewer": "Reviewer 1",
            "concern": "Generalization / external validation",
            "evidence": (
                "BPIC2018 2015-2016 development/final-train and 2017 formal test; "
                "no 2017 tuning; Stage65 independently validates the external chain."
            ),
            "where_to_use": "External validation subsection + limitations",
        },
        {
            "reviewer": "Reviewer 2",
            "concern": "Process/temporal constraints in counterfactual generation",
            "evidence": (
                "Process-valid local neighborhoods with frozen train-side structural "
                "constraints; fixed-prefix interventions; no future information."
            ),
            "where_to_use": "Method + experiment protocol",
        },
        {
            "reviewer": "Reviewer 2",
            "concern": "Whether D-S actually improves explanation quality",
            "evidence": (
                "Stage-wise Fidelity improves from Original→Stage1 and Stage1→Stage2, "
                "but decreases Stage2→Stage3. Final D-S is not significantly better "
                "than the tested baselines."
            ),
            "where_to_use": "Results + discussion; avoid overclaiming",
        },
        {
            "reviewer": "Reviewer 3",
            "concern": "Evidence reliability / support dependence",
            "evidence": (
                "Frozen R(r)=n/(n+5)×Fid(r); support-aware ablation shows materially "
                "higher Fidelity without discounting, revealing the tradeoff between "
                "support conservatism and fidelity."
            ),
            "where_to_use": "Ablation + limitation",
        },
        {
            "reviewer": "Reviewer 3",
            "concern": "Evidence-source dependence / conflict",
            "evidence": (
                "Agreement correlates positively with Fidelity; mean K changes "
                "0.1516→0.0189→0.0643 across the hierarchy; high-agreement groups "
                "have substantially higher Stage2 Fidelity."
            ),
            "where_to_use": "Dependence/conflict analysis",
        },
    ]

    guardrails = [
        {
            "claim": "D-S significantly outperforms best single evidence",
            "status": "FORBIDDEN",
            "reason": "Δ=+0.002589, p=0.567567; not significant.",
        },
        {
            "claim": "D-S significantly outperforms validation-best source",
            "status": "FORBIDDEN",
            "reason": "D-S is lower by 0.012858; Holm-adjusted p=0.212321.",
        },
        {
            "claim": "D-S significantly outperforms uniform vote",
            "status": "FORBIDDEN",
            "reason": "Δ=+0.000182; no significant difference.",
        },
        {
            "claim": "Every fusion stage monotonically improves Fidelity",
            "status": "FORBIDDEN",
            "reason": "Stage2→Stage3 ΔFidelity = -0.044821.",
        },
        {
            "claim": "BPIC2018 confirms external D-S performance improvement",
            "status": "FORBIDDEN",
            "reason": "Only 1/1600 two-class contexts and 0 genuine D-S fusion operations.",
        },
        {
            "claim": "BPIC2018 externally validates protocol applicability under no retuning",
            "status": "ALLOWED WITH QUALIFICATION",
            "reason": (
                "Protocol remained frozen and independently reproducible, but evidence "
                "availability was insufficient for genuine fusion."
            ),
        },
        {
            "claim": "Agreement quality is associated with explanation Fidelity",
            "status": "ALLOWED",
            "reason": "Positive Pearson/Spearman correlations and tiered Fidelity analysis.",
        },
    ]

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    write_json(out / "ETCE_FINAL_EXPERIMENT_LEDGER.json", ledger)
    write_csv(out / "manuscript_core_results.csv", core_rows)
    write_csv(out / "reviewer_evidence_map.csv", reviewer_rows)
    write_csv(out / "claim_guardrails.csv", guardrails)

    sources = collect_sources(root)
    write_csv(out / "source_artifact_manifest.csv", sources)

    summary_md = f"""# ETCE Final Experiment Freeze — Stage66

## Freeze status

**FROZEN**

- Primary dataset: BPIC2017
- External validation dataset: BPIC2018
- Frozen D-S lambda: **5**
- Lambda selected on validation only: **YES**
- Formal-test retuning: **NO**
- BPIC2018 TEST retuning: **NO**
- BPIC2018 Stage65 independent validation: **PASSED**

## BPIC2017 main findings

1. Formal rule bank contains **9846** canonical rules, **7029** CF candidates and **6172** reliability-eligible CF candidates.
2. Hierarchical fusion behavior is not monotonic:
   - Original → Stage1: ΔFidelity = **+0.101551** (N=919)
   - Stage1 → Stage2: ΔFidelity = **+0.035433** (N=614)
   - Stage2 → Stage3: ΔFidelity = **-0.044821** (N=529)
3. Final Stage3 D-S Fidelity = **0.668218**.
4. Baselines on the same N=529:
   - Best single evidence = **0.665629**
   - Validation-best source = **0.681075**
   - Uniform vote = **0.668035**
5. Formal significance tests do **not** support a claim that D-S is significantly superior to these baselines.
6. Agreement is positively associated with Fidelity; high-agreement Stage2 groups reach **0.900389** mean Fidelity.
7. Mean conflict K changes **0.1516 → 0.0189 → 0.0643** from Stage1 to Stage3.
8. Removing support-aware discounting increases Fidelity strongly; this is a limitation/tradeoff result, not a basis for post-hoc retuning.

## BPIC2018 external-validation finding

- Formal targets: **1600**
- Two-class local contexts: **1 (0.0625%)**
- Total CF rules: **2**
- Reliability-eligible CF rules: **1**
- Stage1/2/3 fusable groups: **0 / 0 / 0**
- Genuine pairwise D-S operations: **0**
- Stage65 independent validation: **PASSED**

Interpretation: BPIC2018 is an **external applicability and limitation test**.
The frozen protocol remains reproducible without TEST-side retuning, but the
external cohort provides insufficient compatible counterfactual evidence for
genuine hierarchical D-S fusion.

## Manuscript claim boundary

The revised manuscript must NOT state that D-S significantly outperforms all
baselines, must NOT claim monotonic improvement across all three fusion stages,
and must NOT present BPIC2018 as evidence of external performance gain.

The defensible contribution is instead:
- leakage-controlled and prefix-consistent experimental protocol;
- validation-frozen reliability/BPA/fusion protocol;
- non-degenerate multi-rule Stage1/Stage2 fusion on BPIC2017;
- explicit analysis of conflict, agreement and evidence reliability;
- independently validated external evidence-availability limitation on BPIC2018.
"""
    (out / "FINAL_EXPERIMENT_SUMMARY.md").write_text(summary_md, encoding="utf-8")

    # Hash Stage66 outputs
    manifest = {}
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "SHA256_MANIFEST.json":
            manifest[p.name] = {
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
    write_json(out / "SHA256_MANIFEST.json", manifest)

    print("=" * 120)
    print("ETCE STAGE66 FINAL EXPERIMENT FREEZE")
    print("=" * 120)
    print("Frozen lambda                        : 5")
    print("BPIC2017 formal targets              : 1600")
    print("BPIC2017 Stage3 fusable groups       : 529")
    print("BPIC2017 Stage3 Fidelity             : 0.668218")
    print("D-S significant vs tested baselines : NO")
    print("BPIC2018 formal targets              : 1600")
    print("BPIC2018 two-class contexts          : 1")
    print("BPIC2018 eligible CF rules           : 1")
    print("BPIC2018 genuine D-S operations      : 0")
    print("BPIC2018 independent validation      : PASSED")
    print("Formal/BPIC2018 retuning             : NO / NO")
    print(f"Output                               : {out}")
    print("Status                               : FROZEN")
    print("=" * 120)
    print()
    print("NEXT: revise the manuscript Experimental Results / Discussion and reviewer response using Stage66 as the only result ledger.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
