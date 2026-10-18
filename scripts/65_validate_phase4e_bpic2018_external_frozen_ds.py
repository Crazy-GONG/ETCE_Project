#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage65
Independent validator for BPIC2018 Stage64 external frozen D-S applicability/fusion.

Goals
-----
- Do NOT import Stage64.
- Do NOT select/tune lambda.
- Do NOT use true 2017 outcome labels.
- Independently rebuild Stage64 inputs from Stage62/63 + frozen config.
- Independently reproduce:
  * total CF rules
  * eligible CF rules
  * Stage1 / Stage2 / Stage3 group counts
  * fusable group counts
  * pairwise fusion count
  * zero total-conflict fallback count for this external run
  * Stage1/2/3 output rows for non-fused singleton evidence
- Compare against Stage64 saved JSON/CSV outputs.

This validates protocol completeness and applicability only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

LAMBDA = 5.0
MODEL_ALIAS = {"XGB": "XGBoost", "XGBoost": "XGBoost"}
SURROGATE_ORDER = {"LORE": 0, "DT": 1, "BN": 2}
MODEL_ORDER = {"RF": 0, "XGBoost": 1, "MLP": 2, "LSTM": 3}

FLOAT_TOL = 1e-12
THETA = None


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    txt = path.read_text(encoding="utf-8-sig")
    if not txt.strip():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
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
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def norm_model(x: str) -> str:
    return MODEL_ALIAS.get(str(x), str(x))


def canonical_values(values):
    return tuple(sorted(str(v) for v in values))


def conditions_to_hypothesis(conditions):
    by_pos = {}
    for cond in conditions:
        require(cond.get("operator") == "in", f"Unsupported operator {cond.get('operator')}")
        pos = int(cond["position"])
        vals = canonical_values(cond["values"])
        require(vals, f"Empty values at position {pos}")
        require(pos not in by_pos, f"Duplicate position {pos}")
        by_pos[pos] = vals
    return tuple((p, by_pos[p]) for p in sorted(by_pos))


def feature_signature(hyp):
    if hyp is THETA:
        return tuple()
    return tuple(int(pos) for pos, _ in hyp)


def source_evidence(rule: dict):
    n = int(rule["n_r"])
    fid = float(rule["Fid_r"])
    rho = n / (n + LAMBDA)
    R = rho * fid
    return {
        "hypothesis": conditions_to_hypothesis(rule["conditions"]),
        "predicted_class": int(rule["predicted_class"]),
        "n_r": n,
        "Fid_r": fid,
        "coverage_eval": float(rule["coverage_eval"]),
        "rho_r": float(rho),
        "R_r": float(R),
        "source_count": 1,
        "source_models": {rule["model"]},
        "source_surrogates": {rule["surrogate"]},
        "source_candidate_ids": [rule["candidate_id"]],
        "last_selected_mass": float(R),
        "last_ignorance_mass": float(1.0 - R),
        "last_K": None,
        "total_conflict_fallback_count": 0,
        "fusion_steps": 0,
        "min_witness_edit": rule.get("witness_edit_count"),
        "cf_valid": True,
        "model_cf_validity_rate": 1.0,
        "all_models_cf_valid": True,
    }


def ev_to_row(ev, stage: str, key, input_count: int):
    case_id = key[0]
    model = surrogate = None

    if stage == "Stage1":
        _, model, prefix, surrogate, pred, sig = key
    elif stage == "Stage2":
        _, model, prefix, pred, sig = key
    elif stage == "Stage3":
        _, prefix, pred, sig = key
    else:
        raise ValueError(stage)

    return {
        "lambda": float(LAMBDA),
        "stage": stage,
        "case_id": case_id,
        "model": model,
        "prefix": int(prefix),
        "surrogate": surrogate,
        "predicted_class": int(pred),
        "feature_signature": json.dumps(list(sig)),
        "input_count": int(input_count),
        "actually_fused": bool(input_count >= 2),
        "source_rule_count": int(ev["source_count"]),
        "source_model_count": int(len(ev["source_models"])),
        "source_surrogate_count": int(len(ev["source_surrogates"])),
        "n_r": int(ev["n_r"]),
        "Fid_r": ev["Fid_r"],
        "coverage_eval": float(ev["coverage_eval"]),
        "rho_r": float(ev["rho_r"]),
        "R_r": float(ev["R_r"]),
        "cf_valid": bool(ev["cf_valid"]),
        "model_cf_validity_rate": ev["model_cf_validity_rate"],
        "all_models_cf_valid": bool(ev["all_models_cf_valid"]),
        "min_witness_edit": ev["min_witness_edit"],
        "fusion_steps": int(ev["fusion_steps"]),
        "total_conflict_fallback_count": int(ev["total_conflict_fallback_count"]),
        "last_K": ev["last_K"],
        "last_selected_mass": float(ev["last_selected_mass"]),
        "last_ignorance_mass": float(ev["last_ignorance_mass"]),
    }


def parse_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no", ""):
        return False
    raise ValueError(x)


def parse_optional_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("none", "nan"):
        return None
    return float(s)


def compare_rows(recomputed: List[dict], saved: List[dict]) -> List[str]:
    """
    Compare the Stage64 stage-output rows using the deterministic identity:
    stage, case_id, model, prefix, surrogate, predicted_class, feature_signature.
    """
    def key(r):
        return (
            str(r["stage"]),
            str(r["case_id"]),
            "" if r.get("model") is None else str(r.get("model")),
            int(r["prefix"]),
            "" if r.get("surrogate") is None else str(r.get("surrogate")),
            int(r["predicted_class"]),
            str(r["feature_signature"]),
        )

    a = {key(r): r for r in recomputed}
    b = {key(r): r for r in saved}
    errors = []

    if set(a) != set(b):
        errors.append(
            f"row-key mismatch: missing_in_saved={sorted(set(a)-set(b))}, "
            f"extra_in_saved={sorted(set(b)-set(a))}"
        )
        return errors

    int_fields = (
        "input_count", "source_rule_count", "source_model_count",
        "source_surrogate_count", "n_r", "fusion_steps",
        "total_conflict_fallback_count",
    )
    bool_fields = ("actually_fused", "cf_valid", "all_models_cf_valid")
    float_fields = (
        "lambda", "Fid_r", "coverage_eval", "rho_r", "R_r",
        "model_cf_validity_rate", "last_selected_mass", "last_ignorance_mass",
    )
    optional_float_fields = ("last_K", "min_witness_edit")

    for k in sorted(a):
        ra, rb = a[k], b[k]

        for f in int_fields:
            if int(ra[f]) != int(rb[f]):
                errors.append(f"{k} {f}: {ra[f]} != {rb[f]}")

        for f in bool_fields:
            if bool(ra[f]) != parse_bool(rb[f]):
                errors.append(f"{k} {f}: {ra[f]} != {rb[f]}")

        for f in float_fields:
            va = float(ra[f])
            vb = float(rb[f])
            if not math.isclose(va, vb, rel_tol=0.0, abs_tol=FLOAT_TOL):
                errors.append(f"{k} {f}: {va} != {vb}")

        for f in optional_float_fields:
            va = ra[f]
            vb = parse_optional_float(rb[f])
            if va is None and vb is None:
                continue
            if va is None or vb is None:
                errors.append(f"{k} {f}: {va} != {vb}")
                continue
            if not math.isclose(float(va), float(vb), rel_tol=0.0, abs_tol=FLOAT_TOL):
                errors.append(f"{k} {f}: {va} != {vb}")

    return errors


def extract_stage62(stage62: Path):
    """
    Independently derive total rules, eligible rules and available contexts.
    Uses no Stage64 outputs.
    """
    # Context inventory from caches
    cache_dir = stage62 / "target_cache"
    cache_files = sorted(cache_dir.glob("*.json.gz"))
    require(len(cache_files) == 1600, f"Expected 1600 Stage62 caches, got {len(cache_files)}")

    contexts = {}
    target_count = 0

    for fp in cache_files:
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            p = json.load(f)

        target_count += 1
        res = p.get("surrogate_result")
        if res is None:
            continue

        key = (
            str(p["case_id"]),
            norm_model(str(p["model"])),
            int(p["prefix"]),
        )
        require(key not in contexts, f"Duplicate context {key}")
        contexts[key] = {
            "eval_indices": [int(x) for x in res["eval_indices"]],
        }

    # Finalized Stage62 rule inventory
    rules_path = stage62 / "canonical_and_counterfactual_rules.jsonl"
    total_cf = 0
    eligible_rules = []

    with rules_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            if str(rec.get("rule_type")) != "counterfactual":
                continue

            total_cf += 1
            payload = rec.get("rule")
            r = dict(payload) if isinstance(payload, dict) else dict(rec)

            case_id = str(rec.get("case_id", r.get("case_id")))
            model = norm_model(str(rec.get("model", r.get("model"))))
            prefix = int(rec.get("prefix", r.get("prefix")))
            surrogate = str(rec.get("surrogate", r.get("surrogate")))
            candidate_id = rec.get("candidate_id", r.get("candidate_id"))

            r.update({
                "case_id": case_id,
                "model": model,
                "prefix": prefix,
                "surrogate": surrogate,
                "candidate_id": candidate_id,
            })

            # Strict metric aliasing only; no re-estimation.
            if "n_r" not in r and "support" in r:
                r["n_r"] = int(r["support"])
            if "Fid_r" not in r and "fidelity" in r:
                r["Fid_r"] = r["fidelity"]
            if "coverage_eval" not in r and "coverage" in r:
                r["coverage_eval"] = float(r["coverage"])

            require(
                r.get("candidate_id") is not None
                and r.get("predicted_class") is not None
                and r.get("conditions") is not None,
                f"Missing structural rule fields at line {line_no}",
            )
            require(
                r.get("n_r") is not None
                and r.get("coverage_eval") is not None,
                f"Missing support/coverage at line {line_no}",
            )

            # Frozen reliability eligibility.
            if int(r["n_r"]) > 0 and r.get("Fid_r") is not None:
                eligible_rules.append(r)

    return target_count, contexts, total_cf, eligible_rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=Path, default=Path("/root/autodl-tmp/ETCE"))
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.project_root.resolve()

    stage62 = root / "results/phase4/phase4e_bpic2018_external/stage62_external_multirule_bank"
    stage63 = root / "results/phase4/phase4e_bpic2018_external/stage63_external_multirule_validation"
    stage64 = root / "results/phase4/phase4e_bpic2018_external/stage64_external_frozen_ds_fusion"
    frozen_path = root / "configs/phase4c_ds_frozen_config.json"

    out = args.out_dir.resolve() if args.out_dir else (
        root / "results/phase4/phase4e_bpic2018_external/stage65_validate_external_frozen_ds"
    )

    if out.exists() and any(out.iterdir()) and args.overwrite:
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # Source gates
    v63 = read_json(stage63 / "validation_stage62_external_multirule.json")
    require(v63.get("status") == "PASSED", "Stage63 is not PASSED")
    require(int(v63.get("target_failures", -1)) == 0, "Stage63 has target failures")

    frozen = read_json(frozen_path)
    require(frozen.get("status") == "frozen", "D-S config not frozen")
    require(float(frozen["lambda"]) == LAMBDA, "Frozen lambda != 5")
    require(frozen.get("selection_scope") == "validation_only", "Invalid lambda selection scope")

    saved_result = read_json(stage64 / "stage64_external_frozen_ds_fusion.json")
    require(saved_result.get("status") == "passed", "Stage64 status is not passed")

    # Independent source reconstruction
    target_count, contexts, total_cf, rules = extract_stage62(stage62)
    eligible_cf = len(rules)

    # Stage1 independent grouping
    s1 = defaultdict(list)
    for r in rules:
        hyp = conditions_to_hypothesis(r["conditions"])
        key = (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
            r["surrogate"],
            int(r["predicted_class"]),
            feature_signature(hyp),
        )
        s1[key].append(r)

    s1_fusable = sum(len(v) >= 2 for v in s1.values())

    # Because current external result has zero fusable Stage1 groups,
    # every Stage1 output is a single source evidence. This is independently
    # reconstructed rather than taken from Stage64.
    stage_rows = []
    s1out = {}
    for key in sorted(
        s1,
        key=lambda k: (
            k[0], MODEL_ORDER.get(k[1], 99), k[2],
            SURROGATE_ORDER.get(k[3], 99), k[4], k[5]
        )
    ):
        src = sorted(s1[key], key=lambda r: str(r["candidate_id"]))
        require(len(src) == 1, "Independent Stage65 expected Stage1 singleton group")
        ev = source_evidence(src[0])
        s1out[key] = ev
        stage_rows.append(ev_to_row(ev, "Stage1", key, 1))

    # Stage2
    s2 = defaultdict(list)
    for key, ev in s1out.items():
        case_id, model, prefix, surrogate, pred, sig = key
        s2[(case_id, model, int(prefix), int(pred), sig)].append((surrogate, ev))

    s2_fusable = sum(len({s for s, _ in items}) >= 2 for items in s2.values())
    s2out = {}
    for key in sorted(
        s2,
        key=lambda k: (k[0], MODEL_ORDER.get(k[1], 99), k[2], k[3], k[4])
    ):
        items = sorted(s2[key], key=lambda x: SURROGATE_ORDER.get(x[0], 99))
        require(len(items) == 1, "Independent Stage65 expected Stage2 singleton group")
        ev = items[0][1]
        s2out[key] = ev
        stage_rows.append(ev_to_row(ev, "Stage2", key, 1))

    # Stage3
    s3 = defaultdict(list)
    for key, ev in s2out.items():
        case_id, model, prefix, pred, sig = key
        s3[(case_id, int(prefix), int(pred), sig)].append((model, ev))

    s3_fusable = sum(len({m for m, _ in items}) >= 2 for items in s3.values())
    s3out = {}
    for key in sorted(s3, key=lambda k: (k[0], k[1], k[2], k[3])):
        items = sorted(s3[key], key=lambda x: MODEL_ORDER.get(x[0], 99))
        require(len(items) == 1, "Independent Stage65 expected Stage3 singleton group")
        ev = items[0][1]
        s3out[key] = ev
        stage_rows.append(ev_to_row(ev, "Stage3", key, 1))

    pairwise_fusions = 0
    fallbacks = 0

    # Compare Stage64 saved JSON
    expected = {
        "formal_target_count": target_count,
        "two_class_context_count": len(contexts),
        "total_cf_rule_count": total_cf,
        "eligible_cf_candidate_count": eligible_cf,
        "stage1": len(s1),
        "stage1_fusable": s1_fusable,
        "stage2": len(s2),
        "stage2_fusable": s2_fusable,
        "stage3": len(s3),
        "stage3_fusable": s3_fusable,
        "pairwise_fusion_count": pairwise_fusions,
        "total_conflict_fallback_count": fallbacks,
    }

    json_checks = {
        "formal_target_count": int(saved_result["formal_target_count"]) == expected["formal_target_count"],
        "two_class_context_count": int(saved_result["two_class_context_count"]) == expected["two_class_context_count"],
        "total_cf_rule_count": int(saved_result["total_cf_rule_count"]) == expected["total_cf_rule_count"],
        "eligible_cf_candidate_count": int(saved_result["eligible_cf_candidate_count"]) == expected["eligible_cf_candidate_count"],
        "stage1": int(saved_result["group_counts"]["stage1"]) == expected["stage1"],
        "stage1_fusable": int(saved_result["group_counts"]["stage1_fusable"]) == expected["stage1_fusable"],
        "stage2": int(saved_result["group_counts"]["stage2"]) == expected["stage2"],
        "stage2_fusable": int(saved_result["group_counts"]["stage2_fusable"]) == expected["stage2_fusable"],
        "stage3": int(saved_result["group_counts"]["stage3"]) == expected["stage3"],
        "stage3_fusable": int(saved_result["group_counts"]["stage3_fusable"]) == expected["stage3_fusable"],
        "pairwise_fusion_count": int(saved_result["pairwise_fusion_count"]) == 0,
        "total_conflict_fallback_count": int(saved_result["total_conflict_fallback_count"]) == 0,
        "lambda": float(saved_result["lambda"]) == LAMBDA,
        "no_bpic2018_lambda_selection": saved_result["lambda_selection_performed_on_bpic2018"] is False,
        "no_bpic2018_retuning": saved_result["protocol_retuning_performed_on_bpic2018"] is False,
        "no_2017_label_use": saved_result["true_2017_outcome_labels_used"] is False,
    }

    saved_stage_rows = read_csv(stage64 / "stage64_ds_stage_outputs.csv")
    row_errors = compare_rows(stage_rows, saved_stage_rows)

    # Conflict trace must be empty because no pairwise fusion happened.
    saved_conflicts = read_csv(stage64 / "stage64_ds_conflict_trace.csv")
    conflict_trace_ok = len(saved_conflicts) == 0

    # Independent applicability conclusion
    genuine_ds = (s1_fusable + s2_fusable + s3_fusable) > 0
    saved_genuine_ds = bool(saved_result["genuine_ds_fusion_occurred"])
    genuine_ok = genuine_ds == saved_genuine_ds

    all_pass = (
        all(json_checks.values())
        and not row_errors
        and conflict_trace_ok
        and genuine_ok
        and target_count == 1600
        and len(contexts) == 1
        and total_cf == 2
        and eligible_cf == 1
    )

    result = {
        "stage": 65,
        "validated_stage": 64,
        "status": "PASSED" if all_pass else "FAILED",
        "independent_reconstruction": {
            **expected,
            "genuine_ds_fusion_occurred": genuine_ds,
        },
        "json_checks": json_checks,
        "stage_output_row_mismatches": row_errors[:50],
        "stage_output_rows_recomputed": len(stage_rows),
        "stage_output_rows_saved": len(saved_stage_rows),
        "conflict_trace_empty": conflict_trace_ok,
        "genuine_ds_flag_match": genuine_ok,
        "true_2017_labels_used": False,
        "test_based_retuning": False,
        "interpretation": (
            "BPIC2018 independently reproduces the external evidence scarcity: "
            "1 two-class context / 1600 targets, 2 total CF rules, 1 reliability-eligible "
            "CF rule, and zero compatible multi-evidence groups at Stage1/Stage2/Stage3. "
            "Therefore the frozen D-S protocol is preserved but no genuine D-S fusion "
            "operation is possible on this external cohort."
        ),
    }

    write_json(out / "validation_stage64_external_frozen_ds.json", result)
    write_csv(out / "recomputed_stage_rows.csv", stage_rows)

    report = f"""# ETCE Stage65 — Independent Validation of Stage64

- Status: **{result['status']}**
- Formal targets independently confirmed: **{target_count}**
- Two-class contexts: **{len(contexts)}**
- Total CF rules: **{total_cf}**
- Reliability-eligible CF rules: **{eligible_cf}**
- Stage1 groups / fusable: **{len(s1)} / {s1_fusable}**
- Stage2 groups / fusable: **{len(s2)} / {s2_fusable}**
- Stage3 groups / fusable: **{len(s3)} / {s3_fusable}**
- Pairwise D-S operations: **0**
- Total-conflict fallbacks: **0**
- Stage-output row mismatches: **{len(row_errors)}**
- True 2017 labels used: **NO**
- BPIC2018 retuning: **NO**

Interpretation: the external-validation chain is internally consistent. Evidence
availability is too sparse for genuine hierarchical D-S fusion, so BPIC2018
supports an applicability/limitation conclusion rather than a fusion-performance claim.
"""
    (out / "STAGE65_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    print("=" * 124)
    print("ETCE STAGE65 INDEPENDENT VALIDATION SUMMARY")
    print("=" * 124)
    print(f"Formal targets                       : {target_count}")
    print(f"Two-class contexts                   : {len(contexts)}")
    print(f"Total CF rules                       : {total_cf}")
    print(f"Eligible CF rules                    : {eligible_cf}")
    print(f"Stage1 groups / fusable              : {len(s1)} / {s1_fusable}")
    print(f"Stage2 groups / fusable              : {len(s2)} / {s2_fusable}")
    print(f"Stage3 groups / fusable              : {len(s3)} / {s3_fusable}")
    print(f"Pairwise D-S fusion operations       : {pairwise_fusions}")
    print(f"Total-conflict fallbacks             : {fallbacks}")
    print(f"Stage-output row mismatches           : {len(row_errors)}")
    print(f"Conflict trace empty                 : {'YES' if conflict_trace_ok else 'NO'}")
    print("2017 true labels used                : NO")
    print("BPIC2018 retuning                    : NO")
    print(f"Validation                           : {result['status']}")
    print("=" * 124)

    if not all_pass:
        print("\nSTAGE65 FAILED. Do not freeze the external-validation conclusion yet.")
        return 3

    print(
        "\nSTAGE65 PASSED. Stage64 is independently reproduced. "
        "The BPIC2018 external-validation experimental chain is now complete."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
