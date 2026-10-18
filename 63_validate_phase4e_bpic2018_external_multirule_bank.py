#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage63
Independent validator for BPIC2018 Stage62 external multi-rule bank.

It independently recomputes all 1600 formal targets from frozen inputs and
verifies Stage62 pools, digests, black-box probabilities/classes, pool status,
and the LORE/DT/BN result for every two-class target.

No model tuning, no true 2017 outcome-label use, no D-S fusion.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

MODELS = ("RF", "XGB", "MLP", "LSTM")
PREFIXES = (5, 10, 15, 20)
THRESHOLD = 0.5
EXPECTED_TARGETS = 1600
POOL_SIZE = 320
MAX_EDITS = 3
MAX_OPTIONS = 8

FLOAT_ATOL = 1e-10
PROB_ATOL = {"RF": 1e-10, "XGB": 1e-10, "MLP": 1e-7, "LSTM": 1e-6}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def json_default(x: Any):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, set):
        return sorted(x)
    raise TypeError(type(x))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def find_stage62_script(root: Path) -> Path:
    preferred = [
        root / "scripts/62_phase4e_bpic2018_external_multirule_bank_v4.py",
        root / "scripts/62_phase4e_bpic2018_external_multirule_bank.py",
    ]
    for p in preferred:
        if p.exists():
            return p
    hits = sorted((root / "scripts").glob("62_phase4e_bpic2018_external_multirule_bank*.py"))
    if not hits:
        raise FileNotFoundError("Stage62 script not found.")
    return hits[-1]


def normalize_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): normalize_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [normalize_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return normalize_json(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float):
        if math.isnan(x):
            return "__NaN__"
        if math.isinf(x):
            return "__Inf__" if x > 0 else "__-Inf__"
    return x


def recursive_compare(a: Any, b: Any, path: str = "root", atol: float = FLOAT_ATOL) -> List[str]:
    a = normalize_json(a)
    b = normalize_json(b)
    errors: List[str] = []

    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a), set(b)
        if ka != kb:
            errors.append(
                f"{path}: keys differ; missing_in_recomputed={sorted(kb-ka)}, "
                f"extra_in_recomputed={sorted(ka-kb)}"
            )
        for k in sorted(ka & kb):
            errors.extend(recursive_compare(a[k], b[k], f"{path}.{k}", atol))
        return errors

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        for i, (va, vb) in enumerate(zip(a, b)):
            errors.extend(recursive_compare(va, vb, f"{path}[{i}]", atol))
        return errors

    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            errors.append(f"{path}: {a!r} != {b!r}")
        return errors

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(float(a), float(b), abs_tol=atol, rel_tol=0.0):
            errors.append(f"{path}: {a!r} != {b!r}")
        return errors

    if a != b:
        errors.append(f"{path}: {a!r} != {b!r}")
    return errors


def rule_signature_rows(result: dict) -> List[dict]:
    rows = []
    for branch in result.get("branches", []):
        surrogate = str(branch.get("branch"))
        factual = branch.get("factual_rule")
        if factual is not None:
            rows.append({
                "surrogate": surrogate,
                "rule_type": "factual",
                "candidate_id": factual.get("candidate_id") if isinstance(factual, dict) else None,
                "rule": normalize_json(factual),
            })
        for i, rule in enumerate(branch.get("counterfactual_rules") or []):
            cid = rule.get("candidate_id") if isinstance(rule, dict) else None
            rows.append({
                "surrogate": surrogate,
                "rule_type": "counterfactual",
                "candidate_id": cid if cid is not None else f"cf_{i}",
                "rule": normalize_json(rule),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=Path, default=Path("/root/autodl-tmp/ETCE"))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.project_root.resolve()
    sys.path.insert(0, str(root / "src"))

    stage62_dir = root / "results/phase4/phase4e_bpic2018_external/stage62_external_multirule_bank"
    stage56_dir = root / "results/phase4/phase4e_bpic2018_external/stage56_features"
    stage60 = root / "results/phase4/phase4e_bpic2018_external/stage60_canonical_blackbox"
    stage61h = root / "results/phase4/phase4e_bpic2018_external/stage61h_frozen_constraints"

    out = args.out_dir.resolve() if args.out_dir else (
        root / "results/phase4/phase4e_bpic2018_external/stage63_external_multirule_validation"
    )
    if out.exists() and any(out.iterdir()) and args.overwrite:
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    stage62_script = find_stage62_script(root)
    s62 = import_from_path("etce_stage62_validator_dependency", stage62_script)
    stage56 = import_from_path("etce_stage56_stage63", root / "scripts/56_phase4e_bpic2018_feature_build.py")
    stage58 = import_from_path("etce_stage58_stage63", root / "scripts/58_phase4e_bpic2018_final_refit_formal_test.py")

    import etce.phase3.semantic_core as semantic_core
    import etce.phase4.surrogate_core as surrogate_core
    from etce.phase4.surrogate_core import generate_process_valid_pool
    from etce.phase4.stratified_core import run_all_surrogates_stratified

    constraints, family_map = s62.load_stage61h_constraints(stage61h)
    s62.patch_semantic_adapter(semantic_core, surrogate_core, constraints, family_map)

    mapping = read_json(stage56_dir / "mapping_final_fit2015_2016.json")
    adapters = {
        p: s62.PrefixFeatureAdapter(
            stage56, mapping, stage56_dir / f"final_P{p}_test2017.npz", p
        )
        for p in PREFIXES
    }

    bindings = s62.load_bindings(stage60)
    checkpoint_map = s62.load_checkpoint_map(stage60, root)

    if len(bindings) != EXPECTED_TARGETS:
        raise RuntimeError(f"Expected 1600 Stage60 bindings, got {len(bindings)}")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    status_df = pd.read_csv(stage62_dir / "target_status.csv", dtype={"case_id": str})
    summary62 = read_json(stage62_dir / "stage62_summary.json")
    cache_dir = stage62_dir / "target_cache"

    rules_by_target: Dict[Tuple[str, str, int], List[dict]] = {}
    with (stage62_dir / "canonical_and_counterfactual_rules.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (str(r["case_id"]), str(r["model"]), int(r["prefix"]))
            rules_by_target.setdefault(key, []).append(r)

    status_lookup = {}
    for _, r in status_df.iterrows():
        key = (str(r["case_id"]), str(r["model"]), int(r["prefix"]))
        if key in status_lookup:
            raise RuntimeError(f"Duplicate Stage62 status target: {key}")
        status_lookup[key] = r.to_dict()

    expected_keys = {
        (str(r["case_id"]), str(r["model"]), int(r["prefix"]))
        for _, r in bindings.iterrows()
    }
    if set(status_lookup) != expected_keys:
        raise RuntimeError("Stage62 target_status inventory differs from Stage60 formal targets.")

    print("=" * 124)
    print("ETCE STAGE63 - INDEPENDENT BPIC2018 STAGE62 VALIDATION")
    print("=" * 124)
    print(f"Stage62 source : {stage62_dir}")
    print(f"Stage62 code   : {stage62_script}")
    print(f"Targets        : {len(bindings)}")
    print(f"Device         : {device}")
    print("2017 true outcome labels used : NO")
    print("D-S fusion performed          : NO")
    print("=" * 124)

    predictors = {}
    for model in MODELS:
        for prefix in PREFIXES:
            predictors[(model, prefix)] = s62.CanonicalExternalPredictor(
                model, prefix, checkpoint_map[(model, prefix)],
                mapping, adapters[prefix], stage58, device
            )

    val_rows = []
    mismatches = []
    recomputed_rule_rows = []
    started = time.perf_counter()

    bindings2 = bindings.copy()
    bindings2["_mo"] = bindings2["model"].map({m: i for i, m in enumerate(MODELS)})
    bindings2 = bindings2.sort_values(["_mo", "prefix", "case_id"], kind="mergesort")

    for done, (_, row) in enumerate(bindings2.iterrows(), start=1):
        case_id = str(row["case_id"])
        model = str(row["model"])
        prefix = int(row["prefix"])
        key = (case_id, model, prefix)
        adapter = adapters[prefix]
        predictor = predictors[(model, prefix)]

        factual = adapter.decode_factual_activities(adapter.row_index(case_id))
        seed = s62.stable_local_seed(model, prefix, case_id)

        pool = generate_process_valid_pool(
            factual, constraints, rng_seed=seed, pool_size=POOL_SIZE,
            max_edits=MAX_EDITS, max_options_per_position=MAX_OPTIONS
        )
        pool = [tuple(x) for x in pool]
        digest = s62.pool_digest(pool)

        batch = adapter.perturbed_batch(case_id, pool)
        probs = np.asarray(predictor.predict_proba(batch), dtype=float)
        labels = (probs >= THRESHOLD).astype(int)
        counts = Counter(int(v) for v in labels.tolist())

        if len(counts) >= 2:
            status = "fusable_two_class_pool"
            result = run_all_surrogates_stratified(pool, labels, factual, int(labels[0]))
        else:
            status = "single_class_pool"
            result = None

        token = hashlib.sha256(f"{case_id}|{model}|{prefix}".encode("utf-8")).hexdigest()[:20]
        cache_path = cache_dir / f"{token}.json.gz"
        with gzip.open(cache_path, "rt", encoding="utf-8") as f:
            cached = json.load(f)

        errors = []
        for name, actual, expected in (
            ("case_id", str(cached.get("case_id")), case_id),
            ("model", str(cached.get("model")), model),
            ("prefix", int(cached.get("prefix")), prefix),
            ("seed", int(cached.get("seed")), int(seed)),
            ("pool_size", int(cached.get("pool_size_actual")), len(pool)),
            ("pool_digest", str(cached.get("pool_digest")), digest),
            ("status", str(cached.get("status")), status),
        ):
            if actual != expected:
                errors.append(f"{name}: {actual!r} != {expected!r}")

        cached_pool = [tuple(x) for x in cached.get("pool", [])]
        if cached_pool != pool:
            errors.append("exact pool inventory/order mismatch")

        cached_probs = np.asarray(cached.get("prob_positive", []), dtype=float)
        cached_labels = np.asarray(cached.get("bb_labels", []), dtype=int)

        if cached_probs.shape != probs.shape:
            prob_diff = float("inf")
            errors.append(f"probability shape mismatch: {cached_probs.shape} vs {probs.shape}")
        else:
            prob_diff = float(np.max(np.abs(cached_probs - probs))) if len(probs) else 0.0
            if prob_diff > PROB_ATOL[model]:
                errors.append(f"probability max_abs_diff {prob_diff:.3e} > {PROB_ATOL[model]:.1e}")

        if cached_labels.shape != labels.shape or not np.array_equal(cached_labels, labels):
            errors.append("black-box label vector mismatch")

        cached_counts = {int(k): int(v) for k, v in cached.get("label_counts", {}).items()}
        if cached_counts != dict(sorted(counts.items())):
            errors.append(f"label_counts mismatch: {cached_counts} vs {dict(sorted(counts.items()))}")

        srow = status_lookup[key]
        if str(srow["status"]) != status:
            errors.append("target_status status mismatch")
        if int(srow["pool_size"]) != len(pool):
            errors.append("target_status pool_size mismatch")
        if str(srow["pool_digest"]) != digest:
            errors.append("target_status pool_digest mismatch")

        if result is None:
            if cached.get("surrogate_result") is not None:
                errors.append("single-class target unexpectedly has cached surrogate_result")
            if rules_by_target.get(key):
                errors.append("single-class target unexpectedly has Stage62 rule records")
        else:
            cached_result = cached.get("surrogate_result")
            if cached_result is None:
                errors.append("two-class target missing cached surrogate_result")
            else:
                serr = recursive_compare(result, cached_result, "surrogate_result", FLOAT_ATOL)
                errors.extend(serr[:20])

            recomputed_rules = rule_signature_rows(result)
            for rr in recomputed_rules:
                recomputed_rule_rows.append({
                    "case_id": case_id, "model": model, "prefix": prefix, **rr
                })

            cached_compact = [{
                "surrogate": str(r["surrogate"]),
                "rule_type": str(r["rule_type"]),
                "candidate_id": r.get("candidate_id"),
                "rule": normalize_json(r.get("rule")),
            } for r in rules_by_target.get(key, [])]

            rerr = recursive_compare(recomputed_rules, cached_compact, "rules_jsonl", FLOAT_ATOL)
            errors.extend(rerr[:20])

        ok = not errors
        val_rows.append({
            "case_id": case_id,
            "model": model,
            "prefix": prefix,
            "status": status,
            "pool_size": len(pool),
            "class0_count": counts.get(0, 0),
            "class1_count": counts.get(1, 0),
            "pool_digest": digest,
            "max_probability_abs_diff": prob_diff,
            "pass": ok,
            "error_count": len(errors),
        })

        if not ok and len(mismatches) < 20:
            mismatches.append({
                "case_id": case_id, "model": model, "prefix": prefix, "errors": errors
            })

        if done % 25 == 0 or done == EXPECTED_TARGETS:
            elapsed = time.perf_counter() - started
            failed = sum(not r["pass"] for r in val_rows)
            print(
                f"  progress {done:4d}/{EXPECTED_TARGETS} | failed={failed} | elapsed={elapsed/60:.1f} min",
                flush=True,
            )

    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(out / "target_validation.csv", index=False, encoding="utf-8-sig")

    fusable = int((val_df["status"] == "fusable_two_class_pool").sum())
    single = int((val_df["status"] == "single_class_pool").sum())
    failed = int((~val_df["pass"]).sum())
    factual_rules = sum(1 for r in recomputed_rule_rows if r["rule_type"] == "factual")
    cf_rules = sum(1 for r in recomputed_rule_rows if r["rule_type"] == "counterfactual")

    aggregate = {
        "formal_targets": (int(summary62["formal_targets"]), len(val_df)),
        "fusable_two_class_targets": (int(summary62["fusable_two_class_targets"]), fusable),
        "single_class_pool_targets": (int(summary62["single_class_pool_targets"]), single),
        "surrogate_error_targets": (int(summary62["surrogate_error_targets"]), 0),
        "factual_rules": (int(summary62["factual_rules"]), factual_rules),
        "counterfactual_rules": (int(summary62["counterfactual_rules"]), cf_rules),
    }
    aggregate_pass = all(a == b for a, b in aggregate.values())
    status = "PASSED" if failed == 0 and aggregate_pass else "FAILED"

    result = {
        "stage": 63,
        "validated_stage": 62,
        "formal_targets_recomputed": len(val_df),
        "target_failures": failed,
        "fusable_two_class_pools_recomputed": fusable,
        "single_class_pools_recomputed": single,
        "factual_rules_recomputed": factual_rules,
        "counterfactual_rules_recomputed": cf_rules,
        "max_probability_abs_diff": float(val_df["max_probability_abs_diff"].max()),
        "aggregate_checks": {
            k: {"stage62": a, "recomputed": b, "pass": a == b}
            for k, (a, b) in aggregate.items()
        },
        "mismatch_examples": mismatches,
        "true_2017_outcome_labels_used": False,
        "test_based_retuning": False,
        "ds_fusion_performed": False,
        "status": status,
    }
    write_json(out / "validation_stage62_external_multirule.json", result)

    checks = {
        "all_1600_targets_independently_recomputed": {"pass": len(val_df) == EXPECTED_TARGETS},
        "all_target_caches_match": {"pass": failed == 0, "failed_targets": failed},
        "aggregate_counts_match": {"pass": aggregate_pass},
        "true_2017_labels_used": {"pass": True, "actual": False},
        "test_based_retuning": {"pass": True, "actual": False},
        "ds_fusion_performed": {"pass": True, "actual": False},
    }
    write_json(out / "consistency_checks.json", checks)

    report = f"""# ETCE Stage63 — Independent Stage62 Validation

- Formal targets independently recomputed: {len(val_df)}
- Target mismatches: {failed}
- Two-class/fusable pools: {fusable}
- Single-class pools: {single}
- Factual rules: {factual_rules}
- Counterfactual rules: {cf_rules}
- Maximum cache/recompute probability absolute difference: {result['max_probability_abs_diff']:.3e}
- True 2017 outcome labels used: NO
- TEST-based retuning: NO
- D-S fusion performed: NO

Validation status: **{status}**
"""
    (out / "STAGE63_INDEPENDENT_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    manifest = {}
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "sha256_manifest.json":
            manifest[p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    write_json(out / "sha256_manifest.json", {"stage": 63, "files": manifest})

    print("\n" + "=" * 124)
    print("ETCE STAGE63 INDEPENDENT VALIDATION SUMMARY")
    print("=" * 124)
    print(f"Formal targets recomputed             : {len(val_df)}")
    print(f"Target mismatches                     : {failed}")
    print(f"Fusable two-class pools               : {fusable}")
    print(f"Single-class pools                    : {single}")
    print(f"Factual rules                         : {factual_rules}")
    print(f"Counterfactual rules                  : {cf_rules}")
    print(f"Max probability cache/recompute Δ     : {result['max_probability_abs_diff']:.3e}")
    print("True 2017 labels used                 : NO")
    print("TEST-based retuning                   : NO")
    print("D-S fusion                            : NO")
    print(f"Validation                            : {status}")
    print("=" * 124)

    if status != "PASSED":
        print("\nSTAGE63 FAILED. Do not proceed to external D-S fusion.")
        return 3

    print(
        "\nSTAGE63 PASSED. Stage62 external pools/rules are independently reproduced.\n"
        "NEXT: apply the frozen lambda=5 external D-S protocol to eligible evidence only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
