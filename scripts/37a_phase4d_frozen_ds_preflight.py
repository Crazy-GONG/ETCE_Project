#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-4A: Frozen D-S formal-test preflight.

READ ONLY.
Purpose:
1) verify Phase 4D-2 / Phase 4D-3 passed artifacts exist;
2) print the exact frozen fusion protocol from phase4c_ds_frozen_config.json;
3) inspect the formal TEST fusion-eval-cache manifest;
4) inspect one deterministic target cache schema;
5) inspect the validation Stage-3 lambda=5 output schema if available.

No D-S fusion is executed.
No parameter is selected or changed.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FROZEN = ROOT / "configs/phase4c_ds_frozen_config.json"

TEST_BASE = (
    ROOT
    / "results/phase4/phase4d_formal_test/formal_test_multirule_bank"
)
TEST_VALIDATION = TEST_BASE / "validation_phase4d_formal_test_multirule.json"
TEST_CACHE_MANIFEST = TEST_BASE / "fusion_eval_cache_manifest.json"
TEST_CACHE_DIR = TEST_BASE / "fusion_eval_cache/targets"

GROUP_AUDIT = (
    ROOT
    / "results/phase4/phase4d_formal_test/"
      "formal_test_target_local_group_audit/"
      "formal_test_target_local_group_audit.json"
)

VALID_SWEEP = (
    ROOT
    / "results/phase4/phase4c_v2_hierarchical_ds_sweep/"
      "validation_ds_lambda_sweep.json"
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_gz_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def print_obj(title, obj):
    print()
    print("=" * 112)
    print(title)
    print("=" * 112)
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def summarize(obj, depth=0, max_depth=3):
    if depth >= max_depth:
        if isinstance(obj, dict):
            return {"<dict_keys>": sorted(map(str, obj.keys()))}
        if isinstance(obj, list):
            return {"<list_len>": len(obj)}
        return f"<{type(obj).__name__}>"

    if isinstance(obj, dict):
        return {str(k): summarize(v, depth + 1, max_depth) for k, v in obj.items()}
    if isinstance(obj, list):
        if not obj:
            return []
        return {
            "<list_len>": len(obj),
            "<first_item>": summarize(obj[0], depth + 1, max_depth),
        }
    return obj


def main():
    required = [FROZEN, TEST_VALIDATION, GROUP_AUDIT, TEST_CACHE_MANIFEST, TEST_CACHE_DIR]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required preflight input(s):\n" + "\n".join(missing))

    frozen = load_json(FROZEN)
    val = load_json(TEST_VALIDATION)
    audit = load_json(GROUP_AUDIT)
    manifest = load_json(TEST_CACHE_MANIFEST)

    print("=" * 112)
    print("ETCE PHASE 4D-4A FROZEN D-S FORMAL TEST PREFLIGHT")
    print("=" * 112)

    print(f"Frozen config : {FROZEN}")
    print(f"Frozen status : {frozen.get('status')}")
    print(f"Lambda        : {frozen.get('lambda')}")
    print(f"Selection     : {frozen.get('selection_scope')}")
    print(f"Test seen pre-freeze: {frozen.get('formal_test_seen_before_freeze')}")
    print()

    print("[Phase 4D gates]")
    print(f"Phase 4D-2 validation status-like field: {val.get('status', val.get('result', '<not top-level>'))}")
    print(f"Phase 4D-3 audit status: {audit.get('status')}")
    print(f"Stage1 status: {audit.get('stage1', {}).get('status')}")
    print()

    print_obj("FROZEN RELIABILITY", frozen.get("reliability"))
    print_obj("FROZEN BPA", frozen.get("bpa"))
    print_obj("FROZEN GROUPING", frozen.get("grouping"))
    print_obj("FROZEN FUSION", frozen.get("fusion"))

    print_obj("TEST FUSION CACHE MANIFEST", manifest)

    cache_files = sorted(TEST_CACHE_DIR.glob("*.json.gz"))
    print()
    print(f"TEST cache files found: {len(cache_files)}")
    if not cache_files:
        raise RuntimeError("No *.json.gz files found in formal TEST fusion cache.")

    sample = cache_files[0]
    payload = load_gz_json(sample)
    print(f"Deterministic sample cache: {sample.name}")
    print_obj("TEST CACHE SAMPLE SCHEMA", summarize(payload, max_depth=5))

    # Validation sweep schema is useful to guarantee test outputs match the
    # already-frozen validation implementation. It is optional only because
    # some migrations may not retain this JSON on the current machine.
    if VALID_SWEEP.exists():
        sweep = load_json(VALID_SWEEP)
        print_obj("VALIDATION D-S SWEEP TOP-LEVEL SCHEMA", summarize(sweep, max_depth=4))

        # Print protocol again from sweep to compare against frozen config.
        if "protocol" in sweep:
            print_obj("VALIDATION SWEEP PROTOCOL", sweep["protocol"])

        # Locate any lambda=5 stage records recursively by lightweight traversal.
        hits = []

        def walk(x, path="$"):
            if isinstance(x, dict):
                lam = x.get("lambda")
                stage = x.get("stage")
                try:
                    is5 = lam is not None and abs(float(lam) - 5.0) < 1e-12
                except Exception:
                    is5 = False
                if is5 or (stage is not None and any(k in x for k in (
                    "mean_conflict_K", "fused_mean_fidelity",
                    "fused_cf_validity_rate", "total_conflict_fallback_count"
                ))):
                    hits.append((path, summarize(x, max_depth=4)))
                for k, v in x.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    walk(v, f"{path}[{i}]")

        walk(sweep)
        print()
        print("=" * 112)
        print("VALIDATION LAMBDA/STAGE RECORD SCHEMA HITS")
        print("=" * 112)
        for path, obj in hits[:20]:
            print(f"\nPATH: {path}")
            print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print()
        print(f"[WARN] Validation sweep JSON not found: {VALID_SWEEP}")
        print("The frozen config is still authoritative, but upload/retain script 31 before implementing 37.")

    print()
    print("=" * 112)
    print("ETCE PHASE 4D-4A PREFLIGHT COMPLETE")
    print("=" * 112)
    print("No D-S fusion performed.")
    print("No lambda/BPA/grouping/conflict/tie rule modified.")
    print("NEXT: implement frozen formal TEST fusion from these exact schemas.")


if __name__ == "__main__":
    main()
