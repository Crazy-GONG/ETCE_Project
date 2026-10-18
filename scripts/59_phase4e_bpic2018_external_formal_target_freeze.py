#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage59
BPIC2018 External Formal Target Freeze

Purpose
-------
Freeze the external explanation cohort BEFORE surrogate/rule generation.

Frozen design
-------------
- Population: BPIC2018 formal external TEST (2017) from Stage56/Stage58.
- Select exactly 100 cases.
- Selection is BLIND to:
    * ground-truth label
    * black-box probability
    * black-box predicted class
    * surrogate output
    * counterfactual output
    * explanation quality
- Selection uses only case_id through a deterministic SHA256 rank.
- Prefixes: P5 / P10 / P15 / P20.
- Models: RF / XGB / MLP / LSTM.
- Formal explanation targets: 100 * 4 * 4 = 1600.
- No black-box seed is selected here. Stage59 freezes model/prefix/case target
  identities only; canonical model-instance binding is deferred to the next
  frozen stage so Stage59 itself remains prediction-blind.

This stage does NOT:
- train a model,
- inspect Stage58 probabilities,
- select cases by labels or prediction outcomes,
- generate surrogate rules,
- generate counterfactuals,
- perform D-S fusion.

Default output
--------------
D:\\ETCE\\results\\phase4\\phase4e_bpic2018_external\\stage59_formal_targets
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


STAGE = 59
PREFIXES = (5, 10, 15, 20)
MODELS = ("RF", "XGB", "MLP", "LSTM")
N_CASES = 100

# Protocol constant, fixed before looking at any label/prediction/explanation.
SELECTION_SALT = "ETCE_BP18_EXTERNAL_FORMAL_TARGET_FREEZE_V1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        entries = list(path.iterdir())
        if entries and not overwrite:
            raise FileExistsError(
                f"Output directory already contains files: {path}\n"
                "Use --overwrite only to reproduce the exact same Stage59 protocol."
            )
        if overwrite:
            import shutil
            for p in entries:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
    path.mkdir(parents=True, exist_ok=True)


def verify_stage56(stage56_dir: Path) -> Dict[str, Any]:
    required = (
        "protocol_frozen.json",
        "dataset_summary.json",
        "consistency_checks.json",
        "sha256_manifest.json",
    )
    for name in required:
        if not (stage56_dir / name).exists():
            raise FileNotFoundError(stage56_dir / name)

    protocol = read_json(stage56_dir / "protocol_frozen.json")
    summary = read_json(stage56_dir / "dataset_summary.json")
    checks = read_json(stage56_dir / "consistency_checks.json")

    if protocol.get("stage") != 56:
        raise RuntimeError("Stage59 expects Stage56 representation input.")
    if not protocol.get("all_hard_checks_pass", False):
        raise RuntimeError("Stage56 protocol is not PASS.")
    if not summary.get("all_hard_checks_pass", False):
        raise RuntimeError("Stage56 summary is not PASS.")
    if any(not v.get("pass", False) for v in checks.values()):
        raise RuntimeError("Stage56 consistency checks are not fully PASS.")

    return {"protocol": protocol, "summary": summary}


def verify_stage58(stage58_dir: Path) -> Dict[str, Any]:
    required = (
        "protocol_frozen.json",
        "consistency_checks.json",
        "sha256_manifest.json",
        "final_test_metrics_summary.csv",
    )
    for name in required:
        if not (stage58_dir / name).exists():
            raise FileNotFoundError(stage58_dir / name)

    protocol = read_json(stage58_dir / "protocol_frozen.json")
    checks = read_json(stage58_dir / "consistency_checks.json")

    if protocol.get("stage") != 58:
        raise RuntimeError("Stage59 expects a completed Stage58.")
    if not protocol.get("all_hard_checks_pass", False):
        raise RuntimeError("Stage58 formal external TEST is not PASS.")
    if any(not v.get("pass", False) for v in checks.values()):
        raise RuntimeError("Stage58 consistency checks are not fully PASS.")
    if protocol.get("test_based_tuning", True):
        raise RuntimeError("Stage58 reports TEST-based tuning; external protocol invalid.")

    return {"protocol": protocol, "checks": checks}


def load_case_ids_only(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intentionally access only case_id and year arrays.
    Ground-truth y is not loaded for Stage59 selection.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        if "case_id" not in z.files or "year" not in z.files:
            raise RuntimeError(f"Missing case_id/year in {npz_path}")
        case_id = z["case_id"].astype(str)
        year = z["year"].astype(np.int16)
    return case_id, year


def rank_digest(case_id: str) -> str:
    payload = f"{SELECTION_SALT}|{case_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ETCE Stage59 - BPIC2018 External Formal Target Freeze"
    )
    ap.add_argument("--project_root", type=Path, default=Path(r"D:\ETCE"))
    ap.add_argument("--stage56_dir", type=Path, default=None)
    ap.add_argument("--stage58_dir", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.project_root.resolve()

    stage56_dir = (
        args.stage56_dir.resolve()
        if args.stage56_dir
        else root / "results" / "phase4" / "phase4e_bpic2018_external" / "stage56_features"
    )
    stage58_dir = (
        args.stage58_dir.resolve()
        if args.stage58_dir
        else root / "results" / "phase4" / "phase4e_bpic2018_external" / "stage58_formal_external_test"
    )
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else root / "results" / "phase4" / "phase4e_bpic2018_external" / "stage59_formal_targets"
    )

    ensure_clean_dir(out_dir, args.overwrite)

    print("=" * 118)
    print("ETCE STAGE59 - BPIC2018 EXTERNAL FORMAL TARGET FREEZE")
    print("=" * 118)
    print(f"Stage56        : {stage56_dir}")
    print(f"Stage58        : {stage58_dir}")
    print(f"Output         : {out_dir}")
    print(f"Selection      : deterministic SHA256(case_id), n={N_CASES}")
    print(f"Prefixes       : {list(PREFIXES)}")
    print(f"Models         : {list(MODELS)}")
    print("Uses labels    : NO")
    print("Uses predictions: NO")
    print("Uses explanations: NO")
    print("=" * 118)

    s56 = verify_stage56(stage56_dir)
    print("[Stage56 gate] PASS")
    verify_stage58(stage58_dir)
    print("[Stage58 gate] PASS")

    # Build case sets from each frozen 2017 prefix representation.
    case_ids_by_prefix: Dict[int, np.ndarray] = {}
    index_by_prefix: Dict[int, Dict[str, int]] = {}

    for p in PREFIXES:
        npz_path = stage56_dir / f"final_P{p}_test2017.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)

        case_ids, years = load_case_ids_only(npz_path)

        if len(case_ids) != len(set(case_ids.tolist())):
            raise RuntimeError(f"Duplicate 2017 case_id in P{p}.")
        if set(years.tolist()) != {2017}:
            raise RuntimeError(f"P{p} contains non-2017 rows.")

        case_ids_by_prefix[p] = case_ids
        index_by_prefix[p] = {cid: i for i, cid in enumerate(case_ids.tolist())}

    base = set(case_ids_by_prefix[5].tolist())
    for p in PREFIXES[1:]:
        if set(case_ids_by_prefix[p].tolist()) != base:
            raise RuntimeError(f"2017 case membership differs between P5 and P{p}.")

    expected_n = int(
        s56["summary"]["prediction_year_counts"].get(
            "2017",
            s56["summary"]["prediction_year_counts"].get(2017, -1),
        )
    )
    if len(base) != expected_n:
        raise RuntimeError(
            f"2017 cohort mismatch: actual={len(base)}, Stage56 expected={expected_n}"
        )
    if len(base) < N_CASES:
        raise RuntimeError(f"Not enough 2017 cases to freeze {N_CASES} targets.")

    # Blind deterministic selection.
    ranked = sorted(
        ((rank_digest(cid), cid) for cid in base),
        key=lambda x: (x[0], x[1]),
    )
    selected = ranked[:N_CASES]
    selected_ids = [cid for _, cid in selected]

    # selected_cases.csv intentionally contains no labels/predictions.
    selected_cases_path = out_dir / "selected_cases.csv"
    with selected_cases_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "selection_rank",
                "case_id",
                "selection_sha256",
                "case_year",
                "P5_row_index",
                "P10_row_index",
                "P15_row_index",
                "P20_row_index",
            ]
        )
        for rank, (digest, cid) in enumerate(selected, start=1):
            w.writerow(
                [
                    rank,
                    cid,
                    digest,
                    2017,
                    index_by_prefix[5][cid],
                    index_by_prefix[10][cid],
                    index_by_prefix[15][cid],
                    index_by_prefix[20][cid],
                ]
            )

    # Formal target identities: exactly 1600, still seed-agnostic.
    formal_targets_path = out_dir / "formal_targets.csv"
    formal_target_rows = []
    with formal_targets_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "target_id",
                "case_rank",
                "case_id",
                "case_year",
                "prefix",
                "model",
                "blackbox_seed_binding",
                "selection_basis",
            ]
        )

        target_idx = 0
        for case_rank, cid in enumerate(selected_ids, start=1):
            for p in PREFIXES:
                for model in MODELS:
                    target_idx += 1
                    target_id = f"BP18_EXT_T{target_idx:04d}"
                    row = [
                        target_id,
                        case_rank,
                        cid,
                        2017,
                        p,
                        model,
                        "DEFERRED_TO_NEXT_FROZEN_STAGE",
                        "case_id_sha256_only",
                    ]
                    w.writerow(row)
                    formal_target_rows.append(row)

    # Save selected row indices for later efficient slicing.
    np.savez_compressed(
        out_dir / "selected_case_indices.npz",
        case_id=np.asarray(selected_ids, dtype=str),
        P5=np.asarray([index_by_prefix[5][x] for x in selected_ids], dtype=np.int32),
        P10=np.asarray([index_by_prefix[10][x] for x in selected_ids], dtype=np.int32),
        P15=np.asarray([index_by_prefix[15][x] for x in selected_ids], dtype=np.int32),
        P20=np.asarray([index_by_prefix[20][x] for x in selected_ids], dtype=np.int32),
    )

    # Hard gates.
    target_ids = [r[0] for r in formal_target_rows]
    target_case_ids = [r[2] for r in formal_target_rows]

    checks = {
        "stage56_pass": {"pass": True},
        "stage58_pass": {"pass": True},
        "formal_test_year_only": {"pass": True, "year": 2017},
        "case_membership_identical_across_prefixes": {"pass": True},
        "raw_formal_test_case_count": {
            "pass": len(base) == expected_n,
            "actual": len(base),
            "expected": expected_n,
        },
        "selected_case_count": {
            "pass": len(selected_ids) == N_CASES,
            "actual": len(selected_ids),
            "expected": N_CASES,
        },
        "selected_case_ids_unique": {
            "pass": len(set(selected_ids)) == N_CASES,
            "actual": len(set(selected_ids)),
            "expected": N_CASES,
        },
        "formal_target_count": {
            "pass": len(formal_target_rows) == N_CASES * len(PREFIXES) * len(MODELS),
            "actual": len(formal_target_rows),
            "expected": N_CASES * len(PREFIXES) * len(MODELS),
        },
        "formal_target_ids_unique": {
            "pass": len(set(target_ids)) == len(target_ids),
        },
        "all_formal_targets_from_selected_cases": {
            "pass": set(target_case_ids) == set(selected_ids),
        },
        "selection_uses_label": {"pass": True, "actual": False, "expected": False},
        "selection_uses_prediction": {"pass": True, "actual": False, "expected": False},
        "selection_uses_explanation": {"pass": True, "actual": False, "expected": False},
        "blackbox_seed_binding_deferred": {"pass": True},
    }

    all_pass = all(v.get("pass", False) for v in checks.values())
    write_json(out_dir / "consistency_checks.json", checks)

    protocol = {
        "stage": STAGE,
        "name": "BPIC2018 External Formal Target Freeze",
        "dataset": "BPIC2018",
        "created_local": datetime.now().astimezone().isoformat(),
        "formal_test_year": 2017,
        "formal_test_population": len(base),
        "selected_cases": N_CASES,
        "prefixes": list(PREFIXES),
        "models": list(MODELS),
        "formal_targets": N_CASES * len(PREFIXES) * len(MODELS),
        "selection": {
            "method": "deterministic SHA256 ranking of case_id",
            "salt": SELECTION_SALT,
            "uses_label": False,
            "uses_blackbox_probability": False,
            "uses_blackbox_prediction": False,
            "uses_surrogate_output": False,
            "uses_counterfactual_output": False,
            "uses_explanation_quality": False,
        },
        "blackbox_seed_binding": "deferred to next frozen stage",
        "test_based_case_selection": False,
        "all_hard_checks_pass": all_pass,
    }
    write_json(out_dir / "protocol_frozen.json", protocol)

    selected_set_payload = "\n".join(selected_ids).encode("utf-8")
    selected_set_sha256 = hashlib.sha256(selected_set_payload).hexdigest()

    summary = {
        "stage": STAGE,
        "dataset": "BPIC2018",
        "formal_test_year": 2017,
        "formal_test_cases": len(base),
        "selected_cases": len(selected_ids),
        "formal_targets": len(formal_target_rows),
        "selected_case_set_sha256": selected_set_sha256,
        "all_hard_checks_pass": all_pass,
    }
    write_json(out_dir / "dataset_summary.json", summary)

    md = f"""# ETCE Stage59 — BPIC2018 External Formal Target Freeze

- Formal TEST year: **2017**
- Formal TEST population: **{len(base):,} cases**
- Selected cases: **{len(selected_ids)}**
- Prefixes: **P5 / P10 / P15 / P20**
- Models: **RF / XGB / MLP / LSTM**
- Formal explanation targets: **{len(formal_target_rows):,}**

Selection is based only on deterministic SHA256 ranking of `case_id`.
Ground-truth labels, black-box probabilities/predictions, surrogate outputs,
counterfactual outputs, and explanation-quality measures are not used.

Black-box seed/model-instance binding is intentionally deferred to the next
frozen stage; Stage59 therefore remains prediction-blind.

Selected case-set SHA256:
`{selected_set_sha256}`

Gate: **{"PASS" if all_pass else "FAIL"}**
"""
    (out_dir / "STAGE59_FORMAL_TARGET_FREEZE.md").write_text(md, encoding="utf-8")

    files = sorted(
        p for p in out_dir.iterdir()
        if p.is_file() and p.name != "sha256_manifest.json"
    )
    write_json(
        out_dir / "sha256_manifest.json",
        {
            "stage": STAGE,
            "algorithm": "SHA256",
            "generated_local": datetime.now().astimezone().isoformat(),
            "files": {
                str(p): {
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
                for p in files
            },
        },
    )

    print("\n" + "=" * 118)
    print("ETCE STAGE59 SUMMARY")
    print("=" * 118)
    print(f"2017 formal TEST population : {len(base):,}")
    print(f"Selected cases              : {len(selected_ids)}")
    print(f"Prefixes                    : {list(PREFIXES)}")
    print(f"Models                      : {list(MODELS)}")
    print(f"Formal targets              : {len(formal_target_rows):,}")
    print(f"Selected-set SHA256         : {selected_set_sha256}")
    print("Label used for selection    : NO")
    print("Prediction used selection   : NO")
    print("Explanation used selection  : NO")
    print(f"Hard gates                  : {'PASS' if all_pass else 'FAIL'}")
    print("=" * 118)

    if not all_pass:
        print("STAGE59 FAILED. Do not proceed to surrogate/rule generation.")
        return 2

    print(
        "\nSTAGE59 FORMAL TARGET COHORT FROZEN SUCCESSFULLY.\n"
        "NEXT: freeze the canonical black-box instance binding for the 1600 targets, "
        "then reproduce the external surrogate/rule pipeline without using TEST outcomes for tuning."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
