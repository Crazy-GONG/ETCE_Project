#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4C-v2: build a reusable validation evaluation cache for
hierarchical D-S fusion and lambda selection.

Why this cache is needed
------------------------
Phase 4B-v2 stores canonical rules and held-out rule metrics, but it does
not store the actual local pool / held-out eval sequences. Fused rules
must be evaluated again on the SAME deterministic validation
neighborhoods and the SAME deterministic class-stratified split.

This script therefore reconstructs the already validated 1600 local
targets, verifies every reconstructed neighborhood against the saved
Phase 4B-v2 manifest, and stores:
- full process-valid local pool;
- black-box labels;
- factual sequence / prediction / probability;
- exact fit indices;
- exact eval indices.

It DOES NOT:
- fit surrogates again;
- select lambda;
- perform D-S fusion;
- read formal test cases.

Run:
    python -u scripts/30_phase4c_v2_build_fusion_eval_cache.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ETCE_ROOT", DEFAULT_ROOT)).expanduser().resolve()
LOCAL_DEPS = ROOT / ".deps"
if sys.platform == "win32" and LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from etce.phase2.data import PREFIXES, load_frozen_common_cohort
from etce.phase3.core import CanonicalPredictor, load_process_constraints
from etce.phase4.surrogate_core import generate_process_valid_pool, sequence_hash
from etce.phase4.stratified_core import stratified_local_split
from etce.utils.io import read_json, write_json

RESULTS3 = ROOT / "results/phase3"
PHASE4B_V2 = ROOT / "results/phase4/phase4b_v2_multirule_bank"
OUT = ROOT / "results/phase4/phase4c_v2_fusion_eval_cache"
TARGETS = OUT / "targets"

MODELS = ("RF", "XGBoost", "MLP", "LSTM")
CALIBRATION_CASE_COUNT = 100
POOL_SIZE = 320
MAX_EDITS = 3
MAX_OPTIONS = 8
POOL_SEED_BASE = 0
PROB_TOL = 1e-5
CACHE_SCHEMA = "phase4c-v2-fusion-eval-target-v1"

_STATUS = {
    "target": "startup",
    "stage": "initializing",
    "completed": 0,
    "total": 0,
}
_STATUS_LOCK = threading.Lock()
_STOP_HEARTBEAT = threading.Event()


def fmt_seconds(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def set_status(target=None, stage=None, completed=None, total=None):
    with _STATUS_LOCK:
        if target is not None:
            _STATUS["target"] = target
        if stage is not None:
            _STATUS["stage"] = stage
        if completed is not None:
            _STATUS["completed"] = int(completed)
        if total is not None:
            _STATUS["total"] = int(total)


def heartbeat(started, interval=30):
    while not _STOP_HEARTBEAT.wait(interval):
        with _STATUS_LOCK:
            s = dict(_STATUS)
        elapsed = time.perf_counter() - started
        total = int(s.get("total", 0))
        completed = int(s.get("completed", 0))
        pct = 0.0 if total == 0 else 100.0 * completed / total
        print(
            f"[HEARTBEAT] {completed}/{total} ({pct:.2f}%) | "
            f"elapsed={fmt_seconds(elapsed)} | "
            f"target={s.get('target')} | stage={s.get('stage')}",
            flush=True,
        )


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def local_seed(model, prefix, case_id):
    text = f"{POOL_SEED_BASE}|{model}|{prefix}|{case_id}".encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:16], 16)


def pool_digest(sequences):
    payload = "\n".join(
        sorted(sequence_hash(seq) for seq in sequences)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prob_close(a, b):
    return math.isclose(
        float(a),
        float(b),
        rel_tol=PROB_TOL,
        abs_tol=PROB_TOL,
    )


def target_token(case_id, model, prefix):
    raw = f"{case_id}|{model}|{prefix}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def target_path(case_id, model, prefix):
    return TARGETS / (
        f"{model}_P{int(prefix)}_{target_token(case_id, model, prefix)}.json.gz"
    )


def write_gzip_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=5) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    os.replace(tmp, path)


def read_gzip_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def cache_complete(path, case_id, model, prefix, expected_digest):
    if not path.is_file():
        return False
    try:
        p = read_gzip_json(path)
    except Exception:
        return False
    return (
        p.get("schema_version") == CACHE_SCHEMA
        and p.get("case_id") == case_id
        and p.get("model") == model
        and int(p.get("prefix", -1)) == int(prefix)
        and p.get("pool_digest") == expected_digest
        and isinstance(p.get("pool_sequences"), list)
        and isinstance(p.get("pool_labels"), list)
        and isinstance(p.get("fit_indices"), list)
        and isinstance(p.get("eval_indices"), list)
    )


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    started = time.perf_counter()

    required = [
        PHASE4B_V2 / "validation_phase4b_v2.json",
        PHASE4B_V2 / "calibration_case_selection.json",
        PHASE4B_V2 / "neighborhood_manifest.csv",
        PHASE4B_V2 / "surrogate_metrics_calibration.csv",
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required inputs:\n" + "\n".join(missing)
        )

    validation = read_json(required[0])
    require(
        validation.get("status") == "passed",
        "Phase 4B-v2 validation is not PASSED",
    )
    require(
        validation.get("test_cases_used") is False,
        "Phase 4B-v2 unexpectedly used test data",
    )

    selection = read_json(required[1])
    case_ids = list(selection["case_ids"])
    require(
        len(case_ids) == CALIBRATION_CASE_COUNT,
        "Unexpected calibration case count",
    )

    manifest_rows = read_csv_rows(required[2])
    metric_rows = read_csv_rows(required[3])

    manifest = {
        (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
        ): r
        for r in manifest_rows
    }
    require(
        len(manifest) == CALIBRATION_CASE_COUNT * len(MODELS) * len(PREFIXES),
        "Unexpected Phase 4B-v2 manifest target count",
    )

    # All 3 surrogate metric rows share the same split inventory per target.
    metric_by_target = {}
    for r in metric_rows:
        key = (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
        )
        metric_by_target.setdefault(key, []).append(r)

    require(
        all(len(v) == 3 for v in metric_by_target.values()),
        "Expected exactly 3 surrogate metric rows per target",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    TARGETS.mkdir(parents=True, exist_ok=True)

    cases, _ = load_frozen_common_cohort(ROOT)
    case_map = {case.case_id: case for case in cases}
    require(
        all(cid in case_map for cid in case_ids),
        "A selected validation case is missing from frozen cohort",
    )
    require(
        all(case_map[cid].split == "val" for cid in case_ids),
        "Fusion cache selection contains a non-validation case",
    )

    canonical = read_json(required[4])
    constraints = load_process_constraints(RESULTS3)

    total = CALIBRATION_CASE_COUNT * len(MODELS) * len(PREFIXES)
    set_status(completed=0, total=total)

    hb = threading.Thread(
        target=heartbeat,
        args=(started,),
        daemon=True,
    )
    hb.start()

    completed = 0
    regenerated = 0
    cached = 0
    max_prob_diff = 0.0
    total_pool_rows = 0
    total_eval_rows = 0

    try:
        for model in MODELS:
            for prefix in PREFIXES:
                set_status(
                    target=f"{model}/P{prefix}",
                    stage="load canonical black box",
                )
                predictor = CanonicalPredictor(
                    ROOT,
                    model,
                    prefix,
                    canonical["items"][model][str(prefix)],
                )

                for case_id in case_ids:
                    completed += 1
                    key = (case_id, model, int(prefix))
                    rec = manifest[key]
                    path = target_path(case_id, model, prefix)
                    label = (
                        f"[{completed:04d}/{total}] "
                        f"{model} P{prefix} {case_id}"
                    )
                    set_status(
                        target=label,
                        stage="cache check",
                        completed=completed - 1,
                    )

                    expected_digest = rec["pool_digest"]
                    if cache_complete(
                        path,
                        case_id,
                        model,
                        prefix,
                        expected_digest,
                    ):
                        p = read_gzip_json(path)
                        total_pool_rows += int(p["pool_size"])
                        total_eval_rows += len(p["eval_indices"])
                        cached += 1
                        set_status(
                            stage="cached",
                            completed=completed,
                        )
                        if completed % 25 == 0:
                            elapsed = time.perf_counter() - started
                            print(
                                f"{label} | cached | "
                                f"elapsed={fmt_seconds(elapsed)}",
                                flush=True,
                            )
                        continue

                    case = case_map[case_id]
                    factual = tuple(case.activities[:prefix])
                    timestamps = tuple(case.timestamps[:prefix])
                    seed = local_seed(model, prefix, case_id)

                    t0 = time.perf_counter()
                    set_status(stage="generate_process_valid_pool")
                    pool = generate_process_valid_pool(
                        factual,
                        constraints,
                        rng_seed=seed,
                        pool_size=POOL_SIZE,
                        max_edits=MAX_EDITS,
                        max_options_per_position=MAX_OPTIONS,
                    )

                    digest = pool_digest(pool)
                    require(
                        digest == expected_digest,
                        f"Pool digest mismatch for {key}",
                    )
                    require(
                        len(pool) == int(rec["pool_size"]),
                        f"Pool size mismatch for {key}",
                    )

                    set_status(stage="black-box predict_proba")
                    probs = np.asarray(
                        predictor.predict_proba(
                            pool,
                            [timestamps] * len(pool),
                        ),
                        dtype=float,
                    )
                    labels = (probs >= 0.5).astype(int)

                    factual_idx = pool.index(factual)
                    factual_prob = float(probs[factual_idx])
                    diff = abs(
                        factual_prob
                        - float(rec["factual_probability"])
                    )
                    max_prob_diff = max(max_prob_diff, diff)

                    require(
                        prob_close(
                            factual_prob,
                            rec["factual_probability"],
                        ),
                        f"Factual probability mismatch for {key}",
                    )
                    require(
                        int(np.sum(labels == 0))
                        == int(rec["pool_class0"]),
                        f"Pool class0 mismatch for {key}",
                    )
                    require(
                        int(np.sum(labels == 1))
                        == int(rec["pool_class1"]),
                        f"Pool class1 mismatch for {key}",
                    )

                    factual_label = int(labels[factual_idx])
                    require(
                        factual_label
                        == int(rec["factual_prediction"]),
                        f"Factual prediction mismatch for {key}",
                    )

                    set_status(stage="rebuild stratified 70/30 split")
                    split = stratified_local_split(
                        pool,
                        labels,
                        factual,
                        eval_fraction=0.30,
                    )

                    fit_indices = [
                        int(i) for i in split["fit_indices"]
                    ]
                    eval_indices = [
                        int(i) for i in split["eval_indices"]
                    ]

                    rows = metric_by_target[key]
                    ref = rows[0]
                    require(
                        all(
                            int(r["fit_size"]) == len(fit_indices)
                            and int(r["eval_size"]) == len(eval_indices)
                            for r in rows
                        ),
                        f"Saved metric fit/eval size mismatch for {key}",
                    )

                    fit_labels = labels[fit_indices]
                    eval_labels = labels[eval_indices]
                    require(
                        int(np.sum(fit_labels == 0))
                        == int(ref["fit_class0"]),
                        f"fit class0 mismatch for {key}",
                    )
                    require(
                        int(np.sum(fit_labels == 1))
                        == int(ref["fit_class1"]),
                        f"fit class1 mismatch for {key}",
                    )
                    require(
                        int(np.sum(eval_labels == 0))
                        == int(ref["eval_class0"]),
                        f"eval class0 mismatch for {key}",
                    )
                    require(
                        int(np.sum(eval_labels == 1))
                        == int(ref["eval_class1"]),
                        f"eval class1 mismatch for {key}",
                    )

                    payload = {
                        "schema_version": CACHE_SCHEMA,
                        "case_id": case_id,
                        "model": model,
                        "prefix": int(prefix),
                        "pool_seed": int(seed),
                        "pool_digest": digest,
                        "pool_size": len(pool),
                        "pool_class0": int(np.sum(labels == 0)),
                        "pool_class1": int(np.sum(labels == 1)),
                        "factual_sequence": list(factual),
                        "factual_prediction": factual_label,
                        "factual_probability": factual_prob,
                        "pool_sequences": [
                            list(seq) for seq in pool
                        ],
                        "pool_labels": [
                            int(v) for v in labels.tolist()
                        ],
                        "fit_indices": fit_indices,
                        "eval_indices": eval_indices,
                        "test_cases_used": False,
                    }
                    write_gzip_json(path, payload)

                    regenerated += 1
                    total_pool_rows += len(pool)
                    total_eval_rows += len(eval_indices)
                    set_status(
                        stage="target PASS",
                        completed=completed,
                    )

                    if completed % 10 == 0 or completed == total:
                        elapsed = time.perf_counter() - started
                        avg = elapsed / completed
                        eta = avg * (total - completed)
                        print(
                            f"{label} | PASS | "
                            f"pool={len(pool)} eval={len(eval_indices)} | "
                            f"target={time.perf_counter()-t0:.1f}s | "
                            f"elapsed={fmt_seconds(elapsed)} | "
                            f"ETA={fmt_seconds(eta)}",
                            flush=True,
                        )

    finally:
        _STOP_HEARTBEAT.set()
        hb.join(timeout=1.0)

    # Final completeness pass.
    missing_cache = []
    for model in MODELS:
        for prefix in PREFIXES:
            for case_id in case_ids:
                rec = manifest[(case_id, model, int(prefix))]
                p = target_path(case_id, model, prefix)
                if not cache_complete(
                    p,
                    case_id,
                    model,
                    prefix,
                    rec["pool_digest"],
                ):
                    missing_cache.append(str(p))

    require(
        not missing_cache,
        "Incomplete fusion evaluation cache:\n"
        + "\n".join(missing_cache[:20]),
    )

    result = {
        "scope": "ETCE Phase 4C-v2 validation fusion evaluation cache",
        "status": "passed",
        "schema_version": CACHE_SCHEMA,
        "calibration_case_count": CALIBRATION_CASE_COUNT,
        "local_target_count": total,
        "models": list(MODELS),
        "prefixes": [int(v) for v in PREFIXES],
        "pool_size_requested": POOL_SIZE,
        "targets_regenerated_this_run": regenerated,
        "targets_reused_from_cache": cached,
        "total_pool_rows": total_pool_rows,
        "total_eval_rows": total_eval_rows,
        "probability_validation": {
            "tolerance": PROB_TOL,
            "max_factual_probability_abs_diff": max_prob_diff,
        },
        "checks": {
            "same_phase4b_v2_validation_cases": True,
            "all_pool_digests_match": True,
            "all_blackbox_labels_match_saved_inventory": True,
            "all_factual_probabilities_match": True,
            "all_stratified_fit_eval_splits_match": True,
            "formal_test_data_used": False,
            "lambda_selected": False,
            "ds_fusion_performed": False,
        },
        "test_cases_used": False,
    }

    write_json(
        OUT / "validation_fusion_eval_cache.json",
        result,
    )

    print()
    print("=" * 100)
    print("PHASE 4C-v2 FUSION EVALUATION CACHE PASSED")
    print("=" * 100)
    print(f"Targets: {total}")
    print(f"Pool rows cached: {total_pool_rows}")
    print(f"Held-out eval rows cached: {total_eval_rows}")
    print(
        "Max factual probability abs diff: "
        f"{max_prob_diff:.3e}"
    )
    print(f"Output: {OUT}")
    print("No formal test data read; no lambda selected; no D-S fusion performed.")
    print("NEXT: hierarchical D-S validation sweep using this cache.")


if __name__ == "__main__":
    main()
