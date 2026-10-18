#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage61
BPIC2018 External Frozen Neighborhood Protocol Recovery Gate

Why this stage exists
---------------------
Stage60 revealed that all 1600 canonical BPIC2018 explanation targets are
predicted as class 0. Therefore the external surrogate/counterfactual stage
must NOT enlarge the neighborhood, change class-balancing rules, change the
GA, or introduce new sampling just to obtain two-class local pools.

This gate recovers the exact neighborhood protocol already used in the
BPIC2017 main experiment from the local ETCE project. It does NOT generate
BPIC2018 neighborhoods yet.

Known frozen facts that must be preserved:
- process-valid local neighborhood
- max neighborhood size: 320 sequences
- max edits: 3
- local labels come from the black-box, not real outcome label
- two-class local pool required before stratified split
- black-box-label stratified 70/30 local fit/eval split
- search domain/statistics from TRAIN side only
- process-derived columns are not independently mutated
- only audited mutable fields may be changed
- SS / SD / DS / DD neighborhood quadrants
- LORE / DT / BN surrogate family unchanged
- TEST cannot retune distance / GA / sampling parameters

The script scans local source/config/result/script text files and creates a
ranked recovery report. It intentionally refuses to proceed to neighborhood
generation unless the main generator and a sufficiently strong frozen
protocol artifact are found.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

STAGE = 61

TEXT_EXTS = {
    ".py", ".json", ".yaml", ".yml", ".md", ".txt", ".csv",
    ".ini", ".cfg", ".toml", ".log"
}
MAX_FILE_BYTES = 8 * 1024 * 1024

# Strong evidence terms. These are not new parameters; they are search anchors
# for previously frozen artifacts.
TERMS = [
    "ga_neighborhood",
    "max_edits",
    "320",
    "70/30",
    "0.7",
    "0.3",
    "SS",
    "SD",
    "DS",
    "DD",
    "alpha_record",
    "alpha_prediction",
    "population",
    "generations",
    "mutation",
    "crossover",
    "hall_of_fame",
    "distance",
    "one-standard-error",
    "one standard error",
    "train-only",
    "process-valid",
    "two-class",
    "stratified",
    "LORE",
    "DT",
    "BN",
]

CONFIG_STRONG_TERMS = [
    "population",
    "generations",
    "mutation",
    "crossover",
    "distance",
]

EXPECTED_GENERATOR_SUFFIX = Path("src/etce/neighborhood/ga_neighborhood.py")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already contains files: {path}\n"
                "Use --overwrite only for an exact rerun."
            )
        import shutil
        for p in list(path.iterdir()):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    path.mkdir(parents=True, exist_ok=True)


def verify_stage60(stage60_dir: Path) -> Dict[str, Any]:
    required = (
        "protocol_frozen.json",
        "consistency_checks.json",
        "canonical_blackbox_bindings.csv",
        "canonical_checkpoint_map.csv",
        "canonical_binding_summary.json",
        "sha256_manifest.json",
    )
    for name in required:
        if not (stage60_dir / name).exists():
            raise FileNotFoundError(stage60_dir / name)

    protocol = read_json(stage60_dir / "protocol_frozen.json")
    checks = read_json(stage60_dir / "consistency_checks.json")
    summary = read_json(stage60_dir / "canonical_binding_summary.json")

    if protocol.get("stage") != 60:
        raise RuntimeError("Stage61 expects Stage60 input.")
    if not protocol.get("all_hard_checks_pass", False):
        raise RuntimeError("Stage60 protocol is not PASS.")
    if any(not v.get("pass", False) for v in checks.values()):
        raise RuntimeError("Stage60 consistency checks are not fully PASS.")
    if int(summary.get("formal_targets", -1)) != 1600:
        raise RuntimeError("Stage60 target count is not 1600.")

    return {"protocol": protocol, "checks": checks, "summary": summary}


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8-sig")
        except Exception:
            return ""
    except Exception:
        return ""


def normalized(s: str) -> str:
    return s.lower().replace("_", " ").replace("-", " ")


def find_term_hits(text: str) -> List[str]:
    lo = text.lower()
    hits = []
    for term in TERMS:
        if term.lower() in lo:
            hits.append(term)
    return hits


def extract_snippets(text: str, terms: List[str], radius: int = 2, max_snips: int = 12) -> List[str]:
    lines = text.splitlines()
    wanted = set()
    for i, line in enumerate(lines):
        lo = line.lower()
        if any(t.lower() in lo for t in terms):
            for j in range(max(0, i - radius), min(len(lines), i + radius + 1)):
                wanted.add(j)
    out = []
    block = []
    last = None
    for j in sorted(wanted):
        if last is None or j == last + 1:
            block.append(f"L{j+1}: {lines[j][:400]}")
        else:
            out.append("\n".join(block))
            block = [f"L{j+1}: {lines[j][:400]}"]
        last = j
    if block:
        out.append("\n".join(block))
    return out[:max_snips]


def score_candidate(path: Path, text: str, hits: List[str]) -> int:
    name = path.name.lower()
    full = str(path).lower()
    score = len(hits)

    if "frozen" in name or "freeze" in name:
        score += 8
    if "neighborhood" in name:
        score += 8
    if "phase4" in full:
        score += 4
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        score += 4
    if "config" in full:
        score += 4
    if "validation" in full or "calibration" in full:
        score += 2
    if "test" in name and "formal" in name:
        score -= 3

    # Strong GA config evidence.
    lo = text.lower()
    score += 2 * sum(t.lower() in lo for t in CONFIG_STRONG_TERMS)
    return score


def scan_project(root: Path) -> Tuple[List[Dict[str, Any]], List[Path]]:
    results = []
    generator_candidates = []

    skip_parts = {
        ".git", "__pycache__", ".venv", "venv", "env", "site-packages",
        "node_modules", "stage55_dataset", "stage56_features",
        "stage58_formal_external_test", "stage59_formal_targets",
        "stage60_canonical_blackbox", "stage61_neighborhood_protocol_recovery",
    }

    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_parts]
        b = Path(base)

        for fn in files:
            p = b / fn
            if p.suffix.lower() not in TEXT_EXTS:
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue

            # Always capture exact generator-name candidates.
            if p.name.lower() == "ga_neighborhood.py":
                generator_candidates.append(p)

            text = safe_read_text(p)
            if not text:
                continue
            hits = find_term_hits(text)

            # Keep only meaningful protocol-related candidates.
            if len(hits) < 2 and "ga_neighborhood" not in p.name.lower():
                continue

            results.append(
                {
                    "path": str(p),
                    "relative_path": str(p.relative_to(root)),
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                    "hits": hits,
                    "hit_count": len(hits),
                    "score": score_candidate(p, text, hits),
                    "snippets": extract_snippets(text, hits),
                }
            )

    results.sort(key=lambda x: (-x["score"], -x["hit_count"], x["relative_path"].lower()))
    return results, generator_candidates


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ETCE Stage61 - recover frozen BPIC2017 neighborhood protocol"
    )
    ap.add_argument("--project_root", type=Path, default=Path(r"D:\ETCE"))
    ap.add_argument("--stage60_dir", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.project_root.resolve()
    stage60_dir = (
        args.stage60_dir.resolve()
        if args.stage60_dir
        else root / "results" / "phase4" / "phase4e_bpic2018_external" / "stage60_canonical_blackbox"
    )
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else root / "results" / "phase4" / "phase4e_bpic2018_external" / "stage61_neighborhood_protocol_recovery"
    )

    ensure_clean_dir(out_dir, args.overwrite)

    print("=" * 122)
    print("ETCE STAGE61 - FROZEN NEIGHBORHOOD PROTOCOL RECOVERY GATE")
    print("=" * 122)
    print(f"Project root : {root}")
    print(f"Stage60      : {stage60_dir}")
    print(f"Output       : {out_dir}")
    print("BPIC2018 neighborhood generation : NOT STARTED")
    print("TEST-based retuning              : FORBIDDEN")
    print("=" * 122)

    s60 = verify_stage60(stage60_dir)
    print("[Stage60 gate] PASS")
    print(f"[Canonical target class counts] {s60['summary'].get('predicted_class_counts_on_frozen_targets', {})}")

    results, generators = scan_project(root)

    exact_generator = root / EXPECTED_GENERATOR_SUFFIX
    exact_exists = exact_generator.exists()

    # Candidate "frozen config" means strong enough evidence to inspect next.
    strong_configs = []
    for x in results:
        lo_hits = {h.lower() for h in x["hits"]}
        strong_count = sum(t.lower() in lo_hits for t in CONFIG_STRONG_TERMS)
        rel = x["relative_path"].lower()
        if (
            x["score"] >= 18
            and strong_count >= 3
            and (
                "config" in rel
                or Path(x["relative_path"]).suffix.lower() in {".json", ".yaml", ".yml"}
            )
        ):
            strong_configs.append(x)

    # Write full ranked report.
    write_json(
        out_dir / "protocol_recovery_report.json",
        {
            "stage": STAGE,
            "created_local": datetime.now().astimezone().isoformat(),
            "project_root": str(root),
            "stage60_summary": s60["summary"],
            "expected_generator": str(exact_generator),
            "expected_generator_exists": exact_exists,
            "all_generator_candidates": [str(p) for p in generators],
            "strong_config_candidates": [
                {
                    "path": x["path"],
                    "relative_path": x["relative_path"],
                    "score": x["score"],
                    "hits": x["hits"],
                    "sha256": x["sha256"],
                    "snippets": x["snippets"],
                }
                for x in strong_configs[:20]
            ],
            "top_ranked_candidates": results[:40],
        },
    )

    # CSV for easy inspection.
    with (out_dir / "protocol_recovery_candidates.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.writer(f)
        w.writerow(["rank", "score", "hit_count", "relative_path", "sha256", "hits"])
        for rank, x in enumerate(results[:100], start=1):
            w.writerow(
                [
                    rank,
                    x["score"],
                    x["hit_count"],
                    x["relative_path"],
                    x["sha256"],
                    " | ".join(x["hits"]),
                ]
            )

    # Extract exact generator evidence if present.
    if exact_exists:
        gtxt = safe_read_text(exact_generator)
        write_json(
            out_dir / "ga_neighborhood_evidence.json",
            {
                "path": str(exact_generator),
                "sha256": sha256_file(exact_generator),
                "hits": find_term_hits(gtxt),
                "snippets": extract_snippets(
                    gtxt,
                    find_term_hits(gtxt),
                    radius=3,
                    max_snips=30,
                ),
            },
        )

    # The gate is deliberately conservative: generator must exist and there
    # must be at least one strong frozen config/protocol artifact to inspect.
    recovery_ready = exact_exists and len(strong_configs) >= 1

    checks = {
        "stage60_pass": {"pass": True},
        "no_bpic2018_neighborhood_generated": {"pass": True},
        "no_test_based_retuning": {"pass": True},
        "expected_ga_generator_exists": {
            "pass": exact_exists,
            "actual": str(exact_generator) if exact_exists else None,
            "expected": str(exact_generator),
        },
        "strong_frozen_protocol_candidate_found": {
            "pass": len(strong_configs) >= 1,
            "actual_count": len(strong_configs),
            "top_candidates": [x["relative_path"] for x in strong_configs[:10]],
        },
    }
    write_json(out_dir / "consistency_checks.json", checks)

    protocol = {
        "stage": STAGE,
        "name": "BPIC2018 Frozen Neighborhood Protocol Recovery Gate",
        "dataset": "BPIC2018 external validation",
        "known_frozen_constraints": {
            "max_neighborhood_sequences": 320,
            "max_edits": 3,
            "local_label_source": "black-box",
            "requires_two_class_pool_before_split": True,
            "local_split": "stratified 70/30 by black-box local label",
            "search_domain": "training side only",
            "real_outcome_label_used_for_local_split": False,
            "test_retuning": False,
            "surrogates": ["LORE", "DT", "BN"],
            "quadrants": ["SS", "SD", "DS", "DD"],
        },
        "recovery_ready_for_manual_exact_config_confirmation": recovery_ready,
        "bpic2018_neighborhood_generated": False,
    }
    write_json(out_dir / "protocol_gate.json", protocol)

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
                str(p): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                for p in files
            },
        },
    )

    print("\n" + "=" * 122)
    print("ETCE STAGE61 RECOVERY SUMMARY")
    print("=" * 122)
    print(f"Expected generator exists : {'YES' if exact_exists else 'NO'}")
    print(f"Generator path            : {exact_generator}")
    print(f"Generator candidates      : {len(generators)}")
    print(f"Protocol candidates       : {len(results)}")
    print(f"Strong frozen candidates  : {len(strong_configs)}")

    if strong_configs:
        print("\nTop strong candidates:")
        for i, x in enumerate(strong_configs[:10], start=1):
            print(
                f"  {i:2d}. score={x['score']:2d} | "
                f"{x['relative_path']}"
            )

    print("\nBPIC2018 neighborhoods generated : NO")
    print(f"Recovery gate                  : {'PASS' if recovery_ready else 'NEEDS EXACT PROTOCOL ARTIFACT'}")
    print("=" * 122)

    if not recovery_ready:
        print(
            "\nSTAGE61 STOPPED SAFELY.\n"
            "No external neighborhood was generated because the exact frozen "
            "BPIC2017 neighborhood artifact was not sufficiently recovered.\n"
            "Use protocol_recovery_candidates.csv / protocol_recovery_report.json "
            "to identify the authoritative frozen config instead of inventing parameters."
        )
        return 3

    print(
        "\nSTAGE61 RECOVERY GATE PASSED.\n"
        "The exact top candidate path(s) above should be inspected/frozen next. "
        "Do not start BPIC2018 neighborhood generation until those recovered values "
        "are bound byte-for-byte into the external protocol."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
