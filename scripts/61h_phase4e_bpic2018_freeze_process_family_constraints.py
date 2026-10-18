#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage61H
Freeze BPIC2018 TRAIN-only process constraints and semantic-family adapter.

Frozen adapter
--------------
family(activity) = dominant subprocess of that activity in 2015+2016 TRAIN
                  prefix events, using deterministic count + lexical tie-break.

Why this adapter
----------------
- BPIC2017 A_/O_/W_ name-prefix family logic is not applicable to BPIC2018.
- The adapter uses an existing BPIC2018 process-context field (subprocess).
- It is fit on TRAIN years 2015+2016 only.
- It does not use 2017 labels, predictions, probabilities, or explanations.
- It is more conservative than overlap-based context rules.
- It preserves the Phase4 principle "replacement stays in the same semantic
  process family" without changing pool size, max edits, local split, or
  surrogate definitions.

Outputs
-------
results/phase4/phase4e_bpic2018_external/stage61h_frozen_constraints/
  process_constraints.json
  dfg_edges.csv
  activity_position_counts.csv
  activity_family_map.csv
  family_options.csv
  consistency_checks.json
  protocol_frozen.json
  sha256_manifest.json
  STAGE61H_FROZEN_CONSTRAINTS.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

STAGE = "61H"
TRAIN_YEARS = (2015, 2016)
PREFIX_MAX = 20
FAMILY_SCHEME = "dominant_subprocess"

# These are protocol-semantic exclusions, not data-driven performance choices.
# Only activities actually present in the prefix vocabulary become immutable.
PROTECTED_ACTIVITY_NAMES = {
    "abort external",   # directly participates in the frozen external outcome definition
    "begin payment",    # directly participates in the frozen external outcome definition
    "decide",           # frozen decision-boundary activity
}

ACTIVITY_ALIASES = ["activity", "concept:name", "concept_name", "activity_name"]
DOCTYPE_ALIASES = ["doctype", "document_type", "doc_type", "documenttype"]
SUBPROCESS_ALIASES = ["subprocess", "sub_process", "process", "subprocess_name"]
CASE_ALIASES = ["case_id", "case:concept:name", "case_concept_name", "case"]
YEAR_ALIASES = ["year", "case_year"]
TIMESTAMP_ALIASES = ["time:timestamp", "timestamp", "time_timestamp"]


def pick_col(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    lookup = {str(c).lower(): c for c in columns}
    for a in aliases:
        if a.lower() in lookup:
            return lookup[a.lower()]
    return None


def norm(v: Any) -> str:
    if v is None:
        return "<MISSING>"
    try:
        if pd.isna(v):
            return "<MISSING>"
    except Exception:
        pass
    s = str(v).strip()
    return s if s else "<MISSING>"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite for exact rebuild.")
        import shutil
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def locate_events(root: Path) -> Path:
    preferred = (
        root
        / "results"
        / "phase4"
        / "phase4e_bpic2018_external"
        / "stage55_dataset"
        / "prefix_events.csv.gz"
    )
    if preferred.exists():
        return preferred
    candidates = list(
        (root / "results" / "phase4" / "phase4e_bpic2018_external").rglob(
            "prefix_events.csv.gz"
        )
    )
    if not candidates:
        raise FileNotFoundError("prefix_events.csv.gz not found")
    candidates.sort(key=lambda p: str(p).lower())
    return candidates[0]


def dominant(counter: Counter) -> Tuple[str, int, float]:
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    if not items:
        return "<MISSING>", 0, 0.0
    k, c = items[0]
    total = sum(counter.values())
    return k, int(c), float(c / total) if total else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=Path, default=Path(r"D:\ETCE"))
    ap.add_argument("--events_csv", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.project_root.resolve()
    events_path = args.events_csv.resolve() if args.events_csv else locate_events(root)
    out = (
        args.out_dir.resolve()
        if args.out_dir
        else root
        / "results"
        / "phase4"
        / "phase4e_bpic2018_external"
        / "stage61h_frozen_constraints"
    )
    ensure_clean_dir(out, args.overwrite)

    print("=" * 122)
    print("ETCE STAGE61H - FREEZE BPIC2018 TRAIN-ONLY PROCESS / FAMILY CONSTRAINTS")
    print("=" * 122)
    print(f"Source events : {events_path}")
    print(f"TRAIN years   : {list(TRAIN_YEARS)}")
    print(f"Family scheme : {FAMILY_SCHEME}")
    print("2017 labels/predictions/explanations used : NO")
    print("BPIC2018 neighborhoods generated          : NO")
    print("=" * 122)

    df = pd.read_csv(events_path)
    cols = list(df.columns)

    case_col = pick_col(cols, CASE_ALIASES)
    year_col = pick_col(cols, YEAR_ALIASES)
    activity_col = pick_col(cols, ACTIVITY_ALIASES)
    doctype_col = pick_col(cols, DOCTYPE_ALIASES)
    subprocess_col = pick_col(cols, SUBPROCESS_ALIASES)
    ts_col = pick_col(cols, TIMESTAMP_ALIASES)

    required = {
        "case": case_col,
        "year": year_col,
        "activity": activity_col,
        "doctype": doctype_col,
        "subprocess": subprocess_col,
        "timestamp": ts_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise RuntimeError(f"Required columns missing: {missing}; columns={cols}")

    df["_row_order"] = range(len(df))
    years = pd.to_numeric(df[year_col], errors="coerce")
    train = df[years.isin(TRAIN_YEARS)].copy()

    for c in (activity_col, doctype_col, subprocess_col):
        train[c] = train[c].map(norm)

    # Prefix-events are repeated across P5/P10/P15/P20 exports. Recover the
    # unique visible event stream using the exact same identity used in Stage61G.
    dedup_cols = [case_col, ts_col, activity_col, doctype_col, subprocess_col]
    train = train.drop_duplicates(subset=dedup_cols, keep="first").copy()

    # Stable event order: timestamp first, original file order for ties.
    train["_ts"] = pd.to_datetime(train[ts_col], errors="coerce", utc=True)
    if train["_ts"].isna().any():
        raise RuntimeError(
            f"Unparseable timestamps in TRAIN prefix events: {int(train['_ts'].isna().sum())}"
        )
    train = train.sort_values(
        [case_col, "_ts", "_row_order"], kind="mergesort"
    ).copy()
    train["_position"] = train.groupby(case_col, sort=False).cumcount() + 1

    max_pos = int(train["_position"].max()) if len(train) else 0
    if max_pos > PREFIX_MAX:
        raise RuntimeError(
            f"Recovered visible prefix position exceeds P{PREFIX_MAX}: max={max_pos}. "
            "Do not freeze constraints until prefix-event duplication/order is resolved."
        )

    activities = sorted(
        a for a in train[activity_col].unique().tolist() if a != "<MISSING>"
    )

    # ------------------------------------------------------------------
    # Family map = dominant TRAIN subprocess per activity.
    # ------------------------------------------------------------------
    sub_counts = defaultdict(Counter)
    for _, r in train.iterrows():
        a = r[activity_col]
        if a != "<MISSING>":
            sub_counts[a][r[subprocess_col]] += 1

    family_map = {}
    family_purity = {}
    family_support = {}
    for a in activities:
        fam, count, purity = dominant(sub_counts[a])
        family_map[a] = fam
        family_purity[a] = purity
        family_support[a] = count

    # ------------------------------------------------------------------
    # Train-only DFG and same-position support.
    # ------------------------------------------------------------------
    edge_count = Counter()
    position_count = Counter()

    for _, g in train.groupby(case_col, sort=False):
        seq = g[activity_col].tolist()
        pos = g["_position"].astype(int).tolist()

        for p, a in zip(pos, seq):
            position_count[(int(p), a)] += 1
        for a, b in zip(seq, seq[1:]):
            edge_count[(a, b)] += 1

    # Frozen mutation domain.
    immutable_existing = sorted(
        a for a in activities if a.lower() in PROTECTED_ACTIVITY_NAMES
    )
    forbidden_targets = sorted(PROTECTED_ACTIVITY_NAMES)
    mutable_activities = sorted(
        a for a in activities if a not in set(immutable_existing)
    )

    # Family option gate before DFG/position filtering.
    family_groups = defaultdict(list)
    for a in mutable_activities:
        family_groups[family_map[a]].append(a)

    family_options = {}
    for a in activities:
        if a in immutable_existing:
            family_options[a] = []
        else:
            family_options[a] = sorted(
                b
                for b in family_groups[family_map[a]]
                if b != a and b not in PROTECTED_ACTIVITY_NAMES
            )

    zero_mutable_family = [
        a for a in mutable_activities if len(family_options[a]) == 0
    ]

    # Stronger feasibility gate: there must be at least one observed
    # same-position, same-family alternative somewhere for every mutable activity.
    same_position_same_family_options = {}
    for a in mutable_activities:
        positions_a = {
            p for (p, act), c in position_count.items()
            if act == a and c > 0
        }
        opts = []
        for b in family_options[a]:
            positions_b = {
                p for (p, act), c in position_count.items()
                if act == b and c > 0
            }
            if positions_a & positions_b:
                opts.append(b)
        same_position_same_family_options[a] = sorted(opts)

    zero_same_position = [
        a for a in mutable_activities
        if len(same_position_same_family_options[a]) == 0
    ]

    # ------------------------------------------------------------------
    # Write authoritative CSV artifacts.
    # ------------------------------------------------------------------
    dfg_path = out / "dfg_edges.csv"
    with dfg_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["source_activity", "target_activity", "count"])
        for (a, b), c in sorted(edge_count.items()):
            w.writerow([a, b, int(c)])

    pos_path = out / "activity_position_counts.csv"
    with pos_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["position", "activity", "count"])
        for (p, a), c in sorted(position_count.items()):
            w.writerow([int(p), a, int(c)])

    fam_path = out / "activity_family_map.csv"
    with fam_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "activity",
                "family",
                "dominant_subprocess_support",
                "dominant_subprocess_purity",
                "immutable_existing",
            ]
        )
        for a in activities:
            w.writerow(
                [
                    a,
                    family_map[a],
                    family_support[a],
                    f"{family_purity[a]:.12g}",
                    int(a in immutable_existing),
                ]
            )

    opt_path = out / "family_options.csv"
    with opt_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "activity",
                "family",
                "family_option_count",
                "family_options",
                "same_position_family_option_count",
                "same_position_family_options",
            ]
        )
        for a in activities:
            sp = same_position_same_family_options.get(a, [])
            w.writerow(
                [
                    a,
                    family_map[a],
                    len(family_options[a]),
                    " | ".join(family_options[a]),
                    len(sp),
                    " | ".join(sp),
                ]
            )

    constraints = {
        "dataset": "BPIC2018",
        "external_protocol_stage": STAGE,
        "scope": "TRAIN_2015_2016_visible_prefix_events_up_to_P20",
        "train_years": list(TRAIN_YEARS),
        "train_case_count": int(train[case_col].nunique()),
        "train_event_count_deduplicated": int(len(train)),
        "train_activity_count": int(len(activities)),
        "train_dfg_edge_count": int(len(edge_count)),
        "max_visible_position": max_pos,
        "process_constraint": "directly-follows graph",
        "edge_validity_rule":
            "every adjacent activity pair must occur at least once in TRAIN 2015+2016 visible-prefix DFG",
        "position_support_rule":
            "a replacement activity must have been observed at the same prefix position in TRAIN 2015+2016",
        "mutation_operator":
            "fixed-length activity substitution only",
        "timestamps":
            "preserve factual timestamps exactly",
        "family_adapter": {
            "scheme": FAMILY_SCHEME,
            "definition":
                "family(activity) = dominant subprocess observed for that activity in TRAIN 2015+2016",
            "tie_break":
                "highest count, then lexical subprocess name",
            "source_fields": [activity_col, subprocess_col],
            "fit_uses_2017": False,
            "fit_uses_labels": False,
            "fit_uses_blackbox_outputs": False,
            "fit_uses_explanations": False,
        },
        "protected_activity_names_protocol_semantic": sorted(PROTECTED_ACTIVITY_NAMES),
        "immutable_existing_activities": immutable_existing,
        "forbidden_mutation_targets": forbidden_targets,
        "mutable_activities": mutable_activities,
        "dfg_edges_sha256": sha256_file(dfg_path),
        "position_counts_sha256": sha256_file(pos_path),
        "activity_family_map_sha256": sha256_file(fam_path),
        "raw_prefix_events_sha256": sha256_file(events_path),
    }
    write_json(out / "process_constraints.json", constraints)

    checks = {
        "train_years_only": {
            "pass": True,
            "actual": list(TRAIN_YEARS),
        },
        "max_visible_position_le_20": {
            "pass": max_pos <= PREFIX_MAX,
            "actual": max_pos,
            "expected_max": PREFIX_MAX,
        },
        "activity_count_23": {
            "pass": len(activities) == 23,
            "actual": len(activities),
            "expected": 23,
        },
        "family_scheme_dominant_subprocess": {
            "pass": FAMILY_SCHEME == "dominant_subprocess",
        },
        "all_mutable_have_family_alternative": {
            "pass": len(zero_mutable_family) == 0,
            "zero_option_activities": zero_mutable_family,
        },
        "all_mutable_have_same_position_family_alternative": {
            "pass": len(zero_same_position) == 0,
            "zero_option_activities": zero_same_position,
        },
        "dfg_nonempty": {
            "pass": len(edge_count) > 0,
            "actual": len(edge_count),
        },
        "position_support_nonempty": {
            "pass": len(position_count) > 0,
            "actual": len(position_count),
        },
        "uses_2017_labels": {"pass": True, "actual": False},
        "uses_2017_predictions": {"pass": True, "actual": False},
        "uses_2017_explanations": {"pass": True, "actual": False},
        "neighborhood_generated": {"pass": True, "actual": False},
    }
    all_pass = all(v.get("pass", False) for v in checks.values())
    write_json(out / "consistency_checks.json", checks)

    protocol = {
        "stage": STAGE,
        "name": "BPIC2018 TRAIN-only process constraint and family adapter freeze",
        "created_local": datetime.now().astimezone().isoformat(),
        "family_scheme": FAMILY_SCHEME,
        "family_selection_basis":
            "TRAIN-only structural semantics: zero uncovered mutable activities; "
            "subprocess represents process role; more conservative than overlap schemes",
        "selection_uses_2017": False,
        "selection_uses_test_performance": False,
        "selection_uses_explanation_quality": False,
        "pool_size_changed": False,
        "max_edits_changed": False,
        "surrogate_protocol_changed": False,
        "all_hard_checks_pass": all_pass,
    }
    write_json(out / "protocol_frozen.json", protocol)

    md = f"""# ETCE Stage61H — BPIC2018 Frozen Process Constraints

Family adapter: **dominant subprocess**, fit on **2015+2016 TRAIN only**.

- Cases: {train[case_col].nunique():,}
- Deduplicated visible prefix events: {len(train):,}
- Activities: {len(activities)}
- DFG edges: {len(edge_count)}
- Maximum visible prefix position: {max_pos}
- Immutable present activities: {immutable_existing}
- Mutable activities with no family alternative: {zero_mutable_family}
- Mutable activities with no same-position family alternative: {zero_same_position}

No 2017 label, prediction, probability, explanation, or neighborhood output was
used to select or fit this adapter.

Gate: **{"PASS" if all_pass else "FAIL"}**
"""
    (out / "STAGE61H_FROZEN_CONSTRAINTS.md").write_text(md, encoding="utf-8")

    files = sorted(
        p for p in out.iterdir()
        if p.is_file() and p.name != "sha256_manifest.json"
    )
    write_json(
        out / "sha256_manifest.json",
        {
            "stage": STAGE,
            "algorithm": "SHA256",
            "files": {
                p.name: {
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
                for p in files
            },
        },
    )

    print("\n" + "=" * 122)
    print("ETCE STAGE61H FREEZE SUMMARY")
    print("=" * 122)
    print(f"TRAIN cases                         : {train[case_col].nunique():,}")
    print(f"Deduplicated prefix events          : {len(train):,}")
    print(f"Activities                          : {len(activities)}")
    print(f"DFG edges                           : {len(edge_count)}")
    print(f"Maximum visible position            : {max_pos}")
    print(f"Immutable existing activities       : {immutable_existing}")
    print(f"Mutable activities                  : {len(mutable_activities)}")
    print(f"Zero family alternatives            : {zero_mutable_family}")
    print(f"Zero same-position family alt.      : {zero_same_position}")
    print("2017 information used               : NO")
    print("Neighborhood generated              : NO")
    print(f"Hard gates                          : {'PASS' if all_pass else 'FAIL'}")
    print("=" * 122)

    if not all_pass:
        print("\nSTAGE61H FAILED. Do not proceed to Stage62.")
        return 2

    print(
        "\nSTAGE61H BPIC2018 PROCESS/FAMILY CONSTRAINTS FROZEN SUCCESSFULLY.\n"
        "NEXT: Stage62 may use these exact frozen constraints to construct the "
        "external process-valid local pools and multi-rule evidence bank."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
