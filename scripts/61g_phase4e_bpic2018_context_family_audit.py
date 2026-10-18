#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage61G
BPIC2018 TRAIN-only context-family adapter audit.

Purpose
-------
The frozen BPIC2017 semantic rule uses activity-name families (A_ / O_ / W_).
BPIC2018 activity names do not carry those prefixes, so the original
family_preserved() rejects every cross-activity substitution.

This audit evaluates dataset-native family definitions using ONLY fields that
already exist in BPIC2018 TRAIN-side events (2015+2016):
  - subprocess
  - doctype
  - joint (subprocess, doctype)
  - overlap variants based on observed TRAIN-side contexts

No 2017 labels, predictions, probabilities, explanation outputs, or rule
quality are read. No neighborhood is generated. No protocol is tuned here.

The output is an audit only. Stage62 must bind one adapter explicitly after
reviewing this TRAIN-only structural audit.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

STAGE = "61G"
TRAIN_YEARS = {2015, 2016}

ACTIVITY_ALIASES = [
    "activity", "concept:name", "concept_name", "event_activity", "activity_name"
]
DOCTYPE_ALIASES = [
    "doctype", "document_type", "doc_type", "documenttype"
]
SUBPROCESS_ALIASES = [
    "subprocess", "sub_process", "process", "subprocess_name"
]
CASE_ALIASES = [
    "case_id", "case:concept:name", "case_concept_name", "case"
]
YEAR_ALIASES = [
    "year", "case_year"
]
POSITION_ALIASES = [
    "event_pos", "event_position", "event_index", "event_idx", "position", "idx"
]
TIMESTAMP_ALIASES = [
    "time:timestamp", "timestamp", "time_timestamp"
]


def pick_col(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    exact = {str(c).lower(): c for c in columns}
    for a in aliases:
        if a.lower() in exact:
            return exact[a.lower()]
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


def locate_prefix_events(root: Path) -> Path:
    phase4e = root / "results" / "phase4" / "phase4e_bpic2018_external"
    candidates = list(phase4e.rglob("prefix_events.csv.gz"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Deterministic preference: Stage55 path first, then shortest path.
        candidates.sort(
            key=lambda p: (
                0 if "stage55" in str(p).lower() else 1,
                len(str(p)),
                str(p).lower(),
            )
        )
        return candidates[0]

    # Fallback: inspect CSV.GZ headers for required context columns.
    possible = []
    for p in phase4e.rglob("*.csv.gz"):
        try:
            cols = pd.read_csv(p, nrows=0).columns.tolist()
        except Exception:
            continue
        if (
            pick_col(cols, ACTIVITY_ALIASES)
            and pick_col(cols, DOCTYPE_ALIASES)
            and pick_col(cols, SUBPROCESS_ALIASES)
            and pick_col(cols, CASE_ALIASES)
            and pick_col(cols, YEAR_ALIASES)
        ):
            possible.append(p)

    if not possible:
        raise FileNotFoundError(
            "Could not locate a BPIC2018 event CSV.GZ with "
            "case/year/activity/doctype/subprocess columns."
        )
    possible.sort(key=lambda p: (0 if "stage55" in str(p).lower() else 1, str(p)))
    return possible[0]


def deduplicate_events(
    df: pd.DataFrame,
    case_col: str,
    activity_col: str,
    doctype_col: str,
    subprocess_col: str,
    position_col: Optional[str],
    timestamp_col: Optional[str],
) -> Tuple[pd.DataFrame, List[str]]:
    if position_col:
        subset = [case_col, position_col]
    else:
        subset = [case_col]
        if timestamp_col:
            subset.append(timestamp_col)
        subset += [activity_col, doctype_col, subprocess_col]

    before = len(df)
    out = df.drop_duplicates(subset=subset, keep="first").copy()
    return out, subset


def dominant(counter: Counter) -> Tuple[str, int, float]:
    if not counter:
        return "<MISSING>", 0, 0.0
    # deterministic tie-break
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    key, count = items[0]
    total = sum(counter.values())
    return key, count, count / total if total else 0.0


def quantiles(vals: List[int]) -> Dict[str, float]:
    if not vals:
        return {"min": 0, "median": 0, "mean": 0.0, "max": 0}
    return {
        "min": int(min(vals)),
        "median": float(statistics.median(vals)),
        "mean": float(statistics.mean(vals)),
        "max": int(max(vals)),
    }


def audit_equivalence_scheme(
    activities: List[str],
    family_map: Dict[str, str],
) -> Dict[str, Any]:
    groups = defaultdict(list)
    for a in activities:
        groups[family_map[a]].append(a)

    options = {
        a: sorted([b for b in groups[family_map[a]] if b != a])
        for a in activities
    }
    counts = [len(options[a]) for a in activities]
    preserved_pairs = sum(counts)
    return {
        "family_count": len(groups),
        "singleton_family_count": sum(len(v) == 1 for v in groups.values()),
        "family_sizes": {
            k: len(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))
        },
        "activities_with_zero_options": [a for a in activities if not options[a]],
        "option_count_stats": quantiles(counts),
        "family_preserved_ordered_pairs": preserved_pairs,
        "options_by_activity": options,
        "family_by_activity": family_map,
    }


def audit_overlap_scheme(
    activities: List[str],
    context_sets: Dict[str, set],
) -> Dict[str, Any]:
    options = {}
    for a in activities:
        opts = []
        for b in activities:
            if a == b:
                continue
            if context_sets[a] & context_sets[b]:
                opts.append(b)
        options[a] = sorted(opts)

    counts = [len(options[a]) for a in activities]
    return {
        "activities_with_zero_options": [a for a in activities if not options[a]],
        "option_count_stats": quantiles(counts),
        "family_preserved_ordered_pairs": sum(counts),
        "options_by_activity": options,
        "context_cardinality_by_activity": {
            a: len(context_sets[a]) for a in activities
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ETCE Stage61G - BPIC2018 TRAIN-only context-family audit"
    )
    ap.add_argument("--project_root", type=Path, default=Path(r"D:\ETCE"))
    ap.add_argument("--events_csv", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.project_root.resolve()
    events_path = args.events_csv.resolve() if args.events_csv else locate_prefix_events(root)
    out = (
        args.out_dir.resolve()
        if args.out_dir
        else root
        / "results"
        / "phase4"
        / "phase4e_bpic2018_external"
        / "stage61g_context_family_audit"
    )

    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} exists. Use --overwrite.")
    if out.exists() and args.overwrite:
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 122)
    print("ETCE STAGE61G - BPIC2018 TRAIN-ONLY CONTEXT-FAMILY ADAPTER AUDIT")
    print("=" * 122)
    print(f"Events source : {events_path}")
    print(f"TRAIN years   : {sorted(TRAIN_YEARS)}")
    print("Uses 2017 labels/predictions/explanations : NO")
    print("Generates BPIC2018 neighborhoods          : NO")
    print("=" * 122)

    df = pd.read_csv(events_path)
    cols = df.columns.tolist()

    case_col = pick_col(cols, CASE_ALIASES)
    year_col = pick_col(cols, YEAR_ALIASES)
    activity_col = pick_col(cols, ACTIVITY_ALIASES)
    doctype_col = pick_col(cols, DOCTYPE_ALIASES)
    subprocess_col = pick_col(cols, SUBPROCESS_ALIASES)
    position_col = pick_col(cols, POSITION_ALIASES)
    timestamp_col = pick_col(cols, TIMESTAMP_ALIASES)

    required = {
        "case": case_col,
        "year": year_col,
        "activity": activity_col,
        "doctype": doctype_col,
        "subprocess": subprocess_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise RuntimeError(
            f"Required event fields missing: {missing}. Columns={cols}"
        )

    # Filter TRAIN-side only before any family analysis.
    years_numeric = pd.to_numeric(df[year_col], errors="coerce")
    train = df[years_numeric.isin(TRAIN_YEARS)].copy()

    # Normalize context columns.
    for c in [activity_col, doctype_col, subprocess_col]:
        train[c] = train[c].map(norm)

    train_dedup, dedup_subset = deduplicate_events(
        train,
        case_col=case_col,
        activity_col=activity_col,
        doctype_col=doctype_col,
        subprocess_col=subprocess_col,
        position_col=position_col,
        timestamp_col=timestamp_col,
    )

    activities = sorted(
        a for a in train_dedup[activity_col].unique().tolist()
        if a != "<MISSING>"
    )

    # Per-activity context counts.
    sub_counts = defaultdict(Counter)
    doc_counts = defaultdict(Counter)
    joint_counts = defaultdict(Counter)

    for _, r in train_dedup.iterrows():
        a = r[activity_col]
        if a == "<MISSING>":
            continue
        sp = r[subprocess_col]
        dt = r[doctype_col]
        sub_counts[a][sp] += 1
        doc_counts[a][dt] += 1
        joint_counts[a][f"{sp} || {dt}"] += 1

    dominant_sub = {}
    dominant_doc = {}
    dominant_joint = {}
    purity_rows = []

    sub_sets = {}
    doc_sets = {}
    joint_sets = {}

    for a in activities:
        ds, dsc, dsp = dominant(sub_counts[a])
        dd, ddc, ddp = dominant(doc_counts[a])
        dj, djc, djp = dominant(joint_counts[a])

        dominant_sub[a] = ds
        dominant_doc[a] = dd
        dominant_joint[a] = dj

        sub_sets[a] = set(sub_counts[a].keys())
        doc_sets[a] = set(doc_counts[a].keys())
        joint_sets[a] = set(joint_counts[a].keys())

        purity_rows.append(
            {
                "activity": a,
                "event_count": int(sum(sub_counts[a].values())),
                "dominant_subprocess": ds,
                "subprocess_purity": dsp,
                "subprocess_cardinality": len(sub_sets[a]),
                "dominant_doctype": dd,
                "doctype_purity": ddp,
                "doctype_cardinality": len(doc_sets[a]),
                "dominant_joint": dj,
                "joint_purity": djp,
                "joint_cardinality": len(joint_sets[a]),
            }
        )

    schemes = {
        "dominant_subprocess": audit_equivalence_scheme(activities, dominant_sub),
        "dominant_doctype": audit_equivalence_scheme(activities, dominant_doc),
        "dominant_joint": audit_equivalence_scheme(activities, dominant_joint),
        "overlap_subprocess": audit_overlap_scheme(activities, sub_sets),
        "overlap_doctype": audit_overlap_scheme(activities, doc_sets),
        "overlap_joint": audit_overlap_scheme(activities, joint_sets),
    }

    # Save detailed activity contexts.
    with (out / "activity_context_profile.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = list(purity_rows[0].keys()) if purity_rows else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(purity_rows)

    # Save option counts by scheme/activity.
    option_rows = []
    for scheme_name, s in schemes.items():
        for a in activities:
            option_rows.append(
                {
                    "scheme": scheme_name,
                    "activity": a,
                    "option_count": len(s["options_by_activity"][a]),
                    "options": " | ".join(s["options_by_activity"][a]),
                }
            )

    with (out / "family_scheme_options.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=["scheme", "activity", "option_count", "options"],
        )
        w.writeheader()
        w.writerows(option_rows)

    report = {
        "stage": STAGE,
        "source": str(events_path),
        "source_columns": cols,
        "resolved_columns": required,
        "position_column": position_col,
        "timestamp_column": timestamp_col,
        "train_years": sorted(TRAIN_YEARS),
        "raw_train_rows": len(train),
        "deduplicated_train_rows": len(train_dedup),
        "dedup_subset": dedup_subset,
        "activity_count": len(activities),
        "activities": activities,
        "schemes": schemes,
        "uses_2017_labels": False,
        "uses_2017_predictions": False,
        "uses_2017_explanations": False,
        "neighborhood_generated": False,
        "protocol_frozen_here": False,
    }

    (out / "context_family_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nTRAIN rows before dedup : {len(train):,}")
    print(f"TRAIN rows after dedup  : {len(train_dedup):,}")
    print(f"Distinct activities     : {len(activities)}")
    print(f"Dedup key               : {dedup_subset}")

    print("\n" + "=" * 122)
    print("CONTEXT-FAMILY SCHEME AUDIT")
    print("=" * 122)
    for name, s in schemes.items():
        z = len(s["activities_with_zero_options"])
        q = s["option_count_stats"]
        print(
            f"{name:22s} | "
            f"zero={z:2d}/{len(activities):2d} | "
            f"pairs={s['family_preserved_ordered_pairs']:4d} | "
            f"options min/med/mean/max="
            f"{q['min']}/{q['median']:.1f}/{q['mean']:.2f}/{q['max']}"
        )

    print("\nPer-activity dominant context purity:")
    for r in purity_rows:
        print(
            f"  {r['activity']!r:28s} | "
            f"sub={r['dominant_subprocess']!r} ({r['subprocess_purity']:.3f},k={r['subprocess_cardinality']}) | "
            f"doc={r['dominant_doctype']!r} ({r['doctype_purity']:.3f},k={r['doctype_cardinality']}) | "
            f"joint={r['dominant_joint']!r} ({r['joint_purity']:.3f},k={r['joint_cardinality']})"
        )

    print("\n2017 labels/predictions/explanations used : NO")
    print("Neighborhood generated                  : NO")
    print("Protocol frozen                         : NO")
    print("=" * 122)

    print("\nArtifacts:")
    print(out / "activity_context_profile.csv")
    print(out / "family_scheme_options.csv")
    print(out / "context_family_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
