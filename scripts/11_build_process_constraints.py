"""Build train-only process constraints for ETCE Phase 3A.

The DFG is built only from frozen common-cohort TRAIN cases and uses events strictly
before each case's label boundary. Therefore no validation/test trajectory and no
label-defining event is used to construct the process constraint.

Run:
    python scripts/11_build_process_constraints.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from etce.data.load_bpic2017 import read_cases, timestamp
from etce.phase2.data import DIRECT_PROXIES, EXPECTED_COUNTS, load_frozen_common_cohort
from etce.utils.io import sha256, write_csv, write_json

# Keep this explicit to avoid relying on a Phase 2 constant that does not exist there.
TERMINAL_OFFER_ACTIVITIES = frozenset({"O_Accepted", "O_Refused", "O_Cancelled"})
FORBIDDEN_MUTATION_TARGETS = set(DIRECT_PROXIES) | set(TERMINAL_OFFER_ACTIVITIES)

RESULTS3 = ROOT / "results/phase3"


def read_metadata(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return {row["case_id"]: row for row in csv.DictReader(stream)}


def main():
    # Reuse validated Phase 2 loader to establish the exact frozen train IDs.
    frozen_cases, audit = load_frozen_common_cohort(ROOT)
    train_frozen = [c for c in frozen_cases if c.split == "train"]
    if len(train_frozen) != EXPECTED_COUNTS["train"]:
        raise AssertionError("Frozen train count mismatch")

    train_ids = {c.case_id for c in train_frozen}
    position_counts = Counter()
    prefix_activity_counts = Counter()

    for case in train_frozen:
        for pos, activity in enumerate(case.activities[:20], 1):
            position_counts[(pos, activity)] += 1
            prefix_activity_counts[activity] += 1

    phase1_meta = read_metadata(
        ROOT / "results/phase1_2/audit/common_cohort_case_ids.csv"
    )

    edge_counts = Counter()
    edge_deltas = defaultdict(list)
    activity_counts = Counter()
    start_counts = Counter()
    end_counts = Counter()
    seen_train = set()

    raw = ROOT / "data/raw/BPI Challenge 2017.xes.gz"
    for case in read_cases(raw):
        if case.case_id not in train_ids:
            continue

        row = phase1_meta[case.case_id]
        label_index = int(row["label_event_index"])  # 1-based
        # Strictly before label boundary.
        events = case.events[: label_index - 1]
        if len(events) < 2:
            continue

        activities = [str(e["concept:name"]) for e in events]
        times = [timestamp(e["time:timestamp"]) for e in events]

        start_counts[activities[0]] += 1
        end_counts[activities[-1]] += 1
        activity_counts.update(activities)

        for a, b, ta, tb in zip(activities, activities[1:], times, times[1:]):
            delta = float((tb - ta).total_seconds())
            if delta < 0:
                raise ValueError(f"Timestamp reversal in train case {case.case_id}")
            edge_counts[(a, b)] += 1
            edge_deltas[(a, b)].append(delta)

        seen_train.add(case.case_id)

    if seen_train != train_ids:
        missing = sorted(train_ids - seen_train)[:10]
        raise AssertionError(f"Missing frozen train cases while building DFG: {missing}")

    constraints_dir = RESULTS3 / "constraints"
    constraints_dir.mkdir(parents=True, exist_ok=True)

    edge_rows = []
    for (source, target), count in sorted(edge_counts.items()):
        values = np.asarray(edge_deltas[(source, target)], dtype=float)
        edge_rows.append(
            {
                "source_activity": source,
                "target_activity": target,
                "count": int(count),
                "delta_mean_seconds": float(values.mean()),
                "delta_median_seconds": float(np.median(values)),
                "delta_p10_seconds": float(np.quantile(values, 0.10)),
                "delta_p90_seconds": float(np.quantile(values, 0.90)),
            }
        )
    write_csv(constraints_dir / "dfg_edges.csv", edge_rows)

    position_rows = [
        {"position": pos, "activity": activity, "count": count}
        for (pos, activity), count in sorted(position_counts.items())
    ]
    write_csv(constraints_dir / "activity_position_counts.csv", position_rows)

    mutable_activities = sorted(
        activity
        for activity in prefix_activity_counts
        if activity not in FORBIDDEN_MUTATION_TARGETS
    )

    meta = {
        "scope": "train_only_pre_label_boundary",
        "train_case_count": len(train_ids),
        "raw_sha256": audit["raw_sha256"],
        "process_constraint": "directly-follows graph",
        "edge_validity_rule": "every adjacent activity pair must occur at least once in train pre-label DFG",
        "position_support_rule": "a replacement activity must have been observed at the same prefix position in train",
        "timestamps_phase3a": "preserve factual timestamps exactly",
        "mutation_operator_phase3a": "activity substitution only",
        "immutable_existing_activities": sorted(TERMINAL_OFFER_ACTIVITIES | set(DIRECT_PROXIES)),
        "forbidden_mutation_targets": sorted(FORBIDDEN_MUTATION_TARGETS),
        "mutable_activities": mutable_activities,
        "train_dfg_edge_count": len(edge_counts),
        "train_activity_count": len(activity_counts),
        "dfg_edges_sha256": sha256(constraints_dir / "dfg_edges.csv"),
        "position_counts_sha256": sha256(constraints_dir / "activity_position_counts.csv"),
    }
    write_json(constraints_dir / "process_constraints.json", meta)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
