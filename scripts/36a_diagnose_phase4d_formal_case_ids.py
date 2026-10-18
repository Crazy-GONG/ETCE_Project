#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Phase 4D-3A
Diagnose formal TEST case_id identity mismatch.

READ-ONLY diagnostic:
- does not modify any file
- does not run D-S fusion
- does not retune lambda/BPA/grouping
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("results/phase4/phase4d_formal_test")
RULES = ROOT / "formal_test_multirule_bank" / "canonical_rules_test.jsonl"
SELECTION = ROOT / "formal_test_case_selection.csv"


def raw_key(v):
    return (type(v).__name__, repr(v))


def norm(v):
    if v is None:
        return "<NULL>"
    return str(v).strip()


def main():
    if not RULES.exists():
        raise FileNotFoundError(RULES)
    if not SELECTION.exists():
        raise FileNotFoundError(SELECTION)

    rows = []
    with RULES.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_line"] = i
            rows.append(obj)

    # ---- selection ----
    with SELECTION.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "case_id" not in (reader.fieldnames or []):
            raise RuntimeError(f"'case_id' not found in {SELECTION}; columns={reader.fieldnames}")
        sel_ids = [str(r["case_id"]).strip() for r in reader]

    sel_set = set(sel_ids)

    # ---- rule bank case ids ----
    raw_ids = [r.get("case_id") for r in rows]
    raw_typed = {raw_key(v) for v in raw_ids}
    norm_ids = [norm(v) for v in raw_ids]
    norm_set = set(norm_ids)

    print("=" * 112)
    print("ETCE PHASE 4D-3A FORMAL TEST CASE_ID DIAGNOSTIC")
    print("=" * 112)
    print(f"Rule-bank rows                         : {len(rows)}")
    print(f"Selection rows                         : {len(sel_ids)}")
    print(f"Selection unique case_id               : {len(sel_set)}")
    print(f"Rule bank raw typed unique case_id     : {len(raw_typed)}")
    print(f"Rule bank str-normalized unique case_id: {len(norm_set)}")
    print()

    # Null / blank IDs
    bad = [(r["_line"], r.get("case_id")) for r in rows
           if r.get("case_id") is None or norm(r.get("case_id")) in {"", "<NULL>", "nan", "NaN"}]
    print(f"Null/blank-like case_id rows            : {len(bad)}")
    if bad:
        print("  First 20:", bad[:20])
    print()

    # selection vs bank
    missing = sorted(sel_set - norm_set)
    extra = sorted(norm_set - sel_set)

    print("[Selection vs canonical rule bank]")
    print(f"Missing selected case IDs in bank       : {len(missing)}")
    for x in missing[:20]:
        print(f"  MISSING: {x!r}")
    print(f"Extra bank case IDs not in selection    : {len(extra)}")
    for x in extra[:20]:
        print(f"  EXTRA  : {x!r}")
    print()

    # Detect typed IDs collapsing under str()
    collision_map = defaultdict(set)
    for v in raw_ids:
        collision_map[norm(v)].add(raw_key(v))

    collisions = {
        k: sorted(v)
        for k, v in collision_map.items()
        if len(v) > 1
    }

    print("[Raw-type -> str() collision check]")
    print(f"Normalized IDs backed by >1 raw typed ID: {len(collisions)}")
    for k, vals in list(sorted(collisions.items()))[:20]:
        print(f"  {k!r} <- {vals}")
    print()

    # Per-case structural counts.
    targets = defaultdict(set)
    ts_groups = defaultdict(set)
    row_counts = Counter()
    cf_counts = Counter()
    eligible_cf_counts = Counter()

    for r in rows:
        cid = norm(r.get("case_id"))
        model = str(r.get("model"))
        prefix = str(r.get("prefix"))
        surrogate = str(r.get("surrogate"))
        targets[cid].add((model, prefix))
        ts_groups[cid].add((model, prefix, surrogate))
        row_counts[cid] += 1
        if r.get("rule_type") == "counterfactual":
            cf_counts[cid] += 1
            if r.get("eligible_for_reliability") is True:
                eligible_cf_counts[cid] += 1

    print("[Per-case structural anomalies]")
    anomaly_ids = []
    for cid in sorted(norm_set):
        nt = len(targets[cid])
        ng = len(ts_groups[cid])
        # Each formal case should contribute 4 models x 4 prefixes = 16 targets
        # and 16 x 3 surrogates = 48 target-surrogate groups.
        if nt != 16 or ng != 48:
            anomaly_ids.append(cid)
            print(
                f"  {cid!r}: targets={nt}, target-surrogate={ng}, "
                f"rows={row_counts[cid]}, CF={cf_counts[cid]}, "
                f"eligible_CF={eligible_cf_counts[cid]}"
            )

    if not anomaly_ids:
        print("  No per-present-case structural anomaly: every present bank case has 16 targets and 48 target-surrogate groups.")
    print()

    all_targets = {
        (norm(r.get("case_id")), str(r.get("model")), str(r.get("prefix")))
        for r in rows
    }
    all_ts = {
        (norm(r.get("case_id")), str(r.get("model")), str(r.get("prefix")), str(r.get("surrogate")))
        for r in rows
    }

    print("[Bank-wide structure after str-normalization]")
    print(f"Unique case IDs                         : {len(norm_set)}")
    print(f"Unique formal targets                   : {len(all_targets)}")
    print(f"Unique target-surrogate groups          : {len(all_ts)}")
    print("Expected                                : 100 / 1600 / 4800")
    print()

    print("[Interpretation]")
    if len(missing) == 1 and len(extra) == 0 and not collisions:
        print("RESULT: one frozen selected case_id is genuinely absent from canonical_rules_test.jsonl.")
        print("Do NOT change 100 -> 99. Next inspect why Phase 4D-1/2 emitted no canonical rules for that case.")
    elif collisions:
        print("RESULT: at least one case_id representation collision exists.")
        print("The Phase 4D-3 audit must preserve the exact canonical case identity used by Phase 4D-0/1/2 rather than blindly astype(str).")
    elif len(missing) or len(extra):
        print("RESULT: selection/rule-bank identity mismatch detected.")
        print("Do NOT retune or replace cases; inspect the reported IDs.")
    else:
        print("RESULT: selection set matches after normalization.")
        print("If the count is still 99, inspect raw-type collisions and structural counts above.")

    print("=" * 112)


if __name__ == "__main__":
    main()
