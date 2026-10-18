#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage55
BPIC2018 External Protocol Freeze + Dataset Build

Purpose
-------
Freeze the BPIC2018 external-validation data protocol before any model training.

This stage performs ONLY deterministic data construction/auditing:
  1) Load BPIC2018 XES / XES.GZ.
  2) Stable-sort events within each case by time:timestamp; ties preserve original XES order.
  3) Read case-level year.
  4) Locate first prediction decision boundary:
         doctype    == "payment application"
         subprocess == "application"
         activity   == "decide"
  5) Exclude cases without this decision boundary.
  6) Construct the frozen late-payment label:
         POSITIVE iff
           a) no "begin payment", OR
           b) last "begin payment" occurs in a year later than case-year, OR
           c) an "abort payment" occurs after the last "begin payment".
     These three positive reasons are mutually assigned with the precedence:
           no_begin_payment
           > last_begin_followed_later_by_abort
           > last_begin_payment_later_year
     so every positive case has exactly one audit reason.
  7) Construct prefixes P5/P10/P15/P20 strictly BEFORE the first decision.
  8) Freeze year roles:
         2015 -> development_train
         2016 -> development_valid
         2017 -> formal_external_test
     and final-fit role:
         2015+2016 -> final_train
         2017      -> formal_external_test
  9) Produce common canonical files for later RF/XGB/MLP/LSTM stages.
 10) Run hard consistency gates and create SHA256 fingerprints.

IMPORTANT
---------
- Stage55 does NOT train models.
- Stage55 does NOT tune any parameter.
- 2017 must never be used for model/threshold/hyperparameter selection.
- Frozen D-S lambda remains 5 and is recorded only as protocol metadata.
- Later stages must NOT alter Stage55 files based on BPIC2018 TEST performance.

Expected project root
---------------------
D:\\ETCE

Input discovery
---------------
By default the script searches:
  D:\\ETCE\\data\\raw\\bpic2018\\*.xes
  D:\\ETCE\\data\\raw\\bpic2018\\*.xes.gz
  D:\\ETCE\\data\\raw\\*2018*.xes
  D:\\ETCE\\data\\raw\\*2018*.xes.gz
or use --xes explicitly.

Output
------
D:\\ETCE\\results\\phase4\\phase4e_bpic2018_external\\stage55_dataset\\

Files:
  protocol_frozen.json
  case_manifest.csv
  prefix_manifest.csv
  prefix_events.csv.gz
  class_stats.csv
  consistency_checks.json
  dataset_summary.json
  sha256_manifest.json
  STAGE55_FROZEN_PROTOCOL.md

The long prefix_events.csv.gz contains only events visible at each prefix and is the
shared canonical source for RF/XGB/MLP/LSTM feature builders in later stages.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


STAGE = 55
PREFIXES = (5, 10, 15, 20)
FROZEN_DS_LAMBDA = 5

EXPECTED_COUNTS = {
    "all_cases": 43809,
    "all_events": 2514266,
    "prediction_cases": 43793,
    "no_decision_cases": 16,
    "positive_all_before_decision_filter": 256,
    "case_year_counts": {2015: 14750, 2016: 14552, 2017: 14507},
    "positive_by_year_all_before_decision_filter": {2015: 99, 2016: 77, 2017: 80},
}

DECISION_DOCTYPE = "payment application"
DECISION_SUBPROCESS = "application"
DECISION_ACTIVITY = "decide"

BEGIN_PAYMENT_ACTIVITY = "begin payment"
ABORT_PAYMENT_ACTIVITY = "abort payment"

# Explicitly frozen annual roles.
DEV_ROLE_BY_YEAR = {
    2015: "development_train",
    2016: "development_valid",
    2017: "formal_external_test",
}
FINAL_ROLE_BY_YEAR = {
    2015: "final_train",
    2016: "final_train",
    2017: "formal_external_test",
}


# -----------------------------
# Generic helpers
# -----------------------------

def norm_text(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x).strip()).lower()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_xes_scalar(elem: ET.Element) -> Any:
    kind = local_name(elem.tag)
    value = elem.attrib.get("value")

    if kind in ("string", "id"):
        return "" if value is None else value
    if kind == "date":
        return "" if value is None else value
    if kind == "int":
        try:
            return int(value) if value is not None else None
        except Exception:
            return value
    if kind == "float":
        try:
            return float(value) if value is not None else None
        except Exception:
            return value
    if kind == "boolean":
        if value is None:
            return None
        return value.strip().lower() == "true"
    # Unknown XES scalar/list/container type: preserve textual value when present.
    return value


def parse_attr_children(parent: ET.Element) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for child in parent:
        kind = local_name(child.tag)
        if kind == "event":
            continue
        key = child.attrib.get("key")
        if key is None:
            continue

        # BPIC logs overwhelmingly use scalar attributes.
        # For nested list/container attributes preserve a deterministic JSON-like structure.
        if len(child):
            nested: Dict[str, Any] = {}
            scalar = parse_xes_scalar(child)
            if scalar is not None:
                nested["value"] = scalar
            for gc in child:
                gk = gc.attrib.get("key")
                if gk is not None:
                    nested[gk] = parse_xes_scalar(gc)
            out[key] = nested
        else:
            out[key] = parse_xes_scalar(child)
    return out


def parse_timestamp(value: Any) -> datetime:
    if value is None or value == "":
        raise ValueError("Missing time:timestamp")

    s = str(value).strip()
    # Python fromisoformat understands offsets but not trailing Z on older versions.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        # Fallbacks seen in XES exports.
        fmts = (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        )
        for fmt in fmts:
            try:
                return datetime.strptime(str(value), fmt)
            except ValueError:
                pass
    raise ValueError(f"Unsupported timestamp: {value!r}")


def timestamp_sort_key(value: Any) -> float:
    dt = parse_timestamp(value)
    # Naive and aware datetimes cannot be compared directly. Convert both to numeric.
    if dt.tzinfo is None:
        return dt.timestamp()
    return dt.timestamp()


def get_case_id(trace_attrs: Dict[str, Any], ordinal: int) -> str:
    for key in ("concept:name", "case:concept:name", "caseid", "case_id"):
        if key in trace_attrs and str(trace_attrs[key]).strip():
            return str(trace_attrs[key]).strip()
    return f"__trace_{ordinal:06d}"


def get_case_year(trace_attrs: Dict[str, Any]) -> int:
    candidates = []
    for key, value in trace_attrs.items():
        nk = norm_text(key)
        if nk == "year" or nk.endswith(":year") or nk == "case:year":
            candidates.append(value)

    if not candidates:
        raise KeyError("No case-level year attribute found")

    for v in candidates:
        try:
            y = int(float(str(v)))
            if y in (2015, 2016, 2017):
                return y
        except Exception:
            continue
    raise ValueError(f"Invalid case-level year candidate(s): {candidates!r}")


def get_event_activity(attrs: Dict[str, Any]) -> str:
    # Stage54 diagnostics referred to "activity"; BPIC XES may expose either
    # activity or concept:name. Prefer the explicit activity field if present.
    for key in ("activity", "concept:name"):
        if key in attrs and str(attrs[key]).strip():
            return str(attrs[key]).strip()
    return ""


def get_event_doctype(attrs: Dict[str, Any]) -> str:
    for key in ("doctype", "document type", "document_type"):
        if key in attrs and str(attrs[key]).strip():
            return str(attrs[key]).strip()
    return ""


def get_event_subprocess(attrs: Dict[str, Any]) -> str:
    for key in ("subprocess", "sub process", "sub_process"):
        if key in attrs and str(attrs[key]).strip():
            return str(attrs[key]).strip()
    return ""


def get_event_timestamp(attrs: Dict[str, Any]) -> Any:
    for key in ("time:timestamp", "timestamp", "time"):
        if key in attrs and attrs[key] not in (None, ""):
            return attrs[key]
    raise KeyError("No event timestamp attribute found")


def is_first_decision_event(attrs: Dict[str, Any]) -> bool:
    return (
        norm_text(get_event_doctype(attrs)) == DECISION_DOCTYPE
        and norm_text(get_event_subprocess(attrs)) == DECISION_SUBPROCESS
        and norm_text(get_event_activity(attrs)) == DECISION_ACTIVITY
    )


def is_payment_application_application_event(attrs: Dict[str, Any], activity: str) -> bool:
    """
    Frozen BPIC2018 payment-outcome context.

    IMPORTANT:
    BPIC2018 may contain later payment-related activities in other subprocesses
    (notably Change). The late-payment outcome audited in Stage54 concerns the
    original Payment application / Application payment flow only. Therefore
    begin/abort-payment events used for the outcome MUST match all three fields.
    """
    return (
        norm_text(get_event_doctype(attrs)) == DECISION_DOCTYPE
        and norm_text(get_event_subprocess(attrs)) == DECISION_SUBPROCESS
        and norm_text(get_event_activity(attrs)) == norm_text(activity)
    )


def is_begin_payment(attrs: Dict[str, Any]) -> bool:
    return is_payment_application_application_event(attrs, BEGIN_PAYMENT_ACTIVITY)


def is_abort_payment(attrs: Dict[str, Any]) -> bool:
    return is_payment_application_application_event(attrs, ABORT_PAYMENT_ACTIVITY)


@dataclass
class Event:
    original_index: int
    sorted_index: int
    timestamp_raw: str
    timestamp_epoch: float
    activity: str
    doctype: str
    subprocess: str
    attrs: Dict[str, Any]


@dataclass
class CaseRecord:
    case_id: str
    case_year: int
    trace_attrs: Dict[str, Any]
    events: List[Event]


# -----------------------------
# Streaming XES reader
# -----------------------------

def open_xes(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def iter_xes_traces(path: Path) -> Iterator[Tuple[int, Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Stream one <trace> at a time.

    Memory use is bounded by one trace, which matters because BPIC2018 has
    ~2.5M events and a max trace length around 2973.
    """
    with open_xes(path) as fh:
        context = ET.iterparse(fh, events=("end",))
        trace_ordinal = 0
        for _, elem in context:
            if local_name(elem.tag) != "trace":
                continue

            trace_attrs: Dict[str, Any] = {}
            events: List[Dict[str, Any]] = []

            for child in elem:
                if local_name(child.tag) == "event":
                    events.append(parse_attr_children(child))
                else:
                    key = child.attrib.get("key")
                    if key is not None:
                        trace_attrs[key] = parse_xes_scalar(child)

            yield trace_ordinal, trace_attrs, events
            trace_ordinal += 1
            elem.clear()


# -----------------------------
# Input/output
# -----------------------------

def discover_xes(project_root: Path) -> Path:
    raw = project_root / "data" / "raw"
    patterns = [
        raw / "bpic2018",
        raw,
    ]

    candidates: List[Path] = []
    for base in patterns:
        if not base.exists():
            continue
        for p in base.glob("*"):
            name = p.name.lower()
            if p.is_file() and (name.endswith(".xes") or name.endswith(".xes.gz")):
                if base.name.lower() == "bpic2018" or "2018" in name:
                    candidates.append(p)

    # Deduplicate after resolve.
    unique: List[Path] = []
    seen = set()
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    if len(unique) == 1:
        return unique[0]

    if not unique:
        raise FileNotFoundError(
            "No BPIC2018 .xes/.xes.gz found. Use --xes explicitly, e.g. "
            r'--xes "D:\ETCE\data\raw\bpic2018\BPI Challenge 2018.xes.gz"'
        )

    msg = "\n".join(f"  - {p}" for p in unique)
    raise RuntimeError(
        "Multiple possible BPIC2018 XES files found; use --xes explicitly:\n" + msg
    )


def ensure_clean_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        existing = [p for p in out_dir.iterdir()]
        if existing and not overwrite:
            raise FileExistsError(
                f"Output directory already contains files:\n  {out_dir}\n"
                "Stage55 is a freeze stage. Refusing to overwrite silently.\n"
                "Use --overwrite ONLY if rebuilding from the exact same frozen protocol."
            )
        if overwrite:
            for p in existing:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    import shutil
                    shutil.rmtree(p)
    out_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Label construction
# -----------------------------

def build_label(events: Sequence[Event], case_year: int) -> Tuple[int, str, Dict[str, Any]]:
    begin_positions = [i for i, e in enumerate(events) if is_begin_payment(e.attrs)]
    abort_positions = [i for i, e in enumerate(events) if is_abort_payment(e.attrs)]

    if not begin_positions:
        return 1, "no_begin_payment", {
            "begin_payment_count": 0,
            "abort_payment_count": len(abort_positions),
            "last_begin_payment_sorted_index": None,
            "last_begin_payment_year": None,
            "abort_after_last_begin": bool(abort_positions),
        }

    last_begin_pos = begin_positions[-1]
    last_begin_event = events[last_begin_pos]
    last_begin_year = parse_timestamp(last_begin_event.timestamp_raw).year
    abort_after = any(pos > last_begin_pos for pos in abort_positions)

    # Precedence only determines the unique audit reason.
    if abort_after:
        reason = "last_begin_followed_later_by_abort"
        label = 1
    elif last_begin_year > case_year:
        reason = "last_begin_payment_later_year"
        label = 1
    else:
        reason = "on_time"
        label = 0

    return label, reason, {
        "begin_payment_count": len(begin_positions),
        "abort_payment_count": len(abort_positions),
        "last_begin_payment_sorted_index": last_begin_event.sorted_index,
        "last_begin_payment_year": last_begin_year,
        "abort_after_last_begin": abort_after,
    }


# -----------------------------
# Main build
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="ETCE Stage55 - BPIC2018 External Protocol Freeze + Dataset Build"
    )
    ap.add_argument(
        "--project_root",
        type=Path,
        default=Path(r"D:\ETCE"),
        help=r"ETCE project root (default: D:\ETCE)",
    )
    ap.add_argument(
        "--xes",
        type=Path,
        default=None,
        help="Explicit BPIC2018 .xes or .xes.gz path. If omitted, auto-discover.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory. Default: results/phase4/phase4e_bpic2018_external/stage55_dataset",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow rebuilding Stage55 output directory. Never use this to tune based on TEST.",
    )
    ap.add_argument(
        "--skip_expected_count_gate",
        action="store_true",
        help="DEBUG ONLY: do not fail when Stage54 expected counts differ.",
    )
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    xes_path = args.xes.resolve() if args.xes else discover_xes(project_root)
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else (
            project_root
            / "results"
            / "phase4"
            / "phase4e_bpic2018_external"
            / "stage55_dataset"
        )
    )

    if not xes_path.exists():
        raise FileNotFoundError(xes_path)

    ensure_clean_output_dir(out_dir, args.overwrite)

    started = time.time()

    # Freeze metadata BEFORE parsing.
    protocol = {
        "stage": STAGE,
        "name": "BPIC2018 External Protocol Freeze + Dataset Build",
        "dataset": "BPIC2018",
        "created_local": datetime.now().astimezone().isoformat(),
        "input_xes": str(xes_path),
        "project_root": str(project_root),
        "output_dir": str(out_dir),
        "prefixes": list(PREFIXES),
        "event_sort": {
            "primary": "time:timestamp ascending",
            "tie_break": "original XES event order (stable)",
        },
        "decision_boundary": {
            "doctype": DECISION_DOCTYPE,
            "subprocess": DECISION_SUBPROCESS,
            "activity": DECISION_ACTIVITY,
            "policy": "prefix events must be strictly before first matching decision",
            "no_decision_policy": "exclude from prediction cohort",
        },
        "late_payment_label": {
            "payment_event_context": {
                "doctype": DECISION_DOCTYPE,
                "subprocess": DECISION_SUBPROCESS,
                "activities": [BEGIN_PAYMENT_ACTIVITY, ABORT_PAYMENT_ACTIVITY],
            },
            "positive_if_any": [
                "no Payment application/Application/begin payment",
                "last matching begin payment timestamp year > case-level year",
                "a matching abort payment occurs after the last matching begin payment",
            ],
            "audit_reason_precedence": [
                "no_begin_payment",
                "last_begin_followed_later_by_abort",
                "last_begin_payment_later_year",
                "on_time",
            ],
            "case_year_source": "case-level year attribute only",
        },
        "year_protocol": {
            "development_train": [2015],
            "development_valid": [2016],
            "formal_external_test": [2017],
            "final_refit_train": [2015, 2016],
            "formal_external_test_for_final_refit": [2017],
            "test_selection_forbidden": True,
        },
        "frozen_from_bpic2017": {
            "ds_lambda": FROZEN_DS_LAMBDA,
            "reliability_protocol": "unchanged",
            "bpa_protocol": "unchanged",
            "hierarchical_ds_protocol": "unchanged",
            "baseline_protocol": "unchanged",
            "process_validity_protocol": "unchanged",
            "retuning_on_bpic2018_test_forbidden": True,
        },
        "shared_downstream_models": ["RF", "XGB", "MLP", "LSTM"],
        "stage55_scope": "data-only; no training; no tuning",
    }
    protocol_path = out_dir / "protocol_frozen.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    case_manifest_path = out_dir / "case_manifest.csv"
    prefix_manifest_path = out_dir / "prefix_manifest.csv"
    prefix_events_path = out_dir / "prefix_events.csv.gz"

    case_fields = [
        "case_id",
        "case_year",
        "dev_role",
        "final_role",
        "included_prediction_cohort",
        "exclusion_reason",
        "label",
        "label_reason",
        "trace_event_count",
        "predecision_event_count",
        "first_decision_sorted_index",
        "first_decision_timestamp",
        "begin_payment_count",
        "abort_payment_count",
        "last_begin_payment_sorted_index",
        "last_begin_payment_year",
        "abort_after_last_begin",
        "supports_p5",
        "supports_p10",
        "supports_p15",
        "supports_p20",
        "trace_attrs_json",
    ]

    prefix_fields = [
        "case_id",
        "case_year",
        "dev_role",
        "final_role",
        "label",
        "label_reason",
        "prefix_len",
        "visible_event_count",
        "prefix_last_timestamp",
        "first_decision_sorted_index",
        "first_decision_timestamp",
    ]

    event_fields = [
        "case_id",
        "case_year",
        "dev_role",
        "final_role",
        "label",
        "label_reason",
        "prefix_len",
        "prefix_event_pos",
        "sorted_event_index",
        "original_xes_event_index",
        "timestamp",
        "activity",
        "doctype",
        "subprocess",
        "event_attrs_json",
    ]

    total_cases = 0
    total_events = 0
    no_decision_cases = 0
    prediction_cases = 0
    timestamp_reversal_cases = 0
    timestamp_reversal_pairs = 0
    case_year_counts = Counter()
    prediction_year_counts = Counter()
    all_label_reason_counts = Counter()
    all_label_by_year = Counter()
    prediction_label_reason_counts = Counter()
    prediction_label_by_year = Counter()
    prefix_rows_count = Counter()
    prefix_event_rows_count = Counter()
    duplicate_case_ids: List[str] = []
    seen_case_ids = set()
    missing_ts_cases: List[str] = []
    bad_year_cases: List[str] = []
    early_decision_cases: List[Tuple[str, int]] = []

    with (
        case_manifest_path.open("w", newline="", encoding="utf-8-sig") as case_f,
        prefix_manifest_path.open("w", newline="", encoding="utf-8-sig") as pref_f,
        gzip.open(prefix_events_path, "wt", newline="", encoding="utf-8") as event_f,
    ):
        case_writer = csv.DictWriter(case_f, fieldnames=case_fields)
        pref_writer = csv.DictWriter(pref_f, fieldnames=prefix_fields)
        event_writer = csv.DictWriter(event_f, fieldnames=event_fields)

        case_writer.writeheader()
        pref_writer.writeheader()
        event_writer.writeheader()

        for trace_ordinal, trace_attrs, raw_events in iter_xes_traces(xes_path):
            total_cases += 1
            total_events += len(raw_events)

            case_id = get_case_id(trace_attrs, trace_ordinal)
            if case_id in seen_case_ids:
                duplicate_case_ids.append(case_id)
            seen_case_ids.add(case_id)

            try:
                case_year = get_case_year(trace_attrs)
            except Exception:
                bad_year_cases.append(case_id)
                raise

            case_year_counts[case_year] += 1

            # Count pre-sort adjacent reversals using original XES order.
            original_ts: List[float] = []
            parsed_events: List[Tuple[int, Dict[str, Any], str, float]] = []
            for original_idx, attrs in enumerate(raw_events):
                try:
                    ts_raw = str(get_event_timestamp(attrs))
                    ts_epoch = timestamp_sort_key(ts_raw)
                except Exception:
                    missing_ts_cases.append(case_id)
                    raise
                original_ts.append(ts_epoch)
                parsed_events.append((original_idx, attrs, ts_raw, ts_epoch))

            local_reversal_pairs = sum(
                1 for a, b in zip(original_ts, original_ts[1:]) if b < a
            )
            if local_reversal_pairs:
                timestamp_reversal_cases += 1
                timestamp_reversal_pairs += local_reversal_pairs

            # Python sorted() is stable, so equal timestamps preserve XES order.
            parsed_events.sort(key=lambda x: x[3])

            events: List[Event] = []
            for sorted_idx, (original_idx, attrs, ts_raw, ts_epoch) in enumerate(parsed_events):
                events.append(
                    Event(
                        original_index=original_idx,
                        sorted_index=sorted_idx,
                        timestamp_raw=ts_raw,
                        timestamp_epoch=ts_epoch,
                        activity=get_event_activity(attrs),
                        doctype=get_event_doctype(attrs),
                        subprocess=get_event_subprocess(attrs),
                        attrs=attrs,
                    )
                )

            # Full-trace outcome label. This label is never reconstructed from prefix data.
            label, label_reason, label_meta = build_label(events, case_year)
            all_label_reason_counts[label_reason] += 1
            if label == 1:
                all_label_by_year[case_year] += 1

            first_decision_pos: Optional[int] = None
            for i, e in enumerate(events):
                if is_first_decision_event(e.attrs):
                    first_decision_pos = i
                    break

            included = first_decision_pos is not None
            if included:
                prediction_cases += 1
                prediction_year_counts[case_year] += 1
                prediction_label_reason_counts[label_reason] += 1
                if label == 1:
                    prediction_label_by_year[case_year] += 1
            else:
                no_decision_cases += 1

            predecision_count = first_decision_pos if first_decision_pos is not None else ""
            first_decision_timestamp = (
                events[first_decision_pos].timestamp_raw if first_decision_pos is not None else ""
            )

            supports = {
                p: bool(included and first_decision_pos is not None and first_decision_pos >= p)
                for p in PREFIXES
            }

            if included and not supports[max(PREFIXES)]:
                early_decision_cases.append((case_id, int(first_decision_pos)))

            dev_role = DEV_ROLE_BY_YEAR.get(case_year, "invalid_year")
            final_role = FINAL_ROLE_BY_YEAR.get(case_year, "invalid_year")

            case_writer.writerow(
                {
                    "case_id": case_id,
                    "case_year": case_year,
                    "dev_role": dev_role,
                    "final_role": final_role,
                    "included_prediction_cohort": int(included),
                    "exclusion_reason": "" if included else "no_first_decision",
                    "label": label,
                    "label_reason": label_reason,
                    "trace_event_count": len(events),
                    "predecision_event_count": predecision_count,
                    "first_decision_sorted_index": first_decision_pos if included else "",
                    "first_decision_timestamp": first_decision_timestamp,
                    "begin_payment_count": label_meta["begin_payment_count"],
                    "abort_payment_count": label_meta["abort_payment_count"],
                    "last_begin_payment_sorted_index": (
                        "" if label_meta["last_begin_payment_sorted_index"] is None
                        else label_meta["last_begin_payment_sorted_index"]
                    ),
                    "last_begin_payment_year": (
                        "" if label_meta["last_begin_payment_year"] is None
                        else label_meta["last_begin_payment_year"]
                    ),
                    "abort_after_last_begin": int(bool(label_meta["abort_after_last_begin"])),
                    "supports_p5": int(supports[5]),
                    "supports_p10": int(supports[10]),
                    "supports_p15": int(supports[15]),
                    "supports_p20": int(supports[20]),
                    "trace_attrs_json": json_dumps(trace_attrs),
                }
            )

            if included:
                assert first_decision_pos is not None
                for p in PREFIXES:
                    if not supports[p]:
                        continue

                    visible = events[:p]
                    prefix_rows_count[p] += 1
                    prefix_event_rows_count[p] += len(visible)

                    pref_writer.writerow(
                        {
                            "case_id": case_id,
                            "case_year": case_year,
                            "dev_role": dev_role,
                            "final_role": final_role,
                            "label": label,
                            "label_reason": label_reason,
                            "prefix_len": p,
                            "visible_event_count": len(visible),
                            "prefix_last_timestamp": visible[-1].timestamp_raw,
                            "first_decision_sorted_index": first_decision_pos,
                            "first_decision_timestamp": first_decision_timestamp,
                        }
                    )

                    for j, e in enumerate(visible, start=1):
                        event_writer.writerow(
                            {
                                "case_id": case_id,
                                "case_year": case_year,
                                "dev_role": dev_role,
                                "final_role": final_role,
                                "label": label,
                                "label_reason": label_reason,
                                "prefix_len": p,
                                "prefix_event_pos": j,
                                "sorted_event_index": e.sorted_index,
                                "original_xes_event_index": e.original_index,
                                "timestamp": e.timestamp_raw,
                                "activity": e.activity,
                                "doctype": e.doctype,
                                "subprocess": e.subprocess,
                                "event_attrs_json": json_dumps(e.attrs),
                            }
                        )

            if total_cases % 2000 == 0:
                elapsed = time.time() - started
                print(
                    f"[Stage55] parsed cases={total_cases:,} | events={total_events:,} "
                    f"| prediction={prediction_cases:,} | elapsed={elapsed:.1f}s",
                    flush=True,
                )

    # -----------------------------
    # Frozen gates
    # -----------------------------
    positive_all = (
        all_label_reason_counts["no_begin_payment"]
        + all_label_reason_counts["last_begin_payment_later_year"]
        + all_label_reason_counts["last_begin_followed_later_by_abort"]
    )
    positive_prediction = (
        prediction_label_reason_counts["no_begin_payment"]
        + prediction_label_reason_counts["last_begin_payment_later_year"]
        + prediction_label_reason_counts["last_begin_followed_later_by_abort"]
    )

    checks: Dict[str, Dict[str, Any]] = {}

    def add_check(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks[name] = {
            "pass": bool(passed),
            "actual": actual,
            "expected": expected,
        }

    add_check(
        "all_cases",
        total_cases == EXPECTED_COUNTS["all_cases"],
        total_cases,
        EXPECTED_COUNTS["all_cases"],
    )
    add_check(
        "all_events",
        total_events == EXPECTED_COUNTS["all_events"],
        total_events,
        EXPECTED_COUNTS["all_events"],
    )
    add_check(
        "case_year_counts",
        dict(sorted(case_year_counts.items())) == EXPECTED_COUNTS["case_year_counts"],
        dict(sorted(case_year_counts.items())),
        EXPECTED_COUNTS["case_year_counts"],
    )
    add_check(
        "prediction_cases",
        prediction_cases == EXPECTED_COUNTS["prediction_cases"],
        prediction_cases,
        EXPECTED_COUNTS["prediction_cases"],
    )
    add_check(
        "no_decision_cases",
        no_decision_cases == EXPECTED_COUNTS["no_decision_cases"],
        no_decision_cases,
        EXPECTED_COUNTS["no_decision_cases"],
    )
    add_check(
        "positive_all_cases",
        positive_all == EXPECTED_COUNTS["positive_all_before_decision_filter"],
        positive_all,
        EXPECTED_COUNTS["positive_all_before_decision_filter"],
    )
    add_check(
        "label_reason_counts_all_cases",
        {
            "on_time": all_label_reason_counts["on_time"],
            "last_begin_payment_later_year": all_label_reason_counts["last_begin_payment_later_year"],
            "no_begin_payment": all_label_reason_counts["no_begin_payment"],
            "last_begin_followed_later_by_abort": all_label_reason_counts["last_begin_followed_later_by_abort"],
        }
        == {
            "on_time": 43553,
            "last_begin_payment_later_year": 168,
            "no_begin_payment": 82,
            "last_begin_followed_later_by_abort": 6,
        },
        {
            "on_time": all_label_reason_counts["on_time"],
            "last_begin_payment_later_year": all_label_reason_counts["last_begin_payment_later_year"],
            "no_begin_payment": all_label_reason_counts["no_begin_payment"],
            "last_begin_followed_later_by_abort": all_label_reason_counts["last_begin_followed_later_by_abort"],
        },
        {
            "on_time": 43553,
            "last_begin_payment_later_year": 168,
            "no_begin_payment": 82,
            "last_begin_followed_later_by_abort": 6,
        },
    )
    add_check(
        "positive_by_year_all_cases",
        {y: all_label_by_year[y] for y in (2015, 2016, 2017)}
        == EXPECTED_COUNTS["positive_by_year_all_before_decision_filter"],
        {y: all_label_by_year[y] for y in (2015, 2016, 2017)},
        EXPECTED_COUNTS["positive_by_year_all_before_decision_filter"],
    )
    add_check(
        "duplicate_case_ids",
        len(duplicate_case_ids) == 0,
        duplicate_case_ids[:20],
        [],
    )
    add_check(
        "missing_timestamp_cases",
        len(missing_ts_cases) == 0,
        missing_ts_cases[:20],
        [],
    )
    add_check(
        "bad_year_cases",
        len(bad_year_cases) == 0,
        bad_year_cases[:20],
        [],
    )
    add_check(
        "all_prediction_cases_support_p20",
        len(early_decision_cases) == 0,
        early_decision_cases[:20],
        [],
    )
    add_check(
        "prefix_manifest_counts",
        all(prefix_rows_count[p] == prediction_cases for p in PREFIXES),
        {f"P{p}": prefix_rows_count[p] for p in PREFIXES},
        {f"P{p}": prediction_cases for p in PREFIXES},
    )
    add_check(
        "prefix_event_counts",
        all(prefix_event_rows_count[p] == prediction_cases * p for p in PREFIXES),
        {f"P{p}": prefix_event_rows_count[p] for p in PREFIXES},
        {f"P{p}": prediction_cases * p for p in PREFIXES},
    )

    # Stage54 observed exactly 863 cases / 863 adjacent reversal pairs.
    add_check(
        "timestamp_reversal_cases",
        timestamp_reversal_cases == 863,
        timestamp_reversal_cases,
        863,
    )
    add_check(
        "timestamp_reversal_pairs",
        timestamp_reversal_pairs == 863,
        timestamp_reversal_pairs,
        863,
    )

    consistency_path = out_dir / "consistency_checks.json"
    consistency_path.write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Class statistics.
    class_stats_path = out_dir / "class_stats.csv"
    with class_stats_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scope",
                "year",
                "role",
                "cases",
                "positive",
                "negative",
                "positive_rate",
            ]
        )

        # Prediction cohort only: this is the population downstream models may use.
        for year in (2015, 2016, 2017):
            cases_y = prediction_year_counts[year]
            pos_y = prediction_label_by_year[year]
            neg_y = cases_y - pos_y
            w.writerow(
                [
                    "prediction_cohort",
                    year,
                    DEV_ROLE_BY_YEAR[year],
                    cases_y,
                    pos_y,
                    neg_y,
                    f"{(pos_y / cases_y if cases_y else 0.0):.10f}",
                ]
            )

        pos_all = positive_prediction
        w.writerow(
            [
                "prediction_cohort",
                "ALL",
                "ALL",
                prediction_cases,
                pos_all,
                prediction_cases - pos_all,
                f"{(pos_all / prediction_cases if prediction_cases else 0.0):.10f}",
            ]
        )

    summary = {
        "stage": STAGE,
        "dataset": "BPIC2018",
        "all_cases": total_cases,
        "all_events": total_events,
        "case_year_counts": dict(sorted(case_year_counts.items())),
        "prediction_cases": prediction_cases,
        "no_decision_cases": no_decision_cases,
        "prediction_year_counts": dict(sorted(prediction_year_counts.items())),
        "label_reason_counts_all_cases": dict(sorted(all_label_reason_counts.items())),
        "positive_all_cases": positive_all,
        "positive_by_year_all_cases": {
            y: all_label_by_year[y] for y in (2015, 2016, 2017)
        },
        "label_reason_counts_prediction": dict(sorted(prediction_label_reason_counts.items())),
        "positive_prediction_cases": positive_prediction,
        "positive_by_year_prediction": {
            y: prediction_label_by_year[y] for y in (2015, 2016, 2017)
        },
        "prefix_manifest_rows": {f"P{p}": prefix_rows_count[p] for p in PREFIXES},
        "prefix_event_rows": {f"P{p}": prefix_event_rows_count[p] for p in PREFIXES},
        "timestamp_reversal_cases_original_order": timestamp_reversal_cases,
        "timestamp_reversal_pairs_original_order": timestamp_reversal_pairs,
        "all_hard_checks_pass": all(v["pass"] for v in checks.values()),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    summary_path = out_dir / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Human-readable freeze note.
    md = f"""# ETCE Stage55 — BPIC2018 External Protocol Freeze

## Frozen cohort and time protocol
- Raw cases: **{total_cases:,}**
- Raw events: **{total_events:,}**
- Prediction cohort: **{prediction_cases:,}**
- Excluded because no first decision: **{no_decision_cases:,}**
- Prefixes: **P5 / P10 / P15 / P20**
- Development train: **2015**
- Development validation: **2016**
- Formal external TEST: **2017**
- Final refit before TEST: **2015 + 2016**
- 2017 parameter/model/threshold selection: **FORBIDDEN**

## Frozen decision boundary
`doctype=payment application`, `subprocess=application`, `activity=decide`.

Only events strictly before the first matching decision may enter a prefix.

## Frozen late-payment label
Only events with `doctype=payment application` and `subprocess=application`
participate in the payment-outcome label.

Positive iff at least one condition holds:
1. no matching `begin payment`;
2. last matching `begin payment` year > case-level year;
3. a matching `abort payment` occurs after the last matching `begin payment`.

All-case positives (Stage54 reproduction gate): **{positive_all:,} / {total_cases:,}**
({positive_all / total_cases:.6%})

Prediction-cohort positives after excluding no-decision cases:
**{positive_prediction:,} / {prediction_cases:,}**
({positive_prediction / prediction_cases:.6%})

## Frozen BPIC2017 protocol carried to external validation
- D-S lambda = **{FROZEN_DS_LAMBDA}**
- reliability protocol: unchanged
- BPA protocol: unchanged
- hierarchical D-S protocol: unchanged
- baselines: unchanged
- process-validity protocol: unchanged
- no retuning based on BPIC2018 formal TEST

## Shared downstream dataset
`prefix_events.csv.gz` is the canonical event-level prefix dataset shared by
RF / XGB / MLP / LSTM. Later stages may derive model-specific tensors/features
from it, but case IDs, labels, prefixes, annual roles, and visible events are frozen here.

## Gate status
**{"PASS" if summary["all_hard_checks_pass"] else "FAIL"}**
"""
    md_path = out_dir / "STAGE55_FROZEN_PROTOCOL.md"
    md_path.write_text(md, encoding="utf-8")

    # Hash AFTER all content files are final. Do not hash the hash manifest itself.
    files_to_hash = [
        xes_path,
        protocol_path,
        case_manifest_path,
        prefix_manifest_path,
        prefix_events_path,
        class_stats_path,
        consistency_path,
        summary_path,
        md_path,
    ]
    hashes = {}
    for p in files_to_hash:
        hashes[str(p)] = {
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    hash_manifest = {
        "stage": STAGE,
        "algorithm": "SHA256",
        "generated_local": datetime.now().astimezone().isoformat(),
        "files": hashes,
    }
    hash_path = out_dir / "sha256_manifest.json"
    hash_path.write_text(
        json.dumps(hash_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 112)
    print("ETCE STAGE55 - BPIC2018 EXTERNAL PROTOCOL FREEZE + DATASET BUILD")
    print("=" * 112)
    print(f"Input XES          : {xes_path}")
    print(f"Output             : {out_dir}")
    print(f"Cases              : {total_cases:,}")
    print(f"Events             : {total_events:,}")
    print(f"Case-year          : {dict(sorted(case_year_counts.items()))}")
    print(f"Prediction cohort  : {prediction_cases:,}")
    print(f"No first decision  : {no_decision_cases:,}")
    print(f"Positive (all)     : {positive_all:,} / {total_cases:,} "
          f"= {positive_all / total_cases:.6%}")
    print(
        "Positive/year all  : "
        + str({y: all_label_by_year[y] for y in (2015, 2016, 2017)})
    )
    print(f"Positive (cohort)  : {positive_prediction:,} / {prediction_cases:,} "
          f"= {positive_prediction / prediction_cases:.6%}")
    print(
        "Positive/year pred : "
        + str({y: prediction_label_by_year[y] for y in (2015, 2016, 2017)})
    )
    print(f"Label reasons      : {dict(sorted(prediction_label_reason_counts.items()))}")
    print(f"Original reversals : cases={timestamp_reversal_cases:,} | pairs={timestamp_reversal_pairs:,}")
    print(f"Prefix rows        : {dict(prefix_rows_count)}")
    print(f"Prefix event rows  : {dict(prefix_event_rows_count)}")
    print(f"Frozen D-S lambda  : {FROZEN_DS_LAMBDA}")
    print(f"Hard gates         : {'PASS' if summary['all_hard_checks_pass'] else 'FAIL'}")
    print(f"SHA256 manifest    : {hash_path}")
    print("=" * 112)

    failed = {k: v for k, v in checks.items() if not v["pass"]}
    if failed:
        print("[FAILED CHECKS]")
        for name, info in failed.items():
            print(f"  - {name}: actual={info['actual']!r} expected={info['expected']!r}")

        if args.skip_expected_count_gate:
            print(
                "\nWARNING: --skip_expected_count_gate was used. "
                "Outputs were written but Stage55 is NOT protocol-frozen."
            )
            return 0

        print(
            "\nSTAGE55 FAILED. Do not proceed to Stage56 until every hard gate passes.\n"
            "No model training/tuning should be started from a failed Stage55 build."
        )
        return 2

    print(
        "\nSTAGE55 FROZEN SUCCESSFULLY.\n"
        "NEXT: Stage56 may build development-only model features using 2015/2016; "
        "2017 remains untouched for parameter selection."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
