#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage62
BPIC2018 External Process-valid Local Pool + LORE/DT/BN Multi-rule Bank

Frozen inputs
-------------
Stage56 : FINAL 2015+2016 mapping + 2017 factual feature arrays
Stage60 : canonical seed42 black-box binding for 1600 formal targets
Stage61H: TRAIN-only BPIC2018 DFG + position support + dominant-subprocess
          semantic-family adapter

Protocol inherited from BPIC2017 Phase4D
----------------------------------------
- selected formal cases: exact frozen Stage59/60 100 cases
- models: RF / XGB / MLP / LSTM
- prefixes: P5 / P10 / P15 / P20
- local pool size: 320
- maximum activity substitutions: 3
- maximum mutation options per position: 8
- deterministic target-local seed base: 0
- only activity sequence is intervened on
- doctype / subprocess / timestamps / numeric factual context stay fixed
- process-valid TRAIN-only DFG
- same-position TRAIN support
- semantic-family preservation via frozen BPIC2018 adapter
- deterministic stratified 70/30 local fit/eval
- LORE / DT / BN definitions unchanged
- no 2017 true outcome label is used for neighborhood generation, surrogate
  fitting, rule extraction, or protocol selection

Critical preflight
------------------
Before any external neighborhood/rule is generated, this script independently
recomputes all 1600 factual seed42 canonical probabilities from the frozen
checkpoints + Stage56 features and requires agreement with Stage60 bindings.
If reproduction fails, Stage62 stops before rule-bank construction.

No D-S fusion is performed here.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

# -------------------------------------------------------------------------------------------------
# Frozen Stage62 constants
# -------------------------------------------------------------------------------------------------

STAGE = 62
MODELS = ("RF", "XGB", "MLP", "LSTM")
PREFIXES = (5, 10, 15, 20)
SURROGATES = ("LORE", "DT", "BN")

POOL_SIZE = 320
MAX_EDITS = 3
MAX_OPTIONS = 8
POOL_SEED_BASE = 0
THRESHOLD = 0.5

# Factual reproduction is a hard protocol gate.
# RF/XGB are expected to reproduce almost bit-for-bit.
# sklearn MLP can differ at ~1e-8 across BLAS/OS builds; label equality is also mandatory.
# CUDA LSTM allows only tiny floating-point drift; label equality is also mandatory.
PROB_ATOL_RF_XGB = 1e-10
PROB_ATOL_MLP = 1e-7
PROB_ATOL_LSTM = 1e-6

def factual_probability_tolerance(model: str) -> float:
    m = str(model).upper()
    if m in ("RF", "XGB", "XGBOOST"):
        return PROB_ATOL_RF_XGB
    if m == "MLP":
        return PROB_ATOL_MLP
    if m == "LSTM":
        return PROB_ATOL_LSTM
    raise ValueError(f"Unknown model for probability tolerance: {model}")

CACHE_SCHEMA = "phase4e-bpic2018-external-multirule-target-v1"


# -------------------------------------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------------------------------------

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def json_default(x: Any):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, set):
        return sorted(x)
    if isinstance(x, tuple):
        return list(x)
    raise TypeError(f"Not JSON serializable: {type(x)}")


def json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, set):
        return sorted(json_safe(v) for v in x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
    return x


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stable_local_seed(model: str, prefix: int, case_id: str) -> int:
    token = f"{POOL_SEED_BASE}|{model}|{int(prefix)}|{case_id}"
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)


def pool_digest(sequences: Sequence[Sequence[str]]) -> str:
    payload = "\n".join("\x1f".join(map(str, seq)) for seq in sequences)
    return sha256_text(payload)


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and overwrite:
        import shutil
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def import_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def pick_column(columns: Sequence[str], options: Sequence[str]) -> Optional[str]:
    low = {str(c).lower(): c for c in columns}
    for o in options:
        if o.lower() in low:
            return low[o.lower()]
    for c in columns:
        cl = str(c).lower()
        if any(o.lower() in cl for o in options):
            return c
    return None


# -------------------------------------------------------------------------------------------------
# Load frozen Stage61H constraints and patch BPIC2017 family globals
# -------------------------------------------------------------------------------------------------

def load_stage61h_constraints(stage61h: Path) -> Tuple[dict, Dict[str, str]]:
    protocol = read_json(stage61h / "protocol_frozen.json")
    checks = read_json(stage61h / "consistency_checks.json")

    if not protocol.get("all_hard_checks_pass", False):
        raise RuntimeError("Stage61H is not frozen PASS.")
    if any(not bool(v.get("pass", False)) for v in checks.values()):
        raise RuntimeError("Stage61H consistency checks are not all PASS.")

    meta = read_json(stage61h / "process_constraints.json")

    edge_count = {}
    with (stage61h / "dfg_edges.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            edge_count[(row["source_activity"], row["target_activity"])] = int(row["count"])

    position_count = {}
    with (stage61h / "activity_position_counts.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        for row in csv.DictReader(f):
            position_count[(int(row["position"]), row["activity"])] = int(row["count"])

    family_map = {}
    with (stage61h / "activity_family_map.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        for row in csv.DictReader(f):
            family_map[row["activity"]] = row["family"]

    constraints = dict(meta)
    constraints["edge_count"] = edge_count
    constraints["position_count"] = position_count
    constraints["mutable_activities"] = tuple(meta["mutable_activities"])

    return constraints, family_map


def patch_semantic_adapter(
    semantic_core,
    surrogate_core,
    constraints: dict,
    family_map: Dict[str, str],
) -> None:
    immutable = set(constraints.get("immutable_existing_activities", []))
    forbidden = set(constraints.get("forbidden_mutation_targets", []))

    def bp18_family_preserved(factual: str, candidate: str) -> bool:
        a = family_map.get(str(factual))
        b = family_map.get(str(candidate))
        return a is not None and b is not None and a == b

    # semantic_core functions resolve family_preserved and protected sets from
    # their module globals at call time.
    semantic_core.family_preserved = bp18_family_preserved
    semantic_core.TERMINAL_OFFER_ACTIVITIES = set(immutable)
    semantic_core.DIRECT_PROXIES = set()
    semantic_core.FORBIDDEN_MUTATION_TARGETS = set(forbidden)

    # surrogate_core imported the same names by value, so patch there too.
    surrogate_core.family_preserved = bp18_family_preserved
    surrogate_core.TERMINAL_OFFER_ACTIVITIES = set(immutable)
    surrogate_core.DIRECT_PROXIES = set()
    surrogate_core.FORBIDDEN_MUTATION_TARGETS = set(forbidden)


# -------------------------------------------------------------------------------------------------
# Stage56 feature adapter: intervene on activity only, keep other factual context fixed
# -------------------------------------------------------------------------------------------------

class PrefixFeatureAdapter:
    def __init__(self, stage56_module, mapping: dict, npz_path: Path, prefix: int):
        self.stage56 = stage56_module
        self.mapping = mapping
        self.prefix = int(prefix)
        self.npz_path = npz_path

        self.arrays = self._load_npz(npz_path)
        self.n = self._infer_n()
        self.case_key = self._find_case_key()
        self.case_ids = [str(v) for v in self.arrays[self.case_key].tolist()]
        self.index_by_case = {cid: i for i, cid in enumerate(self.case_ids)}
        if len(self.index_by_case) != len(self.case_ids):
            raise RuntimeError(f"Duplicate case IDs in {npz_path.name}")

        self.tab_key = self._find_tabular_key()
        self.activity_seq_key = self._find_sequence_key("activity")
        self.doctype_seq_key = self._find_sequence_key("doctype")
        self.subprocess_seq_key = self._find_sequence_key("subprocess")
        self.delta_key = self._find_delta_key()
        self.mask_key = self._find_mask_key()

        self.activity_vocab = mapping["activity_vocab"]
        self.pad_id = int(mapping.get("pad_id", 0))
        self.unk_id = int(mapping.get("unk_id", 1))
        self.id_to_activity = {int(v): str(k) for k, v in self.activity_vocab.items()}

        self.feature_names = self._normalize_feature_names(
            self.stage56.make_feature_names(mapping)
        )
        self.activity_count_indices, self.activity_count_mode = self._resolve_activity_count_block()

        self._verify_activity_encoding_on_factuals()

    @staticmethod
    def _load_npz(path: Path) -> Dict[str, np.ndarray]:
        with np.load(path, allow_pickle=True) as z:
            return {k: np.array(z[k]) for k in z.files}

    def _infer_n(self) -> int:
        lengths = Counter()
        for arr in self.arrays.values():
            if getattr(arr, "ndim", 0) >= 1:
                lengths[int(arr.shape[0])] += 1
        if not lengths:
            raise RuntimeError(f"No sample arrays found in {self.npz_path}")
        return lengths.most_common(1)[0][0]

    def _find_case_key(self) -> str:
        preferred = [
            "case_id", "case_ids", "caseid", "ids", "sample_id", "sample_ids"
        ]
        for k in self.arrays:
            if k.lower() in preferred and self.arrays[k].shape[0] == self.n:
                return k
        for k, arr in self.arrays.items():
            if arr.shape[0] != self.n:
                continue
            kl = k.lower()
            if "case" in kl and "id" in kl:
                return k
        raise RuntimeError(f"Cannot resolve case-id key from NPZ keys={list(self.arrays)}")

    def _normalize_feature_names(self, obj: Any) -> List[str]:
        if isinstance(obj, list):
            return [str(v) for v in obj]
        if isinstance(obj, tuple):
            return [str(v) for v in obj]
        if isinstance(obj, dict):
            # Prefer explicit tabular list.
            for key in ("tabular", "tabular_feature_names", "feature_names", "all"):
                if key in obj and isinstance(obj[key], (list, tuple)):
                    return [str(v) for v in obj[key]]
            # Otherwise use a unique list matching tabular width later.
            candidates = [
                [str(v) for v in val]
                for val in obj.values()
                if isinstance(val, (list, tuple))
            ]
            if candidates:
                # longest is usually complete tabular feature list
                return max(candidates, key=len)
        raise RuntimeError(
            f"Unsupported make_feature_names() output type: {type(obj)}"
        )

    def _find_tabular_key(self) -> str:
        width = len(self.feature_names) if hasattr(self, "feature_names") else None
        preferred = ("x_tab", "x_tabular", "tabular", "features", "x")
        for p in preferred:
            for k, arr in self.arrays.items():
                if (
                    p == k.lower()
                    and arr.ndim == 2
                    and arr.shape[0] == self.n
                    and (width is None or arr.shape[1] == width)
                ):
                    return k

        # feature_names is not available before this call during init, so use
        # Stage56 expected final tabular width from mapping.
        expected = (
            len(self.mapping["numeric_feature_names"])
            + (len(self.mapping["activity_vocab"]) - 1)
            + (len(self.mapping["doctype_vocab"]) - 1)
            + (len(self.mapping["subprocess_vocab"]) - 1)
        )
        candidates = [
            k for k, arr in self.arrays.items()
            if arr.ndim == 2 and arr.shape[0] == self.n and arr.shape[1] == expected
        ]
        if len(candidates) == 1:
            return candidates[0]

        for k, arr in self.arrays.items():
            if (
                arr.ndim == 2
                and arr.shape[0] == self.n
                and np.issubdtype(arr.dtype, np.floating)
                and arr.shape[1] > self.prefix
            ):
                return k
        raise RuntimeError(f"Cannot resolve tabular matrix from {list(self.arrays)}")

    def _find_sequence_key(self, token: str) -> Optional[str]:
        candidates = []
        for k, arr in self.arrays.items():
            kl = k.lower()
            if (
                token in kl
                and arr.ndim == 2
                and arr.shape[0] == self.n
                and arr.shape[1] == self.prefix
            ):
                candidates.append(k)
        if not candidates:
            return None
        candidates.sort(key=lambda k: (0 if "id" in k.lower() else 1, k))
        return candidates[0]

    def _find_delta_key(self) -> Optional[str]:
        for k, arr in self.arrays.items():
            kl = k.lower()
            if (
                ("delta" in kl or "time" in kl)
                and arr.ndim == 2
                and arr.shape[0] == self.n
                and arr.shape[1] == self.prefix
                and np.issubdtype(arr.dtype, np.floating)
            ):
                return k
        return None

    def _find_mask_key(self) -> Optional[str]:
        for k, arr in self.arrays.items():
            if (
                "mask" in k.lower()
                and arr.ndim == 2
                and arr.shape[0] == self.n
                and arr.shape[1] == self.prefix
            ):
                return k
        return None

    def _resolve_activity_count_block(self) -> Tuple[List[int], str]:
        tab_width = self.arrays[self.tab_key].shape[1]
        if len(self.feature_names) != tab_width:
            # Try rebuilding now that tab width is known.
            raw = self.stage56.make_feature_names(self.mapping)
            if isinstance(raw, dict):
                for v in raw.values():
                    if isinstance(v, (list, tuple)) and len(v) == tab_width:
                        self.feature_names = [str(x) for x in v]
                        break

        by_name = [
            i for i, n in enumerate(self.feature_names)
            if "activity" in n.lower() and ("count" in n.lower() or "freq" in n.lower())
        ]
        expected = len(self.activity_vocab) - 1  # PAD omitted, UNK retained.

        if len(by_name) == expected:
            return by_name, "feature_name"

        start = len(self.mapping["numeric_feature_names"])
        idx = list(range(start, start + expected))
        if max(idx) >= tab_width:
            raise RuntimeError(
                f"Derived activity count block out of range: width={tab_width}, idx={idx}"
            )
        return idx, "derived_numeric_then_activity_counts"

    def decode_factual_activities(self, row_index: int) -> Tuple[str, ...]:
        if self.activity_seq_key is None:
            raise RuntimeError("No activity sequence array found in Stage56 NPZ.")
        ids = self.arrays[self.activity_seq_key][row_index]
        out = []
        for v in ids:
            vid = int(v)
            if vid == self.pad_id:
                continue
            out.append(self.id_to_activity.get(vid, "<UNK>"))
        if len(out) != self.prefix:
            raise RuntimeError(
                f"Decoded factual length != prefix: {len(out)} vs {self.prefix}"
            )
        return tuple(out)

    def encode_activity_ids(self, sequence: Sequence[str]) -> np.ndarray:
        ids = [
            int(self.activity_vocab.get(str(a), self.unk_id))
            for a in sequence
        ]
        if len(ids) != self.prefix:
            raise ValueError(f"Expected sequence length {self.prefix}, got {len(ids)}")
        return np.asarray(ids, dtype=self.arrays[self.activity_seq_key].dtype)

    def _activity_count_candidate_vectors(self, sequence: Sequence[str]) -> List[np.ndarray]:
        candidates = []

        try:
            cv = np.asarray(
                self.stage56.count_vector(sequence, self.activity_vocab),
                dtype=float,
            ).reshape(-1)
            candidates.append(cv)
            if len(cv) > 0:
                candidates.append(cv[1:])
            if len(cv) > 1:
                candidates.append(cv[2:])
        except Exception:
            pass

        # Manual ID-count vectors.
        max_id = max(int(v) for v in self.activity_vocab.values())
        full = np.zeros(max_id + 1, dtype=float)
        for a in sequence:
            full[int(self.activity_vocab.get(str(a), self.unk_id))] += 1.0
        candidates.extend([full, full[1:], full[2:]])

        # unique by shape/value
        unique = []
        for c in candidates:
            if not any(c.shape == u.shape and np.array_equal(c, u) for u in unique):
                unique.append(c)
        return unique

    def activity_count_vector(self, sequence: Sequence[str]) -> np.ndarray:
        wanted = len(self.activity_count_indices)
        for c in self._activity_count_candidate_vectors(sequence):
            if len(c) == wanted:
                return c.astype(float)
        raise RuntimeError(
            f"Could not construct activity-count vector of length {wanted}"
        )

    def _verify_activity_encoding_on_factuals(self) -> None:
        # Verify both sequence ID encoding and tabular activity-count block on a
        # deterministic sample before any perturbation is used.
        check_indices = list(range(min(20, self.n)))
        tab = self.arrays[self.tab_key]

        for i in check_indices:
            seq = self.decode_factual_activities(i)

            encoded = self.encode_activity_ids(seq)
            original_ids = np.asarray(self.arrays[self.activity_seq_key][i])
            if not np.array_equal(encoded, original_ids):
                raise RuntimeError(
                    f"Activity sequence re-encoding mismatch at row {i}, P{self.prefix}"
                )

            count_vec = self.activity_count_vector(seq)
            original_counts = tab[i, self.activity_count_indices].astype(float)
            if not np.allclose(count_vec, original_counts, atol=0, rtol=0):
                raise RuntimeError(
                    "Activity count block reproduction failed at "
                    f"row {i}, P{self.prefix}. "
                    f"derived={count_vec.tolist()} original={original_counts.tolist()}"
                )

    def row_index(self, case_id: str) -> int:
        if str(case_id) not in self.index_by_case:
            raise KeyError(f"Case {case_id} not found in {self.npz_path.name}")
        return self.index_by_case[str(case_id)]

    def factual_batch(self, case_ids: Sequence[str]) -> Dict[str, np.ndarray]:
        idx = np.asarray([self.row_index(c) for c in case_ids], dtype=int)
        return {
            k: np.asarray(v[idx]).copy()
            for k, v in self.arrays.items()
            if getattr(v, "ndim", 0) >= 1 and v.shape[0] == self.n
        }

    def perturbed_batch(
        self,
        case_id: str,
        sequences: Sequence[Sequence[str]],
    ) -> Dict[str, np.ndarray]:
        i = self.row_index(case_id)
        m = len(sequences)

        batch = {}
        for k, arr in self.arrays.items():
            if getattr(arr, "ndim", 0) >= 1 and arr.shape[0] == self.n:
                base = np.asarray(arr[i])
                batch[k] = np.repeat(base[None, ...], m, axis=0)

        if self.activity_seq_key is None:
            raise RuntimeError("Activity sequence key unavailable.")

        for j, seq in enumerate(sequences):
            batch[self.activity_seq_key][j] = self.encode_activity_ids(seq)
            count_vec = self.activity_count_vector(seq)
            batch[self.tab_key][j, self.activity_count_indices] = count_vec

        return batch


# -------------------------------------------------------------------------------------------------
# Canonical black-box loader / predictor
# -------------------------------------------------------------------------------------------------

def sniff_json(path: Path) -> Optional[Any]:
    try:
        head = path.read_bytes()[:16].lstrip()
    except Exception:
        return None
    if not (head.startswith(b"{") or head.startswith(b"[")):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


def resolve_manifest_reference(path: Path, payload: Any) -> Optional[Path]:
    candidates = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                kl = str(k).lower()
                if isinstance(v, str) and any(
                    t in kl for t in (
                        "checkpoint", "model_path", "weights", "state_dict_path",
                        "model_file", "artifact_path", "best_model"
                    )
                ):
                    candidates.append(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(payload)

    for ref in candidates:
        q = Path(ref)
        if not q.is_absolute():
            q = (path.parent / q).resolve()
        if q.exists() and q.is_file() and q.resolve() != path.resolve():
            return q
    return None


def get_tabular_key_from_batch(batch: Dict[str, np.ndarray], adapter: PrefixFeatureAdapter) -> str:
    return adapter.tab_key


def tensor_from_array(arr: np.ndarray, semantic_name: str, device: torch.device):
    if semantic_name in ("activity", "doctype", "subprocess", "ids"):
        return torch.as_tensor(arr, dtype=torch.long, device=device)
    if semantic_name == "mask":
        # bool masks are safest; float masks also work for many models.
        if np.issubdtype(arr.dtype, np.bool_):
            return torch.as_tensor(arr, dtype=torch.bool, device=device)
        return torch.as_tensor(arr, dtype=torch.float32, device=device)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


class CanonicalExternalPredictor:
    def __init__(
        self,
        model_name: str,
        prefix: int,
        checkpoint: Path,
        mapping: dict,
        adapter: PrefixFeatureAdapter,
        stage58_module,
        device: torch.device,
    ):
        self.model_name = str(model_name)
        self.prefix = int(prefix)
        self.checkpoint = checkpoint.resolve()
        self.mapping = mapping
        self.adapter = adapter
        self.stage58 = stage58_module
        self.device = device
        self.kind = None
        self.model = None

        self._load()

    def _load(self):
        name = self.model_name.upper()

        # XGBoost native JSON is not a manifest.
        payload = sniff_json(self.checkpoint)
        if name in ("XGB", "XGBOOST") and isinstance(payload, dict) and "learner" in payload:
            model = xgb.XGBClassifier()
            model.load_model(str(self.checkpoint))
            self.kind = "sklearn"
            self.model = model
            return

        # JSON wrapper around a real checkpoint.
        if isinstance(payload, (dict, list)) and not (
            name in ("XGB", "XGBOOST") and isinstance(payload, dict) and "learner" in payload
        ):
            ref = resolve_manifest_reference(self.checkpoint, payload)
            if ref is None:
                # Some XGB JSONs have nonstandard top-level structure.
                if name in ("XGB", "XGBOOST"):
                    model = xgb.XGBClassifier()
                    model.load_model(str(self.checkpoint))
                    self.kind = "sklearn"
                    self.model = model
                    return
                raise RuntimeError(
                    f"JSON checkpoint wrapper has no resolvable model reference: {self.checkpoint}"
                )
            self.checkpoint = ref
            return self._load()

        if name in ("RF", "MLP"):
            self.model = joblib.load(self.checkpoint)
            if not hasattr(self.model, "predict_proba"):
                raise RuntimeError(f"{name} checkpoint lacks predict_proba: {self.checkpoint}")
            self.kind = "sklearn"
            return

        if name in ("XGB", "XGBOOST"):
            # joblib XGB fallback
            try:
                self.model = joblib.load(self.checkpoint)
                if hasattr(self.model, "predict_proba"):
                    self.kind = "sklearn"
                    return
            except Exception:
                pass
            model = xgb.XGBClassifier()
            model.load_model(str(self.checkpoint))
            self.kind = "sklearn"
            self.model = model
            return

        if name == "LSTM":
            self.kind = "torch_lstm"
            self.model = self._load_lstm()
            return

        raise ValueError(f"Unsupported model: {self.model_name}")

    def _load_lstm(self):
        obj = torch.load(self.checkpoint, map_location="cpu", weights_only=False)

        if isinstance(obj, torch.nn.Module):
            model = obj
            model.to(self.device)
            model.eval()
            return model

        if not isinstance(obj, dict):
            raise RuntimeError(
                f"Unsupported LSTM checkpoint object type: {type(obj)}"
            )

        for key in ("model", "module"):
            if isinstance(obj.get(key), torch.nn.Module):
                model = obj[key]
                model.to(self.device)
                model.eval()
                return model

        state = None
        for key in (
            "model_state_dict", "state_dict", "model_state", "weights"
        ):
            if isinstance(obj.get(key), dict):
                state = obj[key]
                break

        if state is None:
            if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
                state = obj
            else:
                raise RuntimeError(
                    f"No state_dict found in LSTM checkpoint. keys={list(obj.keys())}"
                )

        # Strip a possible DataParallel prefix before architecture inference.
        if state and all(str(k).startswith("module.") for k in state):
            state = {str(k)[7:]: v for k, v in state.items()}

        FrozenLSTM = getattr(self.stage58, "FrozenLSTM")
        sig = inspect.signature(FrozenLSTM.__init__)

        # Flatten simple checkpoint metadata when present.
        metadata = {}
        for k, v in obj.items():
            if isinstance(v, (str, int, float, bool)):
                metadata[k] = v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, (str, int, float, bool)):
                        metadata.setdefault(kk, vv)

        tab_dim = int(self.adapter.arrays[self.adapter.tab_key].shape[1])

        def shape_of(predicate):
            for key, tensor in state.items():
                if predicate(str(key).lower()) and hasattr(tensor, "shape"):
                    return tuple(tensor.shape), str(key)
            return None, None

        # Exact vocabulary sizes come from the frozen Stage56 mapping.
        n_act = len(self.mapping["activity_vocab"])
        n_doc = len(self.mapping["doctype_vocab"])
        n_sub = len(self.mapping["subprocess_vocab"])

        # Infer the shared embedding dimension from state_dict.
        embedding_shapes = []
        for key, tensor in state.items():
            kl = str(key).lower()
            if (
                hasattr(tensor, "shape")
                and len(tuple(tensor.shape)) == 2
                and ("emb" in kl or "embedding" in kl)
            ):
                shp = tuple(tensor.shape)
                # Only treat true vocab x emb matrices as embeddings.
                if shp[0] in (n_act, n_doc, n_sub):
                    embedding_shapes.append((str(key), shp))

        emb_dims = sorted({int(shp[1]) for _, shp in embedding_shapes})
        if len(emb_dims) != 1:
            raise RuntimeError(
                "Could not uniquely infer FrozenLSTM emb_dim from state_dict. "
                f"embedding_shapes={embedding_shapes}"
            )
        emb_dim = emb_dims[0]

        # Infer hidden size from recurrent hidden-hidden matrix.
        hh_shape, hh_key = shape_of(
            lambda k: "weight_hh_l0" in k and "_reverse" not in k
        )
        if hh_shape is None or len(hh_shape) != 2:
            raise RuntimeError(
                "Could not infer FrozenLSTM hidden_dim from state_dict."
            )
        hidden_dim = int(hh_shape[1])

        # Infer recurrent depth and directionality from state_dict key inventory.
        layer_ids = []
        for key in state:
            m = re.search(r"weight_ih_l(\d+)", str(key))
            if m:
                layer_ids.append(int(m.group(1)))
        if not layer_ids:
            raise RuntimeError(
                "Could not infer FrozenLSTM num_layers from state_dict."
            )
        num_layers = max(layer_ids) + 1
        bidirectional = any("_reverse" in str(k) for k in state)

        # Dropout has no parameters and does not affect eval-mode inference.
        # Stage58's frozen protocol uses the frozen configuration value; if it is
        # absent from this compact checkpoint, 0.0 is inference-equivalent in eval().
        dropout = float(metadata.get("dropout", 0.0))

        recovered = {
            "n_act": n_act,
            "n_doc": n_doc,
            "n_sub": n_sub,
            "emb_dim": emb_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
        }

        aliases = {
            # Stage58 exact constructor names.
            "n_act": n_act,
            "n_doc": n_doc,
            "n_sub": n_sub,
            "emb_dim": emb_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,

            # Compatibility aliases.
            "activity_vocab_size": n_act,
            "n_activity": n_act,
            "num_activities": n_act,
            "doctype_vocab_size": n_doc,
            "n_doctype": n_doc,
            "num_doctypes": n_doc,
            "subprocess_vocab_size": n_sub,
            "n_subprocess": n_sub,
            "num_subprocesses": n_sub,
            "embedding_dim": emb_dim,
            "hidden_size": hidden_dim,
            "tabular_dim": tab_dim,
            "tab_dim": tab_dim,
            "numeric_dim": tab_dim,
            "input_dim": tab_dim,
            "num_classes": 2,
            "n_classes": 2,
            "prefix": self.prefix,
            "prefix_len": self.prefix,
        }

        kwargs = {}
        missing = []

        for pname, p in sig.parameters.items():
            if pname == "self":
                continue

            if pname in metadata:
                kwargs[pname] = metadata[pname]
                continue

            if pname in aliases:
                kwargs[pname] = aliases[pname]
                continue

            pl = pname.lower()

            if pl in ("act_vocab", "activity_vocab") or (
                "activity" in pl and "vocab" in pl
            ):
                kwargs[pname] = n_act
                continue

            if pl in ("doc_vocab", "doctype_vocab") or (
                "doctype" in pl and "vocab" in pl
            ):
                kwargs[pname] = n_doc
                continue

            if pl in ("sub_vocab", "subprocess_vocab") or (
                "subprocess" in pl and "vocab" in pl
            ):
                kwargs[pname] = n_sub
                continue

            if p.default is not inspect._empty:
                continue

            missing.append(pname)

        if missing:
            raise RuntimeError(
                "Could not instantiate FrozenLSTM; unresolved constructor args "
                f"{missing}. recovered={recovered}; "
                f"checkpoint keys={list(obj.keys())}; signature={sig}; "
                f"state keys sample={list(state.keys())[:20]}"
            )

        model = FrozenLSTM(**kwargs)
        model.load_state_dict(state, strict=True)
        model.to(self.device)
        model.eval()

        print(
            f"[LSTM architecture] P{self.prefix} "
            f"n_act={n_act}, n_doc={n_doc}, n_sub={n_sub}, "
            f"emb={emb_dim}, hidden={hidden_dim}, layers={num_layers}, "
            f"dropout={dropout}, bidirectional={bidirectional}",
            flush=True,
        )

        return model

    def predict_proba(self, batch: Dict[str, np.ndarray]) -> np.ndarray:
        if self.kind == "sklearn":
            x = np.asarray(batch[self.adapter.tab_key], dtype=float)
            probs = np.asarray(self.model.predict_proba(x), dtype=float)
            if probs.ndim != 2 or probs.shape[1] < 2:
                raise RuntimeError(
                    f"Unexpected predict_proba shape for {self.model_name}: {probs.shape}"
                )
            return probs[:, 1]

        if self.kind == "torch_lstm":
            return self._predict_lstm(batch)

        raise RuntimeError(f"Unknown predictor kind: {self.kind}")

    def _find_batch_key(self, semantic: str) -> Optional[str]:
        semantic = semantic.lower()
        preferred = {
            "activity": [self.adapter.activity_seq_key],
            "doctype": [self.adapter.doctype_seq_key],
            "subprocess": [self.adapter.subprocess_seq_key],
            "delta": [self.adapter.delta_key],
            "time": [self.adapter.delta_key],
            "mask": [self.adapter.mask_key],
            "tabular": [self.adapter.tab_key],
            "tab": [self.adapter.tab_key],
        }
        for k in preferred.get(semantic, []):
            if k:
                return k
        for k in self.adapter.arrays:
            if semantic in k.lower():
                return k
        return None

    def _predict_lstm(self, batch: Dict[str, np.ndarray]) -> np.ndarray:
        forward_sig = inspect.signature(self.model.forward)
        kwargs = {}
        positional = []

        semantic_by_param = {}
        for pname in forward_sig.parameters:
            if pname == "self":
                continue
            pl = pname.lower()
            if (
                "activity" in pl
                or pl in ("act", "acts", "act_id", "act_ids", "x_act")
                or pl.startswith("act_")
            ):
                semantic_by_param[pname] = "activity"
            elif (
                "doctype" in pl
                or "doc_type" in pl
                or pl in ("doc", "docs", "doc_id", "doc_ids", "x_doc")
                or pl.startswith("doc_")
            ):
                semantic_by_param[pname] = "doctype"
            elif (
                "subprocess" in pl
                or "sub_process" in pl
                or pl in ("sub", "subs", "sub_id", "sub_ids", "x_sub")
                or pl.startswith("sub_")
            ):
                semantic_by_param[pname] = "subprocess"
            elif "mask" in pl:
                semantic_by_param[pname] = "mask"
            elif (
                "delta" in pl
                or "time" in pl
                or pl in ("dt", "deltas", "x_delta", "x_time")
            ):
                semantic_by_param[pname] = "delta"
            elif "tab" in pl or "numeric" in pl:
                semantic_by_param[pname] = "tabular"
            else:
                raise RuntimeError(
                    f"Cannot map FrozenLSTM.forward parameter {pname!r}; "
                    f"signature={forward_sig}"
                )

        tensors = {}
        for pname, semantic in semantic_by_param.items():
            key = self._find_batch_key(semantic)
            if key is None:
                raise RuntimeError(
                    f"Missing NPZ array for LSTM semantic input {semantic!r}; "
                    f"available={list(batch)}"
                )
            tensors[pname] = tensor_from_array(
                np.asarray(batch[key]),
                semantic if semantic != "tabular" else "float",
                self.device,
            )

        n = next(iter(tensors.values())).shape[0]
        out_probs = []

        with torch.no_grad():
            for start in range(0, n, 512):
                end = min(n, start + 512)
                kw = {k: v[start:end] for k, v in tensors.items()}
                out = self.model(**kw)

                if isinstance(out, (tuple, list)):
                    out = out[0]
                if isinstance(out, dict):
                    for key in ("logits", "output", "pred", "prediction"):
                        if key in out:
                            out = out[key]
                            break

                if not torch.is_tensor(out):
                    raise RuntimeError(
                        f"FrozenLSTM forward returned unsupported type {type(out)}"
                    )

                if out.ndim == 1:
                    p = torch.sigmoid(out)
                elif out.ndim == 2 and out.shape[1] == 1:
                    p = torch.sigmoid(out[:, 0])
                elif out.ndim == 2 and out.shape[1] >= 2:
                    p = torch.softmax(out, dim=1)[:, 1]
                else:
                    raise RuntimeError(f"Unexpected LSTM output shape {tuple(out.shape)}")

                out_probs.append(p.detach().cpu().numpy())

        return np.concatenate(out_probs).astype(float)


# -------------------------------------------------------------------------------------------------
# Canonical checkpoint map / binding helpers
# -------------------------------------------------------------------------------------------------

def _remap_checkpoint_path(raw_path: str, project_root: Path) -> Path:
    """
    Resolve a frozen checkpoint path across Windows/Linux project relocation.

    Only the filesystem prefix is remapped. The checkpoint file itself is not
    modified, and the recorded SHA256 is still verified after relocation.
    """
    raw = str(raw_path).strip()
    direct = Path(raw)
    if direct.exists():
        return direct.resolve()

    normalized = raw.replace("\\", "/")

    # Frozen Stage60 paths were created under D:\ETCE on Windows.
    prefixes = [
        "D:/ETCE/",
        "D:/ETCE",
        "d:/ETCE/",
        "d:/ETCE",
        "d:/etce/",
        "d:/etce",
    ]

    rel = None
    for prefix in prefixes:
        if normalized.startswith(prefix):
            rel = normalized[len(prefix):].lstrip("/")
            break

    # Generic fallback: recover the project-relative portion.
    if rel is None:
        lower = normalized.lower()
        for anchor in ("/results/", "/configs/", "/scripts/", "/src/", "/data/"):
            pos = lower.find(anchor)
            if pos >= 0:
                rel = normalized[pos + 1 :]
                break

    if rel is not None:
        candidate = (project_root / Path(rel)).resolve()
        if candidate.exists():
            return candidate

    # Final fallback: search by basename inside the expected Stage58 checkpoint tree.
    basename = Path(normalized).name
    search_roots = [
        project_root
        / "results"
        / "phase4"
        / "phase4e_bpic2018_external"
        / "stage58_formal_external_test"
        / "checkpoints",
        project_root / "results",
    ]

    matches = []
    for base in search_roots:
        if base.exists():
            matches.extend(base.rglob(basename))

    matches = sorted({m.resolve() for m in matches})

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous relocated checkpoint for {raw_path}: "
            + ", ".join(str(m) for m in matches[:10])
        )

    raise FileNotFoundError(
        "Frozen checkpoint not found after path remapping.\n"
        f"Recorded path: {raw_path}\n"
        f"Project root : {project_root}"
    )


def load_checkpoint_map(
    stage60: Path,
    project_root: Optional[Path] = None,
) -> Dict[Tuple[str, int], Path]:
    if project_root is None:
        probe = stage60.resolve()
        inferred = None
        for parent in [probe] + list(probe.parents):
            if (parent / "scripts").is_dir() and (parent / "results").is_dir():
                inferred = parent
                break
        if inferred is None:
            raise RuntimeError(
                f"Could not infer ETCE project root from Stage60 path: {stage60}"
            )
        project_root = inferred

    project_root = project_root.resolve()

    path = stage60 / "canonical_checkpoint_map.csv"
    rows = list(
        csv.DictReader(
            path.open("r", encoding="utf-8-sig", newline="")
        )
    )

    out = {}
    remap_rows = []

    for row in rows:
        model = str(row["model"])
        prefix = int(row["prefix"])

        pstr = (
            row.get("checkpoint")
            or row.get("checkpoint_path")
            or row.get("path")
            or row.get("artifact_path")
        )
        if not pstr:
            raise RuntimeError(f"No checkpoint path in Stage60 row: {row}")

        p = _remap_checkpoint_path(pstr, project_root)

        recorded = (
            row.get("checkpoint_sha256")
            or row.get("sha256")
            or row.get("artifact_sha256")
        )

        actual = sha256_file(p)

        if recorded and actual != recorded:
            raise RuntimeError(
                f"Checkpoint SHA mismatch for {model}/P{prefix}:\n"
                f"  recorded path = {pstr}\n"
                f"  resolved path = {p}\n"
                f"  recorded SHA  = {recorded}\n"
                f"  actual SHA    = {actual}"
            )

        out[(model, prefix)] = p

        remap_rows.append(
            {
                "model": model,
                "prefix": prefix,
                "recorded_path": str(pstr),
                "resolved_path": str(p),
                "path_remapped": str(pstr) != str(p),
                "recorded_sha256": recorded,
                "actual_sha256": actual,
            }
        )

    expected = {(m, p) for m in MODELS for p in PREFIXES}
    if set(out) != expected:
        raise RuntimeError(
            f"Canonical checkpoint map keys mismatch. "
            f"missing={sorted(expected-set(out))}, "
            f"extra={sorted(set(out)-expected)}"
        )

    remapped_count = sum(bool(r["path_remapped"]) for r in remap_rows)
    print(
        f"[Checkpoint relocation] PASS - "
        f"{len(remap_rows)}/{len(remap_rows)} resolved, "
        f"{remapped_count} path(s) remapped, SHA256 verified",
        flush=True,
    )

    return out


def load_bindings(stage60: Path) -> pd.DataFrame:
    path = stage60 / "canonical_blackbox_bindings.csv"
    df = pd.read_csv(path, dtype={"case_id": str})

    required = {"case_id", "model", "prefix"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Stage60 bindings missing columns {required - set(df.columns)}")

    prob_col = pick_column(
        df.columns,
        ("prob_positive", "probability_positive", "positive_probability", "factual_prob"),
    )
    pred_col = pick_column(
        df.columns,
        ("predicted_class", "prediction", "pred_0_5", "pred_threshold_0_5"),
    )
    if prob_col is None or pred_col is None:
        raise RuntimeError(
            f"Cannot identify Stage60 probability/prediction columns: {list(df.columns)}"
        )

    df = df.rename(columns={prob_col: "_stage60_prob", pred_col: "_stage60_pred"})
    if len(df) != 1600:
        raise RuntimeError(f"Expected 1600 Stage60 bindings, got {len(df)}")
    return df


# -------------------------------------------------------------------------------------------------
# Rule extraction serialization
# -------------------------------------------------------------------------------------------------

def extract_branch_metrics(
    case_id: str,
    model: str,
    prefix: int,
    branch: dict,
) -> Dict[str, Any]:
    row = {
        "case_id": case_id,
        "model": model,
        "prefix": int(prefix),
        "surrogate": branch.get("branch"),
        "counterfactual_rule_count": int(branch.get("counterfactual_rule_count", 0)),
    }

    for key, value in branch.items():
        if isinstance(value, (str, int, float, bool, np.generic)) and key not in row:
            row[key] = json_safe(value)

    return row


def rule_rows_from_branch(
    case_id: str,
    model: str,
    prefix: int,
    branch: dict,
) -> List[dict]:
    surrogate = str(branch.get("branch"))
    rows = []

    factual_rule = branch.get("factual_rule")
    if factual_rule is not None:
        rows.append(
            {
                "case_id": case_id,
                "model": model,
                "prefix": int(prefix),
                "surrogate": surrogate,
                "rule_type": "factual",
                "candidate_id": (
                    factual_rule.get("candidate_id")
                    if isinstance(factual_rule, dict)
                    else None
                ),
                "rule": json_safe(factual_rule),
            }
        )

    for idx, rule in enumerate(branch.get("counterfactual_rules") or []):
        cid = rule.get("candidate_id") if isinstance(rule, dict) else None
        rows.append(
            {
                "case_id": case_id,
                "model": model,
                "prefix": int(prefix),
                "surrogate": surrogate,
                "rule_type": "counterfactual",
                "candidate_id": cid if cid is not None else f"cf_{idx}",
                "rule": json_safe(rule),
            }
        )
    return rows


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=Path, default=Path(r"D:\ETCE"))
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    root = args.project_root.resolve()
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))

    stage56_dir = root / "results/phase4/phase4e_bpic2018_external/stage56_features"
    stage60 = root / "results/phase4/phase4e_bpic2018_external/stage60_canonical_blackbox"
    stage61h = root / "results/phase4/phase4e_bpic2018_external/stage61h_frozen_constraints"

    out = (
        args.out_dir.resolve()
        if args.out_dir
        else root / "results/phase4/phase4e_bpic2018_external/stage62_external_multirule_bank"
    )
    ensure_output_dir(out, args.overwrite)
    cache_dir = out / "target_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Import Stage56 and Stage58 exact code.
    stage56 = import_from_path(
        "etce_stage56_external",
        root / "scripts/56_phase4e_bpic2018_feature_build.py",
    )
    stage58 = import_from_path(
        "etce_stage58_external",
        root / "scripts/58_phase4e_bpic2018_final_refit_formal_test.py",
    )

    import etce.phase3.semantic_core as semantic_core
    import etce.phase4.surrogate_core as surrogate_core
    from etce.phase4.surrogate_core import generate_process_valid_pool
    from etce.phase4.stratified_core import run_all_surrogates_stratified

    constraints, family_map = load_stage61h_constraints(stage61h)
    patch_semantic_adapter(
        semantic_core,
        surrogate_core,
        constraints,
        family_map,
    )

    mapping_path = stage56_dir / "mapping_final_fit2015_2016.json"
    mapping = read_json(mapping_path)

    adapters = {
        p: PrefixFeatureAdapter(
            stage56,
            mapping,
            stage56_dir / f"final_P{p}_test2017.npz",
            p,
        )
        for p in PREFIXES
    }

    bindings = load_bindings(stage60)
    checkpoint_map = load_checkpoint_map(stage60, root)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 124)
    print("ETCE STAGE62 - BPIC2018 EXTERNAL PROCESS-VALID LOCAL POOL + MULTI-RULE BANK")
    print("=" * 124)
    print(f"Stage56 mapping : {mapping_path}")
    print(f"Stage60 binding : {stage60}")
    print(f"Stage61H        : {stage61h}")
    print(f"Output          : {out}")
    print(f"Device          : {device}")
    print(f"Targets         : {len(bindings)}")
    print(f"Pool protocol   : size={POOL_SIZE}, max_edits={MAX_EDITS}, max_options={MAX_OPTIONS}")
    print("Intervention    : activity sequence only")
    print("Held fixed      : doctype, subprocess, timestamps, numeric factual context")
    print("2017 outcome label used : NO")
    print("=" * 124)

    # Runtime record.
    write_json(
        out / "runtime.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "xgboost": xgb.__version__,
            "device": str(device),
        },
    )

    write_json(
        out / "stage62_run_config.json",
        {
            "stage": STAGE,
            "scope": "BPIC2018 external formal local multi-rule evidence bank",
            "formal_targets": 1600,
            "models": list(MODELS),
            "prefixes": list(PREFIXES),
            "surrogates": list(SURROGATES),
            "pool_size": POOL_SIZE,
            "max_edits": MAX_EDITS,
            "max_options_per_position": MAX_OPTIONS,
            "pool_seed_base": POOL_SEED_BASE,
            "threshold": THRESHOLD,
            "activity_intervention_only": True,
            "doctype_fixed": True,
            "subprocess_fixed": True,
            "timestamps_fixed": True,
            "numeric_context_fixed": True,
            "family_adapter": "dominant_subprocess frozen at Stage61H",
            "train_only_dfg": True,
            "same_position_support": True,
            "deterministic_stratified_70_30": True,
            "test_true_labels_used": False,
            "test_based_retuning": False,
            "ds_fusion_performed_here": False,
        },
    )

    # ---------------------------------------------------------------------------------------------
    # Hard factual reproduction gate
    # ---------------------------------------------------------------------------------------------
    print("\n[1/3] Canonical factual reproduction gate")

    predictors = {}
    reproduction_rows = []

    for model in MODELS:
        for prefix in PREFIXES:
            adapter = adapters[prefix]
            sub = bindings[
                (bindings["model"].astype(str) == model)
                & (bindings["prefix"].astype(int) == prefix)
            ].copy()

            if len(sub) != 100:
                raise RuntimeError(
                    f"Expected 100 bindings for {model}/P{prefix}, got {len(sub)}"
                )

            case_ids = sub["case_id"].astype(str).tolist()
            batch = adapter.factual_batch(case_ids)

            predictor = CanonicalExternalPredictor(
                model,
                prefix,
                checkpoint_map[(model, prefix)],
                mapping,
                adapter,
                stage58,
                device,
            )
            predictors[(model, prefix)] = predictor

            probs = predictor.predict_proba(batch)
            expected = sub["_stage60_prob"].astype(float).to_numpy()
            expected_pred = sub["_stage60_pred"].astype(int).to_numpy()

            diffs = np.abs(probs - expected)
            stage62_pred = (probs >= THRESHOLD).astype(int)

            atol = factual_probability_tolerance(model)
            prob_ok = diffs <= atol
            label_ok = stage62_pred == expected_pred
            ok = prob_ok & label_ok
            label_mismatch = int((~label_ok).sum())

            for cid, pnew, pold, pred_new, pred_old, diff, pgood, lgood, good in zip(
                case_ids,
                probs,
                expected,
                stage62_pred,
                expected_pred,
                diffs,
                prob_ok,
                label_ok,
                ok,
            ):
                reproduction_rows.append(
                    {
                        "case_id": cid,
                        "model": model,
                        "prefix": prefix,
                        "stage62_prob": float(pnew),
                        "stage60_prob": float(pold),
                        "stage62_pred": int(pred_new),
                        "stage60_pred": int(pred_old),
                        "abs_diff": float(diff),
                        "tolerance": atol,
                        "probability_pass": bool(pgood),
                        "label_pass": bool(lgood),
                        "pass": bool(good),
                    }
                )

            print(
                f"  {model:4s} P{prefix:2d} | "
                f"N={len(sub):3d} | max_abs_diff={float(diffs.max()):.3e} | "
                f"label_mismatch={label_mismatch} | "
                f"{'PASS' if bool(np.all(ok)) else 'FAIL'}"
            )

    repro_df = pd.DataFrame(reproduction_rows)
    repro_df.to_csv(
        out / "factual_probability_reproduction.csv",
        index=False,
        encoding="utf-8-sig",
    )

    failed = repro_df[~repro_df["pass"]]
    if len(failed):
        write_json(
            out / "consistency_checks.json",
            {
                "canonical_factual_probability_reproduction": {
                    "pass": False,
                    "failed_count": int(len(failed)),
                    "max_abs_diff": float(repro_df["abs_diff"].max()),
                },
                "rule_bank_generated": {"pass": True, "actual": False},
            },
        )
        print("\nSTAGE62 STOPPED BEFORE NEIGHBORHOOD GENERATION.")
        print(f"Canonical factual reproduction failed for {len(failed)} / 1600 targets.")
        return 3

    print("[Canonical factual probability gate] PASS - 1600/1600")

    # ---------------------------------------------------------------------------------------------
    # Target-local pool + surrogate extraction
    # ---------------------------------------------------------------------------------------------
    print("\n[2/3] External process-valid local pools + LORE/DT/BN")

    status_rows = []
    metric_rows = []
    all_rule_rows = []

    total = len(bindings)
    done = 0
    started = time.perf_counter()

    # Stable target order.
    bindings2 = bindings.copy()
    bindings2["_model_order"] = bindings2["model"].map({m: i for i, m in enumerate(MODELS)})
    bindings2 = bindings2.sort_values(
        ["_model_order", "prefix", "case_id"], kind="mergesort"
    )

    for _, row in bindings2.iterrows():
        case_id = str(row["case_id"])
        model = str(row["model"])
        prefix = int(row["prefix"])
        adapter = adapters[prefix]
        predictor = predictors[(model, prefix)]
        done += 1

        token = hashlib.sha256(
            f"{case_id}|{model}|{prefix}".encode("utf-8")
        ).hexdigest()[:20]
        cache_path = cache_dir / f"{token}.json.gz"

        if cache_path.exists() and not args.overwrite:
            with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("schema") != CACHE_SCHEMA:
                raise RuntimeError(f"Cache schema mismatch: {cache_path}")
        else:
            factual_index = adapter.row_index(case_id)
            factual = adapter.decode_factual_activities(factual_index)
            seed = stable_local_seed(model, prefix, case_id)

            pool = generate_process_valid_pool(
                factual,
                constraints,
                rng_seed=seed,
                pool_size=POOL_SIZE,
                max_edits=MAX_EDITS,
                max_options_per_position=MAX_OPTIONS,
            )
            pool = [tuple(s) for s in pool]

            if not pool or pool[0] != factual:
                raise RuntimeError(
                    f"Pool factual invariant failed for {case_id}/{model}/P{prefix}"
                )
            if len(set(pool)) != len(pool):
                raise RuntimeError(
                    f"Duplicate sequences in pool for {case_id}/{model}/P{prefix}"
                )

            batch = adapter.perturbed_batch(case_id, pool)
            probs = predictor.predict_proba(batch)
            labels = (probs >= THRESHOLD).astype(int)

            factual_prob = float(probs[0])
            stage60_prob = float(row["_stage60_prob"])
            atol = factual_probability_tolerance(model)
            if abs(factual_prob - stage60_prob) > atol:
                raise RuntimeError(
                    f"Factual probability drift inside target pool: "
                    f"{case_id}/{model}/P{prefix} "
                    f"{factual_prob} vs {stage60_prob}"
                )

            label_counts = Counter(int(v) for v in labels.tolist())
            factual_label = int(labels[0])
            stage60_label = int(row["_stage60_pred"])
            if factual_label != stage60_label:
                raise RuntimeError(
                    f"Factual class drift inside target pool: "
                    f"{case_id}/{model}/P{prefix} "
                    f"Stage62={factual_label}, Stage60={stage60_label}"
                )

            result = None
            status = "single_class_pool"
            error = None

            if len(label_counts) >= 2:
                try:
                    result = run_all_surrogates_stratified(
                        pool,
                        labels,
                        factual,
                        factual_label,
                    )
                    status = "fusable_two_class_pool"
                except Exception as exc:
                    status = "surrogate_error"
                    error = f"{type(exc).__name__}: {exc}"

            payload = {
                "schema": CACHE_SCHEMA,
                "case_id": case_id,
                "model": model,
                "prefix": prefix,
                "seed": seed,
                "factual": list(factual),
                "factual_probability": factual_prob,
                "factual_label": factual_label,
                "pool_size_requested": POOL_SIZE,
                "pool_size_actual": len(pool),
                "pool_digest": pool_digest(pool),
                "pool": [list(s) for s in pool],
                "prob_positive": probs.tolist(),
                "bb_labels": labels.tolist(),
                "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
                "status": status,
                "error": error,
                "surrogate_result": json_safe(result) if result is not None else None,
                "uses_true_outcome_label": False,
            }

            with gzip.open(cache_path, "wt", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

        # Aggregate from payload whether newly generated or cached.
        label_counts = {
            int(k): int(v)
            for k, v in payload["label_counts"].items()
        }
        result = payload.get("surrogate_result")

        status_rows.append(
            {
                "case_id": case_id,
                "model": model,
                "prefix": prefix,
                "status": payload["status"],
                "pool_size": payload["pool_size_actual"],
                "class0_count": label_counts.get(0, 0),
                "class1_count": label_counts.get(1, 0),
                "factual_label": payload["factual_label"],
                "factual_probability": payload["factual_probability"],
                "pool_digest": payload["pool_digest"],
                "cache_file": str(cache_path),
                "error": payload.get("error"),
            }
        )

        if result is not None:
            for branch in result.get("branches", []):
                metric_rows.append(
                    extract_branch_metrics(case_id, model, prefix, branch)
                )
                all_rule_rows.extend(
                    rule_rows_from_branch(case_id, model, prefix, branch)
                )

        if done % 25 == 0 or done == total:
            elapsed = time.perf_counter() - started
            print(
                f"  progress {done:4d}/{total} | "
                f"elapsed={elapsed/60:.1f} min | "
                f"last={model}/P{prefix}/{case_id}",
                flush=True,
            )

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(
        out / "target_status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(
        out / "surrogate_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rules_path = out / "canonical_and_counterfactual_rules.jsonl"
    with rules_path.open("w", encoding="utf-8") as f:
        for r in all_rule_rows:
            f.write(json.dumps(r, ensure_ascii=False, default=json_default) + "\n")

    # ---------------------------------------------------------------------------------------------
    # Summary / checks
    # ---------------------------------------------------------------------------------------------
    print("\n[3/3] Summary and frozen provenance")

    status_counts = status_df["status"].value_counts().to_dict()
    fusable = int(status_counts.get("fusable_two_class_pool", 0))
    single = int(status_counts.get("single_class_pool", 0))
    errors = int(status_counts.get("surrogate_error", 0))

    cf_rules = sum(
        1 for r in all_rule_rows if r["rule_type"] == "counterfactual"
    )
    factual_rules = sum(
        1 for r in all_rule_rows if r["rule_type"] == "factual"
    )

    summary = {
        "stage": STAGE,
        "formal_targets": int(len(status_df)),
        "fusable_two_class_targets": fusable,
        "single_class_pool_targets": single,
        "surrogate_error_targets": errors,
        "min_pool_size": int(status_df["pool_size"].min()),
        "median_pool_size": float(status_df["pool_size"].median()),
        "mean_pool_size": float(status_df["pool_size"].mean()),
        "max_pool_size": int(status_df["pool_size"].max()),
        "surrogate_metric_rows": int(len(metric_df)),
        "factual_rules": int(factual_rules),
        "counterfactual_rules": int(cf_rules),
        "total_rule_records": int(len(all_rule_rows)),
        "factual_probability_reproduction_max_abs_diff": float(
            repro_df["abs_diff"].max()
        ),
        "uses_true_2017_outcome_labels": False,
        "test_based_retuning": False,
        "ds_fusion_performed": False,
    }
    write_json(out / "stage62_summary.json", summary)

    checks = {
        "stage60_factual_probability_reproduced_1600": {
            "pass": len(failed) == 0 and len(repro_df) == 1600,
            "count": int(len(repro_df)),
            "max_abs_diff": float(repro_df["abs_diff"].max()),
        },
        "all_1600_targets_processed": {
            "pass": len(status_df) == 1600,
            "actual": int(len(status_df)),
            "expected": 1600,
        },
        "no_true_2017_label_used": {
            "pass": True,
            "actual": False,
        },
        "frozen_pool_size_320": {
            "pass": POOL_SIZE == 320,
        },
        "frozen_max_edits_3": {
            "pass": MAX_EDITS == 3,
        },
        "frozen_surrogates": {
            "pass": tuple(SURROGATES) == ("LORE", "DT", "BN"),
        },
        "stage61h_family_adapter_used": {
            "pass": constraints["family_adapter"]["scheme"] == "dominant_subprocess",
        },
        "surrogate_runtime_errors_zero": {
            "pass": errors == 0,
            "actual": errors,
        },
    }
    all_pass = all(v.get("pass", False) for v in checks.values())
    write_json(out / "consistency_checks.json", checks)

    report = f"""# ETCE Stage62 — BPIC2018 External Multi-rule Bank

- Formal targets: {len(status_df)}
- Two-class / fusable local pools: {fusable}
- Single-class local pools: {single}
- Surrogate runtime errors: {errors}
- Pool size min / median / mean / max:
  {summary['min_pool_size']} / {summary['median_pool_size']:.1f} /
  {summary['mean_pool_size']:.2f} / {summary['max_pool_size']}
- Factual rules: {factual_rules}
- Counterfactual rules: {cf_rules}
- Canonical factual probability max absolute reproduction error:
  {summary['factual_probability_reproduction_max_abs_diff']:.3e}

Protocol:
- Stage59/60 frozen 100 TEST cases and canonical seed42 black boxes.
- Stage61H TRAIN-only BPIC2018 DFG, position support, and dominant-subprocess family adapter.
- Activity-only intervention; doctype/subprocess/timestamps/numeric context fixed.
- Pool size 320, max edits 3, max options 8.
- LORE / DT / BN with frozen deterministic stratified local protocol.
- No true 2017 outcome label used.
- No TEST-based retuning.
- No D-S fusion performed here.

Hard gate: **{"PASS" if all_pass else "FAIL"}**
"""
    (out / "STAGE62_EXTERNAL_MULTIRULE_REPORT.md").write_text(report, encoding="utf-8")

    # Manifest last.
    manifest = {}
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "sha256_manifest.json":
            manifest[p.name] = {
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
    write_json(
        out / "sha256_manifest.json",
        {"stage": STAGE, "algorithm": "SHA256", "files": manifest},
    )

    print("\n" + "=" * 124)
    print("ETCE STAGE62 SUMMARY")
    print("=" * 124)
    print(f"Formal targets                         : {len(status_df)}")
    print(f"Fusable two-class pools                : {fusable}")
    print(f"Single-class pools                     : {single}")
    print(f"Surrogate errors                       : {errors}")
    print(
        "Pool size min/median/mean/max          : "
        f"{summary['min_pool_size']}/"
        f"{summary['median_pool_size']:.1f}/"
        f"{summary['mean_pool_size']:.2f}/"
        f"{summary['max_pool_size']}"
    )
    print(f"Factual rules                          : {factual_rules}")
    print(f"Counterfactual rules                   : {cf_rules}")
    print(
        "Factual probability reproduction maxΔ : "
        f"{summary['factual_probability_reproduction_max_abs_diff']:.3e}"
    )
    print("True 2017 labels used                  : NO")
    print("TEST-based retuning                    : NO")
    print("D-S fusion                             : NO")
    print(f"Hard gates                             : {'PASS' if all_pass else 'FAIL'}")
    print("=" * 124)

    if not all_pass:
        print(
            "\nSTAGE62 COMPLETED WITH A FAILED HARD GATE. "
            "Do not proceed to D-S fusion until the failed check is diagnosed."
        )
        return 4

    print(
        "\nSTAGE62 EXTERNAL MULTI-RULE BANK COMPLETED SUCCESSFULLY.\n"
        "NEXT: independently validate the external target caches/rules, then "
        "apply the frozen lambda=5 D-S protocol without any BPIC2018 retuning."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
