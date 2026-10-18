#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage57
BPIC2018 External Black-box Refit on Development Period

STRICT PROTOCOL
---------------
- Dataset: BPIC2018, Stage56 frozen representations.
- Models: RF / XGB / MLP / LSTM.
- Prefixes: P5 / P10 / P15 / P20.
- Formal seeds: [13, 21, 42, 73, 101].
- Prediction threshold: 0.5 (frozen from BPIC2017).
- Model hyperparameters: MUST be imported from the frozen BPIC2017
  `blackbox_best_by_model_prefix_final.json`.
- Stage57 performs NO hyperparameter search.
- 2015 is used for fitting.
- 2016 is used ONLY for development evaluation / checkpoint selection where
  the already-frozen neural protocol requires early stopping.
- 2017 formal external TEST files are NOT loaded at all in Stage57.

This stage writes:
  frozen_config_import.json
  development_metrics_seedwise.csv
  development_metrics_summary.csv
  development_predictions.csv.gz
  checkpoints/*
  consistency_checks.json
  protocol_frozen.json
  sha256_manifest.json

NOTE
----
The exact BPIC2017 frozen model configuration file is treated as authoritative.
This script accepts several common nesting layouts and records the exact extracted
parameter dict for every (model, prefix). If any pair cannot be resolved,
Stage57 aborts BEFORE training.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
        log_loss,
    )
except Exception as e:
    raise RuntimeError(
        "Stage57 requires scikit-learn. Install the same compatible environment "
        "used by the frozen ETCE black-box experiments."
    ) from e

try:
    import joblib
except Exception as e:
    raise RuntimeError("Stage57 requires joblib.") from e

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


STAGE = 57
MODELS = ("RF", "XGB", "MLP", "LSTM")
PREFIXES = (5, 10, 15, 20)
SEEDS = (13, 21, 42, 73, 101)
THRESHOLD = 0.5


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
                "Use --overwrite only for an exact protocol rebuild."
            )
        if overwrite:
            import shutil
            for p in entries:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # reproducibility-oriented settings
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def finite_or_none(x: float) -> Optional[float]:
    return None if (x is None or not np.isfinite(x)) else float(x)


# ---------------------------------------------------------------------
# Stage56 gate
# ---------------------------------------------------------------------

def verify_stage56(stage56_dir: Path) -> Dict[str, Any]:
    required = [
        "protocol_frozen.json",
        "dataset_summary.json",
        "consistency_checks.json",
        "sha256_manifest.json",
        "mapping_development_fit2015.json",
        "mapping_final_fit2015_2016.json",
    ]
    for name in required:
        p = stage56_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing Stage56 file: {p}")

    protocol = read_json(stage56_dir / "protocol_frozen.json")
    summary = read_json(stage56_dir / "dataset_summary.json")
    checks = read_json(stage56_dir / "consistency_checks.json")

    if protocol.get("stage") != 56:
        raise RuntimeError("Input is not Stage56.")
    if not protocol.get("all_hard_checks_pass", False):
        raise RuntimeError("Stage56 protocol not frozen/PASS.")
    if not summary.get("all_hard_checks_pass", False):
        raise RuntimeError("Stage56 summary not PASS.")
    failed = {k: v for k, v in checks.items() if not v.get("pass", False)}
    if failed:
        raise RuntimeError(f"Stage56 consistency checks failed: {list(failed)[:10]}")

    # Verify hashes for Stage56 files listed in its manifest.
    hm = read_json(stage56_dir / "sha256_manifest.json")
    for raw, meta in hm.get("files", {}).items():
        p = Path(raw)
        if p.exists():
            got = sha256_file(p)
            exp = meta.get("sha256")
            if got != exp:
                raise RuntimeError(f"Stage56 SHA256 mismatch: {p}")

    return {"protocol": protocol, "summary": summary}


# ---------------------------------------------------------------------
# Frozen config resolution
# ---------------------------------------------------------------------

def canonical_model_name(x: str) -> str:
    s = str(x).strip().lower().replace("-", "").replace("_", "")
    if s in ("rf", "randomforest", "randomforestclassifier"):
        return "RF"
    if s in ("xgb", "xgboost", "xgbclassifier"):
        return "XGB"
    if s in ("mlp", "mlpclassifier", "multilayerperceptron"):
        return "MLP"
    if s in ("lstm", "bilstm", "longshorttermmemory"):
        return "LSTM"
    return str(x).upper()


def prefix_from_any(x: Any) -> Optional[int]:
    if isinstance(x, int) and x in PREFIXES:
        return x
    s = str(x).strip().upper()
    if s.startswith("P"):
        s = s[1:]
    try:
        p = int(s)
        return p if p in PREFIXES else None
    except Exception:
        return None


def looks_like_param_dict(d: Dict[str, Any]) -> bool:
    # A param dict should mostly contain scalar/list values rather than nested model maps.
    if not isinstance(d, dict) or not d:
        return False
    nested = sum(isinstance(v, dict) for v in d.values())
    return nested <= max(1, len(d) // 3)


def recursive_collect_configs(
    obj: Any,
    path: Tuple[str, ...] = (),
    found: Optional[List[Tuple[Tuple[str, ...], Dict[str, Any]]]] = None,
) -> List[Tuple[Tuple[str, ...], Dict[str, Any]]]:
    if found is None:
        found = []
    if isinstance(obj, dict):
        if looks_like_param_dict(obj):
            found.append((path, obj))
        for k, v in obj.items():
            recursive_collect_configs(v, path + (str(k),), found)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            recursive_collect_configs(v, path + (str(i),), found)
    return found


def score_candidate_path(path: Tuple[str, ...], model: str, prefix: int) -> int:
    toks = [str(x) for x in path]
    score = 0
    for t in toks:
        if canonical_model_name(t) == model:
            score += 10
        if prefix_from_any(t) == prefix:
            score += 10
        tl = t.lower()
        if "best" in tl:
            score += 2
        if "param" in tl or "config" in tl:
            score += 2
    return score


def normalize_param_container(d: Dict[str, Any]) -> Dict[str, Any]:
    # common wrappers
    for k in ("params", "hyperparameters", "best_params", "config", "model_params"):
        if k in d and isinstance(d[k], dict):
            return dict(d[k])
    return dict(d)


def resolve_frozen_configs(config_obj: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    leaves = recursive_collect_configs(config_obj)
    resolved: Dict[str, Dict[str, Dict[str, Any]]] = {m: {} for m in MODELS}

    for model in MODELS:
        for p in PREFIXES:
            candidates = []
            for path, d in leaves:
                sc = score_candidate_path(path, model, p)
                if sc >= 20:
                    candidates.append((sc, len(path), path, d))
            if not candidates:
                continue
            candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
            _, _, path, d = candidates[0]
            resolved[model][str(p)] = {
                "source_path": list(path),
                "params": normalize_param_container(d),
            }

    missing = [
        f"{m}/P{p}"
        for m in MODELS
        for p in PREFIXES
        if str(p) not in resolved[m]
    ]
    if missing:
        # Try a direct common layout before aborting.
        for model in MODELS:
            aliases = [model, model.lower(), "XGBoost" if model == "XGB" else model]
            for alias in aliases:
                node = config_obj.get(alias)
                if not isinstance(node, dict):
                    continue
                for p in PREFIXES:
                    if str(p) in resolved[model]:
                        continue
                    for pk in (str(p), f"P{p}", f"p{p}"):
                        if pk in node and isinstance(node[pk], dict):
                            resolved[model][str(p)] = {
                                "source_path": [alias, pk],
                                "params": normalize_param_container(node[pk]),
                            }

        missing = [
            f"{m}/P{p}"
            for m in MODELS
            for p in PREFIXES
            if str(p) not in resolved[m]
        ]

    if missing:
        raise RuntimeError(
            "Could not resolve frozen BPIC2017 parameter dictionaries for: "
            + ", ".join(missing)
            + "\nStage57 aborts rather than inventing new BPIC2018 hyperparameters."
        )
    return resolved


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def binary_metrics(y: np.ndarray, prob: np.ndarray) -> Dict[str, Any]:
    pred = (prob >= THRESHOLD).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    out = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "positive_f1": f1_score(y, pred, pos_label=1, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = roc_auc_score(y, prob)
        out["pr_auc"] = average_precision_score(y, prob)
        out["log_loss"] = log_loss(y, np.vstack([1.0 - prob, prob]).T, labels=[0, 1])
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
        out["log_loss"] = float("nan")
    return out


# ---------------------------------------------------------------------
# Parameter adaptation
# ---------------------------------------------------------------------

def pick(d: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for n in names:
        if n in d:
            return d[n]
    return default


def clean_rf_params(raw: Dict[str, Any], seed: int) -> Dict[str, Any]:
    allowed = {
        "n_estimators", "criterion", "max_depth", "min_samples_split",
        "min_samples_leaf", "min_weight_fraction_leaf", "max_features",
        "max_leaf_nodes", "min_impurity_decrease", "bootstrap",
        "oob_score", "n_jobs", "random_state", "verbose", "warm_start",
        "class_weight", "ccp_alpha", "max_samples", "monotonic_cst",
    }
    p = {k: v for k, v in raw.items() if k in allowed}
    p["random_state"] = seed
    p.setdefault("n_jobs", -1)
    return p


def clean_xgb_params(raw: Dict[str, Any], seed: int) -> Dict[str, Any]:
    if xgb is None:
        raise RuntimeError("xgboost is not installed but XGB is required by frozen protocol.")
    # XGBClassifier tolerates many kwargs; strip obvious non-model controls.
    controls = {
        "seed", "epochs", "batch_size", "patience", "early_stopping_rounds",
        "threshold", "prefix", "model", "model_name",
    }
    p = {k: v for k, v in raw.items() if k not in controls}
    p["random_state"] = seed
    p.setdefault("n_jobs", -1)
    p.setdefault("eval_metric", "logloss")
    return p


def clean_mlp_params(raw: Dict[str, Any], seed: int) -> Dict[str, Any]:
    p = dict(raw)

    # Map common frozen naming variants.
    if "hidden_dims" in p and "hidden_layer_sizes" not in p:
        hd = p.pop("hidden_dims")
        p["hidden_layer_sizes"] = tuple(hd) if isinstance(hd, (list, tuple)) else (int(hd),)
    if "hidden_layers" in p and "hidden_layer_sizes" not in p:
        hd = p.pop("hidden_layers")
        p["hidden_layer_sizes"] = tuple(hd) if isinstance(hd, (list, tuple)) else (int(hd),)
    if "lr" in p and "learning_rate_init" not in p:
        p["learning_rate_init"] = p.pop("lr")

    # Frozen configs may use `learning_rate` to mean a numeric optimizer LR.
    # sklearn MLPClassifier instead reserves `learning_rate` for the schedule
    # string {'constant','invscaling','adaptive'}, so preserve semantics by
    # mapping numeric values to learning_rate_init.
    if "learning_rate" in p:
        lr_value = p["learning_rate"]
        if isinstance(lr_value, (int, float)):
            if "learning_rate_init" not in p:
                p["learning_rate_init"] = float(lr_value)
            p.pop("learning_rate", None)
        elif isinstance(lr_value, str):
            stripped = lr_value.strip()
            try:
                numeric_lr = float(stripped)
            except ValueError:
                p["learning_rate"] = stripped.lower()
            else:
                if "learning_rate_init" not in p:
                    p["learning_rate_init"] = numeric_lr
                p.pop("learning_rate", None)

    if "epochs" in p and "max_iter" not in p:
        p["max_iter"] = p.pop("epochs")
    if "weight_decay" in p and "alpha" not in p:
        p["alpha"] = p.pop("weight_decay")

    # Compatibility normalization only: scikit-learn expects lowercase enum strings.
    # This does NOT change the frozen hyperparameter semantics.
    if "activation" in p and isinstance(p["activation"], str):
        p["activation"] = p["activation"].strip().lower()
    if "solver" in p and isinstance(p["solver"], str):
        p["solver"] = p["solver"].strip().lower()

    allowed = {
        "hidden_layer_sizes", "activation", "solver", "alpha", "batch_size",
        "learning_rate", "learning_rate_init", "power_t", "max_iter",
        "shuffle", "random_state", "tol", "verbose", "warm_start",
        "momentum", "nesterovs_momentum", "early_stopping",
        "validation_fraction", "beta_1", "beta_2", "epsilon",
        "n_iter_no_change", "max_fun",
    }
    q = {k: v for k, v in p.items() if k in allowed}
    q["random_state"] = seed
    return q


# ---------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------

if nn is not None:
    class FrozenLSTM(nn.Module):
        def __init__(
            self,
            n_act: int,
            n_doc: int,
            n_sub: int,
            emb_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            bidirectional: bool,
        ) -> None:
            super().__init__()
            self.act_emb = nn.Embedding(n_act, emb_dim, padding_idx=0)
            self.doc_emb = nn.Embedding(n_doc, emb_dim, padding_idx=0)
            self.sub_emb = nn.Embedding(n_sub, emb_dim, padding_idx=0)
            in_dim = emb_dim * 3 + 1
            self.lstm = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            out_dim = hidden_dim * (2 if bidirectional else 1)
            self.head = nn.Linear(out_dim, 1)

        def forward(self, act, doc, sub, delta):
            x = torch.cat([
                self.act_emb(act),
                self.doc_emb(doc),
                self.sub_emb(sub),
                delta.unsqueeze(-1),
            ], dim=-1)
            out, _ = self.lstm(x)
            z = out[:, -1, :]
            return self.head(z).squeeze(-1)


def train_lstm(
    train: Dict[str, np.ndarray],
    valid: Dict[str, np.ndarray],
    mapping_meta: Dict[str, Any],
    raw: Dict[str, Any],
    seed: int,
    checkpoint_path: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is not installed but LSTM is required by frozen protocol.")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb_dim = int(pick(raw, "emb_dim", "embedding_dim", default=16))
    hidden_dim = int(pick(raw, "hidden_dim", "hidden_size", default=64))
    num_layers = int(pick(raw, "num_layers", "layers", default=1))
    dropout = float(pick(raw, "dropout", default=0.0))
    bidirectional = bool(pick(raw, "bidirectional", default=False))
    batch_size = int(pick(raw, "batch_size", default=128))
    lr = float(pick(raw, "lr", "learning_rate", default=1e-3))
    weight_decay = float(pick(raw, "weight_decay", default=0.0))
    epochs = int(pick(raw, "epochs", "max_epochs", default=50))
    patience = int(pick(raw, "patience", "early_stopping_patience", default=5))

    n_act = len(mapping_meta["activity_vocab"])
    n_doc = len(mapping_meta["doctype_vocab"])
    n_sub = len(mapping_meta["subprocess_vocab"])

    model = FrozenLSTM(
        n_act, n_doc, n_sub,
        emb_dim, hidden_dim, num_layers, dropout, bidirectional
    ).to(device)

    ytr = train["y"].astype(np.float32)
    pos = max(1.0, float(ytr.sum()))
    neg = max(1.0, float(len(ytr) - ytr.sum()))
    # Only use class weighting if the frozen config explicitly requests it.
    use_pos_weight = bool(pick(raw, "use_pos_weight", "class_weight_balanced", default=False))
    if use_pos_weight:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device)
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    opt_name = str(pick(raw, "optimizer", default="adam")).lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(
        torch.from_numpy(train["activity_seq"]).long(),
        torch.from_numpy(train["doctype_seq"]).long(),
        torch.from_numpy(train["subprocess_seq"]).long(),
        torch.from_numpy(train["delta_log1p_seq"]).float(),
        torch.from_numpy(ytr).float(),
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)

    def predict(data: Dict[str, np.ndarray]) -> np.ndarray:
        model.eval()
        probs = []
        bs = 1024
        with torch.no_grad():
            for i in range(0, len(data["y"]), bs):
                act = torch.from_numpy(data["activity_seq"][i:i+bs]).long().to(device)
                doc = torch.from_numpy(data["doctype_seq"][i:i+bs]).long().to(device)
                sub = torch.from_numpy(data["subprocess_seq"][i:i+bs]).long().to(device)
                delta = torch.from_numpy(data["delta_log1p_seq"][i:i+bs]).float().to(device)
                logits = model(act, doc, sub, delta)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs).astype(np.float64)

    # Frozen neural protocol: validation is allowed only for checkpoint selection
    # under an already-fixed architecture/optimizer.
    best_f1 = -1.0
    best_epoch = -1
    bad = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for act, doc, sub, delta, y in loader:
            act, doc, sub, delta, y = (
                act.to(device), doc.to(device), sub.to(device), delta.to(device), y.to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(act, doc, sub, delta)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        pv = predict(valid)
        mf1 = f1_score(valid["y"], (pv >= THRESHOLD).astype(np.int8),
                       average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_macro_f1": float(mf1)})

        if mf1 > best_f1 + 1e-12:
            best_f1 = float(mf1)
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_args": {
                        "n_act": n_act, "n_doc": n_doc, "n_sub": n_sub,
                        "emb_dim": emb_dim, "hidden_dim": hidden_dim,
                        "num_layers": num_layers, "dropout": dropout,
                        "bidirectional": bidirectional,
                    },
                    "seed": seed,
                    "threshold": THRESHOLD,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
        else:
            bad += 1
            if patience > 0 and bad >= patience:
                break

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    prob = predict(valid)
    return prob, {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_valid_macro_f1": best_f1,
        "history": history,
        "resolved_training_params": {
            "emb_dim": emb_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "patience": patience,
            "optimizer": opt_name,
            "use_pos_weight": use_pos_weight,
        },
    }


# ---------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------

def fit_predict_one(
    model_name: str,
    prefix: int,
    seed: int,
    raw_params: Dict[str, Any],
    train: Dict[str, np.ndarray],
    valid: Dict[str, np.ndarray],
    mapping_meta: Dict[str, Any],
    checkpoint_dir: Path,
) -> Tuple[np.ndarray, Dict[str, Any], Path]:
    set_seed(seed)

    if model_name == "RF":
        params = clean_rf_params(raw_params, seed)
        model = RandomForestClassifier(**params)
        model.fit(train["X_tabular"], train["y"])
        prob = model.predict_proba(valid["X_tabular"])[:, 1]
        ckpt = checkpoint_dir / f"RF_P{prefix}_seed{seed}.joblib"
        joblib.dump(model, ckpt)
        return prob.astype(np.float64), {"resolved_params": params}, ckpt

    if model_name == "XGB":
        params = clean_xgb_params(raw_params, seed)
        model = xgb.XGBClassifier(**params)
        model.fit(train["X_tabular"], train["y"])
        prob = model.predict_proba(valid["X_tabular"])[:, 1]
        ckpt = checkpoint_dir / f"XGB_P{prefix}_seed{seed}.json"
        model.save_model(str(ckpt))
        return prob.astype(np.float64), {"resolved_params": params}, ckpt

    if model_name == "MLP":
        params = clean_mlp_params(raw_params, seed)
        model = MLPClassifier(**params)
        model.fit(train["X_tabular"], train["y"])
        prob = model.predict_proba(valid["X_tabular"])[:, 1]
        ckpt = checkpoint_dir / f"MLP_P{prefix}_seed{seed}.joblib"
        joblib.dump(model, ckpt)
        return prob.astype(np.float64), {"resolved_params": params}, ckpt

    if model_name == "LSTM":
        ckpt = checkpoint_dir / f"LSTM_P{prefix}_seed{seed}.pt"
        prob, info = train_lstm(
            train, valid, mapping_meta, raw_params, seed, ckpt
        )
        return prob, info, ckpt

    raise ValueError(model_name)


# ---------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------

METRIC_KEYS = (
    "accuracy", "macro_f1", "positive_f1", "precision", "recall",
    "balanced_accuracy", "roc_auc", "pr_auc", "log_loss",
)


def t95_halfwidth(vals: Sequence[float]) -> float:
    xs = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if len(xs) < 2:
        return float("nan")
    # df=4 for five formal seeds -> t_0.975 = 2.776445...
    tcrit = 2.7764451051977987 if len(xs) == 5 else 1.96
    return tcrit * statistics.stdev(xs) / math.sqrt(len(xs))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="ETCE Stage57 - BPIC2018 development refit using frozen BPIC2017 black-box configs"
    )
    ap.add_argument("--project_root", type=Path, default=Path(r"D:\ETCE"))
    ap.add_argument("--stage56_dir", type=Path, default=None)
    ap.add_argument(
        "--frozen_config",
        type=Path,
        default=None,
        help="BPIC2017 frozen black-box config JSON. Default: configs/blackbox_best_by_model_prefix_final.json",
    )
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.project_root.resolve()
    stage56_dir = (
        args.stage56_dir.resolve()
        if args.stage56_dir
        else root / "results" / "phase4" / "phase4e_bpic2018_external" / "stage56_features"
    )
    frozen_config = (
        args.frozen_config.resolve()
        if args.frozen_config
        else root / "configs" / "blackbox_best_by_model_prefix_final.json"
    )
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else root / "results" / "phase4" / "phase4e_bpic2018_external" / "stage57_development_models"
    )

    ensure_clean_dir(out_dir, args.overwrite)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("ETCE STAGE57 - BPIC2018 DEVELOPMENT BLACK-BOX REFIT WITH FROZEN BPIC2017 CONFIGS")
    print("=" * 120)
    print(f"Stage56       : {stage56_dir}")
    print(f"Frozen config : {frozen_config}")
    print(f"Output        : {out_dir}")
    print(f"Seeds         : {list(SEEDS)}")
    print(f"Threshold     : {THRESHOLD}")
    print("Formal TEST   : 2017 WILL NOT BE LOADED")
    print("=" * 120)

    verify_stage56(stage56_dir)
    print("[Stage56 gate] PASS")

    if not frozen_config.exists():
        raise FileNotFoundError(
            f"Frozen BPIC2017 black-box configuration not found:\n  {frozen_config}\n"
            "Do NOT replace it with newly tuned BPIC2018 parameters."
        )

    cfg_obj = read_json(frozen_config)
    resolved = resolve_frozen_configs(cfg_obj)

    imported = {
        "source": str(frozen_config),
        "source_sha256": sha256_file(frozen_config),
        "resolved": resolved,
        "models": list(MODELS),
        "prefixes": list(PREFIXES),
        "seeds": list(SEEDS),
        "threshold": THRESHOLD,
    }
    write_json(out_dir / "frozen_config_import.json", imported)
    print("[Frozen BPIC2017 config] PASS - all 16 model/prefix configs resolved")

    mapping_dev = read_json(stage56_dir / "mapping_development_fit2015.json")

    metric_rows: List[Dict[str, Any]] = []
    pred_path = out_dir / "development_predictions.csv.gz"

    with gzip.open(pred_path, "wt", encoding="utf-8", newline="") as pf:
        pw = csv.writer(pf)
        pw.writerow([
            "case_id", "case_year", "prefix", "model", "seed",
            "y_true", "prob_positive", "pred_threshold_0_5"
        ])

        for p in PREFIXES:
            train_path = stage56_dir / f"development_P{p}_train2015.npz"
            valid_path = stage56_dir / f"development_P{p}_valid2016.npz"
            if not train_path.exists() or not valid_path.exists():
                raise FileNotFoundError(f"Missing Stage56 P{p} development files.")

            train = load_npz(train_path)
            valid = load_npz(valid_path)

            if set(np.unique(train["year"]).tolist()) != {2015}:
                raise RuntimeError(f"P{p} train data contains non-2015 years.")
            if set(np.unique(valid["year"]).tolist()) != {2016}:
                raise RuntimeError(f"P{p} validation data contains non-2016 years.")

            for model_name in MODELS:
                raw_params = resolved[model_name][str(p)]["params"]

                for seed in SEEDS:
                    print(f"[P{p}] {model_name} seed={seed} ...", flush=True)
                    t0 = time.time()

                    prob, info, ckpt = fit_predict_one(
                        model_name=model_name,
                        prefix=p,
                        seed=seed,
                        raw_params=raw_params,
                        train=train,
                        valid=valid,
                        mapping_meta=mapping_dev,
                        checkpoint_dir=checkpoint_dir,
                    )

                    if len(prob) != len(valid["y"]):
                        raise RuntimeError(
                            f"Prediction length mismatch {model_name} P{p} seed={seed}"
                        )
                    if not np.all(np.isfinite(prob)):
                        raise RuntimeError(
                            f"Non-finite probabilities {model_name} P{p} seed={seed}"
                        )

                    m = binary_metrics(valid["y"], prob)
                    row = {
                        "model": model_name,
                        "prefix": p,
                        "seed": seed,
                        "threshold": THRESHOLD,
                        "train_year": 2015,
                        "validation_year": 2016,
                        "n_train": int(len(train["y"])),
                        "n_valid": int(len(valid["y"])),
                        "positive_train": int(train["y"].sum()),
                        "positive_valid": int(valid["y"].sum()),
                        **m,
                        "checkpoint": str(ckpt),
                        "checkpoint_sha256": sha256_file(ckpt),
                        "runtime_sec": round(time.time() - t0, 3),
                    }
                    metric_rows.append(row)

                    pred = (prob >= THRESHOLD).astype(np.int8)
                    for cid, yy, pp, yh in zip(
                        valid["case_id"], valid["y"], prob, pred
                    ):
                        pw.writerow([
                            str(cid), 2016, p, model_name, seed,
                            int(yy), f"{float(pp):.12g}", int(yh)
                        ])

                    extra_path = out_dir / f"traininfo_{model_name}_P{p}_seed{seed}.json"
                    write_json(extra_path, info)

                    print(
                        f"    Macro-F1={m['macro_f1']:.6f} "
                        f"PR-AUC={m['pr_auc']:.6f} "
                        f"ROC-AUC={m['roc_auc']:.6f} "
                        f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}",
                        flush=True,
                    )

    # Seedwise metrics CSV.
    metrics_path = out_dir / "development_metrics_seedwise.csv"
    fields = [
        "model", "prefix", "seed", "threshold",
        "train_year", "validation_year",
        "n_train", "n_valid", "positive_train", "positive_valid",
        "accuracy", "macro_f1", "positive_f1", "precision", "recall",
        "balanced_accuracy", "roc_auc", "pr_auc", "log_loss",
        "tn", "fp", "fn", "tp",
        "checkpoint", "checkpoint_sha256", "runtime_sec",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in metric_rows:
            rr = dict(r)
            for k in METRIC_KEYS:
                rr[k] = finite_or_none(float(rr[k]))
            w.writerow(rr)

    # Summary across five formal seeds.
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in metric_rows:
        grouped[(r["model"], r["prefix"])].append(r)

    summary_rows = []
    for model_name in MODELS:
        for p in PREFIXES:
            rows = grouped[(model_name, p)]
            if len(rows) != len(SEEDS):
                raise RuntimeError(f"Missing formal seeds for {model_name} P{p}.")
            out = {"model": model_name, "prefix": p, "n_seeds": len(rows)}
            for mk in METRIC_KEYS:
                vals = [float(r[mk]) for r in rows]
                out[f"{mk}_mean"] = float(np.nanmean(vals))
                out[f"{mk}_std"] = float(np.nanstd(vals, ddof=1))
                out[f"{mk}_ci95_halfwidth"] = t95_halfwidth(vals)
            summary_rows.append(out)

    summary_path = out_dir / "development_metrics_summary.csv"
    sum_fields = ["model", "prefix", "n_seeds"]
    for mk in METRIC_KEYS:
        sum_fields += [
            f"{mk}_mean", f"{mk}_std", f"{mk}_ci95_halfwidth"
        ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(summary_rows)

    checks = {
        "stage56_gate": {"pass": True},
        "frozen_config_exists": {"pass": frozen_config.exists(), "path": str(frozen_config)},
        "frozen_config_sha256_recorded": {
            "pass": True,
            "sha256": sha256_file(frozen_config),
        },
        "all_16_model_prefix_configs_resolved": {
            "pass": all(str(p) in resolved[m] for m in MODELS for p in PREFIXES)
        },
        "formal_seeds_exact": {
            "pass": list(SEEDS) == [13, 21, 42, 73, 101],
            "actual": list(SEEDS),
        },
        "threshold_fixed_0_5": {"pass": THRESHOLD == 0.5, "actual": THRESHOLD},
        "only_2015_train_loaded": {
            "pass": all(r["train_year"] == 2015 for r in metric_rows)
        },
        "only_2016_validation_loaded": {
            "pass": all(r["validation_year"] == 2016 for r in metric_rows)
        },
        "all_80_runs_completed": {
            "pass": len(metric_rows) == 4 * 4 * 5,
            "actual": len(metric_rows),
            "expected": 80,
        },
        "2017_not_loaded": {"pass": True},
    }
    all_pass = all(v.get("pass", False) for v in checks.values())
    write_json(out_dir / "consistency_checks.json", checks)

    protocol = {
        "stage": STAGE,
        "dataset": "BPIC2018",
        "created_local": datetime.now().astimezone().isoformat(),
        "models": list(MODELS),
        "prefixes": list(PREFIXES),
        "formal_seeds": list(SEEDS),
        "prediction_threshold": THRESHOLD,
        "frozen_config_source": str(frozen_config),
        "frozen_config_sha256": sha256_file(frozen_config),
        "development_train_year": 2015,
        "development_validation_year": 2016,
        "formal_external_test_year": 2017,
        "formal_test_loaded_in_stage57": False,
        "hyperparameter_search_in_stage57": False,
        "neural_validation_usage": "checkpoint/early-stop only under frozen architecture and optimizer",
        "test_based_retuning_forbidden": True,
        "all_hard_checks_pass": all_pass,
    }
    write_json(out_dir / "protocol_frozen.json", protocol)

    # Hash outputs except the manifest itself.
    files = sorted(
        p for p in out_dir.rglob("*")
        if p.is_file() and p.name != "sha256_manifest.json"
    )
    hm = {
        "stage": STAGE,
        "algorithm": "SHA256",
        "generated_local": datetime.now().astimezone().isoformat(),
        "files": {
            str(p): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in files
        },
    }
    write_json(out_dir / "sha256_manifest.json", hm)

    print("\n" + "=" * 120)
    print("ETCE STAGE57 DEVELOPMENT SUMMARY")
    print("=" * 120)
    for r in summary_rows:
        print(
            f"{r['model']:4s} P{r['prefix']:2d} | "
            f"Macro-F1={r['macro_f1_mean']:.6f}±{r['macro_f1_std']:.6f} | "
            f"PR-AUC={r['pr_auc_mean']:.6f}±{r['pr_auc_std']:.6f} | "
            f"ROC-AUC={r['roc_auc_mean']:.6f}±{r['roc_auc_std']:.6f}"
        )
    print(f"Hard gates : {'PASS' if all_pass else 'FAIL'}")
    print(f"2017 used  : NO")
    print("=" * 120)

    if not all_pass:
        print("STAGE57 FAILED. Do not proceed to final refit/test.")
        return 2

    print(
        "\nSTAGE57 COMPLETED UNDER FROZEN BPIC2017 CONFIGURATION.\n"
        "NEXT: Stage58 may refit each frozen model on 2015+2016 using the "
        "Stage56 FINAL mapping and then evaluate 2017 exactly once."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
