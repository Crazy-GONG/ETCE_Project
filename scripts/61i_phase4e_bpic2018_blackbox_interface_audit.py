#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage61I
Recover the exact BPIC2018 black-box feature/prediction interface needed by Stage62.

READ ONLY:
- no model training
- no TEST label access for tuning
- no neighborhood generation
- no protocol changes

It inspects:
  scripts/56_phase4e_bpic2018_feature_build.py
  scripts/58_phase4e_bpic2018_final_refit_formal_test.py
  Stage56 final mapping
  one final TEST NPZ schema (P5)
  Stage60 canonical seed42 checkpoints

The output is intentionally focused on functions/classes/constants that can
rebuild a perturbed prefix and run the frozen black boxes.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import torch

SCRIPT56_CANDIDATES = [
    "56_phase4e_bpic2018_feature_build.py",
]
SCRIPT58_CANDIDATES = [
    "58_phase4e_bpic2018_final_refit_formal_test.py",
    "58_phase4e_bpic2018_final_refit_formal_test_v2.py",
]

INTEREST = (
    "feature", "build", "encode", "transform", "mapping", "vocab", "scaler",
    "dataset", "npz", "load", "model", "predict", "prob", "lstm", "mlp",
    "rf", "xgb", "torch", "checkpoint", "sequence", "tabular", "count",
)


def safe_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8-sig")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def locate_script(root: Path, candidates: List[str]) -> Path:
    scripts = root / "scripts"
    for n in candidates:
        p = scripts / n
        if p.exists():
            return p
    hits = []
    for p in scripts.glob("*.py"):
        lo = p.name.lower()
        if any(c.split("_", 1)[0] in lo for c in candidates):
            hits.append(p)
    if not hits:
        raise FileNotFoundError(f"Could not locate any of {candidates}")
    hits.sort(key=lambda p: p.name)
    return hits[0]


def sig(n: ast.FunctionDef) -> str:
    try:
        return f"{n.name}{ast.unparse(n.args)}"
    except Exception:
        return n.name


def node_source(lines: List[str], node: ast.AST, max_lines=220) -> str:
    s = getattr(node, "lineno", 1)
    e = getattr(node, "end_lineno", s)
    truncated = (e - s + 1) > max_lines
    e2 = min(e, s + max_lines - 1)
    body = "\n".join(lines[s-1:e2])
    if truncated:
        body += f"\n# ... TRUNCATED after {max_lines} lines ..."
    return body


def inspect_source(path: Path) -> Dict[str, Any]:
    text = safe_text(path)
    lines = text.splitlines()
    tree = ast.parse(text)

    imports = []
    funcs = []
    classes = []
    constants = []

    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            imports.append(node_source(lines, n, 20))
        elif isinstance(n, ast.FunctionDef):
            low = n.name.lower()
            if any(k in low for k in INTEREST):
                funcs.append({
                    "name": n.name,
                    "line": n.lineno,
                    "end": getattr(n, "end_lineno", n.lineno),
                    "signature": sig(n),
                    "source": node_source(lines, n),
                })
        elif isinstance(n, ast.ClassDef):
            low = n.name.lower()
            if any(k in low for k in INTEREST) or "nn.module" in node_source(lines, n, 4).lower():
                classes.append({
                    "name": n.name,
                    "line": n.lineno,
                    "end": getattr(n, "end_lineno", n.lineno),
                    "source": node_source(lines, n),
                })
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            src = node_source(lines, n, 12)
            if any(k in src.lower() for k in INTEREST):
                constants.append({"line": getattr(n, "lineno", None), "source": src})

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "imports": imports,
        "functions": funcs,
        "classes": classes,
        "constants": constants,
    }


def shape(x: Any, depth=0):
    if depth >= 3:
        if isinstance(x, dict):
            return {"type": "dict", "len": len(x), "keys": list(x.keys())[:30]}
        if isinstance(x, (list, tuple)):
            return {"type": type(x).__name__, "len": len(x), "sample": x[:5]}
        return {"type": type(x).__name__, "sample": repr(x)[:200]}
    if isinstance(x, dict):
        return {
            "type": "dict",
            "len": len(x),
            "keys": list(x.keys())[:50],
            "children": {str(k): shape(v, depth+1) for k, v in list(x.items())[:30]},
        }
    if isinstance(x, (list, tuple)):
        return {
            "type": type(x).__name__,
            "len": len(x),
            "items": [shape(v, depth+1) for v in list(x)[:5]],
        }
    return {"type": type(x).__name__, "sample": repr(x)[:220]}


def find_mapping(stage56: Path) -> Path:
    preferred = stage56 / "mapping_final_fit2015_2016.json"
    if preferred.exists():
        return preferred
    maps = sorted(stage56.glob("*final*map*.json"))
    if not maps:
        maps = sorted(stage56.glob("*mapping*.json"))
    if not maps:
        raise FileNotFoundError("Stage56 final mapping JSON not found")
    return maps[0]


def inspect_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        out = {}
        for k in z.files:
            arr = z[k]
            sample = None
            if arr.size:
                try:
                    sample = arr.reshape(-1)[:5].tolist()
                except Exception:
                    sample = repr(arr.reshape(-1)[:5])
            out[k] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sample": sample,
            }
        return out


def checkpoint_schema(path: Path, model: str) -> Dict[str, Any]:
    if path.suffix.lower() in (".joblib", ".pkl"):
        obj = joblib.load(path)
        attrs = {}
        for name in (
            "n_features_in_", "classes_", "feature_names_in_",
            "n_estimators", "max_depth", "hidden_layer_sizes",
        ):
            if hasattr(obj, name):
                v = getattr(obj, name)
                try:
                    v = v.tolist()
                except Exception:
                    pass
                attrs[name] = v
        return {
            "loader": "joblib",
            "python_type": f"{type(obj).__module__}.{type(obj).__name__}",
            "has_predict_proba": hasattr(obj, "predict_proba"),
            "attrs": attrs,
        }

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        return {
            "loader": "torch",
            "type": "dict",
            "keys": list(obj.keys()),
            "shape": shape(obj),
        }
    return {
        "loader": "torch",
        "python_type": f"{type(obj).__module__}.{type(obj).__name__}",
        "shape": shape(obj),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=Path, default=Path(r"D:\ETCE"))
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    root = a.project_root.resolve()
    out = (
        a.out_dir.resolve() if a.out_dir
        else root / "results" / "phase4" / "phase4e_bpic2018_external" / "stage61i_blackbox_interface"
    )
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise FileExistsError(f"{out} exists; use --overwrite.")
    if out.exists() and a.overwrite:
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    s56 = locate_script(root, SCRIPT56_CANDIDATES)
    s58 = locate_script(root, SCRIPT58_CANDIDATES)
    stage56 = root / "results" / "phase4" / "phase4e_bpic2018_external" / "stage56_features"
    stage60 = root / "results" / "phase4" / "phase4e_bpic2018_external" / "stage60_canonical_blackbox"

    mapping_path = find_mapping(stage56)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    npz_path = stage56 / "final_P5_test2017.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    ck_map = stage60 / "canonical_checkpoint_map.csv"
    if not ck_map.exists():
        raise FileNotFoundError(ck_map)

    checkpoints = []
    with ck_map.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["prefix"]) == 5:
                p = Path(row["checkpoint"])
                checkpoints.append({
                    "model": row["model"],
                    "path": str(p),
                    "sha256": row["checkpoint_sha256"],
                    "schema": checkpoint_schema(p, row["model"]),
                })

    report = {
        "stage": "61I",
        "script56": inspect_source(s56),
        "script58": inspect_source(s58),
        "mapping_path": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path),
        "mapping_shape": shape(mapping),
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "npz_schema": inspect_npz(npz_path),
        "p5_seed42_checkpoint_schemas": checkpoints,
        "training_or_tuning_performed": False,
        "neighborhood_generated": False,
    }
    (out / "stage61i_blackbox_interface_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    lines = []
    lines.append("=" * 118)
    lines.append("ETCE STAGE61I - BPIC2018 BLACK-BOX PERTURBATION INTERFACE")
    lines.append("=" * 118)
    lines.append(f"Stage56 script : {s56}")
    lines.append(f"Stage58 script : {s58}")
    lines.append(f"Mapping        : {mapping_path}")
    lines.append(f"P5 TEST NPZ    : {npz_path}")
    lines.append("Training/tuning performed : NO")
    lines.append("Neighborhood generated    : NO")

    for label, info in [("STAGE56", report["script56"]), ("STAGE58", report["script58"])]:
        lines.append("")
        lines.append(f"[{label} INTERESTING FUNCTIONS]")
        for x in info["functions"]:
            lines.append(f"L{x['line']}: {x['signature']}")
        lines.append("")
        lines.append(f"[{label} INTERESTING CLASSES]")
        for x in info["classes"]:
            lines.append(f"L{x['line']}: class {x['name']}")

    lines.append("")
    lines.append("[MAPPING TOP-LEVEL SHAPE]")
    lines.append(json.dumps(report["mapping_shape"], ensure_ascii=False, indent=2))

    lines.append("")
    lines.append("[P5 FINAL TEST NPZ SCHEMA]")
    lines.append(json.dumps(report["npz_schema"], ensure_ascii=False, indent=2))

    lines.append("")
    lines.append("[P5 CANONICAL SEED42 CHECKPOINT SCHEMAS]")
    for x in checkpoints:
        lines.append(f"\n### {x['model']}")
        lines.append(f"path: {x['path']}")
        lines.append(json.dumps(x["schema"], ensure_ascii=False, indent=2, default=str))

    # Include exact bodies only for compact relevant functions/classes.
    for label, info in [("STAGE56", report["script56"]), ("STAGE58", report["script58"])]:
        lines.append("")
        lines.append(f"[{label} RELEVANT SOURCE BODIES]")
        for x in info["functions"]:
            lines.append(f"\n### {x['name']} L{x['line']}-L{x['end']}")
            lines.append(x["source"])
        for x in info["classes"]:
            lines.append(f"\n### class {x['name']} L{x['line']}-L{x['end']}")
            lines.append(x["source"])

    lines.append("")
    lines.append("=" * 118)
    lines.append("STAGE61I COMPLETE. No BPIC2018 neighborhood/rule artifacts were generated.")
    lines.append("=" * 118)

    txt = "\n".join(lines)
    (out / "STAGE61I_BLACKBOX_INTERFACE_SUMMARY.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print("\nArtifacts:")
    print(out / "STAGE61I_BLACKBOX_INTERFACE_SUMMARY.txt")
    print(out / "stage61i_blackbox_interface_report.json")


if __name__ == "__main__":
    main()
