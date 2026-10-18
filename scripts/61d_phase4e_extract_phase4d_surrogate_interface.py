#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage61D
Narrow interface extraction for porting Phase4D to BPIC2018.

READ ONLY. No experiment artifacts are modified.
No BPIC2018 neighborhood/rule generation is performed.

Extracts:
1) imports from scripts/34_phase4d_formal_test_multirule_bank.py
2) exact source bodies for selected helper functions from script 34
3) all calls made inside script34.main()
4) public functions/classes/signatures from src/etce/phase4/surrogate_core.py
5) source bodies of surrogate_core functions referenced by script34
6) referenced constants / path globals

Output is intentionally compact enough to paste back into ChatGPT.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set

SCRIPT34 = "34_phase4d_formal_test_multirule_bank.py"
CORE_REL = Path("src/etce/phase4/surrogate_core.py")

SCRIPT34_KEEP_FUNCS = {
    "local_seed",
    "pool_digest",
    "token",
    "target_cache_path",
    "eval_cache_path",
    "cache_is_complete",
    "eval_cache_is_complete",
    "rule_core",
    "legacy_matches_first",
    "rule_record",
    "require_inputs",
    "main",
}

MAX_BODY_LINES = 260


def safe_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = dotted_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def node_source(lines: List[str], node: ast.AST, max_lines: int = MAX_BODY_LINES) -> str:
    start = max(1, getattr(node, "lineno", 1))
    end = max(start, getattr(node, "end_lineno", start))
    if end - start + 1 > max_lines:
        end = start + max_lines - 1
        suffix = f"\n# ... TRUNCATED at {max_lines} lines ..."
    else:
        suffix = ""
    return "\n".join(lines[start - 1:end]) + suffix


def signature_of(node: ast.AST) -> str:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    args = []
    defaults_offset = len(node.args.args) - len(node.args.defaults)
    for i, a in enumerate(node.args.args):
        s = a.arg
        if a.annotation is not None:
            try:
                s += ": " + ast.unparse(a.annotation)
            except Exception:
                pass
        if i >= defaults_offset:
            d = node.args.defaults[i - defaults_offset]
            try:
                s += "=" + ast.unparse(d)
            except Exception:
                s += "=..."
        args.append(s)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults):
        s = a.arg
        if d is not None:
            try:
                s += "=" + ast.unparse(d)
            except Exception:
                s += "=..."
        args.append(s)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    return f"{node.name}({', '.join(args)})"


def extract_module(path: Path) -> Dict[str, Any]:
    text = safe_text(path)
    lines = text.splitlines()
    tree = ast.parse(text)

    imports = []
    functions = {}
    classes = {}
    constants = []

    for n in tree.body:
        if isinstance(n, ast.Import):
            imports.append(node_source(lines, n, 20))
        elif isinstance(n, ast.ImportFrom):
            imports.append(node_source(lines, n, 20))
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[n.name] = {
                "signature": signature_of(n),
                "lineno": n.lineno,
                "end_lineno": getattr(n, "end_lineno", n.lineno),
                "node": n,
                "source": node_source(lines, n),
            }
        elif isinstance(n, ast.ClassDef):
            classes[n.name] = {
                "lineno": n.lineno,
                "end_lineno": getattr(n, "end_lineno", n.lineno),
                "source": node_source(lines, n),
            }
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            names = []
            value = None
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        names.append(t.id)
                value = n.value
            else:
                if isinstance(n.target, ast.Name):
                    names.append(n.target.id)
                value = n.value
            if names and any(
                nm.isupper() or any(k in nm.lower() for k in [
                    "path", "dir", "prefix", "model", "surrogate",
                    "pool", "edit", "seed", "case", "schema",
                ])
                for nm in names
            ):
                try:
                    val = ast.unparse(value) if value is not None else None
                except Exception:
                    val = None
                constants.append({
                    "line": n.lineno,
                    "names": names,
                    "value": val,
                    "source": node_source(lines, n, 10),
                })

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "text": text,
        "lines": lines,
        "tree": tree,
        "imports": imports,
        "functions": functions,
        "classes": classes,
        "constants": constants,
    }


def calls_inside(func_node: ast.AST) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for n in ast.walk(func_node):
        if isinstance(n, ast.Call):
            name = dotted_name(n.func)
            if not name:
                continue
            key = (name, getattr(n, "lineno", None))
            if key in seen:
                continue
            seen.add(key)
            args = []
            for a in n.args[:8]:
                try:
                    args.append(ast.unparse(a))
                except Exception:
                    args.append("...")
            kwargs = {}
            for k in n.keywords[:12]:
                try:
                    kwargs[k.arg or "**"] = ast.unparse(k.value)
                except Exception:
                    kwargs[k.arg or "**"] = "..."
            out.append({
                "line": getattr(n, "lineno", None),
                "name": name,
                "args": args,
                "kwargs": kwargs,
            })
    return sorted(out, key=lambda x: (x["line"] or 0, x["name"]))


def imported_from_core(module_info: Dict[str, Any]) -> Set[str]:
    names = set()
    tree = module_info["tree"]
    for n in tree.body:
        if isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            if "surrogate_core" in mod:
                for a in n.names:
                    names.add(a.name)
    return names


def names_referenced_in_main(module_info: Dict[str, Any]) -> Set[str]:
    main = module_info["functions"].get("main")
    if not main:
        return set()
    refs = set()
    for n in ast.walk(main["node"]):
        if isinstance(n, ast.Name):
            refs.add(n.id)
        elif isinstance(n, ast.Attribute):
            refs.add(dotted_name(n))
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=Path, default=Path(r"D:\ETCE"))
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.project_root.resolve()
    s34_path = root / "scripts" / SCRIPT34
    core_path = root / CORE_REL
    out = (
        args.out_dir.resolve()
        if args.out_dir
        else root / "results" / "phase4" / "phase4e_bpic2018_external" / "stage61d_interface_extract"
    )

    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} exists. Use --overwrite.")
    if out.exists() and args.overwrite:
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 122)
    print("ETCE STAGE61D - PHASE4D / SURROGATE_CORE INTERFACE EXTRACTION")
    print("=" * 122)
    print(f"Phase4D script : {s34_path} | {'FOUND' if s34_path.exists() else 'MISSING'}")
    print(f"Surrogate core : {core_path} | {'FOUND' if core_path.exists() else 'MISSING'}")
    print("BPIC2018 generation: NO")
    print("=" * 122)

    if not s34_path.exists():
        raise FileNotFoundError(s34_path)
    if not core_path.exists():
        raise FileNotFoundError(core_path)

    s34 = extract_module(s34_path)
    core = extract_module(core_path)

    core_import_names = imported_from_core(s34)
    main_refs = names_referenced_in_main(s34)

    # Also catch module-qualified calls if core imported as module.
    core_public_names = set(core["functions"]) | set(core["classes"])
    directly_used_core = sorted(
        n for n in core_public_names
        if n in main_refs or n in core_import_names
    )

    main_calls = calls_inside(s34["functions"]["main"]["node"])

    report = {
        "stage": "61D",
        "phase4d": {
            "path": s34["path"],
            "sha256": s34["sha256"],
            "imports": s34["imports"],
            "constants": s34["constants"],
            "functions": {
                name: {
                    "signature": x["signature"],
                    "lineno": x["lineno"],
                    "end_lineno": x["end_lineno"],
                }
                for name, x in s34["functions"].items()
            },
            "core_import_names": sorted(core_import_names),
            "main_calls": main_calls,
        },
        "surrogate_core": {
            "path": core["path"],
            "sha256": core["sha256"],
            "imports": core["imports"],
            "constants": core["constants"],
            "functions": {
                name: {
                    "signature": x["signature"],
                    "lineno": x["lineno"],
                    "end_lineno": x["end_lineno"],
                }
                for name, x in core["functions"].items()
            },
            "classes": {
                name: {
                    "lineno": x["lineno"],
                    "end_lineno": x["end_lineno"],
                }
                for name, x in core["classes"].items()
            },
            "directly_used_by_phase4d_main": directly_used_core,
        },
        "bpic2018_generation": False,
    }

    (out / "stage61d_interface_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("ETCE STAGE61D INTERFACE SUMMARY")
    lines.append("=" * 116)
    lines.append("")
    lines.append("[PHASE4D IMPORTS]")
    lines.extend(s34["imports"])
    lines.append("")
    lines.append("[PHASE4D PROTOCOL/PATH CONSTANTS]")
    for x in s34["constants"]:
        lines.append(f"L{x['line']}: {x['source']}")
    lines.append("")
    lines.append("[SURROGATE_CORE IMPORTS]")
    lines.extend(core["imports"])
    lines.append("")
    lines.append("[SURROGATE_CORE PUBLIC FUNCTIONS]")
    for name, x in sorted(core["functions"].items(), key=lambda kv: kv[1]["lineno"]):
        if not name.startswith("_"):
            lines.append(f"L{x['lineno']}: {x['signature']}")
    lines.append("")
    lines.append("[SURROGATE_CORE CLASSES]")
    for name, x in sorted(core["classes"].items(), key=lambda kv: kv[1]["lineno"]):
        lines.append(f"L{x['lineno']}: class {name}")
    lines.append("")
    lines.append("[CORE NAMES DIRECTLY USED/IMPORTED BY PHASE4D]")
    for n in directly_used_core:
        lines.append(f"- {n}")
    lines.append("")
    lines.append("[CALLS INSIDE PHASE4D main()]")
    for c in main_calls:
        argtxt = ", ".join(c["args"])
        if c["kwargs"]:
            if argtxt:
                argtxt += ", "
            argtxt += ", ".join(f"{k}={v}" for k, v in c["kwargs"].items())
        lines.append(f"L{c['line']}: {c['name']}({argtxt})")
    lines.append("")
    lines.append("[SELECTED PHASE4D FUNCTION BODIES]")
    for name in SCRIPT34_KEEP_FUNCS:
        x = s34["functions"].get(name)
        if x:
            lines.append("")
            lines.append(f"### {name} :: {x['signature']} :: L{x['lineno']}-L{x['end_lineno']}")
            lines.append(x["source"])
    lines.append("")
    lines.append("[SURROGATE_CORE FUNCTION BODIES USED BY PHASE4D]")
    for name in directly_used_core:
        x = core["functions"].get(name)
        if x:
            lines.append("")
            lines.append(f"### {name} :: {x['signature']} :: L{x['lineno']}-L{x['end_lineno']}")
            lines.append(x["source"])

    lines.append("")
    lines.append("=" * 116)
    lines.append("BPIC2018 generation: NO")
    lines.append("=" * 116)

    summary = "\n".join(lines)
    (out / "STAGE61D_INTERFACE_SUMMARY.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print("\nArtifacts:")
    print(out / "STAGE61D_INTERFACE_SUMMARY.txt")
    print(out / "stage61d_interface_report.json")
    print("\nSTAGE61D COMPLETE. No BPIC2018 artifacts were generated.")


if __name__ == "__main__":
    main()
