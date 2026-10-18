#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage61C
Code-level extraction of the reconstructed Phase3B -> Phase4D protocol chain.

READ-ONLY. No BPIC2018 neighborhood / rule / counterfactual generation.

Target scripts:
  14_phase3a1_semantic_constraint.py
  16_phase3b_formal_counterfactual.py
  18_phase4a_surrogate_pilot.py
  26_phase4b_v2_multirule_bank.py
  34_phase4d_formal_test_multirule_bank.py

The goal is to recover:
- imported project modules/functions
- top-level constants
- argparse parameters/defaults
- input/output paths
- phase-to-phase references
- hard-coded protocol values
- function definitions
- JSON/CSV/NPZ keys mentioned in code
- semantic/process constraints
"""

from __future__ import annotations
import argparse, ast, csv, hashlib, json, re
from pathlib import Path
from typing import Any, Dict, List

TARGETS = [
    "14_phase3a1_semantic_constraint.py",
    "16_phase3b_formal_counterfactual.py",
    "18_phase4a_surrogate_pilot.py",
    "26_phase4b_v2_multirule_bank.py",
    "34_phase4d_formal_test_multirule_bank.py",
]

KEY_PATTERNS = [
    r"phase3a", r"phase3a1", r"phase3b", r"phase4a", r"phase4b", r"phase4d",
    r"train_only_dfg", r"event_family", r"same_position", r"timestamp",
    r"counterfactual", r"surrogate", r"canonical", r"rule", r"support",
    r"lore", r"decision", r"bayesian", r"bn", r"rf", r"xgb", r"mlp", r"lstm",
    r"prefix", r"seed", r"formal_case", r"pilot_case", r"target",
    r"mutable", r"process", r"dfg", r"factual", r"prob", r"threshold",
    r"json", r"csv", r"npz", r"parquet", r"pickle", r"joblib",
]

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def safe_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8-sig")

def literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None

def collect_ast_info(text: str) -> Dict[str, Any]:
    tree = ast.parse(text)
    imports = []
    funcs = []
    constants = []
    argparse_args = []
    calls = []
    string_literals = []

    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                imports.append({"type":"import","module":a.name,"as":a.asname})
        elif isinstance(n, ast.ImportFrom):
            imports.append({
                "type":"from",
                "module":n.module,
                "names":[a.name for a in n.names]
            })
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append({
                "name":n.name,
                "line":getattr(n,"lineno",None),
                "args":[a.arg for a in n.args.args],
            })
        elif isinstance(n, ast.Assign):
            val = literal(n.value)
            if val is not None:
                for t in n.targets:
                    if isinstance(t, ast.Name) and (
                        t.id.isupper() or
                        any(k in t.id.lower() for k in [
                            "seed","prefix","model","case","threshold",
                            "path","dir","count","support","depth","rule",
                            "surrogate","formal","pilot","event","dfg"
                        ])
                    ):
                        constants.append({
                            "name":t.id,
                            "line":getattr(n,"lineno",None),
                            "value":val
                        })
        elif isinstance(n, ast.Call):
            fn = ""
            if isinstance(n.func, ast.Attribute):
                fn = n.func.attr
            elif isinstance(n.func, ast.Name):
                fn = n.func.id
            calls.append({"name":fn,"line":getattr(n,"lineno",None)})
            if fn == "add_argument" and n.args:
                argname = literal(n.args[0])
                kw = {}
                for k in n.keywords:
                    kw[k.arg] = literal(k.value)
                argparse_args.append({
                    "line":getattr(n,"lineno",None),
                    "name":argname,
                    "kwargs":kw
                })
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            s = n.value
            if len(s) <= 600:
                string_literals.append({
                    "line":getattr(n,"lineno",None),
                    "value":s
                })

    # de-duplicate calls
    seen=set(); call_names=[]
    for x in calls:
        if x["name"] and x["name"] not in seen:
            seen.add(x["name"]); call_names.append(x)

    return {
        "imports":imports,
        "functions":sorted(funcs,key=lambda x:x["line"] or 0),
        "constants":sorted(constants,key=lambda x:x["line"] or 0),
        "argparse":sorted(argparse_args,key=lambda x:x["line"] or 0),
        "calls":call_names,
        "strings":string_literals,
    }

def relevant_lines(text: str) -> List[Dict[str, Any]]:
    regs=[re.compile(p,re.I) for p in KEY_PATTERNS]
    out=[]
    for i,line in enumerate(text.splitlines(),1):
        if any(r.search(line) for r in regs):
            out.append({"line":i,"text":line[:800]})
    return out

def probable_paths(strings):
    out=[]
    for s in strings:
        v=s["value"]
        if (
            any(ext in v.lower() for ext in [".json",".csv",".npz",".pkl",".joblib",".pt",".parquet"])
            or "\\" in v or "/" in v
            or "results" in v.lower()
            or "phase3" in v.lower()
            or "phase4" in v.lower()
        ):
            out.append(s)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project_root",type=Path,default=Path(r"D:\ETCE"))
    ap.add_argument("--out_dir",type=Path,default=None)
    ap.add_argument("--overwrite",action="store_true")
    a=ap.parse_args()

    root=a.project_root.resolve()
    scripts=root/"scripts"
    out=(a.out_dir.resolve() if a.out_dir else
         root/"results"/"phase4"/"phase4e_bpic2018_external"/"stage61c_code_protocol_extract")

    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise FileExistsError(f"{out} exists; use --overwrite.")
    if out.exists() and a.overwrite:
        import shutil; shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)

    print("="*122)
    print("ETCE STAGE61C - CODE-LEVEL PHASE3B -> PHASE4D PROTOCOL EXTRACTION")
    print("="*122)
    print(f"Project root : {root}")
    print("BPIC2018 generation: NO")
    print("="*122)

    report={}
    missing=[]
    for fn in TARGETS:
        p=scripts/fn
        print(f"{fn:48s} : {'FOUND' if p.exists() else 'MISSING'}")
        if not p.exists():
            missing.append(fn); continue
        text=safe_text(p)
        info=collect_ast_info(text)
        info.update({
            "path":str(p),
            "sha256":sha256_file(p),
            "line_count":len(text.splitlines()),
            "relevant_lines":relevant_lines(text),
            "probable_paths":probable_paths(info["strings"]),
        })
        # raw strings can be noisy; omit full list from saved report
        info.pop("strings",None)
        report[fn]=info

    if missing:
        raise FileNotFoundError("Missing target scripts: "+", ".join(missing))

    # Cross-chain references
    cross=[]
    for src,info in report.items():
        for row in info["relevant_lines"]:
            lo=row["text"].lower()
            refs=[x for x in TARGETS if x.split("_",1)[0] in lo]
            # phase refs are more useful than exact script refs
            phases=[p for p in ["phase3a","phase3a1","phase3b","phase4a","phase4b","phase4d"]
                    if p in lo]
            if phases:
                cross.append({
                    "script":src,"line":row["line"],
                    "phases":"|".join(phases),"text":row["text"]
                })

    # Write detailed JSON.
    (out/"stage61c_code_protocol_report.json").write_text(
        json.dumps({
            "stage":"61C",
            "targets":TARGETS,
            "scripts":report,
            "cross_phase_references":cross,
            "bpic2018_generation":False
        },ensure_ascii=False,indent=2),
        encoding="utf-8"
    )

    # Functions/imports CSV.
    with (out/"stage61c_functions.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["script","line","function","args"])
        for fn,info in report.items():
            for x in info["functions"]:
                w.writerow([fn,x["line"],x["name"],", ".join(x["args"])])

    with (out/"stage61c_constants.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["script","line","name","value"])
        for fn,info in report.items():
            for x in info["constants"]:
                w.writerow([fn,x["line"],x["name"],repr(x["value"])])

    with (out/"stage61c_argparse.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["script","line","argument","kwargs"])
        for fn,info in report.items():
            for x in info["argparse"]:
                w.writerow([fn,x["line"],x["name"],json.dumps(x["kwargs"],ensure_ascii=False)])

    with (out/"stage61c_cross_phase_references.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["script","line","phases","text"])
        w.writeheader(); w.writerows(cross)

    # Human-readable concise summary
    lines=[]
    lines.append("ETCE STAGE61C CODE-LEVEL PROTOCOL SUMMARY")
    lines.append("="*110)
    for fn in TARGETS:
        info=report[fn]
        lines.append("")
        lines.append(f"[{fn}]")
        lines.append(f"SHA256: {info['sha256']}")
        lines.append("Functions:")
        for x in info["functions"][:25]:
            lines.append(f"  L{x['line']}: {x['name']}({', '.join(x['args'])})")
        lines.append("Protocol-like constants:")
        for x in info["constants"][:35]:
            lines.append(f"  L{x['line']}: {x['name']} = {repr(x['value'])}")
        lines.append("Argparse:")
        for x in info["argparse"][:25]:
            lines.append(f"  L{x['line']}: {x['name']} {x['kwargs']}")
        lines.append("Likely paths / phase references:")
        for x in info["probable_paths"][:25]:
            v=x["value"].replace("\n"," ")
            lines.append(f"  L{x['line']}: {v[:300]}")

    lines.append("")
    lines.append("="*110)
    lines.append("CROSS-PHASE REFERENCE LINES")
    for x in cross[:160]:
        lines.append(f"{x['script']} L{x['line']} [{x['phases']}] {x['text']}")
    lines.append("")
    lines.append("BPIC2018 generation: NO")
    lines.append("="*110)

    summary="\n".join(lines)
    (out/"STAGE61C_CODE_PROTOCOL_SUMMARY.txt").write_text(summary,encoding="utf-8")
    print("\n"+summary)

    print("\nArtifacts:")
    for n in [
        "STAGE61C_CODE_PROTOCOL_SUMMARY.txt",
        "stage61c_code_protocol_report.json",
        "stage61c_functions.csv",
        "stage61c_constants.csv",
        "stage61c_argparse.csv",
        "stage61c_cross_phase_references.csv",
    ]:
        print(out/n)
    print("\nSTAGE61C COMPLETE. No BPIC2018 artifacts were generated.")

if __name__=="__main__":
    main()
