#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage61E
Exact source/schema extraction for the last three interfaces needed by BPIC2018 Stage62.

READ ONLY.
No BPIC2018 neighborhood/rule generation.
"""

from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

TARGETS = [
    ("src/etce/phase3/core.py", ["load_process_constraints", "flow_valid", "process_support_score", "sequence_edit_count"]),
    ("src/etce/phase4/surrogate_core.py", ["generate_process_valid_pool"]),
    ("src/etce/phase4/stratified_core.py", ["run_all_surrogates_stratified"]),
]

def safe_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8-sig")

def sha256_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def extract_function(text: str, name: str):
    tree=ast.parse(text)
    lines=text.splitlines()
    for n in ast.walk(tree):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name:
            start=n.lineno
            end=getattr(n,"end_lineno",start)
            return {
                "name":name,
                "start_line":start,
                "end_line":end,
                "source":"\n".join(lines[start-1:end]),
            }
    return None

def shape(x: Any, depth=0):
    if depth >= 4:
        if isinstance(x, dict): return {"type":"dict","len":len(x)}
        if isinstance(x, (list,tuple,set)): return {"type":type(x).__name__,"len":len(x)}
        return {"type":type(x).__name__,"sample":repr(x)[:160]}
    if isinstance(x, dict):
        out={"type":"dict","len":len(x),"keys":list(x.keys())[:50],"children":{}}
        for k in list(x.keys())[:20]:
            out["children"][str(k)]=shape(x[k],depth+1)
        return out
    if isinstance(x, (list,tuple,set)):
        vals=list(x)
        return {
            "type":type(x).__name__,
            "len":len(vals),
            "items":[shape(v,depth+1) for v in vals[:5]],
        }
    return {"type":type(x).__name__,"sample":repr(x)[:200]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project_root",type=Path,default=Path(r"D:\ETCE"))
    ap.add_argument("--out_dir",type=Path,default=None)
    ap.add_argument("--overwrite",action="store_true")
    a=ap.parse_args()

    root=a.project_root.resolve()
    out=(a.out_dir.resolve() if a.out_dir else
         root/"results"/"phase4"/"phase4e_bpic2018_external"/"stage61e_exact_interfaces")
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise FileExistsError(f"{out} exists; use --overwrite.")
    if out.exists() and a.overwrite:
        import shutil; shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)

    report={"stage":"61E","files":{},"bpic2018_generation":False}
    summary=[]

    summary.append("ETCE STAGE61E EXACT INTERFACE / CONSTRAINT SCHEMA")
    summary.append("="*110)

    for rel,names in TARGETS:
        p=root/rel
        summary.append(f"\n[{rel}] {'FOUND' if p.exists() else 'MISSING'}")
        if not p.exists():
            raise FileNotFoundError(p)
        text=safe_text(p)
        fs={}
        for n in names:
            item=extract_function(text,n)
            if item is None:
                raise RuntimeError(f"Function {n} not found in {p}")
            fs[n]=item
            summary.append(f"\n### {n} L{item['start_line']}-L{item['end_line']}")
            summary.append(item["source"])
        report["files"][rel]={"sha256":sha256_file(p),"functions":fs}

    # Runtime schema of the already-frozen BPIC2017 constraints.
    import sys
    if str(root/"src") not in sys.path:
        sys.path.insert(0,str(root/"src"))
    from etce.phase3.core import load_process_constraints

    constraints=load_process_constraints(root/"results/phase3")
    schema=shape(constraints)
    report["bpic2017_constraint_schema"]=schema

    summary.append("\n[BPI2017 LOADED PROCESS CONSTRAINT SCHEMA]")
    summary.append(json.dumps(schema,ensure_ascii=False,indent=2))

    # Save exact JSON report.
    (out/"stage61e_exact_interface_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    txt="\n".join(summary)
    (out/"STAGE61E_EXACT_INTERFACE_SUMMARY.txt").write_text(txt,encoding="utf-8")

    print(txt)
    print("\nArtifacts:")
    print(out/"STAGE61E_EXACT_INTERFACE_SUMMARY.txt")
    print(out/"stage61e_exact_interface_report.json")
    print("\nSTAGE61E COMPLETE. No BPIC2018 artifacts were generated.")

if __name__=="__main__":
    main()
