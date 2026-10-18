#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage61B
Trace exact Phase3 neighborhood protocol into Phase4 formal pipeline.

This script is READ-ONLY with respect to experiment artifacts.
It does not generate BPIC2018 neighborhoods and does not inspect TEST labels/predictions.

It:
1) loads the 3 strongest Phase3 run_config JSONs;
2) computes field-level equality/differences;
3) searches local Python/JSON/YAML/MD files for:
   - exact config filenames,
   - their parent directories,
   - distinctive config keys,
   - neighborhood/surrogate/fusion references;
4) ranks scripts that likely generated or consumed these configs;
5) reports which config fields are later referenced by Phase4 scripts.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

CONFIG_REL = [
    Path("results/phase3/phase3a1/phase3a1_run_config.json"),
    Path("results/phase3/phase3b/phase3b_run_config.json"),
    Path("results/phase3/phase3a/phase3a_run_config.json"),
]

TEXT_EXTS = {".py",".json",".yaml",".yml",".md",".txt",".cfg",".ini",".toml"}
MAX_BYTES = 5 * 1024 * 1024

IMPORTANT_KEYWORDS = [
    "neighborhood","neighbourhood","local_pool","two_class","two-class",
    "stratified","surrogate","lore","decisiontree","decision_tree","bayesian",
    "counterfactual","canonical_rule","phase4","formal_test","support",
    "mutable","process_valid","process-valid","distance","ga","population",
    "generation","mutation","crossover","max_edit","edit","320"
]


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def safe_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return p.read_text(encoding="utf-8-sig")
        except Exception:
            return ""
    except Exception:
        return ""

def sha256_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def flatten(x: Any, prefix="") -> Dict[str, Any]:
    out={}
    if isinstance(x,dict):
        for k,v in x.items():
            key=f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v,key))
    elif isinstance(x,list):
        for i,v in enumerate(x):
            key=f"{prefix}[{i}]"
            out.update(flatten(v,key))
    else:
        out[prefix]=x
    return out

def scalar_keys(x: Any) -> List[str]:
    return list(flatten(x).keys())

def basename_keys(paths):
    return [p.name for p in paths]

def walk_text_files(root:Path):
    skip={".git","__pycache__","site-packages","node_modules",".venv","venv","env",
          "stage55_dataset","stage56_features","stage58_formal_external_test",
          "stage59_formal_targets","stage60_canonical_blackbox",
          "stage61_neighborhood_protocol_recovery","stage61b_protocol_trace"}
    for base,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in skip]
        for fn in files:
            p=Path(base)/fn
            if p.suffix.lower() not in TEXT_EXTS: continue
            try:
                if p.stat().st_size>MAX_BYTES: continue
            except OSError:
                continue
            yield p

def line_hits(text:str, needles:List[str], max_lines=80):
    lines=text.splitlines()
    hits=[]
    for i,line in enumerate(lines,1):
        lo=line.lower()
        ms=[n for n in needles if n.lower() in lo]
        if ms:
            hits.append({"line":i,"matches":ms,"text":line[:500]})
            if len(hits)>=max_lines: break
    return hits

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project_root",type=Path,default=Path(r"D:\ETCE"))
    ap.add_argument("--out_dir",type=Path,default=None)
    ap.add_argument("--overwrite",action="store_true")
    a=ap.parse_args()

    root=a.project_root.resolve()
    out=(a.out_dir.resolve() if a.out_dir else
         root/"results"/"phase4"/"phase4e_bpic2018_external"/"stage61b_protocol_trace")

    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise FileExistsError(f"{out} exists. Use --overwrite.")
    if out.exists() and a.overwrite:
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)

    cfg_paths=[root/p for p in CONFIG_REL]
    print("="*120)
    print("ETCE STAGE61B - TRACE PHASE3 NEIGHBORHOOD PROTOCOL INTO PHASE4")
    print("="*120)
    for p in cfg_paths:
        print(f"Config : {p} | {'FOUND' if p.exists() else 'MISSING'}")
    print("BPIC2018 neighborhood generation: NO")
    print("="*120)

    missing=[str(p) for p in cfg_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Phase3 configs:\n"+"\n".join(missing))

    cfgs={p.name:read_json(p) for p in cfg_paths}
    flats={k:flatten(v) for k,v in cfgs.items()}

    # 1) field-by-field comparison
    all_keys=sorted(set().union(*(set(v.keys()) for v in flats.values())))
    diff_rows=[]
    equal_count=0
    diff_count=0
    names=[p.name for p in cfg_paths]

    for key in all_keys:
        vals=[flats[n].get(key,"<MISSING>") for n in names]
        same=all(v==vals[0] for v in vals[1:])
        equal_count += int(same)
        diff_count += int(not same)
        diff_rows.append({
            "field":key,
            "same_all":same,
            names[0]:repr(vals[0]),
            names[1]:repr(vals[1]),
            names[2]:repr(vals[2]),
        })

    with (out/"phase3_config_field_diff.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=["field","same_all",*names]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(diff_rows)

    # dump normalized configs
    for p in cfg_paths:
        (out/f"{p.stem}_pretty.json").write_text(
            json.dumps(cfgs[p.name],ensure_ascii=False,indent=2),encoding="utf-8"
        )

    # 2) search project for filename/path references and config-key references
    config_filenames=[p.name for p in cfg_paths]
    parent_tokens=[p.parent.name for p in cfg_paths]

    # Distinctive leaf keys, reduce common noise.
    leaf_keys=set()
    for k in all_keys:
        leaf=re.sub(r"\[\d+\]","",k.split(".")[-1])
        if len(leaf)>=4:
            leaf_keys.add(leaf)
    key_needles=sorted(leaf_keys)

    candidates=[]
    phase4_refs=[]
    writer_refs=[]

    for p in walk_text_files(root):
        # Don't self-search generated output.
        try:
            rel=str(p.relative_to(root))
        except ValueError:
            rel=str(p)
        text=safe_text(p)
        if not text: continue
        lo=text.lower()

        filename_hits=[x for x in config_filenames if x.lower() in lo]
        parent_hits=[x for x in parent_tokens if x.lower() in lo]
        kw_hits=[x for x in IMPORTANT_KEYWORDS if x.lower() in lo]
        key_hits=[x for x in key_needles if x.lower() in lo]

        score=0
        score += 20*len(filename_hits)
        score += 6*len(parent_hits)
        score += min(12,len(kw_hits))
        score += min(12,len(key_hits))
        if "phase4" in rel.lower(): score += 8
        if "phase3" in rel.lower(): score += 5
        if p.suffix.lower()==".py": score += 3
        if re.search(r"(json\.dump|write_text|open\(.+['\"]w|run_config)",text,re.I):
            score += 5

        if score<10:
            continue

        hits=line_hits(
            text,
            config_filenames+parent_tokens+IMPORTANT_KEYWORDS+key_needles,
            max_lines=100
        )

        row={
            "path":str(p),
            "relative_path":rel,
            "sha256":sha256_file(p),
            "score":score,
            "filename_hits":filename_hits,
            "parent_hits":parent_hits,
            "keyword_hits":kw_hits,
            "config_key_hits":key_hits[:50],
            "line_hits":hits,
        }
        candidates.append(row)

        if "phase4" in rel.lower():
            phase4_refs.append(row)

        # likely config writer
        if filename_hits or (
            "phase3" in rel.lower()
            and re.search(r"(run_config|json\.dump|write_text)",text,re.I)
        ):
            writer_refs.append(row)

    candidates.sort(key=lambda x:(-x["score"],x["relative_path"].lower()))
    phase4_refs.sort(key=lambda x:(-x["score"],x["relative_path"].lower()))
    writer_refs.sort(key=lambda x:(-x["score"],x["relative_path"].lower()))

    # 3) which config fields appear in phase4 scripts?
    phase4_field_usage=defaultdict(list)
    for row in phase4_refs:
        text=safe_text(Path(row["path"])).lower()
        for full_key in all_keys:
            leaf=re.sub(r"\[\d+\]","",full_key.split(".")[-1]).lower()
            if len(leaf)>=4 and leaf in text:
                phase4_field_usage[full_key].append(row["relative_path"])

    with (out/"phase4_config_field_usage.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["field","phase4_reference_count","phase4_files"])
        for k in sorted(phase4_field_usage):
            w.writerow([k,len(set(phase4_field_usage[k])),
                        " | ".join(sorted(set(phase4_field_usage[k])))])

    # 4) ranked summary
    report={
        "stage":"61B",
        "project_root":str(root),
        "config_files":[{
            "path":str(p),
            "sha256":sha256_file(p),
            "top_level_keys":sorted(cfgs[p.name].keys()) if isinstance(cfgs[p.name],dict) else None
        } for p in cfg_paths],
        "field_counts":{
            "union_fields":len(all_keys),
            "equal_across_all_three":equal_count,
            "different_across_three":diff_count,
        },
        "top_candidate_scripts":candidates[:40],
        "likely_phase3_config_writers":writer_refs[:30],
        "phase4_consumers_or_related_scripts":phase4_refs[:40],
        "phase4_field_usage":{
            k:sorted(set(v)) for k,v in phase4_field_usage.items()
        },
        "bpic2018_neighborhood_generated":False,
    }
    (out/"stage61b_protocol_trace_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
    )

    # concise human report
    lines=[]
    lines.append("ETCE STAGE61B PROTOCOL TRACE SUMMARY")
    lines.append("="*100)
    lines.append(f"Fields total/equal/different: {len(all_keys)}/{equal_count}/{diff_count}")
    lines.append("")
    lines.append("LIKELY PHASE3 CONFIG WRITERS / GENERATORS:")
    for i,x in enumerate(writer_refs[:12],1):
        lines.append(f"{i:2d}. score={x['score']:3d} | {x['relative_path']}")
    lines.append("")
    lines.append("TOP PHASE4 CONSUMERS / RELATED SCRIPTS:")
    for i,x in enumerate(phase4_refs[:15],1):
        lines.append(f"{i:2d}. score={x['score']:3d} | {x['relative_path']}")
    lines.append("")
    lines.append("FIELDS DIFFERENT ACROSS THE 3 CONFIGS:")
    for r in diff_rows:
        if not r["same_all"]:
            lines.append(f"- {r['field']}")
            for n in names:
                lines.append(f"    {n}: {r[n]}")
    lines.append("")
    lines.append("BPIC2018 neighborhood generated: NO")
    lines.append("="*100)

    summary="\n".join(lines)
    (out/"STAGE61B_PROTOCOL_TRACE_SUMMARY.txt").write_text(summary,encoding="utf-8")
    print(summary)

    print("\nArtifacts:")
    print(out/"STAGE61B_PROTOCOL_TRACE_SUMMARY.txt")
    print(out/"phase3_config_field_diff.csv")
    print(out/"phase4_config_field_usage.csv")
    print(out/"stage61b_protocol_trace_report.json")
    print("\nSTAGE61B COMPLETE. No BPIC2018 neighborhood was generated.")


if __name__=="__main__":
    main()
