#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage61F
BPIC2018 semantic-family compatibility audit for Phase4 process-valid pool.

READ ONLY:
- does not generate BPIC2018 neighborhoods
- does not inspect TEST labels/predictions
- does not tune any protocol

It prints:
1) exact semantic_core family functions used by Phase4;
2) BPIC2018 FINAL train-side activity vocabulary (2015+2016 mapping);
3) pairwise family-preservation feasibility counts under the existing function.
"""

from __future__ import annotations
import argparse, ast, json
from pathlib import Path
from collections import Counter, defaultdict

TARGET_FUNCS = [
    "event_family",
    "family_preserved",
    "semantic_mutation_options",
    "semantic_mutate_once",
    "semantic_crossover",
]

def safe_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8-sig")

def extract_func(text: str, name: str):
    tree = ast.parse(text)
    lines = text.splitlines()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            start = n.lineno
            end = getattr(n, "end_lineno", start)
            return {
                "start": start,
                "end": end,
                "source": "\n".join(lines[start-1:end]),
            }
    return None

def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=Path, default=Path(r"D:\ETCE"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.project_root.resolve()
    semantic_path = root / "src" / "etce" / "phase3" / "semantic_core.py"
    stage56 = root / "results" / "phase4" / "phase4e_bpic2018_external" / "stage56_features"
    mapping_path = stage56 / "mapping_final_fit2015_2016.json"
    out = root / "results" / "phase4" / "phase4e_bpic2018_external" / "stage61f_semantic_family_audit"

    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} exists; use --overwrite.")
    if out.exists() and args.overwrite:
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    if not semantic_path.exists():
        raise FileNotFoundError(semantic_path)
    if not mapping_path.exists():
        raise FileNotFoundError(mapping_path)

    text = safe_text(semantic_path)
    funcs = {}
    for name in TARGET_FUNCS:
        item = extract_func(text, name)
        if item is not None:
            funcs[name] = item

    # Import exact implementation.
    import sys
    sys.path.insert(0, str(root / "src"))
    from etce.phase3.semantic_core import family_preserved
    try:
        from etce.phase3.semantic_core import event_family
    except Exception:
        event_family = None

    mapping = read_json(mapping_path)
    vocab_obj = mapping["activity_vocab"]

    # Handle either token->id or list-like vocab.
    if isinstance(vocab_obj, dict):
        activities = [
            k for k, v in sorted(vocab_obj.items(), key=lambda kv: kv[1])
            if int(v) >= 2
        ]
    elif isinstance(vocab_obj, list):
        activities = [str(x) for i, x in enumerate(vocab_obj) if i >= 2]
    else:
        raise TypeError(f"Unsupported activity_vocab type: {type(vocab_obj)}")

    compatible = []
    incompatible = []
    family_counts = Counter()

    for a in activities:
        fam = None
        if event_family is not None:
            try:
                fam = event_family(a)
            except Exception as e:
                fam = f"<ERROR:{type(e).__name__}>"
        family_counts[str(fam)] += 1

    for a in activities:
        for b in activities:
            if a == b:
                continue
            try:
                ok = bool(family_preserved(a, b))
            except Exception as e:
                ok = False
            (compatible if ok else incompatible).append((a, b))

    by_source = {}
    for a in activities:
        opts = [b for b in activities if b != a and bool(family_preserved(a, b))]
        by_source[a] = opts

    zero_option = [a for a, opts in by_source.items() if len(opts) == 0]

    summary = {
        "activity_count": len(activities),
        "family_counts": dict(family_counts),
        "ordered_distinct_pairs": len(activities) * max(0, len(activities)-1),
        "family_preserved_pairs": len(compatible),
        "family_rejected_pairs": len(incompatible),
        "activities_with_zero_family_options": zero_option,
        "family_options_by_activity": by_source,
        "bpic2018_neighborhood_generated": False,
    }

    (out / "semantic_family_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("=" * 116)
    lines.append("ETCE STAGE61F - BPIC2018 SEMANTIC-FAMILY COMPATIBILITY AUDIT")
    lines.append("=" * 116)
    lines.append(f"semantic_core : {semantic_path}")
    lines.append(f"mapping       : {mapping_path}")
    lines.append("BPIC2018 neighborhood generation: NO")
    lines.append("")

    for name in TARGET_FUNCS:
        if name in funcs:
            x = funcs[name]
            lines.append(f"### {name} L{x['start']}-L{x['end']}")
            lines.append(x["source"])
            lines.append("")

    lines.append("[BPIC2018 FINAL TRAIN-SIDE ACTIVITY VOCAB]")
    for i, a in enumerate(activities, 1):
        fam = event_family(a) if event_family is not None else "<not exported>"
        lines.append(f"{i:2d}. {a!r} | family={fam!r} | options={len(by_source[a])}")

    lines.append("")
    lines.append("[AUDIT]")
    lines.append(f"Activity count                    : {len(activities)}")
    lines.append(f"Family counts                     : {dict(family_counts)}")
    lines.append(f"Family-preserved ordered pairs    : {len(compatible)}")
    lines.append(f"Family-rejected ordered pairs     : {len(incompatible)}")
    lines.append(f"Activities with zero family opts  : {len(zero_option)}")
    if zero_option:
        for a in zero_option:
            lines.append(f"  - {a!r}")
    lines.append("")
    lines.append("BPIC2018 neighborhood generated   : NO")
    lines.append("=" * 116)

    txt = "\n".join(lines)
    (out / "STAGE61F_SEMANTIC_FAMILY_AUDIT.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print("\nArtifacts:")
    print(out / "STAGE61F_SEMANTIC_FAMILY_AUDIT.txt")
    print(out / "semantic_family_audit.json")

if __name__ == "__main__":
    main()
