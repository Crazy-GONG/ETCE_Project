#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse, csv, gzip, importlib.util, json, sys
from collections import defaultdict
from pathlib import Path

LAMBDA = 5.0
SURROGATE_ORDER = {"LORE": 0, "DT": 1, "BN": 2}
MODEL_ORDER = {"RF": 0, "XGBoost": 1, "MLP": 2, "LSTM": 3}
MODEL_ALIAS = {"XGB": "XGBoost", "XGBoost": "XGBoost"}

def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def write_json(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, default=lambda o: sorted(o) if isinstance(o,set) else list(o) if isinstance(o,tuple) else str(o)), encoding="utf-8")

def write_jsonl(p, rows):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=lambda o: sorted(o) if isinstance(o,set) else list(o) if isinstance(o,tuple) else str(o))+"\n")

def write_csv(p, rows):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    rows=list(rows)
    if not rows:
        p.write_text("", encoding="utf-8"); return
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def import_from_path(name, path):
    spec=importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

def locate_stage37(root):
    for p in [
        root/"scripts/37_phase4d_frozen_ds_formal_test_fusion_v2.py",
        root/"scripts/37_phase4d_frozen_ds_formal_test_fusion.py",
    ]:
        if p.exists(): return p
    hits=sorted((root/"scripts").glob("37*ds*formal*test*fusion*.py"))
    if not hits: raise FileNotFoundError("Stage37 frozen D-S script not found")
    return hits[-1]

def norm_model(x):
    return MODEL_ALIAS.get(str(x), str(x))

def eligible(rule):
    if rule.get("eligible_for_reliability") is False: return False
    try:
        return int(rule["n_r"])>0 and rule["Fid_r"] is not None
    except Exception:
        return False

def extract_external(stage62):
    """
    Reconstruct Stage64 inputs from the two authoritative Stage62 artifacts:

    1) canonical_and_counterfactual_rules.jsonl
       -> authoritative finalized rules, including held-out n_r / Fid_r /
          coverage_eval and candidate_id.

    2) target_cache/*.json.gz
       -> authoritative held-out local evaluation contexts used only for
          fused-rule re-evaluation.

    Raw branch['counterfactual_rules'] inside target_cache contains candidate
    structure but is not the finalized canonical rule record and therefore
    must NOT be used as the reliability-metric source.
    """
    contexts = {}
    targets = []

    cache_dir = stage62 / "target_cache"
    files = sorted(cache_dir.glob("*.json.gz"))
    if len(files) != 1600:
        raise RuntimeError(f"Expected 1600 Stage62 caches, got {len(files)}")

    # ------------------------------------------------------------------
    # Held-out evaluation contexts from Stage62 caches.
    # ------------------------------------------------------------------
    for fp in files:
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            p = json.load(f)

        case_id = str(p["case_id"])
        raw_model = str(p["model"])
        model = norm_model(raw_model)
        prefix = int(p["prefix"])

        targets.append({
            "case_id": case_id,
            "model": model,
            "raw_model": raw_model,
            "prefix": prefix,
            "status": str(p["status"]),
            "pool_size": int(p["pool_size_actual"]),
            "class0_count": int(p["label_counts"].get("0", 0)),
            "class1_count": int(p["label_counts"].get("1", 0)),
        })

        res = p.get("surrogate_result")
        if res is None:
            continue

        key = (case_id, model, prefix)
        if key in contexts:
            raise RuntimeError(f"Duplicate Stage62 context: {key}")

        contexts[key] = {
            "case_id": case_id,
            "model": model,
            "prefix": prefix,
            "pool_sequences": [tuple(x) for x in p["pool"]],
            "pool_labels": [int(x) for x in p["bb_labels"]],
            "eval_indices": [int(x) for x in res["eval_indices"]],
            "factual_sequence": tuple(p["factual"]),
        }

    # ------------------------------------------------------------------
    # Finalized canonical/CF rules from Stage62 JSONL.
    # Stage62 stores rule payload nested under record["rule"]; tolerate a
    # flattened record as a defensive compatibility path.
    # ------------------------------------------------------------------
    rules_path = stage62 / "canonical_and_counterfactual_rules.jsonl"
    if not rules_path.is_file():
        raise FileNotFoundError(rules_path)

    rules = []
    seen = set()

    with rules_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            if str(rec.get("rule_type")) != "counterfactual":
                continue

            payload = rec.get("rule")
            if isinstance(payload, dict):
                r = dict(payload)
            else:
                # Compatibility with a flattened Stage62 writer.
                r = dict(rec)

            case_id = str(rec.get("case_id", r.get("case_id")))
            raw_model = str(rec.get("model", r.get("model")))
            model = norm_model(raw_model)
            prefix = int(rec.get("prefix", r.get("prefix")))
            surrogate = str(rec.get("surrogate", r.get("surrogate")))
            candidate_id = rec.get("candidate_id", r.get("candidate_id"))

            r.update({
                "case_id": case_id,
                "model": model,
                "prefix": prefix,
                "surrogate": surrogate,
                "candidate_id": candidate_id,
                "rule_type": "counterfactual",
            })

            # Stage62 preserves the native surrogate_core metric names:
            #   support / fidelity / coverage
            # while the frozen BPIC2017 Stage37 D-S interface expects:
            #   n_r / Fid_r / coverage_eval
            # These are exact semantic aliases of the same held-out local
            # evaluation quantities; no metric is recomputed or retuned here.
            if "n_r" not in r and "support" in r:
                r["n_r"] = int(r["support"])
            if "Fid_r" not in r and "fidelity" in r:
                r["Fid_r"] = r["fidelity"]
            if "coverage_eval" not in r and "coverage" in r:
                r["coverage_eval"] = float(r["coverage"])

            # Structural fields must exist for every CF rule.
            structural_required = (
                "candidate_id",
                "predicted_class",
                "conditions",
            )
            structural_missing = [
                k for k in structural_required
                if k not in r or r[k] is None
            ]
            if structural_missing:
                raise RuntimeError(
                    f"Finalized Stage62 CF rule missing structural fields "
                    f"{structural_missing} at JSONL line {line_no}: "
                    f"{case_id}/{model}/P{prefix}/{surrogate}"
                )

            # support / coverage must exist as held-out quantities.
            metric_missing = [
                k for k in ("n_r", "coverage_eval")
                if k not in r or r[k] is None
            ]
            if metric_missing:
                raise RuntimeError(
                    f"Finalized Stage62 CF rule missing held-out metrics "
                    f"{metric_missing} at JSONL line {line_no}: "
                    f"{case_id}/{model}/P{prefix}/{surrogate}"
                )

            key = (
                case_id,
                model,
                prefix,
                surrogate,
                str(candidate_id),
            )
            if key in seen:
                raise RuntimeError(f"Duplicate finalized CF rule: {key}")
            seen.add(key)

            # Fidelity may legitimately be None when held-out support is zero.
            # Such a rule remains part of the Stage62 CF rule bank but is NOT
            # eligible for frozen D-S reliability/BPA construction.
            if eligible(r):
                r["eligible_for_reliability"] = True
                rules.append(r)
            else:
                r["eligible_for_reliability"] = False

    return rules, contexts, targets, len(seen)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project_root",type=Path,default=Path("/root/autodl-tmp/ETCE"))
    ap.add_argument("--out_dir",type=Path,default=None)
    ap.add_argument("--overwrite",action="store_true")
    a=ap.parse_args()
    root=a.project_root.resolve()
    stage62=root/"results/phase4/phase4e_bpic2018_external/stage62_external_multirule_bank"
    stage63=root/"results/phase4/phase4e_bpic2018_external/stage63_external_multirule_validation"
    frozen_path=root/"configs/phase4c_ds_frozen_config.json"
    out=a.out_dir.resolve() if a.out_dir else root/"results/phase4/phase4e_bpic2018_external/stage64_external_frozen_ds_fusion"
    if out.exists() and any(out.iterdir()) and a.overwrite:
        import shutil; shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)

    v63=read_json(stage63/"validation_stage62_external_multirule.json")
    if v63.get("status")!="PASSED" or int(v63.get("target_failures",-1))!=0:
        raise RuntimeError("Stage63 validation is not cleanly PASSED")

    frozen=read_json(frozen_path)
    if frozen.get("status")!="frozen" or float(frozen.get("lambda"))!=LAMBDA or frozen.get("selection_scope")!="validation_only":
        raise RuntimeError("Frozen D-S config invariant failed")

    stage37=locate_stage37(root)
    ds=import_from_path("etce_stage37_frozen_ds_core",stage37)
    ds.validate_frozen_config(frozen)

    rules,cache,targets,total_cf_rules=extract_external(stage62)
    s62sum=read_json(stage62/"stage62_summary.json")
    expected_total_cf = int(s62sum["counterfactual_rules"])
    if total_cf_rules != expected_total_cf:
        raise RuntimeError(
            f"Total CF rule count {total_cf_rules} != Stage62 {expected_total_cf}"
        )

    write_jsonl(out/"external_eligible_cf_rules.jsonl",rules)
    write_csv(out/"external_target_applicability.csv",targets)

    print("="*124)
    print("ETCE STAGE64 - BPIC2018 EXTERNAL FROZEN HIERARCHICAL D-S")
    print("="*124)
    print(f"Frozen lambda                : {LAMBDA:g}")
    print(f"Formal targets               : {len(targets)}")
    print(f"Two-class contexts           : {len(cache)}")
    print(f"Total CF rules               : {total_cf_rules}")
    print(f"Eligible CF rules            : {len(rules)}")
    print("2017 true labels used        : NO")
    print("BPIC2018 retuning            : NO")
    print("="*124)

    stage_rows=[]; conflict_rows=[]

    # Stage 1
    s1=defaultdict(list)
    for r in rules: s1[ds.group_key_stage1(r)].append(r)
    s1f=sum(len(v)>=2 for v in s1.values())
    print(f"[Stage 1] groups={len(s1)} | fusable={s1f}")
    s1out={}
    for key in sorted(s1,key=lambda k:(k[0],MODEL_ORDER.get(k[1],99),k[2],SURROGATE_ORDER.get(k[3],99),k[4],k[5])):
        src=sorted(s1[key],key=lambda r:str(r["candidate_id"]))
        case_id,model,prefix,surr,pred,sig=key
        ctx=[cache[(case_id,model,int(prefix))]]
        cur=ds.source_rule(src[0],LAMBDA)
        for step,r in enumerate(src[1:],1):
            cur,tr=ds.pairwise_rule_fusion(cur,ds.source_rule(r,LAMBDA),ctx,LAMBDA,"Stage1",{
                "lambda":LAMBDA,"case_id":case_id,"model":model,"prefix":int(prefix),"surrogate":surr,
                "predicted_class":int(pred),"feature_signature":json.dumps(list(sig)),"step":step})
            conflict_rows.append(tr)
        s1out[key]=cur
        stage_rows.append(ds.ev_to_row(cur,LAMBDA,"Stage1",key,len(src)))

    # Stage 2
    s2=defaultdict(list)
    for key,ev in s1out.items():
        case_id,model,prefix,surr,pred,sig=key
        s2[(case_id,model,int(prefix),int(pred),sig)].append((surr,ev))
    s2f=sum(len({s for s,_ in items})>=2 for items in s2.values())
    print(f"[Stage 2] groups={len(s2)} | fusable={s2f}")
    s2out={}
    for key in sorted(s2,key=lambda k:(k[0],MODEL_ORDER.get(k[1],99),k[2],k[3],k[4])):
        case_id,model,prefix,pred,sig=key
        items=sorted(s2[key],key=lambda x:SURROGATE_ORDER.get(x[0],99))
        cur=items[0][1]; ctx=[cache[(case_id,model,int(prefix))]]
        for step,(surr,right) in enumerate(items[1:],1):
            cur,tr=ds.pairwise_rule_fusion(cur,right,ctx,LAMBDA,"Stage2",{
                "lambda":LAMBDA,"case_id":case_id,"model":model,"prefix":int(prefix),"surrogate":f"cross-surrogate:{surr}",
                "predicted_class":int(pred),"feature_signature":json.dumps(list(sig)),"step":step})
            conflict_rows.append(tr)
        s2out[key]=cur
        stage_rows.append(ds.ev_to_row(cur,LAMBDA,"Stage2",key,len(items)))

    # Stage 3
    s3=defaultdict(list)
    for key,ev in s2out.items():
        case_id,model,prefix,pred,sig=key
        s3[(case_id,int(prefix),int(pred),sig)].append((model,ev))
    s3f=sum(len({m for m,_ in items})>=2 for items in s3.values())
    print(f"[Stage 3] groups={len(s3)} | fusable={s3f}")
    s3out={}
    for key in sorted(s3,key=lambda k:(k[0],k[1],k[2],k[3])):
        case_id,prefix,pred,sig=key
        items=sorted(s3[key],key=lambda x:MODEL_ORDER.get(x[0],99))
        cur=items[0][1]; models=[items[0][0]]
        for step,(model,right) in enumerate(items[1:],1):
            models.append(model)
            ctx=[cache[(case_id,m,int(prefix))] for m in models]
            cur,tr=ds.pairwise_rule_fusion(cur,right,ctx,LAMBDA,"Stage3",{
                "lambda":LAMBDA,"case_id":case_id,"model":f"cross-model:{model}","prefix":int(prefix),"surrogate":None,
                "predicted_class":int(pred),"feature_signature":json.dumps(list(sig)),"step":step})
            conflict_rows.append(tr)
        s3out[key]=cur
        stage_rows.append(ds.ev_to_row(cur,LAMBDA,"Stage3",key,len(items)))

    summaries=[ds.summarize_stage(stage_rows,conflict_rows,s,LAMBDA) for s in ("Stage1","Stage2","Stage3")]
    fallbacks=sum(int(x["total_conflict_fallback"]) for x in conflict_rows)
    audit={
        "stage1":{"group_count":len(s1),"fusable_group_count_ge2_rules":s1f},
        "stage2":{"compatible_sets":len(s2),"sets_with_ge2_surrogates":s2f},
        "stage3":{"compatible_sets":len(s3),"sets_with_ge2_blackboxes":s3f},
    }
    write_json(out/"stage64_external_grouping_audit.json",audit)
    write_csv(out/"stage64_ds_stage_outputs.csv",stage_rows)
    write_csv(out/"stage64_ds_conflict_trace.csv",conflict_rows)
    write_csv(out/"stage64_ds_summary.csv",summaries)
    write_csv(out/"stage64_stage3_rules.csv",[r for r in stage_rows if r["stage"]=="Stage3"])

    genuine=(s1f+s2f+s3f)>0
    result={
        "scope":"ETCE BPIC2018 external frozen hierarchical D-S applicability/fusion",
        "status":"passed","formal_target_count":len(targets),"two_class_context_count":len(cache),
        "total_cf_rule_count":total_cf_rules,"eligible_cf_candidate_count":len(rules),"lambda":LAMBDA,
        "lambda_selection_performed_on_bpic2018":False,"protocol_retuning_performed_on_bpic2018":False,
        "protocol_source":str(frozen_path),"ds_implementation_source":str(stage37),
        "stage63_validation_passed":True,
        "group_counts":{"stage1":len(s1),"stage1_fusable":s1f,"stage2":len(s2),"stage2_fusable":s2f,"stage3":len(s3),"stage3_fusable":s3f},
        "pairwise_fusion_count":len(conflict_rows),"genuine_ds_fusion_occurred":genuine,
        "total_conflict_fallback_count":fallbacks,"summary":{r["stage"]:r for r in summaries},
        "external_evidence_availability_rate":len(cache)/len(targets),
        "true_2017_outcome_labels_used":False,
        "interpretation_guard":"External applicability/protocol-completeness only; N=1 two-class target cannot support broad statistical superiority claims."
    }
    write_json(out/"stage64_external_frozen_ds_fusion.json",result)

    report=[
        "# ETCE Stage64 — BPIC2018 External Frozen D-S","",
        "- Status: **PASSED**", "- Frozen lambda: **5**",
        "- BPIC2018 lambda selection: **False**","- BPIC2018 protocol retuning: **False**",
        f"- Formal targets: **{len(targets)}**",f"- Two-class contexts: **{len(cache)}**",
        f"- Total CF rules: **{total_cf_rules}**",f"- Eligible CF rules: **{len(rules)}**",f"- Evidence availability: **{100*len(cache)/len(targets):.4f}%**","",
        "## Hierarchy applicability","",
        f"- Stage1 groups / fusable: {len(s1)} / {s1f}",
        f"- Stage2 groups / fusable: {len(s2)} / {s2f}",
        f"- Stage3 groups / fusable: {len(s3)} / {s3f}",
        f"- Pairwise D-S operations: {len(conflict_rows)}",
        f"- Total-conflict fallbacks: {fallbacks}","",
        "## Interpretation","",
        "This is an external applicability/protocol-completeness check. The extremely limited evidence availability does not support a population-level performance claim."
    ]
    (out/"STAGE64_EXTERNAL_FROZEN_DS_REPORT.md").write_text("\n".join(report),encoding="utf-8")

    print("\n"+"="*124)
    print("ETCE STAGE64 EXTERNAL FROZEN D-S SUMMARY")
    print("="*124)
    print(f"Formal targets                       : {len(targets)}")
    print(f"Two-class contexts                   : {len(cache)}")
    print(f"Total CF rules                       : {total_cf_rules}")
    print(f"Eligible CF rules                    : {len(rules)}")
    print(f"Stage1 groups / fusable              : {len(s1)} / {s1f}")
    print(f"Stage2 groups / fusable              : {len(s2)} / {s2f}")
    print(f"Stage3 groups / fusable              : {len(s3)} / {s3f}")
    print(f"Pairwise D-S fusion operations       : {len(conflict_rows)}")
    print(f"Total-conflict fallbacks             : {fallbacks}")
    print(f"External evidence availability       : {100*len(cache)/len(targets):.4f}%")
    print("Lambda selected on BPIC2018          : NO")
    print("Protocol retuned on BPIC2018         : NO")
    print("True 2017 labels used                : NO")
    print("Status                               : PASSED")
    print("="*124)
    print("\nNEXT: independently validate Stage64, then freeze the BPIC2018 external-validation conclusion.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
