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
    rules=[]; contexts={}; targets=[]
    files=sorted((stage62/"target_cache").glob("*.json.gz"))
    if len(files)!=1600:
        raise RuntimeError(f"Expected 1600 Stage62 caches, got {len(files)}")
    for fp in files:
        with gzip.open(fp,"rt",encoding="utf-8") as f: p=json.load(f)
        case_id=str(p["case_id"]); raw_model=str(p["model"]); model=norm_model(raw_model); prefix=int(p["prefix"])
        targets.append({
            "case_id":case_id,"model":model,"raw_model":raw_model,"prefix":prefix,
            "status":str(p["status"]),"pool_size":int(p["pool_size_actual"]),
            "class0_count":int(p["label_counts"].get("0",0)),
            "class1_count":int(p["label_counts"].get("1",0)),
        })
        res=p.get("surrogate_result")
        if res is None: continue
        contexts[(case_id,model,prefix)]={
            "case_id":case_id,"model":model,"prefix":prefix,
            "pool_sequences":[tuple(x) for x in p["pool"]],
            "pool_labels":[int(x) for x in p["bb_labels"]],
            "eval_indices":[int(x) for x in res["eval_indices"]],
            "factual_sequence":tuple(p["factual"]),
        }
        for branch in res.get("branches",[]):
            s=str(branch["branch"])
            for rr in branch.get("counterfactual_rules") or []:
                r=dict(rr); r.update({"case_id":case_id,"model":model,"prefix":prefix,"surrogate":s,"rule_type":"counterfactual"})
                req=("candidate_id","predicted_class","conditions","n_r","Fid_r","coverage_eval")
                miss=[k for k in req if k not in r]
                if miss: raise RuntimeError(f"CF rule missing {miss}: {case_id}/{model}/P{prefix}/{s}")
                if eligible(r): rules.append(r)
    return rules,contexts,targets

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

    rules,cache,targets=extract_external(stage62)
    s62sum=read_json(stage62/"stage62_summary.json")
    if len(rules)!=int(s62sum["counterfactual_rules"]):
        raise RuntimeError(f"Eligible CF rule count {len(rules)} != Stage62 {s62sum['counterfactual_rules']}")

    write_jsonl(out/"external_eligible_cf_rules.jsonl",rules)
    write_csv(out/"external_target_applicability.csv",targets)

    print("="*124)
    print("ETCE STAGE64 - BPIC2018 EXTERNAL FROZEN HIERARCHICAL D-S")
    print("="*124)
    print(f"Frozen lambda                : {LAMBDA:g}")
    print(f"Formal targets               : {len(targets)}")
    print(f"Two-class contexts           : {len(cache)}")
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
        "eligible_cf_candidate_count":len(rules),"lambda":LAMBDA,
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
        f"- Eligible CF rules: **{len(rules)}**",f"- Evidence availability: **{100*len(cache)/len(targets):.4f}%**","",
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
