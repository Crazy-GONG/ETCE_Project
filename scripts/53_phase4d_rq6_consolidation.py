#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-18 / Script 53
RQ6 consolidated ablation + statistical significance/effect-size tables.

No new training or TEST-side tuning is performed.

Outputs
-------
Table8_Hypothesis_Driven_Ablation.csv
Table9_Statistical_Significance_Effect_Size.csv
RQ6_Secondary_Component_Evidence.csv
Figure2_Three_Seed_Stability_Data.csv
protocol_alignment_note.json
rq6_consolidation_summary.json

Protocol-alignment note
-----------------------
The old plan listed "w/o Feature-aware Selection". The CURRENT frozen Phase4D
implementation has no separate feature-aware post-selector; representative
selection is highest proper focal mass -> specificity -> lexical tie-break.
Therefore this script does NOT fabricate that obsolete ablation.
"""

from __future__ import annotations
import csv, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "results/phase4/phase4d_formal_test"
DS_STAGE = FORMAL / "formal_test_frozen_ds_fusion/phase4d_frozen_ds_stage_outputs.csv"
BASELINE = FORMAL / "formal_test_fusion_baselines/formal_fusion_strategy_comparison_stage3.csv"
PROPOSAL = FORMAL / "formal_test_process_aware_proposal_ablation"
PN = FORMAL / "formal_test_petri_net_conformance"
MINI = FORMAL / "formal_test_minimality_proximity"
STAB = FORMAL / "formal_test_multiseed_stability"
OUT = FORMAL / "formal_test_rq6_consolidation"

# Completed Phase4D-9 formal paired Fidelity results.
FUSION = {
"best_single_evidence": dict(full=.668218, base=.665629, delta=.002589, lo=-.000842, hi=.006824, p=.567567, holm=1.0, rrb=.1282, w=15,t=503,l=11),
"validation_best_source": dict(full=.668218, base=.681075, delta=-.012858, lo=-.023446, hi=-.002582, p=.0353868, holm=.212321, rrb=-.2885, w=30,t=459,l=40),
"uniform_vote": dict(full=.668218, base=.668035, delta=.000182, lo=-.005074, hi=.005295, p=.959511, holm=1.0, rrb=-.0094, w=21,t=491,l=17),
"validation_weighted_vote": dict(full=.668218, base=.668152, delta=.000065, lo=-.007691, hi=.007490, p=.639114, holm=1.0, rrb=.0831, w=25,t=487,l=17),
"fuzzy_probabilistic_or": dict(full=.668218, base=.665303, delta=.002915, lo=-.000756, hi=.007093, p=.422118, holm=1.0, rrb=.1556, w=21,t=494,l=14),
"bayesian_categorical_noise": dict(full=.668218, base=.664714, delta=.003504, lo=-.001031, hi=.008889, p=.700243, holm=1.0, rrb=.0736, w=20,t=493,l=16),
}

# Completed Phase4D-10 formal support-aware ablation results.
SUPPORT = {
"rule_fidelity": dict(full=.668218, abl=.704924, delta=-.036706, lo=-.052806, hi=-.021031, p=5.7252e-06, rrb=-.4341),
"model_level_validity": dict(full=.849874, abl=.829710, delta=.020164, lo=.004568, hi=.035759, p=.00985071, rrb=.3325),
"all_model_validity": dict(full=.699433, abl=.654064, delta=.045369, p=.00427241),
"coverage": dict(full=.293846, abl=.257642, delta=.036204, p=3.28631e-15),
"support": dict(full=24.805293, abl=16.837429, delta=7.967864, p=1.72714e-20),
}

def req(c,m):
    if not c: raise AssertionError(m)

def rcsv(p):
    with Path(p).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def wcsv(p,rows):
    rows=list(rows); p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if not rows: p.write_text("",encoding="utf-8"); return
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def wjson(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")

def fb(v):
    return str(v).strip().lower() in {"1","true","yes","y"} if not isinstance(v,bool) else v

def ff(v):
    if v is None or str(v).strip().lower() in {"","none","null","nan"}: return None
    return float(v)

def avg(xs):
    a=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(a)/len(a) if a else None

def find(rows, **kw):
    for r in rows:
        if all(str(r.get(k,""))==str(v) for k,v in kw.items()): return r
    return None

def holm(ps):
    m=len(ps); order=sorted(enumerate(map(float,ps)),key=lambda z:z[1])
    out=[None]*m; running=0.0
    for rank,(i,p) in enumerate(order):
        running=max(running,(m-rank)*p); out[i]=min(1.0,running)
    return out

def full_metrics():
    rows=rcsv(DS_STAGE)
    s=[r for r in rows if r.get("stage")=="Stage3" and fb(r.get("actually_fused"))]
    req(len(s)==529,f"Expected 529 fused Stage3 rows, found {len(s)}")
    return dict(
        fidelity=avg(ff(r.get("Fid_r")) for r in s),
        cf=avg(int(fb(r.get("cf_valid"))) for r in s),
        model=avg(ff(r.get("model_cf_validity_rate")) for r in s),
        allmodel=avg(int(fb(r.get("all_models_cf_valid"))) for r in s),
        R=avg(ff(r.get("R_r")) for r in s),
        coverage=avg(ff(r.get("coverage_eval")) for r in s),
        support=avg(ff(r.get("n_r")) for r in s),
    )

def process_sources():
    paths={
      "pcont":PROPOSAL/"proposal_ablation_paired_continuous.csv",
      "pbin":PROPOSAL/"proposal_ablation_paired_binary.csv",
      "pncont":PN/"pn_conformance_paired_continuous.csv",
      "pnbin":PN/"pn_conformance_paired_binary.csv",
    }
    for p in paths.values(): req(p.is_file(),f"Missing {p}")
    return {k:rcsv(v) for k,v in paths.items()}

def build_table8(proc):
    pn=find(proc["pnbin"],metric="has_pn_and_dfg_fit_flip")
    req(pn is not None,"Missing has_pn_and_dfg_fit_flip")
    rows=[
      dict(component="Evidence Fusion",variant="w/o Evidence Fusion (best_single_evidence)",
           primary_endpoint="Stage3 Rule Fidelity",
           full_etce=FUSION["best_single_evidence"]["full"],
           ablated=FUSION["best_single_evidence"]["base"],
           delta_full_minus_ablated=FUSION["best_single_evidence"]["delta"],
           ci95_lo=FUSION["best_single_evidence"]["lo"],ci95_hi=FUSION["best_single_evidence"]["hi"],
           p_raw=FUSION["best_single_evidence"]["p"],effect_type="rank-biserial",
           effect_value=FUSION["best_single_evidence"]["rrb"],
           interpretation="No detectable mean-Fidelity gain over best-single evidence."),
      dict(component="Support-aware Reliability",variant="w/o Support Adjustment (R=Fid)",
           primary_endpoint="Stage3 all-model validity",
           full_etce=SUPPORT["all_model_validity"]["full"],
           ablated=SUPPORT["all_model_validity"]["abl"],
           delta_full_minus_ablated=SUPPORT["all_model_validity"]["delta"],
           ci95_lo=None,ci95_hi=None,p_raw=SUPPORT["all_model_validity"]["p"],
           effect_type="paired risk difference",effect_value=SUPPORT["all_model_validity"]["delta"],
           interpretation="Improves cross-model validity/coverage/support but lowers local Fidelity."),
      dict(component="Process-aware Proposal Constraints",variant="w/o Process-aware Proposal",
           primary_endpoint="Target with >=1 PN+DFG-valid prediction-flipping CF",
           full_etce=ff(pn["full_rate"]),ablated=ff(pn["ablated_rate"]),
           delta_full_minus_ablated=ff(pn["delta_full_minus_ablated"]),
           ci95_lo=None,ci95_hi=None,p_raw=ff(pn["p_value"]),
           effect_type="paired risk difference",effect_value=ff(pn["delta_full_minus_ablated"]),
           interpretation="Large process-validity contribution; nominal GSR itself is not improved."),
    ]
    adj=holm([r["p_raw"] for r in rows])
    for r,a in zip(rows,adj):
        r["p_holm_rq6_primary"]=a; r["significant_after_holm_0.05"]=a<.05
    return rows

def fusion_table9():
    methods=list(FUSION); adj=holm([FUSION[m]["p"] for m in methods]); out=[]
    for m,a in zip(methods,adj):
        x=FUSION[m]
        req(abs(a-x["holm"])<5e-6,f"Holm mismatch for {m}: {a} vs {x['holm']}")
        out.append(dict(family="Fusion Fidelity",contrast=f"D-S ETCE vs {m}",endpoint="Stage3 Rule Fidelity",
          N=529,full_etce=x["full"],comparison=x["base"],delta=x["delta"],
          ci95_lo=x["lo"],ci95_hi=x["hi"],wins=x["w"],ties=x["t"],losses=x["l"],
          p_raw=x["p"],p_holm=a,effect_type="rank-biserial",effect_value=x["rrb"],
          **{"significant_after_holm_0.05": a < .05}))
    return out

def minimality_table9():
    p=MINI/"stage3_minimality_paired_comparison.csv"; req(p.is_file(),f"Missing {p}")
    rows=[r for r in rcsv(p) if r.get("metric")=="sequence_edit_count"]
    req(len(rows)==6,f"Expected 6 sequence-edit comparisons, found {len(rows)}")
    adj=holm([ff(r["p_value"]) for r in rows]); out=[]
    for r,a in zip(rows,adj):
        w=int(float(r["ds_wins_lower_is_better"])); t=int(float(r["ties"])); l=int(float(r["baseline_wins"]))
        eff=(w-l)/(w+l) if w+l else 0.0
        out.append(dict(family="Stage3 Sequence Edit",contrast=f"D-S ETCE vs {r['baseline_method']}",
          endpoint="Sequence edit count",N=int(float(r["N_usable"])),
          full_etce=ff(r["ds_mean"]),comparison=ff(r["baseline_mean"]),
          delta=ff(r["delta_ds_minus_baseline"]),ci95_lo=None,ci95_hi=None,
          wins=w,ties=t,losses=l,p_raw=ff(r["p_value"]),p_holm=a,
          effect_type="non-tie directional effect",effect_value=eff,
          **{"significant_after_holm_0.05": a < .05}))
    return out

def secondary(proc):
    P={r["metric"]:r for r in proc["pcont"]}; B={r["metric"]:r for r in proc["pbin"]}
    PC={r["metric"]:r for r in proc["pncont"]}; PB={r["metric"]:r for r in proc["pnbin"]}
    out=[]
    for k,x in SUPPORT.items():
        out.append(dict(component="Support-aware Reliability",metric=k,full=x["full"],ablated=x["abl"],
                        delta=x["delta"],p_raw=x["p"]))
    specs=[
      ("flow_valid_candidate_rate",P["flow_valid_candidate_rate"]),
      ("process_valid_target_rate",B["has_process_valid_flip"]),
      ("process_invalid_CF_target_rate",B["has_process_invalid_flip"]),
      ("mean_PN_fitness_flips",PC["mean_pn_fitness_flips"]),
      ("PN_fit_CF_target_rate",PB["has_pn_fit_flip"]),
      ("PN_and_DFG_fit_CF_target_rate",PB["has_pn_and_dfg_fit_flip"]),
    ]
    for name,r in specs:
        if "full_mean" in r:
            full,abl,delta=ff(r["full_mean"]),ff(r["ablated_mean"]),ff(r["delta_full_minus_ablated"])
        else:
            full,abl,delta=ff(r["full_rate"]),ff(r["ablated_rate"]),ff(r["delta_full_minus_ablated"])
        out.append(dict(component="Process-aware Proposal Constraints",metric=name,full=full,ablated=abl,
                        delta=delta,p_raw=ff(r["p_value"])))
    return out

def stability():
    ms=STAB/"multiseed_stage3_mean_std.csv"; req(ms.is_file(),f"Missing {ms}")
    rows=rcsv(ms); idx={r["metric"]:r for r in rows}
    keys=["stage3_mean_fidelity","stage3_cf_validity","stage3_model_validity","stage3_all_model_validity",
          "stage3_mean_reliability","stage3_mean_ignorance","stage3_mean_min_witness_edit","stage3_mean_conflict_K"]
    return [dict(metric=k,mean=ff(idx[k]["mean"]),std=ff(idx[k]["std"]),seed_count=int(float(idx[k]["seed_count"]))) for k in keys]

def main():
    full=full_metrics()
    req(abs(full["fidelity"]-.668218)<5e-6,"Full Fidelity cross-check failed")
    req(abs(full["allmodel"]-.699433)<5e-6,"Full all-model validity cross-check failed")
    proc=process_sources()
    t8=build_table8(proc)
    t9_primary=[]
    for r in t8:
        t9_primary.append(dict(family="RQ6 Primary Components",contrast=f"Full ETCE vs {r['variant']}",
          endpoint=r["primary_endpoint"],N="",full_etce=r["full_etce"],comparison=r["ablated"],
          delta=r["delta_full_minus_ablated"],ci95_lo=r["ci95_lo"],ci95_hi=r["ci95_hi"],
          wins="",ties="",losses="",p_raw=r["p_raw"],p_holm=r["p_holm_rq6_primary"],
          effect_type=r["effect_type"],effect_value=r["effect_value"],
          **{"significant_after_holm_0.05": r["significant_after_holm_0.05"]}))
    t9=t9_primary+fusion_table9()+minimality_table9()
    sec=secondary(proc); stab=stability()

    note=dict(
      status="CURRENT_PROTOCOL_ALIGNMENT",
      old_planned_variant="w/o Feature-aware Selection",
      separate_feature_aware_postselector_present=False,
      current_selection="highest normalized proper focal mass -> more specific -> lexical tie-break",
      action="Do not fabricate the obsolete ablation; report sparsity/proximity as evaluation metrics."
    )

    OUT.mkdir(parents=True,exist_ok=True)
    wcsv(OUT/"Table8_Hypothesis_Driven_Ablation.csv",t8)
    wcsv(OUT/"Table9_Statistical_Significance_Effect_Size.csv",t9)
    wcsv(OUT/"RQ6_Secondary_Component_Evidence.csv",sec)
    wcsv(OUT/"Figure2_Three_Seed_Stability_Data.csv",stab)

    for src,name in [
      (STAB/"multiseed_group_stability_pairwise.csv","Figure2_Group_Stability_Pairwise.csv"),
      (STAB/"multiseed_target_footprint_stability_pairwise.csv","Figure2_Target_Footprint_Stability_Pairwise.csv")
    ]:
        req(src.is_file(),f"Missing {src}"); wcsv(OUT/name,rcsv(src))

    wjson(OUT/"protocol_alignment_note.json",note)
    wjson(OUT/"rq6_consolidation_summary.json",dict(
      status="passed",full_stage3=full,table8=t8,table9=t9,stability=stab,
      protocol_alignment=note,test_side_tuning=False,protocol_retuning=False,
      next_step="external-dataset validation then final paper-table regeneration"
    ))

    lines=[
      "# ETCE RQ6 Consolidated Results","",
      "## Table 8. Hypothesis-driven Ablation","",
      "| Component | Ablation | Primary endpoint | Full | Ablated | Delta | Raw p | Holm p |",
      "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in t8:
        lines.append(f"| {r['component']} | {r['variant']} | {r['primary_endpoint']} | "
                     f"{r['full_etce']:.6f} | {r['ablated']:.6f} | {r['delta_full_minus_ablated']:+.6f} | "
                     f"{r['p_raw']:.3g} | {r['p_holm_rq6_primary']:.3g} |")
    lines += ["","## Three-seed robustness","",
              "| Metric | Mean | Std |","|---|---:|---:|"]
    for r in stab:
        lines.append(f"| {r['metric']} | {r['mean']:.6f} | {r['std']:.6f} |")
    (OUT/"PHASE4D_RQ6_CONSOLIDATED_REPORT.md").write_text("\n".join(lines),encoding="utf-8")

    print("="*118)
    print("ETCE PHASE 4D-18 RQ6 CONSOLIDATION PASSED")
    print("="*118)
    print("[Table 8 - Primary component ablations]")
    for r in t8:
        print(f"{r['component']:36s} | Full={r['full_etce']:.6f} | Abl={r['ablated']:.6f} | "
              f"Delta={r['delta_full_minus_ablated']:+.6f} | p={r['p_raw']:.6g} | "
              f"Holm={r['p_holm_rq6_primary']:.6g} | Sig={'YES' if r['significant_after_holm_0.05'] else 'NO'}")
    print()
    print("[Fusion Fidelity Holm family]")
    for r in [x for x in t9 if x["family"]=="Fusion Fidelity"]:
        print(f"{r['contrast']:55s} | Delta={r['delta']:+.6f} | p={r['p_raw']:.6g} | Holm={r['p_holm']:.6g} | r_rb={r['effect_value']:+.4f}")
    print()
    print("[Sequence Edit Holm family]")
    for r in [x for x in t9 if x["family"]=="Stage3 Sequence Edit"]:
        print(f"{r['contrast']:55s} | Delta={r['delta']:+.6f} | p={r['p_raw']:.6g} | Holm={r['p_holm']:.6g}")
    print()
    print("[Three-seed mean ± std]")
    for r in stab:
        print(f"{r['metric']:34s} | {r['mean']:.6f} ± {r['std']:.6f}")
    print()
    print(f"Output: {OUT}")
    print("Feature-aware post-selector in current frozen protocol: False")
    print("TEST-side tuning performed: False")
    print("Protocol retuning performed: False")
    print("NEXT: external-dataset validation.")

if __name__ == "__main__":
    main()
