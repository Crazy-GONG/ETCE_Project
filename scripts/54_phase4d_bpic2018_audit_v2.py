#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4D-19 / Script 54 v2
BPIC2018 external-dataset audit and label/split preflight.

Purpose:
1. Locate the BPIC2018 raw event log.
2. Audit XES structure, case/event counts, years, activities, and decision boundary.
3. Reconstruct the published late-payment label as an AUDIT CANDIDATE only.
4. Check whether the proposed current ETCE prefixes (5/10/15/20) can be used
   without crossing the first decision event.
5. Do NOT train models, build surrogates, tune parameters, or evaluate TEST.

The external-data experiment remains blocked until the raw-log audit is reviewed.
"""
from __future__ import annotations

import csv, gzip, hashlib, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase4/phase4d_external_bpic2018_audit"
PREFIXES = (5, 10, 15, 20)

CANDIDATE_PATHS = [
    ROOT / "data/raw/bpic2018",
    ROOT / "data/raw/BPIC2018",
    ROOT / "data/raw/BPI Challenge 2018.xes.gz",
    ROOT / "data/raw/BPI_Challenge_2018.xes.gz",
    ROOT / "data/raw/bpic2018.xes.gz",
    ROOT / "data/raw/BPIC2018.xes.gz",
]

ACTIVITY_ALIASES = {
    "begin_payment": {"begin payment", "begin payment-application", "begin payment"},
    "abort_payment": {"abort payment", "abort payment-application", "abort payment"},
    "finish_payment": {"finish payment", "finish payment-application", "finish payment"},
    "first_decision": {"decide", "sealed decide", "Payment application-Application-decide", "Payment application-Application-sealed decide"},
}

def norm(x):
    return str(x).strip().lower().replace("_", " ")

def attr_ci(attrs, names):
    """Case-insensitive attribute lookup with lightweight key normalization."""
    if not isinstance(attrs, dict):
        return None
    wanted={norm(n).replace(":", " ") for n in names}
    for k,v in attrs.items():
        nk=norm(k).replace(":", " ")
        if nk in wanted:
            return v
    return None

def case_year(case):
    """Prefer the explicit BPIC2018 case-level year attribute; retain timestamp fallback for audit."""
    v=attr_ci(case.get("attrs",{}), ["year", "case:year", "case year"])
    if v is not None:
        try:
            return int(float(str(v))), "case_attribute"
        except Exception:
            pass
    ev=case.get("events",[])
    if ev:
        return int(ev[0]["timestamp_obj"].year), "first_timestamp_fallback"
    return None, "missing"

def event_doc_context(e):
    attrs=e.get("attrs",{}) if isinstance(e,dict) else {}
    doctype=attr_ci(attrs,["doctype","document type","document_type","doc type"])
    subprocess=attr_ci(attrs,["subprocess","sub process","sub_process"])
    return norm(doctype) if doctype is not None else None, norm(subprocess) if subprocess is not None else None

def is_payment_application_event(e, allowed_activity_names):
    act=norm(e.get("activity",""))
    allowed={norm(x) for x in allowed_activity_names}
    if act not in allowed:
        return False
    doctype,subprocess=event_doc_context(e)
    # If context attributes are available, require Payment application / Application.
    if doctype is not None and subprocess is not None:
        return doctype=="payment application" and subprocess=="application"
    return True

def find_payment_activity(events, aliases):
    return [i for i,e in enumerate(events) if is_payment_application_event(e, aliases)]

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def locate():
    files=[]
    seen=set()
    for p in CANDIDATE_PATHS:
        p=Path(p)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for pat in ("*.xes","*.xes.gz","*.csv"):
                for q in sorted(p.glob(pat)):
                    if q.is_file() and q not in seen:
                        files.append(q); seen.add(q)
    # also recursively inspect data/raw/bpic2018 if present
    for base in (ROOT/"data/raw", ROOT/"data"):
        if base.is_dir():
            for q in base.rglob("*"):
                if q.is_file() and q.suffix.lower() in {".xes",".csv"} or q.name.lower().endswith(".xes.gz"):
                    s=str(q).lower()
                    if "bpic2018" in s or "bpi challenge 2018" in s:
                        if q not in seen:
                            files.append(q); seen.add(q)
    return sorted(files, key=lambda x:(0 if x.suffix.lower() in {".xes",".gz"} or x.name.endswith(".xes.gz") else 1, str(x)))

def load_xes(path):
    try:
        from pm4py.objects.log.importer.xes import importer as xes_importer
    except Exception as e:
        raise RuntimeError("pm4py is required to audit XES. Install it in the ETCE environment.") from e
    log=xes_importer.apply(
        str(path),
        variant=xes_importer.Variants.ITERPARSE,
        parameters={"timestamp_sort":False,"show_progress_bar":False},
    )
    cases=[]
    for trace in log:
        cid=trace.attributes.get("concept:name")
        if cid is None:
            raise ValueError("Trace without concept:name.")
        attrs=dict(trace.attributes)
        events=[]
        for i,e in enumerate(trace):
            act=e.get("concept:name")
            ts=e.get("time:timestamp")
            if act is None or ts is None:
                raise ValueError(f"Missing concept:name/time:timestamp in case {cid!r}, event {i}.")
            events.append({
                "activity":str(act),
                "timestamp":ts.isoformat() if hasattr(ts,"isoformat") else str(ts),
                "timestamp_obj":ts,
                "attrs":dict(e),
            })
        cases.append({"case_id":str(cid),"attrs":attrs,"events":events})
    return cases

def load_csv(path):
    import pandas as pd
    df=pd.read_csv(path, low_memory=False)
    cols=list(df.columns)
    lower={c.lower():c for c in cols}
    def pick(names):
        for n in names:
            if n.lower() in lower: return lower[n.lower()]
        return None
    case=pick(["case:concept:name","case_id","caseid","case id","case"])
    act=pick(["concept:name","activity","event","activity_name","event_name"])
    ts=pick(["time:timestamp","timestamp","event_time","time"])
    if not (case and act and ts):
        raise ValueError(f"CSV needs case/activity/timestamp columns. Columns={cols}")
    df[ts]=pd.to_datetime(df[ts], errors="coerce", utc=True)
    if df[ts].isna().any(): raise ValueError("CSV contains unparseable timestamps.")
    cases=[]
    for cid,g in df.groupby(case,sort=False):
        # Preserve file row order. Do not silently sort events.
        ev=[]
        for _,r in g.iterrows():
            ev.append({"activity":str(r[act]),"timestamp":r[ts].isoformat(),"timestamp_obj":r[ts],"attrs":{}})
        cases.append({"case_id":str(cid),"attrs":{},"events":ev})
    return cases

def load(path):
    if path.name.lower().endswith(".xes") or path.name.lower().endswith(".xes.gz"):
        return load_xes(path)
    if path.suffix.lower()==".csv":
        return load_csv(path)
    raise ValueError(f"Unsupported file: {path}")

def find_activity(events, aliases):
    target={norm(x) for x in aliases}
    return [i for i,e in enumerate(events) if norm(e["activity"]) in target]

def label_candidate(case):
    ev=case["events"]
    start_year,_year_source=case_year(case)
    begin=find_payment_activity(ev,ACTIVITY_ALIASES["begin_payment"])
    abort=find_payment_activity(ev,ACTIVITY_ALIASES["abort_payment"])
    finish=find_payment_activity(ev,ACTIVITY_ALIASES["finish_payment"])
    if not begin:
        return 1, "no_begin_payment", {"begin":0,"abort":len(abort),"finish":len(finish)}
    last=begin[-1]
    immediate=(last+1<len(ev) and is_payment_application_event(ev[last+1], ACTIVITY_ALIASES["abort_payment"]))
    later_abort=any(j>last for j in abort)
    last_begin_year=ev[last]["timestamp_obj"].year
    if immediate:
        return 1, "last_begin_immediately_followed_by_abort", {"begin":len(begin),"abort":len(abort),"finish":len(finish)}
    if later_abort:
        return 1, "last_begin_followed_later_by_abort", {"begin":len(begin),"abort":len(abort),"finish":len(finish)}
    if last_begin_year>start_year:
        return 1, "last_begin_payment_later_year", {"begin":len(begin),"abort":len(abort),"finish":len(finish)}
    return 0, "on_time_candidate", {"begin":len(begin),"abort":len(abort),"finish":len(finish)}

def audit(cases):
    activity=Counter()
    years=Counter()
    first_timestamp_years=Counter()
    year_sources=Counter()
    lengths=[]
    prefix=defaultdict(lambda:Counter())
    label_reason=Counter()
    year_label=defaultdict(lambda:Counter())
    decision_boundary=Counter()
    decision_before_prefix=defaultdict(int)
    begin_before_decision=0
    timestamp_reversal_cases=0
    timestamp_reversal_pairs=0
    decision_contexts=Counter()
    for c in cases:
        ev=c["events"]; lengths.append(len(ev))
        if not ev: continue
        cy,ysrc=case_year(c)
        years[cy]+=1
        year_sources[ysrc]+=1
        first_timestamp_years[ev[0]["timestamp_obj"].year]+=1
        for e in ev: activity[e["activity"]]+=1
        reversals=sum(1 for a,b in zip(ev,ev[1:]) if b["timestamp_obj"] < a["timestamp_obj"])
        timestamp_reversal_pairs += reversals
        timestamp_reversal_cases += int(reversals>0)
        dec=find_payment_activity(ev,ACTIVITY_ALIASES["first_decision"])
        # Diagnostic: capture all decide-like activities and their document/subprocess context.
        for e in ev:
            if "decide" in norm(e["activity"]):
                dc,sp=event_doc_context(e)
                decision_contexts[(e["activity"], dc or "<missing>", sp or "<missing>")] += 1
        if dec:
            first_dec=dec[0]; decision_boundary["has_first_decision"]+=1
            for k in PREFIXES:
                if len(ev)>=k and first_dec<k:
                    decision_before_prefix[k]+=1
        else:
            decision_boundary["no_first_decision"]+=1
        begin=find_payment_activity(ev,ACTIVITY_ALIASES["begin_payment"])
        dec=find_payment_activity(ev,ACTIVITY_ALIASES["first_decision"])
        if begin and dec and begin[0] < dec[0]:
            begin_before_decision+=1
        y,reason,meta=label_candidate(c)
        label_reason[reason]+=1
        year_label[cy][y]+=1
        for k in PREFIXES:
            prefix[k]["eligible_case"] += int(len(ev)>=k)
            prefix[k]["no_decision_in_prefix"] += int(len(ev)>=k and (not dec or dec[0]>=k))
    return {
      "case_count":len(cases),
      "event_count":sum(lengths),
      "year_case_counts":dict(sorted(years.items())),
      "first_timestamp_year_case_counts":dict(sorted(first_timestamp_years.items())),
      "year_source_counts":dict(year_sources),
      "trace_length":{"min":min(lengths) if lengths else None,"median":sorted(lengths)[len(lengths)//2] if lengths else None,"max":max(lengths) if lengths else None},
      "top_activities":activity.most_common(50),
      "candidate_late_label_total":dict(label_reason),
      "candidate_late_label_by_year":{str(y):dict(v) for y,v in sorted(year_label.items())},
      "first_decision":{"counts":dict(decision_boundary),"begin_payment_before_first_decision_cases":begin_before_decision},
      "prefix_audit":{str(k):dict(v) for k,v in prefix.items()},
      "decision_in_prefix_cases":{str(k):decision_before_prefix[k] for k in PREFIXES},
      "timestamp_reversal_cases":timestamp_reversal_cases,
      "timestamp_reversal_pairs":timestamp_reversal_pairs,
      "decide_like_contexts":[{"activity":a,"doctype":d,"subprocess":sp,"event_count":n} for (a,d,sp),n in decision_contexts.most_common(30)],
    }

def write_csv(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0])
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    files=locate()
    print("="*118)
    print("ETCE PHASE 4D-19 BPIC2018 EXTERNAL-DATA AUDIT v2")
    print("="*118)
    if not files:
        print("NO BPIC2018 RAW LOG FOUND")
        print("Expected under: D:\\ETCE\\data\\raw\\bpic2018\\ or D:\\ETCE\\data\\raw\\")
        print("No external experiment has been run.")
        (OUT/"audit_status.json").write_text(json.dumps({"status":"BLOCKED_NO_RAW_LOG"},indent=2),encoding="utf-8")
        return 2
    print(f"Candidate files found: {len(files)}")
    for i,p in enumerate(files,1):
        print(f"[{i}] {p} | {p.stat().st_size:,} bytes | sha256={sha256(p)}")
    path=files[0]
    print(f"Using first candidate: {path}")
    cases=load(path)
    report=audit(cases)
    report.update({"status":"AUDIT_COMPLETE","source_path":str(path),"sha256":sha256(path)})
    (OUT/"bpic2018_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    write_csv(OUT/"bpic2018_top_activities.csv",[{"activity":a,"event_count":n} for a,n in report["top_activities"]])
    write_csv(OUT/"bpic2018_year_label_candidate.csv",[
        {"year":y,"negative":v.get(0,0),"positive":v.get(1,0)}
        for y,v in report["candidate_late_label_by_year"].items()
    ])
    print(f"Cases={report['case_count']:,} | Events={report['event_count']:,}")
    print(f"Case-year attribute/fallback={report['year_case_counts']} | source={report['year_source_counts']}")
    print(f"First timestamp years={report['first_timestamp_year_case_counts']}")
    print(f"TraceLen={report['trace_length']}")
    print(f"Timestamp reversal cases={report['timestamp_reversal_cases']:,} | reversal pairs={report['timestamp_reversal_pairs']:,}")
    print(f"First decision={report['first_decision']}")
    print("[Decide-like activity/context diagnostics]")
    if report["decide_like_contexts"]:
        for r in report["decide_like_contexts"][:20]:
            print(f"  activity={r['activity']!r} | doctype={r['doctype']!r} | subprocess={r['subprocess']!r} | events={r['event_count']:,}")
    else:
        print("  NONE")
    print("[Candidate late-payment label]")
    for k,v in report["candidate_late_label_total"].items(): print(f"  {k:45s} {v:,}")
    print("[Year x candidate label]")
    for y,v in report["candidate_late_label_by_year"].items():
        neg=v.get(0,0); pos=v.get(1,0); n=neg+pos
        rate=(pos/n) if n else float("nan")
        print(f"  {y}: N={n:,} | late={pos:,} ({rate:.4%})")
    print("[Current ETCE prefix audit]")
    for k in PREFIXES:
        x=report["prefix_audit"][str(k)]
        print(f"  P{k}: eligible={x.get('eligible_case',0):,} | no_decision_in_prefix={x.get('no_decision_in_prefix',0):,} | decision_in_prefix={report['decision_in_prefix_cases'][str(k)]:,}")
    print()
    print("IMPORTANT:")
    print("1) Candidate late-payment label is AUDIT ONLY, not yet frozen.")
    print("2) 2015/2016/2017 split is NOT changed or frozen by this script.")
    print("3) No model/surrogate/fusion/test evaluation was performed.")
    print("4) Prefixes crossing first decision must be excluded before formal training.")
    print(f"Output: {OUT}")
    print("NEXT: review audit; then freeze BPIC2018 label + 2015 train / 2016 validation / 2017 test if supported.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
