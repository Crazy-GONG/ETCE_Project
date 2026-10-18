#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Stage58
BPIC2018 Final Refit + Formal External TEST

Frozen protocol:
- Stage56 FINAL representation only.
- Stage57/BPIC2017 frozen model configurations only.
- RF / XGB / MLP / LSTM.
- Prefixes P5/P10/P15/P20.
- Seeds [13,21,42,73,101].
- Threshold 0.5.
- Final fit: 2015+2016.
- Formal external TEST: 2017.
- No TEST-based tuning.
- LSTM epoch count is frozen from each Stage57 development best_epoch
  BEFORE 2017 arrays are loaded.
"""

from __future__ import annotations
import argparse, csv, gzip, hashlib, json, math, random, statistics, time, warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import joblib
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score,
    confusion_matrix, log_loss
)

STAGE = 58
MODELS = ("RF", "XGB", "MLP", "LSTM")
PREFIXES = (5, 10, 15, 20)
SEEDS = (13, 21, 42, 73, 101)
THRESHOLD = 0.5


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(p: Path, x: Any) -> None:
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def ensure_clean_dir(p: Path, overwrite: bool) -> None:
    if p.exists() and any(p.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output exists: {p}. Use --overwrite only for exact rebuild.")
        import shutil
        for x in list(p.iterdir()):
            shutil.rmtree(x) if x.is_dir() else x.unlink()
    p.mkdir(parents=True, exist_ok=True)

def load_npz(p: Path) -> Dict[str, np.ndarray]:
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick(d: Dict[str, Any], *names: str, default=None):
    for n in names:
        if n in d:
            return d[n]
    return default


def verify_stage56(d: Path) -> Dict[str, Any]:
    for n in ("protocol_frozen.json","dataset_summary.json","consistency_checks.json",
              "sha256_manifest.json","mapping_final_fit2015_2016.json"):
        if not (d/n).exists():
            raise FileNotFoundError(d/n)
    p = read_json(d/"protocol_frozen.json")
    s = read_json(d/"dataset_summary.json")
    c = read_json(d/"consistency_checks.json")
    if p.get("stage") != 56 or not p.get("all_hard_checks_pass", False):
        raise RuntimeError("Stage56 protocol is not frozen/PASS.")
    if not s.get("all_hard_checks_pass", False):
        raise RuntimeError("Stage56 summary is not PASS.")
    if any(not v.get("pass",False) for v in c.values()):
        raise RuntimeError("Stage56 checks are not fully PASS.")
    return {"protocol":p,"summary":s}

def verify_stage57(d: Path) -> Dict[str, Any]:
    for n in ("protocol_frozen.json","consistency_checks.json",
              "frozen_config_import.json","development_metrics_seedwise.csv",
              "development_metrics_summary.csv","sha256_manifest.json"):
        if not (d/n).exists():
            raise FileNotFoundError(d/n)
    p = read_json(d/"protocol_frozen.json")
    c = read_json(d/"consistency_checks.json")
    i = read_json(d/"frozen_config_import.json")
    if p.get("stage") != 57 or not p.get("all_hard_checks_pass",False):
        raise RuntimeError("Stage57 is not frozen/PASS.")
    if any(not v.get("pass",False) for v in c.values()):
        raise RuntimeError("Stage57 checks are not fully PASS.")
    if p.get("formal_test_loaded_in_stage57", True):
        raise RuntimeError("Stage57 reports 2017 was loaded; protocol invalid.")
    if i.get("source_sha256") != p.get("frozen_config_sha256"):
        raise RuntimeError("Stage57 frozen-config provenance mismatch.")
    return {"protocol":p,"checks":c,"imported":i}


def frozen_configs(imported: Dict[str, Any]) -> Dict[str, Any]:
    r = imported.get("resolved")
    if not isinstance(r, dict):
        raise RuntimeError("No resolved frozen configs in Stage57 import.")
    missing = [f"{m}/P{p}" for m in MODELS for p in PREFIXES
               if m not in r or str(p) not in r[m]]
    if missing:
        raise RuntimeError("Missing frozen configs: " + ", ".join(missing))
    return r


def clean_rf(raw: Dict[str,Any], seed:int) -> Dict[str,Any]:
    allowed = {
        "n_estimators","criterion","max_depth","min_samples_split","min_samples_leaf",
        "min_weight_fraction_leaf","max_features","max_leaf_nodes","min_impurity_decrease",
        "bootstrap","oob_score","n_jobs","random_state","verbose","warm_start",
        "class_weight","ccp_alpha","max_samples","monotonic_cst"
    }
    p={k:v for k,v in raw.items() if k in allowed}
    p["random_state"]=seed; p.setdefault("n_jobs",-1)
    return p

def clean_xgb(raw: Dict[str,Any], seed:int) -> Dict[str,Any]:
    controls={"seed","epochs","batch_size","patience","early_stopping_rounds",
              "threshold","prefix","model","model_name"}
    p={k:v for k,v in raw.items() if k not in controls}
    p["random_state"]=seed; p.setdefault("n_jobs",-1); p.setdefault("eval_metric","logloss")
    return p

def clean_mlp(raw: Dict[str,Any], seed:int) -> Dict[str,Any]:
    p=dict(raw)
    if "hidden_dims" in p and "hidden_layer_sizes" not in p:
        v=p.pop("hidden_dims"); p["hidden_layer_sizes"]=tuple(v) if isinstance(v,(list,tuple)) else (int(v),)
    if "hidden_layers" in p and "hidden_layer_sizes" not in p:
        v=p.pop("hidden_layers"); p["hidden_layer_sizes"]=tuple(v) if isinstance(v,(list,tuple)) else (int(v),)
    if "lr" in p and "learning_rate_init" not in p:
        p["learning_rate_init"]=p.pop("lr")
    if "learning_rate" in p:
        v=p["learning_rate"]
        if isinstance(v,(int,float)):
            p.setdefault("learning_rate_init",float(v)); p.pop("learning_rate",None)
        elif isinstance(v,str):
            vv=v.strip()
            try:
                num=float(vv)
            except ValueError:
                p["learning_rate"]=vv.lower()
            else:
                p.setdefault("learning_rate_init",num); p.pop("learning_rate",None)
    if "epochs" in p and "max_iter" not in p:
        p["max_iter"]=p.pop("epochs")
    if "weight_decay" in p and "alpha" not in p:
        p["alpha"]=p.pop("weight_decay")
    if isinstance(p.get("activation"),str):
        p["activation"]=p["activation"].strip().lower()
    if isinstance(p.get("solver"),str):
        p["solver"]=p["solver"].strip().lower()
    allowed={
        "hidden_layer_sizes","activation","solver","alpha","batch_size","learning_rate",
        "learning_rate_init","power_t","max_iter","shuffle","random_state","tol","verbose",
        "warm_start","momentum","nesterovs_momentum","early_stopping","validation_fraction",
        "beta_1","beta_2","epsilon","n_iter_no_change","max_fun"
    }
    q={k:v for k,v in p.items() if k in allowed}; q["random_state"]=seed
    return q


def metrics(y:np.ndarray, prob:np.ndarray) -> Dict[str,Any]:
    pred=(prob>=THRESHOLD).astype(np.int8)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    out={
        "accuracy":accuracy_score(y,pred),
        "macro_f1":f1_score(y,pred,average="macro",zero_division=0),
        "positive_f1":f1_score(y,pred,pos_label=1,zero_division=0),
        "precision":precision_score(y,pred,zero_division=0),
        "recall":recall_score(y,pred,zero_division=0),
        "balanced_accuracy":balanced_accuracy_score(y,pred),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
        "roc_auc":roc_auc_score(y,prob),
        "pr_auc":average_precision_score(y,prob),
        "log_loss":log_loss(y,np.c_[1-prob,prob],labels=[0,1]),
    }
    return out

METRICS=("accuracy","macro_f1","positive_f1","precision","recall",
         "balanced_accuracy","roc_auc","pr_auc","log_loss")

def ci95(vals: Sequence[float]) -> float:
    xs=[float(v) for v in vals if np.isfinite(v)]
    if len(xs)<2: return float("nan")
    t=2.7764451051977987 if len(xs)==5 else 1.96
    return t*statistics.stdev(xs)/math.sqrt(len(xs))


class FrozenLSTM(nn.Module):
    def __init__(self,n_act,n_doc,n_sub,emb_dim,hidden_dim,num_layers,dropout,bidirectional):
        super().__init__()
        self.a=nn.Embedding(n_act,emb_dim,padding_idx=0)
        self.d=nn.Embedding(n_doc,emb_dim,padding_idx=0)
        self.s=nn.Embedding(n_sub,emb_dim,padding_idx=0)
        self.rnn=nn.LSTM(
            emb_dim*3+1,hidden_dim,num_layers=num_layers,batch_first=True,
            dropout=dropout if num_layers>1 else 0.0,bidirectional=bidirectional
        )
        self.fc=nn.Linear(hidden_dim*(2 if bidirectional else 1),1)
    def forward(self,a,d,s,delta):
        x=torch.cat([self.a(a),self.d(d),self.s(s),delta.unsqueeze(-1)],dim=-1)
        o,_=self.rnn(x)
        return self.fc(o[:,-1,:]).squeeze(-1)


def stage57_best_epoch(stage57:Path,prefix:int,seed:int)->int:
    p=stage57/f"traininfo_LSTM_P{prefix}_seed{seed}.json"
    if not p.exists(): raise FileNotFoundError(p)
    e=int(read_json(p).get("best_epoch",-1))
    if e<1: raise RuntimeError(f"Invalid Stage57 best_epoch P{prefix} seed={seed}: {e}")
    return e


def train_lstm_final(train,test,mapping,raw,seed,epochs,ckpt):
    set_seed(seed)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb=int(pick(raw,"emb_dim","embedding_dim",default=16))
    hid=int(pick(raw,"hidden_dim","hidden_size",default=64))
    layers=int(pick(raw,"num_layers","layers",default=1))
    drop=float(pick(raw,"dropout",default=0.0))
    bi=bool(pick(raw,"bidirectional",default=False))
    bs=int(pick(raw,"batch_size",default=128))
    lr=float(pick(raw,"lr","learning_rate",default=1e-3))
    wd=float(pick(raw,"weight_decay",default=0.0))
    weighted=bool(pick(raw,"use_pos_weight","class_weight_balanced",default=False))
    model=FrozenLSTM(len(mapping["activity_vocab"]),len(mapping["doctype_vocab"]),
                     len(mapping["subprocess_vocab"]),emb,hid,layers,drop,bi).to(dev)
    y=train["y"].astype(np.float32)
    if weighted:
        pos=max(1.0,float(y.sum())); neg=max(1.0,float(len(y)-y.sum()))
        lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],device=dev))
    else:
        lossfn=nn.BCEWithLogitsLoss()
    optname=str(pick(raw,"optimizer",default="adam")).lower()
    opt=(torch.optim.AdamW if optname=="adamw" else torch.optim.Adam)(
        model.parameters(),lr=lr,weight_decay=wd
    )
    ds=TensorDataset(torch.from_numpy(train["activity_seq"]).long(),
                     torch.from_numpy(train["doctype_seq"]).long(),
                     torch.from_numpy(train["subprocess_seq"]).long(),
                     torch.from_numpy(train["delta_log1p_seq"]).float(),
                     torch.from_numpy(y).float())
    g=torch.Generator(); g.manual_seed(seed)
    dl=DataLoader(ds,batch_size=bs,shuffle=True,generator=g)
    hist=[]
    for ep in range(1,epochs+1):
        model.train(); losses=[]
        for a,d,s,delta,yy in dl:
            a,d,s,delta,yy=a.to(dev),d.to(dev),s.to(dev),delta.to(dev),yy.to(dev)
            opt.zero_grad(set_to_none=True)
            loss=lossfn(model(a,d,s,delta),yy); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        hist.append({"epoch":ep,"train_loss":float(np.mean(losses))})
    torch.save({"state_dict":model.state_dict(),"seed":seed,"epochs":epochs,
                "threshold":THRESHOLD},ckpt)
    model.eval(); probs=[]
    with torch.no_grad():
        for i in range(0,len(test["y"]),1024):
            a=torch.from_numpy(test["activity_seq"][i:i+1024]).long().to(dev)
            d=torch.from_numpy(test["doctype_seq"][i:i+1024]).long().to(dev)
            s=torch.from_numpy(test["subprocess_seq"][i:i+1024]).long().to(dev)
            de=torch.from_numpy(test["delta_log1p_seq"][i:i+1024]).float().to(dev)
            probs.append(torch.sigmoid(model(a,d,s,de)).cpu().numpy())
    return np.concatenate(probs).astype(float),{
        "device":str(dev),"fixed_epochs_from_stage57":epochs,"history":hist,
        "resolved":{"emb_dim":emb,"hidden_dim":hid,"num_layers":layers,"dropout":drop,
                    "bidirectional":bi,"batch_size":bs,"lr":lr,"weight_decay":wd,
                    "optimizer":optname,"use_pos_weight":weighted}
    }


def fit_one(model,prefix,seed,raw,train,test,mapping,stage57,ckptdir):
    set_seed(seed)
    if model=="RF":
        par=clean_rf(raw,seed); m=RandomForestClassifier(**par)
        m.fit(train["X_tabular"],train["y"])
        prob=m.predict_proba(test["X_tabular"])[:,1]
        ck=ckptdir/f"RF_P{prefix}_seed{seed}.joblib"; joblib.dump(m,ck)
        return prob,{"resolved_params":par},ck
    if model=="XGB":
        par=clean_xgb(raw,seed); m=xgb.XGBClassifier(**par)
        m.fit(train["X_tabular"],train["y"])
        prob=m.predict_proba(test["X_tabular"])[:,1]
        ck=ckptdir/f"XGB_P{prefix}_seed{seed}.json"; m.save_model(str(ck))
        return prob,{"resolved_params":par},ck
    if model=="MLP":
        par=clean_mlp(raw,seed); m=MLPClassifier(**par)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            m.fit(train["X_tabular"],train["y"])
        prob=m.predict_proba(test["X_tabular"])[:,1]
        ck=ckptdir/f"MLP_P{prefix}_seed{seed}.joblib"; joblib.dump(m,ck)
        return prob,{"resolved_params":par,"warnings":[str(w.message) for w in ws]},ck
    if model=="LSTM":
        ep=stage57_best_epoch(stage57,prefix,seed)
        ck=ckptdir/f"LSTM_P{prefix}_seed{seed}.pt"
        prob,info=train_lstm_final(train,test,mapping,raw,seed,ep,ck)
        return prob,info,ck
    raise ValueError(model)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project_root",type=Path,default=Path(r"D:\ETCE"))
    ap.add_argument("--stage56_dir",type=Path,default=None)
    ap.add_argument("--stage57_dir",type=Path,default=None)
    ap.add_argument("--out_dir",type=Path,default=None)
    ap.add_argument("--overwrite",action="store_true")
    a=ap.parse_args()
    root=a.project_root.resolve()
    s56=(a.stage56_dir.resolve() if a.stage56_dir else
         root/"results"/"phase4"/"phase4e_bpic2018_external"/"stage56_features")
    s57=(a.stage57_dir.resolve() if a.stage57_dir else
         root/"results"/"phase4"/"phase4e_bpic2018_external"/"stage57_development_models")
    out=(a.out_dir.resolve() if a.out_dir else
         root/"results"/"phase4"/"phase4e_bpic2018_external"/"stage58_formal_external_test")
    ensure_clean_dir(out,a.overwrite)
    ckpt=out/"checkpoints"; ckpt.mkdir()

    print("="*120)
    print("ETCE STAGE58 - BPIC2018 FINAL REFIT + FORMAL EXTERNAL TEST")
    print("="*120)
    print(f"Stage56 : {s56}\nStage57 : {s57}\nOutput  : {out}")
    print(f"Seeds   : {list(SEEDS)}\nThreshold: {THRESHOLD}")
    print("Final train: 2015+2016 | Formal TEST: 2017 | TEST tuning: FORBIDDEN")
    print("="*120)

    v56=verify_stage56(s56); print("[Stage56] PASS")
    v57=verify_stage57(s57); print("[Stage57] PASS")
    resolved=frozen_configs(v57["imported"])
    print("[Frozen config provenance] PASS")

    mapping=read_json(s56/"mapping_final_fit2015_2016.json")

    # Freeze ALL LSTM epochs before any 2017 NPZ is opened.
    epoch_map={f"P{p}_seed{s}":stage57_best_epoch(s57,p,s)
               for p in PREFIXES for s in SEEDS}
    write_json(out/"lstm_final_epochs_frozen_from_stage57.json",epoch_map)
    print("[LSTM epochs frozen before TEST load] PASS")

    rows=[]
    predfile=out/"final_test_predictions.csv.gz"
    with gzip.open(predfile,"wt",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["case_id","year","prefix","model","seed","y_true","prob_positive","pred_0_5"])

        for p in PREFIXES:
            tr=load_npz(s56/f"final_P{p}_train2015_2016.npz")
            if set(np.unique(tr["year"]).tolist())!={2015,2016}:
                raise RuntimeError(f"P{p} final-train year gate failed.")

            te=load_npz(s56/f"final_P{p}_test2017.npz")
            if set(np.unique(te["year"]).tolist())!={2017}:
                raise RuntimeError(f"P{p} formal TEST year gate failed.")

            print(f"\n[P{p}] train={len(tr['y']):,} (+{int(tr['y'].sum())}) | "
                  f"TEST={len(te['y']):,} (+{int(te['y'].sum())})")

            for model in MODELS:
                raw=resolved[model][str(p)]["params"]
                for seed in SEEDS:
                    print(f"[P{p}] {model} seed={seed} FINAL REFIT -> TEST ...",flush=True)
                    t0=time.time()
                    prob,info,cp=fit_one(model,p,seed,raw,tr,te,mapping,s57,ckpt)
                    if len(prob)!=len(te["y"]) or not np.all(np.isfinite(prob)):
                        raise RuntimeError(f"Invalid predictions {model} P{p} seed={seed}")
                    mm=metrics(te["y"],prob)
                    r={"model":model,"prefix":p,"seed":seed,"threshold":THRESHOLD,
                       "train_years":"2015+2016","test_year":2017,
                       "n_train":len(tr["y"]),"n_test":len(te["y"]),
                       "positive_train":int(tr["y"].sum()),"positive_test":int(te["y"].sum()),
                       **mm,"checkpoint":str(cp),"checkpoint_sha256":sha256_file(cp),
                       "runtime_sec":round(time.time()-t0,3)}
                    rows.append(r)
                    yh=(prob>=THRESHOLD).astype(np.int8)
                    for cid,yv,pv,h in zip(te["case_id"],te["y"],prob,yh):
                        w.writerow([str(cid),2017,p,model,seed,int(yv),f"{float(pv):.12g}",int(h)])
                    write_json(out/f"traininfo_{model}_P{p}_seed{seed}.json",info)
                    print(f"    Macro-F1={mm['macro_f1']:.6f} Pos-F1={mm['positive_f1']:.6f} "
                          f"PR-AUC={mm['pr_auc']:.6f} ROC-AUC={mm['roc_auc']:.6f} "
                          f"TP={mm['tp']} FP={mm['fp']} FN={mm['fn']} TN={mm['tn']}")

    fields=["model","prefix","seed","threshold","train_years","test_year","n_train","n_test",
            "positive_train","positive_test",*METRICS,"tn","fp","fn","tp",
            "checkpoint","checkpoint_sha256","runtime_sec"]
    with (out/"final_test_metrics_seedwise.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

    grp=defaultdict(list)
    for r in rows: grp[(r["model"],r["prefix"])].append(r)
    summary=[]
    for m in MODELS:
        for p in PREFIXES:
            rr=grp[(m,p)]
            if len(rr)!=5: raise RuntimeError(f"Missing seeds {m} P{p}")
            z={"model":m,"prefix":p,"n_seeds":5}
            for k in METRICS:
                vals=[float(x[k]) for x in rr]
                z[k+"_mean"]=float(np.mean(vals))
                z[k+"_std"]=float(np.std(vals,ddof=1))
                z[k+"_ci95_halfwidth"]=ci95(vals)
            summary.append(z)

    sf=["model","prefix","n_seeds"]
    for k in METRICS:
        sf += [k+"_mean",k+"_std",k+"_ci95_halfwidth"]
    with (out/"final_test_metrics_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=sf); w.writeheader(); w.writerows(summary)

    sy=v56["summary"]
    years=sy["prediction_year_counts"]; pos=sy["positive_by_year_prediction"]
    n2017=int(years.get("2017",years.get(2017)))
    p2017=int(pos.get("2017",pos.get(2017)))

    checks={
        "stage56_pass":{"pass":True},
        "stage57_pass":{"pass":True},
        "all_lstm_epochs_frozen_before_test":{"pass":len(epoch_map)==20 and all(v>=1 for v in epoch_map.values())},
        "all_80_runs_completed":{"pass":len(rows)==80,"actual":len(rows),"expected":80},
        "all_test_year_2017":{"pass":all(r["test_year"]==2017 for r in rows)},
        "all_test_n_match_stage56":{"pass":all(r["n_test"]==n2017 for r in rows),
                                    "actual_unique":sorted(set(r["n_test"] for r in rows)),"expected":n2017},
        "all_test_positive_match_stage56":{"pass":all(r["positive_test"]==p2017 for r in rows),
                                           "actual_unique":sorted(set(r["positive_test"] for r in rows)),"expected":p2017},
        "threshold_fixed":{"pass":THRESHOLD==0.5},
        "formal_seeds_exact":{"pass":list(SEEDS)==[13,21,42,73,101]},
        "test_based_tuning":{"pass":True,"actual":False,"expected":False},
    }
    allpass=all(v["pass"] for v in checks.values())
    write_json(out/"consistency_checks.json",checks)
    write_json(out/"protocol_frozen.json",{
        "stage":58,"dataset":"BPIC2018","created_local":datetime.now().astimezone().isoformat(),
        "models":list(MODELS),"prefixes":list(PREFIXES),"seeds":list(SEEDS),"threshold":THRESHOLD,
        "final_train_years":[2015,2016],"formal_external_test_year":2017,
        "frozen_config_sha256":v57["imported"]["source_sha256"],
        "lstm_epoch_source":"Stage57 per-prefix/per-seed best_epoch frozen before TEST load",
        "test_based_tuning":False,"all_hard_checks_pass":allpass
    })
    (out/"STAGE58_FORMAL_EXTERNAL_TEST.md").write_text(
        f"# Stage58 BPIC2018 Formal External TEST\n\n"
        f"- Final train: 2015+2016\n- Formal TEST: 2017\n"
        f"- TEST cases/prefix: {n2017}\n- Positive TEST cases/prefix: {p2017}\n"
        f"- Threshold: 0.5\n- Test-based tuning: NO\n- Gate: {'PASS' if allpass else 'FAIL'}\n",
        encoding="utf-8"
    )

    files=[p for p in out.rglob("*") if p.is_file() and p.name!="sha256_manifest.json"]
    write_json(out/"sha256_manifest.json",{
        "stage":58,"algorithm":"SHA256","generated_local":datetime.now().astimezone().isoformat(),
        "files":{str(p):{"sha256":sha256_file(p),"bytes":p.stat().st_size} for p in sorted(files)}
    })

    print("\n"+"="*120)
    print("ETCE STAGE58 FORMAL EXTERNAL TEST SUMMARY")
    print("="*120)
    for r in summary:
        print(f"{r['model']:4s} P{r['prefix']:2d} | "
              f"Macro-F1={r['macro_f1_mean']:.6f}±{r['macro_f1_std']:.6f} | "
              f"Pos-F1={r['positive_f1_mean']:.6f}±{r['positive_f1_std']:.6f} | "
              f"PR-AUC={r['pr_auc_mean']:.6f}±{r['pr_auc_std']:.6f} | "
              f"ROC-AUC={r['roc_auc_mean']:.6f}±{r['roc_auc_std']:.6f}")
    print(f"Hard gates        : {'PASS' if allpass else 'FAIL'}")
    print("TEST-based tuning : NO")
    print("="*120)
    if not allpass:
        raise SystemExit(2)
    print("\nSTAGE58 FORMAL EXTERNAL TEST COMPLETED SUCCESSFULLY.")
    print("No BPIC2018 TEST-based retuning is permitted after this point.")


if __name__=="__main__":
    main()
