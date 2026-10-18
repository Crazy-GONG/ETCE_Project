# ETCE Phase 3A 本地运行说明

本阶段开始进入论文的核心反事实部分，但仍然只是 **Phase 3A 方法/软件 pilot**，不是最终论文统计。

## 目标

Phase 3A 只验证：

1. 可以从 Phase 2B 中无 test 泄漏地选择 canonical black-box；
2. 可以只用训练集、且只用 label boundary 之前的历史构造 DFG 流程约束；
3. 可以在固定 Prefix-5/10/15/20 上生成让 black-box 预测翻转的活动序列；
4. 所有成功反事实都满足：
   - train-only DFG；
   - 时间顺序合法；
   - 不引入 `A_Pending/A_Denied/A_Cancelled`；
   - 不通过新生成 `O_Accepted/O_Refused/O_Cancelled` 直接“改结果”；
   - 已存在的历史 terminal offer event 不被修改。

## Phase 3A 的有意限制

为了先把 reviewer 最关心的 **process validity + leakage + counterfactual validity** 做干净，本 pilot：

- 只允许 activity substitution；
- 固定长度；
- timestamps 完全保持 factual 不变；
- 不声称 causal recourse；
- 不声称 actionability；
- 暂不做 Petri-net conformance；
- 暂不做 LORE / DT / BN；
- 暂不做 BPA / D-S fusion；
- 不作为论文最终统计。

这一步通过以后再进入 Phase 3B / Phase 4。

## 放置位置

把压缩包内容按原目录覆盖到项目中：

```text
D:\ETCE\
├─ src\
│  └─ etce\
│     └─ phase3\
│        ├─ __init__.py
│        └─ core.py
└─ scripts\
   ├─ 10_select_canonical_blackboxes.py
   ├─ 11_build_process_constraints.py
   ├─ 12_phase3a_counterfactual_pilot.py
   └─ 13_validate_phase3a.py
```

不要删除：

```text
results\phase1_2\
results\phase2\
results\phase2b\
```

## 运行顺序

PowerShell：

```powershell
cd D:\ETCE
python .\scripts\10_select_canonical_blackboxes.py
python .\scripts\11_build_process_constraints.py
python .\scripts\12_phase3a_counterfactual_pilot.py
python .\scripts\13_validate_phase3a.py
```

必须按 10 → 11 → 12 → 13 的顺序。

## 主要输出

```text
results\phase3\
├─ canonical_blackboxes\
│  ├─ canonical_manifest.json
│  └─ checkpoints\
├─ constraints\
│  ├─ process_constraints.json
│  ├─ dfg_edges.csv
│  └─ activity_position_counts.csv
└─ phase3a\
   ├─ PHASE3A_REPORT.md
   ├─ pilot_case_selection.json
   ├─ phase3a_run_config.json
   ├─ counterfactual_pilot_targets.csv
   ├─ counterfactual_pilot_summary.csv
   └─ validation_phase3a.json
```

## 运行结束后发给 ChatGPT

优先发：

- `PHASE3A_REPORT.md`
- `counterfactual_pilot_summary.csv`
- `validation_phase3a.json`
- `canonical_manifest.json`
- `process_constraints.json`

如果 Phase 3A GSR 太低，不要自行扩大 mutation 范围或加入结果事件；先分析失败原因。
