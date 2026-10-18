# ETCE Phase 3A.1 运行说明

这是已经通过验证的 Phase 3A 的配对语义约束实验。

## 唯一新增规则

```text
A_* -> A_*
O_* -> O_*
W_* -> W_*
```

除此之外全部保持 Phase 3A 不变：

- 同样的 20 个 pilot cases
- 同样的 canonical black-box checkpoints
- 同样的 Prefix 5/10/15/20
- 同样的 train-only DFG
- 同样的 position support
- 同样的 timestamps
- 同样的 population / generations / mutation rate / crossover rate
- 同样的 deterministic search seed construction

因此可以直接比较 Phase 3A 与 Phase 3A.1 的 GSR 变化。

## 放置位置

把压缩包内容按目录放到 `D:\ETCE\`：

```text
D:\ETCE\
├─ src\
│  └─ etce\
│     └─ phase3\
│        └─ semantic_core.py
└─ scripts\
   ├─ 14_phase3a1_semantic_constraint.py
   └─ 15_validate_phase3a1.py
```

原来的 `core.py`、10/11/12/13 均不要改。

## 运行

PowerShell：

```powershell
cd D:\ETCE
python .\scripts\14_phase3a1_semantic_constraint.py
python .\scripts\15_validate_phase3a1.py
```

不需要重新运行 10、11、12、13。

## 输出

```text
results\phase3\phase3a1\
├─ PHASE3A1_REPORT.md
├─ counterfactual_semantic_targets.csv
├─ counterfactual_semantic_summary.csv
├─ phase3a1_overall_comparison.json
├─ phase3a1_run_config.json
├─ pilot_case_selection.json
└─ validation_phase3a1.json
```

运行完成后优先发：

- `PHASE3A1_REPORT.md`
- `counterfactual_semantic_summary.csv`
- `validation_phase3a1.json`
- `phase3a1_overall_comparison.json`

如果 GSR 下降明显，也不要自行放宽语义约束。先判断下降来自哪些 model/prefix 和哪些 event family。
