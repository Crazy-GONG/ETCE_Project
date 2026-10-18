# ETCE Phase 3B 正式反事实生成实验

Phase 3A.1 已通过验证后，Phase 3B 将其冻结为正式反事实生成器。

## 为什么现在进入 Phase 3B

Phase 3A.1 的 320 个 pilot 目标中：

- Phase 3A GSR = 0.8031
- 加入 event-family preservation 后 GSR = 0.6156
- 197/197 个成功反事实保持：
  - DFG valid
  - temporal valid
  - immutable history
  - event-family preservation

GSR 下降是更严格语义约束的代价，不应为了追求成功率再放松约束。

## 防止 pilot 泄漏

Phase 3A / Phase 3A.1 使用过的 20 个 test case：

- 只作为开发/软件 pilot；
- Phase 3B **永久排除**；
- 正式 Phase 3B 从剩余 test cases 中，按 `SHA256(case_id)` 固定选择 100 个此前未使用样本；
- 选择不读取标签、预测或 test 指标。

## 正式设计

```text
100 formal cases
× 4 black-box models
× 4 prefixes
× 5 formal search seeds
= 8000 searches
```

正式 search seeds：

```text
13, 21, 42, 73, 101
```

Phase 3B 不再调：

- population size
- generations
- mutation rate
- crossover rate
- candidate option count
- process constraints
- semantic constraints

## 运行

将两个脚本放到：

```text
D:\ETCE\scripts\
```

然后：

```powershell
cd D:\ETCE
python .\scripts\16_phase3b_formal_counterfactual.py
```

脚本支持断点续跑。运行中断后重新执行，会跳过已经完成的 `(case, model, prefix, search_seed)`。

完成后：

```powershell
python .\scripts\17_validate_phase3b.py
```

## 主要输出

```text
results\phase3\phase3b\
├─ PHASE3B_REPORT.md
├─ formal_case_selection.json
├─ phase3b_run_config.json
├─ formal_counterfactual_runs.csv
├─ formal_metrics_per_seed.csv
├─ formal_metrics_summary.csv
├─ cross_seed_stability.csv
├─ phase3b_overall.json
└─ validation_phase3b.json
```

## 正式指标

- Generation Success Rate (GSR)
- mean edit count
- normalized sequence distance
- target-side probability margin
- train-only DFG validity
- temporal validity
- event-family validity
- immutable-history validity
- two explicit failure categories:
  - `no_admissible_mutation`
  - `admissible_candidates_no_flip`
- cross-seed modified-position Jaccard
- exact counterfactual repetition
- mean ± std
- 95% Student-t CI

运行结束后优先发：

- `PHASE3B_REPORT.md`
- `formal_metrics_summary.csv`
- `cross_seed_stability.csv`
- `validation_phase3b.json`
- `phase3b_overall.json`
