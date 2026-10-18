# ETCE Phase 4A：Surrogate Rule Extraction Pilot

Phase 3 已冻结。Phase 4A 开始进入 ETCE 的多源规则证据部分。

## 这一步做什么

对每个局部 target：

1. 使用 Phase 3A.1 的流程/语义约束生成黑盒标签无关的局部邻域；
2. 用冻结 canonical black-box 给邻域打标签；
3. 将邻域按 sequence hash 固定划分为 70% fit / 30% eval；
4. 训练三种内部 surrogate evidence source：
   - LORE branch
   - DT branch
   - BN branch
5. 提取 factual rule 和 counterfactual rule；
6. 在 **held-out local eval subset** 上计算：
   - `n_r`：rule support
   - `Fid(r)`：rule fidelity

这一步暂时不做：

- `rho_r`
- `R(r)`
- BPA
- D-S fusion

这些留到下一阶段。

## 重要术语

这里的 **LORE branch** 是 ETCE 内部的 process-aware LORE workflow adaptation：

```text
process-valid local neighborhood
→ class-balanced local fit subset
→ local decision tree
→ factual/counterfactual path rules
```

不是后续实验中的 external official LORE / LORE_sa baseline。

LORE 本身的核心思想就是通过遗传/合成局部邻域训练局部决策树，再从树中提取 factual 与 counterfactual rules；我们的内部 branch 在此基础上增加了 Phase 3 已冻结的流程约束。

## BN 的具体定义

为了避免“Bayesian Network”写得含糊，这里固定为：

```text
Naive-Bayes-structured BN:
Y → X1
Y → X2
...
Y → Xk
```

其中 `Xi` 是第 i 个事件位置的 activity category。

使用 Laplace smoothing `alpha=1`。

## Pilot 范围

只使用 **validation split 的 8 个 case**：

```text
8 cases × 4 black-boxes × 4 prefixes = 128 local targets
```

不使用 test case，因此仍然属于方法/软件开发阶段。

局部 pool 上限：

```text
320 sequences
max edits = 3
```

## 放置

解压到：

```text
D:\ETCE\
```

形成：

```text
D:\ETCE\
├─ src\
│  └─ etce\
│     └─ phase4\
│        ├─ __init__.py
│        └─ surrogate_core.py
└─ scripts\
   ├─ 18_phase4a_surrogate_pilot.py
   └─ 19_validate_phase4a.py
```

## 运行

PowerShell：

```powershell
cd D:\ETCE
python .\scripts\18_phase4a_surrogate_pilot.py
```

完成后：

```powershell
python .\scripts\19_validate_phase4a.py
```

## 主要输出

```text
results\phase4\phase4a\
├─ PHASE4A_REPORT.md
├─ pilot_case_selection.json
├─ phase4a_run_config.json
├─ local_neighborhoods.csv.gz
├─ surrogate_metrics.csv
├─ surrogate_aggregate.csv
├─ canonical_rules.jsonl
└─ validation_phase4a.json
```

运行结束后优先发：

- `PHASE4A_REPORT.md`
- `surrogate_aggregate.csv`
- `surrogate_metrics.csv`
- `validation_phase4a.json`

## 下一步判定标准

Phase 4A 不是要求所有 surrogate 都“越高越好”，而是先看：

- 三个 branch 是否都能稳定产生规则；
- held-out local surrogate fidelity 是否合理；
- counterfactual rule availability 是否足够；
- rule support 是否太低；
- 三种 surrogate 是否确实产生不同规则，而不是三个分支完全重复。

如果规则 availability / support 明显不够，再只在 validation split 内调整 Phase 4 参数；正式 test 实验前必须冻结。
