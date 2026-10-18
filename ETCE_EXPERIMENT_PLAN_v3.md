# ETCE 新版实验设计（v3）

版本日期：2026-09-22  
用途：Codex 代码重建与后续实验的权威设计基准。  
依据：Expert Systems 三位审稿人意见；在 v2 基础上进一步明确“BPIC2018 暂不进入 Phase 1”和“Prefix 端点先统计后确定”。

## 1. 总体原则

1. 所有实验、代码与论文修改都必须能对应 Reviewer 1/2/3 的明确意见。
2. 旧实验代码已丢失，旧论文结果只作历史参考；不得反推、猜测或伪造缺失参数与样本数。
3. 先保证实验协议正确、无明显泄漏、可复现，再运行完整 ETCE 与 RQ1–RQ6。
4. 本文件是实验设计基准；若后续任务文件与本文件冲突，以本文件为准。

## 2. 数据集定位

### 2.1 BPIC2017

- 主数据集。
- Phase 1 只处理 BPIC2017。
- 当前工作版预测任务：`Offer accepted vs. not accepted`。
- 该标签必须先通过原始日志审计确认；若原始字段/活动不能无歧义支持该定义，必须停止并报告，禁止擅自换标签。
- 数据划分：按 case start time 做 case-level temporal split，Train/Validation/Test = 60%/20%/20%。

### 2.2 BPIC2018

- 计划作为跨领域外部验证数据集。
- **当前尚未下载，不属于 Phase 1 的任何实施范围。**
- 当前工作版预测任务为 `Late payment vs. not late`，具体字段、outcome 定义和年份切分必须在数据下载后重新审计确认。
- 在 BPIC2017 主流程稳定前，不允许因为 BPIC2018 缺失而阻塞 Phase 1。

## 3. Prefix 方案（v3 重要更新）

### 3.1 不提前固定最终 Prefix

最终正式实验不再预先写死为 Prefix-10/20/30。Phase 1 先对以下候选端点做覆盖统计：

`[5, 10, 15, 20, 25, 30]`

Phase 1 必须输出每个候选端点的：

- eligible case 数；
- 占全部可用 case 的比例；
- Train/Validation/Test 中的样本数；
- 正负类别样本数与比例；
- 因轨迹过短而被排除的 case 数；
- 不同端点下的 trace/case 分布变化。

生成 `results/phase1/stats/prefix_coverage.csv`。

### 3.2 Phase 1 不得自动决定最终端点

Codex 只负责统计候选端点，不得擅自选择正式实验的 3–4 个 Prefix。最终 Prefix 由人工根据覆盖率、类别平衡和预测阶段代表性确定。

### 3.3 Prefix 定义

对于完整轨迹 `σ=<e1,...,em>`，Prefix-k 定义为：

`σ^k=<e1,...,ek>`，且 `k < m`。

模型只能看到当前 prefix 内已经发生的事件与当时已知属性。任何 prefix 后事件、最终 outcome、未来才产生的属性都禁止作为输入。

## 4. Leakage Control

1. 同一 case 的所有 prefix 必须处于同一 Train/Validation/Test split。
2. case/application/offer/event ID 不作为预测特征。
3. BPIC2017 中与最终 outcome 直接相关的字段/活动（例如 Accepted、Selected 或其他审计确认的直接结果信息）不得提前进入 prefix 特征。
4. 标准化、缺失值填补、类别字典、离散化、特征筛选均只在 training set 拟合。
5. 未来用于 DFG、Petri net 或其他过程结构的任何模型，也只能由 training cases 学习。

## 5. 后续 Black-box 体系（Phase 1 不实现）

后续正式实验拟采用：

- Random Forest
- XGBoost
- MLP
- LSTM（替换旧 SVM，以回应 Reviewer 1 的 sequence-model 质疑）

四种模型必须共享相同 cases、temporal split、最终确定的 prefix endpoints、labels 和 leakage exclusion。

## 6. 后续 Surrogate 体系（Phase 1 不实现）

ETCE 内部固定：

- LORE branch
- Decision Tree (DT)
- Bayesian Network (BN)

统一转成 canonical rule representation：

`r = (C_r, y_r, n_r, Fid(r))`

## 7. 后续 Reliability / BPA（Phase 1 不实现）

`Fid(r) = (# black-box-consistent covered samples)/|S_r|`

`rho_r = |S_r|/(|S_r| + lambda)`

`R(r) = rho_r * Fid(r)`

`m_r(theta_r)=R(r)`

`m_r(Theta)=1-R(r)`

`m_r(not theta_r)=0`

`lambda` 仅由 validation set 选择，并做敏感性分析。

## 8. 后续 Evidence Dependence / Conflict（Phase 1 不实现）

- 计算 LORE/DT/BN 的 empirical output agreement A。
- 使用 validation-set quantiles 划分 Low/Medium/High agreement。
- 记录三阶段 D-S conflict coefficient `K^(1), K^(2), K^(3)`。
- 进行 correlated-source stress test，例如 `LORE + DT + DT'`。

## 9. 后续 Process Validity（Phase 1 不实现）

生成阶段：

- DFG control-flow constraint
- temporal consistency constraint
- immutable-attribute constraint

独立评价：

- training-log-only Petri-net conformance fitness

严禁把 process validity 写成 real-world actionability。

## 10. 后续评价指标（Phase 1 不实现）

核心指标包括：

- Generation Success Rate (GSR)
- Counterfactual Validity (CFV)
- Rule Fidelity
- Rule Support
- Attribute Sparsity
- Sequence Edit Distance
- Attribute Proximity
- Flow Validity
- Temporal Validity
- Petri-net Conformance Fitness
- Stability
- Candidate Diversity
- D-S Conflict K
- Surrogate Agreement A

必须保存 `N_targets / N_generated / N_valid`，避免只报告无法解释的比例。

## 11. 后续 Baseline / Fusion Comparison（Phase 1 不实现）

外部方法候选：

- DiCE
- DiCE4EL
- LORELEY
- 1 个可复现的 process-aware / temporal-constraint CF baseline
- ETCE

内部 surrogate comparison：

- LORE
- DT
- BN
- ETCE

Fusion Strategy Comparison（相同 surrogate 输出，仅替换聚合机制）：

- Best Single Surrogate
- Weighted Voting
- Probability Averaging
- Fuzzy Fusion
- Bayesian Fusion
- D-S Fusion (ETCE)

## 12. RQ1–RQ6

- RQ1 Overall Effectiveness
- RQ2 Multi-source Fusion
- RQ3 Reliability Modeling
- RQ4 Evidence Dependence and Conflict
- RQ5 Process Validity
- RQ6 Component Contribution and Robustness

所有正式结果采用多随机种子并报告 mean±std；ETCE 与 strongest baseline 进一步报告 95% CI、显著性检验和 effect size。

## 13. Worked Example

最终论文新增一个真实 BPIC2017 test-prefix worked example，完整展示：

`Prefix -> Neighborhood -> LORE/DT/BN rules -> Fidelity/Support -> Reliability/BPA -> Stage 1/2/3 (with K) -> Final CF -> validity/process checks`

## 14. 当前实施状态

- [x] 实验设计完成至 v3
- [ ] Phase 1：BPIC2017 数据审计与预处理
- [ ] 最终 Prefix 端点确认
- [ ] Phase 2：BPIC2017 black-box predictors
- [ ] Phase 3：ETCE 主流程
- [ ] BPIC2018 下载与数据审计
- [ ] RQ1–RQ6 正式实验

