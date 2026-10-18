# ETCE Phase 4A.1：Stratified Local Evidence Pilot

Phase 4A 已经通过软件验证，但结果暴露出一个**可修复的局部划分问题**：

- 128 个 local targets 中，有 90 个 pool 本身包含两个 black-box 类；
- 但原来的 label-independent hash 70/30 split 只有 81 个 fit subset 保留了两个类；
- 因此有 9 个 target 明明存在 counterfactual local evidence，却因为随机式哈希划分在 fit 中丢失了少数类。

这不是流程不可达，而是 local fit/eval protocol 的人为问题。

## Phase 4A.1 只改一件事

保持：

- 完全相同的 8 个 validation cases；
- 完全相同的 local neighborhoods；
- 完全相同的 black-box probabilities/labels；
- 完全相同的 LORE / DT / BN；
- 完全相同的 surrogate hyperparameters。

唯一改成：

> **按 black-box local label 做确定性的 stratified 70/30 local fit/eval split。**

真实 outcome label 不参与划分。

规则仍然只在 held-out eval subset 上计算 support 与 fidelity。

## 为什么不直接扩大 neighborhood

Phase 4A 还发现：

- Prefix-5 的 process-valid local pool 平均只有 5 个唯一 sequence；
- 很多早期 prefix 本身缺少足够的合法局部变化。

这属于严格 process/semantic constraints 下真实的 evidence sparsity，不应为了提高 rule availability 随意放松 DFG/event-family/immutability。

因此 Phase 4A.1 **不扩大 pool、不放松流程约束**，只修正可避免的 split artifact。

## 放置

解压到：

```text
D:\ETCE\
```

新增：

```text
src\etce\phase4\stratified_core.py
scripts\20_phase4a1_stratified_local_split.py
scripts\21_validate_phase4a1.py
```

原 Phase 4A 文件全部保留。

## 运行

```powershell
cd D:\ETCE
python .\scripts\20_phase4a1_stratified_local_split.py
python .\scripts\21_validate_phase4a1.py
```

## 输出

```text
results\phase4\phase4a1\
├─ PHASE4A1_REPORT.md
├─ phase4a1_run_config.json
├─ surrogate_metrics_stratified.csv
├─ surrogate_aggregate_stratified.csv
├─ canonical_rules_stratified.jsonl
└─ validation_phase4a1.json
```

跑完后把整个 `phase4a1` 目录压缩发回来即可。
