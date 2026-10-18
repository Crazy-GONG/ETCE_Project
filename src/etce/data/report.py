"""Generate reviewable reports from audit evidence, without choosing a label."""
from collections import Counter
import csv
from pathlib import Path
import platform
import statistics
from .labels import evaluate_label_gate
from .prefix import CANDIDATES
from etce.utils.io import read_jsonl, sha256, write_csv, write_json


def finish_audit(root, summary):
    audit = root / 'results/phase1/audit'
    stats = root / 'results/phase1/stats'
    cases = list(read_jsonl(root / 'data/interim/phase1/bpic2017_case_audit.jsonl'))
    with (audit / 'offer_outcome_audit.csv').open(encoding='utf-8', newline='') as stream:
        offers = list(csv.DictReader(stream))
    offer_cross = Counter((o['Accepted'], int(o['accepted_state_count']) > 0) for o in offers)
    case_cross = Counter((bool(c['accepted_flag_count']), bool(c['accepted_state_count'])) for c in cases)
    cross_rows = [{'unit': unit, 'any_Accepted_true': flag, 'any_O_Accepted': state, 'count': count}
                  for unit, cross in [('offer', offer_cross), ('case', case_cross)]
                  for (flag, state), count in sorted(cross.items())]
    write_csv(audit / 'outcome_crosscheck.csv', cross_rows)
    write_csv(audit / 'outcome_disagreement_examples.csv', [o for o in offers if o['flag_state_agree'] == 'False'][:20])
    gate = evaluate_label_gate(summary)
    write_json(audit / 'label_gate.json', gate)

    # Raw length coverage is an audit statistic, not a labeled prefix dataset.
    coverage = []
    for k in CANDIDATES:
        eligible = [c for c in cases if c['events'] > k]
        row = {'endpoint': k, 'total_eligible_cases': len(eligible),
               'eligible_rate': len(eligible)/len(cases), 'excluded_short_cases': len(cases)-len(eligible)}
        for split in ('train', 'val', 'test'):
            for suffix in ('cases', 'positive', 'negative', 'positive_rate'):
                row[f'{split}_{suffix}'] = None
        lengths = [c['events'] for c in eligible]
        row.update({'eligible_trace_length_min': min(lengths), 'eligible_trace_length_median': statistics.median(lengths),
                    'eligible_trace_length_mean': statistics.mean(lengths), 'eligible_trace_length_max': max(lengths),
                    'status': 'length_only_label_gate_blocked'})
        coverage.append(row)
    write_csv(stats / 'prefix_coverage.csv', coverage)
    (stats / 'BLOCKED.md').write_text(
        '# 标签门槛未通过\n\n未执行真实 temporal split，也未构造任何带标签 prefix dataset。'
        '因此不生成 split_summary.csv；prefix_coverage.csv 只有原始长度覆盖统计，split/类别列为空值，不是 0。'
        '不得将其视为完整 Phase 1 统计。详见 results/phase1/PHASE1_REPORT.md。\n', encoding='utf-8')
    notes = ['# Prefix 候选覆盖：仅原始长度审计', '',
             '严格按 m > k 统计所有原始 case，未过滤 lifecycle、未删除事件、未生成 predictor 数据。',
             '标签核验受阻，因此未划分真实 split；类别分布、类别漂移及各 split 的少数类风险均不可判定。',
             'CSV 空白代表未计算，不代表零。以下数据不能单独用于确认最终端点。', '']
    previous = len(cases)
    for row in coverage:
        notes.append(f"- k={row['endpoint']}: {row['total_eligible_cases']:,} cases ({row['eligible_rate']:.2%}); "
                     f"比前一候选/全部 case 少 {previous-row['total_eligible_cases']:,}; "
                     f"eligible trace median={row['eligible_trace_length_median']}.")
        previous = row['total_eligible_cases']
    notes += ['', '只呈现样本减少和组成变化，不设置未经授权的骤减阈值，不选择任何最终端点。']
    (stats / 'prefix_coverage_notes.md').write_text('\n'.join(notes)+'\n', encoding='utf-8')

    excluded_fields = {
        'case:concept:name': 'application/case identifier; metadata only',
        'event:EventID': 'event/offer/application identifier; metadata only',
        'event:OfferID': 'offer linkage identifier; metadata only',
        'event:Accepted': 'candidate outcome field; occurs at offer creation and disagrees with later acceptance state',
        'event:Selected': 'offer selection outcome marker at creation; event-time availability unverified',
        'event:org:resource': 'resource identifier; no approved predictor role',
    }
    activities = {
        'O_Accepted': 'direct offer acceptance state; candidate outcome source',
        'O_Refused': 'direct offer terminal result', 'O_Cancelled': 'direct offer terminal result',
        'A_Pending': 'application status associated with accepted offers; potential outcome proxy',
        'A_Denied': 'application terminal result; potential outcome proxy',
        'A_Cancelled': 'application terminal result; potential outcome proxy',
        'A_Accepted': 'name is ambiguous; observed in every case, not a discriminating binary label',
    }
    write_json(audit / 'leakage_exclusions.json', {
        'status': 'blocked_pending_label_and_availability_review', 'raw_sha256': summary['raw_sha256'],
        'excluded_fields': excluded_fields, 'excluded_activities': activities,
        'default': 'deny unreviewed attributes', 'approved_event_fields': [], 'approved_static_case_fields': [],
        'activity_handling': 'no real predictor materialization until outcome and exclusion policy approved',
        'no_future_backfill': True, 'schema_fit_partition': 'train only'})
    initial = '# 初始 leakage 审计\n\n' + '\n'.join(f'- `{k}`: {v}' for k, v in excluded_fields.items())
    initial += '\n\n活动风险：\n' + '\n'.join(f'- `{k}`: {v}' for k, v in activities.items())
    initial += ('\n\n`Accepted` 与 `Selected` 各出现 42,995 次，全部位于 `O_Create Offer`。'
                '记录所在时间不等于现实首次可知时间，不能认为它们在 offer 创建时就是合法预测输入。\n')
    (audit / 'leakage_audit_initial.md').write_text(initial, encoding='utf-8')
    final = initial.replace('初始 leakage 审计', '最终 leakage 审计（本次审计受阻，不表示可用于建模）')
    final += ('\n`LoanGoal`、`ApplicationType`、`RequestedAmount` 仅出现在 trace 属性中；原始日志未提供更新历史，'
              '本阶段不把它们自动认证为 prefix 时刻已知。offer 数值属性也需可用性核验。'
              '\n\n接口默认拒绝未审核字段，ID 仅作 metadata；不从完整 case 回填未来属性。'
              '候选 prefix 保留原始事件索引；若包含禁用活动，接口拒绝，绝不删事件再补足 k。'
              '活动排除后的正式样本策略仍需人工审核；此处没有执行实际样本排除。'
              '\n\nFeatureSchema.fit 强制只接收 train prefix，保存类别字典与 fit case IDs；'
              '未知类别映射至 0，缺失保留 null。不实现 scaler/imputer，也没有拟合任何真实 encoder。'
              '\n\n状态：标签门槛未通过。无 predictor dataset 发布，无法声称完成真实数据无泄漏验证。\n')
    (audit / 'leakage_audit_final.md').write_text(final, encoding='utf-8')

    recorded_before = sum(bool(o['first_accepted_index']) and int(o['creation_index']) < int(o['first_accepted_index']) for o in offers)
    # Do not manufacture label_first_known_* for an unresolved label.
    write_json(audit / 'outcome_timing_summary.json', {
        'offers_with_O_Accepted': sum(int(o['accepted_state_count']) > 0 for o in offers),
        'offers_with_flags_recorded_before_O_Accepted': recorded_before,
        'event_index_base': 1,
        'actual_label_first_known_time': None,
        'reason': 'Candidate field recording times and state occurrence times are observed; real-world availability and target semantics unresolved',
        'per_offer_trace': 'offer_outcome_audit.csv', 'per_case_trace': 'data/interim/phase1/bpic2017_case_audit.jsonl'})
    files = [p for base in ('src', 'scripts', 'configs', 'tests') for p in (root/base).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    files += [root/'ETCE_EXPERIMENT_PLAN_v3.md', root/'CODEX_PHASE1_TASK_v2.md', root/'pyproject.toml']
    write_json(audit / 'run_manifest.json', {'python': platform.python_version(), 'raw_sha256': summary['raw_sha256'],
               'seed': 0, 'seed_role': 'software determinism only',
               'source_hashes': {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files)},
               'event_order': 'original XES; all lifecycle events', 'label_gate': gate['status']})
    coverage_table = '\n'.join(f"| {r['endpoint']} | {r['total_eligible_cases']:,} | {r['eligible_rate']:.2%} | {r['excluded_short_cases']:,} |"
                               for r in coverage)
    report = f'''# Phase 1 报告

## 结论：标签核验受阻，按任务第一门槛停止

已完整阅读 `ETCE_EXPERIMENT_PLAN_v3.md` 与 `CODEX_PHASE1_TASK_v2.md`。
本次完成工程框架、真实日志全量审计、结果字段交叉核验、leakage 风险清单与原始长度覆盖统计。
**未完成可交付的带标签数据集。** 严格执行任务 §3：无法无歧义确定标签时停止后续标签与 prefix dataset 构建。

## 1. 原始文件与审计范围

- 路径：`{summary['raw_path']}`；格式：`{summary['format']}`；文件大小：{summary['raw_bytes']:,} bytes。
- SHA-256：`{summary['raw_sha256']}`。
- {summary['cases']:,} cases / {summary['events']:,} events / {summary['offers']:,} offers / {summary['activities']} activities。
- 事件时间范围：{summary['event_time_min']} 至 {summary['event_time_max']}（UTC）。
- 保留原始事件顺序与所有 lifecycle 事件，无重采样、无删事件。重复 case/offer 创建 ID、空轨迹、时间倒流检测共 {summary['anomaly_count']} 条异常。
- {summary['cases_with_multiple_offers']:,} cases 包含多个 offer；{summary['cases_without_offer']} cases 无 offer；{summary['cases_with_mixed_accepted_flags']:,} cases 同时含 Accepted=True/False 的 offer。

## 2. 标签核验

当前 `Offer accepted vs. not accepted` **尚未通过无歧义核验**，没有构造最终标签。
真实字段 `Accepted`、`Selected` 都在 `O_Create Offer` 上；使用该事件的 `EventID` 与后续状态事件的 `OfferID` 逐 offer 关联。
全部 offer 都有 creation 记录和 boolean Accepted；发现 {summary['offer_flag_state_disagreements']:,} 个 offer 的字段与状态不一致。

| Accepted 字段 | 出现 O_Accepted | 未出现 O_Accepted |
|---|---:|---:|
| True | {offer_cross[('True', True)]:,} | {offer_cross[('True', False)]:,} |
| False | {offer_cross[('False', True)]:,} | {offer_cross[('False', False)]:,} |

在 application/case 层面，“存在 Accepted=True offer”产生 {sum(c['accepted_flag_count'] > 0 for c in cases):,} 个候选正例；
“存在 O_Accepted 活动”产生 {sum(c['accepted_state_count'] > 0 for c in cases):,} 个候选正例；两者有 {summary['case_flag_state_disagreements']:,} 个 case 不一致。
这些是**审计候选定义计数，不是最终标签分布**。`A_Accepted` 在全部 31,509 个 case 出现，不能仅凭名称用作该二分类标签。

不能从日志本身断言哪个字段是错误数据，也不能未经确认把“offer 被接受”改成“application 最终进入 pending”。
此外，多 offer case 的 any/selected/final-offer 聚合方式尚未在规范中明确。

## 3. Outcome 时间与泄漏

{recorded_before:,} 个具有 O_Accepted 状态的 offer，在更早的创建事件上已经记录 Accepted/Selected。
创建字段的记录时间与真实可知时间必须区分。审计记录了每个 offer 的创建 index/time 和首次 O_Accepted index/time（1-based）。
最终 `label_first_known_event_index/timestamp` 未赋值；负类何时可以确认还取决于最终标签定义与日志终止/观察窗口语义。
未生成冒充已核验标签的 case-label 文件。

直接/潜在泄漏包括 Accepted、Selected、case concept:name、EventID、OfferID、resource ID，以及 O_Accepted/O_Refused/O_Cancelled、A_Pending/A_Denied/A_Cancelled 等结果活动。
trace 级 LoanGoal/ApplicationType/RequestedAmount 的历史可用性未由日志独立证明，当前不自动放行。
字段、活动及原因见 `results/phase1/audit/leakage_exclusions.json` 和 `leakage_audit_final.md`。

## 4. Temporal split

**未执行真实 split，实际分组数与类别分布不适用。** 任务 §4 明确要求标签核验通过后执行。
已实现并测试按 `(case_start_time, case_id)` 稳定排序和 floor(0.6N)/floor(0.8N) 边界的通用接口，以及跨 split 不相交断言。
未生成 `split_summary.csv`，避免被误认为真实数据层已通过。

## 5. 候选 Prefix 覆盖

以下仅为所有原始 case 的严格 `m > k` 长度资格审计；未构造 prefix dataset，没有决定最终端点。
`prefix_coverage.csv` 的 split/类别列为空值，表示标签门槛阻断；不是 0。

| k | 原始 eligible cases | 占全部 cases | 过短排除 |
|---:|---:|---:|---:|
{coverage_table}

类别分布/漂移、各 split 少数类风险尚不可评估。候选对应的完整 trace 长度分布和减少数量见 CSV 与 `prefix_coverage_notes.md`。

## 6. 工程与可复现接口

提供流式 `.xes/.xes.gz` 读取、split/prefix 接口、默认拒绝的 leakage policy、training-only 可保存/重载 schema、sequence/tabular view。
这些接口仅经合成测试验证；未在真实数据上拟合 schema 或发布 processed 数据。metadata 与 predictor 分离，未知类别为 0，缺失为 null；不回填未来属性。
`configs/bpic2017.yaml` 使用 JSON 形式的合法 YAML，标准库即可读取。seed=0 仅用于软件复现测试，不是正式实验 seed 决定。
审计数据、配置、规范和源码哈希见 `run_manifest.json`。旧版非 Phase 1 实现已移至 `archive/pre_phase1/`，不参与导入与测试。

## 7. 人工审核事项与可选修订

1. 结合数据说明核定 Accepted 字段与 O_Accepted 状态各自业务含义，解释冲突；本次没有自动选择。
2. 明确 case-level 标签究竟使用任一 offer、selected offer 还是其他有业务依据的聚合；明确负类可确认时点。
3. 若确认目标为“日志中存在 O_Accepted”，在权威规范中明确该定义及其与 Accepted 字段差异；这只是待审核选项，未采用。
4. 若选择 Accepted 字段，需要同样明确聚合和字段可用时间；同样未采用。
5. 审核真实 prefix 的 outcome 活动处理、静态/动态属性可用性。标签门槛通过后才能产生正式 split/类别覆盖和 predictor 输出。
6. 最终 3–4 个 Prefix 仍由人工决定；本次一个也未选择。

## 8. 测试与停止点

<!-- VALIDATION_START -->
本阶段在标签歧义门槛处停止；后续修订和验证分别保存在 `results/phase1_1/` 与 `results/phase1_2/`。
<!-- VALIDATION_END -->
BPIC2018 仅预留 `data/raw/bpic2018/` 和未实现入口，没有下载或替代数据。
**没有训练任何预测模型，没有实现或运行 surrogate、GA、D-S、process-validity filter、baseline、RQ1–RQ6，也没有运行 ETCE 正式实验。**
完成报告后停在 Phase 1 标签审核门槛，等待人工审核；不进入 Phase 2。
'''
    (root / 'results/phase1/PHASE1_REPORT.md').write_text(report, encoding='utf-8')
    return gate
