"""Phase 1.1 label, split, prefix-coverage, and leakage audit pipeline."""
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import platform
import statistics

from .labels import derive_case_label, TERMINAL_OFFER_ACTIVITIES
from .load_bpic2017 import read_cases, timestamp
from .prefix import CANDIDATES
from .split import temporal_split, validate_no_overlap
from etce.utils.io import sha256, write_csv, write_json

SENSITIVE_APPLICATION_ACTIVITIES = frozenset({'A_Pending', 'A_Denied', 'A_Cancelled'})


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _median(values):
    return statistics.median(values) if values else None


def build_phase1_1(root, config):
    root = Path(root)
    raw = root / config['raw_path']
    expected = config.get('raw_sha256')
    actual_hash = sha256(raw)
    if expected and expected != actual_hash:
        raise ValueError('Raw log hash differs from configured audit input')

    label_rows, exclusions, case_prefix_audits = [], [], {}
    accepted_field_events = selected_field_events = 0
    total_cases = 0
    for case in read_cases(raw):
        total_cases += 1
        label, exclusion = derive_case_label(case)
        accepted_field_events += sum('Accepted' in event for event in case.events)
        selected_field_events += sum('Selected' in event for event in case.events)
        if exclusion:
            exclusion.update({'trace_length': len(case.events),
                              'case_start_time': min(timestamp(e['time:timestamp']) for e in case.events).isoformat()
                              if case.events else None})
            exclusions.append(exclusion)
            continue
        row = asdict(label)
        label_rows.append(row)
        per_endpoint = {}
        for k in CANDIDATES:
            if k >= label.label_event_index:
                continue
            prefix_events = case.events[:k]
            terminal = Counter(e['concept:name'] for e in prefix_events
                               if e.get('EventOrigin') == 'Offer'
                               and e.get('concept:name') in TERMINAL_OFFER_ACTIVITIES)
            sensitive = Counter(e['concept:name'] for e in prefix_events
                                if e.get('concept:name') in SENSITIVE_APPLICATION_ACTIVITIES)
            per_endpoint[k] = {'terminal': dict(terminal), 'sensitive': dict(sensitive)}
        case_prefix_audits[case.case_id] = per_endpoint
        if total_cases % 5000 == 0:
            print(f'Labeled {total_cases} raw cases', flush=True)

    assignments = temporal_split(label_rows)
    for row in label_rows:
        row['split'] = assignments[row['case_id']]
    label_rows.sort(key=lambda row: (timestamp(row['case_start_time']), row['case_id']))
    labels = {row['case_id']: row['label'] for row in label_rows}
    by_id = {row['case_id']: row for row in label_rows}

    audit_dir = root / 'results/phase1_1/audit'
    stats_dir = root / 'results/phase1_1/stats'
    write_csv(audit_dir / 'case_labels.csv', label_rows)
    write_csv(audit_dir / 'excluded_label_cases.csv', exclusions,
              ['case_id', 'reason', 'candidate_count', 'latest_timestamp', 'latest_activities',
               'trace_length', 'case_start_time'])

    split_rows = []
    for split in ('train', 'val', 'test'):
        group = [row for row in label_rows if row['split'] == split]
        lengths = [row['trace_length'] for row in group]
        positives = sum(row['label'] for row in group)
        split_rows.append({'split': split, 'cases': len(group), 'events': sum(lengths),
                           'positives': positives, 'negatives': len(group)-positives,
                           'positive_rate': _rate(positives, len(group)),
                           'start_time_min': min(row['case_start_time'] for row in group),
                           'start_time_max': max(row['case_start_time'] for row in group),
                           'trace_length_min': min(lengths), 'trace_length_median': _median(lengths),
                           'trace_length_mean': statistics.mean(lengths), 'trace_length_max': max(lengths)})
    write_csv(stats_dir / 'split_summary.csv', split_rows)

    coverage_rows, activity_rows = [], []
    for k in CANDIDATES:
        eligible = [row for row in label_rows if k < row['label_event_index']]
        aggregate_terminal = Counter()
        aggregate_sensitive = Counter()
        history_cases = sensitive_cases = 0
        for row in eligible:
            observed = case_prefix_audits[row['case_id']][k]
            terminal, sensitive = Counter(observed['terminal']), Counter(observed['sensitive'])
            aggregate_terminal.update(terminal)
            aggregate_sensitive.update(sensitive)
            history_cases += bool(terminal)
            sensitive_cases += bool(sensitive)
        coverage_row = {
            'endpoint': k, 'total_labeled_cases': len(label_rows),
            'total_eligible_cases': len(eligible), 'eligible_rate': _rate(len(eligible), len(label_rows)),
            'excluded_by_label_event': len(label_rows)-len(eligible),
            'excluded_unlabeled_cases': len(exclusions),
            'history_terminal_cases': history_cases,
            'history_terminal_case_rate': _rate(history_cases, len(eligible)),
            'history_O_Accepted_events': aggregate_terminal['O_Accepted'],
            'history_O_Refused_events': aggregate_terminal['O_Refused'],
            'history_O_Cancelled_events': aggregate_terminal['O_Cancelled'],
            'sensitive_application_activity_cases': sensitive_cases,
        }
        for split in ('train', 'val', 'test'):
            group = [row for row in eligible if row['split'] == split]
            positives = sum(row['label'] for row in group)
            group_history_cases = sum(bool(case_prefix_audits[row['case_id']][k]['terminal']) for row in group)
            group_sensitive_cases = sum(bool(case_prefix_audits[row['case_id']][k]['sensitive']) for row in group)
            coverage_row.update({f'{split}_cases': len(group), f'{split}_positive': positives,
                                 f'{split}_negative': len(group)-positives,
                                 f'{split}_positive_rate': _rate(positives, len(group)),
                                 f'{split}_history_terminal_cases': group_history_cases,
                                 f'{split}_sensitive_activity_cases': group_sensitive_cases})
        coverage_rows.append(coverage_row)

        for split in ('all', 'train', 'val', 'test'):
            group = eligible if split == 'all' else [row for row in eligible if row['split'] == split]
            for kind, activities in (('historical_terminal_offer', TERMINAL_OFFER_ACTIVITIES),
                                     ('potential_outcome_proxy', SENSITIVE_APPLICATION_ACTIVITIES)):
                for activity in sorted(activities):
                    affected = []
                    event_count = 0
                    for row in group:
                        bucket = case_prefix_audits[row['case_id']][k][
                            'terminal' if kind == 'historical_terminal_offer' else 'sensitive']
                        if bucket.get(activity, 0):
                            affected.append(row)
                            event_count += bucket[activity]
                    positives = sum(row['label'] for row in affected)
                    activity_rows.append({'endpoint': k, 'split': split, 'kind': kind,
                                          'activity': activity, 'eligible_cases': len(group),
                                          'cases_with_activity': len(affected), 'event_count': event_count,
                                          'positive_cases_with_activity': positives,
                                          'negative_cases_with_activity': len(affected)-positives,
                                          'case_rate': _rate(len(affected), len(group))})
    write_csv(stats_dir / 'prefix_coverage_labeled.csv', coverage_rows)
    write_csv(audit_dir / 'prefix_outcome_activity_audit.csv', activity_rows)

    groups = {split: [row['case_id'] for row in label_rows if row['split'] == split]
              for split in ('train', 'val', 'test')}
    validate_no_overlap(groups)
    summary = {
        'phase': '1.1', 'raw_path': config['raw_path'], 'raw_sha256': actual_hash,
        'raw_cases': total_cases, 'labeled_cases': len(label_rows), 'excluded_cases': len(exclusions),
        'excluded_reasons': dict(Counter(row['reason'] for row in exclusions)),
        'positives': sum(row['label'] for row in label_rows),
        'negatives': len(label_rows)-sum(row['label'] for row in label_rows),
        'positive_rate': _rate(sum(row['label'] for row in label_rows), len(label_rows)),
        'accepted_field_events_not_used_as_labels': accepted_field_events,
        'selected_field_events_not_used_as_labels': selected_field_events,
        'split_case_counts': {row['split']: row['cases'] for row in split_rows},
        'prefix_candidates': list(CANDIDATES), 'final_prefix_endpoints': None,
        'models_trained': False, 'phase2_started': False,
    }
    write_json(audit_dir / 'phase1_1_summary.json', summary)
    write_json(audit_dir / 'label_gate_v2.json', {
        'status': 'verified_with_exclusions', 'definition': config['label_definition'],
        'raw_sha256': actual_hash, 'labels_constructed': True, 'split_constructed': True,
        'prefix_dataset_constructed': False, 'prefix_coverage_constructed': True,
        'excluded_cases': len(exclusions)})
    _write_docs(root, summary, split_rows, coverage_rows, activity_rows)
    source_files = [p for base in ('src', 'scripts', 'configs', 'tests')
                    for p in (root / base).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    source_files += [root / 'ETCE_EXPERIMENT_PLAN_v3.md', root / 'CODEX_PHASE1_TASK_v2.md']
    output_files = [audit_dir / name for name in (
        'case_labels.csv', 'excluded_label_cases.csv', 'label_definition.md',
        'leakage_audit_final_v2.md', 'prefix_outcome_activity_audit.csv', 'phase1_1_summary.json')]
    output_files += [stats_dir / 'split_summary.csv', stats_dir / 'prefix_coverage_labeled.csv',
                     root / 'results/phase1_1/PHASE1_1_REPORT.md']
    write_json(audit_dir / 'phase1_1_manifest.json', {
        'python': platform.python_version(), 'raw_sha256': actual_hash,
        'source_hashes': {p.relative_to(root).as_posix(): sha256(p) for p in sorted(source_files)},
        'output_hashes_before_validation': {p.relative_to(root).as_posix(): sha256(p) for p in output_files},
        'split_is_deterministic': True, 'random_sampling_used': False,
        'final_prefix_endpoints': None, 'models_trained': False})
    return summary


def _write_docs(root, summary, split_rows, coverage_rows, activity_rows):
    audit_dir = root / 'results/phase1_1/audit'
    definition = f'''# BPIC2017 Phase 1.1 标签定义

## Case-level target

对每个 case，仅考虑同时满足以下条件的事件：

- `EventOrigin == Offer`
- `concept:name` 属于 `O_Accepted`、`O_Refused`、`O_Cancelled`

按 `(time:timestamp, 原始 XES event index)` 排序，选最后一个 terminal offer event。原始 event index 为 1-based。

- 最终活动为 `O_Accepted`：`label=1`
- 最终活动为 `O_Refused` 或 `O_Cancelled`：`label=0`
- 无 terminal offer event：排除并报告
- 最新时间戳同时存在冲突 terminal outcomes：无法唯一判定，排除并报告
- 最新时间戳若有相同结果的重复事件，结果类别唯一；以原始顺序最后一个事件作为 label boundary，并保留计数供审计

`Accepted` 和 `Selected` 不参与标签定义。每个已标记 case 保存最终事件活动、`label_event_index` 和 UTC `label_event_timestamp`。

## Prefix eligibility

候选 `Prefix-k` 只在 `k < label_event_index` 时 eligible，内容严格为原始事件 `1..k`。不得包含最终标签事件或其后信息，也不删除事件后补足长度。

早期 offer 的 `O_Accepted/O_Refused/O_Cancelled` 可以作为历史轨迹保留并单独计数；其预测含义和是否进入后续模型仍须人工审核。最终 prefix 尚未选择。

## 本次结果

- 原始 cases：{summary['raw_cases']:,}
- 已标记：{summary['labeled_cases']:,}（positive={summary['positives']:,}, negative={summary['negatives']:,}）
- 排除：{summary['excluded_cases']:,}；原因：`{summary['excluded_reasons']}`
'''
    (audit_dir / 'label_definition.md').write_text(definition, encoding='utf-8')

    sensitive_rows = [r for r in activity_rows if r['split'] == 'all' and r['kind'] == 'potential_outcome_proxy']
    leakage = f'''# BPIC2017 Leakage Audit Final v2

## 标签与严格截止点

标签只由最终 terminal offer activity 定义。`Accepted` 和 `Selected` 各在原始日志出现 {summary['accepted_field_events_not_used_as_labels']:,} 次，均不用于标签或 predictor。
任何候选 prefix 必须满足 `k < label_event_index`，因此不会包含最终标签事件及其后信息。

## Predictor 字段排除

以下字段禁止进入 predictor：`Accepted`、`Selected`、case/application ID、`EventID`、`OfferID`、资源标识 `org:resource`。ID 只可作为 metadata 或关联审计使用。不得从完整 case 向早期 prefix 回填未来属性。

## 历史 terminal offer activities

没有机械删除最终标签之前的 `O_Accepted/O_Refused/O_Cancelled`。它们保留原始位置，并在 `prefix_outcome_activity_audit.csv` 按 endpoint、split、activity、最终 label 统计。它们可能高度预测最终 outcome，后续是否作为 predictor 必须人工批准；Phase 1.1 未构造模型输入。

## Application outcome proxy 审计

`A_Pending`、`A_Denied`、`A_Cancelled` 作为潜在直接 outcome proxy 单独统计，不自动放行：

| endpoint | activity | cases | positive cases | negative cases | eligible-case rate |
|---:|---|---:|---:|---:|---:|
'''
    for row in sensitive_rows:
        leakage += (f"| {row['endpoint']} | {row['activity']} | {row['cases_with_activity']} | "
                    f"{row['positive_cases_with_activity']} | {row['negative_cases_with_activity']} | "
                    f"{row['case_rate']:.6f} |\n")
    leakage += '''
## Training-only fit boundary

未来 scaler、imputer、category encoder、feature selection 只能在 training cases 上 fit。Phase 1.1 未拟合真实 encoder，也未生成 predictor dataset。
'''
    (audit_dir / 'leakage_audit_final_v2.md').write_text(leakage, encoding='utf-8')

    split_table = '\n'.join(f"| {r['split']} | {r['cases']:,} | {r['positives']:,} | {r['negatives']:,} | {r['positive_rate']:.4%} | {r['start_time_min']} | {r['start_time_max']} |"
                            for r in split_rows)
    prefix_table = '\n'.join(f"| {r['endpoint']} | {r['total_eligible_cases']:,} | {r['eligible_rate']:.4%} | {r['excluded_by_label_event']:,} | {r['train_cases']:,} | {r['val_cases']:,} | {r['test_cases']:,} |"
                             for r in coverage_rows)
    report = f'''# ETCE Phase 1.1 Report

## 结论

已按人工指定的 benchmark-compatible terminal offer outcome 定义完成标签、case-level temporal split、候选 prefix 覆盖与 leakage 复审。没有选择最终 prefix，没有构造训练特征，没有训练模型，也没有进入 Phase 2。

## 标签结果

- 原始日志：`{summary['raw_path']}`；SHA-256 `{summary['raw_sha256']}`。
- 原始 cases：{summary['raw_cases']:,}；已标记：{summary['labeled_cases']:,}；排除：{summary['excluded_cases']:,}。
- positive：{summary['positives']:,}；negative：{summary['negatives']:,}；positive rate：{summary['positive_rate']:.4%}。
- 排除原因：`{summary['excluded_reasons']}`。
- `Accepted=True` 未用于标签。完整定义与同时间戳处理见 `results/phase1_1/audit/label_definition.md`。

## Temporal split

对所有已标记 case 按 `(case_start_time, case_id)` 排序，以 floor(0.6N)、floor(0.8N) 为边界。同一 case 不跨 split。

| split | cases | positive | negative | positive rate | start min | start max |
|---|---:|---:|---:|---:|---|---|
{split_table}

## Prefix 候选覆盖

Eligibility 严格为 `k < label_event_index`。排除计数是已标记 cases 中由于达到或越过最终标签事件而不可用的数量；未标记 cases 另行报告。最终端点未选择。

| k | eligible | eligible rate | excluded by label event | train | val | test |
|---:|---:|---:|---:|---:|---:|---:|
{prefix_table}

各 split 的正负数、positive rate、历史 terminal offer 活动和 application outcome proxy 计数见 `results/phase1_1/stats/prefix_coverage_labeled.csv` 与 `results/phase1_1/audit/prefix_outcome_activity_audit.csv`。

## Leakage 决策

禁止 predictor 字段：`Accepted`、`Selected`、case/application/event/offer ID、resource ID。早期 offer 历史 terminal activities 未被机械删除，已逐 endpoint/split 审计。`A_Pending/A_Denied/A_Cancelled` 被标记为潜在 outcome proxy，未自动放行。详见 `leakage_audit_final_v2.md`。

## 验证与停止点

<!-- VALIDATION_START -->
等待 `03_validate_bpic2017_dataset.py` 验证。
<!-- VALIDATION_END -->

Phase 1.1 到此停止。没有训练 RF/XGBoost/MLP/LSTM，没有实现或运行 ETCE。
'''
    report_path = root / 'results/phase1_1/PHASE1_1_REPORT.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding='utf-8')
