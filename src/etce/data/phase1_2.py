"""Phase 1.2: benchmark label comparison and leakage-safe prefix cohorts."""
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
from pathlib import Path
import platform

from .labels import derive_benchmark_label, derive_case_label, TERMINAL_OFFER_ACTIVITIES
from .load_bpic2017 import read_cases, timestamp
from .split import temporal_split, validate_no_overlap
from etce.utils.io import sha256, write_csv, write_json

MAIN_ENDPOINTS = (5, 10, 15, 20)
AUDIT_ENDPOINTS = (25, 30)
ALL_ENDPOINTS = MAIN_ENDPOINTS + AUDIT_ENDPOINTS
DIRECT_OUTCOME_PROXIES = frozenset({'A_Pending', 'A_Denied', 'A_Cancelled'})


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _id_hash(case_ids):
    value = '\n'.join(sorted(case_ids)) + '\n'
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _comparison_row(case_id, current, current_exclusion, benchmark, benchmark_exclusion):
    row = {'case_id': case_id,
           'current_status': 'labeled' if current else 'excluded',
           'current_label': current.label if current else None,
           'current_activity': current.label_source_value_or_activity if current else None,
           'current_event_index': current.label_event_index if current else None,
           'current_event_timestamp': current.label_event_timestamp if current else None,
           'current_exclusion_reason': current_exclusion.get('reason') if current_exclusion else None,
           'benchmark_status': 'labeled' if benchmark else 'excluded',
           'benchmark_label': benchmark.label if benchmark else None,
           'benchmark_activity': benchmark.label_source_value_or_activity if benchmark else benchmark_exclusion.get('last_offer_activity'),
           'benchmark_event_index': benchmark.label_event_index if benchmark else benchmark_exclusion.get('last_offer_event_index'),
           'benchmark_event_timestamp': benchmark.label_event_timestamp if benchmark else benchmark_exclusion.get('last_offer_event_timestamp'),
           'benchmark_exclusion_reason': benchmark_exclusion.get('reason') if benchmark_exclusion else None}
    if current and benchmark:
        label_equal = current.label == benchmark.label
        boundary_equal = (current.label_event_index == benchmark.label_event_index
                          and current.label_event_timestamp == benchmark.label_event_timestamp)
        if label_equal and boundary_equal:
            category = 'same_label_same_boundary'
        elif label_equal:
            category = 'same_label_different_boundary'
        else:
            category = 'different_label'
    elif current and not benchmark:
        category = 'current_labeled_benchmark_excluded'
    elif benchmark and not current:
        category = 'current_excluded_benchmark_labeled'
    else:
        category = ('both_excluded_same_reason' if current_exclusion.get('reason') == benchmark_exclusion.get('reason')
                    else 'both_excluded_different_reason')
    row['comparison_category'] = category
    return row


def run_phase1_2(root, config):
    root = Path(root)
    raw = root / config['raw_path']
    raw_hash = sha256(raw)
    if raw_hash != config['raw_sha256']:
        raise ValueError('Raw BPIC2017 SHA-256 does not match audited configuration')

    comparisons, final_labels, prefix_audits = [], [], {}
    for count, case in enumerate(read_cases(raw), 1):
        current, current_exclusion = derive_case_label(case)
        benchmark, benchmark_exclusion = derive_benchmark_label(case)
        comparisons.append(_comparison_row(case.case_id, current, current_exclusion,
                                           benchmark, benchmark_exclusion))
        # The Phase 1.2 instruction explicitly prefers the exact benchmark rule
        # after differences are measured and reported. Final cohorts use benchmark.
        if benchmark:
            final_labels.append(asdict(benchmark))
            per_endpoint = {}
            for k in ALL_ENDPOINTS:
                prefix = case.events[:k]
                proxy = Counter(event.get('concept:name') for event in prefix
                                if event.get('concept:name') in DIRECT_OUTCOME_PROXIES)
                historical = Counter(event.get('concept:name') for event in prefix
                                     if event.get('EventOrigin') == 'Offer'
                                     and event.get('concept:name') in TERMINAL_OFFER_ACTIVITIES
                                     and k < benchmark.label_event_index)
                per_endpoint[k] = {'proxy': dict(proxy), 'historical': dict(historical)}
            prefix_audits[case.case_id] = per_endpoint
        if count % 5000 == 0:
            print(f'Compared {count} raw cases', flush=True)

    audit_dir = root / 'results/phase1_2/audit'
    stats_dir = root / 'results/phase1_2/stats'
    comparison_csv = audit_dir / 'label_definition_comparison.csv'
    write_csv(comparison_csv, comparisons)
    categories = Counter(row['comparison_category'] for row in comparisons)
    _write_label_comparison(audit_dir / 'label_definition_comparison.md', categories,
                            comparisons, raw_hash)

    # The difference report now exists. Adopt the exact benchmark definition.
    assignments = temporal_split(final_labels)
    for row in final_labels:
        row['split'] = assignments[row['case_id']]
    final_labels.sort(key=lambda row: (timestamp(row['case_start_time']), row['case_id']))
    write_csv(audit_dir / 'case_labels_phase1_2.csv', final_labels)
    excluded = [row for row in comparisons if row['benchmark_status'] == 'excluded']
    write_csv(audit_dir / 'excluded_cases_phase1_2.csv', excluded)

    groups = {split: [row['case_id'] for row in final_labels if row['split'] == split]
              for split in ('train', 'val', 'test')}
    validate_no_overlap(groups)
    split_rows = []
    for split in ('train', 'val', 'test'):
        group = [row for row in final_labels if row['split'] == split]
        positives = sum(row['label'] for row in group)
        split_rows.append({'split': split, 'cases': len(group), 'positives': positives,
                           'negatives': len(group)-positives,
                           'positive_rate': _rate(positives, len(group)),
                           'start_time_min': min(row['case_start_time'] for row in group),
                           'start_time_max': max(row['case_start_time'] for row in group)})
    write_csv(stats_dir / 'split_summary_phase1_2.csv', split_rows)

    eligibility_rows, proxy_rows = [], []
    endpoint_cohorts = {}
    historical_rows = []
    for k in ALL_ENDPOINTS:
        role = 'main_candidate' if k in MAIN_ENDPOINTS else 'audit_only'
        boundary = [row for row in final_labels if k < row['label_event_index']]
        endpoint = [row for row in boundary if not prefix_audits[row['case_id']][k]['proxy']]
        endpoint_cohorts[k] = {row['case_id'] for row in endpoint}
        for row in boundary:
            proxy = prefix_audits[row['case_id']][k]['proxy']
            if proxy:
                proxy_rows.append({'case_id': row['case_id'], 'split': row['split'], 'label': row['label'],
                                   'endpoint': k, 'status': 'outcome_proxy_excluded',
                                   'proxy_activities': '|'.join(sorted(proxy)),
                                   'proxy_event_count': sum(proxy.values())})
        for split in ('all', 'train', 'val', 'test'):
            universe = final_labels if split == 'all' else [row for row in final_labels if row['split'] == split]
            boundary_group = boundary if split == 'all' else [row for row in boundary if row['split'] == split]
            endpoint_group = endpoint if split == 'all' else [row for row in endpoint if row['split'] == split]
            proxy_all = [row for row in universe if prefix_audits[row['case_id']][k]['proxy']]
            proxy_excluded = [row for row in boundary_group if prefix_audits[row['case_id']][k]['proxy']]
            positives = sum(row['label'] for row in endpoint_group)
            historical_cases = sum(bool(prefix_audits[row['case_id']][k]['historical']) for row in endpoint_group)
            eligibility_rows.append({
                'endpoint': k, 'candidate_role': role, 'cohort': 'endpoint_specific', 'split': split,
                'total_labeled_cases': len(universe), 'boundary_eligible_cases': len(boundary_group),
                'excluded_by_label_boundary': len(universe)-len(boundary_group),
                'outcome_proxy_present_cases': len(proxy_all),
                'outcome_proxy_excluded_cases': len(proxy_excluded),
                'eligible_cases': len(endpoint_group), 'positive': positives,
                'negative': len(endpoint_group)-positives,
                'positive_rate': _rate(positives, len(endpoint_group)),
                'historical_terminal_cases': historical_cases,
                'historical_terminal_case_rate': _rate(historical_cases, len(endpoint_group))})
            _append_historical(historical_rows, 'endpoint_specific', k, split, endpoint_group, prefix_audits)
    write_csv(stats_dir / 'prefix_eligibility_final.csv', eligibility_rows)
    write_csv(audit_dir / 'outcome_proxy_exclusions.csv', proxy_rows)

    common = [row for row in final_labels
              if 20 < row['label_event_index']
              and all(not prefix_audits[row['case_id']][k]['proxy'] for k in MAIN_ENDPOINTS)]
    common_ids = {row['case_id'] for row in common}
    write_csv(audit_dir / 'common_cohort_case_ids.csv', [
        {'case_id': row['case_id'], 'split': row['split'], 'label': row['label'],
         'label_event_index': row['label_event_index'], 'label_event_timestamp': row['label_event_timestamp']}
        for row in common])
    common_rows = []
    for split in ('all', 'train', 'val', 'test'):
        group = common if split == 'all' else [row for row in common if row['split'] == split]
        ids = {row['case_id'] for row in group}
        positives = sum(row['label'] for row in group)
        row = {'cohort': 'common_prefix_5_10_15_20', 'split': split, 'cases': len(group),
               'positive': positives, 'negative': len(group)-positives,
               'positive_rate': _rate(positives, len(group)), 'case_id_sha256': _id_hash(ids),
               'identical_case_ids_across_prefixes': True}
        for k in MAIN_ENDPOINTS:
            row[f'prefix_{k}_cases'] = len(group)
            row[f'prefix_{k}_case_id_sha256'] = _id_hash(ids)
            _append_historical(historical_rows, 'common', k, split, group, prefix_audits)
        common_rows.append(row)
    write_csv(stats_dir / 'common_cohort_summary.csv', common_rows)
    write_csv(audit_dir / 'historical_terminal_activity_phase1_2.csv', historical_rows)

    comparison_differences = len(comparisons) - categories['same_label_same_boundary']
    summary = {
        'phase': '1.2', 'raw_sha256': raw_hash, 'raw_cases': len(comparisons),
        'comparison_categories': dict(categories), 'comparison_differences': comparison_differences,
        'final_label_definition': 'benchmark_last_offer_origin_event',
        'final_labeled_cases': len(final_labels), 'final_excluded_cases': len(excluded),
        'positives': sum(row['label'] for row in final_labels),
        'negatives': len(final_labels)-sum(row['label'] for row in final_labels),
        'main_prefix_candidates': list(MAIN_ENDPOINTS), 'audit_only_prefixes': list(AUDIT_ENDPOINTS),
        'common_cohort_cases': len(common), 'common_case_id_sha256': _id_hash(common_ids),
        'formal_main_prefixes_fixed': True, 'models_trained': False, 'phase2_started': False}
    write_json(audit_dir / 'phase1_2_summary.json', summary)
    _write_leakage_v3(audit_dir / 'leakage_audit_final_v3.md', summary, eligibility_rows,
                      historical_rows)
    _write_report(root / 'results/phase1_2/PHASE1_2_REPORT.md', summary, split_rows, eligibility_rows,
                  common_rows)
    source_files = [p for base in ('src', 'scripts', 'configs', 'tests')
                    for p in (root / base).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    source_files += [root / 'ETCE_EXPERIMENT_PLAN_v3.md', root / 'CODEX_PHASE1_TASK_v2.md']
    output_files = [audit_dir / name for name in (
        'label_definition_comparison.csv', 'label_definition_comparison.md',
        'case_labels_phase1_2.csv', 'excluded_cases_phase1_2.csv',
        'outcome_proxy_exclusions.csv', 'common_cohort_case_ids.csv',
        'historical_terminal_activity_phase1_2.csv', 'leakage_audit_final_v3.md',
        'phase1_2_summary.json')]
    output_files += [stats_dir / name for name in (
        'split_summary_phase1_2.csv', 'prefix_eligibility_final.csv', 'common_cohort_summary.csv')]
    output_files += [root / 'results/phase1_2/PHASE1_2_REPORT.md']
    write_json(audit_dir / 'phase1_2_manifest.json', {
        'python': platform.python_version(), 'raw_sha256': raw_hash,
        'source_hashes': {p.relative_to(root).as_posix(): sha256(p) for p in sorted(source_files)},
        'output_hashes_before_validation': {p.relative_to(root).as_posix(): sha256(p) for p in output_files},
        'random_sampling_used': False, 'formal_main_prefixes': list(MAIN_ENDPOINTS),
        'audit_only_prefixes': list(AUDIT_ENDPOINTS), 'models_trained': False})
    return summary


def _append_historical(rows, cohort, endpoint, split, group, audits):
    for activity in sorted(TERMINAL_OFFER_ACTIVITIES):
        for final_label in (0, 1):
            label_group = [row for row in group if row['label'] == final_label]
            affected = [row for row in label_group
                        if audits[row['case_id']][endpoint]['historical'].get(activity, 0)]
            event_count = sum(audits[row['case_id']][endpoint]['historical'].get(activity, 0)
                              for row in affected)
            rows.append({'cohort': cohort, 'endpoint': endpoint, 'split': split,
                         'historical_activity': activity, 'final_label': final_label,
                         'denominator_cases': len(label_group), 'cases_with_activity': len(affected),
                         'event_count': event_count, 'case_rate': _rate(len(affected), len(label_group))})


def _write_label_comparison(path, categories, rows, raw_hash):
    current_labeled = sum(row['current_status'] == 'labeled' for row in rows)
    benchmark_labeled = sum(row['benchmark_status'] == 'labeled' for row in rows)
    label_differences = categories['different_label']
    boundary_differences = categories['same_label_different_boundary'] + label_differences
    newly_excluded = [row for row in rows
                      if row['comparison_category'] == 'current_labeled_benchmark_excluded']
    benchmark_exclusion_activities = Counter(
        row['benchmark_activity'] or '<missing>' for row in rows if row['benchmark_status'] == 'excluded')
    text = f'''# BPIC2017 标签定义逐 case 比较

原始日志 SHA-256：`{raw_hash}`；比较 {len(rows):,} 个 case。

## 定义

- Phase 1.1：在 `EventOrigin=Offer` 且 activity 属于 `O_Accepted/O_Refused/O_Cancelled` 的事件中取最后一个。
- Benchmark：取最后一个 `EventOrigin=Offer` 事件；只有其 activity 属于上述三个 terminal activities 时才标记，否则排除。
- 两者均按 `(time:timestamp, 原始 1-based event index)` 确定最后事件。

## 比较结果

- Phase 1.1 labeled：{current_labeled:,}
- Benchmark labeled：{benchmark_labeled:,}
- 标签值不同：{label_differences:,}
- 两者均有标签但 boundary 不同：{boundary_differences:,}

| category | cases |
|---|---:|
'''
    for category, count in sorted(categories.items()):
        text += f'| {category} | {count:,} |\n'
    text += '\nBenchmark 排除 case 的最后 Offer-origin activity 分布：\n\n| activity | cases |\n|---|---:|\n'
    for activity, count in sorted(benchmark_exclusion_activities.items()):
        text += f'| {activity} | {count:,} |\n'
    text += '\nPhase 1.1 有标签、Benchmark 新排除的 4 个 case：\n\n'
    text += '| case_id | Phase 1.1 label/activity/boundary | benchmark last Offer activity/boundary |\n|---|---|---|\n'
    for row in newly_excluded:
        text += (f"| {row['case_id']} | {row['current_label']} / {row['current_activity']} / "
                 f"{row['current_event_index']} | {row['benchmark_activity']} / "
                 f"{row['benchmark_event_index']} |\n")
    text += '''
逐 case 证据见 `label_definition_comparison.csv`，包括两种定义的 label、activity、event index、timestamp 和排除原因。

## Phase 1.2 决定

差异已经先完成记录。依照人工指令，Phase 1.2 的最终标签采用 benchmark 精确定义；Phase 1.1 文件保持不变用于追踪，不静默覆盖。新标签保存为 `case_labels_phase1_2.csv`。
'''
    Path(path).write_text(text, encoding='utf-8')


def _write_leakage_v3(path, summary, eligibility, historical):
    all_rows = [row for row in eligibility if row['split'] == 'all']
    text = '''# BPIC2017 Leakage Audit Final v3

## 永久禁止 predictor 字段

`Accepted`、`Selected`、case/application ID、`EventID`、`OfferID`、resource ID (`org:resource`) 永久禁止进入 predictor。ID 仅用于审计关联和 metadata。禁止从完整 case 向 prefix 回填未来属性。

## Prefix 与 direct application-outcome proxy

不删除原始事件，也不删除后补足 prefix。先要求 `k < label_event_index`；随后若原始前 k 个事件包含 `A_Pending/A_Denied/A_Cancelled`，该 case 在该 endpoint 标记为 `outcome_proxy_excluded`。

| k | role | boundary eligible | proxy excluded | final eligible | positive rate |
|---:|---|---:|---:|---:|---:|
'''
    for row in all_rows:
        text += (f"| {row['endpoint']} | {row['candidate_role']} | {row['boundary_eligible_cases']} | "
                 f"{row['outcome_proxy_excluded_cases']} | {row['eligible_cases']} | "
                 f"{row['positive_rate']:.6f} |\n")
    text += '''
25、30 仅保留 audit-only 统计。正式主实验 endpoints 固定为 5、10、15、20；common cohort 进一步固定为 Prefix-20 eligible 且前 20 个事件无上述 proxy 的同一批 case。

## 历史 terminal offer activities

最终 label boundary 之前的 `O_Accepted/O_Refused/O_Cancelled` 不机械删除。其按 cohort、endpoint、split、activity 和最终 label 的 case/event 计数保存在 `historical_terminal_activity_phase1_2.csv`。这些是真实的早期 offer 历史，但仍可能具有很强预测性；Phase 1.2 只审计，不训练模型。

## 状态

未构造 predictor matrix，未 fit encoder/scaler/imputer，未训练模型，未进入 Phase 2。
'''
    Path(path).write_text(text, encoding='utf-8')


def _write_report(path, summary, splits, eligibility, common):
    comparison = '\n'.join(f'| {key} | {value:,} |' for key, value in sorted(summary['comparison_categories'].items()))
    split_table = '\n'.join(f"| {r['split']} | {r['cases']:,} | {r['positives']:,} | {r['negatives']:,} | {r['positive_rate']:.4%} |"
                            for r in splits)
    endpoint = [row for row in eligibility if row['split'] == 'all']
    endpoint_table = '\n'.join(f"| {r['endpoint']} | {r['candidate_role']} | {r['boundary_eligible_cases']:,} | {r['outcome_proxy_excluded_cases']:,} | {r['eligible_cases']:,} | {r['positive']:,} | {r['negative']:,} | {r['positive_rate']:.4%} |"
                               for r in endpoint)
    common_table = '\n'.join(f"| {r['split']} | {r['cases']:,} | {r['positive']:,} | {r['negative']:,} | {r['positive_rate']:.4%} | `{r['case_id_sha256']}` |"
                             for r in common)
    text = f'''# ETCE Phase 1.2 Report

## 标签定义比较与最终决定

逐 case 比较了 Phase 1.1 的 last-terminal-offer 定义与精确 benchmark last-Offer-origin-event 定义。完整证据见 `results/phase1_2/audit/label_definition_comparison.md` 和对应 CSV。

| comparison category | cases |
|---|---:|
{comparison}

差异已报告。按人工指令，Phase 1.2 最终采用 benchmark 精确定义；Phase 1.1 标签文件未覆盖。最终 labeled={summary['final_labeled_cases']:,}，excluded={summary['final_excluded_cases']:,}，positive={summary['positives']:,}，negative={summary['negatives']:,}。

## Benchmark 标签的 temporal split

| split | cases | positive | negative | positive rate |
|---|---:|---:|---:|---:|
{split_table}

## Endpoint-specific eligible cohort

先要求 `k < label_event_index`，再排除原始 Prefix-k 内出现 `A_Pending/A_Denied/A_Cancelled` 的 case。不删除事件、不补足 prefix。

| k | role | boundary eligible | proxy excluded | eligible | positive | negative | positive rate |
|---:|---|---:|---:|---:|---:|---:|---:|
{endpoint_table}

正式主实验 endpoints 固定为 5/10/15/20；25/30 仅为 audit-only。

## Common cohort

定义：benchmark label 已确定、`label_event_index > 20`，并且 Prefix-5/10/15/20 均不存在 direct application-outcome proxy。common cohort 共 {summary['common_cohort_cases']:,} cases。

| split | cases | positive | negative | positive rate | case-ID SHA-256 |
|---|---:|---:|---:|---:|---|
{common_table}

`common_cohort_summary.csv` 同时保存四个 prefix 的 case 数和 case-ID hash；四个 prefix 完全相同。主实验优先考虑 common cohort。

## Leakage 与停止点

永久禁止 `Accepted`、`Selected`、case/application/event/offer ID 和 resource ID；禁止未来信息回填。历史 terminal offer activities 保留并按最终 label 单独统计。详见 `leakage_audit_final_v3.md`。

<!-- VALIDATION_START -->
等待 Phase 1.2 验证。
<!-- VALIDATION_END -->

Phase 1.2 到此停止。未训练 RF/XGBoost/MLP/LSTM，未实现 surrogate、GA、D-S 或 ETCE，未进入 Phase 2。
'''
    Path(path).write_text(text, encoding='utf-8')
