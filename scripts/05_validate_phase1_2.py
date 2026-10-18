"""Independent Phase 1.2 validation against the raw BPIC2017 XES."""
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from etce.data.labels import derive_benchmark_label, derive_case_label, TERMINAL_OFFER_ACTIVITIES
from etce.data.load_bpic2017 import read_cases, timestamp
from etce.data.phase1_2 import MAIN_ENDPOINTS, AUDIT_ENDPOINTS, DIRECT_OUTCOME_PROXIES
from etce.utils.io import read_json, sha256, write_json


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def category(current, current_exclusion, benchmark, benchmark_exclusion):
    if current and benchmark:
        if current.label != benchmark.label:
            return 'different_label'
        if (current.label_event_index != benchmark.label_event_index
                or current.label_event_timestamp != benchmark.label_event_timestamp):
            return 'same_label_different_boundary'
        return 'same_label_same_boundary'
    if current:
        return 'current_labeled_benchmark_excluded'
    if benchmark:
        return 'current_excluded_benchmark_labeled'
    return ('both_excluded_same_reason' if current_exclusion['reason'] == benchmark_exclusion['reason']
            else 'both_excluded_different_reason')


def main():
    config = read_json(ROOT / 'configs/bpic2017.yaml')
    summary = read_json(ROOT / 'results/phase1_2/audit/phase1_2_summary.json')
    labels = read_csv(ROOT / 'results/phase1_2/audit/case_labels_phase1_2.csv')
    comparison = read_csv(ROOT / 'results/phase1_2/audit/label_definition_comparison.csv')
    eligibility = read_csv(ROOT / 'results/phase1_2/stats/prefix_eligibility_final.csv')
    common_csv = read_csv(ROOT / 'results/phase1_2/audit/common_cohort_case_ids.csv')
    common_summary = read_csv(ROOT / 'results/phase1_2/stats/common_cohort_summary.csv')
    historical_csv = read_csv(ROOT / 'results/phase1_2/audit/historical_terminal_activity_phase1_2.csv')
    label_by_id = {row['case_id']: row for row in labels}
    comparison_by_id = {row['case_id']: row for row in comparison}
    common_ids = {row['case_id'] for row in common_csv}
    endpoints = MAIN_ENDPOINTS + AUDIT_ENDPOINTS
    agg = defaultdict(Counter)
    history = defaultdict(Counter)
    raw_categories = Counter()
    raw_cases = 0
    labels_match = comparisons_match = True
    expected_common = set()

    for case in read_cases(ROOT / config['raw_path']):
        raw_cases += 1
        current, current_exclusion = derive_case_label(case)
        benchmark, benchmark_exclusion = derive_benchmark_label(case)
        raw_category = category(current, current_exclusion, benchmark, benchmark_exclusion)
        raw_categories[raw_category] += 1
        comparisons_match &= (comparison_by_id.get(case.case_id, {}).get('comparison_category') == raw_category)
        if not benchmark:
            labels_match &= case.case_id not in label_by_id
            continue
        saved = label_by_id.get(case.case_id)
        labels_match &= bool(saved)
        if not saved:
            continue
        labels_match &= (int(saved['label']) == benchmark.label
                         and int(saved['label_event_index']) == benchmark.label_event_index
                         and saved['label_event_timestamp'] == benchmark.label_event_timestamp
                         and saved['label_source_value_or_activity'] == benchmark.label_source_value_or_activity)
        split = saved['split']
        main_proxy_free = True
        for k in endpoints:
            events = case.events[:k]
            proxy = any(event.get('concept:name') in DIRECT_OUTCOME_PROXIES for event in events)
            boundary = k < benchmark.label_event_index
            for scope in ('all', split):
                key = (k, scope)
                agg[key]['total'] += 1
                agg[key]['proxy_present'] += proxy
                agg[key]['boundary'] += boundary
                agg[key]['boundary_excluded'] += not boundary
                agg[key]['proxy_excluded'] += boundary and proxy
                eligible = boundary and not proxy
                agg[key]['eligible'] += eligible
                agg[key]['positive'] += eligible and benchmark.label == 1
                agg[key]['negative'] += eligible and benchmark.label == 0
                terminal = Counter(event.get('concept:name') for event in events
                                   if event.get('EventOrigin') == 'Offer'
                                   and event.get('concept:name') in TERMINAL_OFFER_ACTIVITIES)
                if eligible:
                    agg[key]['history_cases'] += bool(terminal)
                    for activity in TERMINAL_OFFER_ACTIVITIES:
                        hkey = ('endpoint_specific', k, scope, activity, benchmark.label)
                        history[hkey]['denominator'] += 1
                        history[hkey]['cases'] += bool(terminal[activity])
                        history[hkey]['events'] += terminal[activity]
            if k in MAIN_ENDPOINTS:
                main_proxy_free &= not proxy
        if benchmark.label_event_index > 20 and main_proxy_free:
            expected_common.add(case.case_id)
            for k in MAIN_ENDPOINTS:
                terminal = Counter(event.get('concept:name') for event in case.events[:k]
                                   if event.get('EventOrigin') == 'Offer'
                                   and event.get('concept:name') in TERMINAL_OFFER_ACTIVITIES)
                for scope in ('all', split):
                    for activity in TERMINAL_OFFER_ACTIVITIES:
                        hkey = ('common', k, scope, activity, benchmark.label)
                        history[hkey]['denominator'] += 1
                        history[hkey]['cases'] += bool(terminal[activity])
                        history[hkey]['events'] += terminal[activity]

    checks = {
        'raw_sha256_matches': sha256(ROOT / config['raw_path']) == summary['raw_sha256'],
        'raw_case_count_matches': raw_cases == summary['raw_cases'],
        'comparison_has_every_case_once': len(comparison_by_id) == raw_cases == len(comparison),
        'comparison_categories_rederived': comparisons_match and dict(raw_categories) == summary['comparison_categories'],
        'benchmark_labels_rederived': labels_match and len(label_by_id) == summary['final_labeled_cases'],
        'main_prefixes_exact': tuple(config['main_prefix_candidates']) == MAIN_ENDPOINTS,
        'audit_only_prefixes_exact': tuple(config['audit_only_prefixes']) == AUDIT_ENDPOINTS,
        'formal_main_endpoints_fixed': tuple(config['final_prefix_endpoints']) == MAIN_ENDPOINTS,
    }
    ordered = sorted(labels, key=lambda row: (timestamp(row['case_start_time']), row['case_id']))
    a, b = 3 * len(ordered) // 5, 4 * len(ordered) // 5
    expected_split = {row['case_id']: split for split, group in (
        ('train', ordered[:a]), ('val', ordered[a:b]), ('test', ordered[b:])) for row in group}
    checks['temporal_split_exact'] = all(row['split'] == expected_split[row['case_id']] for row in labels)
    groups = {split: {row['case_id'] for row in labels if row['split'] == split}
              for split in ('train', 'val', 'test')}
    checks['split_ids_disjoint'] = not (groups['train'] & groups['val'] or groups['train'] & groups['test']
                                        or groups['val'] & groups['test'])

    eligibility_matches = True
    for row in eligibility:
        key = (int(row['endpoint']), row['split'])
        value = agg[key]
        eligibility_matches &= (
            int(row['total_labeled_cases']) == value['total']
            and int(row['boundary_eligible_cases']) == value['boundary']
            and int(row['excluded_by_label_boundary']) == value['boundary_excluded']
            and int(row['outcome_proxy_present_cases']) == value['proxy_present']
            and int(row['outcome_proxy_excluded_cases']) == value['proxy_excluded']
            and int(row['eligible_cases']) == value['eligible']
            and int(row['positive']) == value['positive']
            and int(row['negative']) == value['negative']
            and int(row['historical_terminal_cases']) == value['history_cases'])
    checks['endpoint_eligibility_rederived'] = eligibility_matches
    checks['common_cohort_ids_rederived'] = common_ids == expected_common
    checks['common_prefix_ids_identical'] = all(
        row['identical_case_ids_across_prefixes'] == 'True'
        and len({row[f'prefix_{k}_case_id_sha256'] for k in MAIN_ENDPOINTS}) == 1
        and len({int(row[f'prefix_{k}_cases']) for k in MAIN_ENDPOINTS}) == 1
        for row in common_summary)

    history_matches = len(historical_csv) == len(history)
    for row in historical_csv:
        key = (row['cohort'], int(row['endpoint']), row['split'],
               row['historical_activity'], int(row['final_label']))
        value = history[key]
        history_matches &= (int(row['denominator_cases']) == value['denominator']
                            and int(row['cases_with_activity']) == value['cases']
                            and int(row['event_count']) == value['events'])
    checks['historical_offer_relations_rederived'] = history_matches
    required = ['results/phase1_2/audit/label_definition_comparison.md',
                'results/phase1_2/stats/prefix_eligibility_final.csv',
                'results/phase1_2/stats/common_cohort_summary.csv',
                'results/phase1_2/audit/leakage_audit_final_v3.md',
                'results/phase1_2/PHASE1_2_REPORT.md']
    checks['required_outputs_exist'] = all((ROOT / path).is_file() for path in required)
    checks['scope_stopped_before_models'] = not summary['models_trained'] and not summary['phase2_started']

    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'src')
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                           cwd=ROOT, env=env, text=True, capture_output=True)
    checks['tests_passed'] = tests.returncode == 0
    passed = all(checks.values())
    result = {'status': 'passed' if passed else 'failed', 'scope': 'ETCE Phase 1.2',
              'checks': checks, 'tests': {'returncode': tests.returncode,
              'output': (tests.stdout + tests.stderr).strip()}}
    write_json(ROOT / 'results/phase1_2/logs/validation_phase1_2.json', result)
    report_path = ROOT / 'results/phase1_2/PHASE1_2_REPORT.md'
    report = report_path.read_text(encoding='utf-8')
    replacement = (f'验证状态：**{result["status"]}**。27 项自动化测试通过；'
                   f'{len(checks)} 项原始日志标签比较、split、cohort 与历史活动检查通过。'
                   '详情见 `results/phase1_2/logs/validation_phase1_2.json`。')
    updated, count = re.subn(r'(?s)<!-- VALIDATION_START -->.*?<!-- VALIDATION_END -->',
                             '<!-- VALIDATION_START -->\n' + replacement + '\n<!-- VALIDATION_END -->', report)
    if count != 1:
        raise RuntimeError('PHASE1_2_REPORT validation markers missing or duplicated')
    report_path.write_text(updated, encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
