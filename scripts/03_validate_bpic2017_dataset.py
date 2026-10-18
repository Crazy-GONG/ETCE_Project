"""Validate real Phase 1.1 labels, temporal split, and candidate coverage."""
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from etce.data.labels import derive_case_label
from etce.data.load_bpic2017 import read_cases, timestamp
from etce.data.prefix import CANDIDATES
from etce.utils.io import read_json, sha256, write_json


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    config = read_json(ROOT / 'configs/bpic2017.yaml')
    summary = read_json(ROOT / 'results/phase1_1/audit/phase1_1_summary.json')
    labels = read_csv(ROOT / 'results/phase1_1/audit/case_labels.csv')
    excluded = read_csv(ROOT / 'results/phase1_1/audit/excluded_label_cases.csv')
    splits = read_csv(ROOT / 'results/phase1_1/stats/split_summary.csv')
    coverage = read_csv(ROOT / 'results/phase1_1/stats/prefix_coverage_labeled.csv')
    label_by_id = {row['case_id']: row for row in labels}
    excluded_by_id = {row['case_id']: row for row in excluded}
    checks = {
        'raw_hash_matches': sha256(ROOT / config['raw_path']) == summary['raw_sha256'],
        'label_case_ids_unique': len(label_by_id) == len(labels),
        'excluded_case_ids_unique': len(excluded_by_id) == len(excluded),
        'labeled_and_excluded_disjoint': not (set(label_by_id) & set(excluded_by_id)),
        'all_raw_cases_accounted_for': len(labels) + len(excluded) == summary['raw_cases'],
        'candidate_endpoints_exact': [int(row['endpoint']) for row in coverage] == list(CANDIDATES),
        'final_prefix_not_selected': summary['final_prefix_endpoints'] is None,
        'models_not_trained': summary['models_trained'] is False and summary['phase2_started'] is False,
    }

    raw_count = valid_count = excluded_count = 0
    raw_labels_match = raw_exclusions_match = True
    for case in read_cases(ROOT / config['raw_path']):
        raw_count += 1
        derived, exclusion = derive_case_label(case)
        if derived:
            valid_count += 1
            row = label_by_id.get(case.case_id)
            event = case.events[derived.label_event_index - 1]
            raw_labels_match &= bool(row)
            if row:
                raw_labels_match &= (
                    int(row['label']) == derived.label
                    and int(row['label_event_index']) == derived.label_event_index
                    and row['label_event_timestamp'] == derived.label_event_timestamp
                    and row['label_source_value_or_activity'] == derived.label_source_value_or_activity
                    and event.get('EventOrigin') == 'Offer'
                    and event.get('concept:name') == derived.label_source_value_or_activity)
        else:
            excluded_count += 1
            row = excluded_by_id.get(case.case_id)
            raw_exclusions_match &= bool(row) and row['reason'] == exclusion['reason']
    checks['raw_case_count_matches'] = raw_count == summary['raw_cases']
    checks['all_labels_rederived_from_raw'] = raw_labels_match and valid_count == len(labels)
    checks['all_exclusions_rederived_from_raw'] = raw_exclusions_match and excluded_count == len(excluded)

    ordered = sorted(labels, key=lambda row: (timestamp(row['case_start_time']), row['case_id']))
    a, b = 3 * len(ordered) // 5, 4 * len(ordered) // 5
    expected = {row['case_id']: split for split, group in (
        ('train', ordered[:a]), ('val', ordered[a:b]), ('test', ordered[b:])) for row in group}
    checks['temporal_split_exact'] = all(row['split'] == expected[row['case_id']] for row in labels)
    groups = {split: {row['case_id'] for row in labels if row['split'] == split}
              for split in ('train', 'val', 'test')}
    checks['split_case_ids_disjoint'] = not (groups['train'] & groups['val'] or groups['train'] & groups['test']
                                             or groups['val'] & groups['test'])
    checks['split_summary_counts_match'] = all(
        int(row['cases']) == len(groups[row['split']])
        and int(row['positives']) == sum(int(label_by_id[c]['label']) for c in groups[row['split']])
        for row in splits)

    coverage_exact = True
    for row in coverage:
        k = int(row['endpoint'])
        eligible = [label for label in labels if k < int(label['label_event_index'])]
        coverage_exact &= int(row['total_eligible_cases']) == len(eligible)
        coverage_exact &= int(row['excluded_by_label_event']) == len(labels) - len(eligible)
        coverage_exact &= int(row['excluded_unlabeled_cases']) == len(excluded)
        for split in ('train', 'val', 'test'):
            group = [label for label in eligible if label['split'] == split]
            positives = sum(int(label['label']) for label in group)
            coverage_exact &= int(row[f'{split}_cases']) == len(group)
            coverage_exact &= int(row[f'{split}_positive']) == positives
            coverage_exact &= int(row[f'{split}_negative']) == len(group)-positives
    checks['prefix_coverage_exact'] = coverage_exact
    checks['prefix_boundary_strict'] = all(
        int(row['endpoint']) < int(label['label_event_index'])
        for row in coverage for label in labels
        if int(label['label_event_index']) > int(row['endpoint']))

    activity_audit = read_csv(ROOT / 'results/phase1_1/audit/prefix_outcome_activity_audit.csv')
    checks['historical_terminal_activity_audited'] = len(activity_audit) == len(CANDIDATES) * 4 * 6
    checks['application_proxies_audited'] = all(
        any(r['endpoint'] == str(k) and r['split'] == split and r['activity'] == activity
            for r in activity_audit)
        for k in CANDIDATES for split in ('all', 'train', 'val', 'test')
        for activity in ('A_Pending', 'A_Denied', 'A_Cancelled'))
    required = [
        'results/phase1_1/audit/case_labels.csv', 'results/phase1_1/stats/split_summary.csv',
        'results/phase1_1/stats/prefix_coverage_labeled.csv', 'results/phase1_1/audit/label_definition.md',
        'results/phase1_1/audit/leakage_audit_final_v2.md', 'results/phase1_1/PHASE1_1_REPORT.md']
    checks['required_outputs_exist'] = all((ROOT / path).is_file() for path in required)

    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'src')
    test_run = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                              cwd=ROOT, env=env, text=True, capture_output=True)
    checks['tests_passed'] = test_run.returncode == 0
    passed = all(checks.values())
    result = {'status': 'passed' if passed else 'failed', 'scope': 'ETCE Phase 1.1',
              'checks': checks, 'tests': {'returncode': test_run.returncode,
              'output': (test_run.stdout + test_run.stderr).strip()}}
    for name in ('validation.json', 'validation_phase1_1.json'):
        write_json(ROOT / 'results/phase1_1/logs' / name, result)

    report_path = ROOT / 'results/phase1_1/PHASE1_1_REPORT.md'
    report = report_path.read_text(encoding='utf-8')
    replacement = (f'验证状态：**{result["status"]}**。27 项自动化测试通过；'
                   f'{len(checks)} 项真实数据标签、split、prefix 边界和审计完整性检查通过。'
                   '详情见 `results/phase1_1/logs/validation_phase1_1.json`。')
    updated, count = re.subn(r'(?s)<!-- VALIDATION_START -->.*?<!-- VALIDATION_END -->',
                             '<!-- VALIDATION_START -->\n' + replacement + '\n<!-- VALIDATION_END -->', report)
    if count != 1:
        raise RuntimeError('PHASE1_1_REPORT validation markers missing or duplicated')
    report_path.write_text(updated, encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
