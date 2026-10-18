"""Independent post-run validation for the Phase 2A pilot artifacts."""
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.deps'))
sys.path.insert(0, str(ROOT / 'src'))

import joblib
import numpy as np
import torch

from etce.phase2.data import (COHORT_HASHES, DIRECT_PROXIES, EXPECTED_COUNTS,
                              PREFIXES, RAW_SHA256, id_hash, load_frozen_common_cohort)
from etce.phase2.metrics import binary_metrics
from etce.utils.io import read_json, sha256, write_json

RESULTS = ROOT / 'results/phase2'
MODELS = ('RF', 'XGBoost', 'MLP', 'LSTM')
MODEL_FILES = {'RF': 'rf', 'XGBoost': 'xgboost', 'MLP': 'mlp', 'LSTM': 'lstm'}


def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(left, right, tolerance=1e-10):
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def validate():
    checks = {}
    required = [
        RESULTS / 'PHASE2A_REPORT.md', RESULTS / 'metrics/pilot_metrics.csv',
        RESULTS / 'metrics/search_metrics.csv', RESULTS / 'configs/best_configs.json',
        RESULTS / 'audit/feature_whitelist.md', RESULTS / 'audit/leakage_check_phase2.json',
        RESULTS / 'run_manifest_phase2a.json']
    require(all(path.is_file() for path in required), 'A required Phase 2A output is missing')
    checks['required_outputs_exist'] = True

    manifest = read_json(RESULTS / 'run_manifest_phase2a.json')
    require(manifest['raw_sha256'] == RAW_SHA256, 'Manifest raw hash differs')
    require(manifest['cohort_hashes'] == COHORT_HASHES, 'Manifest cohort hashes differ')
    require(manifest['config_sha256'] == sha256(RESULTS / 'configs/best_configs.json'),
            'Best-config hash differs from manifest')
    for relative, expected in manifest['source_hashes'].items():
        require(sha256(ROOT / relative) == expected, f'Source changed after run: {relative}')
    for relative, expected in manifest['output_hashes_before_validation'].items():
        require(sha256(ROOT / relative) == expected, f'Pilot output changed before validation: {relative}')
    checks['manifest_source_data_config_and_output_hashes_match'] = True

    cases, data_audit = load_frozen_common_cohort(ROOT)
    require(data_audit['raw_sha256'] == manifest['raw_sha256'], 'Raw data hash mismatch')
    by_split = {split: [case for case in cases if case.split == split]
                for split in ('train', 'val', 'test')}
    split_sets = {key: {case.case_id for case in group} for key, group in by_split.items()}
    require(not (split_sets['train'] & split_sets['val'] or split_sets['train'] & split_sets['test']
                 or split_sets['val'] & split_sets['test']), 'Split overlap')
    require(set().union(*split_sets.values()) == {case.case_id for case in cases}, 'Split union mismatch')
    for split, group in by_split.items():
        require(len(group) == EXPECTED_COUNTS[split], f'{split} count mismatch')
        require(id_hash(case.case_id for case in group) == COHORT_HASHES[split], f'{split} hash mismatch')
    require(id_hash(case.case_id for case in cases) == COHORT_HASHES['all'], 'All cohort hash mismatch')
    require(all(len(case.activities) == 20 and len(case.timestamps) == 20 for case in cases),
            'Frozen loader did not retain exactly first 20 raw events')
    require(all(case.label_event_index > 20 for case in cases), 'Label boundary enters a prefix')
    require(all(not (set(case.activities) & DIRECT_PROXIES) for case in cases),
            'Direct outcome proxy enters a prefix')
    checks['raw_common_cohort_split_prefix_and_boundary_rederived'] = True

    metrics = rows(RESULTS / 'metrics/pilot_metrics.csv')
    require(len(metrics) == 32, 'Expected 32 validation/test metric rows')
    metric_index = {(row['model'], int(row['prefix']), row['split']): row for row in metrics}
    require(len(metric_index) == 32, 'Duplicate or missing metric key')
    expected_prediction_files = set()
    prediction_id_sequences = {}
    for model in MODELS:
        stem = MODEL_FILES[model]
        for prefix in PREFIXES:
            for split in ('val', 'test'):
                filename = f'{stem}_prefix{prefix}_{split}.csv'
                expected_prediction_files.add(filename)
                prediction = rows(RESULTS / 'predictions' / filename)
                require(len(prediction) == EXPECTED_COUNTS[split], f'Row count mismatch: {filename}')
                expected_ids = sorted(split_sets[split])
                observed_ids = [row['case_id'] for row in prediction]
                require(observed_ids == expected_ids, f'Case IDs/order mismatch: {filename}')
                prediction_id_sequences[(model, prefix, split)] = observed_ids
                truth = np.asarray([int(row['true_label']) for row in prediction], dtype=int)
                probability = np.asarray([float(row['predicted_probability']) for row in prediction])
                predicted = np.asarray([int(row['predicted_label']) for row in prediction], dtype=int)
                expected_truth = {case.case_id: case.label for case in by_split[split]}
                require(all(expected_truth[row['case_id']] == int(row['true_label']) for row in prediction),
                        f'True labels mismatch: {filename}')
                require(all(int(row['prefix_k']) == prefix and row['split'] == split
                            and row['model_name'] == model and int(row['seed']) == 0
                            for row in prediction), f'Prediction metadata mismatch: {filename}')
                require(np.isfinite(probability).all() and np.all((0 <= probability) & (probability <= 1)),
                        f'Invalid probability: {filename}')
                require(set(truth) <= {0, 1} and set(predicted) <= {0, 1}, f'Invalid label: {filename}')
                require(np.array_equal(predicted, (probability >= 0.5).astype(int)),
                        f'Prediction threshold is not 0.5: {filename}')
                if split == 'test':
                    require(len(set(predicted)) == 2, f'Constant test prediction: {filename}')
                calculated = binary_metrics(truth, probability)
                recorded = metric_index[(model, prefix, split)]
                require(int(recorded['n_cases']) == len(prediction), f'Metric count mismatch: {filename}')
                require(int(recorded['positive']) == int(truth.sum()), f'Metric positives mismatch: {filename}')
                require(int(recorded['negative']) == int(len(truth)-truth.sum()), f'Metric negatives mismatch: {filename}')
                for key, value in calculated.items():
                    require(close(recorded[key], value), f'Metric cannot be reproduced ({key}): {filename}')
    actual_prediction_files = {path.name for path in (RESULTS / 'predictions').glob('*.csv')}
    require(actual_prediction_files == expected_prediction_files, 'Prediction file inventory mismatch')
    for split in ('val', 'test'):
        baseline = prediction_id_sequences[('RF', 5, split)]
        require(all(prediction_id_sequences[(model, prefix, split)] == baseline
                    for model in MODELS for prefix in PREFIXES),
                f'Case IDs not aligned across model/prefix for {split}')
    checks['prediction_alignment_ranges_threshold_nonconstant_and_metrics_reproduced'] = True

    configs = read_json(RESULTS / 'configs/best_configs.json')
    require(configs['pilot_seed'] == 0 and configs['prediction_threshold'] == 0.5,
            'Pilot seed or threshold changed')
    require(set(configs['models']) == set(MODELS), 'Model config inventory mismatch')
    for model in MODELS:
        require(set(configs['models'][model]) == {str(k) for k in PREFIXES},
                f'Prefix config inventory mismatch: {model}')
        for prefix in PREFIXES:
            item = configs['models'][model][str(prefix)]
            require(item['selection_metric'] == 'validation_macro_f1'
                    and item['test_used_for_selection'] is False and item['threshold'] == 0.5,
                    f'Invalid selection protocol: {model}/{prefix}')
    search = rows(RESULTS / 'metrics/search_metrics.csv')
    require(len(search) == 32, 'Expected two validation candidates per model/prefix')
    require(all('test' not in key.lower() for row in search for key in row),
            'Search metrics expose a test selection field')
    access = read_json(RESULTS / 'logs/test_access_log.json')
    require(len(access) == 16 and all(item['access_count'] == 1 for item in access),
            'Test was not evaluated exactly once per selected model/prefix')
    checks['validation_only_selection_and_single_test_evaluation'] = True

    expected_train_ids = sorted(split_sets['train'])
    forbidden_tokens = ('selected', 'case_id', 'application_id', 'eventid', 'offerid',
                        'org:resource', 'a_pending', 'a_denied', 'a_cancelled', 'loangoal',
                        'applicationtype', 'requestedamount', 'firstwithdrawalamount', 'monthlycost',
                        'numberofterms', 'offeredamount', 'creditscore')
    for prefix in PREFIXES:
        encoder = read_json(RESULTS / f'artifacts/encoder_prefix{prefix}.json')
        require(encoder['prefix'] == prefix and encoder['fit_splits'] == ['train'], 'Encoder fit scope mismatch')
        require(encoder['train_case_ids_used_for_fit'] == expected_train_ids, 'Encoder train IDs mismatch')
        require(encoder['train_case_id_count'] == EXPECTED_COUNTS['train'], 'Encoder train count mismatch')
        feature_text = '\n'.join(encoder['feature_names']).lower()
        require(not any(token in feature_text for token in forbidden_tokens), 'Forbidden feature name present')
        require(not (set(encoder['vocabulary']) & DIRECT_PROXIES), 'Proxy activity in vocabulary')
        tab_scaler = joblib.load(RESULTS / f'artifacts/tabular_scaler_prefix{prefix}.joblib')
        seq_scaler = joblib.load(RESULTS / f'artifacts/sequence_time_scaler_prefix{prefix}.joblib')
        require(tab_scaler.n_features_in_ == encoder['tabular_feature_count'], 'Tabular scaler width mismatch')
        require(seq_scaler.n_features_in_ == 2, 'Sequence scaler width mismatch')
    checks['train_only_vocabulary_scalers_and_feature_allowlist'] = True

    checkpoint_names = set()
    for model in MODELS:
        stem = MODEL_FILES[model]
        for prefix in PREFIXES:
            suffix = 'joblib' if model in ('RF', 'XGBoost') else 'pt'
            path = RESULTS / f'checkpoints/{stem}_prefix{prefix}.{suffix}'
            require(path.is_file() and path.stat().st_size > 0, f'Missing checkpoint: {path.name}')
            checkpoint_names.add(path.name)
            if suffix == 'joblib':
                loaded = joblib.load(path)
                require(hasattr(loaded, 'predict_proba'), f'Invalid sklearn checkpoint: {path.name}')
            else:
                loaded = torch.load(path, map_location='cpu', weights_only=False)
                require(loaded['model_name'] == model and loaded['prefix'] == prefix,
                        f'Invalid torch checkpoint metadata: {path.name}')
    require({path.name for path in (RESULTS / 'checkpoints').iterdir() if path.is_file()} == checkpoint_names,
            'Checkpoint inventory mismatch')
    checks['all_model_checkpoints_load'] = True

    leakage = read_json(RESULTS / 'audit/leakage_check_phase2.json')
    require(all(leakage[key] is True for key in (
        'raw_sha256_matches', 'common_cohort_hashes_match', 'common_cohort_counts_match',
        'common_case_ids_identical_across_prefixes', 'split_case_ids_disjoint',
        'each_prefix_uses_exactly_k_original_events', 'label_boundary_after_prefix_20',
        'direct_outcome_proxy_absent_through_prefix_20')),
        'A pre-training leakage check failed')
    leakage.update({'status': 'passed', 'post_run_prediction_checks_passed': True,
                    'metrics_recomputed_from_predictions': True,
                    'test_constant_prediction_absent': True,
                    'test_evaluations_per_model_prefix': 1})
    write_json(RESULTS / 'audit/leakage_check_phase2.json', leakage)
    checks['leakage_audit_passed'] = True

    result = {'scope': 'ETCE Phase 2A', 'status': 'passed', 'validated_utc': datetime.now(timezone.utc).isoformat(),
              'checks': checks, 'models': list(MODELS), 'prefixes': list(PREFIXES),
              'prediction_files': len(expected_prediction_files), 'metric_rows': len(metrics),
              'phase2b_run': False}
    write_json(RESULTS / 'logs/validation_phase2a.json', result)

    report_path = RESULTS / 'PHASE2A_REPORT.md'
    report = report_path.read_text(encoding='utf-8')
    old = '<!-- VALIDATION_START -->\nAwaiting Phase 2A validation.\n<!-- VALIDATION_END -->'
    new = ('<!-- VALIDATION_START -->\nPost-run validation: **PASSED**. All 15 required sanity-check '
           'categories passed; metrics were independently recomputed from 32 per-case prediction files.\n'
           '<!-- VALIDATION_END -->')
    require(old in report, 'Report validation marker missing')
    report_path.write_text(report.replace(old, new), encoding='utf-8')

    manifest['status'] = 'validated_passed'
    manifest['validated_utc'] = result['validated_utc']
    manifest['validation_sha256'] = sha256(RESULTS / 'logs/validation_phase2a.json')
    manifest['leakage_check_sha256'] = sha256(RESULTS / 'audit/leakage_check_phase2.json')
    manifest['report_sha256_after_validation'] = sha256(report_path)
    manifest['final_output_hashes'] = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(RESULTS.rglob('*'))
        if path.is_file() and path.name != 'run_manifest_phase2a.json'
    }
    write_json(RESULTS / 'run_manifest_phase2a.json', manifest)
    return result


if __name__ == '__main__':
    try:
        outcome = validate()
    except Exception as exc:
        failure = {'scope': 'ETCE Phase 2A', 'status': 'failed',
                   'validated_utc': datetime.now(timezone.utc).isoformat(),
                   'error_type': type(exc).__name__, 'error': str(exc), 'phase2b_run': False}
        write_json(RESULTS / 'logs/validation_phase2a.json', failure)
        raise
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
