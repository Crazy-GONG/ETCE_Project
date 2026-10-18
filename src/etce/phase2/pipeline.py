"""Phase 2A single-seed pilot orchestrator."""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier

from etce.utils.io import sha256, write_csv, write_json
from .data import COHORT_HASHES, EXPECTED_COUNTS, PREFIXES, load_frozen_common_cohort
from .encoding import fit_encode_prefix
from .metrics import binary_metrics
from .models import predict_torch, set_pilot_seed, train_torch_candidate

PILOT_SEED = 0


def _search_spaces(prefix):
    return {
        'RF': [
            {'n_estimators': 150, 'max_depth': None, 'min_samples_split': 2,
             'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': 'balanced',
             'random_state': PILOT_SEED, 'n_jobs': 4},
            {'n_estimators': 200, 'max_depth': 16, 'min_samples_split': 4,
             'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': 'balanced',
             'random_state': PILOT_SEED, 'n_jobs': 4},
        ],
        'XGBoost': [
            {'n_estimators': 120, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.9,
             'colsample_bytree': 0.9, 'min_child_weight': 1, 'reg_alpha': 0.0,
             'reg_lambda': 1.0, 'random_state': PILOT_SEED, 'n_jobs': 4,
             'eval_metric': 'logloss'},
            {'n_estimators': 180, 'max_depth': 3, 'learning_rate': 0.03, 'subsample': 1.0,
             'colsample_bytree': 0.9, 'min_child_weight': 2, 'reg_alpha': 0.1,
             'reg_lambda': 1.0, 'random_state': PILOT_SEED, 'n_jobs': 4,
             'eval_metric': 'logloss'},
        ],
        'MLP': [
            {'hidden_dims': [64, 32], 'activation': 'ReLU', 'dropout': 0.2,
             'optimizer': 'Adam', 'learning_rate': 0.001, 'batch_size': 256,
             'max_epochs': 12, 'early_stopping_patience': 3, 'weight_decay': 0.0001,
             'seed': PILOT_SEED},
            {'hidden_dims': [128, 64], 'activation': 'ReLU', 'dropout': 0.2,
             'optimizer': 'Adam', 'learning_rate': 0.0005, 'batch_size': 256,
             'max_epochs': 12, 'early_stopping_patience': 3, 'weight_decay': 0.0001,
             'seed': PILOT_SEED},
        ],
        'LSTM': [
            {'embedding_dim': 16, 'time_feature_dim': 2, 'hidden_size': 32,
             'num_layers': 1, 'dropout': 0.0, 'bidirectional': False,
             'classifier_head': 'Linear(hidden_size,1)', 'optimizer': 'Adam',
             'learning_rate': 0.001, 'batch_size': 256, 'max_epochs': 10,
             'early_stopping_patience': 3, 'weight_decay': 0.0001, 'seed': PILOT_SEED},
            {'embedding_dim': 24, 'time_feature_dim': 2, 'hidden_size': 64,
             'num_layers': 1, 'dropout': 0.0, 'bidirectional': False,
             'classifier_head': 'Linear(hidden_size,1)', 'optimizer': 'Adam',
             'learning_rate': 0.0005, 'batch_size': 256, 'max_epochs': 10,
             'early_stopping_patience': 3, 'weight_decay': 0.0001, 'seed': PILOT_SEED},
        ],
    }


def _prediction_rows(model, prefix, split, data, probability, config_id):
    predicted = (probability >= 0.5).astype(int)
    return [{'case_id': case_id, 'prefix_k': prefix, 'true_label': int(label),
             'predicted_label': int(pred), 'predicted_probability': float(prob),
             'split': split, 'model_name': model, 'config_id': config_id, 'seed': PILOT_SEED}
            for case_id, label, pred, prob in zip(data.case_ids, data.labels, predicted, probability)]


def _metric_row(model, prefix, split, data, probability, config_id):
    result = {'model': model, 'prefix': prefix, 'split': split, 'n_cases': len(data.labels),
              'positive': int(data.labels.sum()), 'negative': int(len(data.labels)-data.labels.sum()),
              'seed': PILOT_SEED, 'config_id': config_id}
    result.update(binary_metrics(data.labels, probability))
    return result


def run_phase2a(root):
    root = Path(root)
    results = root / 'results/phase2'
    for directory in ('audit', 'artifacts', 'checkpoints', 'configs', 'logs', 'metrics', 'predictions'):
        (results / directory).mkdir(parents=True, exist_ok=True)
    set_pilot_seed(PILOT_SEED)
    started = datetime.now(timezone.utc).isoformat()
    cases, data_audit = load_frozen_common_cohort(root)
    write_feature_whitelist(results / 'audit/feature_whitelist.md', data_audit)
    write_json(results / 'audit/leakage_check_phase2.json', {
        'status': 'pre_training_checks_passed',
        'raw_sha256_matches': True,
        'common_cohort_hashes_match': True,
        'common_cohort_counts_match': True,
        'common_case_ids_identical_across_prefixes': True,
        'split_case_ids_disjoint': True,
        'prefixes': list(PREFIXES),
        'each_prefix_uses_exactly_k_original_events': True,
        'label_boundary_after_prefix_20': True,
        'direct_outcome_proxy_absent_through_prefix_20': True,
        'allowed_raw_event_fields': ['concept:name', 'time:timestamp'],
        'future_attribute_backfill_used': False,
        'encoder_fit_split': 'train',
        'test_used_for_model_selection': False,
        'historical_terminal_cases': data_audit['historical_terminal_cases'],
    })

    try:
        from xgboost import XGBClassifier
        import xgboost
    except ImportError as exc:
        raise RuntimeError('Official xgboost is required; no substitute permitted') from exc

    all_metrics, search_metrics, best_configs, training_logs, test_access = [], [], {}, [], []
    for prefix in PREFIXES:
        encoded, encoder = fit_encode_prefix(cases, prefix, results / 'artifacts')
        train, val, test = encoded['train'], encoded['val'], encoded['test']
        if not (len(train.labels) == EXPECTED_COUNTS['train'] and len(val.labels) == EXPECTED_COUNTS['val']
                and len(test.labels) == EXPECTED_COUNTS['test']):
            raise ValueError('Encoded split count mismatch')
        spaces = _search_spaces(prefix)
        for model_name in ('RF', 'XGBoost', 'MLP', 'LSTM'):
            candidate_results = []
            for index, config in enumerate(spaces[model_name], 1):
                config_id = f'{model_name.lower()}_prefix{prefix}_c{index}'
                candidate_started = time.perf_counter()
                extra = {}
                if model_name == 'RF':
                    model = RandomForestClassifier(**config)
                    model.fit(train.tabular, train.labels)
                    val_probability = model.predict_proba(val.tabular)[:, 1]
                elif model_name == 'XGBoost':
                    model = XGBClassifier(**config)
                    model.fit(train.tabular, train.labels)
                    val_probability = model.predict_proba(val.tabular)[:, 1]
                elif model_name == 'MLP':
                    model, extra = train_torch_candidate('mlp', train, val, config, PILOT_SEED)
                    val_probability = predict_torch(model, 'mlp', val, config['batch_size'])
                else:
                    model, extra = train_torch_candidate('lstm', train, val, config, PILOT_SEED,
                                                         len(encoder['vocabulary']) + 1)
                    val_probability = predict_torch(model, 'lstm', val, config['batch_size'])
                val_metrics = binary_metrics(val.labels, val_probability)
                elapsed = time.perf_counter() - candidate_started
                search_metrics.append({'model': model_name, 'prefix': prefix, 'config_id': config_id,
                                       'validation_macro_f1': val_metrics['macro_f1'],
                                       'validation_accuracy': val_metrics['accuracy'],
                                       'validation_roc_auc': val_metrics['roc_auc'],
                                       'validation_balanced_accuracy': val_metrics['balanced_accuracy'],
                                       'training_seconds': elapsed, 'seed': PILOT_SEED})
                candidate_results.append((val_metrics['macro_f1'], config_id, deepcopy_config(config),
                                          model, np.asarray(val_probability), extra, elapsed))
                print(f'{model_name} prefix={prefix} candidate={index} '
                      f'val_macro_f1={val_metrics["macro_f1"]:.4f} seconds={elapsed:.1f}', flush=True)
            candidate_results.sort(key=lambda item: (-item[0], item[1]))
            _, config_id, best_config, best_model, val_probability, extra, elapsed = candidate_results[0]
            # Test is accessed exactly once, after validation-only selection.
            if model_name in ('RF', 'XGBoost'):
                test_probability = best_model.predict_proba(test.tabular)[:, 1]
                joblib.dump(best_model, results / 'checkpoints' / f'{model_name.lower()}_prefix{prefix}.joblib')
            else:
                kind = model_name.lower()
                test_probability = predict_torch(best_model, kind, test, best_config['batch_size'])
                torch.save({'model_name': model_name, 'prefix': prefix, 'config': best_config,
                            'state_dict': best_model.state_dict(), 'encoder_vocabulary_size': len(encoder['vocabulary']) + 1,
                            'seed': PILOT_SEED, 'training': extra},
                           results / 'checkpoints' / f'{kind}_prefix{prefix}.pt')
            test_access.append({'model': model_name, 'prefix': prefix, 'config_id': config_id,
                                'access_count': 1, 'purpose': 'single final evaluation after validation selection'})
            for split, data, probability in (('val', val, val_probability), ('test', test, test_probability)):
                filename = f'{model_name.lower()}_prefix{prefix}_{split}.csv'
                write_csv(results / 'predictions' / filename,
                          _prediction_rows(model_name, prefix, split, data, probability, config_id))
                all_metrics.append(_metric_row(model_name, prefix, split, data, probability, config_id))
            best_configs.setdefault(model_name, {})[str(prefix)] = {
                'config_id': config_id, 'hyperparameters': best_config,
                'selection_metric': 'validation_macro_f1',
                'selection_value': binary_metrics(val.labels, val_probability)['macro_f1'],
                'threshold': 0.5, 'test_used_for_selection': False,
                'training_metadata': extra}
            training_logs.append({'model': model_name, 'prefix': prefix, 'selected_config_id': config_id,
                                  'candidate_count': len(candidate_results), 'selected_training_seconds': elapsed,
                                  'test_evaluations': 1})
    write_csv(results / 'metrics/pilot_metrics.csv', all_metrics)
    write_csv(results / 'metrics/search_metrics.csv', search_metrics)
    write_json(results / 'configs/best_configs.json', {
        'pilot_seed': PILOT_SEED, 'seed_role': 'software/protocol pilot only',
        'primary_validation_metric': 'macro_f1', 'prediction_threshold': 0.5,
        'models': best_configs})
    write_json(results / 'logs/training_log.json', training_logs)
    write_json(results / 'logs/test_access_log.json', test_access)
    write_json(results / 'audit/data_pipeline_audit.json', data_audit)
    write_report(results / 'PHASE2A_REPORT.md', all_metrics, best_configs, data_audit, started)
    return build_manifest(root, results, data_audit, xgboost.__version__, started)


def deepcopy_config(config):
    return json.loads(json.dumps(config))


def write_feature_whitelist(path, data_audit):
    text = f'''# Phase 2A Feature Whitelist

## Allowed predictor information

- `concept:name`: activity identity observed inside the exact Prefix-k.
- Original event order inside Prefix-k.
- `time:timestamp` only to derive prefix-local: delta from previous event, time since case start, elapsed time, inter-event summary, last hour and weekday.

Only these two raw event fields are read by the Phase 2 encoder. Vocabulary and scalers are fitted from train cases only. Prefix length is metadata and is not inserted as a predictor.

## Permanently excluded

- `Accepted`, `Selected`.
- case/application IDs, `EventID`, `OfferID`, `org:resource`.
- label boundary event and every later event.
- `A_Pending`, `A_Denied`, `A_Cancelled`; the frozen common cohort already excludes them through Prefix-20.
- every unlisted event attribute.
- `LoanGoal`, `ApplicationType`, `RequestedAmount`, `FirstWithdrawalAmount`, `MonthlyCost`, `NumberOfTerms`, `OfferedAmount`, `CreditScore`.

No future attribute is backfilled from a full trace. Unknown validation/test activities map to `UNK`; they do not expand the train vocabulary.

## Historical offer outcomes

Earlier `O_Accepted/O_Refused/O_Cancelled` activities remain as observed process history before the final label boundary. Cases containing at least one such activity by prefix are: {data_audit['historical_terminal_cases']}.
'''
    Path(path).write_text(text, encoding='utf-8')


def write_report(path, metrics, configs, data_audit, started):
    test = [row for row in metrics if row['split'] == 'test']
    lines = '\n'.join(f"| {r['model']} | {r['prefix']} | {r['n_cases']} | {r['accuracy']:.4f} | "
                      f"{r['macro_f1']:.4f} | {r['roc_auc']:.4f} | {r['balanced_accuracy']:.4f} |"
                      for r in test)
    text = f'''# ETCE Phase 2A Report

> Phase 2A is a software and protocol pilot. Its results are not used as final paper statistics.

## Frozen protocol

- BPIC2017 SHA-256: `{data_audit['raw_sha256']}`.
- Common cohort: train={EXPECTED_COUNTS['train']}, validation={EXPECTED_COUNTS['val']}, test={EXPECTED_COUNTS['test']}.
- Prefixes: 5/10/15/20 with identical case IDs at every prefix.
- Pilot seed: {PILOT_SEED}, used only for software validation.
- Selection: train fit, validation Macro-F1 selection, fixed threshold 0.5, test evaluated once after selection.

## Test pilot metrics

| model | prefix | cases | accuracy | macro-F1 | ROC-AUC | balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
{lines}

Validation and test metrics are stored in `metrics/pilot_metrics.csv`; candidate validation results are in `metrics/search_metrics.csv`. Per-case probabilities are stored under `predictions/`.

## Inputs and leakage

RF, XGBoost and MLP use the exact same scaled tabular matrices. LSTM uses train-only activity vocabulary plus two scaled time-sequence features: delta time and elapsed time. Only activity and timestamp are used. Historical terminal offer activities are retained; direct application outcome proxies and all prohibited fields are absent.

## Scope and stopping point

The pilot started at {started}. It trains only RF, official XGBoost, MLP and unidirectional LSTM. No counterfactual, surrogate, GA, BPA, D-S, process-validity filter or ETCE implementation is included.

<!-- VALIDATION_START -->
Awaiting Phase 2A validation.
<!-- VALIDATION_END -->

Phase 2B was not run. Human review is required before any multi-seed experiment.
'''
    Path(path).write_text(text, encoding='utf-8')


def build_manifest(root, results, data_audit, xgboost_version, started):
    source_files = [path for path in (root / 'src').rglob('*.py')]
    source_files += [path for path in (root / 'scripts').glob('*.py')]
    source_files += [root / 'CODEX_PHASE2_TASK.md', root / 'ETCE_EXPERIMENT_PLAN_v3.md']
    output_files = [path for path in results.rglob('*') if path.is_file()
                    and path.name != 'run_manifest_phase2a.json']
    config_path = results / 'configs/best_configs.json'
    manifest = {'phase': '2A', 'status': 'pilot_pending_validation', 'started_utc': started,
                'completed_utc': datetime.now(timezone.utc).isoformat(), 'pilot_seed': PILOT_SEED,
                'raw_sha256': data_audit['raw_sha256'], 'cohort_hashes': COHORT_HASHES,
                'python': platform.python_version(), 'platform': platform.platform(),
                'versions': {'numpy': np.__version__, 'sklearn': sklearn.__version__,
                             'torch': torch.__version__, 'xgboost': xgboost_version,
                             'joblib': joblib.__version__},
                'config_sha256': sha256(config_path),
                'source_hashes': {path.relative_to(root).as_posix(): sha256(path) for path in sorted(source_files)},
                'output_hashes_before_validation': {path.relative_to(root).as_posix(): sha256(path)
                                                    for path in sorted(output_files)},
                'test_evaluations': 16, 'phase2b_run': False, 'paper_statistics': False}
    write_json(results / 'run_manifest_phase2a.json', manifest)
    return manifest
