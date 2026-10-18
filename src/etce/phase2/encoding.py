"""Train-only encoders for tabular and sequential Phase 2 representations."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from .data import PREFIXES


@dataclass(frozen=True)
class EncodedSplit:
    case_ids: np.ndarray
    labels: np.ndarray
    tabular: np.ndarray
    activity_sequence: np.ndarray
    time_sequence: np.ndarray


def _seconds(delta):
    return float(delta.total_seconds())


def fit_encode_prefix(cases, prefix_k, artifact_dir):
    if prefix_k not in PREFIXES:
        raise ValueError('Unapproved prefix')
    train = [case for case in cases if case.split == 'train']
    vocabulary = tuple(sorted({activity for case in train for activity in case.activities[:prefix_k]}))
    activity_to_id = {activity: index + 1 for index, activity in enumerate(vocabulary)}
    feature_names = []
    for kind in ('count', 'frequency', 'last', 'first'):
        feature_names.extend(f'{kind}::{activity}' for activity in ('UNK',) + vocabulary)
    feature_names.extend(('elapsed_seconds', 'mean_delta_seconds', 'std_delta_seconds',
                          'max_delta_seconds', 'last_delta_seconds', 'last_hour', 'last_weekday'))

    def raw(case):
        activities = case.activities[:prefix_k]
        times = case.timestamps[:prefix_k]
        ids = np.asarray([activity_to_id.get(activity, 0) for activity in activities], dtype=np.int64)
        size = len(vocabulary) + 1
        counts = np.bincount(ids, minlength=size).astype(np.float64)
        frequency = counts / prefix_k
        last = np.zeros(size, dtype=np.float64); last[ids[-1]] = 1.0
        first = np.zeros(size, dtype=np.float64); first[ids[0]] = 1.0
        deltas = np.asarray([0.0] + [_seconds(b-a) for a, b in zip(times, times[1:])], dtype=np.float64)
        elapsed = np.asarray([_seconds(value-times[0]) for value in times], dtype=np.float64)
        intervals = deltas[1:]
        temporal = np.asarray([elapsed[-1], intervals.mean(), intervals.std(), intervals.max(),
                               deltas[-1], times[-1].hour, times[-1].weekday()], dtype=np.float64)
        tabular = np.concatenate((counts, frequency, last, first, temporal))
        sequence_time = np.column_stack((deltas, elapsed))
        return tabular, ids, sequence_time

    raw_by_id = {case.case_id: raw(case) for case in cases}
    train_ids = sorted(case.case_id for case in train)
    tabular_scaler = StandardScaler().fit(np.vstack([raw_by_id[key][0] for key in train_ids]))
    seq_train = np.vstack([raw_by_id[key][2] for key in train_ids])
    time_scaler = StandardScaler().fit(seq_train)
    output = {}
    for split in ('train', 'val', 'test'):
        group = sorted((case for case in cases if case.split == split), key=lambda item: item.case_id)
        output[split] = EncodedSplit(
            np.asarray([case.case_id for case in group], dtype=str),
            np.asarray([case.label for case in group], dtype=np.int64),
            tabular_scaler.transform(np.vstack([raw_by_id[case.case_id][0] for case in group])).astype(np.float32),
            np.stack([raw_by_id[case.case_id][1] for case in group]).astype(np.int64),
            time_scaler.transform(np.vstack([raw_by_id[case.case_id][2] for case in group]))
                .reshape(len(group), prefix_k, 2).astype(np.float32))
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'prefix': prefix_k, 'unknown_activity_id': 0, 'vocabulary': list(vocabulary),
                'activity_to_id': activity_to_id, 'feature_names': feature_names,
                'tabular_feature_count': len(feature_names), 'sequence_time_features': ['delta_seconds', 'elapsed_seconds'],
                'train_case_ids_used_for_fit': train_ids,
                'train_case_id_count': len(train_ids), 'fit_splits': ['train']}
    (artifact_dir / f'encoder_prefix{prefix_k}.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    joblib.dump(tabular_scaler, artifact_dir / f'tabular_scaler_prefix{prefix_k}.joblib')
    joblib.dump(time_scaler, artifact_dir / f'sequence_time_scaler_prefix{prefix_k}.joblib')
    return output, metadata
