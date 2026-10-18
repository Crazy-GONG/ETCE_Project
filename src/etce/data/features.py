"""Untrained sequence/tabular interface; training-only, persistent schema.

Metadata (case, split, endpoint) is separate from predictor dictionaries.
No imputation, scaling, or future backfill. Missing values remain None.
"""
from collections import Counter
from dataclasses import asdict, dataclass
from .leakage import LeakagePolicy, sanitize
from .load_bpic2017 import timestamp
from etce.utils.io import read_json, write_json


@dataclass
class FeatureSchema:
    policy: LeakagePolicy
    activities: tuple[str, ...] = ()
    categories: dict | None = None
    fit_case_ids: tuple[str, ...] = ()
    fitted: bool = False

    def fit(self, prefixes):
        if self.fitted:
            raise ValueError('Schema already fitted; use a new instance')
        self.policy.validate()
        activities, categories, ids = set(), {}, set()
        for prefix in prefixes:
            if prefix.split != 'train':
                raise ValueError('Only training prefixes may fit schema')
            safe = sanitize(prefix, self.policy)
            ids.add(prefix.case_id)
            for event in safe['events']:
                activities.add(event['activity'])
                for key, value in event['attributes'].items():
                    if isinstance(value, str):
                        categories.setdefault('event:' + key, set()).add(value)
            for key, value in safe['static'].items():
                if isinstance(value, str):
                    categories.setdefault('case:' + key, set()).add(value)
        if not ids:
            raise ValueError('Cannot fit empty training schema')
        self.activities = tuple(sorted(activities))
        self.categories = {key: sorted(values) for key, values in sorted(categories.items())}
        self.fit_case_ids = tuple(sorted(ids))
        self.fitted = True
        return self

    def _encode(self, scope, key, value):
        if isinstance(value, str):
            values = self.categories.get(scope + ':' + key, [])
            return values.index(value) + 1 if value in values else 0
        return value

    def transform(self, prefix):
        if not self.fitted:
            raise ValueError('Fit training schema first')
        safe = sanitize(prefix, self.policy)
        if not safe['events']:
            raise ValueError('Empty prefix')
        start = timestamp(safe['events'][0]['timestamp'])
        sequence, latest = [], {}
        counts = Counter()
        for event in safe['events']:
            activity = event['activity'] if event['activity'] in self.activities else None
            counts[activity] += 1
            attrs = {key: self._encode('event', key, event['attributes'].get(key))
                     for key in self.policy.allowed_event_fields}
            # Only observed values can update the tabular last-known state.
            latest.update({key: value for key, value in attrs.items() if key in event['attributes']})
            sequence.append({'activity_index': self.activities.index(activity) + 1 if activity else 0,
                             'timestamp': event['timestamp'],
                             'elapsed_seconds': (timestamp(event['timestamp'])-start).total_seconds(),
                             'attributes': attrs})
        tabular = {'prefix_length': prefix.endpoint, 'last_activity_index': sequence[-1]['activity_index'],
                   'elapsed_seconds': sequence[-1]['elapsed_seconds'],
                   'activity_counts': {str(i+1): counts[a] for i, a in enumerate(self.activities)},
                   'unknown_activity_count': counts[None],
                   'dynamic': {key: latest.get(key) for key in self.policy.allowed_event_fields},
                   'static': {key: self._encode('case', key, safe['static'].get(key))
                              for key in self.policy.allowed_case_fields}}
        return {'metadata': {'case_id': prefix.case_id, 'split': prefix.split, 'endpoint': prefix.endpoint},
                'sequence': sequence, 'tabular': tabular}

    def save(self, path):
        if not self.fitted:
            raise ValueError('Cannot save unfitted schema')
        write_json(path, asdict(self))

    @classmethod
    def load(cls, path):
        data = read_json(path)
        policy = data.pop('policy')
        policy['allowed_event_fields'] = tuple(policy['allowed_event_fields'])
        policy['allowed_case_fields'] = tuple(policy['allowed_case_fields'])
        data['activities'] = tuple(data['activities'])
        data['fit_case_ids'] = tuple(data['fit_case_ids'])
        instance = cls(LeakagePolicy(**policy), **data)
        instance.policy.validate()
        return instance
