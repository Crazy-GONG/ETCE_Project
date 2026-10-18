"""Frozen Phase 1.2 common-cohort loader for Phase 2A."""
from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path

from etce.data.load_bpic2017 import read_cases, timestamp
from etce.utils.io import sha256

RAW_SHA256 = '183c5e5189282779c811c78c33ff936351b3dd201165d612211fc220936f8249'
COHORT_HASHES = {
    'all': '6c73c5eae4463837fe202443559362aecadebc9cf478b04153ec352d7767619a',
    'train': 'f974837fba171b2b49d66546676d50f574b548e81d6981b7ae61e9d720c005ff',
    'val': '87e216c86442bb6329f08ff8f7dc43d512c84dc9c6c180792342058a38aae815',
    'test': '36f598a2b8439336c338ce6d85758aebf023ce7eaf9c4a4a6bb76c27bfabb653',
}
EXPECTED_COUNTS = {'all': 26592, 'train': 16121, 'val': 5195, 'test': 5276}
PREFIXES = (5, 10, 15, 20)
DIRECT_PROXIES = frozenset({'A_Pending', 'A_Denied', 'A_Cancelled'})


@dataclass(frozen=True)
class FrozenCase:
    case_id: str
    split: str
    label: int
    label_event_index: int
    activities: tuple[str, ...]
    timestamps: tuple


def id_hash(case_ids):
    payload = '\n'.join(sorted(case_ids)) + '\n'
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def load_frozen_common_cohort(root):
    """Read frozen IDs/labels/splits; extract only activity and timestamp from raw XES."""
    root = Path(root)
    raw = root / 'data/raw/BPI Challenge 2017.xes.gz'
    if sha256(raw) != RAW_SHA256:
        raise ValueError('BPIC2017 raw SHA-256 mismatch')
    rows = _read_csv(root / 'results/phase1_2/audit/common_cohort_case_ids.csv')
    metadata = {row['case_id']: row for row in rows}
    if len(metadata) != len(rows):
        raise ValueError('Duplicate common-cohort case ID')
    for scope in ('all', 'train', 'val', 'test'):
        ids = metadata if scope == 'all' else {key: row for key, row in metadata.items() if row['split'] == scope}
        if len(ids) != EXPECTED_COUNTS[scope] or id_hash(ids) != COHORT_HASHES[scope]:
            raise ValueError(f'Frozen common-cohort mismatch: {scope}')
    cases = []
    seen = set()
    historical = {k: 0 for k in PREFIXES}
    for case in read_cases(raw):
        row = metadata.get(case.case_id)
        if row is None:
            continue
        label_index = int(row['label_event_index'])
        if label_index <= 20 or len(case.events) < 20:
            raise ValueError(f'Invalid frozen prefix boundary: {case.case_id}')
        events = case.events[:20]
        if any(event.get('concept:name') in DIRECT_PROXIES for event in events):
            raise ValueError(f'Direct outcome proxy in frozen common cohort: {case.case_id}')
        if label_index > len(case.events):
            raise ValueError(f'Label boundary outside trace: {case.case_id}')
        activities = tuple(str(event['concept:name']) for event in events)
        times = tuple(timestamp(event['time:timestamp']) for event in events)
        if any(a > b for a, b in zip(times, times[1:])):
            raise ValueError(f'Timestamp reversal: {case.case_id}')
        for k in PREFIXES:
            historical[k] += any(activity in {'O_Accepted', 'O_Refused', 'O_Cancelled'}
                                 for activity in activities[:k])
        cases.append(FrozenCase(case.case_id, row['split'], int(row['label']), label_index,
                                activities, times))
        seen.add(case.case_id)
    if seen != set(metadata):
        raise ValueError('Raw XES does not contain every frozen common-cohort case')
    cases.sort(key=lambda item: item.case_id)
    return cases, {'raw_sha256': RAW_SHA256, 'cohort_hashes': COHORT_HASHES,
                   'counts': EXPECTED_COUNTS, 'historical_terminal_cases': historical}
