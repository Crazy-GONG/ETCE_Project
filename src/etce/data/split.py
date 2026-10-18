"""Case-level temporal partition, v3 §2.1 / task §4."""
from .load_bpic2017 import timestamp


def temporal_split(cases):
    cases = list(cases)
    if len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case ID')
    ordered = sorted(cases, key=lambda c: (timestamp(c['case_start_time']), c['case_id']))
    a, b = 3 * len(ordered) // 5, 4 * len(ordered) // 5
    if not 0 < a < b < len(ordered):
        raise ValueError('Insufficient cases for three nonempty partitions')
    result = {c['case_id']: split for split, group in (
        ('train', ordered[:a]), ('val', ordered[a:b]), ('test', ordered[b:])) for c in group}
    validate_no_overlap({s: [key for key, value in result.items() if value == s]
                         for s in ('train', 'val', 'test')})
    return result


def validate_no_overlap(groups):
    seen = set()
    for name in ('train', 'val', 'test'):
        ids = list(groups[name])
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError('Duplicate or overlapping case IDs')
        seen.update(ids)
