"""Candidate endpoints only; there is deliberately no endpoint selector."""
from dataclasses import dataclass
from .load_bpic2017 import Case

CANDIDATES = (5, 10, 15, 20, 25, 30)


@dataclass(frozen=True)
class Prefix:
    case_id: str
    split: str
    endpoint: int
    events: tuple[dict, ...]
    case_attributes: dict


def generate_prefixes(case: Case, split, label_event_index):
    if split not in {'train', 'val', 'test'}:
        raise ValueError('Invalid split')
    if not isinstance(label_event_index, int) or not 1 <= label_event_index <= len(case.events):
        raise ValueError('label_event_index must be a valid 1-based original event index')
    for k in CANDIDATES:
        if k < label_event_index:
            yield Prefix(case.case_id, split, k, case.events[:k], case.attributes)


def coverage(cases, assignments, labels):
    """Labeled coverage under strict k < label_event_index."""
    cases = list(cases)
    if not cases:
        raise ValueError('No audited cases')
    rows = []
    for k in CANDIDATES:
        eligible = [c for c in cases if k < c['label_event_index']]
        row = {'endpoint': k, 'total_eligible_cases': len(eligible),
               'eligible_rate': len(eligible) / len(cases),
               'excluded_by_label_event': len(cases)-len(eligible)}
        for split in ('train', 'val', 'test'):
            group = [c for c in eligible if assignments[c['case_id']] == split]
            positives = sum(labels[c['case_id']] for c in group)
            row.update({f'{split}_cases': len(group), f'{split}_positive': positives,
                        f'{split}_negative': len(group)-positives,
                        f'{split}_positive_rate': positives / len(group) if group else None})
        rows.append(row)
    return rows
