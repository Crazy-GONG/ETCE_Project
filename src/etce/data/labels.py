"""Auditable Phase 1.1 benchmark-compatible case outcome labels."""
from dataclasses import dataclass
from .load_bpic2017 import timestamp

TERMINAL_OFFER_ACTIVITIES = frozenset({'O_Accepted', 'O_Refused', 'O_Cancelled'})


class LabelAmbiguityError(RuntimeError):
    pass


def evaluate_label_gate(summary):
    reasons = []
    if summary['offer_flag_state_disagreements']:
        reasons.append('Accepted field and O_Accepted activity disagree at offer level')
    if summary['case_flag_state_disagreements']:
        reasons.append('Case-level any Accepted=True differs from any O_Accepted')
    if summary['cases_with_multiple_offers']:
        reasons.append('Multiple offers per case: case-level aggregation is not specified')
    if summary['offers_without_creation'] or summary['accepted_flags_without_boolean']:
        reasons.append('Incomplete offer linkage or non-boolean Accepted field')
    if summary['anomaly_count']:
        reasons.append('Raw structural anomalies require review')
    # Even numerical agreement is not proof of event-time availability.
    reasons.append('Outcome semantics and actual first-known time need documented confirmation')
    return {'status': 'blocked', 'task': 'Offer accepted vs. not accepted',
            'raw_sha256': summary['raw_sha256'], 'reasons': reasons,
            'labels_constructed': False, 'split_constructed': False,
            'prefix_dataset_constructed': False, 'human_review_required': True}


def require_verified_labels(gate):
    if gate.get('status') != 'verified':
        raise LabelAmbiguityError('Label audit blocked. See results/phase1/PHASE1_REPORT.md.')


@dataclass(frozen=True)
class CaseLabel:
    case_id: str
    label: int
    label_source_type: str
    label_source_value_or_activity: str
    label_event_index: int
    label_event_timestamp: str
    case_start_time: str
    trace_length: int
    terminal_offer_event_count: int
    earlier_terminal_offer_event_count: int


def derive_case_label(case):
    """Return (label, exclusion), using original 1-based event indices.

    Chronology is timestamp then original XES index. A conflicting outcome at the
    same latest timestamp has no unique chronological terminal result and is excluded.
    """
    if not case.events:
        return None, {'case_id': case.case_id, 'reason': 'empty_trace', 'candidate_count': 0}
    candidates = []
    for index, event in enumerate(case.events, 1):
        activity = event.get('concept:name')
        if event.get('EventOrigin') == 'Offer' and activity in TERMINAL_OFFER_ACTIVITIES:
            candidates.append((timestamp(event['time:timestamp']), index, activity, event['time:timestamp']))
    if not candidates:
        return None, {'case_id': case.case_id, 'reason': 'no_terminal_offer_event', 'candidate_count': 0}
    latest_time = max(item[0] for item in candidates)
    latest = [item for item in candidates if item[0] == latest_time]
    latest_outcomes = {item[2] for item in latest}
    if len(latest_outcomes) != 1:
        return None, {'case_id': case.case_id, 'reason': 'conflicting_terminal_outcomes_at_latest_timestamp',
                      'candidate_count': len(candidates), 'latest_timestamp': latest[0][3],
                      'latest_activities': '|'.join(sorted(latest_outcomes))}
    # Same-outcome duplicates at one timestamp have an unambiguous class. The final
    # original event is the label boundary and duplicate count remains auditable.
    _, index, activity, raw_time = max(latest, key=lambda item: item[1])
    start = min(timestamp(event['time:timestamp']) for event in case.events)
    label = CaseLabel(case.case_id, int(activity == 'O_Accepted'), 'terminal_offer_activity', activity,
                      index, timestamp(raw_time).isoformat(), start.isoformat(), len(case.events),
                      len(candidates), len(candidates) - len(latest))
    return label, None


def derive_benchmark_label(case):
    """Label from the last chronological EventOrigin=Offer event.

    Unlike derive_case_label, non-terminal last offer activity excludes the case.
    Timestamp ties are resolved by original XES event index, matching trace order.
    """
    if not case.events:
        return None, {'case_id': case.case_id, 'reason': 'empty_trace', 'offer_event_count': 0}
    offers = [(timestamp(event['time:timestamp']), index, event.get('concept:name'),
               event['time:timestamp'])
              for index, event in enumerate(case.events, 1)
              if event.get('EventOrigin') == 'Offer']
    if not offers:
        return None, {'case_id': case.case_id, 'reason': 'no_offer_event', 'offer_event_count': 0}
    _, index, activity, raw_time = max(offers, key=lambda item: (item[0], item[1]))
    if activity not in TERMINAL_OFFER_ACTIVITIES:
        return None, {'case_id': case.case_id, 'reason': 'last_offer_event_not_terminal',
                      'offer_event_count': len(offers), 'last_offer_activity': activity,
                      'last_offer_event_index': index,
                      'last_offer_event_timestamp': timestamp(raw_time).isoformat()}
    start = min(timestamp(event['time:timestamp']) for event in case.events)
    label = CaseLabel(case.case_id, int(activity == 'O_Accepted'), 'benchmark_last_offer_activity',
                      activity, index, timestamp(raw_time).isoformat(), start.isoformat(),
                      len(case.events), len(offers), 0)
    return label, None
