"""Raw-log observations, not model labels. Field mappings observed in raw XES.

Offer rows link O_Create Offer.EventID to subsequent OfferID, and compare
Accepted against O_Accepted. Both alternatives remain visible for audit.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
from .load_bpic2017 import read_cases, timestamp
from etce.utils.io import sha256, write_csv, write_json, write_jsonl


def audit(root, config):
    raw = root / config['raw_path']
    if not raw.is_file():
        raise FileNotFoundError(f'BPIC2017 raw log missing: {raw}')
    out = root / 'results/phase1/audit'
    out.mkdir(parents=True, exist_ok=True)
    inventories = {'case': {}, 'event': {}}
    activities, field_activity_values = Counter(), Counter()
    cases, offers, anomalies = [], [], []
    seen = set()
    total_events = 0
    first_time = last_time = None

    def inventory(scope, attrs):
        for key, value in attrs.items():
            entry = inventories[scope].setdefault(key, {'count': 0, 'types': set(), 'examples': []})
            entry['count'] += 1
            entry['types'].add(type(value).__name__)
            serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
            if len(entry['examples']) < 5 and serialized not in entry['examples']:
                entry['examples'].append(serialized)

    for case in read_cases(raw):
        if case.case_id in seen:
            anomalies.append({'case_id': case.case_id, 'issue': 'duplicate case id'})
        seen.add(case.case_id)
        inventory('case', case.attributes)
        if not case.events:
            anomalies.append({'case_id': case.case_id, 'issue': 'empty trace'})
            continue
        times = []
        offer_created, offer_states = {}, defaultdict(list)
        accepted_events, flags, terminal_events = [], [], []
        for index, event in enumerate(case.events, 1):
            inventory('event', event)
            activity = event.get('concept:name')
            activities[activity] += 1
            time = timestamp(event['time:timestamp'])
            times.append(time)
            for key in ('Accepted', 'Selected'):
                if key in event:
                    field_activity_values[(key, activity, str(event[key]))] += 1
            if 'Accepted' in event:
                flags.append({'index': index, 'timestamp': event['time:timestamp'],
                              'activity': activity, 'accepted': event['Accepted'],
                              'selected': event.get('Selected'), 'offer_id': event.get('EventID')})
            if activity == 'O_Create Offer':
                offer_id = event.get('EventID')
                if offer_id in offer_created:
                    anomalies.append({'case_id': case.case_id, 'issue': 'duplicate offer creation id'})
                offer_created[offer_id] = (index, event)
            if 'OfferID' in event:
                offer_states[event['OfferID']].append((index, event))
            if activity == 'O_Accepted':
                accepted_events.append({'index': index, 'timestamp': event['time:timestamp'],
                                        'offer_id': event.get('OfferID')})
            if activity in {'A_Pending', 'A_Cancelled', 'A_Denied', 'O_Accepted', 'O_Cancelled', 'O_Refused'}:
                terminal_events.append({'index': index, 'timestamp': event['time:timestamp'], 'activity': activity})
        reversal = sum(a > b for a, b in zip(times, times[1:]))
        if reversal:
            anomalies.append({'case_id': case.case_id, 'issue': f'{reversal} timestamp reversals'})
        for offer_id in sorted(set(offer_created) | set(offer_states), key=str):
            creation = offer_created.get(offer_id)
            states = offer_states.get(offer_id, [])
            accepted = [(i, e) for i, e in states if e['concept:name'] == 'O_Accepted']
            i, event = creation if creation else (None, {})
            offers.append({'case_id': case.case_id, 'offer_id': offer_id,
                           'creation_index': i, 'creation_timestamp': event.get('time:timestamp'),
                           'Accepted': event.get('Accepted'), 'Selected': event.get('Selected'),
                           'accepted_state_count': len(accepted),
                           'first_accepted_index': accepted[0][0] if accepted else None,
                           'first_accepted_timestamp': accepted[0][1]['time:timestamp'] if accepted else None,
                           'states': '|'.join(e['concept:name'] for _, e in states),
                           'flag_state_agree': event.get('Accepted') == bool(accepted) if creation else False})
        total_events += len(case.events)
        first_time = min(first_time, min(times)) if first_time else min(times)
        last_time = max(last_time, max(times)) if last_time else max(times)
        cases.append({'case_id': case.case_id, 'events': len(case.events),
                      'start_time': min(times).isoformat(), 'end_time': max(times).isoformat(),
                      'offer_count': len(offer_created), 'accepted_flag_count': sum(f['accepted'] is True for f in flags),
                      'rejected_flag_count': sum(f['accepted'] is False for f in flags),
                      'selected_count': sum(f['selected'] is True for f in flags),
                      'accepted_state_count': len(accepted_events), 'flags': flags,
                      'accepted_events': accepted_events, 'terminal_events': terminal_events})
        if len(cases) % 5000 == 0:
            print(f'Audited {len(cases)} cases / {total_events} events', flush=True)

    lengths = [c['events'] for c in cases]
    summary = {'raw_path': config['raw_path'], 'format': '.xes.gz' if raw.suffix == '.gz' else '.xes',
               'raw_sha256': sha256(raw), 'raw_bytes': raw.stat().st_size,
               'cases': len(cases), 'events': total_events,
               'event_time_min': first_time.isoformat() if first_time else None,
               'event_time_max': last_time.isoformat() if last_time else None,
               'activities': len(activities), 'offers': len(offers),
               'cases_with_multiple_offers': sum(c['offer_count'] > 1 for c in cases),
               'cases_without_offer': sum(c['offer_count'] == 0 for c in cases),
               'cases_with_mixed_accepted_flags': sum(c['accepted_flag_count'] > 0 and c['rejected_flag_count'] > 0 for c in cases),
               'case_flag_state_disagreements': sum(bool(c['accepted_flag_count']) != bool(c['accepted_state_count']) for c in cases),
               'offer_flag_state_disagreements': sum(not o['flag_state_agree'] for o in offers),
               'offers_without_creation': sum(o['creation_index'] is None for o in offers),
               'accepted_flags_without_boolean': sum(type(o['Accepted']) is not bool for o in offers),
               'anomaly_count': len(anomalies)}
    write_json(out / 'dataset_summary.json', summary)
    for scope, fields in inventories.items():
        denominator = len(cases) if scope == 'case' else total_events
        write_csv(out / f'{scope}_attributes.csv', [
            {'attribute': key, 'count': entry['count'], 'missing': denominator-entry['count'],
             'types': '|'.join(sorted(entry['types'])), 'examples': ' | '.join(entry['examples'])}
            for key, entry in sorted(fields.items())])
    write_csv(out / 'activity_counts.csv', [{'activity': key, 'count': count} for key, count in sorted(activities.items())])
    write_csv(out / 'trace_length_summary.csv', [{'cases': len(cases), 'min': min(lengths),
              'median': statistics.median(lengths), 'mean': statistics.mean(lengths), 'max': max(lengths)}])
    write_csv(out / 'candidate_outcome_fields.csv', [{'field': key, 'activity': activity, 'value': value, 'count': count}
              for (key, activity, value), count in sorted(field_activity_values.items())])
    write_csv(out / 'offer_outcome_audit.csv', offers)
    write_csv(out / 'raw_anomalies.csv', anomalies, ['case_id', 'issue'])
    write_jsonl(root / 'data/interim/phase1/bpic2017_case_audit.jsonl', cases)
    return summary
