from datetime import datetime, timedelta, timezone
from etce.data.load_bpic2017 import Case
from etce.data.leakage import LeakagePolicy


def case(identifier='case_0', length=31, day=0):
    start = datetime(2016, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
    events = tuple({'concept:name': 'A' if j % 2 else 'B',
                    'time:timestamp': (start + timedelta(seconds=j)).isoformat(),
                    'value': j, 'category': 'known', 'Accepted': True, 'EventID': f'e_{j}'}
                   for j in range(length))
    return Case(identifier, {'concept:name': identifier, 'static': 9, 'future': 99}, events)


def summaries(n=10):
    return [{'case_id': str(i), 'case_start_time': f'2016-01-{i+1:02d}T00:00:00+00:00',
             'events': 31, 'label_event_index': 31}
            for i in range(n)]


def policy():
    return LeakagePolicy(('value', 'category'), ('static',),
                         {'Accepted': 'outcome', 'Selected': 'outcome', 'EventID': 'ID'},
                         {'A_Pending': 'potential outcome proxy', 'A_Denied': 'potential outcome proxy',
                          'A_Cancelled': 'potential outcome proxy'}, True)
