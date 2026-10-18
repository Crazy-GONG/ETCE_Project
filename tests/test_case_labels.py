import unittest
from etce.data.labels import derive_case_label
from etce.data.load_bpic2017 import Case


def event(activity, second, origin='Offer', **extra):
    return {'concept:name': activity, 'EventOrigin': origin,
            'time:timestamp': f'2016-01-01T00:00:{second:02d}Z', **extra}


class CaseLabelTests(unittest.TestCase):
    def test_last_chronological_terminal_defines_label_and_boundary(self):
        case = Case('c', {}, (event('O_Accepted', 2), event('O_Cancelled', 1), event('O_Refused', 3)))
        label, exclusion = derive_case_label(case)
        self.assertIsNone(exclusion)
        self.assertEqual((label.label, label.label_event_index, label.label_source_value_or_activity),
                         (0, 3, 'O_Refused'))

    def test_accepted_boolean_is_ignored(self):
        case = Case('c', {}, (event('O_Create Offer', 1, Accepted=True), event('O_Cancelled', 2)))
        label, _ = derive_case_label(case)
        self.assertEqual(label.label, 0)

    def test_event_origin_is_required(self):
        label, exclusion = derive_case_label(Case('c', {}, (event('O_Accepted', 1, origin='Workflow'),)))
        self.assertIsNone(label)
        self.assertEqual(exclusion['reason'], 'no_terminal_offer_event')

    def test_conflicting_latest_timestamp_is_excluded(self):
        case = Case('c', {}, (event('O_Accepted', 1), event('O_Refused', 1)))
        label, exclusion = derive_case_label(case)
        self.assertIsNone(label)
        self.assertEqual(exclusion['reason'], 'conflicting_terminal_outcomes_at_latest_timestamp')

    def test_same_outcome_latest_duplicate_has_unique_class_and_last_index(self):
        case = Case('c', {}, (event('O_Accepted', 1), event('O_Accepted', 1)))
        label, exclusion = derive_case_label(case)
        self.assertIsNone(exclusion)
        self.assertEqual((label.label, label.label_event_index), (1, 2))
