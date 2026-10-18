import unittest
from etce.data.labels import derive_benchmark_label
from etce.data.load_bpic2017 import Case


def event(activity, second, origin='Offer'):
    return {'concept:name': activity, 'EventOrigin': origin,
            'time:timestamp': f'2016-01-01T00:00:{second:02d}Z'}


class BenchmarkLabelTests(unittest.TestCase):
    def test_last_offer_terminal_is_label(self):
        label, exclusion = derive_benchmark_label(Case('c', {}, (
            event('O_Cancelled', 1), event('O_Accepted', 2))))
        self.assertIsNone(exclusion)
        self.assertEqual((label.label, label.label_event_index), (1, 2))

    def test_last_nonterminal_offer_excludes_even_after_earlier_terminal(self):
        label, exclusion = derive_benchmark_label(Case('c', {}, (
            event('O_Accepted', 1), event('O_Returned', 2))))
        self.assertIsNone(label)
        self.assertEqual(exclusion['reason'], 'last_offer_event_not_terminal')

    def test_non_offer_terminal_name_is_ignored(self):
        label, exclusion = derive_benchmark_label(Case('c', {}, (
            event('O_Accepted', 2, 'Workflow'), event('O_Cancelled', 1))))
        self.assertIsNone(exclusion)
        self.assertEqual(label.label, 0)

    def test_same_timestamp_uses_original_trace_order(self):
        label, _ = derive_benchmark_label(Case('c', {}, (
            event('O_Accepted', 1), event('O_Refused', 1))))
        self.assertEqual((label.label, label.label_event_index), (0, 2))
