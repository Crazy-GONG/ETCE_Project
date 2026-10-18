import unittest
from etce.data.split import temporal_split
from helpers import summaries


class TemporalSplitTests(unittest.TestCase):
    def test_order_ratios_and_rounding(self):
        rows = summaries(11)
        split = temporal_split(reversed(rows))
        self.assertEqual([list(split.values()).count(s) for s in ('train', 'val', 'test')], [6, 2, 3])
        self.assertEqual([split[str(i)] for i in range(11)], ['train']*6 + ['val']*2 + ['test']*3)

    def test_start_time_tie_uses_case_id(self):
        rows = summaries()
        for row in rows:
            row['case_start_time'] = '2016-01-01T00:00:00Z'
        self.assertEqual(temporal_split(rows), temporal_split(reversed(rows)))

    def test_duplicate_identifier_rejected(self):
        rows = summaries()
        rows[-1]['case_id'] = rows[0]['case_id']
        with self.assertRaises(ValueError):
            temporal_split(rows)
