import unittest
from etce.data.prefix import CANDIDATES, coverage, generate_prefixes
from helpers import case, summaries
from etce.data.split import temporal_split


class PrefixTests(unittest.TestCase):
    def test_candidates_and_original_events(self):
        original = case()
        prefixes = list(generate_prefixes(original, 'train', 31))
        self.assertEqual(tuple(p.endpoint for p in prefixes), CANDIDATES)
        for prefix in prefixes:
            self.assertEqual(len(prefix.events), prefix.endpoint)
            self.assertEqual(prefix.events, original.events[:prefix.endpoint])

    def test_strict_boundary(self):
        for length in (5, 10, 15, 20, 25, 30):
            self.assertTrue(all(p.endpoint < length for p in generate_prefixes(case(length=length), 'train', length)))
        self.assertEqual(list(generate_prefixes(case(length=5), 'train', 5)), [])

    def test_coverage_conserves_counts(self):
        rows = summaries()
        rows[0]['events'] = 10
        assignments = temporal_split(rows)
        labels = {row['case_id']: int(row['case_id']) % 2 for row in rows}
        for row in coverage(rows, assignments, labels):
            self.assertEqual(row['total_eligible_cases'] + row['excluded_by_label_event'], 10)
            self.assertEqual(sum(row[f'{s}_cases'] for s in ('train', 'val', 'test')), row['total_eligible_cases'])
            for split in ('train', 'val', 'test'):
                self.assertEqual(row[f'{split}_positive'] + row[f'{split}_negative'], row[f'{split}_cases'])
