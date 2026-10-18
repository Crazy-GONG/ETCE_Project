import unittest
from etce.data.split import temporal_split, validate_no_overlap
from etce.data.prefix import generate_prefixes
from helpers import case, summaries


class NoOverlapTests(unittest.TestCase):
    def test_every_prefix_in_parent_partition(self):
        assignments = temporal_split(summaries())
        groups = {s: set() for s in ('train', 'val', 'test')}
        for identifier, split in assignments.items():
            for prefix in generate_prefixes(case(identifier), split, 31):
                groups[prefix.split].add(prefix.case_id)
                self.assertEqual(prefix.split, assignments[prefix.case_id])
        validate_no_overlap(groups)
        self.assertEqual(len(set.union(*groups.values())), 10)

    def test_overlap_is_error(self):
        with self.assertRaises(ValueError):
            validate_no_overlap({'train': ['same'], 'val': ['same'], 'test': []})
