from pathlib import Path
import random
import tempfile
import unittest
from etce.data.features import FeatureSchema
from etce.data.prefix import generate_prefixes
from etce.data.split import temporal_split
from etce.utils.io import sha256, write_jsonl
from etce.utils.seed import set_seed
from helpers import case, policy, summaries


class DeterminismTests(unittest.TestCase):
    def test_seed_and_byte_identical_processed_views(self):
        hashes = []
        with tempfile.TemporaryDirectory() as tmp:
            for run in range(2):
                set_seed(0)
                rows = summaries()
                random.shuffle(rows)
                assignments = temporal_split(rows)
                schema = FeatureSchema(policy()).fit(generate_prefixes(case(), 'train', 31))
                schema.save(Path(tmp)/f'schema{run}.json')
                reloaded = FeatureSchema.load(Path(tmp)/f'schema{run}.json')
                views = [reloaded.transform(p) for p in generate_prefixes(case(), 'test', 31)]
                self.assertEqual(views, [schema.transform(p) for p in generate_prefixes(case(), 'test', 31)])
                path = Path(tmp)/f'views{run}.jsonl'
                write_jsonl(path, [{'assignments': assignments, 'views': views}])
                hashes.append(sha256(path))
            self.assertEqual(*hashes)
            self.assertEqual(sha256(Path(tmp)/'schema0.json'), sha256(Path(tmp)/'schema1.json'))

    def test_unknown_category_does_not_expand_schema(self):
        schema = FeatureSchema(policy()).fit(generate_prefixes(case(), 'train', 31))
        test_case = case()
        test_case.events[0]['category'] = 'test_only_value'
        before = dict(schema.categories)
        view = schema.transform(next(generate_prefixes(test_case, 'test', 31)))
        self.assertEqual(view['sequence'][0]['attributes']['category'], 0)
        self.assertEqual(schema.categories, before)
