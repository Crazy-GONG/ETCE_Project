import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from etce.phase2.data import FrozenCase
from etce.phase2.encoding import fit_encode_prefix
from etce.phase2.metrics import binary_metrics


def sample(case_id, split, label, activities):
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return FrozenCase(case_id, split, label, 21, tuple(activities),
                      tuple(start + timedelta(minutes=i) for i in range(len(activities))))


class Phase2EncodingTests(unittest.TestCase):
    def test_fit_is_train_only_and_unknown_maps_to_zero(self):
        base = ['A_Create Application', 'A_Submitted', 'W_Handle leads', 'O_Create Offer',
                'O_Created']
        cases = [sample('train-1', 'train', 1, base),
                 sample('train-2', 'train', 0, list(reversed(base))),
                 sample('val-1', 'val', 1, base[:-1] + ['VAL_ONLY']),
                 sample('test-1', 'test', 0, base[:-1] + ['TEST_ONLY'])]
        with tempfile.TemporaryDirectory() as directory:
            encoded, metadata = fit_encode_prefix(cases, 5, Path(directory))
        self.assertNotIn('VAL_ONLY', metadata['vocabulary'])
        self.assertNotIn('TEST_ONLY', metadata['vocabulary'])
        self.assertEqual(metadata['fit_splits'], ['train'])
        self.assertEqual(metadata['train_case_ids_used_for_fit'], ['train-1', 'train-2'])
        self.assertEqual(encoded['val'].activity_sequence[0, -1], 0)
        self.assertEqual(encoded['test'].activity_sequence[0, -1], 0)
        self.assertEqual(encoded['train'].activity_sequence.shape, (2, 5))
        self.assertTrue(np.isfinite(encoded['train'].tabular).all())
        self.assertTrue(np.isfinite(encoded['train'].time_sequence).all())

    def test_metrics_use_fixed_half_threshold(self):
        result = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
        self.assertEqual(result['accuracy'], 1.0)
        self.assertEqual(result['macro_f1'], 1.0)
        self.assertEqual(result['balanced_accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main()
