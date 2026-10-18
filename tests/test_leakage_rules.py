from dataclasses import replace
import unittest
from etce.data.prefix import generate_prefixes
from etce.data.features import FeatureSchema
from etce.data.leakage import sanitize
from etce.data.labels import require_verified_labels, LabelAmbiguityError
from helpers import case, policy


class LeakageTests(unittest.TestCase):
    def test_identifiers_outcome_and_future_do_not_enter_predictors(self):
        original = case()
        prefixes = list(generate_prefixes(original, 'train', 31))
        schema = FeatureSchema(policy()).fit(prefixes)
        view = schema.transform(prefixes[0])
        self.assertEqual(view['tabular']['dynamic']['value'], 4)
        self.assertEqual(view['tabular']['static'], {'static': 9})
        self.assertEqual(view['metadata']['case_id'], original.case_id)
        for event in view['sequence']:
            self.assertEqual(set(event['attributes']), {'value', 'category'})
        self.assertNotIn('case_id', view['tabular'])

    def test_fit_validation_or_test_rejected(self):
        for split in ('val', 'test'):
            with self.assertRaises(ValueError):
                FeatureSchema(policy()).fit(generate_prefixes(case(), split, 31))

    def test_unapproved_policy_and_bad_allowlist_rejected(self):
        prefix = next(generate_prefixes(case(), 'train', 31))
        for bad in (replace(policy(), audit_approved=False),
                    replace(policy(), allowed_event_fields=('Accepted',)),
                    replace(policy(), allowed_event_fields=('OfferID',)),
                    replace(policy(), allowed_event_fields=('org:resource',))):
            with self.assertRaises(ValueError):
                sanitize(prefix, bad)

    def test_potential_outcome_proxy_rejected_without_rewriting_prefix(self):
        original = case()
        original.events[2]['concept:name'] = 'A_Denied'
        prefix = next(generate_prefixes(original, 'train', 31))
        with self.assertRaises(ValueError):
            sanitize(prefix, policy())
        self.assertEqual(len(prefix.events), 5)

    def test_historical_terminal_offer_activity_is_retained(self):
        original = case()
        original.events[2]['concept:name'] = 'O_Cancelled'
        original.events[2]['EventOrigin'] = 'Offer'
        prefix = next(generate_prefixes(original, 'train', 31))
        safe = sanitize(prefix, policy())
        self.assertEqual(safe['events'][2]['activity'], 'O_Cancelled')

    def test_future_event_mutation_cannot_change_early_view(self):
        original = case()
        schema = FeatureSchema(policy()).fit(generate_prefixes(original, 'train', 31))
        before = schema.transform(next(generate_prefixes(original, 'test', 31)))
        original.events[20]['value'] = 999999
        after = schema.transform(next(generate_prefixes(original, 'test', 31)))
        self.assertEqual(before, after)

    def test_label_gate_blocks(self):
        with self.assertRaises(LabelAmbiguityError):
            require_verified_labels({'status': 'blocked'})
