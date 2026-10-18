"""Deny by default; audited policy is required before constructing predictors."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LeakagePolicy:
    allowed_event_fields: tuple[str, ...]
    allowed_case_fields: tuple[str, ...]
    excluded_fields: dict[str, str]
    excluded_activities: dict[str, str]
    audit_approved: bool = False

    def validate(self):
        if not self.audit_approved:
            raise ValueError('Leakage policy has not passed raw-data audit')
        allowed = set(self.allowed_event_fields) | set(self.allowed_case_fields)
        for name in allowed:
            normalized = name.casefold().replace('_', '').replace(':', '')
            if (name in self.excluded_fields or normalized.endswith('id')
                    or normalized in {'accepted', 'selected', 'conceptname', 'orgresource', 'resource'}):
                raise ValueError(f'Forbidden predictor field: {name}')


def sanitize(prefix, policy):
    policy.validate()
    if len(prefix.events) != prefix.endpoint:
        raise ValueError('Prefix length does not match endpoint')
    # Never delete or mask an activity and then shift original prefix endpoints.
    if any(e['concept:name'] in policy.excluded_activities for e in prefix.events):
        raise ValueError('Prefix contains excluded activity; explicit exclusion accounting required')
    return {'events': [{'activity': e['concept:name'], 'timestamp': e['time:timestamp'],
                        'attributes': {key: e[key] for key in policy.allowed_event_fields if key in e}}
                       for e in prefix.events],
            'static': {key: prefix.case_attributes[key] for key in policy.allowed_case_fields
                       if key in prefix.case_attributes}}
