"""Streaming XES reader; preserve original order and all lifecycle events."""
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Case:
    case_id: str
    attributes: dict
    events: tuple[dict, ...]


def timestamp(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('Timezone missing in raw timestamp')
    return value.astimezone(timezone.utc)


def attributes(element):
    result = {}
    for child in element:
        kind = child.tag.rsplit('}', 1)[-1]
        key, value = child.get('key'), child.get('value')
        if key is None:
            continue
        if key in result:
            raise ValueError(f'Duplicate XES attribute: {key}')
        if kind == 'int':
            value = int(value)
        elif kind == 'float':
            value = float(value)
        elif kind == 'boolean':
            if value not in {'true', 'false'}:
                raise ValueError(f'Invalid XES boolean {value}')
            value = value == 'true'
        elif kind in {'list', 'container'}:
            value = attributes(child)
        result[key] = value
    return result


def read_cases(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        context = ET.iterparse(stream, events=('start', 'end'))
        _, root = next(context)
        for action, element in context:
            if action == 'end' and element.tag.rsplit('}', 1)[-1] == 'trace':
                attrs = attributes(element)
                if 'concept:name' not in attrs:
                    raise ValueError('Trace missing concept:name; cannot invent identifier')
                events = tuple(attributes(child) for child in element
                               if child.tag.rsplit('}', 1)[-1] == 'event')
                yield Case(str(attrs['concept:name']), attrs, events)
                root.remove(element)
                element.clear()
