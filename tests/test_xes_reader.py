import gzip
from pathlib import Path
import tempfile
import unittest
from etce.data.load_bpic2017 import read_cases


class ReaderTests(unittest.TestCase):
    def test_gzip_and_plain_match_preserving_lifecycle(self):
        xml = b'''<log xmlns="http://www.xes-standard.org/"><trace>
        <string key="concept:name" value="case"/>
        <event><string key="concept:name" value="A"/><boolean key="Accepted" value="false"/>
        <string key="lifecycle:transition" value="start"/><date key="time:timestamp" value="2016-01-01T00:00:00Z"/></event>
        <event><string key="concept:name" value="A"/><string key="lifecycle:transition" value="complete"/>
        <date key="time:timestamp" value="2016-01-01T00:00:01Z"/></event>
        </trace></log>'''
        with tempfile.TemporaryDirectory() as tmp:
            plain, compressed = Path(tmp)/'raw.xes', Path(tmp)/'raw.xes.gz'
            plain.write_bytes(xml)
            compressed.write_bytes(gzip.compress(xml))
            a, b = list(read_cases(plain)), list(read_cases(compressed))
            self.assertEqual(a, b)
            self.assertEqual(len(a[0].events), 2)
            self.assertIs(a[0].events[0]['Accepted'], False)
