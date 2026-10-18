import json
import tempfile
import unittest
from pathlib import Path

from etce.utils.io import write_json_atomic


class AtomicJsonTests(unittest.TestCase):
    def test_atomic_json_replaces_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            write_json_atomic(path, {"value": 1})
            write_json_atomic(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 2})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
