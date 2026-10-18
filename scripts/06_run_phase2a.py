from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.deps'))
sys.path.insert(0, str(ROOT / 'src'))
from etce.phase2.pipeline import run_phase2a


if __name__ == '__main__':
    manifest = run_phase2a(ROOT)
    print({'phase': manifest['phase'], 'status': manifest['status'],
           'test_evaluations': manifest['test_evaluations']})
