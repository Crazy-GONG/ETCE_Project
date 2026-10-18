"""Build Phase 1.1 labels/splits/coverage only; no predictor dataset or model."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from etce.data.phase1_1 import build_phase1_1
from etce.utils.io import read_json

if __name__ == '__main__':
    summary = build_phase1_1(ROOT, read_json(ROOT / 'configs/bpic2017.yaml'))
    print(summary)
