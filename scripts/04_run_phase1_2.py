from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from etce.data.phase1_2 import run_phase1_2
from etce.utils.io import read_json


if __name__ == '__main__':
    print(run_phase1_2(ROOT, read_json(ROOT / 'configs/bpic2017.yaml')))
