from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from etce.data.audit_bpic2017 import audit
from etce.utils.io import read_json
from etce.data.report import finish_audit
from etce.utils.logging import make_logger

if __name__ == '__main__':
    (ROOT / 'results/phase1/logs').mkdir(parents=True, exist_ok=True)
    logger = make_logger(ROOT / 'results/phase1/logs/audit.log')
    summary = audit(ROOT, read_json(ROOT / 'configs/bpic2017.yaml'))
    gate = finish_audit(ROOT, summary)
    logger.info('Raw audit completed: %s', summary)
    logger.info('Label gate: %s', gate)
    print(summary)
    print('Label gate:', gate['status'], '; see results/phase1/PHASE1_REPORT.md')
