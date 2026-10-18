# ETCE

ETCE is a protocol-constrained experimental reconstruction project for process-prediction explanations and evidence fusion. The repository contains source code, numbered experiment scripts, frozen configuration files, tests, and reproducibility instructions.

## Repository contents

```text
configs/    Frozen experiment configurations
scripts/    Numbered experiment and validation entry points
src/        Reusable ETCE Python package
tests/      Automated tests
```

The experiment design is documented in [ETCE_EXPERIMENT_PLAN_v3.md](ETCE_EXPERIMENT_PLAN_v3.md). Phase-specific commands and validation order are recorded in the `RUN_PHASE*.md` files.

## Data and generated artifacts

Raw datasets, intermediate datasets, model checkpoints, caches, predictions, and experiment results are intentionally excluded from this repository. They must not be committed to Git.

The scripts expect locally supplied datasets under paths such as:

```text
data/raw/BPI Challenge 2017.xes.gz
data/raw/bpic2018/BPI Challenge 2018.xes.gz
```

Generated artifacts are written beneath `results/`. Both `data/` and `results/` are protected by `.gitignore`.

## Environment

Python 3.11 or newer is required. For the frozen server environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
python -m pip install -e . --no-deps
```

Run the automated tests with:

```bash
python -m pytest
```

From Phase 4B onward, use the documented Linux commands and strict build-then-validate order in [RUN_PHASE4B.md](RUN_PHASE4B.md) and the relevant numbered scripts. The project root defaults to the script-relative repository root and may be overridden with `ETCE_ROOT`.

## Reproducibility notes

- Keep the frozen protocols, seeds, selected cases, checkpoints, metrics, and thresholds unchanged.
- Run long experiments with unbuffered logging and preserve complete phase output directories.
- Do not combine partial artifacts created from different code revisions, checkpoint sets, or machines.
- GPU or parallel execution is not implied by the published protocol.

## License

No license has been selected yet. Add an appropriate `LICENSE` file before treating this repository as open source.
