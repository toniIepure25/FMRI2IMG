# fmri2image

Industrial-grade scaffold (Hydra + DVC + Docker + CI + tests) pentru reconstrucție fMRI→Imagine.

## Setup rapid

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .
pre-commit install
cp .env.example .env
export DATA_DIR=$(pwd)/data
make dvc-init
pytest -q
python -m fmri2image.cli
```
