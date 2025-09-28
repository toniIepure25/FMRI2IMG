PY=python

install:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e . && pre-commit install

lint:
	ruff check .
	mypy src

test:
	pytest -q

run:
	$(PY) -m fmri2image.cli

dvc-init:
	dvc init -f
	dvc remote add -d origin $${DVC_REMOTE}
	dvc gc -w

docker-build:
	docker build -t fmri2image:dev .

docker-run:
	docker run --rm -it -v $$(pwd):/workspace -e WANDB_PROJECT -e WANDB_ENTITY fmri2image:dev

data-mock:
	python -m fmri2image.data.download_nsd --raw_root $(DATA_DIR)/raw

dvc-track:
	dvc add data/raw && git add data/raw.dvc && git commit -m "data: update raw" || true
