.PHONY: help install test run run-dev docker-build docker-run

PY ?= python
PORT ?= 7860
IMAGE ?= dataclean-env

help:
	@echo "DataClean-Env tasks"
	@echo ""
	@echo "  make install       Install dependencies"
	@echo "  make test          Run pytest"
	@echo "  make run           Run FastAPI + Gradio (http://localhost:$(PORT)/)"
	@echo "  make run-dev       Run with auto-reload"
	@echo "  make docker-build  Build Docker image ($(IMAGE))"
	@echo "  make docker-run    Run Docker image on :$(PORT)"
	@echo ""

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

run:
	$(PY) -m uvicorn server:app --host 0.0.0.0 --port $(PORT)

run-dev:
	$(PY) -m uvicorn server:app --host 0.0.0.0 --port $(PORT) --reload

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):7860 $(IMAGE)
