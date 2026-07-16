.PHONY: install install-dev download eda train api dashboard simulate test lint format docker-build docker-up docker-down

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

download:
	python scripts/download_data.py

eda:
	python -m retainiq.data.loader

train:
	python scripts/train.py

api:
	uvicorn retainiq.api.main:app --reload --port 8000

dashboard:
	streamlit run src/retainiq/dashboard/app.py

simulate:
	python scripts/simulate_traffic.py --n 300

test:
	python -m pytest -q

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

docker-build:
	docker build -t retainiq:latest .

docker-up:
	docker compose up --build

docker-down:
	docker compose down
