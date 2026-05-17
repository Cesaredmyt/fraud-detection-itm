.PHONY: help setup install install-dev clean lint format test test-unit test-integration db-create db-migrate db-drop pipeline

help:
	@echo "Comandos disponibles:"
	@echo "  setup           - Crea venv e instala todas las dependencias (dev incluido)"
	@echo "  install         - Instala dependencias de producción"
	@echo "  install-dev     - Instala dependencias de desarrollo"
	@echo "  lint            - Ejecuta ruff check sobre src/ y tests/"
	@echo "  format          - Formatea el código con ruff format"
	@echo "  test            - Ejecuta todos los tests"
	@echo "  test-unit       - Ejecuta solo tests unitarios"
	@echo "  test-integration- Ejecuta solo tests de integración"
	@echo "  clean           - Limpia caches, builds y datos intermedios"
	@echo "  db-create       - Crea la base de datos y usuario en PostgreSQL"
	@echo "  db-migrate      - Aplica las migraciones SQL"
	@echo "  db-drop         - Elimina la base de datos (cuidado)"
	@echo "  pipeline        - Ejecuta el pipeline ML end-to-end"

setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	.venv/bin/pre-commit install

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests
	mypy src

format:
	ruff format src tests
	ruff check --fix src tests

test:
	pytest

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov .coverage build dist

db-migrate:
	python scripts/run_migrations.py

pipeline:
	python pipelines/run_full_pipeline.py