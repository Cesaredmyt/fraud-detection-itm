# Guía de reproducibilidad

## Requisitos previos

- Python 3.11+
- PostgreSQL 16
- Git

## Setup paso a paso

1. Clonar el repositorio.
2. Crear entorno virtual: `python -m venv .venv`.
3. Activar: `.\.venv\Scripts\Activate.ps1` (Windows) o `source .venv/bin/activate` (Linux/macOS).
4. Instalar dependencias: `pip install -e ".[dev]"`.
5. Copiar `.env.example` a `.env` y editar credenciales.
6. Crear base de datos PostgreSQL (ver `db/migrations/README.md`).
7. Ejecutar migraciones: `python scripts/run_migrations.py`.
8. Verificar instalación: `pytest`.

## Reproducción de experimentos

(Pendiente de completar cuando los pipelines estén implementados.)

## Semillas aleatorias

Todas las semillas se fijan a `42` por defecto, configurable en `config/config.yaml`.