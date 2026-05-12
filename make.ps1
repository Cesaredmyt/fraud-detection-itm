# Equivalente del Makefile para Windows PowerShell
# Uso: .\make.ps1 <comando>

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

function Show-Help {
    Write-Host "Comandos disponibles:" -ForegroundColor Cyan
    Write-Host "  setup           - Crea venv e instala todas las dependencias"
    Write-Host "  install         - Instala dependencias de produccion"
    Write-Host "  install-dev     - Instala dependencias de desarrollo"
    Write-Host "  lint            - Ejecuta ruff check"
    Write-Host "  format          - Formatea el codigo con ruff"
    Write-Host "  test            - Ejecuta todos los tests"
    Write-Host "  test-unit       - Ejecuta solo tests unitarios"
    Write-Host "  test-integration- Ejecuta solo tests de integracion"
    Write-Host "  clean           - Limpia caches y builds"
    Write-Host "  db-migrate      - Aplica migraciones SQL"
    Write-Host "  pipeline        - Ejecuta pipeline end-to-end"
}

switch ($Command) {
    "help" { Show-Help }

    "setup" {
        python -m venv .venv
        .\.venv\Scripts\Activate.ps1
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        pre-commit install
    }

    "install" { pip install -e . }

    "install-dev" {
        pip install -e ".[dev]"
        pre-commit install
    }

    "lint" {
        ruff check src tests
        mypy src
    }

    "format" {
        ruff format src tests
        ruff check --fix src tests
    }

    "test" { pytest }
    "test-unit" { pytest -m unit }
    "test-integration" { pytest -m integration }

    "clean" {
        Get-ChildItem -Path . -Include "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "*.egg-info" -Recurse -Force |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force htmlcov, .coverage, build, dist -ErrorAction SilentlyContinue
        Write-Host "Limpieza completa." -ForegroundColor Green
    }

    "db-migrate" { python scripts/run_migrations.py }

    "pipeline" { python pipelines/run_full_pipeline.py }

    default {
        Write-Host "Comando no reconocido: $Command" -ForegroundColor Red
        Show-Help
    }
}