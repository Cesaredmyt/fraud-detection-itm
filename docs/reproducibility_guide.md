# Guía de reproducibilidad

## Versiones soportadas

- Python 3.11 o 3.12. El paquete declara `>=3.11,<3.13`.
- Git.
- PostgreSQL 16 solo para migraciones y pruebas de integración con base de
  datos.
- `uv` para regenerar o sincronizar el lockfile. No es necesario para la
  instalación editable básica con `pip`.

No configure `PYTHONPATH`: el paquete se instala desde el layout `src/` y debe
poder importarse desde cualquier directorio.

## Dependencias y lockfiles

`pyproject.toml` es la fuente de verdad. Las dependencias opcionales están
separadas por uso:

- `dev`: calidad, pruebas y el superconjunto necesario para ejecutar la suite
  completa actual.
- `training`: entrenamiento, notebooks, visualización y acceso a datos.
- `api`: servicio HTTP y persistencia de la API.
- `observability`: métricas y telemetría.

`uv.lock` es el lockfile autoritativo y conserva resoluciones y hashes para
Python 3.11 y 3.12. `requirements.lock` es su exportación con hashes para
flujos basados en `pip`; `requirements.txt` solo apunta a esa exportación por
compatibilidad y nunca instala el repositorio desde Git.

## Instalación editable para desarrollo

Desde un checkout limpio:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
```

En Linux o macOS, active el entorno con `source .venv/bin/activate`. Para
validar Python 3.12, repita el procedimiento en otro entorno creado con ese
intérprete.

La instalación bloqueada equivalente con `uv` es:

```powershell
uv sync --all-extras --frozen
```

Si una política exige verificar los hashes con `pip`, instale primero la
exportación y después el proyecto editable sin volver a resolver dependencias:

```powershell
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps -e .
```

## Verificación desde fuera del repositorio

Con el entorno activado, cambie a cualquier directorio que no sea el checkout
y ejecute:

```powershell
Set-Location $env:TEMP
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
python -c "import fraud_detection; print(fraud_detection.__file__)"
```

El comando debe terminar con código `0` y mostrar una ruta bajo
`src/fraud_detection`, sin depender del directorio actual.

## Comprobaciones del proyecto

Desde la raíz del repositorio:

```powershell
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy
python -m pytest -m "not integration and not slow"
```

Las pruebas de integración requieren las variables de `.env.example` y una
base PostgreSQL preparada según `db/migrations/README.md`. No almacene
credenciales reales en el repositorio.

## Regeneración controlada del lock

Regenerar archivos solo cuando cambie `pyproject.toml` o se apruebe una
actualización de dependencias:

```powershell
uv lock
uv export --all-extras --no-emit-project --no-editable `
  --format requirements.txt --output-file requirements.lock
uv lock --check
git diff -- pyproject.toml uv.lock requirements.lock requirements.txt
```

Después, repita la instalación y todas las comprobaciones en Python 3.11 y
3.12. Los cuatro archivos deben entrar en el mismo commit para evitar que la
declaración y la resolución diverjan.

## Reproducción de experimentos

Las semillas se fijan a `42` por defecto y se configuran en
`config/config.yaml`. Los resultados previos al protocolo v2 están marcados
como `academic_legacy`; no deben usarse para elegir modelos futuros hasta
repetirlos con los controles de datos y leakage definidos en el roadmap.
