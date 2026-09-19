# Auditoría del Día 2 — instalación y dependencias

Fecha: 2026-09-19
Rama: `feature/antifraud-mvp`
Ticket: `FD-001`

## Checklist

- [x] Eliminada de `requirements.txt` la instalación editable remota del
  propio repositorio.
- [x] Separados los extras `dev`, `training`, `api` y `observability`.
- [x] Generados `uv.lock` y `requirements.lock`; la exportación para `pip`
  contiene hashes SHA-256.
- [x] Instalado el checkout con `pip install -e ".[dev]"` en entornos vacíos.
- [x] Probados Python 3.11 y 3.12 sin `PYTHONPATH` manual.
- [x] Actualizada la guía de reproducibilidad.

## Decisiones

`uv.lock` es el lockfile autoritativo porque resuelve en un solo archivo el
rango soportado `>=3.11,<3.13`. `requirements.lock` se versiona como
exportación con hashes para consumidores que solo admiten `pip`.

El extra `dev` es, por ahora, un superconjunto intencional de los demás extras:
la suite existente importa componentes de entrenamiento y persistencia. Esto
garantiza que el comando exacto exigido por el roadmap pueda ejecutar todas
las comprobaciones. Cuando los tests estén separados por job en el Día 3 se
podrá reducir ese solapamiento sin romper el entorno de desarrollo.

## Evidencia ejecutada

Se crearon dos entornos virtuales nuevos y separados. En ambos se ejecutó:

```text
python -m pip install --require-hashes -r requirements.lock
python -m pip install -e ".[dev]"
python -m pip check
```

Resultados:

| Comprobación | Python 3.11 | Python 3.12 |
| --- | --- | --- |
| Versión probada | 3.11 | 3.12.14 |
| Instalación con hashes | Correcta | Correcta |
| Instalación editable `.[dev]` | Correcta | Correcta |
| `pip check` | Sin conflictos | Sin conflictos |
| Import externo sin `PYTHONPATH` | Correcto | Correcto |
| Pytest sin `integration`/`slow` | 196 passed, 8 deselected | 196 passed, 8 deselected |
| Cobertura observada | 78% | 78% |

En ambos intérpretes, desde un directorio temporal fuera del repositorio,
`fraud_detection.__file__` resolvió al checkout actual bajo
`src/fraud_detection/__init__.py`.

Comprobaciones adicionales en Python 3.11:

```text
ruff check src tests                 -> correcto
ruff format --check src tests        -> 59 files already formatted
mypy                                 -> Success: no issues found in 33 source files
```

## Impacto en días posteriores

- El Día 3 ya puede crear jobs independientes sobre un entorno reproducible.
- La cobertura actual de 78% no cumple todavía el mínimo global de 80% que
  exigirá el Día 3; debe elevarse antes de activar ese gate.
- Los extras `api` y `observability` reservan desde ahora las dependencias de
  los días de servicio y telemetría, sin introducir código de esos días.
- El hook de Ruff quedó alineado con la versión `0.16.8` resuelta en el lock;
  así el formateo local y el futuro job de CI aplican la misma regla.
- Los locks deben actualizarse junto con `pyproject.toml` en cualquier cambio
  futuro de dependencias, incluidos los días 22–30 y 50–56.

## Criterio de aceptación

Cumplido: un checkout limpio, instalado de forma editable, importa
`fraud_detection` desde un directorio externo tanto en Python 3.11 como en
Python 3.12 y sin modificar `PYTHONPATH`.
