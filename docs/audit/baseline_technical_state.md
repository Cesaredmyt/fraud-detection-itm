# Estado técnico inicial — Día 1

> Fotografía reproducible tomada el 2026-09-19 a las 01:46:14 -06:00
> (2026-09-19T07:46:14Z). No se instalaron dependencias ni se modificaron
> variables de entorno antes de ejecutar las comprobaciones.

## Checklist del Día 1

- [x] Tickets `FD-001`–`FD-032` registrados y tablero local creado con
  `Backlog / In progress / Review / Done`.
- [x] Commit, rama, `git status`, Python, pip, SO y hardware registrados.
- [x] Ruff, Mypy y Pytest invocados sin alterar variables; salida, código y
  duración conservados, incluidos los fallos por dependencias ausentes.
- [x] Fallo de instalación editable documentado sin aplicar la corrección del
  Día 2.
- [x] Fotografía técnica creada en este documento.
- [x] Métricas existentes clasificadas como `academic_legacy`.

La aceptación del día queda satisfecha como fotografía técnica fechada. La
sincronización de tickets con un servicio remoto no forma parte de esta
fotografía: el repositorio no especifica uno y la CLI de GitHub no estaba
disponible. El registro versionado evita atribuir una integración externa que
no existe.

## Identidad del checkout

| Campo | Valor |
| --- | --- |
| Repositorio | `https://github.com/Cesaredmyt/fraud-detection-itm.git` |
| Commit | `ba053d26397141bb21c8557ef8cbb195cf0cba07` |
| Rama general del roadmap | `feature/antifraud-mvp` |
| Tag que apunta al commit | `v1.0.0` |
| Python | CPython 3.11.0, 64 bits |
| Ejecutable | `C:\Users\dcesa\AppData\Local\Programs\Python\Python311\python.exe` |
| pip | 22.3 |
| SO | Microsoft Windows 11 Home Single Language, 10.0.26200 (build 26200) |
| CPU | AMD Ryzen 5 5600H with Radeon Graphics, 12 procesadores lógicos |
| RAM física | 29,909,643,264 bytes |
| GPU | AMD Radeon(TM) Graphics; NVIDIA GeForce RTX 3050 Ti Laptop GPU |

`git status --short --branch` antes de crear archivos de auditoría:

```text
## feature/antifraud-mvp
?? docs/roadmap_producto_90_dias.md
?? scripts/create_notebook_05_evaluacion_final.py
```

Los dos archivos sin seguimiento ya existían en `main` antes de abrir la rama
de auditoría. Se preservaron y no se modificaron.

## Entorno Python observado

No había un entorno virtual o Conda activo. `PYTHONPATH` no estaba definido.
`python -m pip list --format=freeze` devolvió únicamente:

```text
pillow==12.3.0
pip==22.3
pymupdf==1.28.2
setuptools==65.5.0
```

## Fallo de instalación editable

El paquete del proyecto no está instalado en el intérprete activo:

```text
> python -m pip show fraud-detection-itm
WARNING: Package(s) not found: fraud-detection-itm

> python -c "import fraud_detection"
ModuleNotFoundError: No module named 'fraud_detection'
```

Además, `requirements.txt` contiene una instalación editable del propio
repositorio fijada al commit anterior
`7bff04eeca184fed115da7b5b8e5a98fcd719210`:

```text
-e git+https://github.com/Cesaredmyt/fraud-detection-itm.git@7bff04eeca184fed115da7b5b8e5a98fcd719210#egg=fraud_detection_itm
```

Ese origen no representa el checkout auditado (`ba053d2…`) y puede instalar
código desde otra revisión/ruta de origen. En este equipo no quedó un
`.egg-link`, `.pth` ni distribución `fraud*` en los directorios de
`site-packages`; el fallo observable es una instalación editable ausente,
con una referencia obsoleta en el archivo de dependencias. La corrección se
reserva para el Día 2, tal como indica el roadmap.

## Línea base de calidad

Las órdenes se ejecutaron desde la raíz del repositorio, sin instalar paquetes
y sin alterar variables. Las duraciones son tiempo de pared observado por el
runner.

| Comprobación | Orden | Resultado | Código | Duración |
| --- | --- | --- | ---: | ---: |
| Ruff lint | `python -m ruff check .` | No ejecutable: módulo ausente | 1 | 1.135 s |
| Ruff format | `python -m ruff format --check .` | No ejecutable: módulo ausente | 1 | 1.124 s |
| Mypy | `python -m mypy` | No ejecutable: módulo ausente | 1 | 1.190 s |
| Pytest unitario/no integración | `python -m pytest -m "not integration"` | No ejecutable: módulo ausente | 1 | 1.104 s |

Salida íntegra de Ruff lint:

```text
C:\Users\dcesa\AppData\Local\Programs\Python\Python311\python.exe: No module named ruff
```

Salida íntegra de Ruff format:

```text
C:\Users\dcesa\AppData\Local\Programs\Python\Python311\python.exe: No module named ruff
```

Salida íntegra de Mypy:

```text
C:\Users\dcesa\AppData\Local\Programs\Python\Python311\python.exe: No module named mypy
```

Salida íntegra de Pytest:

```text
C:\Users\dcesa\AppData\Local\Programs\Python\Python311\python.exe: No module named pytest
```

También se intentó `ruff check .` directamente. Falló porque el ejecutable no
está en `PATH`; la invocación canónica con `python -m ruff` confirmó que el
módulo tampoco está instalado.

## Estado de las métricas

Todas las métricas producidas antes del protocolo v2 quedan clasificadas como
`academic_legacy`. El inventario autoritativo está en
[`metric_registry.md`](metric_registry.md). La etiqueta no invalida su uso en
la tesis; impide tratarlas como evidencia de producto, seleccionar el futuro
modelo candidato con ellas o presentarlas como resultados comerciales.

## Hallazgos que condicionan días posteriores

1. **Día 2 bloqueado por dependencias:** el checkout no puede importar el
   paquete y no tiene instaladas las herramientas de desarrollo. Debe repararse
   la instalación antes de interpretar resultados de lint, tipos o tests.
2. **Día 3:** CI usa `mypy` con `continue-on-error: true`, por lo que los errores
   de tipos no bloquean merges actualmente.
3. **Días 8–14:** los resultados heredados usan split estratificado aleatorio;
   varios experimentos PaySim calculan features antes del split. Deben
   considerarse evidencia académica heredada hasta repetirlos con el protocolo
   sin leakage.
4. **Regla TEST:** `docs/baseline_results.md` registra selección de
   `min_rules_to_flag=4` y recalibración de `contamination` sobre TEST. Esos
   resultados no pueden usarse para selección/tuning futuro.
5. **Días 15–21:** no existe todavía un manifest de dataset que vincule las
   métricas con checksum, licencia y ventana temporal.
6. **Trazabilidad:** `reports/metrics/` y `reports/experiment_logs/` solo
   contienen `.gitkeep`; las tablas consolidadas no conservan por sí solas toda
   la salida de ejecución.

## Repetición de la fotografía

Una vez completado el Día 2, ejecutar desde un checkout limpio y sin
`PYTHONPATH` manual:

```powershell
python --version
python -m pip --version
python -m pip show fraud-detection-itm
python -c "import fraud_detection; print(fraud_detection.__file__)"
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest -m "not integration"
```
