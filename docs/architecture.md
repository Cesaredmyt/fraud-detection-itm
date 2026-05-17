# Arquitectura del sistema

Documento técnico de referencia del PMV.

## Capas

1. **Data Layer** — Almacenamiento en PostgreSQL y filesystem.
2. **Processing Layer** — ETL y feature engineering.
3. **ML Layer** — Entrenamiento de modelos supervisados, no supervisados e híbridos.
4. **Evaluation Layer** — Cálculo de métricas y comparación contra baseline.
5. **Orchestration Layer** — Scripts de pipeline end-to-end.
6. **Presentation Layer** — Notebooks de análisis y figuras para la tesis.

## Principio rector

> Todo lo que se ejecuta dos veces vive en `src/`. Todo lo que cuenta una historia vive en `notebooks/`.

(Documento en construcción.)