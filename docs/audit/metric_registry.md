# Registro de métricas heredadas

**Clasificación:** `academic_legacy`

Todas las métricas existentes al commit
`ba053d26397141bb21c8557ef8cbb195cf0cba07` son resultados académicos
heredados. No son métricas de candidato, champion ni piloto y no deben usarse
para tuning futuro.

## Artefactos cubiertos

| Artefacto | Clasificación | Motivo principal |
| --- | --- | --- |
| `docs/baseline_results.md` | `academic_legacy` | Consolida los experimentos 001–009 con protocolo anterior |
| `config/experiments/*.yaml` | `academic_legacy` | Configuraciones del protocolo experimental anterior |
| `notebooks/05_evaluacion_final_y_comparacion.ipynb` | `academic_legacy` | Evalúa TEST y documenta ajustes hechos sobre TEST |
| `reports/tablas_evaluacion_final.xlsx` | `academic_legacy` | Exporta las tablas del notebook de evaluación final |
| Métricas persistidas por corridas 001–009, si existen fuera de Git | `academic_legacy` | Fueron generadas antes del protocolo v2 |

## Restricciones de uso

- Pueden citarse como resultados históricos de tesis, siempre con la etiqueta.
- No pueden decidir features, modelos, reglas, contaminación ni umbrales.
- No constituyen evidencia comercial ni validación sobre datos de cliente.
- Los resultados del protocolo v2 deberán usar una clasificación distinta,
  un manifest de datos y una identidad de corrida nueva.
