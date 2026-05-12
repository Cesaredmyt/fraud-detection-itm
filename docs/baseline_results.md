# Benchmarks de la línea base experimental

Resultados del sistema basado en reglas (`RulesBaseline`) sobre los datasets
completos limpios. Estos números son la **referencia contra la que se compara**
todos los modelos de aprendizaje automático del proyecto.

## Credit Card Fraud Detection

**Fecha**: 2026-11
**Versión del baseline**: v1
**Hiperparámetros**:

- `amount_high_threshold`: 5000.0
- `amount_medium_threshold`: 200.0
- `v17_negative_threshold`: -5.0
- `min_rules_to_flag`: 2

**Métricas globales**:

| Métrica   | Valor  |
| --------- | ------ |
| F1-score  | 0.1747 |
| AUC-ROC   | 0.7491 |
| Precision | 0.1709 |
| Recall    | 0.1786 |
| Accuracy  | 0.9973 |

**Matriz de confusión**:

|               | Pred. Legítima | Pred. Fraude |
| ------------- | -------------- | ------------ |
| Real Legítima | 281,082        | 388          |
| Real Fraude   | 368            | 80           |

**Tasas de activación por regla**:

| Regla                        | Tasa   |
| ---------------------------- | ------ |
| R1 (`is_night`)              | 17.66% |
| R2 (`amount_high`)           | 0.02%  |
| R3 (`round_medium_amount`)   | 1.05%  |
| R4 (`v17_strongly_negative`) | 0.11%  |

**Lectura clave**: el sistema basado en reglas detecta solo el 17.86% de los
fraudes reales y genera 388 falsos positivos. El F1-score de 0.17 y el AUC de
0.75 establecen el techo experimental que los modelos de ML deben superar para
validar la hipótesis del proyecto.

## PaySim

Pendiente de ejecución. Se realizará en la fase de evaluación final.
