# Benchmarks experimentales

Resultados de la línea base experimental (`RulesBaseline`) y del primer modelo
de aprendizaje automático (`RandomForestModel`) sobre el dataset Credit Card
Fraud Detection. Estos números son la referencia oficial del proyecto y se
citan en el documento de tesis.

## Configuración experimental

- **Dataset**: Credit Card Fraud Detection (Dal Pozzolo et al., 2015)
- **Filas tras limpieza**: 281,918 (1,889 duplicados eliminados del original)
- **Tasa de fraude global**: 0.159%
- **Split estratificado**: 70% train (197,342) / 15% val (42,288) / 15% test (42,288)
- **Tasa de fraude preservada**: ~0.159% en cada split
- **Random seed**: 42
- **Sin SMOTE aplicado** (primer experimento; se evaluará el impacto en Día 4-5)

## Resultados sobre el conjunto de validación

### Sistema basado en reglas (RulesBaseline)

**Hiperparámetros**:

- `amount_high_threshold`: 5000.0
- `amount_medium_threshold`: 200.0
- `v17_negative_threshold`: -5.0
- `min_rules_to_flag`: 2

**Matriz de confusión**:

|               | Pred. Legítima | Pred. Fraude |
| ------------- | -------------- | ------------ |
| Real Legítima | 42,158         | 63           |
| Real Fraude   | 58             | 9            |

**Métricas (clase fraude)**:

| Métrica   | Valor  |
| --------- | ------ |
| Precision | 0.1250 |
| Recall    | 0.1343 |
| F1-score  | 0.1295 |
| AUC-ROC   | 0.7254 |

### Random Forest (RandomForestModel)

**Hiperparámetros**:

- `n_estimators`: 200
- `max_depth`: 20
- `min_samples_split`: 5
- `min_samples_leaf`: 2
- `max_features`: sqrt
- `class_weight`: balanced
- `random_state`: 42

**Matriz de confusión**:

|               | Pred. Legítima | Pred. Fraude |
| ------------- | -------------- | ------------ |
| Real Legítima | 42,219         | 2            |
| Real Fraude   | 19             | 48           |

**Métricas (clase fraude)**:

| Métrica   | Valor  |
| --------- | ------ |
| Precision | 0.9600 |
| Recall    | 0.7164 |
| F1-score  | 0.8205 |
| AUC-ROC   | 0.9356 |

**Top 10 features más importantes**:

| Rank | Feature | Importancia |
| ---- | ------- | ----------- |
| 1    | V14     | 17.37%      |
| 2    | V10     | 11.28%      |
| 3    | V4      | 10.54%      |
| 4    | V12     | 10.47%      |
| 5    | V11     | 8.82%       |
| 6    | V17     | 8.48%       |
| 7    | V3      | 3.73%       |
| 8    | V16     | 3.56%       |
| 9    | V7      | 2.90%       |
| 10   | V2      | 1.95%       |

## Comparativa Random Forest vs Baseline

| Métrica              | Baseline (Reglas) | Random Forest | Diferencia absoluta | Mejora relativa |
| -------------------- | ----------------- | ------------- | ------------------- | --------------- |
| F1-score             | 0.1295            | 0.8205        | +0.6910             | **+533.6%**     |
| AUC-ROC              | 0.7254            | 0.9356        | +0.2102             | +29.0%          |
| Precision            | 0.1250            | 0.9600        | +0.8350             | +668.0%         |
| Recall               | 0.1343            | 0.7164        | +0.5821             | +433.4%         |
| Falsos positivos     | 63                | 2             | -61                 | -96.8%          |
| Verdaderos positivos | 9                 | 48            | +39                 | +433.3%         |

## Lectura clave

El modelo Random Forest supera al sistema basado en reglas en todas las
métricas relevantes para la hipótesis del proyecto:

- **F1-score 6.3 veces mayor** (0.82 vs 0.13).
- **AUC 1.29 veces mayor** (0.94 vs 0.73).
- **Reducción del 96.8% en falsos positivos** (de 63 a 2).
- **Incremento del 433% en verdaderos positivos** (de 9 a 48).

Estos resultados validan la hipótesis del proyecto en el dataset Credit Card:
un modelo de aprendizaje automático supera significativamente a un sistema
tradicional basado en reglas en la detección de fraude financiero.

La importancia relativa de las features confirma además la consistencia entre
el análisis exploratorio (Notebook 01) y el modelo entrenado: las variables
V14, V10, V12 y V17, identificadas en el EDA como las de mayor correlación
con la clase, son también las que el Random Forest considera más predictivas.

## Trabajo pendiente

- Evaluación sobre el conjunto de **test** (final, una sola vez al cierre).
- Aplicación de SMOTE y comparación de impacto.
- Entrenamiento de XGBoost y comparación con Random Forest.
- Implementación del modelo híbrido (supervisado + no supervisado).
- Replicación completa de la metodología sobre el dataset PaySim.
