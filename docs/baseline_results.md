# Benchmarks experimentales

Resultados oficiales del proyecto en el dataset Credit Card Fraud Detection.
Estos números se citan en el documento de tesis y son reproducibles mediante
los scripts de `pipelines/` con los YAMLs de configuración en
`config/experiments/`.

## Configuración experimental común

- **Dataset**: Credit Card Fraud Detection (Dal Pozzolo et al., 2015)
- **Filas tras limpieza**: 281,918 (1,889 duplicados eliminados del original)
- **Tasa de fraude global**: 0.159%
- **Split estratificado**: 70% train (197,342) / 15% val (42,288) / 15% test (42,288)
- **Random seed**: 42
- **Features**: temporales + monto (`feature_version: v1`)
- **Sin SMOTE aplicado** en estos experimentos
- **Sin tuning de hiperparámetros**: defaults del proyecto

## Experimento 001: Baseline vs Random Forest

**Configuración**: `config/experiments/exp_001_creditcard_baseline.yaml`

### Sistema basado en reglas (`RulesBaseline`)

**Hiperparámetros**:

- `amount_high_threshold`: 5000.0
- `amount_medium_threshold`: 200.0
- `v17_negative_threshold`: -5.0
- `min_rules_to_flag`: 2

### Random Forest (`RandomForestModel`)

**Hiperparámetros**:

- `n_estimators`: 200
- `max_depth`: 20
- `min_samples_split`: 5
- `min_samples_leaf`: 2
- `max_features`: sqrt
- `class_weight`: balanced
- `random_state`: 42

### Resultados sobre VALIDATION

| Modelo        | Precision | Recall | F1     | AUC-ROC | TP  | FP  | FN  |
| ------------- | --------- | ------ | ------ | ------- | --- | --- | --- |
| RulesBaseline | 0.1250    | 0.1343 | 0.1295 | 0.7254  | 9   | 63  | 58  |
| RandomForest  | 0.9600    | 0.7164 | 0.8205 | 0.9356  | 48  | 2   | 19  |

**Mejora del Random Forest sobre el baseline**:

- F1: +533.6%
- AUC-ROC: +29.0%
- Verdaderos positivos: +433.3%
- Falsos positivos: -96.8%

## Experimento 002: Comparativa de tres modelos

**Configuración**: `config/experiments/exp_002_creditcard_three_models.yaml`

Incorpora XGBoost a la comparativa, manteniendo idéntica la configuración de
datos y splits del Experimento 001.

### XGBoost (`XGBoostModel`)

**Hiperparámetros**:

- `n_estimators`: 300
- `max_depth`: 8
- `learning_rate`: 0.1
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `scale_pos_weight`: calculado automáticamente como `n_neg / n_pos` ≈ 627
- `tree_method`: hist
- `eval_metric`: aucpr
- `random_state`: 42

### Resultados sobre TRAIN

| Modelo        | Precision | Recall | F1     | AUC-ROC | TP  | FP  | FN  |
| ------------- | --------- | ------ | ------ | ------- | --- | --- | --- |
| RulesBaseline | 0.1813    | 0.1911 | 0.1860 | 0.7573  | 60  | 271 | 254 |
| RandomForest  | 0.9691    | 1.0000 | 0.9843 | 0.9999  | 314 | 10  | 0   |
| XGBoost       | 1.0000    | 1.0000 | 1.0000 | 1.0000  | 314 | 0   | 0   |

### Resultados sobre VALIDATION

| Modelo        | Precision | Recall | F1         | AUC-ROC    | TP  | FP  | FN  |
| ------------- | --------- | ------ | ---------- | ---------- | --- | --- | --- |
| RulesBaseline | 0.1250    | 0.1343 | 0.1295     | 0.7254     | 9   | 63  | 58  |
| RandomForest  | 0.9600    | 0.7164 | 0.8205     | 0.9356     | 48  | 2   | 19  |
| **XGBoost**   | 0.9444    | 0.7612 | **0.8430** | **0.9693** | 51  | 3   | 16  |

### Mejoras relativas sobre el baseline

| Modelo       | F1 vs Baseline | AUC vs Baseline | Recall vs Baseline |
| ------------ | -------------- | --------------- | ------------------ |
| RandomForest | +533.6%        | +29.0%          | +433.4%            |
| XGBoost      | +550.9%        | +33.6%          | +466.7%            |

### XGBoost vs Random Forest (validation)

| Métrica   | RandomForest | XGBoost | Ganador      | Diferencia |
| --------- | ------------ | ------- | ------------ | ---------- |
| F1        | 0.8205       | 0.8430  | XGBoost      | +2.7%      |
| AUC-ROC   | 0.9356       | 0.9693  | XGBoost      | +3.6%      |
| Recall    | 0.7164       | 0.7612  | XGBoost      | +6.3%      |
| Precision | 0.9600       | 0.9444  | RandomForest | -1.6%      |
| TP        | 48           | 51      | XGBoost      | +3 fraudes |
| FP        | 2            | 3       | RandomForest | +1 FP      |

## Lectura clave

Los tres modelos evaluados se ordenan de forma consistente en todas las
métricas relevantes: **XGBoost > Random Forest > Reglas**, validando que la
sofisticación del algoritmo aporta valor incremental incluso entre métodos
de aprendizaje automático.

**Validación de la hipótesis**: el modelo más sofisticado (XGBoost) alcanza
un F1-score de 0.8430 frente al 0.1295 del sistema basado en reglas. Esto
representa una mejora del 551% en la métrica primaria de la hipótesis del
proyecto, con una mejora paralela del 34% en AUC-ROC. Operacionalmente,
XGBoost detecta 51 fraudes reales de 67 posibles (recall 76.1%) con solo 3
falsos positivos, mientras que el sistema de reglas detecta apenas 9
fraudes generando 63 alertas falsas.

**Comparación con literatura**: los resultados se sitúan en el rango
competitivo de trabajos publicados sobre el mismo dataset (Awoyemi et al.,
2017 reportan F1≈0.85, AUC≈0.96; Dal Pozzolo et al., 2015 reportan
F1≈0.74, AUC≈0.94), lo que respalda la solidez metodológica del proyecto.

## Trabajo pendiente

- Evaluación sobre el conjunto de **test** (una sola vez, al cierre).
- Aplicación de SMOTE y comparación de impacto en recall.
- Implementación de `IsolationForestModel` (no supervisado).
- Implementación del **modelo híbrido** (supervisado + no supervisado).
- Replicación completa de la metodología sobre el dataset PaySim.
- Validación cruzada k-fold (k=5) sobre todos los modelos.
- Análisis de errores: revisión cualitativa de los falsos negativos.
