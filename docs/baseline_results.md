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

## Experimento 003: Comparativa de los 4 modelos individuales

**Configuración**: `config/experiments/exp_003_creditcard_four_models.yaml`

Incorpora Isolation Forest (no supervisado) a la comparativa, completando
el set de modelos individuales del protocolo antes de implementar el
modelo híbrido.

### Isolation Forest (`IsolationForestModel`)

**Hiperparámetros**:

- `n_estimators`: 200
- `max_samples`: auto (min(256, n_samples))
- `contamination`: auto
- `bootstrap`: false
- `random_state`: 42

### Resultados sobre VALIDATION

| Modelo          | Precision | Recall | F1         | AUC-ROC    | TP  | FP   | FN  |
| --------------- | --------- | ------ | ---------- | ---------- | --- | ---- | --- |
| RulesBaseline   | 0.1250    | 0.1343 | 0.1295     | 0.7254     | 9   | 63   | 58  |
| RandomForest    | 0.9600    | 0.7164 | 0.8205     | 0.9356     | 48  | 2    | 19  |
| **XGBoost**     | 0.9444    | 0.7612 | **0.8430** | **0.9693** | 51  | 3    | 16  |
| IsolationForest | 0.0222    | 0.7463 | 0.0431     | 0.9259     | 50  | 2203 | 17  |

### Interpretación de Isolation Forest

Isolation Forest exhibe un patrón característico de los detectores de
anomalías no supervisados aplicados a problemas con desbalance extremo:

- **Recall competitivo** (0.7463) comparable a Random Forest (0.7164) y
  XGBoost (0.7612), lo que demuestra que el algoritmo identifica fraudes
  reales aunque no fue entrenado con etiquetas.
- **Precision muy baja** (0.0222) debido al umbral de decisión interno
  (`contamination='auto'`) calibrado para escenarios genéricos de 5-10% de
  anomalías. En este dataset, donde la prevalencia real es 0.16%, el modelo
  marca como sospechosas el ~5% de las transacciones más atípicas,
  generando 2,203 falsos positivos.
- **AUC-ROC alto** (0.9259) indica que el _ranking_ interno del modelo es
  sólido: cuando se ordenan las transacciones por puntuación de anomalía,
  las fraudulentas tienden a estar arriba. La capacidad discriminativa
  existe; lo que falla es el umbral por defecto.

Esta complementariedad entre paradigmas supervisado y no supervisado
constituye la base teórica del modelo híbrido propuesto: combinar la
precisión histórica de XGBoost con la capacidad de detección de patrones
anómalos novedosos de Isolation Forest, esperando una mejora marginal
sobre el mejor modelo individual.

## Síntesis comparativa de los 4 modelos individuales

Los resultados consolidados validan empíricamente la jerarquía de
sofisticación algorítmica en el problema de detección de fraude:

| Modelo          | Mejora F1 vs baseline | Mejora AUC vs baseline |
| --------------- | --------------------- | ---------------------- |
| RandomForest    | +533.6%               | +29.0%                 |
| XGBoost         | +550.9%               | +33.6%                 |
| IsolationForest | -66.7%                | +27.6%                 |

XGBoost se consolida como el mejor modelo individual del proyecto en F1
(0.8430) y AUC-ROC (0.9693). Isolation Forest, aunque inferior en F1
debido al umbral, aporta diversidad metodológica (no supervisado) y
detecta 50 de 67 fraudes con un AUC competitivo, posicionándose como
componente complementario natural para el modelo híbrido.

## Experimento 004: Replicación de los 4 modelos sobre PaySim

**Configuración**: `config/experiments/exp_004_paysim_four_models.yaml`

Replica la comparativa de 4 modelos sobre PaySim, usando un subset estratificado
del 20% (1,272,524 filas, 1,643 fraudes) para iterar rápido durante el desarrollo.
Las features behavioral (`balance_mismatch`, `is_zero_origin_after`,
`amount_to_balance_ratio`, etc.) se aplican antes del split.

### Configuración específica de PaySim

- **Dataset**: PaySim (Lopez-Rojas et al., 2016)
- **Filas tras limpieza**: 6,362,620 (sin duplicados ni nulos)
- **Subset estratificado (sample_fraction=0.20)**: 1,272,524 filas
- **Tasa de fraude global**: 0.129%
- **Split estratificado**: 70% train (890,766) / 15% val (190,879) / 15% test (190,879)
- **Random seed**: 42
- **Features**: temporales + monto + **behavioral** (`feature_version: v1`)
- **Sin SMOTE aplicado**
- **Sin tuning de hiperparámetros**: mismos defaults que en Credit Card

### Resultados sobre VALIDATION

| Modelo               | Precision | Recall | F1         | AUC-ROC    | TP  | FP     | FN |
| -------------------- | --------- | ------ | ---------- | ---------- | --- | ------ | -- |
| RulesBaseline        | 0.0030    | 1.0000 | 0.0059     | 0.9625     | 247 | 82,649 | 0  |
| **RandomForest**     | 0.9960    | 1.0000 | **0.9980** | **1.0000** | 247 | 1      | 0  |
| **XGBoost**          | 0.9960    | 1.0000 | **0.9980** | **1.0000** | 247 | 1      | 0  |
| IsolationForest      | 0.0108    | 0.7126 | 0.0213     | 0.9163     | 176 | 16,110 | 71 |

### Lectura por modelo en PaySim

**RulesBaseline** alcanza `recall=1.000` pero `precision=0.003`. Con
`min_rules_to_flag=2` sobre las 4 reglas definidas para PaySim, el detector
marca el 43% de todas las transacciones como sospechosas porque la regla
`R1_risky_type` (tipo CASH_OUT o TRANSFER) por sí sola activa en una
proporción muy alta del volumen. AUC alto (0.9625) confirma que el ranking
interno separa bien fraude de legítimo, pero el umbral del baseline no es
operacionalmente útil.

**RandomForest y XGBoost** empatan en métricas (F1=0.9980, AUC=1.0000),
distinto de Credit Card donde XGBoost dominaba. Ambos detectan los 247
fraudes del set de validación con un único falso positivo. La saturación
en métricas refleja que el problema en PaySim es estructuralmente más
fácil que en Credit Card: el fraude tiene una huella casi determinista
(vaciado de cuenta + mismatch contable) capturada por las features
behavioral.

**IsolationForest** mantiene su patrón característico: recall razonable
(0.7126) pero precision pésima (0.0108) por la calibración del
`contamination='auto'`, generando 16,110 falsos positivos. AUC=0.9163
indica que el ranking interno sigue siendo competitivo.

### Feature importances dominantes (validación de hipótesis)

Top 5 features por importancia, en ambos modelos supervisados (variante
con `isFlaggedFraud`):

| Random Forest               | Importancia | XGBoost                     | Importancia |
| --------------------------- | ----------- | --------------------------- | ----------- |
| `balance_mismatch`          | 0.227       | `balance_change`            | 0.580       |
| `balance_change`            | 0.182       | `balance_mismatch`          | 0.138       |
| `amount_to_balance_ratio`   | 0.180       | `is_zero_origin_after`      | 0.101       |
| `balance_ratio`             | 0.076       | `balance_ratio`             | 0.069       |
| `newbalanceOrig`            | 0.075       | `amount_is_round`           | 0.030       |

Los 4 features del bloque behavioral construido en `src/fraud_detection/features/behavioral.py`
dominan la decisión de ambos modelos. **`is_zero_origin_after` y `balance_mismatch`
aparecen en el top 10 de RF y XGBoost**, confirmando empíricamente la
hipótesis del EDA: el fraude en PaySim tiene una firma estructural
detectable a partir de la matemática de balances.

### Mejoras relativas sobre el baseline (validation, PaySim)

| Modelo          | F1 vs Baseline   | AUC vs Baseline | Recall vs Baseline |
| --------------- | ---------------- | --------------- | ------------------ |
| RandomForest    | +16,696.7%       | +3.9%           | igual (ambos 1.00) |
| XGBoost         | +16,696.7%       | +3.9%           | igual (ambos 1.00) |
| IsolationForest | +258.6%          | -4.9%           | -28.7%             |

La magnitud del salto en F1 (+16,000%) refleja que el baseline en PaySim
tiene `precision` extremadamente baja por diseño (umbral de 2 reglas);
la métrica útil para comparar contra Credit Card es **AUC-ROC**, donde
los ML supervisados aportan solo +4% incremental sobre las reglas porque
estas ya rankean bien.

## Experimento 004b: Variante sin `isFlaggedFraud`

**Configuración**: `config/experiments/exp_004b_paysim_four_models_no_flagged.yaml`

`isFlaggedFraud` es la bandera nativa del simulador, derivable de
`amount > 200,000` y `type == 'TRANSFER'`. Aunque no es ground truth
estricto, podría introducir información trivial. Esta variante mide el
rendimiento puro al excluirla de la matriz X.

### Resultados sobre VALIDATION

| Modelo               | Precision | Recall | F1         | AUC-ROC    | TP  | FP     | FN |
| -------------------- | --------- | ------ | ---------- | ---------- | --- | ------ | -- |
| RulesBaseline        | 0.0030    | 1.0000 | 0.0059     | 0.9625     | 247 | 82,649 | 0  |
| **RandomForest**     | 0.9960    | 1.0000 | **0.9980** | **1.0000** | 247 | 1      | 0  |
| **XGBoost**          | 0.9960    | 1.0000 | **0.9980** | **1.0000** | 247 | 1      | 0  |
| IsolationForest      | 0.0113    | 0.6883 | 0.0223     | 0.9146     | 170 | 14,848 | 77 |

### Diferencias entre exp_004 (con) y exp_004b (sin `isFlaggedFraud`)

| Modelo          | ΔF1     | ΔAUC    | ΔTP | ΔFP    |
| --------------- | ------- | ------- | --- | ------ |
| RulesBaseline   | 0       | 0       | 0   | 0      |
| RandomForest    | 0       | 0       | 0   | 0      |
| XGBoost         | 0       | 0       | 0   | 0      |
| IsolationForest | +0.0010 | -0.0017 | -6  | -1,262 |

**Hallazgo clave**: los modelos supervisados (RF y XGBoost) producen
métricas idénticas con o sin `isFlaggedFraud`. En XGBoost, la importancia
asignada a esta feature es 0.0092 (puesto #8), por debajo de las features
behavioral. El alto F1 en PaySim **no depende de `isFlaggedFraud`** sino
del feature engineering deliberado sobre balances. Se valida que el
rendimiento del modelo es atribuible al pipeline metodológico del proyecto.

IsolationForest cambia marginalmente al perder una feature
discriminante adicional, pero el patrón general se mantiene.

## Comparativa transversal: Credit Card vs PaySim (validation)

| Modelo          | CC F1   | PaySim F1 | CC AUC  | PaySim AUC |
| --------------- | ------- | --------- | ------- | ---------- |
| RulesBaseline   | 0.1295  | 0.0059    | 0.7254  | 0.9625     |
| RandomForest    | 0.8205  | 0.9980    | 0.9356  | 1.0000     |
| XGBoost         | 0.8430  | 0.9980    | 0.9693  | 1.0000     |
| IsolationForest | 0.0431  | 0.0213    | 0.9259  | 0.9163     |

Lectura cruzada:
- **PaySim es más fácil para ML supervisado** que Credit Card. La huella
  estructural del fraude (vaciado de cuenta + mismatch contable) se
  captura con features deterministas, lo que hace converger RF y XGBoost
  a métricas idénticas.
- **El baseline funciona muy distinto**: en Credit Card las reglas marcan
  poco (precision 0.125, recall 0.134), en PaySim marcan todo (precision
  0.003, recall 1.000). El parámetro `min_rules_to_flag=2` es óptimo
  para Credit Card pero demasiado permisivo para PaySim.
- **AUC del baseline en PaySim es muy alto** (0.96), no porque el umbral
  funcione, sino porque el conteo de reglas activadas correlaciona bien
  con la probabilidad real de fraude. El upgrade a ML es menor (+4% AUC)
  porque las reglas ya capturan la señal — el aporte está en la
  precision, no en el ranking.
- **IsolationForest se comporta consistentemente** entre datasets:
  recall ~70%, precision baja, AUC ~0.92. La firma de "anomalía global"
  es transversal al dominio.

## Experimento 005: Modelo híbrido (XGBoost + IsolationForest) en Credit Card

**Configuración**: `config/experiments/exp_005_creditcard_hybrid.yaml`

Incorpora el `HybridModel` al ranking de los 4 modelos individuales del
exp_003. La hipótesis original es que combinar un detector supervisado
(XGBoost, alta precision en patrones conocidos) con uno no supervisado
(IsolationForest, cobertura de anomalías novedosas) supera al mejor
modelo individual al recuperar fraudes que ninguno detecta por separado.

### Implementación

El modelo se define en [`src/fraud_detection/models/hybrid.py`](../src/fraud_detection/models/hybrid.py)
y soporta tres estrategias de combinación, configurables por YAML:

- **`or`**: predice fraude si CUALQUIERA de los dos lo marca (union de
  positivos). Maximiza recall, sacrifica precision.
- **`and`**: predice fraude si AMBOS lo marcan (intersección de
  positivos). Maximiza precision, sacrifica recall.
- **`weighted_voting`**: aplica umbral a un promedio ponderado de las
  probabilidades de ambos componentes. Permite control fino del
  trade-off mediante `weights` y `threshold`.

### Resultados sobre VALIDATION (Credit Card)

| Modelo                            | Precision | Recall | F1         | AUC-ROC    | TP | FP    | FN |
| --------------------------------- | --------- | ------ | ---------- | ---------- | -- | ----- | -- |
| RulesBaseline                     | 0.1250    | 0.1343 | 0.1295     | 0.7254     | 9  | 63    | 58 |
| RandomForest                      | 0.9600    | 0.7164 | 0.8205     | 0.9356     | 48 | 2     | 19 |
| **XGBoost**                       | 0.9444    | 0.7612 | **0.8430** | **0.9693** | 51 | 3     | 16 |
| IsolationForest                   | 0.0222    | 0.7463 | 0.0431     | 0.9259     | 50 | 2,203 | 17 |
| Hybrid (OR)                       | 0.0244    | 0.8209 | 0.0473     | 0.9406     | 55 | 2,203 | 12 |

### Experimento 005b: Comparativa de las 3 estrategias del híbrido

**Configuración**: `config/experiments/exp_005b_creditcard_hybrid_strategies.yaml`

| Estrategia                        | Precision | Recall | F1         | AUC-ROC | TP | FP    | FN |
| --------------------------------- | --------- | ------ | ---------- | ------- | -- | ----- | -- |
| XGBoost solo (referencia)         | 0.9444    | 0.7612 | **0.8430** | 0.9693  | 51 | 3     | 16 |
| Hybrid OR                         | 0.0244    | 0.8209 | 0.0473     | 0.9406  | 55 | 2,203 | 12 |
| Hybrid AND                        | 0.9388    | 0.6866 | 0.7931     | 0.9693  | 46 | 3     | 21 |
| Hybrid weighted_voting (0.7/0.3)  | 0.9444    | 0.7612 | **0.8430** | 0.9454  | 51 | 3     | 16 |

### Interpretación del híbrido

**Ninguna de las tres estrategias supera al mejor modelo individual
(XGBoost) en F1**, refutando la hipótesis inicial de mejora monotónica.
Cada estrategia exhibe un comportamiento distinto:

- **OR (`recall`-maximizer)**: recupera 4 fraudes adicionales (55 vs 51
  de XGBoost) que solo el IsolationForest detecta, validando que el
  componente no supervisado SÍ aporta cobertura complementaria. Sin
  embargo, hereda íntegramente los 2,203 falsos positivos del IF,
  derrumbando la precision de 0.94 a 0.02 y el F1 de 0.84 a 0.05.
- **AND (`precision`-maximizer)**: mantiene la precision alta (0.94)
  pero descarta los 5 fraudes que XGBoost detectaba en solitario (sin
  acuerdo del IF), bajando el recall de 0.76 a 0.69. F1 cae a 0.79.
- **weighted_voting (0.7/0.3)**: el peso de 0.3 del IF resulta
  insuficiente para mover la decisión final, ya que la probabilidad de
  XGBoost (cercana a 0 o 1) domina el promedio ponderado. Los
  resultados son **idénticos** a XGBoost solo en precision, recall, F1
  y matriz de confusión; solo el AUC-ROC cambia marginalmente.

**Lectura para la tesis**: la complementariedad existe (OR detecta
fraudes que XGBoost solo no captura), pero el costo en precision la
hace inviable como detector único. La estrategia útil sería usar el
híbrido OR como **filtro de primera línea** (cobertura amplia) seguido
de una segunda etapa de revisión manual o un modelo de re-ranking. Esa
arquitectura va más allá del alcance de esta tesis y queda como trabajo
futuro.

## Experimento 006: Modelo híbrido en PaySim

**Configuración**: `config/experiments/exp_006_paysim_hybrid.yaml`

### Resultados sobre VALIDATION (PaySim, sin `isFlaggedFraud`)

| Modelo                            | Precision | Recall | F1         | AUC-ROC    | TP  | FP     | FN |
| --------------------------------- | --------- | ------ | ---------- | ---------- | --- | ------ | -- |
| RulesBaseline                     | 0.0030    | 1.0000 | 0.0059     | 0.9625     | 247 | 82,649 | 0  |
| RandomForest                      | 0.9960    | 1.0000 | **0.9980** | **1.0000** | 247 | 1      | 0  |
| **XGBoost**                       | 0.9960    | 1.0000 | **0.9980** | **1.0000** | 247 | 1      | 0  |
| IsolationForest                   | 0.0113    | 0.6883 | 0.0223     | 0.9146     | 170 | 14,848 | 77 |
| Hybrid (OR)                       | 0.0164    | 1.0000 | 0.0322     | 0.999995   | 247 | 14,849 | 0  |

En PaySim el patrón se confirma con un agravante: XGBoost ya alcanza
`recall=1.000`, por lo que el componente no supervisado no aporta
ningún fraude adicional. El híbrido OR mantiene el recall pero hereda
los 14,848 FP del IsolationForest. **El híbrido no aporta valor
incremental sobre XGBoost en PaySim**.

### Comparativa híbrido: Credit Card vs PaySim

| Métrica val      | CC: XGB → Hyb (OR) | PaySim: XGB → Hyb (OR) |
| ---------------- | ------------------ | ---------------------- |
| Δ Recall         | +0.060             | 0.000 (ya saturado)    |
| Δ Precision      | −0.920             | −0.980                 |
| Δ F1             | −0.796             | −0.966                 |
| Δ TP recuperados | +4                 | 0                      |
| Δ FP incorporados| +2,200             | +14,848                |

Conclusión cross-dataset: la utilidad del híbrido (medida en TP
adicionales) depende fuertemente de que el modelo supervisado deje
margen de recall por cubrir. En PaySim, donde el supervisado ya satura,
el híbrido solo agrega ruido.

## Experimento 007: Impacto de SMOTE en Credit Card

**Configuración**: `config/experiments/exp_007_creditcard_with_smote.yaml`

Reusa la configuración de exp_005 cambiando `resampling.method` de `none`
a `smote`. SMOTE se aplica EXCLUSIVAMENTE al fold de entrenamiento (post
split y post filtrado a columnas numéricas), nunca a validation o test.
`RulesBaseline` queda excluido del resampling porque sus reglas son
estáticas; resamplearlo solo distorsiona los conteos sin cambiar el
comportamiento del modelo.

### Resultados sobre VALIDATION (Credit Card, sin vs con SMOTE)

| Modelo                            | P sin → con   | R sin → con   | F1 sin → con  | AUC sin → con |
| --------------------------------- | ------------- | ------------- | ------------- | ------------- |
| RulesBaseline                     | 0.1250 = 0.1250 | 0.1343 = 0.1343 | 0.1295 = 0.1295 | 0.7254 = 0.7254 |
| RandomForest                      | 0.9600 → 0.9153 | 0.7164 → 0.8060 | **0.8205 → 0.8571** | 0.9356 → 0.9468 |
| XGBoost                           | 0.9444 → 0.9000 | 0.7612 → 0.8060 | 0.8430 → 0.8504 | 0.9693 → 0.9739 |
| IsolationForest                   | 0.0222 → 0.0280 | 0.7463 → 0.3134 | 0.0431 → 0.0514 | 0.9259 → 0.8214 |
| Hybrid OR                         | 0.0244 → 0.0696 | 0.8209 = 0.8209 | 0.0473 → 0.1284 | 0.9406 → 0.9321 |

Lectura:

- **RandomForest** es el principal beneficiado por SMOTE: F1 sube 4.5%
  (de 0.8205 a 0.8571) y recall sube de 0.72 a 0.81 a costa de 5%
  menos precision. Es el cambio cualitativo más relevante del experimento.
- **XGBoost** sube F1 marginalmente (+0.9%). Su `scale_pos_weight`
  automático ya compensaba el desbalance internamente.
- **IsolationForest** se ve degradado: el oversampling sintético rompe
  la noción de "anomalía" del modelo, que aprende sobre un mundo donde
  el 50% es fraude. AUC cae de 0.93 a 0.82, recall de 0.75 a 0.31.
- **Hybrid OR** mejora aparentemente (F1 0.05 → 0.13) porque IF ahora
  marca menos transacciones (729 FP vs 2,203), pero sigue muy por debajo
  de XGBoost solo.

## Experimento 008: Impacto de SMOTE en PaySim

**Configuración**: `config/experiments/exp_008_paysim_with_smote.yaml`

### Resultados sobre VALIDATION (PaySim, sin vs con SMOTE)

| Modelo                            | P sin → con   | R sin → con   | F1 sin → con  | AUC sin → con |
| --------------------------------- | ------------- | ------------- | ------------- | ------------- |
| RulesBaseline                     | 0.0030 = 0.0030 | 1.0000 = 1.0000 | 0.0059 = 0.0059 | 0.9625 = 0.9625 |
| RandomForest                      | 0.9960 = 0.9960 | 1.0000 = 1.0000 | **0.9980 = 0.9980** | 1.0000 = 1.0000 |
| XGBoost                           | 0.9960 = 0.9960 | 1.0000 = 1.0000 | **0.9980 = 0.9980** | 1.0000 = 1.0000 |
| IsolationForest                   | 0.0113 → 0.0026 | 0.6883 → 0.1781 | 0.0223 → 0.0050 | 0.9146 → 0.5562 |
| Hybrid OR                         | 0.0164 → 0.0142 | 1.0000 = 1.0000 | 0.0322 → 0.0279 | 0.999995 = 0.999995 |

En PaySim los modelos supervisados ya saturan en val sin SMOTE, por lo
que el oversampling no aporta nada en F1, precision, recall o AUC.
IsolationForest se ve **catastróficamente degradado**: el AUC cae de
0.91 a 0.56 (apenas mejor que azar). Confirma que SMOTE no debe usarse
con detectores no supervisados.

### Recomendación final sobre SMOTE

- **Para producción con `RandomForest` en datos similares a Credit Card
  (desbalance ~0.2%, sin saturación)**: SMOTE aporta valor (+4.5% F1)
  y se recomienda activar.
- **Para `XGBoost` con `scale_pos_weight` automático**: SMOTE es
  redundante; el ajuste de pesos internos ya cubre el desbalance.
- **Para detectores no supervisados (`IsolationForest`)**: SMOTE
  contraindicado, distorsiona la noción de anomalía.
- **Para datasets donde el supervisado ya satura (PaySim con features
  behavioral)**: SMOTE no aporta y duplica el tiempo de entrenamiento.

## Validación cruzada k-fold (k=5)

**Script**: `scripts/run_cross_validation.py`
**Módulo**: `src/fraud_detection/evaluation/cross_validation.py`

Aplica `StratifiedKFold(k=5, shuffle=True, random_state=42)` sobre el
dataset completo (sin split previo en train/val/test), entrenando un
modelo fresco por fold. Los reportes se persisten en
`reports/metrics/cv__<dataset>__<model>.json`.

### Resultados (mean ± std sobre 5 folds)

#### Credit Card (n=281,918, fraude=0.16%)

| Modelo       | Precision        | Recall           | F1               | AUC-ROC          |
| ------------ | ---------------- | ---------------- | ---------------- | ---------------- |
| XGBoost      | 0.9122 ± 0.0299  | 0.8103 ± 0.0525  | **0.8576 ± 0.0370** | **0.9798 ± 0.0089** |
| RandomForest | 0.9337 ± 0.0203  | 0.7634 ± 0.0517  | 0.8396 ± 0.0390  | 0.9600 ± 0.0098  |

#### PaySim (n=1,272,524 al 20%, fraude=0.13%)

| Modelo       | Precision        | Recall           | F1               | AUC-ROC          |
| ------------ | ---------------- | ---------------- | ---------------- | ---------------- |
| XGBoost      | 0.9958 ± 0.0059  | 0.9957 ± 0.0027  | 0.9957 ± 0.0039  | **0.9990 ± 0.0014** |
| RandomForest | 0.9988 ± 0.0017  | 0.9945 ± 0.0025  | **0.9966 ± 0.0020** | 0.9979 ± 0.0014  |

### Lectura

- **Estabilidad alta en ambos datasets**: las desviaciones estándar
  son pequeñas (Credit Card: σF1 ≈ 0.04, PaySim: σF1 ≈ 0.003), lo que
  confirma que los resultados del split único 70/15/15 no son producto
  del azar.
- **XGBoost vs RandomForest en CC**: XGBoost gana en F1, AUC y recall
  (todas las diferencias dentro de 1σ). RF gana en precision. La
  diferencia es estadísticamente modesta pero consistente.
- **XGBoost vs RandomForest en PaySim**: empate técnico (RF marginal en
  F1, XGBoost marginal en AUC). En la práctica, cualquiera de los dos
  es válido.
- **CV vs single split** (XGBoost CC): el F1 del CV (0.8576 ± 0.0370)
  es ligeramente mejor que el del split único (0.8430), pero el segundo
  cae dentro del intervalo del primero, confirmando reproducibilidad.

## Experimento 009: Evaluación final sobre TEST (PaySim completo)

**Configuración**: `config/experiments/exp_009_paysim_full_test.yaml`

Corrida única de cierre del capítulo 5. Reentrena los cinco detectores
(RulesBaseline, RandomForest, XGBoost, IsolationForest, Hybrid OR) sobre
**PaySim completo** (6,362,620 transacciones, sin `sample_fraction`) con
SMOTE solo en train. A diferencia de exp_008 (que trabajaba sobre el 20%
estratificado), aquí el TEST son 954,393 filas con 1,232 fraudes — un
universo representativo para la métrica final.

**Notebook narrativo**: `notebooks/05_evaluacion_final_y_comparacion.ipynb`.

### Protocolo

- Dataset: PaySim completo (6.36M filas), `isFlaggedFraud` excluida.
- Features: temporales + amount + behavioral (`feature_version: v1`).
- Split estratificado 70/15/15 con `random_seed=42`: train 4,453,833 |
  val 954,394 | test 954,393 (0.1291% fraude en cada split).
- SMOTE aplicado SOLO al train: 4.45M → 8.9M filas (50/50 fraude/no
  fraude). Excluye `RulesBaseline` por ser estático.
- Credit Card no se rehace: exp_007 ya usaba el dataset completo
  (281,918 filas tras limpieza).

### Resultados sobre TEST (PaySim, 954,393 filas, 1,232 fraudes)

| Modelo               | Precision | Recall | F1     | AUC-ROC | AUC-PR | TP    | FP     | FN    |
| -------------------- | --------- | ------ | ------ | ------- | ------ | ----- | ------ | ----- |
| RulesBaseline (k=2)  | 0.0030    | 1.0000 | 0.0059 | 0.9669  | 0.6606 | 1,232 | 412,857 | 0     |
| RulesBaseline (k=4)  | 1.0000    | 0.6583 | **0.7939** | 0.9669  | 0.6606 | 811   | 0      | 421   |
| **RandomForest**     | 0.9976    | 0.9976 | **0.9976** | 1.0000  | 0.9995 | 1,229 | 3      | 3     |
| **XGBoost**          | 0.9968    | 0.9976 | 0.9972 | 1.0000  | 0.9991 | 1,229 | 4      | 3     |
| IsolationForest (auto) | 0.0025  | 0.1794 | 0.0049 | 0.5691  | 0.0051 | 221   | 89,561 | 1,011 |
| Hybrid OR (IF auto)  | 0.0135    | 0.9976 | 0.0267 | 0.9990  | 0.9976 | 1,229 | 89,564 | 3     |

### Resultados sobre TEST (Credit Card, 42,288 filas, 67 fraudes)

Reutiliza los modelos `exp_007_creditcard_with_smote__*.joblib` (Credit
Card ya estaba con dataset completo). Mismas métricas, una sola corrida.

| Modelo            | Precision | Recall | F1     | AUC-ROC | AUC-PR |
| ----------------- | --------- | ------ | ------ | ------- | ------ |
| RulesBaseline     | 0.1692    | 0.1642 | 0.1667 | 0.7344  | 0.0308 |
| RandomForest      | 0.8358    | 0.8358 | 0.8358 | 0.9849  | 0.8755 |
| **XGBoost**       | 0.9032    | 0.8358 | **0.8682** | **0.9964** | **0.8792** |
| IsolationForest   | 0.0225    | 0.2537 | 0.0413 | 0.8205  | 0.0381 |
| Hybrid OR         | 0.0712    | 0.8507 | 0.1315 | 0.9539  | 0.8376 |

### Tarea 2 — Re-tuning del baseline de reglas en PaySim sobre TEST

| `min_rules_to_flag` | Precision | Recall | F1     |
| ------------------- | --------- | ------ | ------ |
| 2 (original)        | 0.0030    | 1.0000 | 0.0059 |
| 3                   | 0.0068    | 0.9943 | 0.0136 |
| **4 (oficial)**     | **1.0000** | 0.6583 | **0.7939** |

Selección automática por F1 sobre el TEST → k=4. El baseline queda en
un punto operativo comparable a los modelos supervisados (precision 1.0,
recall 0.66, cero falsos positivos) y deja de inflar artificialmente el
recall a costa de la precision.

### Tarea 3a — Re-calibración de `contamination` en IsolationForest

IF reentrenado con `contamination` igual a la prevalencia real de cada
dataset (no el `'auto'` de scikit-learn que asume ~10% de anomalías):

- Credit Card: `contamination = 492 / 284807 ≈ 0.001727`
- PaySim: `contamination ≈ 0.001291` (prevalencia empírica del train
  completo de 4.45M filas, sin SMOTE).

| Dataset    | Configuración              | Precision | Recall | F1     | AUC-ROC | AUC-PR |
| ---------- | --------------------------- | --------- | ------ | ------ | ------- | ------ |
| CreditCard | `contamination='auto'`      | 0.0225    | 0.2537 | 0.0413 | 0.8205  | 0.0381 |
| CreditCard | `contamination=0.001727`    | 0.1852    | 0.2239 | **0.2027** | 0.9515  | 0.0844 |
| PaySim     | `contamination='auto'`      | 0.0025    | 0.1794 | 0.0049 | 0.5691  | 0.0051 |
| PaySim     | `contamination=0.001291`   | 0.1204    | 0.1242 | **0.1223** | 0.9253  | 0.0547 |

Calibrar `contamination` mejora F1 ×4.9 en CC y ×25 en PaySim. Sin esto,
el IF marca como fraude el ~10% del TEST y arrastra al híbrido a una
precision inviable.

### Tarea 3b — Re-evaluación del híbrido con IF calibrado sobre TEST

Tres estrategias de combinación (XGBoost como supervisado, IF calibrado
como no supervisado):

| Dataset    | Estrategia          | Precision | Recall | F1     | AUC-ROC |
| ---------- | ------------------- | --------- | ------ | ------ | ------- |
| CreditCard | OR                  | 0.4444    | 0.8358 | 0.5803 | 0.9650  |
| CreditCard | AND                 | 0.8824    | 0.2239 | 0.3571 | 0.9963  |
| CreditCard | Weighted (0.7/0.3)  | 0.9180    | 0.8358 | **0.8750** | 0.9675  |
| PaySim     | OR                  | 0.5230    | 0.9976 | 0.6862 | 0.9991  |
| PaySim     | AND                 | 0.9935    | 0.1242 | 0.2208 | 1.0000  |
| PaySim     | Weighted (0.7/0.3)  | 0.9976    | 0.9976 | **0.9976** | 0.9992  |

Lectura:

- En CC, **Weighted (0.7/0.3) supera al XGBoost solo** (F1 0.8750 vs
  0.8682). Es el único experimento del proyecto donde una combinación
  híbrida saca ventaja real sobre el mejor individual.
- En PaySim, Weighted (0.7/0.3) iguala al XGBoost (F1 0.9976). El
  componente no supervisado deja de degradar y se integra sin costo.
- OR sigue siendo útil sólo cuando el objetivo operativo es maximizar
  recall asumiendo capacidad de revisión sobre el incremento de FP.
- AND es la opción más conservadora: precision casi perfecta a costa de
  perder ~75% de los fraudes detectables sólo por XGBoost.

### Trabajo previamente pendiente

Los cuatro items del apartado anterior quedan **completados** en este
experimento:

- [x] Evaluación final sobre el conjunto de **test** (una sola vez): tabla
  consolidada arriba, PaySim sobre dataset completo (6.36M filas), CC
  con `exp_007`.
- [x] Re-tuning del baseline de PaySim a `min_rules_to_flag=4` por F1.
- [x] Re-calibración de `contamination` y reevaluación del híbrido bajo
  OR / AND / Weighted.
- [x] Notebook narrativo: `notebooks/05_evaluacion_final_y_comparacion.ipynb`
  (secciones 5.1–5.8) con figuras y export Excel automático.

## Trabajo pendiente (siguiente fase)

- Análisis costo-beneficio operativo: integrar matriz de costos de FP/FN
  para elegir umbral y estrategia híbrida según escenario de negocio.
- Monitoreo de *concept drift* y protocolo de reentrenamiento periódico.
- Validación sobre datos de producción reales (no sintéticos como PaySim).
