# Fraud Detection ITM

Detección de fraude y phishing en transacciones financieras digitales mediante técnicas de aprendizaje automático.

**Proyecto de Investigación Tecnológica**
Instituto Tecnológico de Morelia — Departamento de Sistemas y Computación

---

## Autores

- Cesar Enrique Diaz Maldonado
- José Antonio Medina Ayala

## Asesor

- Rubén Lara Bárcenas

## Resumen

Sistema experimental que aplica técnicas supervisadas y no supervisadas de aprendizaje automático para detectar transacciones financieras fraudulentas y patrones de phishing. La investigación contrasta un modelo híbrido de ML contra un sistema basado en reglas (línea base experimental) utilizando métricas estandarizadas (F1-score, AUC).

## Hipótesis

Un modelo de aprendizaje automático que integra enfoques supervisados y no supervisados para el análisis conductual de transacciones financieras alcanzará un rendimiento superior al de un sistema basado en reglas, evaluado mediante F1-score y AUC.

## Stack tecnológico

- **Lenguaje**: Python 3.11+
- **ML**: scikit-learn, XGBoost, imbalanced-learn
- **Datos**: Pandas, NumPy, PyArrow
- **Base de datos**: PostgreSQL 16
- **Visualización**: Matplotlib, Seaborn
- **Configuración**: PyYAML, python-dotenv
- **Testing**: Pytest, coverage
- **Calidad**: Ruff, Mypy
- **Entorno**: virtualenv, VS Code, GitHub

## Metodología

El desarrollo sigue **CRISP-DM** (Cross-Industry Standard Process for Data Mining).

## Datasets

- **Credit Card Fraud Detection** (Dal Pozzolo et al., 2015) — Kaggle
- **PaySim** (Lopez-Rojas & Axelsson, 2016) — Kaggle

## Estructura del proyecto

\`\`\`
fraud-detection-itm/
├── config/ Configuración YAML (modelos, paths, hiperparámetros)
├── data/ Datasets (gitignored)
├── db/migrations/ Migraciones SQL
├── docs/ Documentación técnica
├── models/ Modelos serializados (gitignored)
├── notebooks/ Notebooks de EDA y análisis
├── pipelines/ Scripts de orquestación end-to-end
├── reports/ Métricas, figuras y logs de experimentos
├── scripts/ Scripts utilitarios
├── src/ Código fuente (paquete instalable)
├── tests/ Tests unitarios y de integración
├── pyproject.toml Configuración del paquete
├── Makefile Comandos comunes (Linux/macOS)
└── make.ps1 Comandos comunes (Windows)
\`\`\`

## Setup rápido

\`\`\`powershell

# 1. Clonar el repositorio

git clone https://github.com/TU_USUARIO/fraud-detection-itm.git
cd fraud-detection-itm

# 2. Crear entorno virtual e instalar

.\make.ps1 setup

# 3. Configurar variables de entorno

copy .env.example .env

# Editar .env con tus credenciales de PostgreSQL

# 4. Crear base de datos y aplicar migraciones

.\make.ps1 db-migrate

# 5. Ejecutar tests

.\make.ps1 test
\`\`\`

Documentación detallada de setup en \`docs/reproducibility_guide.md\`.

## Licencia

MIT License. Ver \`LICENSE\`.

## Estado actual del proyecto (2026-05-17)

**Capítulo 5 de la tesis cerrado** — evaluación final sobre el conjunto de
TEST con ambos datasets en su versión completa.

- _Credit Card Fraud Detection_: 281,918 transacciones tras limpieza, TEST
  de 42,288 filas (67 fraudes).
- _PaySim_: dataset COMPLETO de 6,362,620 transacciones (experimento
  `exp_009_paysim_full_test`), TEST de 954,393 filas (1,232 fraudes).

**Mejor modelo por dataset (sobre TEST, una sola corrida):**

| Dataset    | Modelo                    | F1     | AUC-ROC | AUC-PR |
| ---------- | ------------------------- | ------ | ------- | ------ |
| CreditCard | Hybrid Weighted (0.7/0.3) | 0.8750 | 0.9675  | 0.7917 |
| PaySim     | RandomForest / Weighted   | 0.9976 | 1.0000  | 0.9995 |

**Entregables del capítulo 5:**

- Notebook narrativo: `notebooks/05_evaluacion_final_y_comparacion.ipynb`
  (secciones 5.1 a 5.8 con figuras y export Excel automático).
- Tablas consolidadas: `reports/tablas_evaluacion_final.xlsx` (6 hojas).
- Figuras: `reports/figures/evaluation/final_test/fig_5_{1..4}.png`.
- Documentación experimental: `docs/baseline_results.md` (exp_001 a exp_009).

**Trabajo cerrado en esta fase:**

- [x] Evaluación final sobre TEST (una sola vez, datasets completos).
- [x] Re-tuning del baseline de PaySim (`min_rules_to_flag=4` por F1).
- [x] Re-calibración de `contamination` en IsolationForest a la prevalencia
      real (F1 ×4.9 en CC, ×25 en PaySim).
- [x] Re-evaluación del híbrido bajo OR / AND / Weighted con IF calibrado.

**Siguiente fase:**

- Análisis costo-beneficio operativo (matriz de costos FP/FN para elegir
  umbral y estrategia híbrida por escenario).
- Monitoreo de _concept drift_ y protocolo de reentrenamiento periódico.
- Validación sobre datos de producción reales.
