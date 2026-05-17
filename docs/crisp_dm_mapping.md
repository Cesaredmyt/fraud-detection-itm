# Trazabilidad CRISP-DM ↔ código

| Fase CRISP-DM          | Módulo / Notebook                                | Estado   |
| ---------------------- | ------------------------------------------------ | -------- |
| Business Understanding | `docs/architecture.md`, protocolo                | ✅       |
| Data Understanding     | `notebooks/01_eda_creditcard.ipynb`              | ✅       |
| Data Understanding     | `notebooks/02_eda_paysim.ipynb`                  | ✅       |
| Data Preparation       | `src/fraud_detection/data/*`                     | ✅       |
| Data Preparation       | `src/fraud_detection/features/*`                 | ✅       |
| Modeling               | `src/fraud_detection/models/rules_baseline.py`   | ✅       |
| Modeling               | `src/fraud_detection/models/random_forest.py`    | ✅       |
| Modeling               | `src/fraud_detection/models/xgboost_model.py`    | Semana 4 |
| Modeling               | `src/fraud_detection/models/isolation_forest.py` | Semana 4 |
| Modeling               | `src/fraud_detection/models/hybrid.py`           | Semana 5 |
| Evaluation             | `src/fraud_detection/evaluation/*`               | ✅       |
| Evaluation             | `pipelines/run_full_pipeline.py`                 | ✅       |
| Deployment             | N/A (proyecto académico)                         | N/A      |
