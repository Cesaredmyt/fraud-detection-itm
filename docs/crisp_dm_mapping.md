# Trazabilidad CRISP-DM ↔ código

| Fase CRISP-DM          | Módulo / Notebook                                      | Estado         |
| ---------------------- | ------------------------------------------------------ | -------------- |
| Business Understanding | `docs/architecture.md`, protocolo                      | ✅             |
| Data Understanding     | `notebooks/01_eda_creditcard.ipynb`                    | ✅ Credit Card |
| Data Understanding     | `notebooks/02_eda_paysim.ipynb`                        | ✅ PaySim      |
| Data Preparation       | `src/fraud_detection/data/loaders.py`, `validators.py` | ✅ Loaders     |
| Data Preparation       | `src/fraud_detection/data/cleaners.py`                 | Pendiente      |
| Data Preparation       | `src/fraud_detection/features/*`                       | Pendiente      |
| Modeling               | `src/fraud_detection/models/rules_baseline.py`         | Pendiente      |
| Modeling               | `src/fraud_detection/models/random_forest.py`          | Pendiente      |
| Modeling               | `src/fraud_detection/models/xgboost_model.py`          | Pendiente      |
| Modeling               | `src/fraud_detection/models/isolation_forest.py`       | Pendiente      |
| Modeling               | `src/fraud_detection/models/hybrid.py`                 | Pendiente      |
| Evaluation             | `src/fraud_detection/evaluation/*`                     | Pendiente      |
| Deployment             | N/A (proyecto académico, no producción)                | N/A            |
