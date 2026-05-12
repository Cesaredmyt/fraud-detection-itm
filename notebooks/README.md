# Notebooks

Análisis exploratorio y narrativo. Cada notebook responde preguntas concretas y exporta sus figuras a `reports/figures/`.

## Convenciones

- Numerados secuencialmente (`01_`, `02_`, ...).
- Los outputs (`.ipynb` con imágenes embebidas) se limpian antes de cada commit:

  \`\`\`powershell
  jupyter nbconvert --clear-output --inplace notebooks/NN\_\*.ipynb
  \`\`\`

- El kernel a usar es `Python (fraud-detection-itm)` (registrado con `python -m ipykernel install --user --name fraud-detection-itm`).
- Los notebooks importan desde `fraud_detection.*`. **No contienen lógica reusable**, solo orquestación y análisis.

## Notebooks disponibles

| #   | Notebook                                   | Fase CRISP-DM      | Estado    |
| --- | ------------------------------------------ | ------------------ | --------- |
| 01  | `01_eda_creditcard.ipynb`                  | Data Understanding | ✅        |
| 02  | `02_eda_paysim.ipynb`                      | Data Understanding | Pendiente |
| 03  | `03_feature_engineering_exploration.ipynb` | Data Preparation   | Pendiente |
| 04  | `04_baseline_rules_analysis.ipynb`         | Modeling           | Pendiente |
| 05  | `05_supervised_models_comparison.ipynb`    | Modeling           | Pendiente |
| 06  | `06_unsupervised_models.ipynb`             | Modeling           | Pendiente |
| 07  | `07_hybrid_model_evaluation.ipynb`         | Evaluation         | Pendiente |
| 08  | `08_final_results_thesis.ipynb`            | Evaluation         | Pendiente |
