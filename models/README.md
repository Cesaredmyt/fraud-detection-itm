# Modelos serializados

Los modelos entrenados no se versionan (gitignored). Se regeneran ejecutando los pipelines de entrenamiento.

## Formato

- Serialización: `joblib`.
- Naming: `<dataset>__<model_name>__<feature_version>__<timestamp>.joblib`.
- Ejemplo: `creditcard__random_forest__v1__20260315_142233.joblib`.

## Reproducción

Los hiperparámetros y semillas de cada modelo están registrados en la tabla `experiments` de PostgreSQL.