# Carpeta de datos

Los datasets no se versionan en el repositorio.

## Estructura

- `raw/`: datos originales descargados, **inmutables**.
- `interim/`: datos en proceso de limpieza.
- `processed/`: datos listos para modelado.
- `external/`: referencias externas (CONDUSEF, etc.).

## Cómo obtener los datasets

### Credit Card Fraud Detection
1. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Descargar y descomprimir en `data/raw/creditcard.csv`.

### PaySim
1. https://www.kaggle.com/datasets/ealaxi/paysim1
2. Descargar y descomprimir en `data/raw/paysim.csv`.