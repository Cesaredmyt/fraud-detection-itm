-- =============================================================
-- Migración 001: esquema inicial
-- fraud-detection-itm
-- =============================================================
-- Crea las cinco tablas principales del proyecto:
--   - raw_transactions     : datos originales unificados
--   - clean_transactions   : datos post limpieza
--   - feature_store        : features derivados
--   - experiments          : registro de cada corrida experimental
--   - model_metrics        : métricas por experimento
-- =============================================================

BEGIN;

-- -----------------------------------------------------------
-- Tabla: raw_transactions
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_transactions (
    id                  BIGSERIAL PRIMARY KEY,
    source              VARCHAR(20)      NOT NULL,
    external_id         VARCHAR(64),
    transaction_time    DOUBLE PRECISION,
    amount              NUMERIC(15, 2)   NOT NULL,
    transaction_type    VARCHAR(20),
    origin_account      VARCHAR(64),
    destination_account VARCHAR(64),
    features_json       JSONB,
    is_fraud            BOOLEAN          NOT NULL,
    ingested_at         TIMESTAMPTZ      DEFAULT NOW(),
    CONSTRAINT chk_raw_source CHECK (source IN ('creditcard', 'paysim'))
);

CREATE INDEX IF NOT EXISTS idx_raw_source ON raw_transactions(source);
CREATE INDEX IF NOT EXISTS idx_raw_fraud  ON raw_transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_raw_time   ON raw_transactions(transaction_time);

COMMENT ON TABLE  raw_transactions IS 'Espejo de los CSV originales unificados (Credit Card + PaySim).';
COMMENT ON COLUMN raw_transactions.source IS 'Dataset de origen: creditcard | paysim.';
COMMENT ON COLUMN raw_transactions.features_json IS 'Almacena V1..V28 de Credit Card o campos adicionales en JSONB.';

-- -----------------------------------------------------------
-- Tabla: clean_transactions
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS clean_transactions (
    id                  BIGSERIAL PRIMARY KEY,
    raw_id              BIGINT           REFERENCES raw_transactions(id) ON DELETE CASCADE,
    source              VARCHAR(20)      NOT NULL,
    amount              NUMERIC(15, 2)   NOT NULL,
    transaction_type    VARCHAR(20),
    origin_account      VARCHAR(64),
    destination_account VARCHAR(64),
    transaction_step    INTEGER,
    is_fraud            BOOLEAN          NOT NULL,
    cleaning_version    VARCHAR(10)      NOT NULL,
    processed_at        TIMESTAMPTZ      DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_clean_raw_id  ON clean_transactions(raw_id);
CREATE INDEX IF NOT EXISTS idx_clean_source  ON clean_transactions(source);
CREATE INDEX IF NOT EXISTS idx_clean_version ON clean_transactions(cleaning_version);

COMMENT ON TABLE clean_transactions IS 'Datos post limpieza y normalización, listos para feature engineering.';

-- -----------------------------------------------------------
-- Tabla: feature_store
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS feature_store (
    id                    BIGSERIAL PRIMARY KEY,
    transaction_id        BIGINT           REFERENCES clean_transactions(id) ON DELETE CASCADE,
    feature_version       VARCHAR(10)      NOT NULL,
    -- Conductuales
    txn_frequency_1h      INTEGER,
    txn_frequency_24h     INTEGER,
    avg_amount_user       NUMERIC(15, 2),
    amount_zscore         NUMERIC(10, 4),
    amount_deviation_pct  NUMERIC(10, 4),
    -- Temporales
    hour_of_day           SMALLINT,
    is_night              BOOLEAN,
    time_since_last_txn   NUMERIC(10, 2),
    -- Anomalías por usuario
    user_anomaly_score    NUMERIC(10, 4),
    -- Split
    split                 VARCHAR(10),
    created_at            TIMESTAMPTZ      DEFAULT NOW(),
    CONSTRAINT chk_feature_split CHECK (split IN ('train', 'val', 'test') OR split IS NULL)
);

CREATE INDEX IF NOT EXISTS idx_feature_split   ON feature_store(split);
CREATE INDEX IF NOT EXISTS idx_feature_version ON feature_store(feature_version);
CREATE INDEX IF NOT EXISTS idx_feature_txn_id  ON feature_store(transaction_id);

COMMENT ON TABLE feature_store IS 'Variables derivadas del comportamiento transaccional para entrenamiento de modelos.';

-- -----------------------------------------------------------
-- Tabla: experiments
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiments (
    id                SERIAL PRIMARY KEY,
    experiment_name   VARCHAR(100)     NOT NULL,
    model_type        VARCHAR(50)      NOT NULL,
    dataset           VARCHAR(20)      NOT NULL,
    feature_version   VARCHAR(10)      NOT NULL,
    hyperparameters   JSONB            NOT NULL,
    random_seed       INTEGER          NOT NULL,
    git_commit        VARCHAR(40),
    started_at        TIMESTAMPTZ      DEFAULT NOW(),
    finished_at       TIMESTAMPTZ,
    status            VARCHAR(20)      DEFAULT 'running',
    notes             TEXT,
    CONSTRAINT chk_exp_model_type CHECK (
        model_type IN ('rules', 'random_forest', 'xgboost', 'isolation_forest', 'hybrid')
    ),
    CONSTRAINT chk_exp_dataset CHECK (dataset IN ('creditcard', 'paysim')),
    CONSTRAINT chk_exp_status CHECK (status IN ('running', 'success', 'failed', 'aborted'))
);

CREATE INDEX IF NOT EXISTS idx_exp_model_type ON experiments(model_type);
CREATE INDEX IF NOT EXISTS idx_exp_dataset    ON experiments(dataset);
CREATE INDEX IF NOT EXISTS idx_exp_status     ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_exp_started_at ON experiments(started_at);

COMMENT ON TABLE experiments IS 'Registro de cada corrida experimental con hiperparámetros, seed y commit Git para reproducibilidad.';

-- -----------------------------------------------------------
-- Tabla: model_metrics
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_metrics (
    id              SERIAL PRIMARY KEY,
    experiment_id   INTEGER          REFERENCES experiments(id) ON DELETE CASCADE,
    split           VARCHAR(10)      NOT NULL,
    precision_score NUMERIC(6, 4),
    recall_score    NUMERIC(6, 4),
    f1_score        NUMERIC(6, 4),
    auc_roc         NUMERIC(6, 4),
    true_positives  INTEGER,
    false_positives INTEGER,
    true_negatives  INTEGER,
    false_negatives INTEGER,
    threshold       NUMERIC(6, 4),
    extra_metrics   JSONB,
    recorded_at     TIMESTAMPTZ      DEFAULT NOW(),
    CONSTRAINT chk_metric_split CHECK (split IN ('train', 'val', 'test'))
);

CREATE INDEX IF NOT EXISTS idx_metric_exp_id ON model_metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_metric_split  ON model_metrics(split);

COMMENT ON TABLE model_metrics IS 'Métricas de evaluación por experimento y split (train/val/test).';

-- -----------------------------------------------------------
-- Tabla de control de migraciones
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     VARCHAR(20)  PRIMARY KEY,
    applied_at  TIMESTAMPTZ  DEFAULT NOW(),
    description TEXT
);

INSERT INTO schema_migrations (version, description)
VALUES ('001', 'Esquema inicial: raw_transactions, clean_transactions, feature_store, experiments, model_metrics.')
ON CONFLICT (version) DO NOTHING;

COMMIT;
