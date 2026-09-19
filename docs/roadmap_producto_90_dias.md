# Roadmap de 90 días: de tesis académica a producto antifraude

## Objetivo

Al terminar el día 90 debe existir un MVP pilotable en **shadow mode** que reciba transacciones por API o batch, produzca scores reproducibles, registre modelo/política/explicación, reciba feedback tardío, monitorice calidad y drift, y pueda desplegarse o revertirse sin manipulación manual.

El MVP no bloqueará transacciones automáticamente. Las decisiones serán `approve`, `review` o una recomendación de `decline`; el cliente conservará la decisión final hasta completar validación con datos reales.

## Supuestos de ejecución

| Rol | Dedicación | Responsabilidad |
|---|---:|---|
| ML Engineer (`ML`) | Tiempo completo | Datos, features, evaluación, modelos, calibración, drift |
| Backend/MLOps (`BE`) | Tiempo completo | API, persistencia, CI/CD, despliegue, observabilidad |
| Producto/Riesgo (`PR`) | 25% | Costos, política, KPIs, flujo de revisión y piloto |
| Seguridad/Legal (`SL`) | 10% | Privacidad, threat model, controles y contratos |

Si una sola persona ejecuta el plan, debe conservar el orden y convertirlo en aproximadamente 18 semanas laborables. No se debe recortar validación, seguridad ni pruebas para conservar la fecha.

## Prioridades

- `P0`: bloquea la validez del modelo o el piloto.
- `P1`: obligatorio antes de incorporar datos de un cliente.
- `P2`: mejora posterior; no debe distraer del MVP.

## Definition of Done

Una tarea se considera terminada cuando:

- [ ] Tiene responsable, ticket y criterio de aceptación.
- [ ] Incluye pruebas positivas, negativas y de borde.
- [ ] `ruff check` y `ruff format --check` pasan.
- [ ] `mypy` pasa sin ser ignorado por CI.
- [ ] `pytest -m "not integration"` pasa desde un checkout limpio.
- [ ] No introduce acceso a TEST durante selección o tuning.
- [ ] Actualiza documentación, configuración y runbook afectados.
- [ ] Incluye plan de rollback cuando cambia datos, modelos o infraestructura.
- [ ] No registra secretos, PAN ni payloads financieros completos.

## Reglas metodológicas no negociables

- TEST no selecciona features, modelos, reglas, contaminación ni umbrales.
- Toda transformación aprendida se ajusta exclusivamente con train.
- SMOTE solo se aplica dentro de train/fold y nunca a IsolationForest.
- La evaluación principal es out-of-time; el split aleatorio es secundario.
- Una feature debe existir en el instante real de decisión.
- Score, calibración, política y decisión se versionan por separado.
- PaySim y Credit Card son benchmarks académicos, no evidencia comercial.

---

# Plan diario

## Semana 1 — Control, instalación y arquitectura

### Día 1 — Congelar el estado inicial

**Responsables:** ML + BE · **Prioridad:** P0

- [ ] Crear tickets `FD-001` en adelante y un tablero `Backlog / In progress / Review / Done`.
- [ ] Registrar commit, rama, `git status`, Python, pip, SO y hardware.
- [ ] Ejecutar Ruff, Mypy y Pytest sin alterar variables; guardar salida y duración.
- [ ] Documentar el fallo del editable install que apunta a una ruta anterior.
- [ ] Crear `docs/audit/baseline_technical_state.md`.
- [ ] Etiquetar métricas existentes como `academic_legacy`.

**Aceptación:** existe una fotografía técnica reproducible y fechada.

### Día 2 — Reparar instalación y dependencias

**Responsable:** BE · **Prioridad:** P0

- [ ] Quitar de `requirements.txt` la instalación editable del propio repositorio.
- [ ] Separar extras `dev`, `training`, `api` y `observability` en `pyproject.toml`.
- [ ] Elegir y generar un lockfile con hashes.
- [ ] Crear un entorno vacío e instalar con `pip install -e ".[dev]"`.
- [ ] Probar Python 3.11 y 3.12 sin `PYTHONPATH` manual.
- [ ] Actualizar `docs/reproducibility_guide.md`.

**Aceptación:** un checkout limpio importa `fraud_detection` desde cualquier directorio.

### Día 3 — Endurecer CI

**Responsable:** BE · **Prioridad:** P0

- [ ] Eliminar `continue-on-error` de Mypy.
- [ ] Separar jobs de lint, tipos, unit, integration, build y security scan.
- [ ] Incluir entrypoints productivos fuera de `src` o moverlos al paquete.
- [ ] Publicar cobertura XML y fijar mínimo global de 80%.
- [ ] Exigir 85% en módulos nuevos de training, inference, contracts y decisioning.
- [ ] Añadir secret scan y dependency audit.

**Aceptación:** lint, tipos, tests o vulnerabilidades críticas bloquean merge.

### Día 4 — ADRs de producto

**Responsables:** ML + BE + PR · **Prioridad:** P0

- [ ] Crear ADR del límite del MVP: scoring y revisión, no procesamiento de pagos.
- [ ] Crear ADR de paridad training-serving.
- [ ] Crear ADR de PostgreSQL como control/auditoría, no agregador online.
- [ ] Crear ADR de candidate/champion, shadow y rollback.
- [ ] Crear ADR que prohíba tuning sobre TEST.
- [ ] Registrar alternativas y condiciones para revisarlas.

**Aceptación:** las cinco decisiones están revisadas y versionadas.

### Día 5 — Inventario de features

**Responsables:** ML + PR · **Prioridad:** P0

- [ ] Inventariar nombre, fuente, tipo, sensibilidad y momento de disponibilidad.
- [ ] Marcar saldos posteriores, labels y chargebacks como postdecisión.
- [ ] Marcar V1–V28 como no reproducibles para clientes nuevos.
- [ ] Separar features del request, derivadas stateless y dependientes de estado.
- [ ] Definir zona horaria, moneda, precisión y política de nulos.
- [ ] Crear `docs/data/feature_availability_matrix.md`.

**Aceptación:** el modelo productivo no usa información futura.

### Día 6 — Revisión y buffer

- [ ] Revisar PRs de los días 1–5.
- [ ] Resolver únicamente defectos y bloqueos.
- [ ] Confirmar propietarios de riesgos P0.
- [ ] Actualizar tablero y decisiones pendientes.

**Aceptación:** no hay PR P0 abierto de la semana.

### Día 7 — Gate 1

- [ ] Instalar y ejecutar suite desde un runner limpio.
- [ ] Revisar ADRs y matriz de features.
- [ ] Confirmar que TEST permanece inaccesible para tuning.
- [ ] Firmar checklist de entrada a correcciones metodológicas.

**Aceptación:** reproducibilidad y reglas experimentales aprobadas.

## Semana 2 — Leakage y feature engineering

### Día 8 — Corregir historial entre usuarios

**Responsable:** ML · **Prioridad:** P0

- [ ] Escribir test que reproduzca contaminación A→B.
- [ ] Añadir casos con índices no consecutivos, filas desordenadas y tiempos empatados.
- [ ] Sustituir `shift(1)` global por desplazamiento dentro del grupo.
- [ ] Definir desempate determinista para transacciones simultáneas.
- [ ] Preservar explícitamente identidad y orden de filas.
- [ ] Ejecutar regresión sobre muestra PaySim.

**Aceptación:** la primera transacción de cada usuario tiene historial cero.

### Día 9 — Rehacer `amount_decile`

**Responsable:** ML · **Prioridad:** P0

- [ ] Crear transformador `fit/transform`.
- [ ] Ajustar cuantiles solo en train.
- [ ] Resolver límites duplicados y valores fuera de rango.
- [ ] Serializar límites y versión.
- [ ] Probar que modificar TEST no cambia bins.
- [ ] Retirar el ranking calculado sobre el dataset completo.

**Aceptación:** validation y test no afectan estadísticas aprendidas.

### Día 10 — Split temporal y grupal

**Responsable:** ML · **Prioridad:** P0

- [ ] Implementar `temporal_split` sin shuffle.
- [ ] Implementar opción de agrupación por cuenta/comercio/cliente.
- [ ] Añadir periodo de embargo configurable.
- [ ] Validar ausencia de solapamiento temporal y de entidades restringidas.
- [ ] Reportar rangos de fecha, filas, entidades y prevalencia.
- [ ] Mantener el split estratificado solo como benchmark secundario.

**Aceptación:** ninguna observación futura entra en train.

### Día 11 — Transformadores versionados

- [ ] Definir interfaz `fit`, `transform`, `get_feature_names_out` y `version`.
- [ ] Clasificar transformaciones stateless y stateful.
- [ ] Fijar orden, nombres y tipos de columnas.
- [ ] Fallar ante columnas faltantes o tipos incompatibles.
- [ ] Definir tratamiento de columnas extra y nulos.
- [ ] Añadir round-trip de serialización.

**Aceptación:** el mismo objeto transforma train, validation, test e inferencia.

### Día 12 — Resampling por componente

- [ ] Mover resampling de configuración global a configuración por modelo.
- [ ] Prohibir SMOTE en modelos no supervisados.
- [ ] Evitar SMOTE + `scale_pos_weight` sin experimento explícito.
- [ ] Entrenar IF con datos originales.
- [ ] Decidir si IF usa población completa o legítima madura.
- [ ] Añadir test que impida pasar train sintético al IF del híbrido.

**Aceptación:** ningún detector no supervisado recibe datos SMOTE.

### Día 13 — Validación cruzada sin leakage

- [ ] Ejecutar feature fitting dentro de cada fold.
- [ ] Ejecutar resampling solo dentro del fold train.
- [ ] Añadir rolling/temporal validation.
- [ ] Añadir Average Precision, Precision@K y Recall@K.
- [ ] Guardar distribución por fold e intervalos.
- [ ] Crear test sentinela que detecte información exclusiva de validation.

**Aceptación:** preprocessing y resampling quedan encapsulados por fold.

### Día 14 — Gate 2

- [ ] Revisar tests de leakage.
- [ ] Bloquear scripts que seleccionen sobre TEST.
- [ ] Aprobar protocolo experimental v2.
- [ ] Registrar commit del protocolo.
- [ ] No reentrenar hasta aprobar el gate.

**Aceptación:** protocolo v2 auditable y sin leakage conocido.

## Semana 3 — Dataset y modelo candidato

### Día 15 — Manifest del dataset

- [ ] Registrar fuente, licencia, checksum y ventana temporal.
- [ ] Registrar filas, columnas, prevalencia y criterios de exclusión.
- [ ] Definir identificador único de transacción.
- [ ] Documentar retraso y madurez de etiquetas.
- [ ] Rechazar datasets sin procedencia verificable.

**Aceptación:** cada corrida identifica exactamente sus datos.

### Día 16 — Dataset builder v2

- [ ] Separar extracción, validación, limpieza y materialización.
- [ ] Escribir Parquet particionado.
- [ ] Añadir muestra determinista para CI.
- [ ] Evitar copias completas innecesarias.
- [ ] Emitir reporte de calidad, tiempo y memoria.

**Aceptación:** sample/full utilizan el mismo código y producen manifest.

### Día 17 — Métricas de producto

- [ ] Definir costo de FP, FN y revisión.
- [ ] Definir capacidad diaria de revisión.
- [ ] Implementar costo esperado y fraude monetario capturado.
- [ ] Añadir métricas por monto, tiempo y segmento permitido.
- [ ] Separar labels preliminares de maduros.

**Aceptación:** existe una función objetivo aprobada por PR.

### Día 18 — Reentrenar RF y XGBoost

- [ ] Entrenar con split v2.
- [ ] Ejecutar tuning solo con train/validation.
- [ ] Registrar hiperparámetros, seed, tiempo, memoria y tamaño.
- [ ] Evaluar estabilidad temporal y por segmento.
- [ ] No abrir holdout final.

**Aceptación:** hay candidatos reproducibles y comparables.

### Día 19 — Evaluar IsolationForest

- [ ] Entrenar sin SMOTE.
- [ ] Comparar población completa vs. legítima madura.
- [ ] Seleccionar umbral en validation.
- [ ] Medir estabilidad de score y falsos positivos.
- [ ] Medir uplift respecto al supervisado.
- [ ] Documentar eliminación si no aporta valor.

**Aceptación:** decisión objetiva sobre el rol de IF.

### Día 20 — Calibrar scores y política

- [ ] Comparar Platt e isotonic.
- [ ] Medir Brier score y calibration curve.
- [ ] Definir `review_threshold` y `decline_threshold` por costo.
- [ ] Deshabilitar auto-decline en MVP.
- [ ] Persistir calibrador y umbrales.

**Aceptación:** no se usa el `predict()` implícito a 0.5.

### Día 21 — Gate 3

- [ ] Revisar métricas, estabilidad y features dominantes.
- [ ] Aprobar candidato y baseline simple.
- [ ] Crear model card.
- [ ] Documentar limitaciones de datasets públicos.

**Aceptación:** candidato elegido sin consultar TEST.

## Semana 4 — Contratos y bundle

### Día 22 — `TransactionV1`

- [ ] Crear schema Pydantic versionado.
- [ ] Incluir IDs, timestamp, monto, moneda, canal y token de cuenta.
- [ ] Definir límites, opcionales y campos extra.
- [ ] Exigir RFC 3339 con zona horaria.
- [ ] Prohibir PAN y secretos.
- [ ] Crear contract tests válidos e inválidos.

**Aceptación:** input público independiente de DataFrames internos.

### Día 23 — Respuesta de scoring

- [ ] Definir `risk_score`, `decision`, `reason_codes` y versiones.
- [ ] Incluir `request_id` y advertencias de degradación.
- [ ] Definir códigos de error estables.
- [ ] No exponer detalles que faciliten evasión.
- [ ] Añadir ejemplos OpenAPI.

**Aceptación:** un cliente puede integrar solo con la especificación.

### Día 24 — Model bundle

- [ ] Empaquetar schema, transformadores, modelo, calibrador y policy.
- [ ] Incluir manifest, commit y versiones de librerías.
- [ ] Calcular checksum SHA-256.
- [ ] Rechazar incompatibilidad de schema o checksum.
- [ ] Añadir test save/load/raw-input/score.

**Aceptación:** score idéntico antes y después de serializar.

### Día 25 — `ScoringService`

- [ ] Crear servicio de dominio independiente de HTTP.
- [ ] Precargar bundle una vez.
- [ ] Validar, transformar, puntuar, calibrar y decidir.
- [ ] Emitir estructura de auditoría.
- [ ] Cubrir individual, batch y errores.

**Aceptación:** inferencia se prueba sin servidor web.

### Día 26 — FastAPI

- [ ] Crear app factory.
- [ ] Implementar `/health`, `/ready` y `/v1/score`.
- [ ] Añadir request ID y medición de latencia.
- [ ] Mapear excepciones de dominio.
- [ ] Redactar logs sensibles.

**Aceptación:** OpenAPI y contract tests pasan.

### Día 27 — Batch

- [ ] Implementar `/v1/score:batch` con límites.
- [ ] Mantener orden e IDs.
- [ ] Devolver errores por elemento cuando proceda.
- [ ] Diseñar job asíncrono para archivos grandes.
- [ ] Medir memoria 1/10/100/máximo.

**Aceptación:** batch no recarga el modelo ni pierde trazabilidad.

### Día 28 — Gate 4

- [ ] Comparar scores offline, servicio y API.
- [ ] Probar artifact corrupto y columnas inválidas.
- [ ] Ejecutar contract tests.
- [ ] Publicar quickstart local.

**Aceptación:** paridad training-serving demostrada.

## Semana 5 — Contenedor y persistencia

### Día 29 — Docker

- [ ] Crear Dockerfile multi-stage.
- [ ] Ejecutar como usuario no root.
- [ ] Fijar imagen por digest.
- [ ] Excluir datos, modelos no aprobados, `.env` y notebooks.
- [ ] Generar SBOM y scan.

**Aceptación:** imagen reproducible sin secretos ni vulnerabilidad crítica aceptada.

### Día 30 — Settings

- [ ] Crear settings tipados por entorno.
- [ ] Validar variables al arrancar.
- [ ] Añadir model URI, shadow mode, timeouts y límites.
- [ ] Eliminar defaults sensibles.
- [ ] Documentar precedencia.

**Aceptación:** configuración inválida impide readiness.

### Día 31 — Esquema operativo

- [ ] Diseñar `tenants`, `model_versions`, `policy_versions`.
- [ ] Diseñar `score_events`, `feedback_events`, `review_cases`.
- [ ] Incluir event/received/decision timestamps.
- [ ] Definir índices, particionado y retención.
- [ ] Minimizar PII.

**Aceptación:** esquema revisado por BE, ML y SL.

### Día 32 — Migración 002

- [ ] Crear nueva migración sin editar 001.
- [ ] Añadir constraints e índices.
- [ ] Documentar rollback.
- [ ] Probar DB nueva y actualización 001→002.
- [ ] Añadir schema integration test.

**Aceptación:** ambas rutas producen el mismo schema.

### Día 33 — Persistencia asíncrona

- [ ] Separar scoring de escritura no crítica.
- [ ] Implementar cola/buffer y backpressure.
- [ ] Definir comportamiento con DB caída.
- [ ] Garantizar idempotencia y DLQ.
- [ ] Medir impacto en latencia.

**Aceptación:** DB caída no bloquea scoring según política.

### Día 34 — Multi-tenancy e idempotencia

- [ ] Crear clave tenant+transaction.
- [ ] Rechazar misma clave con payload distinto.
- [ ] Filtrar todas las consultas por tenant.
- [ ] Probar acceso cruzado.
- [ ] Documentar reintentos.

**Aceptación:** tenants no pueden acceder o afectar datos ajenos.

### Día 35 — Gate 5

- [ ] Levantar API y PostgreSQL con compose.
- [ ] Aplicar migraciones.
- [ ] Cargar bundle y ejecutar smoke test.
- [ ] Destruir/recrear el stack.
- [ ] Probar desde documentación únicamente.

**Aceptación:** desarrollador nuevo levanta el sistema sin conocimiento tribal.

## Semana 6 — Feedback y decisioning

### Día 36 — Contrato de feedback

- [ ] Definir fraude confirmado, legítimo, chargeback, review y unknown.
- [ ] Incluir fuente, fecha, monto y confianza.
- [ ] Permitir revisiones sin borrar historia.
- [ ] Definir madurez por tipo.
- [ ] Crear contract tests.

**Aceptación:** label provisional evoluciona a maduro auditadamente.

### Día 37 — Endpoint de feedback

- [ ] Implementar `/v1/feedback`.
- [ ] Validar propiedad del tenant.
- [ ] Implementar idempotencia y revisiones.
- [ ] Emitir evento de actualización.
- [ ] Probar concurrencia y duplicados.

**Aceptación:** duplicados no cuentan dos veces.

### Día 38 — Policy engine

- [ ] Separar score y decisión.
- [ ] Versionar policy por tenant.
- [ ] Implementar approve/review/decline.
- [ ] Deshabilitar decline por defecto.
- [ ] Aplicar capacidad de revisión.
- [ ] Registrar causa de decisión.

**Aceptación:** policy cambia sin reentrenar.

### Día 39 — Motor de reglas

- [ ] Reemplazar reglas hard-coded por configuración tipada.
- [ ] Validar operadores y features permitidas.
- [ ] Añadir prioridad, acción y versión.
- [ ] Ejecutar nuevas reglas en shadow.
- [ ] Prohibir `eval` y código arbitrario.

**Aceptación:** reglas auditables, testeables y reversibles.

### Día 40 — Resolver modelo + reglas + policy

- [ ] Definir orden de ejecución.
- [ ] Definir precedencia de conflictos.
- [ ] Guardar score puro y decisión final.
- [ ] Consolidar reason codes.
- [ ] Probar matriz de conflictos.

**Aceptación:** decisión reconstruible con versiones.

### Día 41 — Labels retrasados

- [ ] Implementar ventanas 7/30/60/90 días.
- [ ] Excluir labels inmaduros de métricas finales.
- [ ] Medir cobertura de labels.
- [ ] Manejar eventos fuera de orden.
- [ ] Evitar considerar “sin chargeback” como legítimo.

**Aceptación:** métricas comparan poblaciones con igual madurez.

### Día 42 — Gate 6

- [ ] Ejecutar transaction→score→decision→feedback→metric.
- [ ] Corregir duplicidad y trazabilidad.
- [ ] Reconstruir un caso completo desde DB.
- [ ] Aprobar ciclo cerrado.

**Aceptación:** flujo cerrado funciona end-to-end.

## Semana 7 — Explainability

### Día 43 — Catálogo de reason codes

- [ ] Crear códigos estables para monto, velocidad, tiempo, dispositivo y reglas.
- [ ] Mapear features a conceptos operativos.
- [ ] Separar códigos del modelo y reglas.
- [ ] Definir severidad y audiencia.
- [ ] Versionar catálogo.

**Aceptación:** analista entiende códigos sin conocer SHAP.

### Día 44 — TreeSHAP offline

- [ ] Añadir dependencia compatible.
- [ ] Generar explicaciones en validation.
- [ ] Comparar con permutation importance.
- [ ] Detectar proxies/features dominantes sospechosas.
- [ ] Versionar background dataset.

**Aceptación:** explicaciones reproducibles por modelo.

### Día 45 — Explicación local

- [ ] Extraer contribuciones positivas/negativas.
- [ ] Convertir top contribuciones en reason codes.
- [ ] Limitar cantidad y precisión expuesta.
- [ ] Manejar features correlacionadas.
- [ ] Probar score bajo/medio/alto.

**Aceptación:** casos review tienen razón útil o degradación explícita.

### Día 46 — Presupuesto de latencia

- [ ] Medir p95 con/sin SHAP.
- [ ] Fijar presupuesto.
- [ ] Mantener reason codes rápidos síncronos.
- [ ] Mover detalle a worker si excede SLO.
- [ ] Definir fallback.

**Aceptación:** explainability no rompe SLO.

### Día 47 — Persistir explicaciones

- [ ] Guardar versión de explainer/catálogo.
- [ ] Evitar PII innecesaria.
- [ ] Enlazar explicación con score y decisión.
- [ ] Definir retención.
- [ ] Crear consulta para analista.

**Aceptación:** explicación se audita junto a decisión.

### Día 48 — Seguridad y estabilidad

- [ ] Probar perturbaciones pequeñas.
- [ ] Identificar códigos inestables.
- [ ] Revisar filtración de lógica y umbrales.
- [ ] Revisar features sensibles/proxies.
- [ ] Definir vistas cliente/analista/auditor.

**Aceptación:** explicación no facilita evasión.

### Día 49 — Gate 7

- [ ] Revisar 30 casos.
- [ ] Registrar ejemplos buenos y fallidos.
- [ ] Aprobar catálogo v1.
- [ ] Actualizar model card.

**Aceptación:** PR acepta utilidad operacional.

## Semana 8 — Observabilidad

### Día 50 — Métricas HTTP

- [ ] Medir requests, errores, latencia, throughput y payload.
- [ ] Controlar cardinalidad de labels.
- [ ] Exponer métricas protegidas.
- [ ] Crear dashboard inicial.
- [ ] Añadir model/policy version.

**Aceptación:** degradación visible sin inspección manual.

### Día 51 — Logs y trazas

- [ ] Emitir logs JSON.
- [ ] Añadir request ID, versión y duración.
- [ ] Seudonimizar tenant.
- [ ] Redactar payloads y secretos.
- [ ] Trazar validación, features, modelo y persistencia.

**Aceptación:** request rastreable sin filtrar datos sensibles.

### Día 52 — Calidad online

- [ ] Medir nulos, rangos, categorías nuevas y parsing failures.
- [ ] Comparar con schema del bundle.
- [ ] Definir fallo duro vs warning.
- [ ] Alertar por tenant.
- [ ] Registrar degradación de features.

**Aceptación:** integración rota se detecta antes de degradar silenciosamente.

### Día 53 — Drift baseline

- [ ] Guardar distribuciones de referencia.
- [ ] Definir bins PSI estables.
- [ ] Añadir JS/KS cuando corresponda.
- [ ] Medir feature, score y decision drift.
- [ ] Segmentar sin crear cardinalidad incontrolable.

**Aceptación:** cada modelo tiene baseline versionada.

### Día 54 — Alertas

- [ ] Definir niveles info/warning/critical.
- [ ] Alertar error, latencia, schema, drift y alert rate.
- [ ] Añadir cooldown.
- [ ] Asignar dueño/canal.
- [ ] Vincular runbook.

**Aceptación:** toda alerta tiene acción y propietario.

### Día 55 — Runbooks

- [ ] Documentar modelo no cargable.
- [ ] Documentar DB/cola caída.
- [ ] Documentar latencia alta.
- [ ] Documentar schema/score drift.
- [ ] Incluir diagnóstico, mitigación y rollback.

**Aceptación:** una persona distinta puede responder al incidente.

### Día 56 — Gate 8

- [ ] Simular bundle corrupto.
- [ ] Simular DB caída.
- [ ] Enviar payload inválido.
- [ ] Inyectar drift sintético.
- [ ] Verificar alertas y runbooks.

**Aceptación:** cuatro fallos detectables y accionables.

## Semana 9 — Registry y despliegue

### Día 57 — Manifest de modelo

- [ ] Incluir ID, versión, checksum, fechas y commit.
- [ ] Referenciar datasets y feature schema.
- [ ] Incluir métricas, calibrador, policy compatible y limitaciones.
- [ ] Incluir estado y aprobadores.
- [ ] Validar con JSON Schema.

**Aceptación:** bundle incompleto no se registra.

### Día 58 — Artefact storage

- [ ] Configurar object storage.
- [ ] Habilitar versionado e inmutabilidad lógica.
- [ ] Verificar checksum upload/download.
- [ ] Separar permisos read/write.
- [ ] Definir cifrado y retención.

**Aceptación:** modelo no se sobrescribe silenciosamente.

### Día 59 — Estados y promoción

- [ ] Implementar candidate/approved/champion/retired/rejected.
- [ ] Exigir evidencia para aprobación.
- [ ] Auditar actor y timestamp.
- [ ] Bloquear transiciones inválidas.
- [ ] Conservar campeón anterior.

**Aceptación:** promociones y rollback auditados.

### Día 60 — Shadow deployment

- [ ] Ejecutar champion y candidate sobre el mismo evento.
- [ ] Usar champion para respuesta.
- [ ] Persistir desacuerdos.
- [ ] Aislar errores del candidate.
- [ ] Añadir porcentaje y kill switch.

**Aceptación:** candidate puede fallar sin afectar scoring.

### Día 61 — Champion–challenger

- [ ] Comparar score, decisión y desacuerdo.
- [ ] Esperar labels maduros.
- [ ] Medir costo y revisión adicional.
- [ ] Reportar por periodo/segmento.
- [ ] Exigir muestra mínima.

**Aceptación:** no se promueve por variación puntual.

### Día 62 — Rollback atómico

- [ ] Validar bundle antes de activar.
- [ ] Mantener último bundle válido.
- [ ] Implementar swap atómico.
- [ ] Probar rollback bajo carga.
- [ ] Definir permisos de producción.

**Aceptación:** no existe ventana con modelo parcial.

### Día 63 — Gate 9

- [ ] Registrar candidate.
- [ ] Aprobar y desplegar shadow.
- [ ] Promover y revertir.
- [ ] Verificar auditoría.
- [ ] Documentar tiempos.

**Aceptación:** lifecycle sin manipular archivos manualmente.

## Semana 10 — Seguridad y privacidad

### Día 64 — Threat model

- [ ] Diagramar activos y trust boundaries.
- [ ] Evaluar STRIDE.
- [ ] Incluir extraction, adversarial input y feedback poisoning.
- [ ] Priorizar probabilidad/impacto.
- [ ] Asignar mitigaciones.

**Aceptación:** amenazas P0/P1 tienen dueño y acción.

### Día 65 — Autenticación/autorización

- [ ] Implementar autenticación de servicio.
- [ ] Separar roles score/feedback/analyst/admin.
- [ ] Añadir expiración, rotación y revocación.
- [ ] Probar denegación y acceso cruzado.
- [ ] Redactar credenciales en logs.

**Aceptación:** mínimo privilegio demostrado.

### Día 66 — Secretos y cifrado

- [ ] Usar secret manager fuera de local.
- [ ] Habilitar TLS.
- [ ] Verificar cifrado de DB y artefactos.
- [ ] Definir rotación.
- [ ] Escanear historial Git.

**Aceptación:** secretos ausentes de repo, imagen y logs.

### Día 67 — Privacidad

- [ ] Inventariar tratamientos y finalidades.
- [ ] Tokenizar identificadores.
- [ ] Definir retención por almacenamiento.
- [ ] Crear borrado/exportación.
- [ ] Documentar transferencias y subprocesadores.

**Aceptación:** cada dato tiene finalidad, dueño y expiración.

### Día 68 — Secure SDLC

- [ ] Habilitar SAST, dependency, container y secret scan.
- [ ] Definir SLA de vulnerabilidades.
- [ ] Proteger ramas y reviews.
- [ ] Fijar acciones CI por commit.
- [ ] Generar SBOM por release.

**Aceptación:** critical findings bloquean release.

### Día 69 — Incidentes y continuidad

- [ ] Crear plan de respuesta.
- [ ] Definir severidades y contactos.
- [ ] Definir RTO/RPO.
- [ ] Configurar backups.
- [ ] Restaurar en entorno aislado.
- [ ] Definir fail-open/fail-closed.

**Aceptación:** restore probado y medido.

### Día 70 — Gate 10

- [ ] Revisar threat model y controles.
- [ ] Cerrar P0 y calendarizar P1.
- [ ] Aprobar alcance de datos.
- [ ] Confirmar ausencia de PAN crudo.
- [ ] Autorizar staging seudonimizado.

**Aceptación:** aprobación SL documentada.

## Semana 11 — Rendimiento y resiliencia

### Día 71 — SLOs

- [ ] Definir TPS, burst, batch y concurrencia.
- [ ] Definir p95/p99.
- [ ] Definir disponibilidad/error budget.
- [ ] Definir límites CPU/memoria.
- [ ] Documentar infraestructura objetivo.

**Aceptación:** escalabilidad expresada con números.

### Día 72 — Load tests

- [ ] Crear steady, burst, soak y overload.
- [ ] Usar datos sintéticos.
- [ ] Medir individual y batch.
- [ ] Medir explainability on/off.
- [ ] Guardar resultados por imagen/commit.

**Aceptación:** pruebas repetibles.

### Día 73 — Profiling

- [ ] Perfilar cada transformación.
- [ ] Medir copias y asignaciones.
- [ ] Medir cold start y carga.
- [ ] Identificar hotspots.
- [ ] No optimizar sin evidencia.

**Aceptación:** backlog de rendimiento basado en perfiles.

### Día 74 — Optimización

- [ ] Limitar threads y `n_jobs`.
- [ ] Reusar bundle, objetos y conexiones.
- [ ] Evaluar microbatch solo con datos.
- [ ] Comparar RF/XGBoost en costo.
- [ ] Añadir regression test numérico.

**Aceptación:** SLO con 20% de margen.

### Día 75 — Overload

- [ ] Limitar concurrencia.
- [ ] Responder overload controladamente.
- [ ] Configurar timeout/pool.
- [ ] Probar cola llena y DB lenta.
- [ ] Evitar retry storms.

**Aceptación:** degradación controlada y observable.

### Día 76 — Failure tests

- [ ] Reiniciar instancia bajo carga.
- [ ] Retirar bundle candidate.
- [ ] Interrumpir DB/storage.
- [ ] Inyectar latencia.
- [ ] Verificar recuperación e idempotencia.

**Aceptación:** no hay duplicidad o pérdida silenciosa.

### Día 77 — Gate 11

- [ ] Ejecutar suite de carga.
- [ ] Comparar con SLO.
- [ ] Documentar capacidad segura/máxima.
- [ ] Definir autoscaling inicial.
- [ ] Bloquear features que incumplan SLO.

**Aceptación:** capacidad del piloto cuantificada.

## Semana 12 — Cliente piloto

### Día 78 — Perfil de cliente

- [ ] Elegir un solo vertical/caso de uso.
- [ ] Definir rango de volumen.
- [ ] Exigir historial etiquetado y feedback.
- [ ] Definir integraciones soportadas.
- [ ] Preparar cuestionario técnico.

**Aceptación:** no se acepta un piloto imposible de medir.

### Día 79 — Data readiness

- [ ] Solicitar schema, diccionario y muestra seudonimizada.
- [ ] Validar timestamp/zona.
- [ ] Medir nulos, duplicados y labels.
- [ ] Confirmar features en tiempo real.
- [ ] Emitir go/no-go.

**Aceptación:** gaps conocidos antes de fechas contractuales.

### Día 80 — Éxito del piloto

- [ ] Definir baseline del cliente.
- [ ] Definir KPIs y fórmulas.
- [ ] Definir capacidad de revisión.
- [ ] Definir duración shadow y label maturity.
- [ ] Definir éxito, fracaso y extensión.

**Aceptación:** método de medición acordado por ambas partes.

### Día 81 — Mapping del cliente

- [ ] Mapear hacia `TransactionV1`.
- [ ] Definir moneda, unidades y categorías.
- [ ] Rechazar campos ambiguos.
- [ ] Crear fixtures seudonimizados.
- [ ] Versionar mapping/fallbacks.

**Aceptación:** no hay transformación manual fuera del código.

### Día 82 — Backtest real

- [ ] Aplicar split temporal.
- [ ] Entrenar solo con pasado.
- [ ] Calibrar en validation.
- [ ] Evaluar por periodo, monto y segmento.
- [ ] Comparar baseline del cliente.

**Aceptación:** potencial económico sin uso de futuro.

### Día 83 — Tenant piloto

- [ ] Crear tenant y credenciales.
- [ ] Configurar shadow policy.
- [ ] Configurar límites/retención.
- [ ] Configurar dashboards/alertas.
- [ ] Probar aislamiento.

**Aceptación:** tenant listo sin activar decisiones reales.

### Día 84 — Gate 12

- [ ] Revisar datos, seguridad y KPIs.
- [ ] Confirmar soporte/escalamiento.
- [ ] Ejecutar smoke con cliente.
- [ ] Probar kill switch.
- [ ] Firmar checklist go-live.

**Aceptación:** aprobación ML, BE, PR y SL.

## Días 85–90 — Shadow y cierre

### Día 85 — 5% de tráfico

- [ ] Activar subconjunto controlado.
- [ ] Verificar schema, latencia, errores y score.
- [ ] Comparar volumen esperado/real.
- [ ] Registrar versiones y hora.
- [ ] Corregir solo P0/P1.

**Aceptación:** dos horas estables y sin filtración.

### Día 86 — 25–50%

- [ ] Revisar día 85.
- [ ] Aumentar solo si cumple gates.
- [ ] Revisar reason codes/categorías nuevas.
- [ ] Verificar idempotencia/feedback.
- [ ] Documentar anomalías.

**Aceptación:** ampliación basada en métricas.

### Día 87 — 100% shadow

- [ ] Ampliar si los gates pasan.
- [ ] Simular cola de revisión.
- [ ] Medir alertas por hora.
- [ ] Versionar cualquier ajuste de policy.
- [ ] No reentrenar con labels inmaduros.

**Aceptación:** revisión simulada cabe en capacidad acordada.

### Día 88 — Simulacro y rollback

- [ ] Simular modelo inválido o latencia crítica.
- [ ] Activar kill switch.
- [ ] Revertir a champion.
- [ ] Verificar continuidad/auditoría.
- [ ] Medir RTO y actualizar runbook.

**Aceptación:** rollback dentro de RTO sin editar archivos.

### Día 89 — Revisión y backlog

- [ ] Consolidar métricas técnicas y preliminares.
- [ ] Separar labels maduros/inmaduros.
- [ ] Revisar incidentes y feedback.
- [ ] Decidir destino de IF/híbrido.
- [ ] Estimar costo por millón de transacciones.
- [ ] Priorizar siguiente 30/60/90.

**Aceptación:** decisiones posteriores con dueño y fecha.

### Día 90 — Release del piloto

- [ ] Ejecutar checklist de release.
- [ ] Etiquetar `v0.1.0-pilot`.
- [ ] Publicar model card, API docs, runbooks, SBOM y changelog.
- [ ] Registrar champion, policy, reason codes y manifests.
- [ ] Confirmar shadow/review; auto-decline continúa apagado.
- [ ] Firmar go/no-go y deuda aceptada con expiración.

**Aceptación:** otra persona puede operar, auditar, medir y revertir el MVP.

---

# Backlog ordenado

## P0 — empezar inmediatamente

- [ ] `FD-001` Reparar instalación y lock.
- [ ] `FD-002` Hacer CI bloqueante.
- [ ] `FD-003` Corregir historial multiusuario.
- [ ] `FD-004` Convertir deciles a transformer.
- [ ] `FD-005` Implementar split temporal/grupal.
- [ ] `FD-006` Rehacer CV sin leakage.
- [ ] `FD-007` Impedir tuning sobre TEST.
- [ ] `FD-008` Separar resampling y proteger IF.
- [ ] `FD-009` Definir schema predecisión.
- [ ] `FD-010` Crear model bundle.
- [ ] `FD-011` Crear ScoringService.
- [ ] `FD-012` Crear API score/batch/feedback/health.
- [ ] `FD-013` Persistir con tenancy/idempotencia.
- [ ] `FD-014` Crear registry/promoción/rollback.
- [ ] `FD-015` Añadir observabilidad mínima.

## P1 — antes del piloto

- [ ] `FD-016` Policy engine y reglas.
- [ ] `FD-017` Calibración y umbral por costo.
- [ ] `FD-018` Reason codes y SHAP.
- [ ] `FD-019` Labels tardíos.
- [ ] `FD-020` Threat model y RBAC.
- [ ] `FD-021` Privacidad y retención.
- [ ] `FD-022` Load/failure tests y SLO.
- [ ] `FD-023` Runbooks y restore.
- [ ] `FD-024` Champion–challenger shadow.
- [ ] `FD-025` Model card/release checklist.

## P2 — después de validar necesidad

- [ ] `FD-026` Redis para features online.
- [ ] `FD-027` Kafka/Flink o streaming equivalente.
- [ ] `FD-028` Feature store dedicado.
- [ ] `FD-029` Consola completa de analistas.
- [ ] `FD-030` Retraining programado con aprobación.
- [ ] `FD-031` Modelos por tenant/vertical.
- [ ] `FD-032` VPC/on-prem.

# Criterio para conservar IsolationForest/híbrido

- [ ] IF entrenado sin SMOTE.
- [ ] Umbral elegido solo en validation.
- [ ] Score tratado como anomalía o calibrado; no llamado probabilidad sin evidencia.
- [ ] Uplift medido sobre supervisado solo.
- [ ] FP compatibles con revisión.
- [ ] Estabilidad temporal y por segmento.
- [ ] Costo/latencia aceptables.

Si algún criterio falla, IF queda como experimento offline y se elimina del bundle piloto.

# Checklist de no-go

No conectar tráfico si:

- [ ] Se usan datos postdecisión.
- [ ] Se ajustó sobre TEST.
- [ ] No se reconstruye una decisión por versión.
- [ ] No existe rollback probado.
- [ ] Falla aislamiento por tenant.
- [ ] Se registran PAN, secretos o payloads sensibles.
- [ ] No hay acuerdo de label maturity.
- [ ] El cliente espera auto-decline.
- [ ] No se conoce throughput seguro.
- [ ] No hay responsable de incidentes.

# Evidencia para declarar el MVP pilotable

- [ ] Build reproducible y CI verde.
- [ ] Evaluación v2 sin leakage conocido.
- [ ] Evaluación out-of-time.
- [ ] API/batch versionados.
- [ ] Paridad offline/online.
- [ ] Score/decision/feedback auditables.
- [ ] Registry y rollback.
- [ ] Calidad, drift y servicio monitorizados.
- [ ] Reason codes operativos.
- [ ] Security review y restore probados.
- [ ] Privacidad/retención definidas.
- [ ] SLO medido bajo carga.
- [ ] Piloto shadow con KPIs acordados.
- [ ] Sin usar PaySim como promesa comercial.
