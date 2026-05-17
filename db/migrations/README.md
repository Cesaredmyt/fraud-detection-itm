# Migraciones SQL

Versionado del esquema de PostgreSQL del proyecto.

## Convención

Las migraciones se nombran `NNN_descripcion_breve.sql` donde `NNN` es un número de tres dígitos secuencial.

## Cómo se ejecutan

Las migraciones se aplican mediante el script `scripts/run_migrations.py` que:

1. Conecta a PostgreSQL usando las credenciales de `.env`.
2. Lee todas las migraciones de esta carpeta en orden alfabético.
3. Aplica solo las que aún no figuran en la tabla `schema_migrations`.

Comando:

\`\`\`powershell
python scripts/run_migrations.py
\`\`\`

## Reglas

- **Las migraciones son inmutables.** Una vez aplicada en producción (o en repos compartidos), no se modifica. Si hay error, se crea una nueva migración que lo corrige.
- **Idempotencia.** Toda migración debe usar `IF NOT EXISTS` y `ON CONFLICT DO NOTHING` para que pueda ejecutarse dos veces sin romper nada.
- **Transaccionalidad.** Toda migración va dentro de `BEGIN ... COMMIT`.

## Migraciones registradas

| Versión | Descripción                                                   |
| ------- | ------------------------------------------------------------- |
| 001     | Esquema inicial (5 tablas principales + control de versiones) |
