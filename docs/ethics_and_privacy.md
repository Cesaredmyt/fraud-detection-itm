# Consideraciones éticas y de privacidad

## Marco legal

- Ley Federal de Protección de Datos Personales en Posesión de los Particulares (México, 2010).
- Ley para Regular las Instituciones de Tecnología Financiera (Ley Fintech, 2018).

## Principios aplicados

- **Datos anonimizados o sintéticos.** No se usa información personal identificable.
- **Repositorios públicos con licencia abierta.** Kaggle y UCI Machine Learning Repository.
- **Almacenamiento controlado.** PostgreSQL local con acceso restringido al equipo.
- **Uso académico exclusivo.** Los resultados no se transfieren a terceros sin autorización.

## Prácticas técnicas

- Las contraseñas y secretos se gestionan vía variables de entorno (`.env`), excluidos de Git.
- Los datasets crudos están excluidos del repositorio (`.gitignore`).
- Los modelos serializados se manejan localmente y no se distribuyen sin revisión.