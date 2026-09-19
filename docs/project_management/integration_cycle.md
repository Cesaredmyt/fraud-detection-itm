# Ciclo de integración del roadmap

## Rama de trabajo

Todo el trabajo de los días 1–90 se desarrolla en `feature/antifraud-mvp`. `main`
no recibe trabajo diario directamente.

## Integración cada 10 días

Los puntos de integración previstos son los días 10, 20, 30, 40, 50, 60,
70, 80 y 90. En cada punto:

1. Completar las tareas y la evidencia acumulada del bloque.
2. Ejecutar Ruff lint y format, Mypy y Pytest unitario/no integración.
3. Ejecutar las pruebas de integración, seguridad, datos o rendimiento que el
   bloque haya incorporado.
4. Revisar que no se usó TEST para selección o tuning y que los artefactos no
   contienen secretos, PAN ni payloads financieros completos.
5. Registrar resultados, duración, commit y cualquier excepción pendiente.
6. Integrar en `develop` únicamente si las comprobaciones requeridas pasan.

Un fallo conserva el trabajo en `feature/antifraud-mvp`; no se fuerza el merge y se
documenta el bloqueo. La integración en `main` queda fuera de este ciclo y
requiere el proceso de release correspondiente.
