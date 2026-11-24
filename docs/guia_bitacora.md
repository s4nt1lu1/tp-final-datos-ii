# Guía y Estándar para la Bitácora de Sesiones del Equipo

## Propósito
Este documento define el estándar de documentación interna del equipo para el **Trabajo Práctico Final de Laboratorio de Datos II**. 

Cada vez que un integrante realice avances y registre un `commit` en el repositorio, deberá documentar una nueva entrada en el archivo [`bitacora_sesiones.md`](./bitacora_sesiones.md). El objetivo es garantizar la **trazabilidad metodológica**: que cualquier compañero pueda retomar el trabajo entendiendo no solo *qué código se escribió*, sino *por qué se tomaron esas decisiones*, *qué problemas se encontraron en los datos* y *qué falta hacer a continuación*, facilitando además la redacción del informe final y la preparación de la defensa oral.

---

## Estructura Obligatoria de cada Entrada de Sesión

Cada registro en `bitacora_sesiones.md` debe incluir los siguientes 6 bloques:

### 1. Encabezado y Metadatos
* **Fecha:** Día en que se realizó el trabajo.
* **Autor/es:** Integrante(s) que trabajaron en la sesión.
* **Rama de Git (`branch`):** Rama activa donde se realizaron los cambios (ej. `feature/eda`, `feature/preprocessing`).
* **Fase / Sección de la consigna abordada:** Ej. *Fase 1 — Sección 1.1: Análisis Descriptivo y Versionado Inicial*.
* **Archivos creados o modificados:** Lista de notebooks, scripts o archivos JSON de trazabilidad actualizados.

### 2. Objetivo de la Sesión
Breve descripción (2 a 4 líneas) de qué puntos específicos de la consigna se buscaron resolver en este avance.

### 3. Razonamiento Metodológico (El "Por Qué")
Explicación lógica del camino seguido antes y durante la escritura del código:
* ¿Por qué se eligió cierto orden de operaciones?
* ¿Qué hipótesis se plantearon sobre el problema físico o la estructura de los archivos?
* ¿Qué criterios técnicos (ej. reglas de desempate, manejo de errores, prevención de *data leakage*) fundamentan las decisiones tomadas?

### 4. Hallazgos Cuantitativos y Resultados Clave (Con Números en la Mano)
Todo hallazgo debe quedar respaldado por los **valores numéricos exactos** medidos en el código (evitando afirmaciones vagas como *"había pocos nulos"* o *"algunos días se solapan"*):
* Cantidades exactas de filas, columnas y fechas.
* Fechas puntuales donde se detectaron anomalías, solapamientos o valores extremos.
* Estadísticas descriptivas (media, mediana, desvíos, mínimos y máximos).
* Hashes `SHA-256` registrados cuando se creen o modifiquen datasets.

### 5. Problemas Técnicos Encontrados y Solución Aplicada
Registro breve de cualquier obstáculo de datos o de librerías (por ejemplo, cambios de comportamiento en `pandas 3.1.0`, formatos de fecha, celdas con errores de Excel) y cómo se resolvió en el código.

### 6. Estado al Cierre y Próximos Pasos (Hoja de Ruta)
Lista clara de qué quedó terminado y cuáles son las tareas inmediatas para quien tome la posta en el siguiente `commit`.

---

## Plantilla Copiable para Nuevas Entradas

```markdown
## Sesión [N] — [Fecha: AAAA-MM-DD]

* **Autor/es:** [Nombre]
* **Rama:** `[nombre-de-rama]`
* **Fase / Sección:** [Ej. Fase 1 - Sección 1.2]
* **Archivos modificados:** `[lista de archivos]`

### 1. Objetivo de la Sesión
[Describir qué se propuso resolver]

### 2. Razonamiento Metodológico y Decisiones Tomadas
* [Decisión 1 y su justificación]
* [Decisión 2 y su justificación]

### 3. Hallazgos y Resultados Cuantitativos
* [Resultado 1 con números exactos]
* [Resultado 2 con números exactos]

### 4. Próximos Pasos
1. [Paso siguiente 1]
2. [Paso siguiente 2]
```

