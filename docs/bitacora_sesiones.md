# Bitácora de Sesiones y Trazabilidad Metodológica

En este documento se registran cronológicamente los avances, decisiones, hallazgos cuantitativos y próximos pasos de cada sesión de trabajo, siguiendo el estándar definido en [`guia_bitacora.md`](./guia_bitacora.md).

---

## Sesión 1 — 2026-11-06: Configuración Local y Resolución Completa de la Fase 1 (Sección 1.1)

* **Rama:** `feature/eda`
* **Fase / Sección:** Validación de Fase 0 e implementación completa de **Fase 1 — Sección 1.1 (Análisis Descriptivo y Versionado Inicial de Datos)**.
* **Archivos creados / modificados:**
  * `notebooks/eda.ipynb` *(creado — Celdas 1 a 10)*
  * `data/checksums.json` *(actualizado con el hash SHA-256 de `dataset_unificado_inicial.csv`)*
  * `data/raw/dataset_unificado_inicial.csv` *(generado localmente; excluido de Git por `.gitignore`)*
  * `docs/guia_bitacora.md` y `docs/bitacora_sesiones.md` *(creados)*

---

### 1. Objetivo de la Sesión
1. Reproducir y validar el entorno local configurado en la **Fase 0** (`Python 3.12.14` con `uv` y `requirements.txt`), verificando la integridad criptográfica (`SHA-256`) de los 3 archivos Excel crudos en `data/raw/`.
2. Crear la rama `feature/eda` conforme a la estrategia de *branching* exigida.
3. Resolver paso a paso todos los puntos de la **Sección 1.1**: inspección de hojas, análisis de solapamiento temporal entre los 3 archivos Excel, definición y justificación de la regla de desempate, control de jornadas apiladas (`> 25` registros por fecha), comparación del criterio de cierre diario (`23:00` vs. `23:59`), cálculo del checksum `SHA-256` del dataset unificado inicial, análisis de dimensiones/tipos/rangos/faltantes y detección de los 3 tipos de registros inválidos.

---

### 2. Razonamiento Metodológico y Decisiones Tomadas

1. **Inmutabilidad de los archivos crudos (`data/raw/`):**
   * En Excel, la columna `DIA` se visualiza como `#####` debido al ancho de columna, pero internamente almacena objetos `datetime` nativos válidos. Se estableció no abrir ni guardar cambios visuales en Excel para no alterar los metadatos binarios ni invalidar los hashes `SHA-256` de `data/checksums.json`.
2. **Por qué inspeccionar cada archivo por separado antes de concatenar (`pd.concat`):**
   * Si se apilaran los 3 archivos Excel a ciegas antes de analizarlos individualmente, se mezclarían dos anomalías de distinta naturaleza:
     * **Anomalía A (Solapamiento entre archivos):** Fechas normales de 25 filas que aparecen en dos archivos distintos y al apilarse sumarían 50 filas.
     * **Anomalía B (Jornadas apiladas dentro de un mismo archivo):** Fechas que ya traen 50, 51, 99 o 100 filas dentro de un único archivo Excel.
   * Analizar primero cada archivo por separado permitió descubrir la causa exacta de las fechas con `> 25` registros.
3. **Resolución de fechas con más de 25 registros (Plantilla de 31 días):**
   * Se comprobó que **todas** las fechas con más de 25 registros en los 3 archivos corresponden al último día de los meses que no tienen 31 días:
     * Meses de **30 días** (`30 de abril`, `30 de junio`, `30 de septiembre`, `30 de noviembre`) $\rightarrow$ tienen **50 o 51 filas** ($2 \times 25$ horas).
     * Mes de **28 días** (`28 de febrero`) $\rightarrow$ tiene **99 o 100 filas** ($4 \times 25$ horas).
   * Al inspeccionar el caso del `2020-09-30` ( junto con el entorno del `28/09` al `03/10`), se verificó que todos los días del calendario están presentes y que en el día `30/09` la plantilla de 31 bloques diarios de Excel dejó apilada una segunda jornada (que se reinicia a `00:00:00` en la fila 25) bajo la misma etiqueta de fecha.
   * **Decisión aplicada:** Dentro de cada archivo, agrupar por `DIA` y conservar únicamente las **primeras 25 filas (`.groupby("DIA").head(25)`)**, que corresponden al ciclo horario real (`00:00` a `23:59`) de esa fecha. Así se garantiza que **ninguna fecha supere los 25 registros** antes de agregar por día y que el futuro `shift(-1)` no quede desalineado.
4. **Regla de Desempate para el Solapamiento entre Archivos:**
   * Se compararon los cierres diarios (`23:59`) de `Frio (Kw)` en las fechas compartidas entre archivos:
     * En el último día de los archivos antiguos (`2022-03-16` en `2020_2022.xlsx` y `2023-03-06` en `2021_2023.xlsx`), la jornada está truncada (no tiene lectura de `23:59`, dando `NaN`), mientras que el archivo siguiente sí posee el día completo (`24.066,0 kW` y `19.199,0 kW`, respectivamente).
     * En las **5 fechas** donde `2020_2022.xlsx` y `2021_2023.xlsx` muestran valores distintos (`2021-07-28`, `2021-08-25`, `2021-11-28`, `2022-02-02` y `2022-02-18`), el archivo antiguo (`2020_2022.xlsx`) contiene **valores corruptos del orden de los millones** (`2.900.770 kW`, `2.907.122 kW`, `6.155.928 kW`, `8.861.470 kW`), mientras que en el archivo más nuevo (`2021_2023.xlsx`) esos errores ya fueron corregidos a valores operativos reales (`30.764 kW`, `37.122 kW`, `25.719 kW`, `19.887 kW` y `21.470 kW`).
   * **Regla de desempate definida:** **Priorizar siempre el archivo más reciente** (`2022_2023` > `2021_2023` > `2020_2022`).
5. **Elección del Criterio de Cierre Diario (`23:59` vs. `23:00`):**
   * Se cuantificó la diferencia entre la lectura de las `23:59` y la de las `23:00` en `Frio (Kw)`, obteniendo una mediana de entre **`830 kW` y `965 kW`** (promedio de `811,19 kW` en `2022_2023`, `972,58 kW` en `2021_2023`, y `5.725,43 kW` en `2020_2022` influenciado por valores atípicos).
   * **Decisión aplicada:** Utilizar de manera consistente la lectura de las **`23:59`**, dado que captura el consumo acumulado total de las 24 horas completas de la jornada (tal como recomienda la documentación del dataset).
6. **Justificación de `SHA-256` frente a `MD5`:**
   * Se utiliza `SHA-256` en lugar de `MD5` porque `MD5` es criptográficamente vulnerable a colisiones (dos archivos distintos pueden producir el mismo hash), mientras que `SHA-256` es resistente a colisiones y garantiza la integridad inmutable de los artefactos de datos.

---

### 3. Hallazgos Cuantitativos y Resultados Clave (Sección 1.1)

* **Estructura de hojas en los 3 archivos Excel:**
  * `2020_2022.xlsx` y `2022_2023.xlsx` poseen **15 hojas** comunes (`Consolidado KPI`, `Consolidado Produccion`, `Totalizadores Produccion`, `Consolidado EE`, `Totalizadores Energia`, `Consolidado Agua`, `Totalizadores Agua`, `Consolidado GasVapor`, `Totalizadores Gas y Vapor`, `Consolidado Aire`, `Totalizadores Aire`, `Totalizadores CO2`, `Totalizadores Efluentes`, `Totalizadores Glicol` y `Auxiliar`).
  * `2021_2023.xlsx` posee **16 hojas** (agrega la hoja `'Seguimiento Dia'`), confirmando la necesidad de acceder siempre a hojas y columnas por nombre y nunca por posición.
* **Dimensiones individuales antes de unificar (Hoja `Consolidado EE`):**
  * `2020_2022.xlsx`: `16.054` filas (`479` sin fecha válida), `612` fechas únicas (`2020-07-01` al `2022-03-16`), `8` fechas con $> 25$ registros.
  * `2021_2023.xlsx`: `15.450` filas (`134` sin fecha válida), `602` fechas únicas (`2021-07-01` al `2023-03-06`), `8` fechas con $> 25$ registros.
  * `2022_2023.xlsx`: `12.011` filas (`2` sin fecha válida), `474` fechas únicas (`2022-07-01` al `2023-10-26`), `6` fechas con $> 25$ registros.
* **Cuantificación del solapamiento temporal:**
  * Entre `2020_2022` y `2021_2023`: **254 fechas solapadas** (`2021-07-01` a `2022-03-16`). Coinciden exactamente en **246 fechas**, difieren en **5 fechas** (corregidas en `2021_2023`), en 2 fechas (`2021-07-13` y `2022-03-02`) ambos carecen de registro a las `23:59`, y en `2022-03-16` el primer archivo está truncado.
  * Entre `2021_2023` y `2022_2023`: **244 fechas solapadas** (`2022-07-01` a `2023-03-06`). Coinciden exactamente en **243 fechas** y en `2023-03-06` el segundo archivo está truncado.
  * Entre `2020_2022` y `2022_2023`: **0 fechas solapadas**.
  * **Total de fechas solapadas:** **498 fechas** ($\approx 12.450$ filas horarias duplicadas eliminadas al desempate).
* **Dataset unificado inicial (`dataset_unificado_inicial.csv`):**
  * **Dimensiones:** **`29.698` filas x `21` columnas**, abarcando **`1.190` días únicos** desde el `2020-07-01` hasta el `2023-10-26`, con **0 fechas con más de 25 registros** (`1.180` días de 25 filas, `2` días de 24 filas y `8` días de menos de 24 filas).
  * **Hash `SHA-256` registrado en `data/checksums.json`:** `41e1d971dbb16c5d5a8068e63937c116d2975582794c8dd2e5a0ee3cacfcc9a4`.
* **Valores faltantes en `Consolidado EE` unificado:**
  * Solo `Linea 2 (Kw)` presenta **4 valores faltantes (`0,01 %`)** y `KW Gral Planta` presenta **1 valor faltante (`0,00 %`)**. El resto de las columnas (incluida `Frio (Kw)`) tiene `0` nulos.
* **Detección de registros inválidos y valores atípicos:**
  1. **Días incompletos (`< 24` lecturas horarias):** **8 días** (`2021-07-01` con 23h, `2021-07-13` con 23h, `2021-11-01` con 23h, `2023-01-17` con 16h, `2023-04-13` con 20h, `2023-04-19` con 17h, `2023-07-01` con 23h y `2023-10-26` con 8h). Además existen **2 días con 24 lecturas** a los que les falta el registro de cierre de las `23:59`.
  2. **Valores imposibles (orden de millones/billones y negativos):**
     * En `Frio (Kw)` se detectaron **3 filas horarias** con valores imposibles (ocurridas en horas intermedias del día y no a las `23:59`):
       * `2021-01-31` a las `11:00:00`: $3,277 \times 10^{16}\text{ kW}$
       * `2021-04-22` a las `16:00:00`: $3,277 \times 10^{16}\text{ kW}$
       * `2021-06-12` a las `01:00:00`: $4.111.695\text{ kW}$
     * En el conjunto de columnas de `Consolidado EE` se detectaron **81 filas** con valores $\ge 1.000.000\text{ kW}$ y **1.822 filas** con valores negativos ($< 0\text{ kW}$).
  3. **Días sin reinicio del totalizador a medianoche (`00:00:00`):**
     * Se detectaron **2 días** con `Frio (Kw) > 0` a las `00:00:00`: el **`2020-07-11`** (inicia en **`29.113,0 kW`** y `Planta (Kw) = 70.738,47 kW`, confirmando que no se reinició a medianoche) y el **`2022-02-05`** (inicia en `7,0 kW`, valor marginal).
* **Estadísticas del consumo diario de `Frio (Kw)` a las `23:59` (`1.179` días válidos):**
  * **Media:** `23.709,07 kW` | **Desvío estándar:** `7.720,40 kW`
  * **Mínimo:** `2.310,00 kW` | **Percentil 25%:** `17.996,00 kW` | **Mediana (50%):** `23.778,00 kW` | **Percentil 75%:** `29.345,50 kW` | **Máximo:** `56.128,00 kW`.

---

### 4. Cómo Debemos Seguir en la Próxima Sesión (Plan para la Sección 1.2: Análisis Temporal)

A partir de los resultados obtenidos hoy en `df_diario_eda` (`1.179` días válidos entre julio 2020 y octubre 2023), el camino metodológico para resolver la **Sección 1.2** y las visualizaciones temporales de la **Sección 1.4** es el siguiente:

1. **Verificar el máximo de `56.128,00 kW` en `df_diario_eda`:**
   * Como la documentación indica que el rango típico diario de `Frio (Kw)` llega hasta `35.000 kW`, antes de graficar verificaremos qué días superan los `40.000 kW` a las `23:59`: comprobaremos si alguno de esos días tuvo una falla de reinicio a la `01:00:00` (aunque a las `00:00:00` figurara en `0`) o si son picos reales de verano.
2. **Evolución histórica y patrones anuales/mensuales (Estacionalidad):**
   * Crear las columnas temporales `anio`, `mes` y `dia_semana` a partir de `DIA`.
   * Graficar la **serie temporal completa de `Frio (Kw)`** (con media móvil de 30 días superpuesta para visualizar la tendencia y los ciclos verano/invierno).
   * Construir los **boxplots mensuales (`mes` del 1 al 12)** y comparar promedios por año y por mes. Dado que la planta está en **Tecate (Baja California, México — Hemisferio Norte)**, verificaremos el fuerte incremento estacional en los meses cálidos de verano boreal (junio a septiembre) frente al invierno (diciembre a febrero).
3. **Medición cuantitativa del efecto del Día de la Semana (Punto crítico de la consigna 1.2 y Conclusión 3):**
   * La consigna exige explícitamente: *"Midan el efecto del día de la semana sobre el consumo, en lugar de asumirlo. Si el efecto resulta despreciable, díganlo con el número en la mano y descarten esas variables justificando la decisión"*.
   * Calcularemos la **media, mediana y desvío estándar de `Frio (Kw)` para cada uno de los 7 días de la semana** (Lunes a Domingo), el porcentaje de variación entre días laborales y fin de semana, y aplicaremos un **test estadístico ANOVA de una vía (`scipy.stats.f_oneway`) / Kruskal-Wallis** junto con el tamaño del efecto ($\eta^2$, porcentaje de varianza explicada por el día de la semana).
   * Físicamente, debido a la **inercia térmica de la fermentación y guarda en Bodega** (los tanques de fermentación deben mantenerse refrigerados las 24 horas los 7 días de la semana sin importar si es domingo), es muy probable que el efecto del día de la semana sea estadísticamente despreciable frente a la estacionalidad mensual. Con el $p$-valor y el porcentaje de varianza en mano, dejaremos fundamentada la decisión.
4. **Distribución del Target (`Frio (Kw)`):**
   * Generar el **histograma con curva de densidad (KDE)** y el **gráfico QQ-plot (`scipy.stats.probplot`)** para evaluar la normalidad y simetría de la variable objetivo.

