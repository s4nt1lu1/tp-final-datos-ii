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

---

## Sesión 2 — 2026-11-11 / 2026-11-12: Registros inválidos ocultos y serie temporal (Sección 1.2, pasos 1 y 2)

* **Rama:** `feature/eda`
* **Fase / Sección:** Fase 1 — Sección 1.2 (Análisis Temporal), pasos 1 y 2 del plan de la Sesión 1.
* **Commits:**
  * `0a20e10` — `fix(eda): verificar_checksums verifica solo los Excel crudos por defecto`
  * `9af02f7` — `feat(eda): verifico dias de mas de 40.000 kW con incrementos horarios y clasifico registros invalidos`
  * `b088aca` — `feat(eda): serie temporal con media movil de 30 dias y analisis de fiestas de fin de año`
* **Archivos modificados:**
  * `src/auxiliar_functions.py` *(constante `ARCHIVOS_CRUDOS` y parámetro opcional `archivos` en `verificar_checksums`)*
  * `notebooks/eda.ipynb` *(celdas de la Sección 1.2 a continuación de la construcción de `df_diario_eda`)*

---

### 1. Objetivo de la Sesión
1. Reproducir el proyecto desde un clon limpio en otra máquina (Windows) y corregir lo que impidiera ejecutarlo.
2. Resolver el **paso 1** del plan de la Sesión 1: verificar si los días con más de `40000 kW` son picos reales o errores del medidor.
3. Resolver el **paso 2**: graficar la serie temporal de `Frio (Kw)` con media móvil de 30 días e identificar los patrones estacionales y anuales.

---

### 2. Razonamiento Metodológico y Decisiones Tomadas

1. **Verificación de checksums en un clon limpio:**
   * `verificar_checksums()` recorría todas las entradas de `data/checksums.json`, incluida `dataset_unificado_inicial.csv`, que no se versiona y que el notebook genera más abajo. En un clon nuevo, la primera celda fallaba con `FileNotFoundError`.
   * **Decisión aplicada:** la lista de los 3 Excel se centralizó en la constante `ARCHIVOS_CRUDOS`, que usan tanto `generar_checksums_crudos()` como `verificar_checksums()`. Por defecto se verifican solo los crudos; los derivados se verifican pasándolos en el parámetro `archivos` una vez generados. Si se pide un archivo sin hash registrado, se lanza un error que indica cuál falta.
2. **Uso del incremento horario para validar los cierres diarios:**
   * `Frio (Kw)` es un totalizador que se reinicia a las `00:00` y acumula durante el día. La diferencia entre lecturas consecutivas **del mismo día** (`groupby("DIA")["Frio (Kw)"].diff()`) es el consumo de cada hora. Un error del medidor deja una huella distinta en los incrementos según cómo falle: una lectura errónea aislada (sube y baja), un salto que se queda (sube y no baja, el cierre queda inflado), un contador congelado (incrementos en 0, el cierre queda bajo) o un reinicio (incremento negativo).
   * El análisis se hace sobre una copia (`df_horario`) para no modificar `df_ee_unificado`, que usan otras secciones.
3. **El filtro de la celda de `df_diario_eda` solo mira el cierre:**
   * Las condiciones `Frio (Kw) >= 0` y `<= 100000` se aplican sobre `df_diario_bruto`, que contiene únicamente la fila de las `23:59`. Un valor imposible a otra hora del día no las afecta. Los criterios de días incompletos y sin reinicio sí consideran el día completo.
   * **Decisión aplicada:** los días con un valor imposible en una hora intermedia **se mantienen** si el acumulado retoma la progresión normal, porque su cierre de las `23:59` es correcto. Para los análisis sobre datos horarios se excluyen los saltos mayores a `1000000 kW`.
4. **Umbral para un salto horario anómalo:**
   * Se toma el **percentil 99,9 del incremento horario (`5153 kW`)**: menos de 1 hora cada 1.000 consume eso en condiciones normales, por lo que un salto mayor no puede explicarse por consumo real.
5. **Atípico no es lo mismo que inválido:**
   * Un día **inválido** es un error de medición (el cierre no representa el consumo real) y se trata en la Fase 2.1. Un día **atípico pero válido** es un consumo real fuera de lo habitual (por ejemplo, una parada de planta) y se conserva, documentándolo como limitación del modelo.
6. **Serie sobre el calendario completo:**
   * La serie diaria se reindexó con `asfreq("D")` para que los días faltantes queden como `NaN`. Así, `rolling(30)` abarca 30 días de calendario (no 30 filas) y el gráfico no une con rectas los huecos. Se usó `min_periods=15` para no promediar ventanas con muy pocos datos.

---

### 3. Hallazgos y Resultados Cuantitativos

* **Incremento horario de `Frio (Kw)` en días válidos** (sin las 3 lecturas del orden de los millones, `28.286` horas):
  * **Mediana:** `970 kW/h` | **Media:** `988 kW/h` | **P90:** `1487 kW/h` | **P99:** `1923 kW/h` | **P99,9:** `5153 kW/h`.
* **Lecturas imposibles aisladas dentro de días válidos:** 3 lecturas que generan 6 saltos (±`3,28·10¹⁶ kW` y ±`4,1·10⁶ kW`), las mismas detectadas en la Sección 1.1. Entre la hora anterior y la posterior al error, el acumulado aumenta `1.390 kW` (`2021-01-31`, 11:00), `2.379 kW` (`2021-04-22`, 16:00) y `3558 kW` (`2021-06-12`, 01:00), compatibles con dos horas de consumo normal en cada época del año. **El cierre es correcto y los días se mantienen.**
* **Días con el acumulado que baja en alguna hora:** **19 días**, clasificados en:
  * **Error que vuelve en la hora siguiente** (subida y bajada de tamaño parecido; cierre correcto o inflado entre `1000` y `3000 kW`): 10 días, por ejemplo `2021-03-18` (`+29166` y `−27138 kW`).
  * **Error que vuelve varias horas después** (cierre correcto): `2020-09-11` y `2021-04-04` (`+33521 kW` a las 15:00 y `−31131 kW` a las 18:00).
  * **Contador congelado:** `2021-01-03`. Entre las 09:00 y las 18:00 hay 10 lecturas consecutivas con incremento ≤ `10 kW`. Cierre de `4925 kW`, el **36 %** de la media de los cierres de enero (`13760 kW`). **Inválido.**
  * **Ruido de medición** (bajadas de `13` y `40 kW`): `2021-01-21` y `2021-07-14`.
  * **Sin revisar** (bajadas de `2000` a `7000 kW` sin subida en la hora anterior): `2021-06-27`, `2021-08-03`, `2022-04-16` y `2022-07-03`.
* **Días con cierre mayor a `40000 kW`:** **18 días**, todos entre abril y agosto (13 de 2021, 4 de 2022 y 1 de 2020).
  * **13 días son picos reales de verano:** sus saltos horarios máximos van de `1975` a `2359 kW`.
  * **5 días tienen un salto mayor al P99,9 que no se compensa con una bajada posterior** (cierre inflado):

    | Día | Cierre (kW) | Hora del salto | Salto (kW) | % del día explicado por esa hora |
    |---|---|---|---|---|
    | 2021-08-01 | 56128 | 22:00 | 27963 | 50 % |
    | 2021-07-21 | 54681 | 19:00 | 17689 | 32 % |
    | 2021-07-27 | 41148 | 23:59 | 14751 | 36 % |
    | 2021-08-24 | 42604 | 23:59 | 11161 | 26 % |
    | 2021-07-12 | 46001 | 21:00 | 6597 | 14 % |

  * El máximo de `56128 kW` reportado en la Sección 1.1 es un error del medidor; **el máximo real de la serie es `48046 kW` (`2021-06-22`)**.
* **Días marcados como inválidos para la Fase 2.1 (6):** `2021-01-03` (contador congelado), `2021-08-01`, `2021-07-21`, `2021-07-27`, `2021-08-24` y `2021-07-12` (salto sin compensar).
* **Serie temporal** (`1212` días de calendario, `1179` con dato, **`33` faltantes**):
  * **Estacionalidad anual marcada:** valle en diciembre–enero y pico entre junio y agosto (verano boreal en Tecate). Media móvil de 30 días: valles de `~11000 kW` (2021), `~10200 kW` (2022) y `~12400 kW` (2023); picos de `~31300 kW` (2020), `~36400 kW` (2021), `~34000 kW` (2022) y `~28900 kW` (2023). Del invierno al verano el consumo se multiplica por ~3.
  * **Tendencia descendente entre años:** los picos de verano bajan de `36400` a `34000` y a `28900 kW`.
* **Fiestas de fin de año:** 8 de los 10 días con menor consumo caen entre el 20 de diciembre y el 6 de enero (de los otros dos, `2021-01-03` es el contador congelado y `2021-01-08` queda a revisar). Respecto de la media de diciembre–enero sin fiestas:
  * **2020/21** (media `13596 ± 3268 kW`): mínimo `6808 kW` el `2020-12-28` (z ≈ 2,1).
  * **2021/22** (media `13357 ± 3294 kW`): **5 días consecutivos con |z| > 3**, del `2021-12-29` al `2022-01-03` (entre `2310` y `3106 kW`, z entre 3,11 y 3,35). El `2022-01-04` vuelve a `16176 kW`.
  * **2022/23** (media `15832 ± 3321 kW`): mínimo `6488 kW` el `2022-12-29` (z ≈ 2,8).
  * El `2022-01-03` (`2310 kW`, mínimo de la serie) **no es un contador congelado**: tiene ceros sueltos alternados con horas de `100` a `200 kW`, y el acumulado nunca baja ni se detiene más de 2 horas. Es un sistema a carga mínima, probablemente por una **parada de planta**. **Atípico pero válido: se conserva.** Es una limitación del modelo, porque nada en los datos del día anterior anticipa la parada.

---

### 4. Problemas Técnicos Encontrados y Solución Aplicada

* **`verificar_checksums()` fallaba en un clon limpio** (`FileNotFoundError` por `dataset_unificado_inicial.csv`): resuelto en el commit `0a20e10` (ver Decisión 1).
* **`pywinpty==3.1.2` en `requirements.txt`:** es un paquete solo de Windows. En Linux y macOS, `uv pip install -r requirements.txt` falla al compilarlo, lo que afecta la condición de aprobación 3. Solución propuesta (pendiente de decisión del grupo): `pywinpty==3.1.2; sys_platform == "win32"`.
* **El hash de `dataset_unificado_inicial.csv` depende del sistema operativo:** en Windows `to_csv` escribe saltos de línea `\r\n` y en Linux/macOS `\n`, por lo que el mismo contenido genera hashes distintos (`41e1d971…` en Windows, `405937be…` en Linux). Al ejecutar el notebook en Linux se modifica `data/checksums.json`. Solución propuesta (pendiente de decisión del grupo): `to_csv(..., lineterminator="\n")`.
* **Instalación en Windows:** PowerShell bloquea `.venv\Scripts\activate` por la política de ejecución de scripts. Se resuelve una vez con `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

---

### 5. Próximos Pasos (resto de la Sección 1.2)

1. **Boxplots mensuales** y tabla de media, mediana y desvío de `Frio (Kw)` por mes y por año; cuantificar cuánta varianza explica el mes.
2. **Efecto del día de la semana:** media, mediana y desvío por día, diferencia entre laborables y fin de semana, ANOVA / Kruskal-Wallis y $\eta^2$. Dado que la estacionalidad es tan marcada, medirlo también sobre el consumo sin estacionalidad (por ejemplo, restando la media móvil), porque la variación entre meses puede ocultar un efecto semanal chico.
3. **Patrón diario:** perfil del consumo por hora del día a partir de los incrementos horarios.
4. **Distribución del target:** histograma con KDE y QQ-plot.
5. Revisar los 4 días con bajadas sin subida previa y el `2021-01-08`.

---

## Sesión 4 — 2026-11-14

* **Rama:** `feature/eda`
* **Notebook:** `notebooks/eda.ipynb` (celdas 56–74)
* **Objetivo:** Auditoría de cierres diarios pendientes, cierre completo de la Sección 1.2 (estacionalidad mensual, efecto semanal, perfil horario) y desarrollo de la Sección 1.4 (distribución del target y desglose de consumo por áreas de planta).

### 1. Qué se Hizo y Resultados Clave

1. **Auditoría de fechas en revisión y consolidación de la base limpia (`1.173` días):**
   * Se verificaron `2021-01-08` (`5.578 kW`), `2021-06-27`, `2021-08-03`, `2022-04-16`, `2022-07-03` y `2023-05-20`: corresponden a saltos intermedios compensados dentro del acumulado horario o a paradas válidas de baja carga invernal.
   * **Regla consensuada del equipo:** 17 días inválidos totales (11 descartados en `df_diario_eda` + 6 errores de medidor: `2021-01-03`, `2021-07-12`, `2021-07-21`, `2021-07-27`, `2021-08-01`, `2021-08-24`), quedando `df_diario_limpio` con **`1.173` días válidos**.

2. **Estacionalidad mensual (`Frio (Kw)`):**
   * **Valle invernal:** diciembre (`12.699,2 kW` media; mediana `12.749,0`; mín `2.706,0`) y enero (`13.856,8 kW` media; mediana `14.793,0`; mín `2.310,0`; `n = 91` días).
   * **Pico de verano:** julio (`30.650,5 kW` media; mediana `30.534,0`; máx `41.206,0`; `n = 117`), junio (`30.451,5 kW` media; mediana `31.116,0`; máx `48.046,0`; IQR `9.455,6`; `n = 90`) y agosto (`28.970,1 kW` media; `n = 111`). El aumento promedio invierno–verano supera el **`+141%`** (~2,4×).
   * ANOVA por `mes_numero`: $F = 152,35$, $p = 2,92 \times 10^{-216}$, **$\eta^2 = 59,07\%$** de la varianza diaria explicada por el mes.

3. **Efecto del día de la semana (ANOVA de un factor y $\eta^2$):**
   * Lunes–viernes (`23.752,5 kW`) vs. fin de semana (`23.293,8 kW`): diferencia de **`-458,7 kW` (`-1,93%`)**.
   * Serie cruda: $F = 1,0839$, $p = 0,3700$ ($> 0,05$), $\eta^2 = 0,55\%$.
   * Serie desestacionalizada (restando media propia mes-año): $F = 4,1915$, $p = 0,0004$ (miércoles `+984,5 kW`, jueves `+512,7 kW` vs. viernes `-740,3 kW`, sábado `-436,7 kW`), $\eta^2 = 2,11\%$.

4. **Perfil horario (`01:00`–`23:59`, incrementos filtrados en `[0, 5.153]` kW):**
   * Carga base continua 24/7: mínimo promedio a las `07:00` (`921,1 kW/h`, mediana `906,0`) y máximo a las `14:00` (`1.038,6 kW/h`, mediana `1.023,0`), con meseta alta entre `13:00` y `21:00` (`1.007,6–1.038,6 kW/h`). Amplitud intradiaria de solo `117,5 kW/h` (`+12,8%`; la hora valle es el `88,7%` de la hora pico). El cierre `23:59` suma en promedio `954,0 kW` (mediana `949,0`).
   * Por estación, la tasa horaria de verano (`~1.170–1.330 kW/h`) casi duplica la de invierno (`~585–675 kW/h`).

5. **Distribución del target `Frio (Kw)` (`N = 1.173`):**
   * Media `23.621,1 kW` | Mediana `23.701,0 kW` | Desvío `7.543,0 kW` (`CV = 31,9%`) | Mín/Máx `2.310,0 / 48.046,0 kW`.
   * Asimetría `-0,066` y curtosis en exceso `-0,435` (distribución platicúrtica con meseta `17.000–31.000 kW` por superposición de regímenes estacionales).
   * Shapiro-Wilk $W = 0,9957$ ($p = 2,27 \times 10^{-3}$) y D'Agostino $K^2 = 15,8620$ ($p = 3,59 \times 10^{-4}$): el cuerpo central se alinea bien con la normal en el QQ-plot, con desvíos esperables en las colas acotadas (`0 kW` y capacidad máxima instalada).

6. **Consumo por áreas de planta (`Consolidado EE` a las `23:59`, filtro exploratorio `[0, 150.000]` kW):**
   * Total `Planta (Kw)`: media `63.748,8 kW` | mediana `66.802,6 kW`.
   * Macro-áreas: `Servicios` `38.706,2 kW` (`60,7%`, $r = 0,947$), `Envasado` `14.152,9 kW` (`22,2%`, $r = 0,543$), `Elaboracion` `6.973,0 kW` (`10,9%`, $r = 0,729$), `Restos Planta` `4.038,1 kW` (`6,3%`, $r = 0,115`).
   * Sub-áreas principales: **`Frio` `23.621,1 kW` (`37,1%`, mayor consumidor individual)**, `Aire` `6.402,2 kW` (`10,0%`, $r = 0,605$), `Bodega` `6.008,3 kW` (`9,4%`, $r = 0,651$), `Linea 4` `6.379,2 kW` (`10,0%`, $r = 0,479`), `Linea 3` `5.388,1 kW` (`8,5%`, $r = 0,418`), `Linea 2` `4.423,9 kW` (`6,9%`, $r = 0,370`), `Cocina` `1.725,4 kW` (`2,7%`, $r = 0,448`). El resto de sub-áreas (`Pta Agua / Eflu`, `Efluentes`, `Calderas`, `Prod Agua`) suman entre `0,9%` y `2,5%` cada una.
   * La correlación `r = 0,947` con `Servicios` es estructural (`Frio` está contenido dentro de `Servicios`), mientras que los vínculos operativos reales más claros aparecen con `Elaboracion`/`Bodega` ($r = 0,651$) y `Aire` ($r = 0,605$).

### 2. Decisiones Metodológicas Tomadas

* **Descarte de variables calendario semanales:** Se descartan `dia_semana` y `es_fin_de_semana` para el modelado principal; en una planta 24/7 con carga térmica continua, la varianza semanal es marginal (`0,55%` cruda, `2,11%` desestacionalizada) frente al `59,07%` explicado por el mes.
* **Mantener `Frio (Kw)` en escala original (`kW`):** Sin transformaciones logarítmicas ni Box-Cox dada la simetría prácticamente nula (`-0,066`) y para preservar la interpretabilidad directa de la métrica objetivo (`MAE < 3.000 kW`).
* **Uso de ANOVA paramétrico unificado:** Dado `N ≈ 167` observaciones por día de la semana (TLC), se mantuvo ANOVA de un factor + $\eta^2$ en el notebook para mantener la presentación clara y defendible.

### 3. Próximos Pasos

1. Integrar el análisis multivariado/producción (Sección 1.3) y consolidar el dataset unificado multi-hoja aplicando el criterio consensuado de `1.173` días válidos.
2. Avanzar a la Fase 2 (Preprocesamiento y feature engineering temporal/climático) en la rama `feature/preprocessing`.

---

## Sesión 5 — 2026-11-15

* **Rama:** `feature/preprocessing`
* **Notebooks y scripts trabajados:**
  * `notebooks/correccion_fechas.ipynb` y `src/correccion_fechas.py` *(unificación multi-hoja y filtrado horario sin corrimiento de fechas)*
  * `notebooks/preprocesamiento_lio.ipynb` *(Sección 2.2A: clima de Tecate con radiación solar, variables autorregresivas de `Frio (Kw)` y calendario)*
  * `notebooks/preprocesamiento_gena.ipynb`, `src/limpieza_predictoras.py` y `src/features_planta.py` *(Secciones 2.1 y 2.2B: selección de representantes sin proxies, limpieza e imputación ajustadas en `train`, ratios, eficiencia vs. `Meta` e interacciones)*
  * `src/preprocessing_pipeline.py` *(Sección 2.4: partición temporal 70/15/15, `StandardScaler` en `train`, checksums y `data_lineage.json`)*
  * `notebooks/eda.ipynb` y `requirements.txt` *(cierre de Secciones 1.3–1.4 y correcciones de reproducibilidad multiplataforma)*

### 1. Qué se Hizo y Resultados Clave

1. **Corrección de `notebooks/correccion_fechas.ipynb` y generación de `data/processed/dataset_trabajo.csv` (commit `7188628`):**
   * Se eliminó el desplazamiento acumulado de fechas (`cumsum()` sobre bloques repetidos `> 25` filas) en la Celda 5, conservando únicamente el `Bloque 0` (`00:00` a `23:59`) de cada día y evitando desfasar meses de `< 31` días entre hojas o inventar fechas en noviembre de 2023.
   * En la Celda 11 se unificaron las reglas consensuadas de filtrado sobre `Frio (Kw)_Consolidado EE`: `MIN_HORAS_DIA = 24`, reinicio a las `00:00 <= 100 kW`, exclusión de contador congelado (`2021-01-03`) y de saltos horarios positivos `> P99,9 = 5.153 kW` sin compensación posterior, más validación MAD (`UMBRAL_MAD = 3,5`).
   * **Resultado:** `data/processed/dataset_trabajo.csv` con **`29.296` filas × `421` columnas**, **`1.172` días válidos** reales (`2020-07-01` a `2023-10-25`) y hash `SHA-256: bd015abb722ba2151d64ae9f229364cd61861b8682ffe846084d215aa083ae78` registrado en `data/checksums.json` (exportado con `lineterminator="\n"`).

2. **Desarrollo de la Sección 2.2A en `notebooks/preprocesamiento_lio.ipynb` (commit `0f8b432`):**
   * **Alineación del target $D+1$:** sobre los `1.172` cierres diarios (`HORA == "23:59"`), se construyó `Frio_manana_y` verificando contigüidad calendario exacta (`(DIA.shift(-1) - DIA).dt.days == 1`), obteniendo **`1.140` pares consecutivos válidos ($D \rightarrow D+1$)**.
   * **Clima diario de Tecate + Radiación Solar (Open-Meteo, `32,5722° N, -116,6283° W`):** descarga reproducible con caché local en `data/raw/clima_tecate_openmeteo.csv` (`1.212` días, `0` nulos) incluyendo temperaturas (`temp_max_c`, `temp_min_c`, `temp_media_c`, `temp_aparente_c`) y radiación solar acumulada (`radiacion_mj_m2`), más rezagos y medias móviles de 3 y 7 días.
     * **`radiacion_roll7`** resultó el mejor predictor climático del consumo de frío de mañana ($r = 0,742$, Spearman $= 0,743$, `0` nulos), seguido por `radiacion_roll3` ($r = 0,719$), `radiacion_mj_m2` ($r = 0,670$), `temp_aparente_c` ($r = 0,576$, Spearman $= 0,593$), `temp_media_roll7` ($r = 0,560$) y `temp_min_c` ($r = 0,551$, Spearman $= 0,574$). Se descartó `amplitud_termica_c` ($r = 0,105$).
   * **Variables autorregresivas de `Frio (Kw)` y calendario sobre grilla continua (`asfreq("D")`):**
     * **`frio_roll_mean_7`** ($r = 0,843$, Spearman $= 0,847$, solo `3` nulos) y **`frio_roll_mean_14`** ($r = 0,842$, Spearman $= 0,850$, `6` nulos) superan al dato aislado del día `frio_hoy_d` ($r = 0,841$, `0` nulos) y a los rezagos puntuales `frio_lag_1` ($r = 0,744$, `32` nulos), `frio_lag_7` ($r = 0,736$, `46` nulos) y `frio_lag_2` ($r = 0,725$, `39` nulos).
     * Se incorporaron `frio_roll_std_7` ($r = 0,178$, volatilidad semanal), `frio_diff_1` ($r = 0,163$, cambio en 24 h) y `mes_numero` ($r = -0,038$ lineal por simetría invernal-estival, pero no linealmente explicativo según ANOVA de Fase 1 con $\eta^2 = 59,07\%$).

3. **Mejoras de reproducibilidad y orden del repositorio (commit `0f8b432`):**
   * En `requirements.txt` se acotó `pywinpty==3.1.2; sys_platform == "win32"` para garantizar que `uv pip install -r requirements.txt` funcione sin errores en Linux y macOS (Condición de Aprobación 3).
   * En `notebooks/eda.ipynb` se redirigió la exportación de `dataset_unificado_inicial.csv` desde `RAW_DATA_DIR` hacia `PROCESSED_DATA_DIR` (`data/processed/dataset_unificado_inicial.csv`) con `lineterminator="\n"`.

4. **Avances integrados del equipo en `feature/preprocessing` (commits `7ef90cc` a `06c2dfc`):**
   * **Cierre de Fase 1 (`7ef90cc` / `af38a41`):** integración completa de la Sección 1.3 y scatter plots del Top 5 en `notebooks/eda.ipynb` y `notebooks/exploracion_produccion.ipynb`.
   * **Sección 2.1 (`2270e98` y `c5c61c1`, `src/limpieza_predictoras.py`):** selección sin fuga sobre `train` (`820` días, hasta `2022-10-27`) de **15 predictoras operativas** de planta (excluyendo proxies, columnas con `> 20%` nulos en `train`, $|\rho_{\text{Spearman}}| < 0,40$ en `train` y colineales $|\rho| \ge 0,90$), seguidas por `LimpiadorExtremos` (`< 0` o `> 5 × P99_train` $\rightarrow$ `NaN`: 9 valores imposibles detectados) e `ImputadorTemporal` (`ffill(limit=3)` + mediana de `train`, quedando `0` nulos).
   * **Sección 2.2B (`06c2dfc`, `src/features_planta.py`):** transformador `FeaturesPlanta` ajustado solo en `train` que construye ratios entre áreas (`ratio_frio_servicios` $\rho = 0,714$), eficiencia energética contra meta mensual (`eficiencia_frio_vs_meta` $\rho = 0,301$, imputando metas previas a jul-2021 con la mediana mensual de `train`) e interacciones producción × clima, destacándose **`bodega_x_radiacion_roll7` ($\rho_{\text{Spearman}} = 0,767$ en `train`)**.
   * **Sección 2.4 y modularización (`9648c53`, `src/correccion_fechas.py` y `src/preprocessing_pipeline.py`):** implementación de `temporal_split()` (70/15/15 cronológico), `build_scaler_pipeline()` (`StandardScaler` ajustado solo en `X_train`), actualización de `data/checksums.json` y generación de `data/processed/data_lineage.json`.

### 2. Próximos Pasos (Cierre de Fase 2)

1. Unificar en el notebook entregable oficial **`notebooks/preprocesamiento.ipynb`** las Secciones **2.1**, **2.2A** y **2.2B** (calculando también las correlaciones de 2.2A sobre el tramo `train` para mantener criterio 100% homogéneo sin fuga temporal).
2. Ejecutar la **Sección 2.3 (Selección de Variables)** exclusivamente sobre `train` combinando todas las features candidatas (15 operativas limpias + 16 de clima/lags/calendario + 4 de ratios/meta/interacciones) mediante **al menos 2 métodos** (ej. **LASSO / `LassoCV`**, **Importancia con Random Forest** y **RFE** con `SEMILLA = 42`).
3. Conectar la función de Feature Engineering y la lista de variables ganadoras de la 2.3 en `run_pipeline()` de `src/preprocessing_pipeline.py` para exportar el **`data/processed/dataset_final.csv`** definitivo, actualizar `data/checksums.json` y `data/processed/data_lineage.json`, y abrir el Pull Request de `feature/preprocessing`.

---

## Sesión 6 — 2026-11-15

- **Autoría** · Lio con asistencia de Codex
- **Rama** · `feature/preprocessing`
- **Fase** · Cierre local de la Fase 2, secciones 2.1 a 2.4
- **Punto de partida** · `0203fd1`, verificado contra `origin/feature/preprocessing`
- **Archivos** · `notebooks/preprocesamiento.ipynb`, los scripts de limpieza, ingeniería, selección y exportación en `src/`, el caché climático fijo, `data/checksums.json`, `data/processed/data_lineage.json` y `README.md`

### Objetivo

Corregir el desfase entre los borradores, reunir el entregable de la Fase 2 en `preprocesamiento.ipynb` y dejar un script que reproduzca la selección y los datos procesados. Los borradores individuales y el notebook de corrección horaria se conservan en sus versiones publicadas como registro del trabajo del grupo. El corte y las cifras vigentes están en el entregable unificado.

### Razonamiento y decisiones

- Mantuvimos el cierre de las 23:59 y buscamos el consumo de D+1 por fecha exacta. Excluimos pares sin cierre válido de D o de D+1 antes de aplicar el reparto 70/15/15. Los cierres válidos sin etiqueta siguen en el historial para construir ventanas de fechas posteriores.
- En la limpieza horaria, la mediana y el MAD se calibran con el primer año, hasta el 30/06/2021. El P99,9 de incrementos de ese tramo fue 4.956 kW. Aplicamos la centena superior (5.000 kW) como umbral para evitar que diferencias de pocos kW decidan un descarte. La regeneración desde los tres Excel conservó 18 exclusiones y reprodujo el mismo CSV horario y SHA-256 del grupo.
- Para las predictoras partimos de hojas operativas elegidas por dominio. Aplicamos los criterios de completitud, Spearman y colinealidad con los pares de Train. Conservamos los mínimos de Lio para las medias de 3, 7 y 14 días y el desvío de 7 días. El arrastre de faltantes cuenta hasta tres días calendario y el resto usa medianas de Train.
- Calculamos ratios, eficiencia eléctrica por hectolitro contra Meta e interacciones. Los cocientes Frío/Servicios fuera de [0, 1] se tratan como faltantes. Las metas sustituidas usan la mediana mensual de Train. El clima histórico de Tecate quedó como diagnóstico reproducible con copia fija, porque falta verificar cuándo estuvo disponible cada observación al cierre de D. Para el modelo se evaluó una interacción entre producción de Bodega e índice estacional conocido desde el calendario.
- En 2.3 ajustamos LASSO estandarizada con `alpha=100` y Random Forest con semilla 42, usando solo Train. Fijamos un máximo de 12 variables para el bosque. Usamos el mismo Ridge estandarizado con `alpha=10` para comparar en Validation todos los subconjuntos y desempatar por cantidad de columnas a una precisión de 0,01 kW. El Test se reserva para el reporte de la Fase 3.
- En 2.4 exportamos los CSV sin escalar, ajustamos el `StandardScaler` dentro de un `Pipeline` solo con `X_train` y dejamos en el linaje los scripts, la fecha, las reglas, las variables, los cortes y las huellas. El preparador ajustado puede transformar un tramo con su historial sin necesitar la etiqueta futura.

### Hallazgos cuantitativos

- El CSV horario conserva 29.296 filas y 421 columnas, con 1.172 días a las 23:59 y SHA-256 `bd015abb722ba2151d64ae9f229364cd61861b8682ffe846084d215aa083ae78`.
- De las 1.172 fechas, 32 quedan sin etiqueta por huecos de calendario o por ser el último cierre. Quedaron 1.140 pares. El último D de Train es el 26/10/2022, el último D de Validation es el 29/04/2023 y el último D de Test es el 24/10/2023.
- El filtro operativo revisó 169 candidatas y retuvo 28. Con las derivadas quedaron 43. En Train, el mes explicó el 68,01 % de la variación del target, el día de semana el 0,46 % y el indicador de fin de semana el 0,07 %. Se sustituyeron 348 metas mensuales faltantes.
- Los baselines en Validation dieron 2.727,13 kW de MAE para persistencia y 2.395,32 kW para la media móvil de 7 días.

| Subconjunto | Predictoras | MAE de Ridge en Validation (kW) |
|---|---:|---:|
| Random Forest | 12 | 2.149,43 |
| Intersección | 9 | 2.227,85 |
| LASSO | 22 | 2.286,60 |
| Todas | 43 | 2.286,66 |

El subconjunto elegido por menor MAE fue el de Random Forest. Los shapes finales de X son **798 × 12**, **171 × 12** y **171 × 12**, respectivamente. `dataset_final.csv` quedó con SHA-256 `2d023012ed97d340f30d79d57744c71fd8575ac03b0b56347bad3817932b6228`.

### Controles de ejecución

- Verificamos el target con fechas desordenadas, día bisiesto, huecos, un cierre nulo, uno infinito y el último día. También comprobamos que cuatro días de separación agotaran un arrastre limitado a tres.
- Reejecutamos las celdas del notebook con Python 3.12.14 en Windows y ejecutamos los scripts desde un directorio nuevo partiendo de los tres Excel y de la copia fija del clima. Dos corridas independientes reprodujeron los cinco SHA-256 de salida.
- Alterar los cierres posteriores al primer D de Test mantuvo las features y la selección en Validation. Pasar 16 días de contexto también reprodujo exactamente las features del último cierre del historial. Se comprobaron los 116 requisitos aplicables del entorno.

### Estado al cierre

La Fase 2 está lista para revisar en la rama local. Queda pendiente integrar el trabajo mediante el Pull Request previsto en la consigna. La evaluación de Test, el registro de modelos y el script de predicción corresponden a las fases siguientes.

Se registró la comparación de ventanas y la justificación de la disponibilidad del clima histórico en [Decisiones a resolver antes de la Fase 3](DECISIONES%20A%20RESOLVER%20ANTES%20DE%20LA%20FASE%203.md).

---

## Sesión 7 — 2026-11-15

- **Autoría** · Lio con asistencia de Codex
- **Rama** · `feature/modeling`, creada desde el preprocesamiento del commit `53d3a62`
- **Fase** · 3.1 a 3.4, entrenamiento, evaluación y versionado
- **Archivos** · `notebooks/modelado.ipynb`, `src/train_model.py`, `results/experiment_logs.csv`, `results/seleccion_modelado.json`, `results/evaluacion_final.json`, `models/modelo_v1.0.0.pkl`, `models/model_registry.json` y `README.md`

### Objetivo

Comparar cuatro métodos de regresión y los dos baselines exigidos, elegir con Validation y dejar un modelo que pueda recuperar la Fase 4. También se abrió el [PR de EDA](https://github.com/s4nt1lu1/tp-final-datos-ii/pull/2) hacia `main`. El [PR de preprocesamiento](https://github.com/s4nt1lu1/tp-final-datos-ii/pull/1) queda después de EDA en el orden de revisión.

### Razonamiento y decisiones

- Calculamos primero **persistencia** y **media móvil de siete días**. La media utiliza la ventana calendario que termina en D y el mínimo de cuatro cierres establecido en Fase 2. Verificamos que ambas referencias cubrieran todas las fechas evaluadas. Así, los errores se comparan sobre las mismas filas y en las mismas unidades.
- Probamos **Ridge, Random Forest, XGBoost y LightGBM**. Usamos una grilla chica para ver el efecto de regularizar Ridge, limitar el tamaño de los árboles y cambiar la pérdida en los árboles. Probamos tres conjuntos de variables con 18 configuraciones por conjunto. Los cuatro estimadores se ajustaron en las mismas 798 filas de Train. El log registra parámetros y métricas de las 54 corridas candidatas y de las dos referencias de Validation.
- Elegimos por **MAE en Validation** porque el trabajo mide el error con esa métrica. Probamos las 12 columnas de Fase 2, conservar solo las medias de 3 y 14 días, o conservar la de 14. Ganó Ridge con `alpha=10` y las 12 columnas. La mejor variante con dos medias obtuvo **2.161,15 kW**. La ganadora conservó las tres porque obtuvo **2.149,43 kW** con la misma cobertura.
- Las variables meteorológicas siguen en diagnóstico. Para comparar radiación como predictora queda pendiente una fuente cuya disponibilidad al cierre de D se pueda verificar. Así se mantiene el mismo contrato temporal para el entrenamiento y la predicción futura.
- Una vez congeladas las columnas y los parámetros, conservamos el preparador de Fase 2 y reconstruimos las predictoras de Train y Validation para compararlas con los CSV. Reentrenamos Ridge y su escalador con **969 filas**. Se usó el commit `3df5f36db9993d80e9019492cbb1ae242cf1e608` y se abrió el Test local después del ajuste.

### Resultados en Validation

| Mejor configuración por familia | MAE (kW) | RMSE (kW) | R² |
|---|---:|---:|---:|
| Ridge | 2.149,43 | 2.790,71 | 0,461 |
| Random Forest | 2.300,33 | 2.871,57 | 0,429 |
| XGBoost | 2.300,50 | 2.904,68 | 0,416 |
| LightGBM | 2.404,29 | 2.965,39 | 0,391 |

Persistencia obtuvo **2.727,13 kW** y la media móvil de siete días **2.395,32 kW**. El residuo medio de Ridge, definido como real menos predicción, fue **−202,66 kW**. Eso indica una leve sobreestimación media en Validation. Al permutar las predictoras, la mayor subida del MAE vino de la media de frío de 14 días (**209,48 kW**) y después de la interacción de Bodega con la estacionalidad (**125,48 kW**). Esta medición describe qué señal usa el modelo en Validation. La relación causal queda fuera del alcance de la prueba.

### Evaluación final sobre las mismas 171 fechas de Test local

| Pronóstico | MAE (kW) | RMSE (kW) | R² |
|---|---:|---:|---:|
| Persistencia | 3.367,84 | 4.412,38 | 0,397 |
| Media móvil de siete días | 3.042,61 | 3.923,80 | 0,523 |
| Ridge final | **2.824,12** | **3.714,61** | **0,573** |

El modelo mejoró **218,49 kW (7,18 %)** respecto del mejor baseline y **543,73 kW (16,14 %)** respecto de persistencia. Como la ventaja sobre el mejor baseline quedó por debajo del 10 %, la decisión para el informe es discutir si esa reducción justifica mantener el preprocesamiento y las mediciones de planta. El modelo elegido es lineal y liviano, pero depende de más mediciones que el baseline. La evaluación de la cátedra con el archivo oculto sigue pendiente.

### Reproducibilidad y controles

- Clonamos la versión publicada de Fase 2 en otra carpeta, creamos un entorno virtual independiente de **Python 3.12.14** e instalamos las **116** dependencias de `requirements.txt` con `uv`. Copiamos solo los tres Excel de entrada. Regeneramos el CSV horario y los cinco CSV procesados con los mismos SHA-256.
- Repetimos la elección en el entorno nuevo. Coincidieron la configuración y las 56 filas de métricas de Validation al excluir las columnas con las fechas de cada corrida. Ejecutamos el notebook con el kernel de Jupyter del entorno nuevo.
- Cargamos el modelo guardado, verificamos su hash, sus 12 columnas y las 969 filas vistas por el escalador. A partir de los cierres diarios reconstruimos las predictoras de Validation y obtuvimos las mismas predicciones que al usar el CSV. También verificamos que el script detectara la evaluación persistida antes de intentar repetir Test.
- El SHA-256 del modelo `v1.0.0` es `ccbd269aaebf2232582ab6a6a051786e5bd3368b936a80a8b6495c84ea61c902`. El registro apunta al commit de entrenamiento. El tag de la versión identifica el commit que agrega el artefacto y el resultado final al repositorio.

### Próximos pasos

Revisar los PR por fase en orden EDA, preprocesamiento y modelado. En Fase 4 se implementa `predict.py` para cargar el registro, preparar un Excel nuevo con contexto histórico y producir una predicción diaria de D+1 con la versión elegida.

---

## Sesión 8 — 2026-11-15

- **Autoría** · Lio con asistencia de Codex
- **Ramas** · `feature/preprocessing` para la revisión de Fase 2 y `feature/modeling` para Fase 3
- **Archivos** · `notebooks/preprocesamiento.ipynb`, `notebooks/modelado.ipynb` y esta bitácora

### Objetivo

Mostrar las cuentas y decisiones de preprocesamiento en un orden que se pueda seguir desde los borradores de Lio y Genaro. Ubicar la ejecución de los dos baselines y el primer ajuste de los cuatro modelos en la sección 3.1, antes de la búsqueda de hiperparámetros.

### Cómo se revisó Fase 2

- En **2.1** mostramos los motivos de los 18 descartes horarios y un ejemplo con un día sin cierre. Buscamos D+1 por fecha y mostramos por qué contar primero los 1.172 cierres dejaba el último día de Train el 27/10/2022. Al repartir los 1.140 pares válidos, Train termina el 26/10/2022 y las particiones tienen 798, 171 y 171 filas.
- Tomamos los criterios de Genaro de faltantes, extremos y colinealidad. Su borrador partía de 28 columnas de EDA. El notebook unificado muestra la elección por dominio entre 169 candidatas y el filtro calculado con Train, que queda con 28 representantes. Graficamos el salto de Envasado antes y después de limpiarlo y mostramos la mediana de Train como respaldo para los huecos.
- En **2.2A** recuperamos las fórmulas de Lio para rezagos y ventanas calendario. Repetimos el ranking de Open-Meteo en Train y mostramos gráficamente la relación de radiación semanal con la etiqueta. Quedó como diagnóstico porque el caché no documenta cuándo se publicó cada observación. Mostramos también, con Train, el patrón mensual y la correlación entre las medias de 3, 7 y 14 días.
- En **2.2B** escribimos directamente los ratios, la imputación de meta mensual y las interacciones de Genaro. Contamos 348 metas sustituidas en Train y dos cocientes Frío/Servicios mayores a 1. En el gráfico de interacción, radiación sola tuvo Spearman 0,772 y Bodega por radiación 0,767 en Train. Para la lista candidata conservamos la interacción de Bodega con el índice estacional del calendario, que podemos calcular al cierre de D.
- En **2.3** ajustamos LASSO y el bosque en celdas separadas del notebook y comparamos sus propuestas con el mismo Ridge en Validation. De las 43 candidatas volvieron a quedar las mismas 12. La propuesta del bosque obtuvo 2.149,43 kW. Mantener las medias de 3 y 14 días dio 2.161,15 kW, y mantener solo la de 14 dio 2.193,52 kW para ese Ridge. La comparación mantiene la media de siete días como baseline de la fase siguiente.
- En **2.4** igualamos las variables y las etiquetas calculadas en el notebook con las del pipeline. Comprobamos que el escalador aprendiera de X_train, que el CSV saliera sin escalar y que los hashes y shapes coincidieran con el linaje. Las 12 columnas y el orden de exportación siguen siendo los que usa el modelo serializado.

### Cómo se revisó Fase 3

La sección 3.1 ahora calcula primero las predicciones de persistencia y de la media móvil. Después entrena una configuración inicial de Ridge, Random Forest, XGBoost y LightGBM, registra seis resultados y muestra sus métricas en las mismas fechas de Validation. En 3.2 seguimos con la búsqueda de 54 configuraciones, incluidas las variantes de ventanas identificadas en 2.3. La selección reproducida coincide con la elección publicada de Ridge con las 12 columnas. La tabla final se lee del resultado de Test ya registrado para esa configuración.

### Reproducibilidad

Regeneramos la limpieza horaria desde los tres Excel en un clon de GitHub con una instalación nueva de Python 3.12.14 y los 116 paquetes fijados. Ejecutamos todas las celdas de los dos notebooks en Jupyter. Revisamos seis gráficos de preprocesamiento y el gráfico de predicciones y residuos de modelado. Los cinco CSV de Fase 2 conservaron sus hashes. La ejecución de `python -m src.train_model` reprodujo la selección y las 56 filas de métricas de Validation al comparar sus valores sin las fechas de corrida. La evaluación final registrada corresponde al mismo modelo `v1.0.0`.
