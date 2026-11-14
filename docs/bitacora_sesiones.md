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

## Sesión 3 — 2026-11-13: Pipeline de Unificación Multi-Hoja y Auditoría Estructural/Física (`exploracion_santi.ipynb`)

* **Autor/es:** Santino Lui
* **Rama:** `dev/feature/eda`
* **Fase / Sección:** Fase 1 / Prototipo de Fase 2.1 — Unificación completa de todas las hojas de los Excel y auditoría estructural y física sobre el target (`Frio (Kw)_Consolidado EE`).
* **Archivos creados / modificados:**
  * `notebooks/exploracion_santi.ipynb` *(creado — Celdas 1 a 8)*
  * `docs/bitacora_sesiones.md` *(actualizado con la Sesión 3)*

---

### 1. Objetivo de la Sesión
1. Extender la carga y unificación de datos más allá de `Consolidado EE`, integrando **todas las hojas** de los tres archivos Excel crudos en una única tabla indexada por `(DIA, HORA)` con sufijos por hoja.
2. Diseñar y ejecutar un pipeline reproducible de normalización horaria, eliminación de registros duplicados/clonados y reparación de errores tipográficos en `HORA` (`limpiar_y_auditar_hoja`), registrando cada acción en contenedores de trazabilidad.
3. Aplicar una auditoría en dos instancias (estructural sobre `HORA` y física sobre `Frio (Kw)_Consolidado EE`) para cuantificar exactamente qué días se eliminan, qué filas se descartan y qué anomalías subsanables se conservan.

---

### 2. Razonamiento Metodológico (El "Por Qué")

1. **Integración de todas las hojas por `(DIA, HORA)` en lugar de trabajar solo con `Consolidado EE`:**
   * En las sesiones previas únicamente se había consolidado `Consolidado EE`, dejando fuera variables de producción, energía, agua, gas/vapor, aire, $\text{CO}_2$, efluentes, glicol y KPIs que serán necesarias como *features* exógenas.
   * **Decisión aplicada:** En cada archivo Excel se recorren todas sus hojas (por nombre, considerando que `2021_2023` incluye `'Seguimiento Dia'`), se exige la presencia de `DIA` y `HORA`, se renombran las columnas como `{col}_{hoja}` y se realiza un `pd.concat(..., axis=1, join="outer")` sobre `(DIA, HORA)`. Luego se apilan los tres archivos en orden de prioridad (`2022_2023` > `2021_2023` > `2020_2022`) con `drop_duplicates(subset=["DIA", "HORA"], keep="first")` y se coerciona toda columna numérica con `pd.to_numeric(..., errors="coerce")` para convertir errores de Excel (`#VALUE!`, `#DIV/0!`) en `NaN`.
2. **Clasificación de hojas según naturaleza física y definición de grilla horaria canónica:**
   * Se separaron explícitamente las hojas no acumulativas (`HOJAS_NO_ACUMULATIVAS = {"Consolidado KPI", "Auxiliar", "Seguimiento Dia"}`) del resto de hojas totalizadoras (`cols_acumulativas`), y se fijó `HORAS_CANONICAS` con los **25 tramos horarios esperados** (`00:00` a `23:00` más el cierre `23:59`).
3. **Limpieza y normalización de `HORA` dentro de cada hoja antes de unir (`limpiar_y_auditar_hoja`):**
   * **Prefijos de fecha anómalos:** Se detectan cadenas con fecha antepuesta distinta de `1899-12-30` (como `1900-01-01`), se registra la reparación y se recorta el prefijo conservando la hora, además de rellenar ceros a la izquierda (`^(\d):` $\rightarrow$ `0\1:`) y truncar segundos residuales.
   * **Registros `'23:59:59'`:** Se eliminan explícitamente antes de truncar segundos para no colisionar ni sobrescribir el cierre real de las `23:59`.
   * **Bloques horarios clonados (meses de `< 31` días):** En lugar de cortar ciegamente con `.head(25)`, se detecta el reinicio de jornada dentro de una misma fecha mediante `(HORA == "00:00") & (DIA == DIA.shift(1))` y se descarta todo bloque con `id_bloque > 0`.
   * **Reparación posicional de typos humanos en `HORA`:** Si un día completo de 25 registros inicia en `00:00` y cierra en `23:59` pero difiere de `HORAS_CANONICAS`, se realinea contra `HORAS_CANONICAS`; si la jornada tiene otra longitud, se corrige cualquier salto unitario rodeado de horas válidas (`idx_n - idx_p == 2`).
4. **Auditoría diaria en dos instancias y criterio para el pico aislado a la `01:00`:**
   * **Instancia 1 (Estructural sobre `HORA`):** Exige inicio en `00:00`, cierre en `23:59` y al menos `24` registros por día para descartar jornadas truncadas antes de evaluar valores físicos.
   * **Instancia 2 (Física sobre `Frio (Kw)_Consolidado EE` con tolerancia `TOL = 1e-3`):** Exige datos numéricos no vacíos, ausencia de negativos (`< 0`), cierre `23:59` positivo y $\le 100.000\text{ kW}$, reinicio a las `00:00` (`val_00 <= 5000` y `val_00 <= val_01 + TOL`) y monotonía creciente (`diffs >= -TOL`).
   * **Excepción controlada para pico a la `01:00`:** Si `val_01 > val_02 + TOL` pero desde las `02:00` hasta las `23:59` la serie crece monótonamente, el día se conserva y se registra en `log_reparaciones` (caso `2021-06-12`), evitando perder una jornada cuyo cierre diario es físicamente consistente.

---

### 3. Hallazgos Cuantitativos y Resultados Clave

* **Verificación inicial e inspección de hojas:**
  * `verificar_checksums()` validó los **3 archivos Excel crudos** (`True`).
  * Se confirmaron **15 hojas** en `2020_2022` y `2022_2023`, y **16 hojas** en `2021_2023` (hoja adicional: `Seguimiento Dia`).
* **Dimensiones del dataset unificado y resultado global del filtrado:**
  * **Pre-filtrado de días (`df_unificado`):** **`29.700` filas** | **`1.190` fechas únicas**.
  * **Días eliminados en total (`df_bajas_dias`):** **`31` días** (**`729` filas** descartadas).
  * **Dataset limpio conservado (`df_limpio`):** **`1.159` días válidos** (**`28.971` filas**).
* **Desglose de los `31` días eliminados (`log_dias_eliminados`):**
  * **Por archivo de origen:** `13` días de `2020_2022`, `10` días de `2021_2023` y `8` días de `2022_2023`.
  * **Instancia `1. Estructural (Horas)` — `8` días (`154` filas):**
    * `2021-07-13` (`2021_2023`, 23 filas: sin cierre `23:59`, última hora `23:00` | `23/25`), `2021-11-01` (`2021_2023`, 23 filas: `23/25`), `2022-03-02` (`2021_2023`, 24 filas: sin cierre `23:59`, última hora `23:00`), `2023-01-17` (`2022_2023`, 16 filas: `16/25`), `2023-04-13` (`2022_2023`, 20 filas: sin cierre `23:59`, última hora `19:00` | `20/25`), `2023-04-19` (`2022_2023`, 17 filas: sin cierre `23:59`, última hora `16:00` | `17/25`), `2023-07-01` (`2022_2023`, 23 filas: primera hora `02:00` | `23/25`) y `2023-10-26` (`2022_2023`, 8 filas: sin cierre `23:59`, última hora `07:00` | `8/25`).
  * **Instancia `2. Física (Target Acumulativo)` sobre `Frio (Kw)_Consolidado EE` — `23` días (`575` filas):**
    * **Sin reinicio a las `00:00` + caída:** `2020-07-11` (`2020_2022`: `00:00 = 29.113,0 kW`, cae `-29.113,0 kW` a las `01:00`).
    * **Cierre `23:59` nulo:** `2023-02-28` (`2022_2023`: cierre `NaN`).
    * **Violaciones de monotonía con magnitudes extremas ($\sim 3{,}28 \times 10^{16}\text{ kW}$):** `2021-01-31` (`-32.771.277.123.571.584,0 kW` a las `12:00`) y `2021-04-22` (`-32.771.276.799.998.576,0 kW` a las `17:00`), ambos en `2020_2022`.
    * **Caídas menores (ruido de medición):** `2021-01-03` (`-10,0 kW` a las `18:00`), `2021-01-21` (`-13,0 kW` a las `06:00`) y `2021-07-14` (`-40,0 kW` a las `15:00`).
    * **Otras `16` violaciones de monotonía (`-2.128,0 kW` a `-31.131,0 kW`):** `2020-09-11` (`-28.674,0`), `2021-03-18` (`-27.138,0`), `2021-03-19` (`-26.003,0`), `2021-04-04` (`-31.131,0`), `2021-04-21` (`-5.405,0`), `2021-04-25` (`-8.660,0`), `2021-04-30` (`-8.446,0`), `2021-06-27` (`-2.128,0`), `2021-07-11` (`-8.790,0`), `2021-07-27` (`-8.868,0`), `2021-08-03` (`-6.726,0`), `2021-10-01` (`-5.474,0`), `2022-04-16` (`-6.796,0`), `2022-04-18` (`-5.676,0`), `2022-07-03` (`-5.418,0`) y `2023-05-20` (`-27.999,0` a las `23:59`).
* **Filas extra eliminadas por hoja en `limpiar_y_auditar_hoja` (`df_bajas_filas`):**

  | Archivo | Motivo | Días únicos afectados | Total filas borradas (suma entre hojas) |
  |---|---|---|---|
  | `2020_2022` | `Jornada apilada/clonada (mes < 31 días)` | `610` | `13.320` |
  | `2020_2022` | `Registro extra '23:59:59'` | `2` | `15` |
  | `2021_2023` | `Jornada apilada/clonada (mes < 31 días)` | `8` | `4.800` |
  | `2021_2023` | `Registro extra '23:59:59'` | `1` | `16` |
  | `2022_2023` | `Jornada apilada/clonada (mes < 31 días)` | `6` | `3.000` |
  | `2022_2023` | `Registro extra '23:59:59'` | `1` | `15` |

* **Días reparados / conservados con anomalías subsanables (`6` registros en `resumen_rep`):**
  * `2021-06-12` (`2020_2022`, `Consolidado EE`): `Pico 01:00 aislado (Día conservado)` — `00:00 = 0.0`, `01:00 = 4.111.695,0`, `02:00 = 3.558,0`; retoma desde `02:00` y cierra en `23:59` con `37.992,0 kW`.
  * `2021-07-21` (`2021_2023`, `Consolidado Agua`): `Prefijo de fecha en HORA` — se eliminó el prefijo `'1900-01-01'` conservando la hora.
  * `2020-07-20` (`2020_2022`, `Totalizadores CO2`): `Typo de secuencia horaria` (`23:00->11:00`).
  * `2020-09-05` (`2020_2022`, `Consolidado KPI`): `Typo de secuencia horaria` (`15:00->16:00`).
  * `2021-09-16` (`2021_2023`, `Seguimiento Dia`): `Typo de secuencia horaria` (`09:00->08:00, 08:00->09:00`).
  * `2022-12-10` (`2022_2023`, `Consolidado KPI`): `Typo de secuencia horaria` (`16:00->14:00`).

---

### 4. Problemas Técnicos Encontrados y Solución Aplicada

* **Desalineación de `HORA` por prefijos de fecha (`1900-01-01`), segundos sueltos (`23:59:59`) y errores de tipeo en hojas secundarias:** al unificar las 15/16 hojas por `(DIA, HORA)`, un solo typo horario en una hoja (ej. `23:00` en lugar de `11:00` en `Totalizadores CO2`) generaba filas huérfanas tras el `outer` join. Se resolvió normalizando el formato de texto, purgando `'23:59:59'` antes del truncado de segundos y aplicando reparación posicional por día en `limpiar_y_auditar_hoja`.
* **Discrepancia de criterio entre la regla de monotonía estricta y los hallazgos de la Sesión 2:**
  * La regla de `exploracion_santi.ipynb` descarta días cuyo cierre `23:59` es correcto pero tuvieron un pico intermedio aislado luego compensado (ej. `2020-09-11`, `2021-01-31`, `2021-03-18`, `2021-04-22`) o ruido mínimo de medición (`-10` a `-40 kW`), mientras que no filtra los días con saltos horarios positivos sin caída posterior mayores al P99,9 (`5.153 kW`) que inflan el cierre (como `2021-08-01`, `2021-07-21`, `2021-08-24` y `2021-07-12`). Queda pendiente unificar ambos criterios junto con el equipo.
* **Conteo de bloques clonados en `2020_2022` (`610` días / `13.320` filas):** mientras `2021_2023` (`8` días) y `2022_2023` (`6` días) coinciden con los cierres de meses `< 31` días, en `2020_2022` la condición `(HORA == "00:00") & (DIA == DIA.shift(1))` se disparó en `610` fechas únicas al aplicarse sobre hojas auxiliares/secundarias; conviene auditar qué hoja específica de `2020_2022` dispara esa regla.
* **Alcance parcial de la auditoría física y persistencia:** se definió `cols_acumulativas` excluyendo `HOJAS_NO_ACUMULATIVAS`, pero la validación física solo evalúa `Frio (Kw)_Consolidado EE` y `df_limpio` aún no se exporta a `data/processed/` ni registra su `SHA-256` en `data/checksums.json`.

---

### 5. Estado al Cierre y Próximos Pasos (Hoja de Ruta)

Quedó operativo el prototipo de unificación multi-hoja y auditoría en `notebooks/exploracion_santi.ipynb` (`1.159` días / `28.971` filas limpias en memoria). Tareas inmediatas para el siguiente avance:

1. **Inspeccionar qué hojas de `2020_2022` generan los `610` días marcados como bloques clonados** en `log_filas_eliminadas`, para confirmar que no se estén descartando filas válidas en hojas distintas de `Consolidado EE`.
2. **Conciliar el criterio de días inválidos con la Sesión 2:** decidir si se conserva el filtro estricto de monotonía o si se recuperan los días con error intermedio compensado, e incorporar la detección de saltos positivos sin compensación (`> P99,9`) y de contador congelado (`2021-01-03` ya eliminado por `-10 kW`, pero útil generalizarlo).
3. **Extender la auditoría física al resto de variables de `cols_acumulativas`** (energía, agua, gas/vapor, aire, $\text{CO}_2$, efluentes, glicol, producción) que se usarán como *features* en las fases siguientes.
4. **Exportar `df_limpio` y las tablas de auditoría a `data/processed/`** (con `lineterminator="\n"`) y registrar su `SHA-256` en `data/checksums.json`, además de completar los ítems pendientes de la **Sección 1.2** (`boxplots` mensuales, efecto del día de la semana con ANOVA/Kruskal-Wallis + $\eta^2$, patrón horario y distribución/QQ-plot del target).