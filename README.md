# Laboratorio de Datos II - Trabajo Práctico Final

Proyecto de Laboratorio de Datos II de Ingeniería en Inteligencia Artificial (UNL).
El objetivo es predecir al cierre de cada día el consumo de frío de una planta
cervecera para el día calendario siguiente.

La versión entregada es `v2.0.0`. El modelo ya está entrenado y versionado en
`models/modelo_v2.0.0.pkl`: para pronosticar un Excel nuevo no hace falta entrenar
ni tener los datos de entrenamiento.

## 1. Entorno

Cloná el repositorio y creá un entorno aislado de Python 3.12 con las versiones fijadas.

```bash
git clone https://github.com/s4nt1lu1/tp-final-datos-ii.git
cd tp-final-datos-ii
conda create -n cervecera_env python=3.12
conda activate cervecera_env
python -m pip install uv
uv pip install -r requirements.txt
```

Si ya tenés `uv`, también podés usar `uv venv --python 3.12` y luego
`uv pip install -r requirements.txt`. Para activar ese entorno usá
`.\.venv\Scripts\Activate.ps1` en PowerShell o `source .venv/bin/activate` en Linux/macOS.

## 2. Predecir con un Excel nuevo (Fase 4)

1. Copiá el Excel nuevo en cualquier carpeta, por ejemplo `data/raw/`.
2. Desde la raíz del repositorio, con el entorno activado, ejecutá:

```bash
python -m src.predict "data/raw/Totalizadores Planta de Cerveza 2024_2025.xlsx"
```

El resultado queda en `results/predicciones.csv`, con una fila por día D:

| fecha | prediccion_frio_kw |
| --- | --- |
| día D (AAAA-MM-DD) | consumo de frío estimado para D+1 (kW) |

El script carga la versión más alta de `models/model_registry.json` (`v2.0.0`) y
verifica el SHA-256 del .pkl. Después aplica la misma limpieza horaria, el mismo
preprocesamiento y las mismas variables creadas que en el entrenamiento, todo ajustado
con Train y guardado dentro del .pkl. No usa `data/processed/` ni los Excel de
entrenamiento.

Opciones:

| Opción | Para qué |
| --- | --- |
| `--salida ruta.csv` | Guardar las predicciones en otro archivo. |
| `--historial anterior.xlsx` | Aportar los días previos al primer día del Excel nuevo (por ejemplo, el Excel del período anterior). |
| `--desde AAAA-MM-DD` | Exportar predicciones solo desde esa fecha. |
| `--version v2.0.0` | Elegir una versión del registro (por defecto, la más alta). |

**Contexto de los primeros días.** Las ventanas del frío usan hasta 28 días anteriores.
Si el Excel nuevo empieza sin ese historial, las ventanas de los primeros días se
completan con las medianas de Train y el script lo avisa. Con `--historial` y el Excel
del período anterior, esos días se pronostican con su historia real:

```bash
python -m src.predict "data/raw/Totalizadores Planta de Cerveza 2024_2025.xlsx" --historial "data/raw/Totalizadores Planta 2022_2023.xlsx"
```

**Contrato con el archivo de entrada.** Las hojas y columnas se buscan por nombre:
puede haber hojas y columnas de más, en cualquier orden. Si falta alguna de las que usa
el modelo, el script termina con un mensaje que indica cuál y en qué hoja. Las columnas
del entrenamiento que el modelo no usa pueden faltar: se completan con la imputación
aprendida en Train. También avisa si faltan cierres o si descarta jornadas, y por qué.

Mediciones mínimas de `v2.0.0` (cada hoja también debe tener `DIA` y `HORA`):

| Hoja | Columnas de medición |
| --- | --- |
| Consolidado EE | `Frio (Kw)` |
| Consolidado KPI | `EE Frio / Hl`, `EE Bodega / Hl`, `Meta EE Frio`, `Meta ET Cocina` |
| Consolidado Produccion | `Hl Producido Bodega`, `Hl Cerveza Envasada` |
| Consolidado Agua | `Agua Bodega (Hl)` |
| Consolidado GasVapor | `Vapor Envasado (Kg)` |
| Totalizadores Energia | `KW Trafo 9`, `KW Trafo 12`, `KW Planta de Agua` |
| Totalizadores Produccion | `HL Mosto Sol` |
| Totalizadores Aire | `Totalizador_Aire_Bodega` |
| Totalizadores CO2 | `Totalizador_Filtracion` |
| Totalizadores Glicol | `Tot  Trasiego` |

## 3. Reproducir el entrenamiento (opcional)

Solo hace falta para auditar o volver a generar el modelo. Copiá los tres Excel
provistos por la cátedra a `data/raw/`:

- `Totalizadores Planta 2020_2022.xlsx`
- `Totalizadores Planta - 2021_2023.xlsx`
- `Totalizadores Planta 2022_2023.xlsx`

Los Excel no se versionan. La limpieza horaria los comprueba contra los SHA-256 de
`data/checksums.json`. Cada etapa verifica el SHA-256 de la salida de la anterior.

```bash
python -m src.auxiliar_functions --verificar-hashes   # Excel -> data/processed/dataset_trabajo.csv
python -m src.preprocessing_pipeline                  # Fase 2: EDA, preprocesamiento y selección
python -m src.train_model                             # Fase 3 -> results/modelado/ y results/experiment_logs.csv
python -m src.auxiliar_functions --comparar-variantes # Diagnóstico en Validation -> results/comparacion_variantes/
```

`python -m src.preprocessing_pipeline --completo` corre antes la limpieza horaria, así que
deja todo listo hasta la selección en un solo comando. Con `--etapas` se corre una sola
etapa (`eda`, `preprocesamiento` o `seleccion`). La semilla de todas las particiones y
estimadores es `SEMILLA = 42` (`src/auxiliar_functions.py`).

| Script | Qué hace |
| --- | --- |
| `src/auxiliar_functions.py` | Rutas, semilla y SHA-256 de los Excel. Limpieza horaria: une los tres libros, repara la secuencia horaria, calibra con el primer año los umbrales del cierre 23:59 y descarta días incompletos, totales imposibles y días sin reinicio del totalizador. `comparar_variantes()` (`--comparar-variantes`): variantes probadas solo en Validation (filtro de redundancia y objetivo relativo). |
| `src/preprocessing_pipeline.py` | Tres etapas. **EDA:** decisiones que dependen de la estructura de la planilla y no de `y` (metadatos, componentes y proxies del frío, columnas calculadas por diferencia, renombradas y copias). **Preprocesamiento:** cierre diario, objetivo D+1, división 70/15/15 y pasos `fit`/`transform` ajustados solo con Train (columnas no informativas, duplicadas, nulos, outliers e imputación); los CSV quedan sin escalar. **Selección:** candidatas (ventanas del frío, calendario, ciclo anual, binning, interacciones, ratios y eficiencia contra la meta) y cuatro métodos ajustados con Train. |
| `src/train_model.py` | Dos baselines, Optuna para Ridge, Lasso, Random Forest, XGBoost y LightGBM con validación temporal en Train, elección por MAE de Validation, evaluación única en Test y registro del modelo. |
| `src/predict.py` | Pronóstico de un Excel nuevo con el modelo del registro. |

Los resultados de cada etapa quedan en `results/` (`eda/`, `preprocesamiento/`, `seleccion/`
y `modelado/`). No se versionan: los regeneran estos comandos y las notebooks los muestran.
Solo se versiona `results/experiment_logs.csv`.

Las notebooks `notebooks/eda.ipynb`, `preprocesamiento.ipynb`, `seleccion.ipynb` y
`modelado.ipynb` muestran cada decisión con sus números. `notebooks/v1/` guarda, como
registro, las notebooks de la versión 1, que se ejecutan desde el tag `v1.0.0`.

### Modelo final y registro

```bash
python -m src.train_model --evaluar-test         # una sola vez, con la elección congelada
python -m src.train_model --regenerar-registro   # vuelve a generar el .pkl desde el commit actual
```

`--evaluar-test` reentrena el ganador con Train + Validation, lo evalúa una vez en Test
junto a los dos baselines y registra la versión en `models/` (.pkl y
`model_registry.json` con hiperparámetros, métricas, ruta, SHA-256 y commit), además de
`data/processed/dataset_final.csv` y `data_lineage.json` con sus SHA-256 en
`data/checksums.json`. `v2.0.0` ya está evaluada.

`--regenerar-registro` reajusta la configuración registrada desde el código del commit
actual, comprueba que predice exactamente lo mismo que el modelo evaluado
(`results/modelado/predicciones_modelo_final.csv`, que escribe `--evaluar-test`) y
actualiza el .pkl, su SHA-256 y el commit del registro, sin volver a leer Test.

Cada corrida de `python -m src.train_model` agrega sus trials, candidatos y baselines a
`results/experiment_logs.csv` (`--sin-log` lo evita).

## 4. Versiones

| Versión | Modelo | Código | MAE Validation | MAE Test |
| --- | --- | --- | ---: | ---: |
| `v2.0.0` | Random Forest, 23 variables | tag `v2.0.0` | 2.297,6 | 2.836,7 |
| `v1.0.0` | Ridge, 10 variables | tag `v1.0.0` | 2.126,7 | 2.796,9 |

En Test, los baselines de `v2.0.0` dan 3.367,8 kW (persistencia) y 3.013,8 kW (media
móvil de 7 días). El .pkl de `v1.0.0` sigue en `models/`, pero su código ya no está en
esta rama: `python -m src.predict archivo.xlsx --version v1.0.0` indica que hay que hacer
checkout del tag `v1.0.0`.
