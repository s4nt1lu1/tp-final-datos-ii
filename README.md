# Laboratorio de Datos II - Trabajo Práctico Final

Proyecto de Laboratorio de Datos II de Ingeniería en Inteligencia Artificial (UNL).
El objetivo es predecir al cierre de cada día el consumo de frío de una planta
cervecera para el día calendario siguiente.

## Entorno

Cloná el repositorio y ubicate en la rama del pipeline de predicción.

```bash
git clone https://github.com/s4nt1lu1/tp-final-datos-ii.git
cd tp-final-datos-ii
git switch feature/prediction-pipeline
```

Creá un entorno aislado de Python 3.12 e instalá las versiones fijadas.

```bash
conda create -n cervecera_env python=3.12
conda activate cervecera_env
python -m pip install uv
uv pip install -r requirements.txt
```

Si ya tenés `uv`, también podés usar `uv venv --python 3.12` y luego
`uv pip install -r requirements.txt`. Para activar ese entorno usá
`.\.venv\Scripts\Activate.ps1` en PowerShell o `source .venv/bin/activate` en Linux/macOS.
La semilla de los estimadores es `SEMILLA = 42`.

## Datos de entrada

Copiar los tres Excel provistos por la cátedra a `data/raw/`. Estos nombres
corresponden a los libros efectivamente usados por el grupo.

- `Totalizadores Planta 2020_2022.xlsx`
- `Totalizadores Planta - 2021_2023.xlsx`
- `Totalizadores Planta 2022_2023.xlsx`

La limpieza horaria comprueba esos archivos contra `data/checksums.json`.
El diagnóstico climático usa una copia fija de
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
en `data/reference/clima_tecate_openmeteo.csv`, para Tecate
(32,5722° N, 116,6283° O, zona `America/Tijuana`). El diagnóstico y el pipeline
pueden ejecutarse con esa copia. La disponibilidad de cada observación
al cierre de D queda pendiente de verificar para incorporarla al modelo.

## Ejecutar la Fase 2

Desde la raíz, con el entorno activado y los tres Excel originales en
`data/raw/`, ejecutá este comando.

```bash
python -m src.preprocessing_pipeline --verificar-hashes
```

El comando limpia los Excel, genera `dataset_trabajo.csv` y la tabla de
descartes, y prepara los datos diarios con los parámetros ajustados en Train.
Genera `dataset_final.csv`, `dataset_diario_base.csv`, `train.csv`,
`validation.csv`, `test.csv` y `data_lineage.json` en `data/processed/`.
Los CSV se guardan sin escalar. En este modo, el comando compara los siete
CSV contra `data/checksums.json` y conserva ese registro.

Abrí `notebooks/preprocesamiento.ipynb` con el mismo entorno para ver las
decisiones y los controles paso a paso. Es el notebook que se entrega en la
Fase 2. Los notebooks auxiliares registran el trabajo previo del grupo.
`notebooks/eda.ipynb` contiene la exploración de los Excel y las decisiones
por sesión están en `docs/bitacora_sesiones.md`.

## Reutilizar la preparación diaria

El resultado de `run_pipeline()` contiene el `preparador` ajustado. Su método
`transform` recibe cierres diarios y su historial anterior. Para reproducir
la ventana de 14 días y el arrastre de hasta tres días, incluir al menos
16 días calendario de contexto previo al primer día que se quiera transformar.
Las columnas finales y su orden figuran en el linaje. Se puede transformar el
último cierre del historial aunque todavía no se conozca su etiqueta D+1.

## Ejecutar la Fase 3

Después de generar los CSV de Fase 2, ejecutá el modelado desde la raíz
con una carpeta de salida nueva para la corrida.

```bash
python -m src.train_model --salida resultados_locales
python -m src.train_model --salida resultados_locales --evaluar-test
```

El primer comando calcula los dos baselines, ajusta Ridge, Random Forest,
XGBoost y LightGBM con Train, y elige por MAE de Validation. Guarda el log
en `resultados_locales/results/experiment_logs.csv` y la elección en
`resultados_locales/results/seleccion_modelado.json`. La notebook explica
la búsqueda, los residuos y la importancia de las variables.

El segundo comando usa esa elección, reentrena con Train + Validation y
guarda las métricas de Test en `resultados_locales/results/evaluacion_final.json`.
Los dos comandos usan la misma salida. Si esa carpeta ya contiene una
evaluación final, elegí otra para una corrida completa.

## Predecir con un Excel nuevo (Fase 4)

Desde la raíz, con las dependencias instaladas y los archivos de `models/`
de esta rama, ejecutá:

```bash
python -m src.predict "ruta/al/archivo_nuevo.xlsx" --salida "results/predicciones.csv"
```

El script carga el modelo y su preparación guardada, verifica el SHA-256
del .pkl y emite una fila por fecha D. El valor de esa fila pronostica el
consumo de D+1. Para este paso alcanzan el Excel nuevo y el modelo publicado
con su registro. Las hojas y columnas se buscan por nombre.

Para el Ridge publicado, estas son las mediciones mínimas. Cada una de las
hojas correspondientes también debe tener las columnas `DIA` y `HORA`.

| Hoja | Columnas de medición |
| --- | --- |
| Consolidado EE | `Frio (Kw)` |
| Consolidado GasVapor | `ET Servicios (Mj)` |
| Totalizadores Energia | `KW Secador Kaeser`, `KW Linea 3` |
| Consolidado Agua | `Red Paste L4` |
| Consolidado Produccion | `Hl Producido Bodega` |

Las medias móviles usan el historial anterior a cada fecha. Si el Excel
incluye los primeros días como contexto, usá `--desde AAAA-MM-DD` para
exportar desde el día solicitado. Si el historial está en otro Excel, usá:

```bash
python -m src.predict "ruta/al/archivo_nuevo.xlsx" --historial "ruta/al/historial.xlsx"
```

El segundo Excel aporta las fechas anteriores a las del principal. Para el
primer pronóstico conviene aportar al menos 16 días calendario de contexto.
El comando avisa por la terminal si faltan cierres al comienzo o si se omiten
fechas, e indica el motivo al descartar una jornada.

El modelo auditado de esta rama es el .pkl indicado en
`models/model_registry.json`. El commit `b4f0002` permite regenerar ese
binario byte por byte. La etiqueta histórica `v1.0.0` está en un commit con
un binario anterior. Para esta entrega se usa el archivo y el registro de
la rama del pipeline de predicción.
