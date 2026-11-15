# Laboratorio de Datos II - Trabajo Práctico Final

Proyecto de Laboratorio de Datos II de Ingeniería en Inteligencia Artificial (UNL).
El objetivo es predecir al cierre de cada día el consumo de frío de una planta
cervecera para el día calendario siguiente.

## Entorno

Cloná el repositorio y ubicate en la rama de modelado.

```bash
git clone https://github.com/s4nt1lu1/tp-final-datos-ii.git
cd tp-final-datos-ii
git switch feature/modeling
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

Ejecutá los dos comandos desde la raíz, con el entorno activado.

```bash
python -m src.correccion_fechas
python -m src.preprocessing_pipeline
```

El primer comando parte de los tres Excel y genera `dataset_trabajo.csv` y la
tabla de días descartados. El segundo verifica los hashes de sus entradas,
ajusta la preparación en Train y elige las variables en Validation. Genera
`dataset_final.csv`, `dataset_diario_base.csv`, `train.csv`, `validation.csv`,
`test.csv` y `data_lineage.json` en `data/processed/`. Los CSV se guardan
sin escalar y sus hashes quedan registrados en `data/checksums.json`.

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

Después de generar los CSV de Fase 2, ejecutá el modelado desde la raíz.

```bash
python -m src.train_model
```

El script calcula los dos baselines y ajusta Ridge, Random Forest, XGBoost y
LightGBM con Train. Agrega cada prueba a `results/experiment_logs.csv` y
registra la elección por MAE de Validation en `results/seleccion_modelado.json`.
`notebooks/modelado.ipynb` explica el espacio de búsqueda y reproduce las
comparaciones, el análisis de residuos y la importancia de variables.

Para trabajar con un registro de experimentos separado, agregá
`--salida resultados_locales`. Usá la misma carpeta de salida para los pasos
de una corrida.

Cuando la selección esté cerrada, la evaluación final se ejecuta con el
siguiente comando.

```bash
python -m src.train_model --evaluar-test
```

Ese paso reentrena el estimador con Train + Validation y guarda las métricas
agregadas de Test en `results/evaluacion_final.json`. El comando evita repetir
la evaluación en una salida que ya tenga ese archivo. La notebook muestra la
tabla guardada. `models/modelo_v1.0.0.pkl` contiene el estimador, el preparador
de Fase 2 y el orden de las columnas. La versión, el commit de entrenamiento
y el hash del artefacto quedan en `models/model_registry.json`.
