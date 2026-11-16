# Trabajo Práctico Final 2026

## Predicción del consumo energético industrial

Este archivo resume los requisitos del PDF **Trabajo Práctico Final 2026**. Para las consignas completas, los criterios de aprobación y las indicaciones de la defensa se toma como referencia el PDF de la cátedra.

El objetivo es pronosticar el consumo diario de refrigeración de una planta cervecera en Tecate. Al cierre del día **D**, se usan las mediciones disponibles para predecir el cierre de **D+1**. Las planillas contienen totalizadores horarios que se acumulan a lo largo del día. Elegir un horario de cierre, justificarlo y mantener el mismo criterio en todo el trabajo. La etiqueta se alinea con el día calendario siguiente, teniendo en cuenta los huecos en la serie.

## Fase 0. Entorno y Git

- Preparar un entorno reproducible con Python 3.12, documentar la instalación con `uv` y fijar las versiones en `requirements.txt`. La consigna presenta Conda con `uv` para este paso.
- Mantener un repositorio remoto con instrucciones de instalación y ejecución en `README.md`. Usar `SEMILLA = 42` donde haya aleatoriedad.
- Trabajar en equipo de dos o tres integrantes, con commits de todos a lo largo del trabajo. Mantener `main` estable e integrar por Pull Request desde las ramas de cada fase.
- Mantener los Excel fuera de Git. Explicar en el README de dónde salen y versionar las huellas en `data/checksums.json`.

## Fase 1. EDA

Rama `feature/eda`. Entregable `notebooks/eda.ipynb` y su PR de integración.

- Unificar los archivos, contar los días solapados, comparar sus cierres y justificar la prioridad entre las fuentes. Controlar las jornadas apiladas y verificar que cada fecha tenga hasta 25 registros antes de pasar a frecuencia diaria.
- Calcular el SHA-256 del dataset unificado inicial leyendo por bloques, registrarlo y explicar la elección frente a MD5. Mostrar dimensiones, tipos, rangos, faltantes y registros inválidos como días incompletos, lecturas imposibles o falta de reinicio del medidor.
- Explorar cómo cambia el consumo en el tiempo, en cada temporada y a lo largo del día. Medir el efecto semanal y justificar qué se hace con las variables asociadas.
- Mostrar correlaciones, las 15 variables más relacionadas con el objetivo y la multicolinealidad. Incluir un mapa de calor, serie temporal, boxplots mensuales, histograma y gráfico Q-Q del objetivo, dispersión de las cinco variables más correlacionadas y análisis por áreas de planta.

## Fase 2. Preprocesamiento y selección

Rama `feature/preprocessing`. Entregables `notebooks/preprocesamiento.ipynb`, `src/preprocessing_pipeline.py` y su PR.

- Documentar las decisiones de limpieza sobre días inválidos, faltantes y valores extremos. Crear y justificar las variables derivadas, como rezagos, medias móviles, calendario, ratios, metas por hectolitro e interacciones.
- Aplicar al menos dos métodos de selección de variables y justificar el subconjunto elegido.
- Dividir por fecha en Train (70 %), Validation (15 %) y Test (15 %), sin mezclar los tramos. Seleccionar el modelo y sus parámetros usando Validation o validación temporal dentro de Train. Reservar Test para la evaluación final.
- Ajustar la escala dentro de un pipeline de scikit-learn con datos de entrenamiento. Informar las dimensiones de las particiones y guardar el dataset final limpio sin escalar en `data/processed/dataset_final.csv`.
- Registrar el SHA-256 del dataset final y documentar las transformaciones, el script y la fecha en el registro de linaje `data/processed/data_lineage.json`.

## Fase 3. Entrenamiento y versionado del modelo

Rama `feature/modeling`. Entregables `notebooks/modelado.ipynb`, `src/train_model.py`, modelo serializado, `models/model_registry.json` y su PR.

- Construir los baselines de persistencia y media móvil de siete días. Compararlos con los modelos sobre los mismos datos y con el mismo MAE.
- Comparar al menos cuatro familias. La consigna pide XGBoost, Random Forest, LightGBM y una regresión regularizada Ridge o Lasso.
- Registrar automáticamente fecha de la corrida, hiperparámetros y métricas MAE, RMSE y R² en `results/experiment_logs.csv`. Explicar la búsqueda, mostrar predicciones frente a valores reales, analizar residuos e interpretar la importancia de las variables.
- Elegir por MAE de Validation y reentrenar el ganador con Train y Validation. Guardar el modelo con versión semántica, como `models/modelo_v1.0.0.pkl`. Registrar versión, parámetros, métricas de Validation, ruta y commit de entrenamiento.
- Crear una etiqueta Git anotada que coincida con la versión registrada y apunte al commit de la entrega que contiene ese modelo.

## Fase 4. Predicción con datos nuevos

Rama `feature/prediction-pipeline`. Entregable `src/predict.py` funcional y PR final.

- Cargar la versión del modelo desde el registro y reutilizar el preprocesamiento del entrenamiento.
- Buscar las hojas y las columnas por nombre. Aceptar hojas o columnas extras y cambios de orden. Informar con claridad si falta un insumo requerido.
- Generar `results/predicciones.csv` con `fecha` y `prediccion_frio_kw`. Cada fila corresponde a un día **D** de emisión y el valor pronosticado corresponde a **D+1**.

## Informe, defensa y condiciones de aprobación

El informe debe explicar las decisiones, justificarlas y contar las limitaciones. Las cinco preguntas de cierre se responden con números del trabajo.

1. ¿Qué mejora da el modelo sobre persistencia y media móvil en el mismo Test? Si mejora menos del 10 % al mejor baseline, discutir el costo de usarlo y si vale la pena.
2. ¿Qué mediciones se parecen demasiado al objetivo? Explicar qué se hizo con ellas y qué datos están disponibles cuando se emite el pronóstico.
3. ¿Qué resultado contradijo la intuición sobre el patrón semanal? Mostrar la medición y cómo afectó la selección.
4. ¿Qué mejora mínima de MAE en kW justifica reemplazar un modelo? Derivar el umbral de una medición o un costo concreto.
5. ¿Qué dos técnicas vistas durante la materia se dejaron afuera y por qué?

Para aprobar se deben cumplir las cuatro condiciones eliminatorias del PDF:

1. Reportar y superar los dos baselines de MAE sobre el mismo Test.
2. Lograr un MAE menor a 3.000 kW en el Test oculto de la cátedra.
3. Poder clonar el repositorio, instalar las dependencias y ejecutar los scripts en otra máquina siguiendo el README.
4. Entregar un predictor funcional y registrar el trabajo de cada fase con sus ramas y Pull Requests.

La defensa final es oral, con el proyecto funcionando. Cada integrante debe poder explicar las decisiones y el código del trabajo.

