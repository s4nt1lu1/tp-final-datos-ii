# Auditoría de Trazabilidad, Reproducibilidad y Ejecución Integral (`v2.0.0`)

**Fecha de auditoría:** 23/11/2026  
**Repositorio auditado:** `https://github.com/s4nt1lu1/tp-final-datos-ii` (rama `task/revision`, commit base `7acd00a`)  
**Directorio de pruebas:** `tp-final-datos-ii-github/`  
**Marco de referencia:** [Trabajo Práctico Final 2026.pdf](../../Trabajo%20Pr%C3%A1ctico%20Final%202026.pdf) — *Condición de aprobación eliminatoria N.º 3 (Reproducibilidad)* y *Fase 0: Arquitectura y trazabilidad (Actividades 0.1, 0.2 y 0.3)*.

---

## 1. Objetivo de la Auditoría

Simular en un entorno limpio el procedimiento que realiza la cátedra al evaluar el entregable:

1. Clonar el repositorio desde GitHub en una carpeta limpia (`tp-final-datos-ii-github`).
2. Copiar los 3 archivos Excel crudos provistos por la cátedra en `data/raw/`.
3. Seguir paso a paso las instrucciones de [README.md](../README.md) para:
   - Instalar dependencias desde [requirements.txt](../requirements.txt).
   - Probar el pronóstico directo sobre datos crudos con [src/predict.py](../src/predict.py) antes de correr el pipeline de entrenamiento.
   - Ejecutar los 4 módulos de `src/` ([src/auxiliar_functions.py](../src/auxiliar_functions.py), [src/preprocessing_pipeline.py](../src/preprocessing_pipeline.py), [src/train_model.py](../src/train_model.py), [src/predict.py](../src/predict.py)) y verificar todas las huellas SHA-256 contra [data/checksums.json](../data/checksums.json), [data/processed/data_lineage.json](../data/processed/data_lineage.json) y [models/model_registry.json](../models/model_registry.json).
4. Ejecutar de principio a fin las 4 notebooks principales ([notebooks/eda.ipynb](../notebooks/eda.ipynb), [notebooks/preprocesamiento.ipynb](../notebooks/preprocesamiento.ipynb), [notebooks/seleccion.ipynb](../notebooks/seleccion.ipynb), [notebooks/modelado.ipynb](../notebooks/modelado.ipynb)).
5. Diagnosticar, corregir y documentar línea por línea cualquier error o inconsistencia surgida durante la simulación.

---

## 2. Resumen de Ejecución y Verificación de Checksums SHA-256

Todos los archivos crudos, intermedios, particiones procesadas y modelos serializados fueron verificados criptográficamente con SHA-256:

| Artefacto | Ruta relativa | SHA-256 esperado (`checksums.json` / `model_registry.json`) | SHA-256 obtenido al ejecutar | Resultado |
| :--- | :--- | :--- | :--- | :---: |
| Excel crudo 1 | `data/raw/Totalizadores Planta 2020_2022.xlsx` | `7bb3474c90ec0de46edca4f6c0bbbf93823d0b5d9a155fe595c32c3d79e3788b` | `7bb3474c90ec0de46edca4f6c0bbbf93823d0b5d9a155fe595c32c3d79e3788b` | **OK** |
| Excel crudo 2 | `data/raw/Totalizadores Planta - 2021_2023.xlsx` | `b4dbfacdb9dfa34688ef5edfa0ec28884fb62db08e5a770eb204c3da7d31f92d` | `b4dbfacdb9dfa34688ef5edfa0ec28884fb62db08e5a770eb204c3da7d31f92d` | **OK** |
| Excel crudo 3 | `data/raw/Totalizadores Planta 2022_2023.xlsx` | `aba9f9ff8b12dd8fbff8a6efea5680334d349e6b2b05368b01e840560a102657` | `aba9f9ff8b12dd8fbff8a6efea5680334d349e6b2b05368b01e840560a102657` | **OK** |
| Dataset horario limpio | `data/processed/dataset_trabajo.csv` | `bd015abb722ba2151d64ae9f229364cd61861b8682ffe846084d215aa083ae78` | `bd015abb722ba2151d64ae9f229364cd61861b8682ffe846084d215aa083ae78` | **OK** |
| Tabla días borrados | `data/processed/tabla_dias_borrados.csv` | `6407891eb6902a4673bc6f8d386b724df1279b9c63434d1d08eb7ba5d97023d3` | `6407891eb6902a4673bc6f8d386b724df1279b9c63434d1d08eb7ba5d97023d3` | **OK** |
| Clima externo Tecate | `data/reference/clima_tecate_openmeteo.csv` | `dc1d29df295bd86bc3c436283fd26b9ef1f3d16667f9f5785d40fbff75d81eb9` | `dc1d29df295bd86bc3c436283fd26b9ef1f3d16667f9f5785d40fbff75d81eb9` | **OK** |
| Historial preprocesado | `data/processed/historial_completo.csv` | `b46f1a30306021c627b6d5cf3348905e3a33460d504493933a3a4aaeb6f90db1` | `b46f1a30306021c627b6d5cf3348905e3a33460d504493933a3a4aaeb6f90db1` | **OK** |
| Partición Train | `data/processed/train.csv` | `650e978ee1fd78d8bf633d87dd787b8a82bf3e6f63150248292bf9df4b0c9b14` | `650e978ee1fd78d8bf633d87dd787b8a82bf3e6f63150248292bf9df4b0c9b14` | **OK** |
| Partición Validation | `data/processed/validation.csv` | `3056e2e1eb0f3ac2e846b823309d44303815df99968c488f50e2b0d895d226e7` | `3056e2e1eb0f3ac2e846b823309d44303815df99968c488f50e2b0d895d226e7` | **OK** |
| Partición Test | `data/processed/test.csv` | `ebe046ac616aa3aa31d0c167007c6a10a94d80ff2014c00690bc57c4b85e5152` | `ebe046ac616aa3aa31d0c167007c6a10a94d80ff2014c00690bc57c4b85e5152` | **OK** |
| Dataset final (`23` features) | `data/processed/dataset_final.csv` | `722849ad6cb1542362d6c536bb4ec067fb040735867d2d92cf54d8f2856150ec` | `722849ad6cb1542362d6c536bb4ec067fb040735867d2d92cf54d8f2856150ec` | **OK** |
| Modelo `v1.0.0` | `models/modelo_v1.0.0.pkl` | `6cef9d08e5ab1d494957e5045567b7e34a4021303f821124f20572bb0f0c8570` | `6cef9d08e5ab1d494957e5045567b7e34a4021303f821124f20572bb0f0c8570` | **OK** |
| Modelo `v2.0.0` | `models/modelo_v2.0.0.pkl` | `a2b7da7890eb5d4649c0e61c5be466a3f03923eb0821bc3966e5fa17d015b5bf` | `a2b7da7890eb5d4649c0e61c5be466a3f03923eb0821bc3966e5fa17d015b5bf` | **OK** |

---

## 3. Errores Detectados durante la Simulación y Correcciones Aplicadas

Durante la simulación estricta de clon limpio se detectaron **4 problemas de reproducibilidad/ejecución**, todos ellos originados en archivos ignorados por `.gitignore` (`results/*`) o detalles de codificación/documentación al clonar desde cero. A continuación se detalla cada error, su causa raíz y las líneas exactas modificadas.

---

### Error 1: Caracter BOM UTF-8 (`\xef\xbb\xbf`) al inicio de `requirements.txt`

* **Archivo afectado:** [requirements.txt](../requirements.txt)
* **Línea modificada:** Línea `1`
* **Síntoma / Error detectado:**
  Al leer [requirements.txt](../requirements.txt) en modo binario / UTF-8 estricto, los primeros bytes eran `b'\xef\xbb\xbfalembic==1.20.1'`. En analizadores de dependencias o entornos Linux que no descartan automáticamente el Byte Order Mark (BOM), el primer paquete se interpreta como `\ufeffalembic==1.20.1` y falla la validación.
* **Corrección aplicada:**
  Se eliminaron los 3 bytes iniciales `\xef\xbb\xbf`, dejando el archivo en codificación UTF-8 estándar sin BOM:
  ```diff
  -﻿<BOM>alembic==1.20.1
  +alembic==1.20.1
  ```

---

### Error 2: `FileNotFoundError` al ejecutar `python -m src.train_model --regenerar-registro` y ausencia de `evaluacion_test.csv` / `dataset_final.csv` en un clon limpio

* **Archivo afectado:** [src/train_model.py](../src/train_model.py)
* **Líneas modificadas:**
  * Líneas `483–493` (dentro de `ejecutar()`)
  * Líneas `715–791` (nueva función `restaurar_artefactos_registrados()` y ajuste en `regenerar_registro()`)
* **Síntoma / Error detectado:**
  Al ejecutar el segundo comando de la subsección *Modelo final y registro* de [README.md](../README.md) (`python -m src.train_model --regenerar-registro`) en el repositorio recién clonado:
  ```text
  Traceback (most recent call last):
    File "src/train_model.py", line 833, in <module>
      main()
    File "src/train_model.py", line 824, in main
      regenerar_registro()
    File "src/train_model.py", line 732, in regenerar_registro
      raise FileNotFoundError(f"Falta {ruta_referencia}: sin esa referencia no se puede comprobar "
  FileNotFoundError: Falta .../results/modelado/predicciones_modelo_final.csv: sin esa referencia no se puede comprobar que el modelo regenerado sea el registrado.
  ```
* **Causa raíz:**
  1. [.gitignore](../.gitignore) (línea `45`: `results/*`, línea `46`: `!results/experiment_logs.csv`) ignora todos los archivos de `results/` salvo `results/experiment_logs.csv`.
  2. Cuando se clona el repositorio y se ejecuta `python -m src.train_model` (sin `--evaluar-test`, ya que Test se evalúa una única vez por versión), `ejecutar()` escribía `optuna_trials.csv`, `comparacion_validation.csv` y `resumen.json`, pero **no restauraba** `results/modelado/predicciones_modelo_final.csv`, `results/modelado/evaluacion_test.csv` ni `data/processed/dataset_final.csv`.
  3. En consecuencia:
     - `python -m src.train_model --regenerar-registro` fallaba en la línea `732` al no encontrar `results/modelado/predicciones_modelo_final.csv`.
     - La celda `In[16]` (índice `32`) de [notebooks/modelado.ipynb](../notebooks/modelado.ipynb) también fallaba al detectar que `v2.0.0` ya estaba en `models/model_registry.json` e intentar leer `pd.read_csv(tm.SALIDA / "evaluacion_test.csv")`.
* **Corrección aplicada:**
  1. Se implementó en [src/train_model.py](../src/train_model.py#L717-L767) la función `restaurar_artefactos_registrados(d, salida=SALIDA, version=VERSION)` que, cuando `version` ya figura en `models/model_registry.json`:
     - Si falta `results/modelado/predicciones_modelo_final.csv`, carga el bundle versionado en Git (`models/modelo_v2.0.0.pkl`, tras verificar su SHA-256 contra `model_registry.json`), calcula `predicciones_historial()` (usando solo las predictoras `X`, sin tocar `y_test`) y restaura el CSV.
     - Si falta `results/modelado/evaluacion_test.csv`, reconstruye la tabla de Test de la evaluación única registrada (`corrida_utc`) a partir de las filas con `particion == "test"` de `results/experiment_logs.csv` (que sí está versionado en Git) o de `baselines_test` / `metricas_test` en `models/model_registry.json`, sin volver a evaluar Test.
     - Si falta `data/processed/dataset_final.csv`, lo exporta mediante `exportar_datos_finales(d, previo["columnas"], X_te)` para que todos los archivos declarados en `data/checksums.json` y `data/processed/data_lineage.json` queden presentes en disco.
  2. En `ejecutar()` ([src/train_model.py](../src/train_model.py#L483-L493)), cuando `not evaluar_test`, se invoca `resultado_test = restaurar_artefactos_registrados(d, salida, VERSION)`.
  3. En `regenerar_registro()` ([src/train_model.py](../src/train_model.py#L770-L791)), se carga `d = cargar_datos()` y se llama a `restaurar_artefactos_registrados(d, salida, version)` antes de leer `ruta_referencia`.
* **Verificación posterior:**
  Al borrar `predicciones_modelo_final.csv`, `evaluacion_test.csv` y `dataset_final.csv` y ejecutar `python -m src.train_model --regenerar-registro`, el comando finalizó con `ExitCode: 0`, diferencia máxima de `0.0e+00 kW` y coincidencia exacta de SHA-256 en `data/checksums.json`.

---

### Error 3: `FileNotFoundError` al ejecutar `notebooks/seleccion.ipynb` y `notebooks/modelado.ipynb` en un clon limpio por falta de `results/comparacion_variantes/`

* **Archivos afectados:**
  * [notebooks/seleccion.ipynb](../notebooks/seleccion.ipynb) — Celda de código `In[8]` (índice de celda `15`, líneas `14–17`)
  * [notebooks/modelado.ipynb](../notebooks/modelado.ipynb) — Celda de código `In[11]` (índice de celda `21`, líneas `1–5`) y Celda de código `In[16]` (índice de celda `32`, líneas `14–16`)
  * [README.md](../README.md) — Sección 3, líneas `100–105`
* **Síntoma / Error detectado:**
  Al ejecutar las 4 notebooks tras haber corrido los 3 comandos de la sección 3 del `README.md`:
  1. En [notebooks/seleccion.ipynb](../notebooks/seleccion.ipynb) (Celda `In[8]` / índice `15`):
     ```text
     FileNotFoundError: [Errno 2] No such file or directory: '.../results/comparacion_variantes/comparacion_mejoras.csv'
     ```
  2. En [notebooks/modelado.ipynb](../notebooks/modelado.ipynb) (Celda `In[11]` / índice `21`):
     ```text
     FileNotFoundError: [Errno 2] No such file or directory: '.../results/comparacion_variantes/comparacion_por_objetivo.csv'
     ```
* **Causa raíz:**
  La carpeta `results/comparacion_variantes/` está ignorada por `.gitignore` (`results/*`) y es generada por `python -m src.auxiliar_functions --comparar-variantes`, comando que no estaba listado en el bloque principal de la Sección 3 del [README.md](../README.md). Además, las celdas de las notebooks asumían que los CSV ya existían en disco sin generarlos si faltaban.
* **Correcciones aplicadas:**
  1. **[README.md](../README.md#L100-L105) (Línea `104`):** Se incorporó el comando `python -m src.auxiliar_functions --comparar-variantes` en el bloque de reproducción del pipeline de la Sección 3:
     ```bash
     python -m src.auxiliar_functions --verificar-hashes   # Excel -> data/processed/dataset_trabajo.csv
     python -m src.preprocessing_pipeline                  # Fase 2: EDA, preprocesamiento y selección
     python -m src.train_model                             # Fase 3 -> results/modelado/ y results/experiment_logs.csv
     python -m src.auxiliar_functions --comparar-variantes # Diagnóstico en Validation -> results/comparacion_variantes/
     ```
  2. **[notebooks/seleccion.ipynb](../notebooks/seleccion.ipynb) (Celda `In[8]`, índice `15`, líneas `14–17`):** Se agregó la verificación automática para ejecutar `af.comparar_variantes()` si `comparacion_mejoras.csv` no existe:
     ```python
     ruta_mejoras = raiz / "results" / "comparacion_variantes" / "comparacion_mejoras.csv"
     if not ruta_mejoras.exists():
         af.comparar_variantes()
     comparacion_variantes = pd.read_csv(ruta_mejoras)
     ```
  3. **[notebooks/modelado.ipynb](../notebooks/modelado.ipynb) (Celda `In[11]`, índice `21`, líneas `1–5`):** Se agregó la guarda equivalente para generar `comparacion_por_objetivo.csv` y `comparacion_mejoras.csv` si no existen:
     ```python
     carpeta_variantes = raiz / "results" / "comparacion_variantes"
     if not (carpeta_variantes / "comparacion_por_objetivo.csv").exists() or not (carpeta_variantes / "comparacion_mejoras.csv").exists():
         from src import auxiliar_functions as af
         af.comparar_variantes(carpeta_variantes)
     por_objetivo = pd.read_csv(carpeta_variantes / "comparacion_por_objetivo.csv")
     ```
  4. **[notebooks/modelado.ipynb](../notebooks/modelado.ipynb) (Celda `In[16]`, índice `32`, líneas `14–16`):** Se agregó la restauración automática de `evaluacion_test.csv` cuando `registro_previo is not None` y el archivo aún no está en disco:
     ```python
     if not (tm.SALIDA / "evaluacion_test.csv").exists():
         tm.restaurar_artefactos_registrados(d, tm.SALIDA, tm.VERSION)
     tabla_test = pd.read_csv(tm.SALIDA / "evaluacion_test.csv")
     ```

---

## 4. Resultado Final de Ejecución de Scripts y Notebooks

Tras aplicar las correcciones anteriores, se verificó la ejecución completa de todos los componentes:

| Componente | Comando / Método de ejecución | Tiempo | Estado | Observaciones |
| :--- | :--- | :---: | :---: | :--- |
| **Limpieza horaria e integridad** | `python -m src.auxiliar_functions --verificar-hashes` | ~2,5 min | **OK (`ExitCode 0`)** | 3 Excels crudos + `dataset_trabajo.csv` (`29.296 × 421`, `1.172` días) + `tabla_dias_borrados.csv` (`18` días) coinciden exactamente con `data/checksums.json`. |
| **Fase 2 completa (EDA + Preproc + Selección)** | `python -m src.preprocessing_pipeline` | ~1,1 min | **OK (`ExitCode 0`)** | `419 -> 321 -> 269` predictoras; causalidad `OK`; `360` candidatas; ganador en Validation: `forward` (`23` variables, MAE promedio `2.320,9 kW`). |
| **Fase 3 (Optuna + Validation)** | `python -m src.train_model` | ~2,1 min | **OK (`ExitCode 0`)** | 320 trials reproducidos al 100%; ganador: `Random Forest` (`forward`, 23 cols, MAE Validation `2.297,6 kW`, RMSE `2.859,6`, $R^2 = 0,434$). |
| **Regeneración y verificación del `.pkl`** | `python -m src.train_model --regenerar-registro` | ~2,5 min | **OK (`ExitCode 0`)** | Diferencia máxima frente al modelo evaluado en Test: `0.0e+00 kW`. `dataset_final.csv` coincide con `data/checksums.json`. |
| **Comparación de variantes** | `python -m src.auxiliar_functions --comparar-variantes` | ~12,0 min | **OK (`ExitCode 0`)** | Genera `comparacion_mejoras.csv` y `comparacion_por_objetivo.csv` idénticos a las tablas de las notebooks. |
| **Fase 4 (Predicción en frío)** | `python -m src.predict "data/raw/Totalizadores Planta 2022_2023.xlsx"` | ~35 s | **OK (`ExitCode 0`)** | Funciona directamente sobre un clon limpio sin requerir `data/processed/`. Genera `468` predicciones en `results/predicciones.csv` con diferencia `0.000000 kW` en los `171` días de Test. |
| **Notebook 1 (`eda.ipynb`)** | `nbclient.execute()` (120 celdas) | ~1,5 min | **OK (0 errores)** | Todas las tablas, gráficos y conclusiones verificadas. |
| **Notebook 2 (`preprocesamiento.ipynb`)** | `nbclient.execute()` (66 celdas) | ~2,0 min | **OK (0 errores)** | Todos los pasos `fit` en Train, test de causalidad y LOF verificados. |
| **Notebook 3 (`seleccion.ipynb`)** | `nbclient.execute()` (43 celdas) | ~2,5 min | **OK (0 errores)** | Selección por 4 métodos y estudio de ablación ejecutados sin errores. |
| **Notebook 4 (`modelado.ipynb`)** | `nbclient.execute()` (36 celdas) | ~3,5 min | **OK (0 errores)** | Optuna, diagnóstico por par/objetivo, importancia por permutación, tabla final de Test (`2.836,7 kW`) y verificación de `predict.py` ejecutados sin errores. |
