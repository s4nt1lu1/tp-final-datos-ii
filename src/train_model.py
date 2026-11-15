"""Fase 3. Ajuste en Train, elección en Validation y evaluación final en Test."""

import argparse  # Permite ejecutar la comparación y el cierre desde la terminal.
import json  # Guarda la selección y el registro de modelos en formato legible.
import subprocess  # Obtiene el commit del código con el que se entrenó el modelo.
from datetime import datetime, timezone  # Registra la fecha de cada experimento en UTC.
from pathlib import Path  # Construye rutas sin depender del separador del sistema.

import joblib  # Guarda juntos el preparador, el estimador y el orden de columnas.
import numpy as np  # Comprueba que las entradas numéricas tengan valores finitos.
import pandas as pd  # Lee las particiones y organiza tablas de resultados.
from lightgbm import LGBMRegressor  # Árboles que se agregan de manera secuencial.
from sklearn.ensemble import RandomForestRegressor  # Promedia árboles entrenados por separado.
from sklearn.linear_model import Ridge  # Regresión lineal regularizada con penalización L2.
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error  # Métricas del TP.
from sklearn.pipeline import make_pipeline  # Une la escala y el estimador lineal.
from sklearn.preprocessing import StandardScaler  # Centra y escala cada columna con el ajuste.
from xgboost import XGBRegressor  # Segundo método de boosting con árboles.

from src.auxiliar_functions import PROJECT_ROOT, SEMILLA, huella  # Rutas, semilla y SHA-256 compartidos.
from src.feature_engineering import PreparadorFeatures, features_temporales, ordenar_diario  # Ingeniería de Fase 2.


VERSION = "v1.0.0"  # Versión del primer modelo que se registra.
PROCESADOS = PROJECT_ROOT / "data" / "processed"  # Entradas sin escalar de Fase 2.
NOMBRE_LOG = "experiment_logs.csv"  # Historial de baselines, ajustes y evaluación final.

# Espacio de búsqueda fijado antes de abrir Test. Las listas tienen configuraciones completas.
BUSQUEDA = {
    "Ridge": [
        {"alpha": 0.1}, {"alpha": 1}, {"alpha": 10}, {"alpha": 50}, {"alpha": 150}, {"alpha": 400},
    ],
    "Random Forest": [
        {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 8, "criterion": "squared_error"},
        {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 6, "criterion": "squared_error"},
        {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 3, "criterion": "squared_error"},
        {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 10, "criterion": "absolute_error"},
    ],
    "XGBoost": [
        {"n_estimators": 120, "max_depth": 2, "learning_rate": 0.05, "objective": "reg:squarederror"},
        {"n_estimators": 180, "max_depth": 3, "learning_rate": 0.04, "objective": "reg:squarederror"},
        {"n_estimators": 120, "max_depth": 2, "learning_rate": 0.05, "objective": "reg:absoluteerror"},
        {"n_estimators": 180, "max_depth": 3, "learning_rate": 0.04, "objective": "reg:absoluteerror"},
    ],
    "LightGBM": [
        {"n_estimators": 120, "num_leaves": 7, "learning_rate": 0.05, "min_child_samples": 18, "objective": "regression"},
        {"n_estimators": 180, "num_leaves": 15, "learning_rate": 0.04, "min_child_samples": 18, "objective": "regression"},
        {"n_estimators": 120, "num_leaves": 7, "learning_rate": 0.05, "min_child_samples": 18, "objective": "regression_l1"},
        {"n_estimators": 180, "num_leaves": 15, "learning_rate": 0.04, "min_child_samples": 18, "objective": "regression_l1"},
    ],
}


def leer_json(ruta):
    """Lee un JSON con la codificación usada en el repositorio."""
    return json.loads(ruta.read_text(encoding="utf-8"))  # Convierte el archivo en un diccionario.


def escribir_json(ruta, datos):
    """Guarda números, fechas y parámetros en un archivo fácil de revisar."""
    texto = json.dumps(datos, ensure_ascii=False, indent=2)  # Conserva los caracteres en español.
    ruta.write_text(texto + "\n", encoding="utf-8", newline="\n")  # Fija saltos de línea.


def leer_csv_verificado(nombre, checksums):
    """Lee un CSV regenerado por la Fase 2 y comprueba su huella."""
    ruta = PROCESADOS / nombre  # Encuentra el archivo de entrada.
    clave = f"data/processed/{nombre}"  # Usa la misma clave que la Fase 2.
    if not ruta.is_file():  # Detecta un clon antes de la generación de datos.
        raise FileNotFoundError(f"Falta {clave}. Ejecutá la Fase 2 siguiendo el README.")
    if huella(ruta) != checksums[clave]:  # Rechaza particiones distintas a las versionadas.
        raise ValueError(f"El SHA-256 de {clave} difiere del registro.")
    tabla = pd.read_csv(ruta, parse_dates=["DIA"], index_col="DIA")  # Usa D como índice.
    if tabla.empty or tabla.index.has_duplicates or not tabla.index.is_monotonic_increasing:
        raise ValueError(f"{clave} debe tener fechas únicas y ordenadas.")
    return tabla  # Las columnas numéricas siguen en las unidades de la Fase 2.


def hacer_baselines(diario, fechas):
    """Predice con el frío de D y con la media de los siete días hasta D."""
    previo = diario.loc[:fechas.max()]  # El último cierre permitido es el último D del tramo.
    temporal = features_temporales(previo)  # Reutiliza el calendario y el mínimo de cuatro cierres.
    baselines = temporal.loc[fechas, ["frio_hoy_d", "frio_roll_mean_7"]].copy()  # Mismas fechas que el modelo.
    baselines.columns = ["Persistencia", "Media móvil de 7 días"]  # Etiquetas para tablas y registros.
    if not np.isfinite(baselines.to_numpy()).all():  # Mantiene la misma cobertura para los tres pronósticos.
        raise ValueError("Falta historial para calcular los dos baselines sobre las fechas evaluadas.")
    return baselines  # Entrega los pronósticos listos para comparar con y.


def crear_modelo(nombre, parametros):
    """Crea un estimador nuevo para que cada prueba se ajuste desde cero."""
    if nombre == "Ridge":  # Ridge usa un coeficiente por variable y requiere escalas comparables.
        return make_pipeline(StandardScaler(), Ridge(random_state=SEMILLA, **parametros))
    if nombre == "Random Forest":  # Los árboles trabajan con las unidades originales.
        return RandomForestRegressor(random_state=SEMILLA, n_jobs=1, **parametros)
    if nombre == "XGBoost":  # Se fija el algoritmo de histogramas y un solo hilo en CPU.
        return XGBRegressor(random_state=SEMILLA, n_jobs=1, tree_method="hist", **parametros)
    if nombre == "LightGBM":  # Se fija la forma de construir histogramas para repetir la corrida.
        return LGBMRegressor(
            random_state=SEMILLA, n_jobs=1, deterministic=True, force_col_wise=True,
            verbosity=-1, **parametros,
        )
    raise ValueError(f"Modelo desconocido: {nombre}")  # Evita reutilizar una configuración inválida.


def medir(y, predicciones):
    """Calcula el error en kW y la proporción de variación explicada."""
    return {
        "mae": round(float(mean_absolute_error(y, predicciones)), 6),  # Error absoluto medio.
        "rmse": round(float(root_mean_squared_error(y, predicciones)), 6),  # Da más peso a errores grandes.
        "r2": round(float(r2_score(y, predicciones)), 6),  # Se compara con predecir la media de y.
    }


def registrar(ruta, tipo, nombre, variante, parametros, particion, n_ajuste, fechas, metricas, corrida):
    """Agrega una fila por baseline o ajuste sin borrar experimentos anteriores."""
    fila = {
        "corrida_utc": corrida,  # Agrupa todas las filas de la misma sesión de entrenamiento.
        "fecha_utc": datetime.now(timezone.utc).isoformat(),  # Fecha del registro de esta prueba.
        "tipo": tipo,  # Identifica baseline, candidato o modelo final.
        "modelo": nombre,  # Nombre de la técnica evaluada.
        "variante": variante,  # Subconjunto de predictoras.
        "particion": particion,  # Período usado para medir el error.
        "n_ajuste": n_ajuste,  # Cantidad de filas que vieron los parámetros.
        "n_evaluacion": len(fechas),  # Cantidad de fechas que entraron en las métricas.
        "desde_D": fechas.min().date().isoformat(),  # Primera fecha de origen evaluada.
        "hasta_D": fechas.max().date().isoformat(),  # Última fecha de origen evaluada.
        "hiperparametros": json.dumps(parametros, sort_keys=True),  # Conserva la configuración completa.
    }
    fila.update(metricas)  # Incorpora MAE, RMSE y R² para comparar los experimentos.
    encabezado = not ruta.exists() or ruta.stat().st_size == 0  # Escribe los títulos en la primera fila.
    pd.DataFrame([fila]).to_csv(ruta, mode="a", header=encabezado, index=False, lineterminator="\n")


def comparar_modelos(salida):
    """Evalúa los modelos en Validation y guarda la selección para el cierre."""
    resultados = salida / "results"  # Agrupa el log y la elección de la fase.
    resultados.mkdir(parents=True, exist_ok=True)  # Permite correr desde una carpeta recién clonada.
    ruta_log = resultados / NOMBRE_LOG  # El log conserva filas de corridas anteriores.
    corrida = datetime.now(timezone.utc).isoformat()  # Etiqueta común para esta tanda.
    checksums = leer_json(PROJECT_ROOT / "data" / "checksums.json")  # Huellas de entrada.
    train = leer_csv_verificado("train.csv", checksums)  # Datos para todos los ajustes.
    val = leer_csv_verificado("validation.csv", checksums)  # Datos para elegir.
    diario = leer_csv_verificado("dataset_diario_base.csv", checksums)  # Historial para los baselines.
    if train.index.max() >= val.index.min():  # Corrobora el orden temporal de la Fase 2.
        raise ValueError("Validation debe empezar después del último D de Train.")
    columnas = train.columns.drop("y").tolist()  # y es la etiqueta de D+1.
    y_train = train["y"]  # Etiquetas usadas para ajustar.
    y_val = val["y"]  # Etiquetas usadas para medir.
    baselines = hacer_baselines(diario, val.index)  # Se calculan antes de entrenar modelos.
    tabla_baselines = []  # Reúne las métricas para el notebook.
    for nombre in baselines.columns:  # Cada baseline recibe exactamente las fechas de Validation.
        metricas = medir(y_val, baselines[nombre])  # Compara el pronóstico con el consumo real siguiente.
        registrar(ruta_log, "baseline", nombre, "", {}, "validation", 0, val.index, metricas, corrida)
        tabla_baselines.append({"modelo": nombre, **metricas})  # Agrega una fila para la tabla.

    # Partimos de las 12 columnas de Fase 2 y probamos dos formas de reducir las medias.
    variantes = {
        "Fase 2 completa": columnas,
        "Medias 3 y 14": [c for c in columnas if c != "frio_roll_mean_7"],
        "Solo media 14": [c for c in columnas if c not in ["frio_roll_mean_3", "frio_roll_mean_7"]],
    }
    pruebas = []  # Contendrá una fila por combinación de modelo, parámetros y columnas.
    mejor_ajuste = None  # Conserva el modelo ya entrenado para los gráficos de diagnóstico.
    mejor_criterio = (float("inf"), float("inf"))  # Orden de elección: MAE y cantidad de columnas.
    for variante, elegidas in variantes.items():  # Prueba variantes con las mismas fechas.
        X_train = train[elegidas]  # Toma las columnas en el orden que viene de la Fase 2.
        X_val = val[elegidas]  # Validation usa las mismas variables y el mismo orden.
        for nombre, configuraciones in BUSQUEDA.items():  # Recorre los cuatro modelos pedidos.
            for parametros in configuraciones:  # Cada configuración es una corrida independiente.
                modelo = crear_modelo(nombre, parametros)  # Reinicia el estimador y la escala, si aplica.
                modelo.fit(X_train, y_train)  # Este ajuste ve solamente Train.
                metricas = medir(y_val, modelo.predict(X_val))  # Validation se usa para comparar.
                criterio = (metricas["mae"], len(elegidas))  # Aplica el mismo orden de la tabla final.
                if criterio < mejor_criterio:  # En empate exacto conserva la primera corrida.
                    mejor_criterio = criterio  # Guarda el resultado que tienen que superar los demás.
                    mejor_ajuste = modelo  # La notebook reutiliza este ajuste para el diagnóstico.
                registrar(ruta_log, "candidato", nombre, variante, parametros, "validation",
                          len(train), val.index, metricas, corrida)  # Registra la corrida en disco.
                pruebas.append({
                    "modelo": nombre, "variante": variante, "n_columnas": len(elegidas),
                    "columnas": elegidas, "hiperparametros": parametros, **metricas,
                })  # Conserva los parámetros que permiten repetir esta prueba.

    comparacion = pd.DataFrame(pruebas)  # Convierte la lista en una tabla para ordenar.
    comparacion = comparacion.sort_values(["mae", "n_columnas"], kind="stable").reset_index(drop=True)
    mejor = comparacion.iloc[0].to_dict()  # Menor MAE. En empate se elige el grupo más corto.
    mejor["sha256_entradas"] = {
        f"data/processed/{nombre}": checksums[f"data/processed/{nombre}"]
        for nombre in ["train.csv", "validation.csv", "dataset_diario_base.csv"]
    }  # Guarda las huellas del ajuste y la elección para el cierre.
    escribir_json(resultados / "seleccion_modelado.json", mejor)  # Congela la decisión antes de abrir Test.
    return pd.DataFrame(tabla_baselines), comparacion, mejor, mejor_ajuste  # Incluye el ajuste para diagnóstico.


def cerrar_modelo(salida, version):
    """Reentrena el ganador congelado, mide Test y registra el modelo."""
    resultados = salida / "results"  # Contiene la elección de la etapa anterior.
    carpeta_modelos = salida / "models"  # Almacena el bundle serializado y su registro.
    carpeta_modelos.mkdir(parents=True, exist_ok=True)  # Crea la carpeta en un clon limpio.
    ruta_evaluacion = resultados / "evaluacion_final.json"  # Guarda un único informe final por entrega.
    if ruta_evaluacion.exists():  # Permite consultar el resultado sin repetir la evaluación de Test.
        raise FileExistsError(f"La evaluación final ya está guardada en {ruta_evaluacion}.")
    mejor = leer_json(resultados / "seleccion_modelado.json")  # Recupera la decisión de Validation.
    checksums = leer_json(PROJECT_ROOT / "data" / "checksums.json")  # Carga el registro vigente.
    for clave, esperada in mejor["sha256_entradas"].items():  # Comprueba la versión de los datos de elección.
        if checksums.get(clave) != esperada:
            raise ValueError(f"Cambió la versión de {clave} después de elegir el modelo.")
    train = leer_csv_verificado("train.csv", checksums)  # Primer tramo de ajuste.
    val = leer_csv_verificado("validation.csv", checksums)  # Se incorpora al ajuste final.
    diario = leer_csv_verificado("dataset_diario_base.csv", checksums)  # Fuente de features y baselines.
    columnas = mejor["columnas"]  # Mantiene las columnas elegidas en Validation.
    train_val = pd.concat([train, val])  # Une los dos tramos cronológicos que ya se pueden usar.

    # Recupera el estado de preparación de Fase 2, que produjo los CSV sin escalar.
    historial_train = diario.loc[:train.index.max()].reset_index()  # Historial conocido al ajustar Fase 2.
    preparador = PreparadorFeatures().fit(historial_train, train["y"])  # Aprende exactamente del mismo Train.
    x_reconstruido = preparador.transform(diario.loc[:val.index.max()].reset_index())  # Aplica el estado congelado.
    np.testing.assert_allclose(x_reconstruido.loc[train_val.index, columnas], train_val[columnas])
    modelo = crear_modelo(mejor["modelo"], mejor["hiperparametros"])  # Reinicia el ganador ya elegido.
    modelo.fit(train_val[columnas], train_val["y"])  # Ajusta de nuevo el estimador y la escala, si aplica.
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()

    # Test se abre después de reconstruir el preparador y ajustar el modelo final.
    test = leer_csv_verificado("test.csv", checksums)  # Último tramo, reservado hasta este punto.
    if train_val.index.max() >= test.index.min():  # Detecta contaminación por fechas repetidas.
        raise ValueError("Test debe empezar después del último D de Validation.")
    baselines = hacer_baselines(diario, test.index)  # Produce ambas referencias con las mismas fechas.
    metricas = {}  # Reúne el resultado final para el informe y el registro.
    corrida = datetime.now(timezone.utc).isoformat()  # Etiqueta la corrida que evaluó el Test.
    ruta_log = resultados / NOMBRE_LOG  # Usa el mismo esquema que en Validation.
    for nombre in baselines.columns:  # Registra persistencia y media móvil con el y de Test.
        metricas[nombre] = medir(test["y"], baselines[nombre])  # Se mide con la misma cobertura.
        registrar(ruta_log, "baseline", nombre, "", {}, "test", 0, test.index, metricas[nombre], corrida)
    metricas["Modelo final"] = medir(test["y"], modelo.predict(test[columnas]))  # Único MAE del modelo en Test.
    registrar(ruta_log, "final", mejor["modelo"], mejor["variante"], mejor["hiperparametros"],
              "test", len(train_val), test.index, metricas["Modelo final"], corrida)

    nombre_archivo = f"modelo_{version}.pkl"  # Incluye la versión semántica en el nombre.
    ruta_modelo = carpeta_modelos / nombre_archivo  # Ubicación local del archivo de modelo.
    bundle = {"modelo": modelo, "preparador": preparador, "columnas": columnas, "version": version}
    joblib.dump(bundle, ruta_modelo, compress=3)  # Guarda el modelo y el estado necesario para Fase 4.
    ruta_registro = carpeta_modelos / "model_registry.json"  # Índice para cargar modelos por versión.
    registro = {}  # Un clon inicial puede tener un registro vacío.
    if ruta_registro.exists() and ruta_registro.stat().st_size > 0:  # Conserva versiones previas válidas.
        registro = leer_json(ruta_registro)
    registro[version] = {
        "path": f"models/{nombre_archivo}", "sha256": huella(ruta_modelo), "git_commit": commit,
        "modelo": mejor["modelo"], "hiperparametros": mejor["hiperparametros"], "columnas": columnas,
        "semilla": SEMILLA, "n_train_validation": len(train_val), "test_sha256": checksums["data/processed/test.csv"],
        "metricas_validation": {"mae": mejor["mae"], "rmse": mejor["rmse"], "r2": mejor["r2"]},
        "metricas_test": metricas["Modelo final"],
    }  # Registra ruta, huella, commit, métricas y parámetros de la versión.
    escribir_json(ruta_registro, registro)  # Deja el modelo listo para ser encontrado por su versión.
    escribir_json(ruta_evaluacion, {
        "version": version, "modelo": mejor["modelo"], "variante": mejor["variante"],
        "desde_D": test.index.min().date().isoformat(), "hasta_D": test.index.max().date().isoformat(),
        "filas_test": len(test), "metricas": metricas,
    })  # Guarda agregados para reproducir la tabla del informe sin volver a evaluar.
    return metricas  # Permite ver inmediatamente el resultado final.


def main():
    """Ejecuta la comparación o el cierre usando argumentos de terminal."""
    parser = argparse.ArgumentParser(description="Modelado cronológico del consumo de frío D+1.")
    parser.add_argument("--salida", type=Path, default=PROJECT_ROOT, help="Raíz de resultados y modelos.")
    parser.add_argument("--evaluar-test", action="store_true", help="Reentrena el ganador y evalúa Test.")
    argumentos = parser.parse_args()  # Lee las opciones elegidas en la terminal.
    if argumentos.evaluar_test:  # La evaluación final requiere una decisión explícita.
        print(json.dumps(cerrar_modelo(argumentos.salida, VERSION), indent=2, ensure_ascii=False))
    else:  # El modo normal solo ajusta con Train y compara en Validation.
        baselines, pruebas, mejor, _ = comparar_modelos(argumentos.salida)
        print(baselines.to_string(index=False))  # Muestra las dos referencias de Validation.
        print(pruebas[["modelo", "variante", "mae", "rmse", "r2"]].head(8).to_string(index=False))
        print(f"Ganador en Validation: {mejor['modelo']} | {mejor['variante']} | MAE {mejor['mae']:.2f} kW")


if __name__ == "__main__":  # Ejecuta el CLI. Importar este archivo desde un notebook no entrena modelos.
    main()
