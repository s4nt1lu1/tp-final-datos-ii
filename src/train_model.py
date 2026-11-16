"""Fase 3. Ajuste en Train, elección en Validation y evaluación final en Test."""

import argparse  
import json  
import subprocess  
from datetime import datetime, timezone 
from pathlib import Path  

import joblib  
import numpy as np  
import pandas as pd  
from lightgbm import LGBMRegressor  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.linear_model import Ridge  
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error  
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler  
from xgboost import XGBRegressor  

# Al ejecutar el .py directamente, Python empieza a buscar módulos en src/.
# Agregamos la raíz para que funcionen los mismos imports que con python -m.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.auxiliar_functions import PROJECT_ROOT, SEMILLA, huella  # Rutas, semilla y SHA-256 compartidos.
from src.preprocessing_pipeline import (
    PreparadorFeatures,
    calibrar_limpieza,
    features_temporales,
    get_default_excel_sources,
    ordenar_diario,
    unificar_archivos_excel,
)


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
    ruta = PROCESADOS / nombre  
    clave = f"data/processed/{nombre}"  
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
    temporal = features_temporales(previo)  
    baselines = temporal.loc[fechas, ["frio_hoy_d", "frio_roll_mean_7"]].copy()  # Mismas fechas que el modelo.
    baselines.columns = ["Persistencia", "Media móvil de 7 días"]  # Etiquetas para tablas y registros.
    if not np.isfinite(baselines.to_numpy()).all(): 
        raise ValueError("Falta historial para calcular los dos baselines sobre las fechas evaluadas.")
    return baselines  


def crear_modelo(nombre, parametros):
    """Crea un estimador nuevo para que cada prueba se ajuste desde cero."""
    if nombre == "Ridge":  
        return make_pipeline(StandardScaler(), Ridge(random_state=SEMILLA, **parametros))
    if nombre == "Random Forest": 
        return RandomForestRegressor(random_state=SEMILLA, n_jobs=1, **parametros)
    if nombre == "XGBoost":  
        return XGBRegressor(random_state=SEMILLA, n_jobs=1, tree_method="hist", **parametros)
    if nombre == "LightGBM":  
        return LGBMRegressor(
            random_state=SEMILLA, n_jobs=1, deterministic=True, force_col_wise=True,
            verbosity=-1, **parametros,
        )
    raise ValueError(f"Modelo desconocido: {nombre}")  


def medir(y, predicciones):
    """Calcula el error en kW y la proporción de variación explicada."""
    return {
        "mae": round(float(mean_absolute_error(y, predicciones)), 6),  
        "rmse": round(float(root_mean_squared_error(y, predicciones)), 6),  
        "r2": round(float(r2_score(y, predicciones)), 6),  
    }


def registrar(ruta, tipo, nombre, variante, parametros, particion, n_ajuste, fechas, metricas, corrida):
    """Agrega una fila por baseline o ajuste sin borrar experimentos anteriores."""
    fila = {
        "corrida_utc": corrida,  # Agrupa todas las filas de la misma sesión de entrenamiento.
        "fecha_utc": datetime.now(timezone.utc).isoformat(),  # Fecha del registro de esta prueba.
        "tipo": tipo,  
        "modelo": nombre, 
        "variante": variante, 
        "particion": particion,  
        "n_ajuste": n_ajuste,  
        "n_evaluacion": len(fechas),  
        "desde_D": fechas.min().date().isoformat(),  
        "hasta_D": fechas.max().date().isoformat(),  
        "hiperparametros": json.dumps(parametros, sort_keys=True),  
    }
    fila.update(metricas)  # Incorpora MAE, RMSE y R² para comparar los experimentos.
    encabezado = not ruta.exists() or ruta.stat().st_size == 0  # Escribe los títulos en la primera fila.
    pd.DataFrame([fila]).to_csv(ruta, mode="a", header=encabezado, index=False, lineterminator="\n")


def comparar_modelos(salida):
    """Evalúa los modelos en Validation y guarda la selección para el cierre."""
    resultados = salida / "results"  
    resultados.mkdir(parents=True, exist_ok=True)  
    ruta_log = resultados / NOMBRE_LOG  
    corrida = datetime.now(timezone.utc).isoformat()  
    checksums = leer_json(PROJECT_ROOT / "data" / "checksums.json")  
    train = leer_csv_verificado("train.csv", checksums)  
    val = leer_csv_verificado("validation.csv", checksums)  
    diario = leer_csv_verificado("dataset_diario_base.csv", checksums)  
    if train.index.max() >= val.index.min():  
        raise ValueError("Validation debe empezar después del último D de Train.")
    columnas = train.columns.drop("y").tolist()  
    y_train = train["y"]  
    y_val = val["y"]  
    baselines = hacer_baselines(diario, val.index)  # Se calculan antes de entrenar modelos.
    tabla_baselines = []  # Reúne las métricas para el notebook.
    for nombre in baselines.columns:  # Cada baseline recibe exactamente las fechas de Validation.
        metricas = medir(y_val, baselines[nombre])  # Compara el pronóstico con el consumo real siguiente.
        registrar(ruta_log, "baseline", nombre, "", {}, "validation", 0, val.index, metricas, corrida)
        tabla_baselines.append({"modelo": nombre, **metricas})  # Agrega una fila para la tabla.

    # Partimos de las 12 columnas de Fase 2 y probamos dos formas de reducir las medias.
    excluir_ruido = ["ratio_frio_servicios", "KW Trafo 12_Totalizadores Energia"]
    variantes = {
        "Fase 2 completa": columnas,
        "Medias 3 y 14": [c for c in columnas if c != "frio_roll_mean_7"],
        "Solo media 14": [c for c in columnas if c not in ["frio_roll_mean_3", "frio_roll_mean_7"]],
        "Sin variables negativas (10 vars)": [c for c in columnas if c not in excluir_ruido],
    }

    
    pruebas = []  # Contendrá una fila por combinación de modelo, parámetros y columnas.
    mejor_ajuste = None  # Conserva el modelo ya entrenado para los gráficos de diagnóstico.
    mejor_criterio = (float("inf"), float("inf"))  
    for variante, elegidas in variantes.items():  
        X_train = train[elegidas]  
        X_val = val[elegidas]  
        for nombre, configuraciones in BUSQUEDA.items(): 
            for parametros in configuraciones: 
                modelo = crear_modelo(nombre, parametros)  
                modelo.fit(X_train, y_train)  
                metricas = medir(y_val, modelo.predict(X_val)) 
                criterio = (metricas["mae"], len(elegidas))  
                if criterio < mejor_criterio:  
                    mejor_criterio = criterio  
                    mejor_ajuste = modelo  
                registrar(ruta_log, "candidato", nombre, variante, parametros, "validation",
                          len(train), val.index, metricas, corrida) 
                pruebas.append({
                    "modelo": nombre, "variante": variante, "n_columnas": len(elegidas),
                    "columnas": elegidas, "hiperparametros": parametros, **metricas,
                })  

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
    resultados = salida / "results"  
    carpeta_modelos = salida / "models"  # Almacena el bundle serializado y su registro.
    carpeta_modelos.mkdir(parents=True, exist_ok=True)  # Crea la carpeta en un clon limpio.
    ruta_evaluacion = resultados / "evaluacion_final.json"  
    if ruta_evaluacion.exists():  
        raise FileExistsError(f"La evaluación final ya está guardada en {ruta_evaluacion}.")
    mejor = leer_json(resultados / "seleccion_modelado.json")  # Recupera la decisión de Validation.
    checksums = leer_json(PROJECT_ROOT / "data" / "checksums.json")  # Carga el registro vigente.
    for clave, esperada in mejor["sha256_entradas"].items():  
        if checksums.get(clave) != esperada:
            raise ValueError(f"Cambió la versión de {clave} después de elegir el modelo.")
    train = leer_csv_verificado("train.csv", checksums)  # Primer tramo de ajuste.
    val = leer_csv_verificado("validation.csv", checksums)  # Se incorpora al ajuste final.
    diario = leer_csv_verificado("dataset_diario_base.csv", checksums)  
    columnas = mejor["columnas"]  
    train_val = pd.concat([train, val])  

    # Recupera el estado de preparación de Fase 2, que produjo los CSV sin escalar.
    historial_train = diario.loc[:train.index.max()].reset_index()  
    preparador = PreparadorFeatures().fit(historial_train, train["y"])  
    x_reconstruido = preparador.transform(diario.loc[:val.index.max()].reset_index())  # Aplica el estado congelado.
    np.testing.assert_allclose(x_reconstruido.loc[train_val.index, columnas], train_val[columnas])
    modelo = crear_modelo(mejor["modelo"], mejor["hiperparametros"])  
    modelo.fit(train_val[columnas], train_val["y"])  # Ajusta de nuevo el estimador y la escala, si aplica.
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()

    # Test se abre después de reconstruir el preparador y ajustar el modelo final.
    test = leer_csv_verificado("test.csv", checksums)  
    if train_val.index.max() >= test.index.min():  # Detecta contaminación por fechas repetidas.
        raise ValueError("Test debe empezar después del último D de Validation.")
    baselines = hacer_baselines(diario, test.index)  
    metricas = {}  
    corrida = datetime.now(timezone.utc).isoformat()  # Etiqueta la corrida que evaluó el Test.
    ruta_log = resultados / NOMBRE_LOG  
    for nombre in baselines.columns:  # Registra persistencia y media móvil con el y de Test.
        metricas[nombre] = medir(test["y"], baselines[nombre])  # Se mide con la misma cobertura.
        registrar(ruta_log, "baseline", nombre, "", {}, "test", 0, test.index, metricas[nombre], corrida)
    metricas["Modelo final"] = medir(test["y"], modelo.predict(test[columnas]))
    registrar(ruta_log, "final", mejor["modelo"], mejor["variante"], mejor["hiperparametros"],
              "test", len(train_val), test.index, metricas["Modelo final"], corrida)

    nombre_archivo = f"modelo_{version}.pkl"  # Incluye la versión semántica en el nombre.
    ruta_modelo = carpeta_modelos / nombre_archivo  # Ubicación local del archivo de model74)
    df_crudo_cal, *_ = unificar_archivos_excel(get_default_excel_sources())
    df_mad_ee, _, umbral_salto = calibrar_limpieza(df_crudo_cal)
    bundle = {
        "modelo": modelo,
        "preparador": preparador,
        "columnas": columnas,
        "version": version,
        "df_mad_ee": df_mad_ee,
        "umbral_salto": umbral_salto,
    }
    joblib.dump(bundle, ruta_modelo, compress=3) 
    ruta_registro = carpeta_modelos / "model_registry.json"  
    registro = {}  
    if ruta_registro.exists() and ruta_registro.stat().st_size > 0:  
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
    return metricas 


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


if __name__ == "__main__": 
    main()
