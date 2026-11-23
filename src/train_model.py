"""Fase 3. Baselines, cinco modelos con Optuna, elección en Validation, Test y registro.

Entrada: `data/processed/historial_completo.csv` (SHA-256 verificado contra
`data/checksums.json`), de `src/preprocessing_pipeline.py`. Las candidatas y los
subconjuntos de la selección (etapa 3 de ese módulo) se recalculan en memoria: la
selección es determinista y tarda alrededor de un minuto.

Etapas:
    1. Baselines en Validation: persistencia (frío de D) y media de los últimos 7 cierres.
    2. Para cada modelo (Ridge, Lasso, Random Forest, XGBoost, LightGBM), Optuna busca los
       hiperparámetros, el conjunto de variables (uno de los subconjuntos de la selección)
       y qué aprende el modelo (`OBJETIVOS`: el nivel de y, o su cambio respecto del frío
       de D o de la media de 7 cierres) que minimizan el MAE de la validación temporal
       interna en Train (TimeSeriesSplit de 5 bloques con 1 día de hueco). Las métricas
       siempre se miden sobre y en kW: el modelo devuelve el nivel aunque aprenda el cambio.
    3. La mejor configuración de cada modelo se ajusta con todo Train y se evalúa una vez
       en Validation. Gana el menor MAE de Validation (criterio de la consigna).
    4. Solo si se pide (`evaluar_test=True`): se reentrena el ganador con Train y
       Validation, se evalúa una única vez en Test junto a los dos baselines y se registra:
         * `models/modelo_v2.0.0.pkl` con el modelo, el preprocesamiento ajustado en Train y
           todo lo que `src/predict.py` necesita para pronosticar un Excel nuevo;
         * la versión en `models/model_registry.json` (hiperparámetros, métricas de
           Validation y Test, ruta, SHA-256 y commit de Git);
         * `data/processed/dataset_final.csv` y `data_lineage.json`, con sus
           SHA-256 en `data/checksums.json`.
    5. `regenerar_registro()`: vuelve a ajustar la configuración congelada con
       Train + Validation desde el commit actual, comprueba que predice exactamente lo
       mismo que el .pkl registrado y actualiza el .pkl, el SHA-256 y el commit del
       registro. Conserva las métricas de Test de la evaluación única: no lee Test.

Reglas contra la fuga de información:
    * Optuna nunca ve Validation ni Test: el objetivo es el error interno en Train.
    * El escalado de Ridge y Lasso vive dentro del Pipeline y se ajusta en cada bloque.
    * Test se lee solo en `evaluar_en_test`, después de congelar la elección.

Registro: cada trial, cada candidato en Validation, los baselines y la evaluación final
se agregan a `results/experiment_logs.csv` (hiperparámetros, MAE, RMSE, R² y fecha).

Uso desde la raíz del proyecto:
    python -m src.train_model
    python -m src.train_model --n-trials 20 --modelos Ridge LightGBM
    python -m src.train_model --evaluar-test         # una sola vez, con la elección congelada
    python -m src.train_model --regenerar-registro   # mismo modelo, commit actual, sin leer Test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src import auxiliar_functions as af
from src import preprocessing_pipeline as pp
from src.auxiliar_functions import DATA_DIR, PROJECT_ROOT, RESULTS_DIR, SEMILLA, huella

RUTA_LOG = RESULTS_DIR / "experiment_logs.csv"
MODELOS = ("Ridge", "Lasso", "Random Forest", "XGBoost", "LightGBM")
VERSION = "v2.0.0"  # cambiarla al registrar un modelo con otra configuración
CARPETA_MODELOS = PROJECT_ROOT / "models"
RUTA_CHECKSUMS = DATA_DIR / "checksums.json"
DIAS_CONTEXTO = 28  # la ventana más larga del frío (frio_media_28)


@dataclass
class ConfigModelado:
    modelos: Tuple[str, ...] = MODELOS
    # Trials de Optuna por modelo: más para los que tienen más hiperparámetros. Los lineales
    # buscan un alpha y el conjunto de variables.
    n_trials: Tuple[Tuple[str, int], ...] = (
        ("Ridge", 40), ("Lasso", 40), ("Random Forest", 60), ("XGBoost", 80), ("LightGBM", 100),
    )
    # Subconjuntos de la selección entre los que Optuna elige ("todas" = todas las candidatas).
    conjuntos: Tuple[str, ...] = ("mutual_information", "forward", "consenso_2_metodos",
                                  "ridge", "lasso", "todas")
    # Qué aprende el modelo (claves de OBJETIVOS). DECISIÓN ABIERTA, ver sección 3.1 y 5.2 de
    # notebooks/modelado.ipynb (comparar_variantes() de src/auxiliar_functions.py).
    #   ("nivel",)  -> aprende y (versión registrada v2.0.0).
    #   ("cambio_media7",) -> aprende y − media de 7 cierres (u otra clave de OBJETIVOS).
    #   varias claves -> Optuna elige entre ellas con la validación interna en Train.
    # Con una sola clave no se agrega ningún parámetro a la búsqueda: los trials son los mismos
    # que sin esta opción.
    objetivos: Tuple[str, ...] = ("nivel",)
    n_splits: int = 5
    gap: int = 1
    n_jobs_bloques: int = -1  # los bloques de la validación interna se ajustan en paralelo
    registrar_log: bool = True

    def trials(self, modelo: str) -> int:
        return dict(self.n_trials).get(modelo, 30)


# ===========================================================================
# Métricas y registro de experimentos
# ===========================================================================
def medir(y, predicciones) -> Dict[str, float]:
    """Error en kW (MAE, RMSE) y proporción de variación explicada (R²)."""
    return {
        "mae": round(float(mean_absolute_error(y, predicciones)), 6),
        "rmse": round(float(root_mean_squared_error(y, predicciones)), 6),
        "r2": round(float(r2_score(y, predicciones)), 6),
    }


def registrar(ruta: Path, tipo: str, nombre: str, variante: str, parametros: Dict[str, object],
              particion: str, n_ajuste: int, fechas: pd.DatetimeIndex, metricas: Dict[str, float],
              corrida: str) -> None:
    """Agrega una fila a `results/experiment_logs.csv` sin borrar experimentos anteriores."""
    fila = {
        "corrida_utc": corrida,  # agrupa las filas de una misma ejecución
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
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
    fila.update(metricas)
    encabezado = not ruta.exists() or ruta.stat().st_size == 0
    pd.DataFrame([fila]).to_csv(ruta, mode="a", header=encabezado, index=False, lineterminator="\n")


# ===========================================================================
# Datos
# ===========================================================================
def cargar_datos(cfg_sel: Optional[pp.ConfigSeleccion] = None) -> Dict[str, object]:
    """Historial (SHA-256 verificado), candidatas y subconjuntos de la selección.

    La selección se recalcula con `pp.calcular_seleccion` sobre el mismo historial y la misma
    configuración, así que los subconjuntos siempre corresponden a las candidatas."""
    cfg_sel = cfg_sel or pp.ConfigSeleccion()
    historial = pp.cargar_preprocesado()
    seleccion = pp.calcular_seleccion(historial, cfg_sel)
    candidatas = seleccion["candidatas"]
    datos = pp.particiones(historial, candidatas)
    return {"historial": historial, "candidatas": candidatas, "subconjuntos": seleccion["subconjuntos"],
            "seleccion": seleccion["resumen"], "train": datos["train"], "validation": datos["validation"]}


def particion_test(historial: pd.DataFrame, candidatas: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Filas completas de Test. Se usa solo en la evaluación final."""
    filas = historial.index[historial["particion"] == "test"]
    X = candidatas.loc[filas]
    completas = X.notna().all(axis=1)
    return X[completas], historial.loc[filas[completas], pp.COL_Y]


def baselines(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Persistencia (frío de D) y media de los últimos 7 cierres, sobre las mismas filas."""
    return pd.DataFrame([
        {"modelo": "Persistencia", **medir(y, X[pp.COL_FRIO])},
        {"modelo": "Media móvil de 7 días", **medir(y, X["frio_media_7"])},
    ])


# ===========================================================================
# Qué aprende el modelo (objetivo)
# ===========================================================================
# clave -> (columna base conocida al cierre de D, operación). Con "resta" el modelo aprende
# y − base y al predecir suma la base; con "cociente" aprende y / base y multiplica.
# Un árbol no puede predecir valores fuera de los que vio en Train: si el nivel del frío se
# corre (por ejemplo, un invierno que consume más), aprender el cambio lo deja en rango.
# Para agregar una opción, sumar una entrada; para quitarla, sacarla de ConfigModelado.objetivos.
OBJETIVOS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "nivel": (None, None),                         # y tal cual (modelado anterior)
    "cambio": (pp.COL_FRIO, "resta"),             # y − frío de D
    "cambio_media7": ("frio_media_7", "resta"),    # y − media de los últimos 7 cierres
    "cociente_media7": ("frio_media_7", "cociente"),  # y / media de los últimos 7 cierres
}


class ObjetivoRelativo(BaseEstimator, RegressorMixin):
    """Entrena `estimador` sobre y transformado respecto de una columna base y devuelve el nivel.

    La base es una predictora conocida al cierre de D (el frío de D o su media de 7 cierres),
    así que la transformación no usa información futura.
    """

    def __init__(self, estimador=None, base: str = pp.COL_FRIO, operacion: str = "resta"):
        self.estimador = estimador
        self.base = base
        self.operacion = operacion

    def _base(self, X):
        return np.asarray(X[self.base], dtype=float)

    def fit(self, X, y):
        base = self._base(X)
        y = np.asarray(y, dtype=float)
        objetivo = y - base if self.operacion == "resta" else y / np.clip(base, 1.0, None)
        self.estimador_ = clone(self.estimador).fit(X, objetivo)
        return self

    def predict(self, X):
        pred = self.estimador_.predict(X)
        base = self._base(X)
        return pred + base if self.operacion == "resta" else pred * np.clip(base, 1.0, None)


def columnas_de(columnas: List[str], objetivo: str) -> List[str]:
    """Columnas que recibe el modelo: el subconjunto más la base del objetivo, si no estaba."""
    base = OBJETIVOS[objetivo][0]
    return list(columnas) + ([base] if base and base not in columnas else [])


# ===========================================================================
# Modelos y espacios de búsqueda
# ===========================================================================
def crear_modelo(modelo: str, params: Dict[str, object]):
    """Estimador nuevo, con la semilla fija, para que cada prueba se ajuste desde cero.
    Ridge y Lasso estandarizan dentro de su Pipeline (se ajusta con los datos de cada ajuste)."""
    if modelo == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(random_state=SEMILLA, **params))
    if modelo == "Lasso":
        return make_pipeline(StandardScaler(), Lasso(max_iter=50_000, random_state=SEMILLA, **params))
    if modelo == "Random Forest":
        return RandomForestRegressor(random_state=SEMILLA, n_jobs=1, **params)
    if modelo == "XGBoost":
        return XGBRegressor(random_state=SEMILLA, n_jobs=1, tree_method="hist", **params)
    if modelo == "LightGBM":
        return LGBMRegressor(random_state=SEMILLA, n_jobs=1, deterministic=True, force_col_wise=True,
                             verbosity=-1, **params)
    raise ValueError(f"Modelo desconocido: {modelo}")


def crear_estimador(modelo: str, params: Dict[str, object]):
    """Estimador nuevo. `params["objetivo_y"]` (clave de OBJETIVOS, por defecto "nivel")
    define qué aprende."""
    params = dict(params)
    objetivo = params.pop("objetivo_y", "nivel")
    estimador = crear_modelo(modelo, params)
    base, operacion = OBJETIVOS[objetivo]
    return estimador if base is None else ObjetivoRelativo(estimador, base, operacion)


def sugerir(trial: optuna.Trial, modelo: str) -> Dict[str, object]:
    """Espacio de búsqueda de cada modelo. Los rangos se fijaron antes de mirar Validation."""
    if modelo == "Ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-2, 1e4, log=True)}
    if modelo == "Lasso":
        return {"alpha": trial.suggest_float("alpha", 1e-1, 3e3, log=True)}
    if modelo == "Random Forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.6, 1.0]),
        }
    if modelo == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 30, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            "objective": trial.suggest_categorical("objective", ["reg:squarederror", "reg:absoluteerror"]),
        }
    if modelo == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            "objective": trial.suggest_categorical("objective", ["regression", "regression_l1"]),
        }
    raise ValueError(f"Modelo desconocido: {modelo}")


ESPACIO_BUSQUEDA = {
    "Ridge": "alpha ∈ [0,01; 10.000] (log)",
    "Lasso": "alpha ∈ [0,1; 3.000] (log)",
    "Random Forest": "n_estimators 100–600; max_depth 3–20; min_samples_leaf 1–30; max_features ∈ {sqrt, 0,3, 0,6, 1}",
    "XGBoost": "n_estimators 100–1.000; learning_rate 0,01–0,3 (log); max_depth 2–8; min_child_weight 1–30 (log); "
               "subsample 0,5–1; colsample_bytree 0,2–1; reg_lambda 0,001–30 (log); pérdida cuadrática o absoluta",
    "LightGBM": "n_estimators 100–1.000; learning_rate 0,01–0,3 (log); num_leaves 4–64 (log); min_child_samples 5–60; "
                "subsample 0,5–1; colsample_bytree 0,2–1; reg_lambda 0,001–30 (log); pérdida cuadrática o absoluta",
}


# ===========================================================================
# Optuna (solo Train)
# ===========================================================================
def validacion_interna(cfg: ConfigModelado) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=cfg.n_splits, gap=cfg.gap)


def evaluar_cv(modelo: str, params: Dict[str, object], X: pd.DataFrame, y: pd.Series,
               cfg: ConfigModelado) -> Dict[str, float]:
    """MAE, RMSE y R² medios (y desvío del MAE) en la validación temporal interna."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = cross_validate(crear_estimador(modelo, params), X, y, cv=validacion_interna(cfg),
                             scoring={"mae": "neg_mean_absolute_error",
                                      "rmse": "neg_root_mean_squared_error", "r2": "r2"},
                             n_jobs=cfg.n_jobs_bloques)
    return {"mae": float(-res["test_mae"].mean()), "mae_desvio": float(res["test_mae"].std(ddof=1)),
            "rmse": float(-res["test_rmse"].mean()), "r2": float(res["test_r2"].mean())}


def optimizar(modelo: str, X_tr: pd.DataFrame, y_tr: pd.Series, subconjuntos: Dict[str, List[str]],
              cfg: ConfigModelado, corrida: Optional[str] = None) -> optuna.Study:
    """Busca hiperparámetros y conjunto de variables con TPE (semilla fija, trials en serie)."""
    conjuntos = [c for c in cfg.conjuntos if c in subconjuntos]
    evaluados = X_tr.index[np.concatenate([e for _, e in validacion_interna(cfg).split(X_tr)])]

    def objetivo(trial: optuna.Trial) -> float:
        conjunto = trial.suggest_categorical("conjunto", conjuntos)
        params = sugerir(trial, modelo)
        objetivo_y = (cfg.objetivos[0] if len(cfg.objetivos) == 1
                      else trial.suggest_categorical("objetivo_y", cfg.objetivos))
        trial.set_user_attr("objetivo_y", objetivo_y)
        params = params | {"objetivo_y": objetivo_y}
        columnas = columnas_de(subconjuntos[conjunto], objetivo_y)
        inicio = time.perf_counter()
        metricas = evaluar_cv(modelo, params, X_tr[columnas], y_tr, cfg)
        trial.set_user_attr("segundos", time.perf_counter() - inicio)
        trial.set_user_attr("n_columnas", len(columnas))
        for k in ("mae_desvio", "rmse", "r2"):
            trial.set_user_attr(k, metricas[k])
        if corrida and cfg.registrar_log:
            registrar(RUTA_LOG, "optuna", modelo, f"v2 · {conjunto}", params | {"trial": trial.number},
                      "train_cv", len(X_tr), evaluados,
                      {k: round(metricas[k], 6) for k in ("mae", "rmse", "r2")}, corrida)
        return metricas["mae"]

    estudio = optuna.create_study(direction="minimize", study_name=modelo,
                                  sampler=optuna.samplers.TPESampler(seed=SEMILLA))
    estudio.optimize(objetivo, n_trials=cfg.trials(modelo))
    return estudio


def mejor_configuracion(estudio: optuna.Study) -> Tuple[str, Dict[str, object]]:
    """Conjunto y parámetros del mejor trial, con los fijos que agrega `sugerir`."""
    params = dict(estudio.best_trial.params)
    conjunto = params.pop("conjunto")
    params.setdefault("objetivo_y", estudio.best_trial.user_attrs.get("objetivo_y", "nivel"))
    if estudio.study_name == "LightGBM":
        params["subsample_freq"] = 1
    return conjunto, params


def tabla_trials(estudios: Dict[str, optuna.Study]) -> pd.DataFrame:
    filas = []
    for modelo, estudio in estudios.items():
        for t in estudio.trials:
            filas.append({"modelo": modelo, "trial": t.number, "mae_interno": t.value,
                          "conjunto": t.params.get("conjunto"),
                          **{k: v for k, v in t.user_attrs.items()},
                          "hiperparametros": json.dumps({k: v for k, v in t.params.items() if k != "conjunto"}
                                                        | {"objetivo_y": t.user_attrs.get("objetivo_y", "nivel")})})
    return pd.DataFrame(filas)


# ===========================================================================
# Validation y Test
# ===========================================================================
def evaluar_en_validation(estudios: Dict[str, optuna.Study], subconjuntos: Dict[str, List[str]],
                          X_tr, y_tr, X_val, y_val, corrida: Optional[str] = None,
                          cfg: Optional[ConfigModelado] = None):
    """Ajusta cada mejor configuración con todo Train y la evalúa una vez en Validation."""
    cfg = cfg or ConfigModelado()
    filas, ajustados, predicciones = [], {}, {}
    for modelo, estudio in estudios.items():
        conjunto, params = mejor_configuracion(estudio)
        columnas = columnas_de(subconjuntos[conjunto], params.get("objetivo_y", "nivel"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimador = crear_estimador(modelo, params).fit(X_tr[columnas], y_tr)
        pred = pd.Series(estimador.predict(X_val[columnas]), index=X_val.index)
        metricas = medir(y_val, pred)
        ajustados[modelo], predicciones[modelo] = (estimador, columnas), pred
        filas.append({"modelo": modelo, "conjunto": conjunto, "n_columnas": len(columnas),
                      "mae_interno": estudio.best_value, **metricas,
                      "sesgo": float((pred - y_val).mean()), "hiperparametros": json.dumps(params)})
        if corrida and cfg.registrar_log:
            registrar(RUTA_LOG, "candidato", modelo, f"v2 · {conjunto}", params, "validation",
                      len(X_tr), X_val.index, metricas, corrida)
    tabla = pd.DataFrame(filas).sort_values(["mae", "n_columnas"]).reset_index(drop=True)
    return tabla, ajustados, predicciones


def evaluar_en_test(modelo: str, conjunto: str, params: Dict[str, object], columnas: List[str],
                    X_tr, y_tr, X_val, y_val, X_te, y_te, corrida: Optional[str] = None,
                    cfg: Optional[ConfigModelado] = None):
    """Reentrena con Train + Validation y evalúa una única vez en Test, con los baselines."""
    cfg = cfg or ConfigModelado()
    X_tv, y_tv = pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        final = crear_estimador(modelo, params).fit(X_tv[columnas], y_tv)
    pred = pd.Series(final.predict(X_te[columnas]), index=X_te.index)
    tabla = pd.concat([baselines(X_te, y_te),
                       pd.DataFrame([{"modelo": f"{modelo} (final)", **medir(y_te, pred)}])], ignore_index=True)
    if corrida and cfg.registrar_log:
        for _, fila in tabla.iloc[:2].iterrows():
            registrar(RUTA_LOG, "baseline", fila["modelo"], "v2", {}, "test", 0, X_te.index,
                      {k: fila[k] for k in ("mae", "rmse", "r2")}, corrida)
        registrar(RUTA_LOG, "final", modelo, f"v2 · {conjunto}", params, "test", len(X_tv), X_te.index,
                  medir(y_te, pred), corrida)
    return tabla, final, pred


# ===========================================================================
# Orquestación
# ===========================================================================
def ejecutar(cfg: Optional[ConfigModelado] = None, evaluar_test: bool = False) -> Dict[str, object]:
    cfg = cfg or ConfigModelado()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    corrida = datetime.now(timezone.utc).isoformat()
    d = cargar_datos()
    X_tr, y_tr, _ = d["train"]
    X_val, y_val, _ = d["validation"]

    tabla_baselines = baselines(X_val, y_val)
    if cfg.registrar_log:
        for _, fila in tabla_baselines.iterrows():
            registrar(RUTA_LOG, "baseline", fila["modelo"], "v2", {}, "validation", 0, X_val.index,
                      {k: fila[k] for k in ("mae", "rmse", "r2")}, corrida)

    estudios = {}
    for modelo in cfg.modelos:
        inicio = time.perf_counter()
        estudios[modelo] = optimizar(modelo, X_tr, y_tr, d["subconjuntos"], cfg, corrida)
        print(f"{modelo:<14} {cfg.trials(modelo):>3} trials en {time.perf_counter() - inicio:6.0f} s | "
              f"MAE interno {estudios[modelo].best_value:,.1f} | {mejor_configuracion(estudios[modelo])}")
    comparacion, ajustados, _ = evaluar_en_validation(estudios, d["subconjuntos"], X_tr, y_tr, X_val, y_val,
                                                      corrida, cfg)
    ganador = comparacion.iloc[0]

    resultado_test = registro = None
    if evaluar_test:
        X_te, y_te = particion_test(d["historial"], d["candidatas"])
        conjunto, params = mejor_configuracion(estudios[ganador["modelo"]])
        columnas = columnas_de(d["subconjuntos"][conjunto], params.get("objetivo_y", "nivel"))
        resultado_test, final, _ = evaluar_en_test(ganador["modelo"], conjunto, params, columnas,
                                                   X_tr, y_tr, X_val, y_val, X_te, y_te, corrida, cfg)
        registro = registrar_modelo(ganador, conjunto, params, columnas, final, resultado_test, d, X_te, corrida)

    print(pd.concat([comparacion.drop(columns="hiperparametros"), tabla_baselines]).round(1).to_string(index=False))
    print(f"Ganador en Validation: {ganador['modelo']} ({ganador['conjunto']}, {ganador['n_columnas']} columnas)")
    if resultado_test is not None:
        print(resultado_test.round(1).to_string(index=False))
    return {"estudios": estudios, "comparacion": comparacion, "baselines": tabla_baselines,
            "ajustados": ajustados, "test": resultado_test, "registro": registro, "datos": d}


# ===========================================================================
# Registro del modelo final (Fase 3.4) y versionado de datos (Fase 2.4)
# ===========================================================================
def ajustar_preprocesamiento(historial: pd.DataFrame) -> Pipeline:
    """Vuelve a ajustar con Train el pipeline del preprocesamiento (configuración por defecto,
    `pp.Config()`) y comprueba que reproduce exactamente las predictoras de
    `historial_completo.csv`. Es el pipeline que usa la predicción."""
    cfg = pp.Config()
    X, y, cortes, _ = pp.cargar_datos_diarios(af.DEFAULT_INPUT_CSV, af.DEFAULT_CHECKSUMS_JSON, cfg)
    fin_train = cortes["train"][1]
    pipeline = Pipeline(pp.construir_pasos(cfg, pp.cargar_decisiones()))
    pipeline.fit(X.loc[:fin_train], y.reindex(X.loc[:fin_train].index))
    reconstruido = pipeline.transform(X)
    predictoras = [c for c in historial.columns if c not in pp.COLUMNAS_NO_PREDICTORAS]
    try:
        pd.testing.assert_frame_equal(reconstruido[predictoras], historial[predictoras],
                                      check_freq=False, check_names=False, check_dtype=False)
    except (AssertionError, KeyError) as error:
        raise ValueError("El preprocesamiento con la configuración por defecto no reproduce "
                         "historial_completo.csv. Volver a correr python -m src.preprocessing_pipeline "
                         "sin opciones antes de registrar el modelo.") from error
    return pipeline


def columnas_requeridas(columnas_modelo: List[str], predictoras: List[str], cfg_sel: pp.ConfigSeleccion,
                        creadas: pp.VariablesCreadas) -> List[str]:
    """Mediciones del Excel que hacen falta para calcular las predictoras del modelo."""
    binning = {alias: col for alias, col, _ in cfg_sel.binning}
    interaccion = dict(cfg_sel.variables_interaccion)
    necesarias = {pp.COL_FRIO}  # también se usa para validar y limpiar cada jornada
    for c in columnas_modelo:
        if c in predictoras:
            necesarias.add(c.removesuffix(pp.SUFIJO_INDICADOR))
        elif c.startswith("frio_"):
            necesarias.add(pp.COL_FRIO)
        elif c.startswith("bin_"):
            necesarias.add(binning[c.split("_")[1]])
        elif c.startswith("int_"):
            a, b = c[len("int_"):].split("_x_")
            necesarias.update(interaccion[n] for n in (a, b) if n in interaccion)
        elif c in creadas.topes_:
            necesarias.update(creadas.topes_[c][:2])
        elif c in creadas.indicadores_produccion_:
            necesarias.add(creadas.indicadores_produccion_[c])
        elif pp.familia_de(c) not in ("calendario", "ciclo anual"):
            raise ValueError(f"No están definidos los insumos de la predictora '{c}'.")
    return sorted(necesarias)


def _commit_actual() -> Tuple[str, bool]:
    """Hash del commit de HEAD y si el código que entrena (src/ y requirements.txt) tiene
    cambios sin commitear. Los archivos que escribe el propio registro no cuentan."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    cambios = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "requirements.txt"],
        cwd=PROJECT_ROOT, text=True).strip()
    return commit, bool(cambios)


def _escribir_json(ruta: Path, datos: Dict[str, object]) -> None:
    with open(ruta, "w", encoding="utf-8", newline="\n") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")


def exportar_datos_finales(d: Dict[str, object], columnas: List[str], X_te: pd.DataFrame) -> Dict[str, str]:
    """Escribe dataset_final.csv (las filas que usó el modelo, sin escalar) y agrega a
    data/checksums.json los SHA-256 de los datos procesados. Devuelve los checksums."""
    historial, candidatas = d["historial"], d["candidatas"]
    X_tr, X_val = d["train"][0], d["validation"][0]
    filas = X_tr.index.append(X_val.index).append(X_te.index)
    final_df = candidatas.loc[filas, columnas].join(historial.loc[filas, [pp.COL_Y, "particion"]])
    carpeta_datos = pp.CARPETA_PREPROCESADO
    final_df.to_csv(carpeta_datos / "dataset_final.csv", index=True, encoding="utf-8", lineterminator="\n",
                    date_format="%Y-%m-%d")
    with open(RUTA_CHECKSUMS, encoding="utf-8") as f:
        checksums = json.load(f)
    for nombre in ("historial_completo.csv", "train.csv", "validation.csv", "test.csv", "dataset_final.csv"):
        checksums[f"data/processed/{nombre}"] = huella(carpeta_datos / nombre)
    _escribir_json(RUTA_CHECKSUMS, checksums)
    return checksums


def escribir_linaje(d: Dict[str, object], columnas: List[str], X_te: pd.DataFrame, version: str,
                    modelo: str, conjunto: str, corrida: str, pipeline: Pipeline) -> Dict[str, object]:
    """data/processed/data_lineage.json: scripts, transformaciones y huellas.

    `pipeline` es el preprocesamiento ajustado que se guarda en el .pkl: de él salen las
    decisiones de la EDA y las columnas que deja cada paso."""
    candidatas = d["candidatas"]
    X_tr, X_val = d["train"][0], d["validation"][0]
    carpeta_datos = pp.CARPETA_PREPROCESADO
    ruta_final = carpeta_datos / "dataset_final.csv"
    ruta_modelo = CARPETA_MODELOS / f"modelo_{version}.pkl"
    final_df = pd.read_csv(ruta_final, index_col=0)
    with open(RUTA_CHECKSUMS, encoding="utf-8") as f:
        checksums = json.load(f)
    eda = pipeline.named_steps["aplicar_decisiones_eda"].decisiones
    salidas_prep = [f"data/processed/{n}" for n in ("historial_completo.csv", "train.csv", "validation.csv", "test.csv")]

    linaje = {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version_modelo": version,
        "etapas": [
            {"etapa": "Limpieza horaria y unificación de los Excel", "script": "src/auxiliar_functions.py",
             "salida": pp.CLAVE_ENTRADA, "sha256": eda["entrada"]["sha256"]},
            {"etapa": "Decisiones estructurales sobre las columnas (EDA)", "script": "src/preprocessing_pipeline.py (etapa eda)",
             "transformaciones": eda["conteo_por_categoria"],
             "columnas": f"{eda['entrada']['columnas']} -> {eda['columnas_salida']}"},
            {"etapa": "Preprocesamiento ajustado con Train", "script": "src/preprocessing_pipeline.py",
             "fin_ajuste": str(pp.fin_de_ajuste(d["historial"]).date()),
             "transformaciones": {n: f"{p.resumen()['columnas_entrada']} -> {p.resumen()['columnas_salida']}"
                                  for n, p in pipeline.steps},
             "salidas": {ruta: checksums.get(ruta) for ruta in salidas_prep}},
            {"etapa": "Variables creadas y selección", "script": "src/preprocessing_pipeline.py (etapa seleccion)",
             "candidatas": len(candidatas.columns),
             "subconjuntos": {n: len(c) for n, c in d["subconjuntos"].items()}},
            {"etapa": "Modelado, reentrenamiento con Train + Validation y evaluación en Test",
             "script": "src/train_model.py", "fecha_utc": corrida, "modelo": modelo,
             "conjunto": conjunto, "columnas": len(columnas), "archivo_modelo": f"models/{ruta_modelo.name}"},
        ],
        "dataset_final": {
            "archivo": "data/processed/dataset_final.csv", "sha256": huella(ruta_final),
            "filas": int(len(final_df)), "columnas_predictoras": len(columnas),
            "shapes": {"X_train": list(X_tr[columnas].shape), "X_validation": list(X_val[columnas].shape),
                       "X_test": list(X_te[columnas].shape)},
        },
        "escalado": "Los CSV quedan sin escalar. Ridge y Lasso estandarizan dentro de su Pipeline, "
                    "que se ajusta con los datos de entrenamiento de cada ajuste.",
    }
    _escribir_json(carpeta_datos / "data_lineage.json", linaje)
    return linaje


PIPELINE_REGISTRADO = "src/auxiliar_functions.py, src/preprocessing_pipeline.py, src/train_model.py"


def guardar_bundle(final, columnas: List[str], d: Dict[str, object], version: str = VERSION
                   ) -> Tuple[Path, Pipeline]:
    """models/modelo_<version>.pkl: el modelo con todo lo que `src/predict.py` necesita para
    pronosticar un Excel nuevo (preprocesamiento y variables creadas ajustados con Train).
    Devuelve la ruta y el pipeline del preprocesamiento guardado."""
    historial = d["historial"]
    X_tr = d["train"][0]
    cfg_sel = pp.ConfigSeleccion()
    _, creadas = pp.construir_candidatas_y_ajuste(historial, cfg_sel)
    pipeline = ajustar_preprocesamiento(historial)
    predictoras = [c for c in historial.columns if c not in pp.COLUMNAS_NO_PREDICTORAS]

    # Calibración de la limpieza horaria (primer año), la misma que usa la limpieza de entrenamiento.
    crudo, *_ = af.unificar_archivos_excel(af.get_default_excel_sources())
    df_mad_ee, _, umbral_salto = af.calibrar_limpieza(crudo)

    bundle = {
        "tipo": "v2", "version": version, "modelo": final, "columnas": list(columnas),
        "pipeline": pipeline, "columnas_entrada": list(pipeline.steps[0][1].columnas_entrada_),
        "cfg_seleccion": cfg_sel, "variables_creadas": creadas,
        "requeridas": columnas_requeridas(columnas, predictoras, cfg_sel, creadas),
        "medianas_train": X_tr[columnas].median(), "dias_contexto": DIAS_CONTEXTO,
        "df_mad_ee": df_mad_ee, "umbral_salto": umbral_salto,
    }
    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    ruta_modelo = CARPETA_MODELOS / f"modelo_{version}.pkl"
    joblib.dump(bundle, ruta_modelo, compress=3)
    return ruta_modelo, pipeline


def predicciones_historial(final, columnas: List[str], d: Dict[str, object]) -> pd.DataFrame:
    """Predicción del modelo final para cada día con todas sus predictoras (Train, Validation,
    Test y días sin etiqueta). No usa y: sirve para verificar que un modelo regenerado es el mismo."""
    X = d["candidatas"][columnas]
    completas = X.index[X.notna().all(axis=1)]
    pred = pd.DataFrame({"particion": d["historial"].loc[completas, "particion"],
                         "prediccion_frio_kw": final.predict(X.loc[completas])}, index=completas)
    pred.index.name = "fecha"
    return pred


def registrar_modelo(ganador: pd.Series, conjunto: str, params: Dict[str, object], columnas: List[str],
                     final, resultado_test: pd.DataFrame, d: Dict[str, object], X_te: pd.DataFrame,
                     corrida: str, version: str = VERSION) -> Dict[str, object]:
    """Guarda el modelo final con su preprocesamiento, lo registra y versiona los datos finales."""
    commit, cambios = _commit_actual()  # antes de escribir cualquier archivo
    X_tr, X_val = d["train"][0], d["validation"][0]
    ruta_modelo, pipeline = guardar_bundle(final, columnas, d, version)
    checksums = exportar_datos_finales(d, columnas, X_te)

    fila_val = ganador.to_dict()
    metricas_test = {k: float(v) for k, v in resultado_test.iloc[-1][["mae", "rmse", "r2"]].items()}
    ruta_registro = CARPETA_MODELOS / "model_registry.json"
    registro = json.loads(ruta_registro.read_text(encoding="utf-8")) if ruta_registro.exists() else {}
    registro[version] = {
        "path": f"models/{ruta_modelo.name}", "sha256": huella(ruta_modelo),
        "git_commit": commit, "cambios_sin_commitear_al_entrenar": cambios,
        "pipeline": PIPELINE_REGISTRADO,
        "modelo": ganador["modelo"], "conjunto": conjunto, "hiperparametros": params,
        "columnas": list(columnas), "semilla": SEMILLA, "n_train_validation": int(len(X_tr) + len(X_val)),
        "metricas_validation": {k: float(fila_val[k]) for k in ("mae", "rmse", "r2")},
        "metricas_test": metricas_test,
        "baselines_test": {f["modelo"]: float(f["mae"]) for _, f in resultado_test.iloc[:2].iterrows()},
        "test_sha256": checksums["data/processed/test.csv"],
        "corrida_utc": corrida,
    }
    _escribir_json(ruta_registro, registro)

    escribir_linaje(d, columnas, X_te, version, ganador["modelo"], conjunto, corrida, pipeline)
    print(f"Modelo {version} registrado en models/model_registry.json ({ruta_modelo.name}).")
    return registro[version]


def regenerar_registro(version: str = VERSION, tolerancia_kw: float = 1e-6) -> Dict[str, object]:
    """Vuelve a ajustar con Train + Validation la configuración registrada, desde el código del
    commit actual, y actualiza el .pkl, su SHA-256 y el commit del registro.

    Antes de escribir nada comprueba que el modelo nuevo predice lo mismo que el .pkl
    registrado (SHA-256 verificado) en cada día del historial, hasta `tolerancia_kw`. Las
    métricas de Test son las de la evaluación única: esta función no calcula ninguna métrica
    sobre Test.
    """
    commit, cambios = _commit_actual()
    ruta_registro = CARPETA_MODELOS / "model_registry.json"
    registro = json.loads(ruta_registro.read_text(encoding="utf-8"))
    if version not in registro:
        raise ValueError(f"{version} no está en el registro: primero hay que evaluarla con --evaluar-test.")
    previo = registro[version]
    ruta_previa = PROJECT_ROOT / previo["path"]
    if not ruta_previa.exists() or huella(ruta_previa) != previo["sha256"]:
        raise ValueError(f"{previo['path']} falta o no coincide con el SHA-256 del registro: sin el modelo "
                         "registrado no se puede comprobar que el regenerado sea el mismo.")
    modelo_previo = joblib.load(ruta_previa)["modelo"]  # se lee antes de sobrescribir el .pkl

    d = cargar_datos()
    modelo, conjunto, params = previo["modelo"], previo["conjunto"], dict(previo["hiperparametros"])
    columnas = columnas_de(d["subconjuntos"][conjunto], params.get("objetivo_y", "nivel"))
    if columnas != previo["columnas"]:
        raise ValueError(f"Las columnas del subconjunto '{conjunto}' no son las registradas para {version}.")
    X_tv = pd.concat([d["train"][0], d["validation"][0]])
    y_tv = pd.concat([d["train"][1], d["validation"][1]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        final = crear_estimador(modelo, params).fit(X_tv[columnas], y_tv)

    pred = predicciones_historial(final, columnas, d)["prediccion_frio_kw"]
    referencia = predicciones_historial(modelo_previo, columnas, d)["prediccion_frio_kw"]
    diferencia = float((pred - referencia).abs().max())
    if diferencia > tolerancia_kw:
        raise ValueError(f"El modelo regenerado no es el registrado: difiere hasta {diferencia:.6f} kW.")

    X_te = particion_test(d["historial"], d["candidatas"])[0]  # solo predictoras, para el linaje
    ruta_modelo, pipeline = guardar_bundle(final, columnas, d, version)
    checksums = exportar_datos_finales(d, columnas, X_te)
    previo.update({
        "path": f"models/{ruta_modelo.name}", "sha256": huella(ruta_modelo),
        "git_commit": commit, "cambios_sin_commitear_al_entrenar": cambios,
        "pipeline": PIPELINE_REGISTRADO,
        "test_sha256": checksums["data/processed/test.csv"],
        "regenerado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nota_regeneracion": ("Reajustado desde git_commit con la configuración congelada; predice lo mismo "
                              f"que el modelo evaluado (diferencia máxima {diferencia:.1e} kW). Las métricas "
                              "de Test son las de la evaluación única de corrida_utc."),
    })
    registro[version] = previo
    _escribir_json(ruta_registro, registro)
    escribir_linaje(d, columnas, X_te, version, modelo, conjunto, previo["corrida_utc"], pipeline)
    print(f"{version} regenerado desde {commit[:7]}{' (con cambios sin commitear en src/)' if cambios else ''}: "
          f"{ruta_modelo.name}, diferencia máxima con el modelo registrado {diferencia:.1e} kW.")
    return previo


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 3: modelado con Optuna, Test y registro del modelo.")
    parser.add_argument("--modelos", nargs="*", choices=list(MODELOS), default=list(MODELOS))
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Trials por modelo (por defecto, los de ConfigModelado).")
    parser.add_argument("--evaluar-test", action="store_true",
                        help="Reentrena el ganador con Train+Validation y lo evalúa una vez en Test.")
    parser.add_argument("--regenerar-registro", action="store_true",
                        help="Reajusta el modelo registrado desde el commit actual, sin búsqueda ni Test.")
    parser.add_argument("--sin-log", action="store_true", help="No agrega filas a experiment_logs.csv.")
    args = parser.parse_args()
    if args.regenerar_registro:
        regenerar_registro()
        return
    cfg = ConfigModelado(modelos=tuple(args.modelos), registrar_log=not args.sin_log)
    if args.n_trials:
        cfg.n_trials = tuple((m, args.n_trials) for m in MODELOS)
    ejecutar(cfg, evaluar_test=args.evaluar_test)


if __name__ == "__main__":
    main()
