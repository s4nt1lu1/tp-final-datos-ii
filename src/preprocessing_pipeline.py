"""Fase 2. Preprocesamiento previo a la selección de variables.

Parte del dataset horario que genera `src/limpieza_horaria.py`
(`data/processed/dataset_trabajo.csv`), toma el cierre diario 23:59, arma el
target D+1 y la división cronológica, y aplica una secuencia de pasos
independientes sobre las predictoras:

    0. aplicar_decisiones_eda    : columnas que la EDA excluye o unifica por la estructura
                                   de la planilla (metadatos, componentes y proxies del frío,
                                   columnas calculadas por diferencia, renombradas y copias en
                                   toda la serie). Ver `src/eda.py`. No estima nada.
    1. descartar_no_informativas : columnas vacías o constantes en Train.
    2. descartar_duplicadas      : columnas idénticas a otra en Train.
    3. filtrar_por_nulos         : clasificación de nulos y descarte de las no utilizables.
    4. outliers_univariados      : errores de medición -> NaN, extremos -> winsorización.
    5. imputar_nulos             : método elegido por columna con enmascaramiento en Train.
    6. (análisis) LOF            : días anómalos en el espacio multivariado.

Reglas contra la fuga de información (data leakage):
    * Todo `fit` recibe SOLO filas con fecha <= último día de Train.
    * Todo `transform` es causal: la fila del día t depende solo de filas con
      fecha <= t y de lo aprendido en `fit`. Por eso se puede transformar la
      serie completa de una vez. `verificar_causalidad` lo comprueba.
    * El target `y` nunca se transforma; solo se usa en Train para el reporte.

Cada paso deja un reporte por columna (`reporte_`) y tablas adicionales
(`tablas_`) que se exportan a `results/preprocesamiento/`.
Para agregar, quitar o reordenar pasos, editar `construir_pasos()` o usar
`--omitir` desde la terminal.

Salida (sin escalar) en `data/processed/`: historial_completo.csv, train.csv,
validation.csv y test.csv. El escalado de Ridge y Lasso vive dentro de su Pipeline
de sklearn en `src/train_model.py` y se ajusta solo con los datos de entrenamiento.

Uso desde la raíz del proyecto:
    python -m src.preprocessing_pipeline --completo   # Excel -> limpieza -> EDA -> preprocesamiento
    python -m src.preprocessing_pipeline              # solo el preprocesamiento (ya existen las
                                                      # salidas de src.limpieza_horaria y src.eda)
    python -m src.preprocessing_pipeline --omitir outliers_univariados --tratamiento-lof excluir_train
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import eda
from src import limpieza_horaria as lh
from src.auxiliar_functions import DATA_DIR, PROJECT_ROOT, RESULTS_DIR, SEMILLA, huella

COL_FECHA = lh.DEFAULT_DATE_COL
COL_FRIO = lh.DEFAULT_TARGET_COL
COL_Y = lh.DEFAULT_TARGET_NAME
SUFIJO_INDICADOR = "__imputado"


# ===========================================================================
# Configuración (todo umbral vive acá y se guarda en el resumen de auditoría)
# ===========================================================================
@dataclass
class Config:
    # División cronológica
    proporciones: Tuple[float, float, float] = (0.70, 0.15, 0.15)

    # 0. Las columnas que se excluyen o unifican por la estructura de la planilla
    #    (metadatos, componentes del frío, renombradas, copias...) las decide la
    #    EDA y se leen de results/eda/resumen.json.

    # 2. Duplicadas en Train: qué copia se conserva
    columnas_obligatorias: Tuple[str, ...] = (lh.DEFAULT_TARGET_COL,)
    hojas_preferidas_duplicados: Tuple[str, ...] = (r"_Totalizadores ",)

    # 3. Nulos
    # Columnas que no se descartan por porcentaje de nulos (sí si están vacías o
    # discontinuadas). Las metas son mensuales y se fijan por adelantado; faltan
    # el primer año porque no estaban en el Excel 2020-2022.
    proteger_de_filtro_nulos: Tuple[str, ...] = (r"^Meta ",)
    dias_recientes: int = 60  # sin datos en los últimos N días de Train => discontinuada
    max_nulos_train: float = 0.30  # sobre nulos "efectivos" (excluye los indefinidos sin producción)
    dias_bloque: int = 7  # racha >= N filas => falla prolongada
    patron_kpi_por_hl: str = r"/\s*H[lL]"  # KPI que se dividen por hectolitros
    patron_produccion: str = r"^H[lL] "  # columnas de producción (denominadores candidatos)
    min_cobertura_no_aplica: float = 0.95  # % de nulos explicados por producción <= 0

    # 4. Outliers univariados
    factor_error_grueso: float = 5.0  # > factor x P99 de Train => error de medidor
    cuantil_error_grueso: float = 0.99
    metodo_outliers: str = "iqr"  # clave de DETECTORES_UNIVARIADOS
    tratamiento_outliers: str = "winsorizar"  # "winsorizar" | "nan" | "ninguno"
    k_iqr: float = 3.0
    z_mad: float = 5.0
    percentiles: Tuple[float, float] = (0.005, 0.995)
    frac_ceros_legitimos: float = 0.05  # si hay más ceros, se excluyen del cálculo de límites
    # Un total diario bajo es físicamente posible (el equipo trabajó menos): por
    # defecto solo se recorta el extremo superior.
    recortar_inferior: bool = False

    # 5. Imputación
    metodos_imputacion: Tuple[str, ...] = (
        "mediana_train", "ffill_3", "mediana_dia_semana", "mediana_mes", "media_movil_7", "lag_7",
    )
    frac_enmascarar: float = 0.10
    min_enmascarados: int = 30
    max_largo_racha_simulada: int = 30
    politica_no_aplica: str = "cero"  # "cero" | "imputar"
    umbral_indicador: float = 0.05  # agrega <col>__imputado si los nulos efectivos superan este %

    # 6. LOF
    lof_vecinos: Tuple[int, ...] = (10, 20, 35, 50)
    lof_umbral: float = 1.5
    lof_varianza_pca: Optional[float] = 0.90  # None = sin PCA
    tratamiento_lof: str = "reportar"  # "reportar" | "excluir_train"


# ===========================================================================
# Utilidades
# ===========================================================================
def _largos_de_rachas(mascara: pd.Series) -> List[int]:
    """Largos (en filas) de las rachas de True consecutivas."""
    m = mascara.to_numpy(dtype=bool)
    if not m.any():
        return []
    ids = np.cumsum(np.r_[True, m[1:] != m[:-1]])
    largos = pd.Series(m).groupby(ids).agg(["first", "size"])
    return largos.loc[largos["first"], "size"].astype(int).tolist()


def _en_calendario(s: pd.Series, fn: Callable[[pd.Series], pd.Series]) -> pd.Series:
    """Aplica `fn` sobre la serie con todos los días calendario (los faltantes como NaN)."""
    if s.empty:
        return s
    calendario = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return fn(s.reindex(calendario)).reindex(s.index)


# ===========================================================================
# Clasificación de nulos (función independiente, la usan los pasos 3 y 5)
# ===========================================================================
MECANISMO = {
    "completa": "Sin nulos en Train",
    "vacia": "Estructural: la variable no se registra",
    "discontinuada": "Estructural: dejó de registrarse antes del fin de Train",
    "inicio_tardio": "Estructural: comienza a registrarse después del inicio de Train",
    "no_aplica_sin_produccion": "MNAR determinista: KPI por Hl indefinido sin producción",
    "bloques": "Falla prolongada de registro",
    "esporadicos": "Faltantes aislados",
}


def componentes_nulos(
    s: pd.Series, denominadores: Dict[str, pd.Series], cfg: Config
) -> Dict[str, object]:
    """Separa los nulos de una columna en previos al inicio, posteriores al fin,
    indefinidos por producción cero y el resto (interiores)."""
    nulo = s.isna()
    vacio = pd.Series(False, index=s.index)
    if nulo.all():
        return {"previos": nulo, "posteriores": vacio, "no_aplica": vacio,
                "resto": vacio, "denominador": None, "cobertura_denominador": np.nan}

    primera, ultima = s.first_valid_index(), s.last_valid_index()
    previos = nulo & (s.index < primera)
    posteriores = nulo & (s.index > ultima)
    interior = nulo & ~previos & ~posteriores

    no_aplica, denominador, cobertura = vacio, None, np.nan
    if interior.any() and re.search(cfg.patron_kpi_por_hl, s.name) and denominadores:
        aciertos = {c: int((sin_prod & interior).sum()) for c, sin_prod in denominadores.items()}
        denominador = max(aciertos, key=aciertos.get)
        cobertura = aciertos[denominador] / int(interior.sum())
        if cobertura >= cfg.min_cobertura_no_aplica:
            no_aplica = interior & denominadores[denominador]
        else:
            denominador = None
    return {"previos": previos, "posteriores": posteriores, "no_aplica": no_aplica,
            "resto": interior & ~no_aplica, "denominador": denominador,
            "cobertura_denominador": cobertura}


def mascaras_sin_produccion(X: pd.DataFrame, cfg: Config) -> Dict[str, pd.Series]:
    """Para cada columna de producción, True en los días sin producción (<= 0 o NaN)."""
    return {c: ~(X[c] > 0) for c in X.columns if re.search(cfg.patron_produccion, c)}


def clasificar_nulos(X: pd.DataFrame, y: Optional[pd.Series], cfg: Config) -> pd.DataFrame:
    """Tabla con el tipo de nulo dominante de cada columna (solo usar con Train)."""
    fin = X.index.max()
    denominadores = mascaras_sin_produccion(X, cfg)
    y_al = y.reindex(X.index) if y is not None else None
    filas = []
    for c in X.columns:
        s = X[c]
        comp = componentes_nulos(s, denominadores, cfg)
        efectivos = comp["previos"] | comp["posteriores"] | comp["resto"]
        rachas = _largos_de_rachas(comp["resto"])
        fila = {
            "columna": c,
            "pct_nulos": float(s.isna().mean()),
            "pct_nulos_efectivos": float(efectivos.mean()),
            "n_previos_inicio": int(comp["previos"].sum()),
            "n_posteriores_fin": int(comp["posteriores"].sum()),
            "n_sin_produccion": int(comp["no_aplica"].sum()),
            "n_interiores": int(comp["resto"].sum()),
            "n_rachas": len(rachas),
            "racha_max": max(rachas, default=0),
            "denominador": comp["denominador"],
            "cobertura_denominador": comp["cobertura_denominador"],
            "primera_obs": s.first_valid_index(),
            "ultima_obs": s.last_valid_index(),
        }
        if s.isna().all():
            tipo = "vacia"
        elif (fin - fila["ultima_obs"]).days > cfg.dias_recientes:
            tipo = "discontinuada"
        elif fila["n_previos_inicio"] > max(fila["n_interiores"], fila["n_sin_produccion"]):
            tipo = "inicio_tardio"
        elif fila["n_sin_produccion"] > fila["n_interiores"]:
            tipo = "no_aplica_sin_produccion"
        elif fila["racha_max"] >= cfg.dias_bloque:
            tipo = "bloques"
        elif fila["n_interiores"] > 0:
            tipo = "esporadicos"
        else:
            tipo = "completa"
        fila["tipo"] = tipo
        fila["mecanismo_probable"] = MECANISMO[tipo]

        # ¿Los días con nulo tienen un y distinto? (sugiere MAR/MNAR en vez de MCAR)
        fila["p_valor_y"] = np.nan
        if y_al is not None:
            con_y = y_al.notna()
            grupo_nulo = y_al[efectivos & con_y]
            grupo_ok = y_al[s.notna() & con_y]
            if len(grupo_nulo) >= 10 and len(grupo_ok) >= 10:
                fila["p_valor_y"] = float(mannwhitneyu(grupo_nulo, grupo_ok).pvalue)
        filas.append(fila)
    return pd.DataFrame(filas)


# ===========================================================================
# Base de los pasos
# ===========================================================================
class PasoAuditable(BaseEstimator, TransformerMixin):
    """Contrato de un paso: `fit` con Train, `transform` causal y reporte por columna."""

    def _iniciar_ajuste(self, X: pd.DataFrame) -> None:
        if not isinstance(X.index, pd.DatetimeIndex) or not X.index.is_monotonic_increasing:
            raise ValueError(f"{type(self).__name__}: se espera un índice de fechas ordenado.")
        self.columnas_entrada_ = list(X.columns)
        self.fin_ajuste_ = X.index.max()
        self.tablas_: Dict[str, pd.DataFrame] = {}

    def _validar_entrada(self, X: pd.DataFrame) -> pd.DataFrame:
        faltan = [c for c in self.columnas_entrada_ if c not in X.columns]
        if faltan:
            raise ValueError(f"{type(self).__name__}: faltan columnas {faltan[:5]}...")
        return X[self.columnas_entrada_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columnas_salida_, dtype=object)

    def resumen(self) -> Dict[str, object]:
        res = {
            "columnas_entrada": len(self.columnas_entrada_),
            "columnas_salida": len(self.columnas_salida_),
            "fin_ajuste": str(self.fin_ajuste_.date()),
        }
        if "decision" in getattr(self, "reporte_", pd.DataFrame()).columns:
            res["decisiones"] = self.reporte_["decision"].value_counts().to_dict()
        return res


# ---------------------------------------------------------------------------
# Paso 0. Decisiones estructurales de la EDA
# ---------------------------------------------------------------------------
class AplicarDecisionesEDA(PasoAuditable):
    """Aplica las decisiones de `src/eda.py`: descarta las columnas que la EDA excluye y
    completa cada columna renombrada con su nombre viejo.

    No estima nada: las decisiones dependen de la estructura de la planilla (nombres, hojas,
    identidades entre medidores), no de los datos de Train. El reporte trae la categoría y el
    motivo de cada columna, tomados de `results/eda/decisiones_columnas.csv`.
    """

    def __init__(self, decisiones: Dict[str, object]):
        self.decisiones = decisiones

    def fit(self, X, y=None):
        self._iniciar_ajuste(X)
        descartar = set(self.decisiones["descartar"])
        self.pares_ = dict(self.decisiones["unificar"])
        faltan = sorted((descartar | set(self.pares_) | set(self.pares_.values())) - set(X.columns))
        if faltan:
            raise ValueError(f"Las decisiones de la EDA nombran columnas que no están: {faltan[:5]}")
        tabla = self.decisiones.get("tabla")
        motivo = dict(zip(tabla["columna"], tabla["motivo"])) if tabla is not None else {}
        categoria = dict(zip(tabla["columna"], tabla["categoria"])) if tabla is not None else {}
        self.reporte_ = pd.DataFrame([{
            "columna": c,
            "decision": "unificar" if c in self.pares_ else "descartar" if c in descartar else "conservar",
            "categoria": categoria.get(c, ""),
            "unida_en": self.pares_.get(c, ""),
            "motivo": motivo.get(c, ""),
        } for c in X.columns]).fillna("")
        self.columnas_salida_ = [c for c in X.columns if c not in descartar and c not in self.pares_]
        return self

    def transform(self, X):
        X = self._validar_entrada(X).copy()
        for vieja, nueva in self.pares_.items():
            X[nueva] = X[nueva].combine_first(X[vieja])
        return X[self.columnas_salida_]


# ---------------------------------------------------------------------------
# Paso 1. Columnas no informativas
# ---------------------------------------------------------------------------
class DescartarNoInformativas(PasoAuditable):
    """Descarta columnas vacías o constantes en Train: no aportan variación para aprender."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fit(self, X, y=None):
        self._iniciar_ajuste(X)
        filas = []
        for c in X.columns:
            if X[c].isna().all():
                motivo = "Vacía en Train"
            elif X[c].nunique(dropna=True) <= 1:
                motivo = f"Constante en Train (valor {X[c].dropna().iloc[0]})"
            else:
                motivo = ""
            filas.append({"columna": c, "decision": "descartar" if motivo else "conservar",
                          "motivo": motivo})
        self.reporte_ = pd.DataFrame(filas)
        self.columnas_salida_ = self.reporte_.loc[
            self.reporte_["decision"] == "conservar", "columna"].tolist()
        return self

    def transform(self, X):
        return self._validar_entrada(X)[self.columnas_salida_]


# ---------------------------------------------------------------------------
# Paso 2. Columnas duplicadas en Train
# ---------------------------------------------------------------------------
class DescartarDuplicadas(PasoAuditable):
    """Descarta columnas idénticas (valores y nulos) a otra, comparando en Train.

    Las copias en toda la serie (la misma medición en dos hojas) ya las quitó la EDA.
    Acá quedan las que coinciden solo en Train: el modelo no puede distinguirlas al aprender.

    Entre copias se conserva, en este orden: una columna obligatoria (el frío,
    del que sale el target), una de las hojas preferidas (por defecto las
    "Totalizadores", que son la lectura directa de los medidores) y, si no,
    la primera en el orden original.
    """

    def __init__(self, cfg: Config, decimales: int = 6):
        self.cfg = cfg
        self.decimales = decimales

    def _prioridad(self, columna: str) -> int:
        if columna in self.cfg.columnas_obligatorias:
            return 0
        if any(re.search(p, columna) for p in self.cfg.hojas_preferidas_duplicados):
            return 1
        return 2

    def fit(self, X, y=None):
        self._iniciar_ajuste(X)
        redondeado = X.round(self.decimales)
        huellas = {c: pd.util.hash_pandas_object(redondeado[c], index=False).sum()
                   for c in X.columns}
        orden = sorted(X.columns, key=lambda c: (self._prioridad(c), X.columns.get_loc(c)))
        representante: Dict[int, List[str]] = {}
        duplicado_de: Dict[str, str] = {}
        for c in orden:
            original = next((r for r in representante.get(huellas[c], [])
                             if redondeado[r].equals(redondeado[c])), None)
            if original is None:
                representante.setdefault(huellas[c], []).append(c)
            else:
                duplicado_de[c] = original
        self.reporte_ = pd.DataFrame([{
            "columna": c,
            "decision": "descartar" if c in duplicado_de else "conservar",
            "duplicado_de": duplicado_de.get(c, ""),
        } for c in X.columns])
        self.columnas_salida_ = [c for c in X.columns if c not in duplicado_de]
        return self

    def transform(self, X):
        return self._validar_entrada(X)[self.columnas_salida_]


# ---------------------------------------------------------------------------
# Paso 3. Filtrado por nulos
# ---------------------------------------------------------------------------
class FiltrarPorNulos(PasoAuditable):
    """Clasifica los nulos de cada columna en Train y descarta las que no se pueden usar:
    vacías, discontinuadas (no disponibles al predecir) o con demasiados nulos efectivos."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fit(self, X, y=None):
        self._iniciar_ajuste(X)
        tabla = clasificar_nulos(X, y, self.cfg)
        motivos, protegidas = [], []
        for _, f in tabla.iterrows():
            protegida = any(re.search(p, f["columna"]) for p in self.cfg.proteger_de_filtro_nulos)
            protegidas.append(protegida)
            if f["tipo"] in ("vacia", "discontinuada"):
                motivos.append(f["mecanismo_probable"])
            elif f["pct_nulos_efectivos"] > self.cfg.max_nulos_train and not protegida:
                motivos.append(f"{f['pct_nulos_efectivos']:.1%} de nulos efectivos en Train "
                               f"(> {self.cfg.max_nulos_train:.0%})")
            else:
                motivos.append("")
        tabla["protegida"] = protegidas
        tabla["decision"] = ["descartar" if m else "conservar" for m in motivos]
        tabla["motivo"] = [m or ("Protegida del umbral de nulos" if p and pct > self.cfg.max_nulos_train
                                 else "")
                           for m, p, pct in zip(motivos, protegidas, tabla["pct_nulos_efectivos"])]
        self.reporte_ = tabla
        self.columnas_salida_ = tabla.loc[tabla["decision"] == "conservar", "columna"].tolist()
        return self

    def transform(self, X):
        return self._validar_entrada(X)[self.columnas_salida_]


# ---------------------------------------------------------------------------
# Paso 4. Outliers univariados
# ---------------------------------------------------------------------------
def _limites_iqr(v: pd.Series, cfg: Config) -> Tuple[float, float]:
    q1, q3 = v.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (-np.inf, np.inf) if iqr <= 0 else (q1 - cfg.k_iqr * iqr, q3 + cfg.k_iqr * iqr)


def _limites_mad(v: pd.Series, cfg: Config) -> Tuple[float, float]:
    mediana = v.median()
    mad = 1.4826 * (v - mediana).abs().median()
    return (-np.inf, np.inf) if mad <= 0 else (mediana - cfg.z_mad * mad, mediana + cfg.z_mad * mad)


def _limites_percentil(v: pd.Series, cfg: Config) -> Tuple[float, float]:
    inf, sup = v.quantile(list(cfg.percentiles))
    return float(inf), float(sup)


# Para agregar un detector: función (valores_train, cfg) -> (límite_inf, límite_sup).
DETECTORES_UNIVARIADOS: Dict[str, Callable[[pd.Series, Config], Tuple[float, float]]] = {
    "iqr": _limites_iqr,
    "mad": _limites_mad,
    "percentil": _limites_percentil,
}


class TratarOutliersUnivariados(PasoAuditable):
    """Dos niveles, ambos con parámetros de Train:

    * Error grueso (imposible físicamente): negativo en una variable que en Train
      es >= 0, o mayor a `factor_error_grueso` x P99. Se pasa a NaN y lo imputa el paso 6.
    * Extremo (posible pero atípico): fuera de los límites del detector elegido.
      Se winsoriza, se pasa a NaN o se deja, según `tratamiento_outliers`.

    Los ceros frecuentes (días sin producción) son legítimos: no entran al cálculo
    de límites ni se recortan. Todos los detectores se reportan para compararlos.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fit(self, X, y=None):
        self._iniciar_ajuste(X)
        if self.cfg.metodo_outliers not in DETECTORES_UNIVARIADOS:
            raise ValueError(f"Detector desconocido: {self.cfg.metodo_outliers}")
        filas, self.limites_ = [], {}
        for c in X.columns:
            v = X[c].dropna()
            no_negativa = bool((v >= 0).mean() >= 0.99)
            p_alto = v.quantile(self.cfg.cuantil_error_grueso)
            tope = self.cfg.factor_error_grueso * p_alto if p_alto > 0 else np.inf
            errores = (v > tope) | ((v < 0) & no_negativa)
            base = v[~errores]
            ceros_frecuentes = bool((base == 0).mean() > self.cfg.frac_ceros_legitimos)
            base_limites = base[base != 0] if ceros_frecuentes else base

            fila = {"columna": c, "no_negativa": no_negativa, "tope_error_grueso": tope,
                    "n_errores_train": int(errores.sum()),
                    "ceros_frecuentes": ceros_frecuentes}
            for nombre, detector in DETECTORES_UNIVARIADOS.items():
                inf, sup = detector(base_limites, self.cfg) if len(base_limites) else (-np.inf, np.inf)
                if not self.cfg.recortar_inferior:
                    inf = 0.0 if no_negativa else -np.inf
                if no_negativa:
                    inf = max(inf, 0.0)
                if ceros_frecuentes:
                    inf = min(inf, 0.0)
                fila[f"{nombre}_inf"], fila[f"{nombre}_sup"] = inf, sup
                fila[f"{nombre}_n_extremos_train"] = int(((base < inf) | (base > sup)).sum())
            self.limites_[c] = {
                "no_negativa": no_negativa, "tope": tope,
                "inf": fila[f"{self.cfg.metodo_outliers}_inf"],
                "sup": fila[f"{self.cfg.metodo_outliers}_sup"],
            }
            filas.append(fila)
        self.reporte_ = pd.DataFrame(filas)
        self.columnas_salida_ = list(X.columns)
        # Comparación agregada de detectores (cuánto afecta cada uno a Train).
        n_valores = int(X.notna().sum().sum())
        self.tablas_["comparacion_detectores"] = pd.DataFrame([{
            "detector": nombre,
            "valores_extremos_train": int(self.reporte_[f"{nombre}_n_extremos_train"].sum()),
            "pct_valores_train": self.reporte_[f"{nombre}_n_extremos_train"].sum() / n_valores,
            "columnas_afectadas": int((self.reporte_[f"{nombre}_n_extremos_train"] > 0).sum()),
            "elegido": nombre == self.cfg.metodo_outliers,
        } for nombre in DETECTORES_UNIVARIADOS])
        return self

    def transform(self, X):
        X = self._validar_entrada(X).copy()
        self.conteos_ultimo_transform_ = {}
        for c, lim in self.limites_.items():
            s = X[c]
            error = (s > lim["tope"]) | ((s < 0) & lim["no_negativa"])
            s = s.mask(error)
            fuera = (s < lim["inf"]) | (s > lim["sup"])
            if self.cfg.tratamiento_outliers == "winsorizar":
                s = s.clip(lim["inf"], lim["sup"])
            elif self.cfg.tratamiento_outliers == "nan":
                s = s.mask(fuera)
            X[c] = s
            self.conteos_ultimo_transform_[c] = (int(error.sum()), int(fuera.sum()))
        return X


# ---------------------------------------------------------------------------
# Paso 5. Imputación con búsqueda del mejor método por columna
# ---------------------------------------------------------------------------
def _estadisticos(s: pd.Series) -> Dict[str, object]:
    return {"mediana": s.median(),
            "mediana_dow": s.groupby(s.index.day_of_week).median(),
            "mediana_mes": s.groupby(s.index.month).median()}


def _imp_mediana(s, est):
    return s.fillna(est["mediana"])


def _imp_ffill(limite: int):
    def imputar(s, est):
        return _en_calendario(s, lambda t: t.ffill(limit=limite)).fillna(est["mediana"])
    return imputar


def _imp_dia_semana(s, est):
    por_dia = pd.Series(s.index.day_of_week, index=s.index).map(est["mediana_dow"])
    return s.fillna(por_dia).fillna(est["mediana"])


def _imp_mes(s, est):
    """Mediana del mismo mes calendario en Train (útil para metas mensuales)."""
    por_mes = pd.Series(s.index.month, index=s.index).map(est["mediana_mes"])
    return s.fillna(por_mes).fillna(est["mediana"])


def _imp_media_movil(dias: int):
    def imputar(s, est):
        previa = _en_calendario(s, lambda t: t.shift(1).rolling(dias, min_periods=1).mean())
        return s.fillna(previa).fillna(est["mediana"])
    return imputar


def _imp_lag(dias: int):
    def imputar(s, est):
        return s.fillna(_en_calendario(s, lambda t: t.shift(dias))).fillna(est["mediana"])
    return imputar


# Todos causales: usan solo días anteriores y estadísticos de Train.
# Para agregar un método: función (serie, estadisticos_train) -> serie sin nulos.
METODOS_IMPUTACION: Dict[str, Callable[[pd.Series, Dict[str, object]], pd.Series]] = {
    "mediana_train": _imp_mediana,
    "ffill_3": _imp_ffill(3),
    "mediana_dia_semana": _imp_dia_semana,
    "mediana_mes": _imp_mes,
    "media_movil_7": _imp_media_movil(7),
    "lag_7": _imp_lag(7),
}


class ImputarNulos(PasoAuditable):
    """Imputa cada columna con el método que menor error obtuvo al simular nulos en Train.

    Búsqueda: en Train se ocultan valores observados en rachas con los largos de
    los nulos reales de esa columna, se imputan con cada método y se mide
    MAE / IQR. Gana el menor (empates: el primero de `metodos_imputacion`).
    Los KPI indefinidos por producción cero se completan con 0 si
    `politica_no_aplica == "cero"`: la columna de producción ya lleva esa información.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fit(self, X, y=None):
        self._iniciar_ajuste(X)
        desconocidos = set(self.cfg.metodos_imputacion) - set(METODOS_IMPUTACION)
        if desconocidos:
            raise ValueError(f"Métodos de imputación desconocidos: {desconocidos}")
        rng = np.random.default_rng(SEMILLA)
        denominadores = mascaras_sin_produccion(X, self.cfg)

        self.estadisticos_, self.metodo_, self.denominador_ = {}, {}, {}
        filas, busqueda = [], []
        for c in X.columns:
            s = X[c]
            comp = componentes_nulos(s, denominadores, self.cfg)
            if self.cfg.politica_no_aplica == "cero" and comp["denominador"]:
                self.denominador_[c] = comp["denominador"]
            sin_produccion = comp["no_aplica"] if c in self.denominador_ else s.isna() & False
            efectivos = s.isna() & ~sin_produccion
            self.estadisticos_[c] = _estadisticos(s)

            # Se simulan nulos como los reales: rachas interiores y, si la columna
            # empieza tarde (las metas faltan todo el primer año), un bloque al
            # comienzo de la serie, donde los métodos que miran días previos no tienen historia.
            errores = self._evaluar_metodos(
                s, _largos_de_rachas(comp["resto"]), int(comp["previos"].sum()), rng)
            for metodo, err in errores.items():
                busqueda.append({"columna": c, "metodo": metodo, "mae_relativo": err})
            validos = {m: e for m, e in errores.items() if np.isfinite(e)}
            mejor = (min(validos, key=lambda m: (round(validos[m], 4),
                                                 self.cfg.metodos_imputacion.index(m)))
                     if validos else "mediana_train")
            self.metodo_[c] = mejor
            filas.append({
                "columna": c, "nulos_train": int(s.isna().sum()),
                "pct_nulos_efectivos": float(efectivos.mean()),
                "completados_con_cero_sin_produccion": int(comp["no_aplica"].sum())
                if c in self.denominador_ else 0,
                "denominador": self.denominador_.get(c, ""),
                "metodo_elegido": mejor,
                "mae_relativo_elegido": errores.get(mejor, np.nan),
                "mae_relativo_mediana": errores.get("mediana_train", np.nan),
                "agrega_indicador": bool(efectivos.mean() > self.cfg.umbral_indicador),
            })
        self.reporte_ = pd.DataFrame(filas)
        self.tablas_["busqueda_imputacion"] = pd.DataFrame(busqueda)
        # Un indicador por patrón de nulos: las columnas que faltan exactamente los
        # mismos días (p. ej. todas las metas) comparten un único indicador.
        self.con_indicador_, representante = [], {}
        for c in self.reporte_.loc[self.reporte_["agrega_indicador"], "columna"]:
            patron = X[c].isna().to_numpy().tobytes()
            if patron not in representante:
                representante[patron] = c
                self.con_indicador_.append(c)
            self.reporte_.loc[self.reporte_["columna"] == c, "indicador"] = (
                representante[patron] + SUFIJO_INDICADOR)
        self.columnas_salida_ = list(X.columns) + [c + SUFIJO_INDICADOR for c in self.con_indicador_]
        return self

    def _evaluar_metodos(self, s: pd.Series, rachas: List[int], bloque_inicial: int,
                         rng) -> Dict[str, float]:
        """Oculta valores observados de Train y mide el error de cada método."""
        observados = np.flatnonzero(s.notna().to_numpy())
        objetivo = max(self.cfg.min_enmascarados, int(self.cfg.frac_enmascarar * len(observados)))
        if len(observados) < 2 * objetivo:
            return {m: np.nan for m in self.cfg.metodos_imputacion}
        largos = [min(r, self.cfg.max_largo_racha_simulada) for r in rachas] or [1]
        ocultar = np.zeros(len(s), dtype=bool)
        if bloque_inicial:
            # Los primeros valores observados quedan sin historia previa, como el bloque real.
            ocultar[observados[:min(bloque_inicial, len(observados) // 3)]] = True
            # Si la columna no tiene otros nulos, se simula solo ese bloque.
            objetivo = objetivo + int(ocultar.sum()) if rachas else 0
        for _ in range(20 * objetivo):
            if ocultar.sum() >= objetivo:
                break
            inicio = rng.choice(observados)
            ocultar[inicio:inicio + rng.choice(largos)] = True
        ocultar &= s.notna().to_numpy()
        mascara = pd.Series(ocultar, index=s.index)
        s_oculta = s.mask(mascara)
        verdad = s[mascara]
        q1, q3 = s.quantile([0.25, 0.75])
        escala = (q3 - q1) or s.std() or 1.0
        est = _estadisticos(s_oculta)
        return {m: float((METODOS_IMPUTACION[m](s_oculta, est)[mascara] - verdad).abs().mean() / escala)
                for m in self.cfg.metodos_imputacion}

    def transform(self, X):
        X = self._validar_entrada(X).copy()
        crudo = X.copy()
        for c, denominador in self.denominador_.items():
            sin_prod = crudo[c].isna() & ~(crudo[denominador] > 0)
            X[c] = X[c].mask(sin_prod, 0.0)
        indicadores = {c + SUFIJO_INDICADOR: X[c].isna().astype(int) for c in self.con_indicador_}
        for c, metodo in self.metodo_.items():
            if X[c].isna().any():
                X[c] = METODOS_IMPUTACION[metodo](X[c], self.estadisticos_[c])
        return pd.concat([X, pd.DataFrame(indicadores, index=X.index)], axis=1)[self.columnas_salida_]


# ===========================================================================
# Análisis 6. Anomalías multivariadas con LOF (no modifica X)
# ===========================================================================
class AnalisisLOF:
    """Local Outlier Factor sobre las predictoras ya imputadas.

    * Estandarización y PCA ajustados en Train (LOF sufre con cientos de columnas).
      Se usa StandardScaler y no RobustScaler: las variables de producción con
      casi todos ceros tienen IQR ~0 y el escalado robusto las infla a z de miles,
      lo que degenera el LOF. Los extremos ya fueron winsorizados en el paso 5.
    * Búsqueda de `n_neighbors`: se elige el valor cuyo conjunto de días anómalos
      es más estable (Jaccard medio con el resto de la grilla).
    * Train se puntúa con el LOF de entrenamiento; Validation/Test con
      `novelty=True` (comparados contra Train, sin reajustar).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _matriz(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.escalador_.transform(X[self.columnas_])
        return self.pca_.transform(Z) if self.pca_ is not None else Z

    def fit(self, X_train: pd.DataFrame) -> "AnalisisLOF":
        self.columnas_ = [c for c in X_train.columns if not c.endswith(SUFIJO_INDICADOR)]
        self.escalador_ = StandardScaler().fit(X_train[self.columnas_])
        self.pca_ = None
        if self.cfg.lof_varianza_pca:
            self.pca_ = PCA(n_components=self.cfg.lof_varianza_pca, svd_solver="full",
                            random_state=SEMILLA).fit(self.escalador_.transform(X_train[self.columnas_]))
        Z = self._matriz(X_train)

        marcados = {}
        for k in self.cfg.lof_vecinos:
            lof = LocalOutlierFactor(n_neighbors=k).fit(Z)
            marcados[k] = set(X_train.index[-lof.negative_outlier_factor_ > self.cfg.lof_umbral])

        def jaccard(a, b):
            return len(a & b) / len(a | b) if (a | b) else 1.0

        sensibilidad = []
        for k in self.cfg.lof_vecinos:
            otros = [jaccard(marcados[k], marcados[j]) for j in self.cfg.lof_vecinos if j != k]
            sensibilidad.append({"n_neighbors": k, "dias_anomalos_train": len(marcados[k]),
                                 "jaccard_medio": float(np.mean(otros)) if otros else 1.0})
        self.sensibilidad_ = pd.DataFrame(sensibilidad)
        self.k_ = int(self.sensibilidad_.sort_values(
            ["jaccard_medio", "n_neighbors"], ascending=[False, True]).iloc[0]["n_neighbors"])
        self.modelo_ = LocalOutlierFactor(n_neighbors=self.k_, novelty=True).fit(Z)
        self.indice_train_ = X_train.index
        self.puntaje_train_ = pd.Series(-self.modelo_.negative_outlier_factor_, index=X_train.index)
        return self

    def puntuar(self, X: pd.DataFrame) -> pd.DataFrame:
        """Puntaje LOF y marca por día. Las fechas de Train usan el puntaje de ajuste."""
        puntaje = pd.Series(-self.modelo_.score_samples(self._matriz(X)), index=X.index)
        en_train = X.index.isin(self.indice_train_)
        puntaje[en_train] = self.puntaje_train_.reindex(X.index[en_train])
        # Variables que más se apartan ese día (|z| respecto de Train), para auditar el motivo.
        z = pd.DataFrame(self.escalador_.transform(X[self.columnas_]),
                         index=X.index, columns=self.columnas_).abs()
        top = z.apply(lambda fila: ", ".join(fila.nlargest(5).index), axis=1)
        return pd.DataFrame({"lof_score": puntaje, "anomalo_lof": puntaje > self.cfg.lof_umbral,
                             "variables_mas_atipicas": top})


# ===========================================================================
# Orquestación
# ===========================================================================
def construir_pasos(cfg: Config, decisiones: Optional[Dict[str, object]] = None
                    ) -> List[Tuple[str, PasoAuditable]]:
    """Secuencia de pasos. Reordenar, quitar o agregar líneas libremente:
    cada paso solo necesita las columnas que le deja el anterior.

    `decisiones` son las de la EDA; si no se pasan, se leen de results/eda/."""
    decisiones = decisiones if decisiones is not None else eda.cargar_decisiones()
    return [
        ("aplicar_decisiones_eda", AplicarDecisionesEDA(decisiones)),
        ("descartar_no_informativas", DescartarNoInformativas(cfg)),
        ("descartar_duplicadas", DescartarDuplicadas(cfg)),
        ("filtrar_por_nulos", FiltrarPorNulos(cfg)),
        ("outliers_univariados", TratarOutliersUnivariados(cfg)),
        ("imputar_nulos", ImputarNulos(cfg)),
    ]


def cargar_datos_diarios(input_csv: Path, checksums_path: Path, cfg: Config):
    """Cierres diarios, target D+1 y fechas de corte (reutiliza la Parte F del pipeline)."""
    with open(checksums_path, encoding="utf-8") as f:
        registrado = lh._sha_registrado(json.load(f), "data/processed/dataset_trabajo.csv")
    sha = huella(input_csv)
    if registrado is None or sha != registrado:
        raise ValueError("dataset_trabajo.csv no coincide con el SHA-256 registrado en checksums.json.")

    horario, _ = lh.load_cleaned_data(input_csv)
    diario, _ = lh.aggregate_daily_data(horario)
    etiquetadas, _ = lh.prepare_target(diario)
    *_, info = lh.temporal_split(etiquetadas[[COL_FECHA, COL_FRIO, COL_Y]], *cfg.proporciones)
    cortes = {p: (pd.Timestamp(info[p]["date_start"]), pd.Timestamp(info[p]["date_end"]))
              for p in ("train", "validation", "test")}

    X = diario.set_index(COL_FECHA).sort_index().apply(pd.to_numeric, errors="coerce")
    y = etiquetadas.set_index(COL_FECHA)[COL_Y]
    return X, y, cortes, sha


def verificar_causalidad(pipeline: Pipeline, X: pd.DataFrame, cortes: List[pd.Timestamp]) -> pd.DataFrame:
    """Comprueba que transformar la historia hasta t da lo mismo que transformar todo y
    recortar en t. Si difiere, algún paso está usando información futura."""
    completo = pipeline.transform(X)
    filas = []
    for corte in cortes:
        parcial = pipeline.transform(X.loc[:corte])
        referencia = completo.loc[parcial.index, parcial.columns]
        dif = float(np.nanmax(np.abs(parcial.to_numpy(float) - referencia.to_numpy(float))))
        filas.append({"corte": str(corte.date()), "filas": len(parcial),
                      "max_diferencia": dif, "causal": bool(dif < 1e-9)})
    return pd.DataFrame(filas)


def impacto_por_paso(pipeline: Pipeline, X: pd.DataFrame, particion: pd.Series) -> pd.DataFrame:
    """Cuántos valores modificó cada paso en cada partición (sobre las columnas que conserva).

    Sirve para detectar tratamientos que alteran mucho Validation/Test, por ejemplo
    una winsorización con límites de Train que recorta un cambio de régimen real.
    """
    filas, actual = [], X
    for nombre, paso in pipeline.steps:
        nuevo = paso.transform(actual)
        comunes = [c for c in nuevo.columns if c in actual.columns]
        antes, despues = actual[comunes], nuevo[comunes]
        cambiado = ~((antes == despues) | (antes.isna() & despues.isna()))
        for part, idx in particion.groupby(particion).groups.items():
            por_col = cambiado.loc[idx].sum()
            filas.append({
                "paso": nombre, "particion": part,
                "valores_modificados": int(por_col.sum()),
                "pct_valores": float(por_col.sum() / max(cambiado.loc[idx].size, 1)),
                "columnas_afectadas": int((por_col > 0).sum()),
                "top_columnas": "; ".join(f"{c} ({n})" for c, n in
                                          por_col[por_col > 0].nlargest(5).items()),
            })
        actual = nuevo
    return pd.DataFrame(filas)


def _particion(indice: pd.DatetimeIndex, y: pd.Series, cortes) -> pd.Series:
    particion = pd.Series("sin_etiqueta", index=indice)
    etiquetado = indice.isin(y.index)
    for nombre, (ini, fin) in cortes.items():
        particion[etiquetado & (indice >= ini) & (indice <= fin)] = nombre
    return particion


def ejecutar(
    cfg: Optional[Config] = None,
    omitir: Tuple[str, ...] = (),
    input_csv: Path = lh.DEFAULT_INPUT_CSV,
    checksums_path: Path = lh.DEFAULT_CHECKSUMS_JSON,
    salida_datos: Path = DATA_DIR / "processed",
    salida_auditoria: Path = RESULTS_DIR / "preprocesamiento",
) -> Dict[str, object]:
    cfg = cfg or Config()
    X, y, cortes, sha_entrada = cargar_datos_diarios(Path(input_csv), Path(checksums_path), cfg)
    fin_train = cortes["train"][1]

    decisiones = eda.cargar_decisiones(checksums_path=Path(checksums_path))
    todos = construir_pasos(cfg, decisiones)
    pasos = [(n, p) for n, p in todos if n not in omitir]
    desconocidos = set(omitir) - {n for n, _ in todos}
    if desconocidos:
        raise ValueError(f"Pasos inexistentes en --omitir: {sorted(desconocidos)}")
    pipeline = Pipeline(pasos)

    # Ajuste SOLO con fechas de Train (incluye días sin etiqueta anteriores al corte).
    X_ajuste = X.loc[:fin_train]
    pipeline.fit(X_ajuste, y.reindex(X_ajuste.index))
    X_proc = pipeline.transform(X)

    causalidad = verificar_causalidad(
        pipeline, X, [fin_train, cortes["validation"][1], cortes["test"][0] + pd.Timedelta(days=30)])
    if not causalidad["causal"].all():
        raise ValueError(f"Algún paso usa información futura:\n{causalidad}")

    particion = _particion(X_proc.index, y, cortes)
    impacto = impacto_por_paso(pipeline, X, particion)

    # LOF necesita la matriz completa; si se omitió la imputación, se saltea.
    columnas_con_nulos = X_proc.columns[X_proc.isna().any()].tolist()
    lof = None
    if columnas_con_nulos:
        print(f"AVISO: {len(columnas_con_nulos)} columnas conservan nulos; se omite el análisis LOF.")
        anomalias = pd.DataFrame({"lof_score": np.nan, "anomalo_lof": False,
                                  "variables_mas_atipicas": ""}, index=X_proc.index)
    else:
        lof = AnalisisLOF(cfg).fit(X_proc.loc[:fin_train])
        anomalias = lof.puntuar(X_proc)

    datos = X_proc.copy().join(y).assign(particion=particion, **anomalias[["lof_score", "anomalo_lof"]])
    excluidos = []
    if lof is not None and cfg.tratamiento_lof == "excluir_train":
        excluidos = datos.index[(datos["particion"] == "train") & datos["anomalo_lof"]].tolist()

    # ---------------- Exportación ----------------
    salida_datos, salida_auditoria = Path(salida_datos), Path(salida_auditoria)
    salida_datos.mkdir(parents=True, exist_ok=True)
    salida_auditoria.mkdir(parents=True, exist_ok=True)

    def guardar(df: pd.DataFrame, ruta: Path, index: bool = False) -> Path:
        df.to_csv(ruta, index=index, encoding="utf-8", lineterminator="\n", date_format="%Y-%m-%d")
        return ruta

    exportados = {"historial_completo.csv": guardar(datos, salida_datos / "historial_completo.csv", True)}
    for nombre in ("train", "validation", "test"):
        parte = datos[datos["particion"] == nombre].drop(columns=["particion"])
        if nombre == "train":
            parte = parte.drop(index=excluidos)
        exportados[f"{nombre}.csv"] = guardar(parte, salida_datos / f"{nombre}.csv", True)

    # Los reportes por paso se numeran como los pasos (00 = decisiones de la EDA). Se borran
    # los de una corrida anterior para que no queden reportes de pasos que ya no existen.
    for viejo in salida_auditoria.glob("[0-9][0-9]_*.csv"):
        viejo.unlink()
    resumen_pasos = {}
    for i, (nombre, paso) in enumerate(pipeline.steps):
        guardar(paso.reporte_, salida_auditoria / f"{i:02d}_{nombre}.csv")
        for tabla, df in paso.tablas_.items():
            guardar(df, salida_auditoria / f"{i:02d}_{nombre}__{tabla}.csv")
        resumen_pasos[nombre] = paso.resumen()
    guardar(causalidad, salida_auditoria / "verificacion_causalidad.csv")
    guardar(impacto, salida_auditoria / "impacto_por_paso.csv")
    if lof is not None:
        guardar(lof.sensibilidad_, salida_auditoria / "lof_sensibilidad.csv")
    guardar(anomalias.assign(particion=particion), salida_auditoria / "lof_dias.csv", True)

    resumen = {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "src/preprocessing_pipeline.py",
        "semilla": SEMILLA,
        "config": asdict(cfg),
        "pasos_omitidos": list(omitir),
        "entrada": {"archivo": str(Path(input_csv).relative_to(PROJECT_ROOT).as_posix()),
                    "sha256": sha_entrada, "dias": len(X), "columnas": X.shape[1]},
        "decisiones_eda": {"archivo": "results/eda/resumen.json",
                           "generado_utc": decisiones["generado_utc"],
                           "sha256_decisiones": decisiones["sha256_decisiones"],
                           "columnas_descartadas": len(decisiones["descartar"]),
                           "columnas_unificadas": len(decisiones["unificar"])},
        "cortes": {p: [str(a.date()), str(b.date())] for p, (a, b) in cortes.items()},
        "fin_ajuste": str(fin_train.date()),
        "pasos": resumen_pasos,
        "causalidad_ok": bool(causalidad["causal"].all()),
        "columnas_con_nulos_al_final": columnas_con_nulos,
        "lof": {
            "ejecutado": lof is not None,
            "n_neighbors_elegido": lof.k_ if lof is not None else None,
            "componentes_pca": (int(lof.pca_.n_components_)
                                if lof is not None and lof.pca_ is not None else None),
            "dias_anomalos": anomalias.assign(particion=particion)
                                     .groupby("particion")["anomalo_lof"].sum().astype(int).to_dict(),
            "tratamiento": cfg.tratamiento_lof,
            "dias_excluidos_de_train": [str(d.date()) for d in excluidos],
        },
        "salida": {"columnas_predictoras": X_proc.shape[1],
                   "sha256": {n: huella(r) for n, r in exportados.items()}},
    }
    with open(salida_auditoria / "resumen.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")

    _imprimir_resumen(resumen, pipeline)
    return {"datos": datos, "pipeline": pipeline, "lof": lof, "resumen": resumen}


def _imprimir_resumen(resumen: Dict[str, object], pipeline: Pipeline) -> None:
    print("=" * 78)
    print("PREPROCESAMIENTO (previo a la selección de variables)")
    print("=" * 78)
    print(f"Entrada        : {resumen['entrada']['dias']} días x {resumen['entrada']['columnas']} columnas")
    print(f"Ajuste hasta   : {resumen['fin_ajuste']} (solo Train)")
    for nombre, paso in pipeline.steps:
        r = paso.resumen()
        print(f"  {nombre:<27}: {r['columnas_entrada']:>4} -> {r['columnas_salida']:>4}  "
              f"{r.get('decisiones', '')}")
    print(f"Causalidad     : {'OK' if resumen['causalidad_ok'] else 'FALLA'}")
    lof = resumen["lof"]
    print(f"LOF            : k={lof['n_neighbors_elegido']}, PCA={lof['componentes_pca']} comp., "
          f"anómalos {lof['dias_anomalos']}")
    print(f"Predictoras    : {resumen['salida']['columnas_predictoras']}")
    print("=" * 78)


def main() -> None:
    nombres = [n for n, _ in construir_pasos(Config())]
    parser = argparse.ArgumentParser(description="Preprocesamiento previo a la selección de variables.")
    parser.add_argument("--omitir", nargs="*", default=[], choices=nombres,
                        help="Pasos a omitir.")
    parser.add_argument("--metodo-outliers", choices=list(DETECTORES_UNIVARIADOS), default="iqr")
    parser.add_argument("--tratamiento-outliers", choices=["winsorizar", "nan", "ninguno"],
                        default="winsorizar")
    parser.add_argument("--max-nulos-train", type=float, default=0.30)
    parser.add_argument("--tratamiento-lof", choices=["reportar", "excluir_train"], default="reportar")
    parser.add_argument("--completo", action="store_true",
                        help="Corre antes la limpieza horaria de los Excel (verificando sus SHA-256) y la EDA.")
    args = parser.parse_args()
    if args.completo:
        lh.run_cleaning_pipeline()
        eda.ejecutar()
    cfg = Config(metodo_outliers=args.metodo_outliers,
                 tratamiento_outliers=args.tratamiento_outliers,
                 max_nulos_train=args.max_nulos_train,
                 tratamiento_lof=args.tratamiento_lof)
    ejecutar(cfg, omitir=tuple(args.omitir))


if __name__ == "__main__":
    main()
