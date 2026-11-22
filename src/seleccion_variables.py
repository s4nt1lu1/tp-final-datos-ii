"""Selección de variables, a partir del preprocesamiento.

Etapas:
    1. Candidatas: predictoras de `data/processed/historial_completo.csv`
       + rezagos, medias móviles y diferencias del frío + calendario
       + variables creadas:
         * ciclo anual : seno y coseno de la fracción del año (armónicos 1 y 2).
         * binning     : intervalos por cuantiles de Train, codificados como indicadores.
         * interacción : productos de a pares de variables principales estandarizadas,
                         y de cada una con el ciclo anual.
         * ratios      : cocientes entre consumos de distintas áreas y por hectolitro.
         * eficiencia  : consumo por hectolitro real dividido por la meta (hoja Consolidado KPI).
         * maquinaria  : (desactivada por defecto) qué compresores funcionan cada día. La EDA
                         excluye los compresores porque son componentes del frío.
    2. Filtro de redundancia (ajustado con Train): de cada grupo de candidatas con
       |Spearman| >= `umbral_redundancia` entre sí queda una sola, la más relacionada con y.
    3. Métodos independientes, todos ajustados SOLO con Train:
         * mutual_information : ranking por información mutua con y.
         * ridge              : ranking por |coeficiente estandarizado| de Ridge.
         * lasso              : columnas con coeficiente no nulo en LassoCV.
         * forward            : selección hacia adelante con Ridge.
       Los métodos de ranking eligen cuántas columnas conservar con una curva
       k -> MAE en validación temporal interna (TimeSeriesSplit sobre Train) y la
       regla de "un error estándar": el k más chico cuyo MAE no supera el mínimo
       más un error estándar.
    4. Comparación de subconjuntos en Validation con estimadores fijos.
       Test no se lee.

Reglas contra la fuga de información:
    * Las variables derivadas del frío usan solo días <= D (shift/rolling hacia atrás).
    * Los cortes del binning, la media y el desvío de las interacciones, los topes de los
      ratios y los umbrales de "equipo funcionando" se estiman solo con fechas de Train;
      la fila del día D usa solo valores de D y de días anteriores.
    * La validación interna usa TimeSeriesSplit con `gap=1`: el último y de cada
      bloque de ajuste es el frío del día siguiente, que es predictora del primer día
      evaluado; el hueco evita ese solapamiento.
    * El escalado vive dentro de cada Pipeline y se ajusta en cada bloque de ajuste.

Para agregar o quitar un método, editar `METODOS`.

Uso desde la raíz del proyecto:
    python -m src.seleccion_variables
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.auxiliar_functions import DATA_DIR, RESULTS_DIR, SEMILLA, huella

COL_FRIO = "Frio (Kw)_Consolidado EE"
COL_BODEGA = "Hl Producido Bodega_Consolidado Produccion"
COL_ENVASADA = "Hl Cerveza Envasada_Consolidado Produccion"
COL_MOSTO = "Hl de Mosto_Consolidado Produccion"  # entrada a fermentación
COL_Y = "y"
COLUMNAS_NO_PREDICTORAS = ("y", "particion", "lof_score", "anomalo_lof")
COLUMNAS_CALENDARIO = ("mes", "dia_semana_sin", "dia_semana_cos", "fin_de_semana", "mes_sin", "mes_cos")
# Prefijo del nombre -> familia. Los nombres los genera este módulo; ninguna
# columna del preprocesamiento empieza con estos prefijos.
PREFIJOS_FAMILIA = (
    ("frio_", "derivada del frío"),
    ("anio_", "ciclo anual"),
    ("bin_", "binning"),
    ("int_", "interacción"),
    ("maq_", "maquinaria"),
    ("ratio_", "ratio entre áreas"),
    ("sin_produccion_", "ratio entre áreas"),
    ("efic_", "eficiencia vs meta"),
)
CARPETA_PREPROCESADO = DATA_DIR / "processed"
RESUMEN_PREPROCESADO = RESULTS_DIR / "preprocesamiento" / "resumen.json"


@dataclass
class ConfigSeleccion:
    # Variables derivadas del frío (en días calendario, siempre hacia atrás)
    rezagos_frio: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 14)
    ventanas_media_frio: Tuple[int, ...] = (2, 3, 5, 7, 10, 14, 21, 28)
    ventanas_mediana_frio: Tuple[int, ...] = (3, 7, 14)
    ventanas_desvio_frio: Tuple[int, ...] = (3, 7, 14)
    ventanas_extremos_frio: Tuple[int, ...] = (3, 7, 14)  # mínimo y máximo de la ventana
    spans_media_exponencial_frio: Tuple[int, ...] = (3, 7, 14)
    diferencias_frio: Tuple[int, ...] = (1, 2, 3, 7)  # frío de D menos frío de D-k
    max_arrastre_dias: int = 3  # días sin cierre que un rezago puede cubrir con el cierre anterior
    incluir_calendario: bool = True

    # Variables circulares anuales: seno y coseno de 2*pi*k*(fracción del año).
    # k = 1 es el ciclo anual; k = 2 (semestral) permite estaciones asimétricas.
    # () = no se crean.
    armonicos_anuales: Tuple[int, ...] = (1, 2)

    # Binning: (alias, columna, intervalos). Los cortes son cuantiles de Train y cada
    # intervalo es un indicador 0/1. Si más de `frac_ceros_intervalo_propio` de Train
    # son ceros (días sin producción), el cero es un intervalo aparte y los cuantiles
    # se calculan sobre los valores positivos. () = sin binning.
    binning: Tuple[Tuple[str, str, int], ...] = (
        ("frio", COL_FRIO, 5),
        ("bodega", COL_BODEGA, 4),
        ("envasada", COL_ENVASADA, 4),
        ("mosto", COL_MOSTO, 4),
    )
    frac_ceros_intervalo_propio: float = 0.05

    # Interacciones: producto de cada par de estas variables, estandarizadas con la
    # media y el desvío de Train (centrar reduce la colinealidad con los efectos
    # principales). Si `interacciones_con_ciclo_anual`, también cada una por el seno y
    # el coseno anual (k = 1): con los dos, el modelo elige la fase. () = sin interacciones.
    variables_interaccion: Tuple[Tuple[str, str], ...] = (
        ("frio", COL_FRIO),
        ("bodega", COL_BODEGA),
        ("envasada", COL_ENVASADA),
        ("mosto", COL_MOSTO),
    )
    interacciones_con_ciclo_anual: bool = True

    # Maquinaria funcionando: (grupo, patrón de columnas). Un equipo funciona el día D si
    # su consumo de ese día supera `frac_umbral_funcionando` x su percentil 95 de Train.
    # Por grupo se crean: un indicador por equipo, cuántos funcionan, la carga media por
    # equipo encendido, el cambio en la cantidad respecto del día anterior y la media de
    # esa cantidad en los últimos `ventana_maquinaria` cierres. () = no se crean.
    # Por defecto no se crean: la EDA excluye los compresores Mycom y los condensadores
    # porque son componentes del frío (Frio (Kw) es su suma) y su reparto cambia por rotación.
    # Para recrearlos habría que conservar esas columnas en src/eda.py y usar, por ejemplo,
    # (("mycom", r"^KW Mycom \d+_"), ("cond", r"^KW Cond ")).
    patrones_maquinaria: Tuple[Tuple[str, str], ...] = ()
    frac_umbral_funcionando: float = 0.05
    ventana_maquinaria: int = 7

    # Ratios entre áreas (consigna 2.2): (nombre, numerador, denominador), con valores del día D.
    # Si el denominador es <= 0 (día sin producción) el ratio vale 0, y para los denominadores de
    # producción el indicador `sin_produccion_<columna>` marca esos días. No se suma un épsilon al
    # denominador: con producción 0 daría valores del orden de 10^9 que dominan cualquier modelo.
    # Con denominadores chicos el cociente tiene colas extremas: se recorta a los percentiles
    # `percentiles_ratios` de Train (sobre los días con denominador > 0).
    ratios: Tuple[Tuple[str, str, str], ...] = (
        ("ratio_bodega_elaboracion_kw", "Bodega (Kw)_Consolidado EE", "Elaboracion (Kw)_Consolidado EE"),
        ("ratio_frio_elaboracion_kw", COL_FRIO, "Elaboracion (Kw)_Consolidado EE"),
        ("ratio_frio_envasado_kw", COL_FRIO, "Envasado (Kw)_Consolidado EE"),
        ("ratio_frio_por_hl_mosto", COL_FRIO, COL_MOSTO),
        ("ratio_mosto_envasada", COL_MOSTO, COL_ENVASADA),
        ("ratio_agua_bodega_por_hl", "Agua Bodega (Hl)_Consolidado Agua", COL_BODEGA),
        ("ratio_agua_elab_por_hl_mosto", "Agua Elaboracion (Hl)_Consolidado Agua", COL_MOSTO),
        ("ratio_et_elab_por_hl_mosto", "ET Elaboracion (Mj)_Consolidado GasVapor", COL_MOSTO),
    )
    # Eficiencia energética contra la meta (consigna 2.2): consumo por Hl / meta por Hl.
    # Mayor a 1 = el día consumió por hectolitro más de lo que fija la meta.
    eficiencias: Tuple[Tuple[str, str, str], ...] = (
        ("efic_frio_vs_meta", "EE Frio / Hl_Consolidado KPI", "Meta EE Frio_Consolidado KPI"),
        ("efic_planta_vs_meta", "EE Planta / Hl_Consolidado KPI", "Meta EE Planta_Consolidado KPI"),
        ("efic_bodega_vs_meta", "EE Bodega / Hl_Consolidado KPI", "Meta EE Bodega_Consolidado KPI"),
        ("efic_elaboracion_vs_meta", "EE Elaboracion / Hl_Consolidado KPI",
         "Meta EE Elaboracion_Consolidado KPI"),
    )
    percentiles_ratios: Tuple[float, float] = (0.005, 0.995)
    patron_produccion: str = r"^H[lL] "

    # Filtro de redundancia (ajustado con Train). Las candidatas se ordenan por |Spearman| con y y
    # se descarta cada una que tenga |Spearman| >= umbral con otra ya conservada. None = sin filtro.
    # Desactivado: con 0.90 deja 265 de 360 candidatas pero empeora todos los modelos en Validation
    # (results/comparacion_variantes/): quita ventanas del frío útiles y el ciclo anual.
    umbral_redundancia: Optional[float] = None
    proteger_de_redundancia: Tuple[str, ...] = (COL_FRIO,)  # siempre se conservan, y primero

    # Validación interna en Train
    n_splits: int = 5
    gap: int = 1

    # Ridge (para rankings, curvas k y forward)
    alphas_ridge: Tuple[float, ...] = tuple(np.logspace(-2, 4, 25).round(4))

    # Curva k -> MAE para métodos de ranking
    grilla_k: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 80, 120)
    regla_un_error_estandar: bool = True

    # Mutual information
    mi_vecinos: int = 3

    # Lasso
    lasso_alphas: int = 60
    lasso_max_iter: int = 50_000

    # Forward
    forward_max_variables: int = 25
    forward_tolerancia_kw: float = 5.0  # mejora mínima de MAE interno para seguir agregando

    # Comparación en Validation
    estimadores_validacion: Tuple[str, ...] = ("ridge", "random_forest")


# ===========================================================================
# 1. Candidatas
# ===========================================================================
def cargar_preprocesado(carpeta: Path = CARPETA_PREPROCESADO,
                        resumen: Path = RESUMEN_PREPROCESADO) -> pd.DataFrame:
    """Lee el historial preprocesado y verifica su SHA-256 contra el resumen de auditoría."""
    ruta = carpeta / "historial_completo.csv"
    with open(resumen, encoding="utf-8") as f:
        esperado = json.load(f)["salida"]["sha256"]["historial_completo.csv"]
    if huella(ruta) != esperado:
        raise ValueError("historial_completo.csv no coincide con el SHA-256 del preprocesamiento. "
                         "Volver a correr python -m src.preprocessing_pipeline.")
    return pd.read_csv(ruta, index_col=0, parse_dates=True).sort_index()


def variables_frio(frio: pd.Series, cfg: ConfigSeleccion) -> pd.DataFrame:
    """Rezagos, ventanas móviles y diferencias del frío. Solo usan días <= D.

    Familias (cada una configurable en `ConfigSeleccion`):
        frio_lag_k        frío de D-k
        frio_media_v      media de D-v+1 a D
        frio_mediana_v    mediana de la ventana (robusta a un día atípico)
        frio_desvio_v     desvío estándar de la ventana (variabilidad reciente)
        frio_min_v/max_v  extremos de la ventana
        frio_ewm_s        media exponencial con span s (pesa más los días recientes)
        frio_diff_k       frío de D menos frío de D-k (tendencia)

    Hay días calendario sin cierre (los descartó la limpieza):
      * Rezagos y diferencias se miden en días calendario. Un día faltante toma el
        último cierre anterior disponible (hasta `max_arrastre_dias`), que es lo que
        se conocería en ese momento.
      * Las ventanas (medias, medianas, desvíos, extremos y exponenciales) usan los
        últimos `v` cierres observados. Con un día faltante, la ventana de 7 cierres
        abarca 8 días calendario, pero nunca queda vacía.
    """
    frio = frio.sort_index()
    arrastrada = frio.asfreq("D").ffill(limit=cfg.max_arrastre_dias)

    def ventana(v):
        return frio.rolling(v, min_periods=max(2, v // 2))

    nuevas = {}
    for k in cfg.rezagos_frio:
        nuevas[f"frio_lag_{k}"] = arrastrada.shift(k).reindex(frio.index)
    for v in cfg.ventanas_media_frio:
        nuevas[f"frio_media_{v}"] = ventana(v).mean()
    for v in cfg.ventanas_mediana_frio:
        nuevas[f"frio_mediana_{v}"] = ventana(v).median()
    for v in cfg.ventanas_desvio_frio:
        nuevas[f"frio_desvio_{v}"] = ventana(v).std()
    for v in cfg.ventanas_extremos_frio:
        nuevas[f"frio_min_{v}"] = ventana(v).min()
        nuevas[f"frio_max_{v}"] = ventana(v).max()
    for s in cfg.spans_media_exponencial_frio:
        nuevas[f"frio_ewm_{s}"] = frio.ewm(span=s, min_periods=2).mean()
    for k in cfg.diferencias_frio:
        nuevas[f"frio_diff_{k}"] = frio - arrastrada.shift(k).reindex(frio.index)
    return pd.DataFrame(nuevas, index=frio.index)


def variables_calendario(indice: pd.DatetimeIndex) -> pd.DataFrame:
    """Mes (1-12), día de la semana y mes con codificación cíclica, y fin de semana.
    Se conocen en D. `mes` como número sirve a los árboles; seno y coseno, a los
    modelos lineales (diciembre y enero quedan cerca)."""
    dia = indice.day_of_week.to_numpy()
    mes = indice.month.to_numpy()
    return pd.DataFrame({
        "mes": mes,
        "dia_semana_sin": np.sin(2 * np.pi * dia / 7),
        "dia_semana_cos": np.cos(2 * np.pi * dia / 7),
        "fin_de_semana": (dia >= 5).astype(int),
        "mes_sin": np.sin(2 * np.pi * (mes - 1) / 12),
        "mes_cos": np.cos(2 * np.pi * (mes - 1) / 12),
    }, index=indice)


def fraccion_del_anio(indice: pd.DatetimeIndex) -> np.ndarray:
    """Posición del día en su año, en [0, 1): 1 de enero = 0. Contempla años bisiestos."""
    dias_del_anio = np.where(indice.is_leap_year, 366, 365)
    return (indice.day_of_year.to_numpy() - 1) / dias_del_anio


def variables_ciclo_anual(indice: pd.DatetimeIndex, armonicos: Tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    """Variables circulares anuales: seno y coseno de 2*pi*k*t, con t la fracción del año.

    A diferencia de `mes_sin`/`mes_cos` (constantes dentro de cada mes), varían día a
    día y el 31 de diciembre queda pegado al 1 de enero. Con seno y coseno juntos, un
    modelo lineal representa una onda anual con cualquier fase: a*sin + b*cos =
    A*cos(2*pi*t - fase). El armónico 2 agrega una onda semestral, que permite que la
    subida y la bajada de la estación no sean simétricas. Se conocen en D.
    """
    t = fraccion_del_anio(indice)
    nuevas = {}
    for k in armonicos:
        nuevas[f"anio_sin_{k}"] = np.sin(2 * np.pi * k * t)
        nuevas[f"anio_cos_{k}"] = np.cos(2 * np.pi * k * t)
    return pd.DataFrame(nuevas, index=indice)


class VariablesCreadas(BaseEstimator, TransformerMixin):
    """Binning, interacciones, ratios, eficiencias y maquinaria, con parámetros aprendidos en Train.

    `fit` recibe las candidatas de las fechas de Train y guarda:
      * `cortes_`: para cada variable del binning, si el cero es un intervalo propio
        y los cuantiles que separan los intervalos.
      * `escala_`: media y desvío de cada variable de las interacciones.
      * `topes_`: para cada ratio y eficiencia, los percentiles de Train a los que se recorta.
      * `umbrales_`: para cada grupo de maquinaria, el consumo diario a partir del
        cual se considera que cada equipo funcionó.
    `transform` es causal: la fila del día D usa solo los valores de D (y, para el
    cambio y la media de equipos funcionando, de días anteriores) y lo aprendido en
    `fit`. `reporte_` describe cada variable creada.
    """

    def __init__(self, cfg: ConfigSeleccion):
        self.cfg = cfg

    def fit(self, X: pd.DataFrame, y=None):
        cfg, filas = self.cfg, []
        self.cortes_ = {}
        for alias, columna, n in cfg.binning:
            v = X[columna].dropna()
            cero_propio = bool((v <= 0).mean() > cfg.frac_ceros_intervalo_propio)
            valores = v[v > 0] if cero_propio else v
            n_cuantiles = n - 1 if cero_propio else n
            cortes = np.unique(np.quantile(valores, np.linspace(0, 1, n_cuantiles + 1)[1:-1]))
            self.cortes_[alias] = (cero_propio, cortes)
            limites = np.r_[-np.inf, cortes, np.inf]
            if cero_propio:
                filas.append({"variable": f"bin_{alias}_cero", "familia": "binning", "origen": columna,
                              "definicion": f"{alias} <= 0 (sin actividad)",
                              "pct_dias_train": 100 * float((v <= 0).mean())})
            for i in range(len(cortes) + 1):
                inf, sup = limites[i], limites[i + 1]
                dentro = (valores >= inf) & (valores < sup)
                texto = (f"{alias} < {sup:,.0f}" if i == 0 else f"{alias} >= {inf:,.0f}"
                         if i == len(cortes) else f"{inf:,.0f} <= {alias} < {sup:,.0f}")
                filas.append({"variable": f"bin_{alias}_q{i + 1}", "familia": "binning", "origen": columna,
                              "definicion": texto + (" (y > 0)" if cero_propio else ""),
                              "pct_dias_train": 100 * float(dentro.sum() / len(v))})

        self.escala_ = {}
        for alias, columna in cfg.variables_interaccion:
            desvio = float(X[columna].std())
            self.escala_[alias] = (columna, float(X[columna].mean()), desvio if desvio > 0 else 1.0)
        for a, b in self._pares():
            filas.append({"variable": f"int_{a}_x_{b}", "familia": "interacción",
                          "origen": " × ".join(self._origen(n) for n in (a, b)),
                          "definicion": f"z({a}) × " + (b if b.startswith("anio_") else f"z({b})"),
                          "pct_dias_train": np.nan})

        self.topes_ = {}
        cocientes = list(cfg.ratios) + list(cfg.eficiencias)
        faltan = sorted({c for _, num, den in cocientes for c in (num, den)} - set(X.columns))
        if faltan:
            raise ValueError(f"Ratios/eficiencias: faltan columnas en las candidatas: {faltan}")
        for nombre, num, den in cocientes:
            valido = X[den] > 0
            crudo = X.loc[valido, num] / X.loc[valido, den]
            inf, sup = crudo.quantile(list(cfg.percentiles_ratios))
            self.topes_[nombre] = (num, den, float(inf), float(sup))
            es_eficiencia = nombre in {n for n, _, _ in cfg.eficiencias}
            filas.append({"variable": nombre,
                          "familia": "eficiencia vs meta" if es_eficiencia else "ratio entre áreas",
                          "origen": f"{num} / {den}",
                          "definicion": f"{num.split('_')[0]} / {den.split('_')[0]} (0 si el denominador <= 0; "
                                        f"recortado a [{inf:,.3g}; {sup:,.3g}])",
                          "pct_dias_train": 100 * float(valido.mean())})
        self.indicadores_produccion_ = {}
        for den in dict.fromkeys(den for _, _, den in cfg.ratios):
            if re.search(cfg.patron_produccion, den):
                alias = re.sub(r"[^a-z0-9]+", "_", den.split("_")[0].lower()).strip("_")
                self.indicadores_produccion_[f"sin_produccion_{alias}"] = den
                filas.append({"variable": f"sin_produccion_{alias}", "familia": "ratio entre áreas",
                              "origen": den, "definicion": f"1 si {den.split('_')[0]} <= 0 (día sin producción)",
                              "pct_dias_train": 100 * float((X[den] <= 0).mean())})

        self.umbrales_ = {}
        for grupo, patron in cfg.patrones_maquinaria:
            columnas = [c for c in X.columns if re.search(patron, c)]
            if not columnas:
                raise ValueError(f"Maquinaria '{grupo}': ninguna columna coincide con {patron!r}.")
            self.umbrales_[grupo] = {c: cfg.frac_umbral_funcionando * float(X[c].quantile(0.95))
                                     for c in columnas}
            for i, (columna, umbral) in enumerate(self.umbrales_[grupo].items(), 1):
                filas.append({"variable": f"maq_{grupo}_{i}", "familia": "maquinaria", "origen": columna,
                              "definicion": f"1 si {columna.split('_')[0]} > {umbral:,.0f} "
                                            f"({cfg.frac_umbral_funcionando:.0%} de su P95 en Train)",
                              "pct_dias_train": 100 * float((X[columna] > umbral).mean())})
            todas = " + ".join(c.split("_")[0] for c in columnas)
            v = cfg.ventana_maquinaria
            for variable, definicion in [
                (f"maq_n_{grupo}", f"cantidad de equipos funcionando (0 a {len(columnas)})"),
                (f"maq_carga_{grupo}", "consumo total del grupo / equipos funcionando (0 si ninguno)"),
                (f"maq_delta_n_{grupo}", "equipos funcionando hoy − el día anterior (arranques y paradas)"),
                (f"maq_n_{grupo}_media_{v}", f"media de equipos funcionando en los últimos {v} cierres"),
            ]:
                filas.append({"variable": variable, "familia": "maquinaria", "origen": todas,
                              "definicion": definicion, "pct_dias_train": np.nan})
        self.reporte_ = pd.DataFrame(filas)
        self.columnas_ = self.reporte_["variable"].tolist()
        return self

    def _pares(self) -> List[Tuple[str, str]]:
        alias = [a for a, _ in self.cfg.variables_interaccion]
        pares = list(combinations(alias, 2))
        if self.cfg.interacciones_con_ciclo_anual:
            pares += [(a, f"anio_{f}_1") for a in alias for f in ("sin", "cos")]
        return pares

    def _origen(self, nombre: str) -> str:
        return self.escala_[nombre][0] if nombre in self.escala_ else nombre

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        nuevas = {}
        for alias, columna, _ in self.cfg.binning:
            cero_propio, cortes = self.cortes_[alias]
            v = X[columna].to_numpy(float)
            falta = np.isnan(v)
            intervalo = np.digitize(v, cortes)  # 0 .. len(cortes)
            if cero_propio:
                nuevas[f"bin_{alias}_cero"] = np.where(falta, np.nan, (v <= 0).astype(float))
            for i in range(len(cortes) + 1):
                en_i = (intervalo == i) & ~(cero_propio & (v <= 0))
                nuevas[f"bin_{alias}_q{i + 1}"] = np.where(falta, np.nan, en_i.astype(float))

        z = {alias: (X[col] - media) / desvio for alias, (col, media, desvio) in self.escala_.items()}
        ciclo = variables_ciclo_anual(X.index, (1,))
        for a, b in self._pares():
            nuevas[f"int_{a}_x_{b}"] = z[a] * (ciclo[b] if b in ciclo else z[b])

        for nombre, (num, den, inf, sup) in self.topes_.items():
            d = X[den]
            nuevas[nombre] = (X[num] / d.where(d > 0)).clip(inf, sup).where(d > 0, 0.0)
        for nombre, den in self.indicadores_produccion_.items():
            nuevas[nombre] = (X[den] <= 0).astype(float).where(X[den].notna())

        v = self.cfg.ventana_maquinaria
        for grupo, umbrales in self.umbrales_.items():
            funciona = pd.DataFrame({c: (X[c] > u).astype(float) for c, u in umbrales.items()})
            for i, columna in enumerate(umbrales, 1):
                nuevas[f"maq_{grupo}_{i}"] = funciona[columna]
            n = funciona.sum(axis=1)
            nuevas[f"maq_n_{grupo}"] = n
            total = X[list(umbrales)].clip(lower=0).sum(axis=1)
            nuevas[f"maq_carga_{grupo}"] = (total / n.where(n > 0)).fillna(0.0)
            # Igual que los rezagos del frío: un día sin cierre toma el cierre anterior.
            anterior = n.asfreq("D").ffill(limit=self.cfg.max_arrastre_dias).shift(1).reindex(n.index)
            nuevas[f"maq_delta_n_{grupo}"] = n - anterior
            nuevas[f"maq_n_{grupo}_media_{v}"] = n.rolling(v, min_periods=max(2, v // 2)).mean()
        return pd.DataFrame(nuevas, index=X.index)[self.columnas_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columnas_, dtype=object)


def fin_de_ajuste(historial: pd.DataFrame) -> pd.Timestamp:
    """Último día de Train: todo parámetro de las variables creadas se estima hasta acá."""
    return historial.index[historial["particion"] == "train"].max()


def candidatas_directas(historial: pd.DataFrame, cfg: ConfigSeleccion) -> pd.DataFrame:
    """Predictoras del preprocesamiento + derivadas del frío + calendario + ciclo anual.
    Ninguna necesita parámetros estimados con datos."""
    base = historial.drop(columns=[c for c in COLUMNAS_NO_PREDICTORAS if c in historial])
    partes = [base, variables_frio(historial[COL_FRIO], cfg)]
    if cfg.incluir_calendario:
        partes.append(variables_calendario(historial.index))
    if cfg.armonicos_anuales:
        partes.append(variables_ciclo_anual(historial.index, cfg.armonicos_anuales))
    return pd.concat(partes, axis=1)


def construir_candidatas_y_ajuste(historial: pd.DataFrame, cfg: ConfigSeleccion
                                  ) -> Tuple[pd.DataFrame, Optional[VariablesCreadas]]:
    """Como `construir_candidatas`, y además devuelve el `VariablesCreadas` ajustado
    (None si no hay variables creadas) para inspeccionar cortes, escalas y topes."""
    directas = candidatas_directas(historial, cfg)
    if not (cfg.binning or cfg.variables_interaccion or cfg.patrones_maquinaria
            or cfg.ratios or cfg.eficiencias):
        return directas, None
    creadas = VariablesCreadas(cfg).fit(directas.loc[:fin_de_ajuste(historial)])
    return pd.concat([directas, creadas.transform(directas)], axis=1), creadas


def construir_candidatas(historial: pd.DataFrame, cfg: ConfigSeleccion) -> pd.DataFrame:
    """Matriz de candidatas para todo el historial (incluye días sin etiqueta).
    El binning y las interacciones se ajustan con las fechas de Train."""
    return construir_candidatas_y_ajuste(historial, cfg)[0]


def familia_de(columna: str) -> str:
    """Familia de una candidata, para tablas y resúmenes."""
    for prefijo, familia in PREFIJOS_FAMILIA:
        if columna.startswith(prefijo):
            return familia
    return "calendario" if columna in COLUMNAS_CALENDARIO else "preprocesamiento"


def filtrar_redundantes(X_tr: pd.DataFrame, y_tr: pd.Series, cfg: ConfigSeleccion
                        ) -> Tuple[List[str], pd.DataFrame]:
    """Columnas que pasan el filtro de redundancia (en el orden original) y reporte por columna.

    Solo usa Train. Orden de preferencia: las protegidas, después mayor |Spearman| con y. Una
    candidata se descarta si su |Spearman| con alguna ya conservada es >= `umbral_redundancia`.
    Las constantes en Train se descartan: no tienen variación para aprender.
    """
    if cfg.umbral_redundancia is None:
        return list(X_tr.columns), pd.DataFrame(columns=["columna", "decision"])
    constantes = set(X_tr.columns[X_tr.nunique(dropna=True) <= 1])
    rho_y = X_tr.corrwith(y_tr, method="spearman").abs()
    posicion = {c: i for i, c in enumerate(X_tr.columns)}
    orden = sorted([c for c in X_tr.columns if c not in constantes],
                   key=lambda c: (c not in cfg.proteger_de_redundancia, -rho_y.get(c, 0), posicion[c]))
    corr = X_tr[orden].rank().corr().abs()  # Spearman = Pearson sobre rangos
    conservadas, filas = [], []
    for c in orden:
        representante = None
        if c not in cfg.proteger_de_redundancia:
            representante = next((k for k in conservadas if corr.at[c, k] >= cfg.umbral_redundancia), None)
        if representante is None:
            conservadas.append(c)
        filas.append({"columna": c, "familia": familia_de(c),
                      "decision": "descartar" if representante else "conservar",
                      "redundante_con": representante or "",
                      "spearman_con_representante": float(corr.at[c, representante]) if representante else np.nan,
                      "spearman_con_y": float(rho_y.get(c, np.nan))})
    filas += [{"columna": c, "familia": familia_de(c), "decision": "descartar",
               "redundante_con": "(constante en Train)", "spearman_con_representante": np.nan,
               "spearman_con_y": np.nan} for c in X_tr.columns if c in constantes]
    elegidas = set(conservadas)
    return [c for c in X_tr.columns if c in elegidas], pd.DataFrame(filas)


def particiones(historial: pd.DataFrame, candidatas: pd.DataFrame):
    """Train y Validation con filas completas. Test no se devuelve."""
    salida = {}
    for nombre in ("train", "validation"):
        filas = historial.index[historial["particion"] == nombre]
        X = candidatas.loc[filas]
        completas = X.notna().all(axis=1)
        salida[nombre] = (X[completas], historial.loc[filas[completas], COL_Y],
                          int((~completas).sum()))
    return salida


# ===========================================================================
# 2. Utilidades de evaluación (siempre dentro de Train)
# ===========================================================================
def validacion_interna(cfg: ConfigSeleccion) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=cfg.n_splits, gap=cfg.gap)


def modelo_ridge(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=SEMILLA))


def mae_interno(X: pd.DataFrame, y: pd.Series, columnas: List[str], alpha: float,
                cfg: ConfigSeleccion) -> Tuple[float, float]:
    """MAE medio y error estándar entre bloques de la validación temporal interna."""
    puntajes = -cross_val_score(modelo_ridge(alpha), X[columnas], y,
                                cv=validacion_interna(cfg), scoring="neg_mean_absolute_error")
    return float(puntajes.mean()), float(puntajes.std(ddof=1) / np.sqrt(len(puntajes)))


def elegir_alpha_ridge(X: pd.DataFrame, y: pd.Series, cfg: ConfigSeleccion) -> float:
    """Alpha de Ridge elegido con la validación temporal interna sobre todas las candidatas."""
    modelo = make_pipeline(StandardScaler(),
                           RidgeCV(alphas=cfg.alphas_ridge, cv=validacion_interna(cfg),
                                   scoring="neg_mean_absolute_error"))
    return float(modelo.fit(X, y)[-1].alpha_)


def elegir_k(ranking: List[str], X: pd.DataFrame, y: pd.Series, alpha: float,
             cfg: ConfigSeleccion) -> Tuple[List[str], pd.DataFrame]:
    """Recorre los primeros k del ranking y elige k con la regla de un error estándar."""
    grilla = sorted({k for k in cfg.grilla_k if k <= len(ranking)} | {len(ranking)})
    filas = []
    for k in grilla:
        mae, ee = mae_interno(X, y, ranking[:k], alpha, cfg)
        filas.append({"k": k, "mae_interno": mae, "error_estandar": ee})
    curva = pd.DataFrame(filas)
    mejor = curva.loc[curva["mae_interno"].idxmin()]
    if cfg.regla_un_error_estandar:
        umbral = mejor["mae_interno"] + mejor["error_estandar"]
        k = int(curva.loc[curva["mae_interno"] <= umbral, "k"].min())
    else:
        k = int(mejor["k"])
    curva["elegido"] = curva["k"] == k
    return ranking[:k], curva


@dataclass
class Resultado:
    metodo: str
    columnas: List[str]
    tabla: pd.DataFrame  # puntaje por candidata
    curva: Optional[pd.DataFrame] = None  # k -> MAE interno o recorrido de forward
    detalle: Dict[str, object] = field(default_factory=dict)


# ===========================================================================
# 3. Métodos de selección (independientes entre sí)
# ===========================================================================
def metodo_mutual_information(X, y, alpha, cfg) -> Resultado:
    """Ranking por información mutua (no lineal, no considera redundancia)."""
    mi = mutual_info_regression(X, y, n_neighbors=cfg.mi_vecinos, random_state=SEMILLA)
    tabla = pd.DataFrame({"columna": X.columns, "puntaje": mi}).sort_values(
        "puntaje", ascending=False, kind="stable").reset_index(drop=True)
    columnas, curva = elegir_k(tabla["columna"].tolist(), X, y, alpha, cfg)
    return Resultado("mutual_information", columnas, tabla, curva)


def metodo_ridge(X, y, alpha, cfg) -> Resultado:
    """Ranking por |coeficiente| de Ridge sobre variables estandarizadas."""
    modelo = modelo_ridge(alpha).fit(X, y)
    coef = modelo[-1].coef_
    tabla = pd.DataFrame({"columna": X.columns, "coeficiente": coef, "puntaje": np.abs(coef)}).sort_values(
        "puntaje", ascending=False, kind="stable").reset_index(drop=True)
    columnas, curva = elegir_k(tabla["columna"].tolist(), X, y, alpha, cfg)
    return Resultado("ridge", columnas, tabla, curva, {"alpha": alpha})


def metodo_lasso(X, y, alpha, cfg) -> Resultado:
    """LassoCV con validación temporal: se conservan los coeficientes no nulos."""
    modelo = make_pipeline(StandardScaler(), LassoCV(
        alphas=cfg.lasso_alphas, cv=validacion_interna(cfg), max_iter=cfg.lasso_max_iter,
        random_state=SEMILLA, n_jobs=-1)).fit(X, y)
    lasso = modelo[-1]
    tabla = pd.DataFrame({"columna": X.columns, "coeficiente": lasso.coef_,
                          "puntaje": np.abs(lasso.coef_)}).sort_values(
        "puntaje", ascending=False, kind="stable").reset_index(drop=True)
    columnas = tabla.loc[tabla["puntaje"] > 0, "columna"].tolist()
    curva = pd.DataFrame({"alpha": lasso.alphas_,
                          "rmse_interno": lasso.mse_path_.mean(axis=1) ** 0.5,
                          "n_no_nulos": [np.nan] * len(lasso.alphas_)})
    curva["elegido"] = np.isclose(curva["alpha"], lasso.alpha_)
    return Resultado("lasso", columnas, tabla, curva, {"alpha": float(lasso.alpha_)})


def metodo_forward(X, y, alpha, cfg) -> Resultado:
    """Agrega de a una la variable que más baja el MAE interno de Ridge.
    Se detiene si la mejora es menor a `forward_tolerancia_kw` o al llegar al máximo."""
    elegidas: List[str] = []
    restantes = list(X.columns)
    recorrido, mae_actual = [], np.inf
    with Parallel(n_jobs=-1, max_nbytes=None) as paralelo:
        while restantes and len(elegidas) < cfg.forward_max_variables:
            resultados = paralelo(delayed(mae_interno)(X, y, elegidas + [c], alpha, cfg)
                                  for c in restantes)
            maes = pd.Series([r[0] for r in resultados], index=restantes)
            mejor = maes.idxmin()
            mejora = mae_actual - maes[mejor]
            if mejora < cfg.forward_tolerancia_kw:
                recorrido.append({"paso": len(elegidas) + 1, "candidata": mejor,
                                  "mae_interno": maes[mejor], "mejora_kw": mejora,
                                  "agregada": False})
                break
            elegidas.append(mejor)
            restantes.remove(mejor)
            mae_actual = maes[mejor]
            recorrido.append({"paso": len(elegidas), "candidata": mejor,
                              "mae_interno": mae_actual, "mejora_kw": mejora, "agregada": True})
    curva = pd.DataFrame(recorrido)
    tabla = pd.DataFrame({"columna": X.columns,
                          "orden_entrada": [elegidas.index(c) + 1 if c in elegidas else np.nan
                                            for c in X.columns]}).sort_values("orden_entrada")
    tabla["puntaje"] = -tabla["orden_entrada"]
    return Resultado("forward", elegidas, tabla, curva)


# Para agregar un método: función (X_train, y_train, alpha_ridge, cfg) -> Resultado.
METODOS: Dict[str, Callable[..., Resultado]] = {
    "mutual_information": metodo_mutual_information,
    "ridge": metodo_ridge,
    "lasso": metodo_lasso,
    "forward": metodo_forward,
}


# ===========================================================================
# 4. Comparación en Validation
# ===========================================================================
def crear_estimador(nombre: str, alpha: float):
    if nombre == "ridge":
        return modelo_ridge(alpha)
    if nombre == "random_forest":
        return RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                     n_jobs=-1, random_state=SEMILLA)
    raise ValueError(f"Estimador desconocido: {nombre}")


def subconjuntos_a_comparar(resultados: Dict[str, Resultado], todas: List[str]) -> Dict[str, List[str]]:
    """Cada método, el consenso (elegidas por al menos dos métodos) y todas las candidatas."""
    subconjuntos = {r.metodo: r.columnas for r in resultados.values()}
    votos = pd.Series([c for r in resultados.values() for c in r.columnas]).value_counts()
    consenso = [c for c in todas if votos.get(c, 0) >= 2]
    if consenso:
        subconjuntos["consenso_2_metodos"] = consenso
    subconjuntos["todas"] = list(todas)
    return subconjuntos


def comparar_en_validation(subconjuntos, X_tr, y_tr, X_val, y_val, alpha, cfg) -> pd.DataFrame:
    filas = []
    for nombre, columnas in subconjuntos.items():
        fila = {"subconjunto": nombre, "columnas": len(columnas)}
        for est in cfg.estimadores_validacion:
            modelo = crear_estimador(est, alpha).fit(X_tr[columnas], y_tr)
            fila[f"mae_val_{est}"] = mean_absolute_error(y_val, modelo.predict(X_val[columnas]))
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    cols_mae = [f"mae_val_{e}" for e in cfg.estimadores_validacion]
    tabla["mae_val_promedio"] = tabla[cols_mae].mean(axis=1)
    return tabla.sort_values(["mae_val_promedio", "columnas"]).reset_index(drop=True)


def baselines_validation(X_val: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
    """Persistencia (frío de D) y media de 7 días (D-6..D), sobre las mismas filas."""
    return pd.DataFrame([
        {"subconjunto": "baseline_persistencia", "columnas": 0,
         "mae_val_promedio": mean_absolute_error(y_val, X_val[COL_FRIO])},
        {"subconjunto": "baseline_media_7", "columnas": 0,
         "mae_val_promedio": mean_absolute_error(y_val, X_val["frio_media_7"])},
    ])


# ===========================================================================
# Orquestación
# ===========================================================================
def seleccionar(X_tr: pd.DataFrame, y_tr: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                cfg: ConfigSeleccion, metodos: Optional[List[str]] = None):
    """Alpha de Ridge, métodos, subconjuntos y comparación en Validation, sin exportar.
    Sirve para repetir la selección sobre otro conjunto de candidatas."""
    metodos = metodos or list(METODOS)
    alpha = elegir_alpha_ridge(X_tr, y_tr, cfg)
    resultados = {m: METODOS[m](X_tr, y_tr, alpha, cfg) for m in metodos}
    subconjuntos = subconjuntos_a_comparar(resultados, list(X_tr.columns))
    comparacion = comparar_en_validation(subconjuntos, X_tr, y_tr, X_val, y_val, alpha, cfg)
    return alpha, resultados, subconjuntos, comparacion


def ejecutar(cfg: Optional[ConfigSeleccion] = None, metodos: Optional[List[str]] = None,
             salida: Path = RESULTS_DIR / "seleccion") -> Dict[str, object]:
    cfg = cfg or ConfigSeleccion()
    metodos = metodos or list(METODOS)
    historial = cargar_preprocesado()
    candidatas, creadas = construir_candidatas_y_ajuste(historial, cfg)
    datos = particiones(historial, candidatas)
    X_tr, y_tr, descartadas_tr = datos["train"]
    X_val, y_val, descartadas_val = datos["validation"]
    if X_tr.index.max() >= X_val.index.min():
        raise ValueError("Validation debe empezar después de Train.")

    n_candidatas = X_tr.shape[1]
    por_familia_inicial = pd.Series([familia_de(c) for c in X_tr.columns]).value_counts()
    columnas, redundancia = filtrar_redundantes(X_tr, y_tr, cfg)
    X_tr, X_val = X_tr[columnas], X_val[columnas]

    alpha, resultados, subconjuntos, comparacion = seleccionar(X_tr, y_tr, X_val, y_val, cfg, metodos)
    ganador = comparacion.iloc[0]["subconjunto"]
    # Los baselines usan el frío y su media de 7 cierres, aunque el filtro las haya quitado.
    tabla_final = pd.concat([comparacion, baselines_validation(candidatas.loc[X_val.index], y_val)],
                            ignore_index=True)

    # Matriz de qué método eligió cada variable.
    elegidas = sorted({c for r in resultados.values() for c in r.columnas},
                      key=lambda c: list(X_tr.columns).index(c))
    votos = pd.DataFrame({m: [c in r.columnas for c in elegidas] for m, r in resultados.items()},
                         index=elegidas)
    votos["n_metodos"] = votos.sum(axis=1)
    votos = votos.sort_values("n_metodos", ascending=False, kind="stable")

    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)

    def guardar(df, nombre, index=False):
        df.to_csv(salida / nombre, index=index, encoding="utf-8", lineterminator="\n")

    for m, r in resultados.items():
        guardar(r.tabla, f"{m}__ranking.csv")
        if r.curva is not None:
            guardar(r.curva, f"{m}__curva.csv")
    guardar(votos.assign(familia=votos.index.map(familia_de)), "votos_por_variable.csv", index=True)
    guardar(tabla_final, "comparacion_validation.csv")
    guardar(redundancia, "filtro_redundancia.csv")
    if creadas is not None:
        guardar(creadas.reporte_, "variables_creadas.csv")
    por_familia = pd.Series([familia_de(c) for c in X_tr.columns]).value_counts()
    elegidas_por_familia = {m: pd.Series([familia_de(c) for c in r.columnas], dtype=object)
                            .value_counts().to_dict() for m, r in resultados.items()}

    resumen = {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "src/seleccion_variables.py",
        "semilla": SEMILLA,
        "config": asdict(cfg),
        "entrada_sha256": huella(CARPETA_PREPROCESADO / "historial_completo.csv"),
        "filas": {"train": len(X_tr), "validation": len(X_val),
                  "train_descartadas_por_historia_incompleta": descartadas_tr,
                  "validation_descartadas_por_historia_incompleta": descartadas_val},
        "fechas": {"train": [str(X_tr.index.min().date()), str(X_tr.index.max().date())],
                   "validation": [str(X_val.index.min().date()), str(X_val.index.max().date())]},
        "candidatas": n_candidatas,
        "candidatas_por_familia": {k: int(v) for k, v in por_familia_inicial.items()},
        "candidatas_tras_filtro_redundancia": X_tr.shape[1],
        "candidatas_tras_filtro_por_familia": {k: int(v) for k, v in por_familia.items()},
        "alpha_ridge": alpha,
        "metodos": {m: {"n_columnas": len(r.columnas), "por_familia": elegidas_por_familia[m],
                        "columnas": r.columnas, **r.detalle}
                    for m, r in resultados.items()},
        "subconjuntos": subconjuntos,
        "ganador": ganador,
        "columnas_ganador": subconjuntos[ganador],
        "criterio": "Menor MAE promedio en Validation entre los estimadores; empate: menos columnas.",
    }
    with open(salida / "resumen.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")

    print("=" * 78)
    print("SELECCIÓN DE VARIABLES")
    print("=" * 78)
    print(f"Candidatas: {n_candidatas} -> {X_tr.shape[1]} tras el filtro de redundancia | "
          f"Train: {len(X_tr)} filas | Validation: {len(X_val)} filas")
    print("  " + " | ".join(f"{f}: {n}" for f, n in por_familia.items()))
    print(f"Alpha de Ridge (validación temporal interna): {alpha}")
    for m, r in resultados.items():
        print(f"  {m:<20}: {len(r.columnas):>3} variables")
    print(tabla_final.round(1).to_string(index=False))
    print(f"Ganador: {ganador}")
    print("=" * 78)
    return {"resultados": resultados, "comparacion": tabla_final, "votos": votos,
            "subconjuntos": subconjuntos, "ganador": ganador, "alpha": alpha,
            "X_train": X_tr, "y_train": y_tr, "X_val": X_val, "y_val": y_val, "resumen": resumen}


def main() -> None:
    parser = argparse.ArgumentParser(description="Selección de variables.")
    parser.add_argument("--metodos", nargs="*", choices=list(METODOS), default=list(METODOS))
    args = parser.parse_args()
    ejecutar(metodos=args.metodos)


if __name__ == "__main__":
    main()
