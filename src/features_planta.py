"""
Features de planta (Sección 2.2, parte B): ratios entre áreas, eficiencia contra
la Meta e interacciones con el clima.

`FeaturesPlanta` es un transformador de sklearn. En `fit()` (llamado SOLO con train)
aprende lo que depende de los datos:
- topes de valores imposibles y medianas de las columnas base (5 x P99 de train),
- la Meta típica de cada mes del año (para los meses sin Meta, antes de jul-2021).
En `transform()` calcula las features nuevas con esos valores, sin recalcular nada,
por lo que validación, test y predict.py no usan información del futuro.
Todas las features usan datos del día D (o anteriores): ninguna mira el día D+1.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

try:
    from src.limpieza_predictoras import ImputadorTemporal, LimpiadorExtremos
except ImportError:  # permite usar el módulo fuera del paquete src
    from limpieza_predictoras import ImputadorTemporal, LimpiadorExtremos

# Columnas del dataset_trabajo que usa esta sección (nombre_hoja)
COLUMNAS_BASE: Dict[str, str] = {
    "frio": "Frio (Kw)_Consolidado EE",
    "servicios": "Servicios (Kw)_Consolidado EE",
    "bodega": "Bodega (Kw)_Consolidado EE",
    "elaboracion": "Elaboracion (Kw)_Consolidado EE",
    "agua_cond": "Agua Cond Evaporativos_Consolidado Agua",
    "agua_planta": "Agua Planta (Hl)_Consolidado Agua",
    "hl_bodega": "Hl Producido Bodega_Consolidado Produccion",
    "ee_frio_hl": "EE Frio / Hl_Consolidado KPI",
    "meta_frio": "Meta EE Frio_Consolidado KPI",
}
# Columnas que se limpian (imposibles a NaN + imputación). Agua Cond Evaporativos y la Meta
# no se imputan: les falta un año entero y rellenarlo con una mediana inventaría datos.
COLUMNAS_A_LIMPIAR: List[str] = [
    COLUMNAS_BASE[k] for k in ("frio", "servicios", "bodega", "elaboracion", "agua_planta", "hl_bodega", "ee_frio_hl")
]


def _ratio(numerador: pd.Series, denominador: pd.Series) -> pd.Series:
    """Cociente seguro: si el denominador es <= 0 (sin actividad o error), el resultado es NaN."""
    return numerador / denominador.where(denominador > 0)


class FeaturesPlanta(BaseEstimator, TransformerMixin):
    """Crea ratios entre áreas, eficiencia contra la Meta e interacciones con el clima.

    Espera un DataFrame diario ordenado por fecha con la columna `col_fecha`,
    las columnas de COLUMNAS_BASE y las columnas de clima de `columnas_clima`.
    """

    def __init__(self, col_fecha: str = "DIA", columnas_clima: tuple = ("temp_media_c", "radiacion_roll7")):
        self.col_fecha = col_fecha
        self.columnas_clima = columnas_clima

    def _verificar_columnas(self, X: pd.DataFrame) -> None:
        requeridas = [self.col_fecha] + list(COLUMNAS_BASE.values()) + list(self.columnas_clima)
        faltantes = [c for c in requeridas if c not in X.columns]
        if faltantes:
            raise ValueError(f"FeaturesPlanta: faltan columnas en los datos de entrada: {faltantes}")

    def fit(self, X: pd.DataFrame, y=None):
        self._verificar_columnas(X)
        # Topes de imposibles y medianas aprendidos en train
        self.limpieza_ = Pipeline([
            ("extremos", LimpiadorExtremos(factor=5.0, cuantil=0.99)),
            ("imputacion", ImputadorTemporal(limite=3)),
        ]).fit(X[COLUMNAS_A_LIMPIAR])
        # Meta típica de cada mes del año (las metas son mensuales y cambian con la estación)
        meses = pd.to_datetime(X[self.col_fecha]).dt.month
        self.meta_por_mes_ = X[COLUMNAS_BASE["meta_frio"]].groupby(meses).median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._verificar_columnas(X)
        X = X.copy()
        base = self.limpieza_.transform(X[COLUMNAS_A_LIMPIAR])
        c = COLUMNAS_BASE
        nuevas = pd.DataFrame(index=X.index)

        # 1) Ratios entre áreas (proporciones del mismo día D)
        nuevas["ratio_frio_servicios"] = _ratio(base[c["frio"]], base[c["servicios"]])
        nuevas["ratio_bodega_elaboracion"] = _ratio(base[c["bodega"]], base[c["elaboracion"]])
        agua_cond = X[c["agua_cond"]].where(X[c["agua_cond"]] >= 0)  # sin imputar: falta el primer año
        nuevas["ratio_aguacond_aguaplanta"] = _ratio(agua_cond, base[c["agua_planta"]])

        # 2) Eficiencia contra la Meta: > 1 = se consumió más frío por Hl que lo planificado
        meses = pd.to_datetime(X[self.col_fecha]).dt.month
        meta = X[c["meta_frio"]].where(X[c["meta_frio"]] > 0)
        nuevas["meta_frio_imputada"] = meta.isna().astype(int)  # trazabilidad: 1 = Meta tomada del mes típico
        meta = meta.fillna(meses.map(self.meta_por_mes_))
        nuevas["eficiencia_frio_vs_meta"] = _ratio(base[c["ee_frio_hl"]], meta)

        # 3) Interacciones producción x clima: más cerveza en bodega en días calurosos -> más carga de frío
        for col_clima in self.columnas_clima:
            nuevas[f"bodega_x_{col_clima}"] = base[c["hl_bodega"]] * X[col_clima]
        return nuevas

    def get_feature_names_out(self, input_features=None):
        nombres = ["ratio_frio_servicios", "ratio_bodega_elaboracion", "ratio_aguacond_aguaplanta",
                   "meta_frio_imputada", "eficiencia_frio_vs_meta"]
        return nombres + [f"bodega_x_{c}" for c in self.columnas_clima]