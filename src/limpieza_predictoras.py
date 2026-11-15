"""
Limpieza de las variables predictoras operativas (Sección 2.1).

Dos transformadores compatibles con sklearn, pensados para ir dentro del Pipeline:
1. LimpiadorExtremos: pasa a NaN los valores imposibles (negativos o mayores a
   `factor` veces el percentil `cuantil` de cada columna).
2. ImputadorTemporal: completa los NaN sin mirar el futuro (forward fill limitado)
   y, lo que quede, con la mediana de cada columna.

Los umbrales y las medianas se calculan en `fit()`, que se llama SOLO con train.
Validación, test y los datos nuevos de predict.py usan esos mismos valores en
`transform()`, sin recalcular nada: así no hay fuga de información del futuro.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _como_dataframe(X, columnas_esperadas: Optional[List[str]] = None) -> pd.DataFrame:
    """Convierte X a DataFrame y verifica que tenga las columnas esperadas (por nombre)."""
    X = pd.DataFrame(X).copy()
    if columnas_esperadas is not None:
        faltantes = [c for c in columnas_esperadas if c not in X.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas predictoras en los datos de entrada: {faltantes}")
        X = X[columnas_esperadas]
    return X


class LimpiadorExtremos(BaseEstimator, TransformerMixin):
    """Pasa a NaN los valores imposibles de cada columna.

    Imposible = negativo (un totalizador diario no puede ser < 0) o mayor a
    `factor` x percentil `cuantil` de train (errores del medidor, del orden de millones).
    """

    def __init__(self, factor: float = 5.0, cuantil: float = 0.99):
        self.factor = factor
        self.cuantil = cuantil

    def fit(self, X, y=None):
        X = _como_dataframe(X)
        self.columnas_ = list(X.columns)
        self.topes_ = X.quantile(self.cuantil) * self.factor
        return self

    def transform(self, X):
        X = _como_dataframe(X, self.columnas_)
        imposibles = (X < 0) | (X > self.topes_)
        self.reemplazos_ = imposibles.sum()  # cuántos valores se pasaron a NaN (último transform)
        return X.mask(imposibles)

    def get_feature_names_out(self, input_features=None):
        return list(self.columnas_)


class ImputadorTemporal(BaseEstimator, TransformerMixin):
    """Completa los NaN sin usar información futura.

    1. Forward fill limitado a `limite` filas: el valor faltante de un día se toma
       del último día anterior con dato (nunca de un día posterior).
    2. Lo que quede (por ejemplo, al comienzo de la serie o huecos largos) se
       completa con la mediana de train.
    Supone que las filas están en orden cronológico.
    """

    def __init__(self, limite: int = 3):
        self.limite = limite

    def fit(self, X, y=None):
        X = _como_dataframe(X)
        self.columnas_ = list(X.columns)
        self.medianas_ = X.median()
        return self

    def transform(self, X):
        X = _como_dataframe(X, self.columnas_)
        nulos_antes = X.isna().sum()
        X_ffill = X.ffill(limit=self.limite)
        self.imputados_ffill_ = nulos_antes - X_ffill.isna().sum()
        self.imputados_mediana_ = X_ffill.isna().sum()
        return X_ffill.fillna(self.medianas_)

    def get_feature_names_out(self, input_features=None):
        return list(self.columnas_)