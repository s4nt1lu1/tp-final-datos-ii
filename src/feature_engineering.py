"""Limpieza de predictoras y creación de variables (secciones 2.1 y 2.2)."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.auxiliar_functions import DATA_DIR
from src.features_planta import COLUMNAS_BASE, FeaturesPlanta
from src.limpieza_predictoras import ImputadorTemporal, LimpiadorExtremos


FRIO = COLUMNAS_BASE["frio"]
CLIMA_CSV = DATA_DIR / "reference" / "clima_tecate_openmeteo.csv"

# Hojas con medidas operativas que pueden estar disponibles al cierre del día.
# El criterio es de dominio. El ranking de toda la serie del EDA queda exploratorio.
HOJAS_OPERATIVAS = (
    "Consolidado Produccion",
    "Totalizadores Produccion",
    "Consolidado EE",
    "Totalizadores Energia",
    "Consolidado Agua",
    "Consolidado GasVapor",
    "Consolidado Aire",
    "Totalizadores CO2",
)
PROXIES = [
    FRIO,
    "Sala Maq (Kw)_Consolidado EE",
    "Servicios (Kw)_Consolidado EE",
    "Planta (Kw)_Consolidado EE",
    "KW Gral Planta_Consolidado EE",
    "KW Gral Planta_Totalizadores Energia",
]
for numero in range(1, 8):
    PROXIES.append(f"KW Mycom {numero}_Totalizadores Energia")

VARIABLES_PLANTA = [
    "ratio_frio_servicios",
    "ratio_bodega_elaboracion",
    "eficiencia_frio_vs_meta",
    "bodega_x_calor_estacional",
]


def ordenar_diario(diario):
    """Prepara un índice de fechas único para buscar días exactos."""
    base = diario.copy()
    base["DIA"] = pd.to_datetime(base["DIA"]).dt.normalize()
    if base.empty or base["DIA"].isna().any() or base["DIA"].duplicated().any():
        raise ValueError("Se necesita un historial diario con fechas válidas y únicas.")
    return base.sort_values("DIA").set_index("DIA", drop=False)


def columnas_numericas(base, columnas):
    """Convierte mediciones inválidas a NaN y avisa qué columnas faltan."""
    faltantes = [columna for columna in columnas if columna not in base.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas para calcular las predictoras: {faltantes}")
    valores = base[columnas].apply(pd.to_numeric, errors="coerce")
    return valores.replace([np.inf, -np.inf], np.nan)


def elegir_operativas(train, y_train):
    """Aplica las reglas del equipo de completitud, relevancia y colinealidad."""
    columnas = []
    for columna in train.columns:
        hoja = columna.rsplit("_", 1)[-1]
        if hoja in HOJAS_OPERATIVAS and columna not in PROXIES:
            columnas.append(columna)

    datos = columnas_numericas(train, columnas)
    # La selección ignora lecturas imposibles según topes estimados en Train.
    datos = LimpiadorExtremos().fit_transform(datos)
    faltantes = datos.isna().mean()
    con_variacion = datos.nunique() > 1
    aptas = datos.columns[(faltantes <= 0.20) & con_variacion]
    rho = datos[aptas].corrwith(y_train, method="spearman")
    rho_frio_hoy = datos[aptas].corrwith(train[FRIO], method="spearman")

    # Las variables que casi duplican Frío(D) aportan una medición redundante.
    relevantes = rho.index[(rho.abs() >= 0.40) & (rho_frio_hoy.abs() < 0.98)]
    orden = rho.loc[relevantes].abs().sort_values(ascending=False, kind="stable").index
    colinealidad = datos[orden].corr(method="spearman").abs()
    elegidas = []
    decisiones = {}
    for columna in datos.columns:
        if faltantes[columna] > 0.20:
            decisiones[columna] = "Más de 20 % faltante"
        elif not con_variacion[columna]:
            decisiones[columna] = "Sin variación"
        elif abs(rho_frio_hoy[columna]) >= 0.98:
            decisiones[columna] = "Redundante con Frío de hoy"
        else:
            decisiones[columna] = "Spearman con y débil o indefinido"

    # Se conserva la primera del ranking compatible con las ya elegidas.
    for columna in orden:
        relacionadas = colinealidad.loc[columna, elegidas]
        redundantes = relacionadas.index[relacionadas >= 0.90]
        if len(redundantes) > 0:
            decisiones[columna] = "Colineal con " + redundantes[0]
        else:
            elegidas.append(columna)
            decisiones[columna] = "Elegida"

    tabla = pd.DataFrame({
        "faltantes_train": faltantes,
        "spearman_y_train": rho,
        "spearman_frio_hoy_train": rho_frio_hoy,
        "decision": pd.Series(decisiones),
    })
    return elegidas, tabla


def features_temporales(base):
    """Calcula rezagos y ventanas calendario con la información disponible en D."""
    # Se usa el historial completo de cierres, incluso cuando un día carece de y.
    frio = pd.to_numeric(base[FRIO], errors="coerce")
    frio = frio.where(np.isfinite(frio) & (frio >= 0)).asfreq("D")
    historial = pd.DataFrame(index=frio.index)
    historial["frio_hoy_d"] = frio
    historial["frio_lag_1"] = frio.shift(1)
    historial["frio_lag_2"] = frio.shift(2)
    historial["frio_lag_7"] = frio.shift(7)
    # Conservamos los mínimos de cobertura del borrador de 2.2A.
    historial["frio_roll_mean_3"] = frio.rolling(3, min_periods=2).mean()
    historial["frio_roll_mean_7"] = frio.rolling(7, min_periods=4).mean()
    historial["frio_roll_mean_14"] = frio.rolling(14, min_periods=7).mean()
    historial["frio_roll_std_7"] = frio.rolling(7, min_periods=4).std()
    historial["mes_numero"] = historial.index.month
    # Un índice sin unidades para la interacción producción × época cálida.
    angulo = 2 * np.pi * historial["mes_numero"] / 12
    historial["calor_estacional"] = (1 - np.cos(angulo)) / 2
    return historial.loc[base.index].copy()


class PreparadorFeatures:
    """Guarda umbrales y medianas de Train para aplicarlos a otros períodos."""

    def fit(self, historial_train, y_train):
        base = ordenar_diario(historial_train)
        if base.index.max() > y_train.index.max():
            raise ValueError("El historial de ajuste no puede superar el último D de Train.")
        # Solo los pares supervisados pueden estimar topes, medianas o correlaciones.
        train = base.loc[y_train.index]
        self.operativas_, self.tabla_operativas_ = elegir_operativas(train, y_train)
        self.limpieza_ = Pipeline([
            ("extremos", LimpiadorExtremos()),
            ("imputacion", ImputadorTemporal(limite=3)),
        ])
        if self.operativas_:
            self.limpieza_.fit(columnas_numericas(train, self.operativas_))

        temporal = features_temporales(base)
        self.imputador_temporal_ = ImputadorTemporal(limite=3).fit(temporal.loc[train.index])
        # Calculamos ratios y KPI con la lógica de Genaro. El ratio de agua queda
        # para el diagnóstico porque en el primer libro falta uno de sus insumos.
        self.planta_ = FeaturesPlanta(
            columnas_clima=("calor_estacional",),
            incluir_agua_cond=False,
        )
        entrada_planta = base.join(temporal[["calor_estacional"]])
        self.planta_.fit(entrada_planta.loc[train.index])
        planta_train = self._features_planta(entrada_planta)
        self.imputador_planta_ = ImputadorTemporal(limite=3).fit(planta_train.loc[train.index])
        return self

    def _features_planta(self, entrada):
        planta = self.planta_.transform(entrada)[VARIABLES_PLANTA].copy()
        # Frío está contenido en Servicios. Un cociente mayor a 1 indica
        # inconsistencia entre ambos medidores y se trata como dato faltante.
        proporcion = planta["ratio_frio_servicios"]
        planta["ratio_frio_servicios"] = proporcion.where(proporcion.between(0, 1))
        return planta

    def transform(self, historial):
        """Genera X sin etiquetas. Pasar también los cierres previos al tramo pedido."""
        base = ordenar_diario(historial)
        temporal = features_temporales(base)
        # Calculamos la diferencia desde el rezago imputado para mantener su significado.
        temporal_limpio = self.imputador_temporal_.transform(temporal)
        temporal_limpio["frio_diff_1"] = (
            temporal_limpio["frio_hoy_d"] - temporal_limpio["frio_lag_1"]
        )
        entrada_planta = base.join(temporal[["calor_estacional"]])
        planta = self.imputador_planta_.transform(self._features_planta(entrada_planta))
        partes = [temporal_limpio, planta]
        if self.operativas_:
            operativas = columnas_numericas(base, self.operativas_)
            partes.insert(0, self.limpieza_.transform(operativas))
        return pd.concat(partes, axis=1)


def clima_historico(path=CLIMA_CSV):
    """Lee la copia fija de Open-Meteo y calcula el diagnóstico de 2.2A."""
    clima = pd.read_csv(path, parse_dates=["DIA"]).sort_values("DIA").set_index("DIA")
    calendario = pd.date_range(clima.index.min(), clima.index.max(), freq="D")
    if not clima.index.equals(calendario) or clima.isna().any().any():
        raise ValueError("El clima histórico debe tener una fila completa por fecha calendario.")
    clima["amplitud_termica_c"] = clima["temp_max_c"] - clima["temp_min_c"]
    clima["temp_media_lag1"] = clima["temp_media_c"].shift(1)
    clima["temp_media_roll3"] = clima["temp_media_c"].rolling(3, min_periods=1).mean()
    clima["temp_media_roll7"] = clima["temp_media_c"].rolling(7, min_periods=1).mean()
    clima["temp_max_roll3"] = clima["temp_max_c"].rolling(3, min_periods=1).mean()
    clima["radiacion_roll3"] = clima["radiacion_mj_m2"].rolling(3, min_periods=1).mean()
    clima["radiacion_roll7"] = clima["radiacion_mj_m2"].rolling(7, min_periods=1).mean()
    return clima
