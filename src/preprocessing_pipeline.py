"""Fase 2. Decisiones estructurales (EDA), preprocesamiento y selección de variables.

Parte del dataset horario que genera `src/auxiliar_functions.py`
(`data/processed/dataset_trabajo.csv`) y tiene tres etapas, en este orden:

Etapa 1. Decisiones estructurales sobre las columnas (EDA)
    Decide, antes de cualquier ajuste estadístico, qué columnas no deben llegar al
    preprocesamiento y cuáles son la misma medición con otro nombre:
        1. metadatos        : identificadores, columnas sin encabezado y la hora del último dato.
        2. componentes_frio : `Frio (Kw)` es la suma exacta de 7 compresores Mycom, 3 grupos de
                              condensadores y 2 bombas de glicol; su reparto cambia por rotación.
        3. proxies_frio     : medidores que contienen al frío (Sala de Máquinas, Servicios,
                              Planta y KW Gral Planta).
        4. residuales       : columnas calculadas por diferencia, con valores negativos.
        5. renombradas      : la misma medición con otro nombre según el Excel; se unen.
        6. copias           : columnas idénticas en toda la serie; se conserva una.
    Dependen de la estructura de la planilla y no de `y` ni de parámetros estimados, así que
    pueden tomarse con la serie completa sin fuga de información. No se exportan: el
    preprocesamiento las recalcula en memoria (`cargar_decisiones`, unos segundos).

Etapa 2. Preprocesamiento ajustado con Train -> `data/processed/`
    Toma el cierre diario 23:59, arma el target D+1 y la división cronológica, y aplica
    pasos independientes sobre las predictoras:
        0. aplicar_decisiones_eda    : las decisiones de la etapa 1. No estima nada.
        1. descartar_no_informativas : columnas vacías o constantes en Train.
        2. descartar_duplicadas      : columnas idénticas a otra en Train.
        3. filtrar_por_nulos         : clasificación de nulos y descarte de las no utilizables.
        4. outliers_univariados      : errores de medición -> NaN, extremos -> winsorización.
        5. imputar_nulos             : método elegido por columna con enmascaramiento en Train.
        6. (análisis) LOF            : días anómalos en el espacio multivariado.
    Salida sin escalar: historial_completo.csv, train.csv, validation.csv y test.csv, con
    sus SHA-256 en `data/checksums.json`.

Etapa 3. Variables creadas y selección
    Candidatas: predictoras del preprocesamiento + rezagos, medias móviles y diferencias
    del frío + calendario + variables creadas (ciclo anual, binning, interacciones, ratios
    entre áreas y eficiencia contra la meta). Métodos ajustados solo con Train (Mutual
    Information, Ridge, Lasso y forward) y comparación de los subconjuntos en Validation.
    No se exporta: `src/train_model.py` la recalcula en memoria (`calcular_seleccion`).

Las tablas de cada decisión no se guardan en archivos: las funciones `ejecutar_*` las
devuelven y las notebooks las muestran.

Reglas contra la fuga de información (data leakage):
    * Todo `fit` recibe SOLO filas con fecha <= último día de Train.
    * Todo `transform` es causal: la fila del día t depende solo de filas con fecha <= t y
      de lo aprendido en `fit`. `verificar_causalidad` lo comprueba.
    * El target `y` nunca se transforma. Test no se lee en ninguna etapa.
    * La validación interna usa TimeSeriesSplit con `gap=1`, y el escalado vive dentro de
      cada Pipeline, ajustado en cada bloque de ajuste.

Uso desde la raíz del proyecto:
    python -m src.preprocessing_pipeline                      # las tres etapas
    python -m src.preprocessing_pipeline --completo           # antes, la limpieza horaria de los Excel
    python -m src.preprocessing_pipeline --etapas eda         # solo una etapa
    python -m src.preprocessing_pipeline --omitir outliers_univariados --tratamiento-lof excluir_train
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import nnls
from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src import auxiliar_functions as af
from src.auxiliar_functions import DATA_DIR, PROJECT_ROOT, SEMILLA, huella

COL_FECHA = af.DEFAULT_DATE_COL
COL_FRIO = af.DEFAULT_TARGET_COL
COL_Y = af.DEFAULT_TARGET_NAME
ETAPAS = ("eda", "preprocesamiento", "seleccion")
NOMBRES_PASOS = ("aplicar_decisiones_eda", "descartar_no_informativas", "descartar_duplicadas",
                 "filtrar_por_nulos", "outliers_univariados", "imputar_nulos")


# ###########################################################################
# ETAPA 1. DECISIONES ESTRUCTURALES SOBRE LAS COLUMNAS (EDA)
# ###########################################################################
CLAVE_ENTRADA = "data/processed/dataset_trabajo.csv"


@dataclass
class ConfigEDA:
    # 1. Metadatos sin significado físico
    patrones_metadatos: Tuple[str, ...] = (
        r"^Id_",                   # identificadores de fila de cada hoja
        r"^\.?\d*_",               # columnas sin encabezado en el Excel ("_Auxiliar", ".3_Auxiliar")
        r"^Ultimo Dato del Dia_",  # hora del último registro, no una medición
    )

    # 2. Componentes del frío: Frio (Kw) = suma de estos medidores (se verifica)
    grupos_componentes_frio: Tuple[Tuple[str, str], ...] = (
        ("Compresores Mycom", r"^KW Mycom \d+_Totalizadores Energia$"),
        ("Condensadores", r"^KW Cond .*_Totalizadores Energia$"),
        ("Bombas de glicol", r"^KW Bba Glicol .*_Totalizadores Energia$"),
    )
    tolerancia_identidad_kw: float = 50.0  # |frío − suma| aceptable en una lectura horaria

    # 3. Medidores que contienen al frío (se verifica que superen al frío casi todos los días)
    proxies_frio: Tuple[str, ...] = (
        "Sala Maq (Kw)_Consolidado EE",
        "Servicios (Kw)_Consolidado EE",
        "Planta (Kw)_Consolidado EE",
        "KW Gral Planta_Consolidado EE",
        "KW Gral Planta_Totalizadores Energia",
    )

    # 4. Columnas calculadas por diferencia (total menos áreas)
    columnas_residuales: Tuple[str, ...] = (
        "Resto Serv (Kw)_Consolidado EE",
        "Restos Planta (Kw)_Consolidado EE",
        "Agua Servicios (Hl)_Consolidado Agua",
        "Aire Servicios (M3)_Consolidado Aire",
    )

    # 5. Renombradas entre archivos Excel
    similitud_nombres: float = 0.85    # difflib sobre el nombre normalizado
    max_solape_renombradas: int = 3    # días con ambas columnas presentes
    rango_ratio_nivel: Tuple[float, float] = (0.67, 1.5)  # continuidad de nivel esperada
    # Pares con salto de nivel que el grupo confirmó como la misma medición
    renombradas_confirmadas: Tuple[str, ...] = ("Tot L3, L4 y Planta de CO2_Totalizadores Glicol",)

    # 6. Copias exactas: qué versión se conserva
    columnas_obligatorias: Tuple[str, ...] = (COL_FRIO,)
    hojas_preferidas: Tuple[str, ...] = (r"_Totalizadores ",)
    decimales_copias: int = 6


# ===========================================================================
# Carga
# ===========================================================================
def cargar_datos_eda(input_csv: Path = af.DEFAULT_INPUT_CSV,
                 checksums_path: Path = af.DEFAULT_CHECKSUMS_JSON) -> Dict[str, object]:
    """Horario numérico y cierres diarios (23:59) de `dataset_trabajo.csv`, con su SHA-256 verificado."""
    with open(checksums_path, encoding="utf-8") as f:
        registrado = af._sha_registrado(json.load(f), CLAVE_ENTRADA)
    sha = huella(input_csv)
    if registrado is None or sha != registrado:
        raise ValueError("dataset_trabajo.csv no coincide con el SHA-256 registrado en checksums.json.")
    horario, _ = af.load_cleaned_data(input_csv)
    diario, _ = af.aggregate_daily_data(horario)
    X = diario.set_index(COL_FECHA).sort_index().apply(pd.to_numeric, errors="coerce")
    columnas = [c for c in horario.columns if c not in (COL_FECHA, "HORA")]
    horario = horario.sort_values([COL_FECHA, "HORA"]).reset_index(drop=True)
    horario[columnas] = horario[columnas].apply(pd.to_numeric, errors="coerce")
    return {"horario": horario, "diario": X, "sha256": sha}


def _normalizar_nombre(nombre: str) -> str:
    return re.sub(r"[^a-z0-9]", "", nombre.lower())


def _hoja(columna: str) -> str:
    return columna.rsplit("_", 1)[-1]


# ===========================================================================
# 1. Metadatos
# ===========================================================================
def columnas_metadatos(columnas: List[str], cfg: ConfigEDA) -> Dict[str, str]:
    salida = {}
    for c in columnas:
        patron = next((p for p in cfg.patrones_metadatos if re.search(p, c)), None)
        if patron:
            salida[c] = f"Metadato (patrón {patron})"
    return salida


# ===========================================================================
# 2. El frío como suma de componentes
# ===========================================================================
def componentes_frio(columnas: List[str], cfg: ConfigEDA) -> Dict[str, str]:
    """Columna -> grupo, para cada componente del frío presente en el dataset."""
    salida = {}
    for grupo, patron in cfg.grupos_componentes_frio:
        for c in columnas:
            if re.search(patron, c):
                salida[c] = grupo
    return salida


def verificar_identidad_frio(X: pd.DataFrame, horario: pd.DataFrame, cfg: ConfigEDA) -> Dict[str, object]:
    """Comprueba que Frio (Kw) sea la suma de sus componentes.

    * Ajuste no negativo del frío diario sobre los 12 componentes: si es una suma,
      todos los coeficientes valen 1 y el R² es 1.
    * Diferencia `frío − suma` en cada lectura horaria.
    """
    comp = componentes_frio(list(X.columns), cfg)
    columnas = list(comp)
    ok = X[columnas + [COL_FRIO]].notna().all(axis=1)
    A, b = X.loc[ok, columnas].to_numpy(float), X.loc[ok, COL_FRIO].to_numpy(float)
    coef, _ = nnls(A, b)
    pred = A @ coef
    r2 = 1 - ((b - pred) ** 2).sum() / ((b - b.mean()) ** 2).sum()

    suma_diaria = X[columnas].sum(axis=1, min_count=len(columnas))
    dif_diaria = (X[COL_FRIO] - suma_diaria).dropna()
    suma_horaria = horario[columnas].sum(axis=1, min_count=len(columnas))
    dif_horaria = (horario[COL_FRIO] - suma_horaria).dropna()

    coeficientes = pd.DataFrame({"grupo": [comp[c] for c in columnas], "coeficiente": coef},
                                index=[c.rsplit("_", 1)[0] for c in columnas])
    return {
        "columnas": columnas,
        "coeficientes": coeficientes,
        "r2": float(r2),
        "mae_ajuste_kw": float(np.abs(b - pred).mean()),
        "dias_evaluados": int(ok.sum()),
        "dif_diaria_mediana_abs_kw": float(dif_diaria.abs().median()),
        "pct_dias_dif_menor_tolerancia": float((dif_diaria.abs() <= cfg.tolerancia_identidad_kw).mean()),
        "pct_horas_dif_menor_tolerancia": float((dif_horaria.abs() <= cfg.tolerancia_identidad_kw).mean()),
        "horas_evaluadas": int(len(dif_horaria)),
    }


def composicion_frio(X: pd.DataFrame, cfg: ConfigEDA) -> pd.DataFrame:
    """Participación de cada grupo de componentes en el frío, por año calendario."""
    comp = componentes_frio(list(X.columns), cfg)
    partes = pd.DataFrame({g: X[[c for c, gr in comp.items() if gr == g]].sum(axis=1, min_count=1)
                           for g, _ in cfg.grupos_componentes_frio})
    ok = X[COL_FRIO] > 0
    participacion = partes[ok].div(X.loc[ok, COL_FRIO], axis=0)
    tabla = participacion.groupby(participacion.index.year).median().mul(100)
    tabla.insert(0, "días", participacion.groupby(participacion.index.year).size())
    tabla.index.name = "año"
    return tabla


def rotacion_mycom(X: pd.DataFrame, frac_umbral: float = 0.05) -> Dict[str, pd.DataFrame]:
    """Qué compresores funcionan cada día (consumo > frac_umbral × su P95) y cuánto suman."""
    mycom = sorted([c for c in X.columns if re.search(r"^KW Mycom \d+_", c)],
                   key=lambda c: int(re.search(r"\d+", c).group()))
    funciona = pd.DataFrame({c.split("_")[0]: X[c] > frac_umbral * X[c].quantile(0.95) for c in mycom})
    trimestre = X.index.to_period("Q")
    pct_dias = funciona.groupby(trimestre).mean().mul(100).round(0)
    pct_dias.index = pct_dias.index.astype(str)
    suma = X[mycom].sum(axis=1, min_count=1)
    resumen = pd.DataFrame({
        "compresores funcionando (media)": funciona.sum(axis=1).groupby(trimestre).mean(),
        "Σ Mycom (kW/día, mediana)": suma.groupby(trimestre).median(),
        "Σ Mycom / Frío (mediana)": (suma / X[COL_FRIO].where(X[COL_FRIO] > 0)).groupby(trimestre).median(),
    })
    resumen.index = resumen.index.astype(str)
    apagados = (X[mycom].fillna(0) == 0).all(axis=1)
    return {"pct_dias_funcionando": pct_dias, "resumen_trimestral": resumen,
            "dias_todos_apagados": X.index[apagados]}


def verificar_proxies(X: pd.DataFrame, cfg: ConfigEDA) -> pd.DataFrame:
    """Para cada medidor que contiene al frío: qué parte es frío y cuánto se parece al frío del día."""
    filas = []
    for c in cfg.proxies_frio:
        if c not in X.columns:
            continue
        ok = X[[c, COL_FRIO]].notna().all(axis=1) & (X[c] > 0)
        filas.append({
            "columna": c,
            "frío / columna (mediana)": float((X.loc[ok, COL_FRIO] / X.loc[ok, c]).median()),
            "% días columna ≥ frío": float((X.loc[ok, c] >= X.loc[ok, COL_FRIO]).mean() * 100),
            "Spearman con frío de D": float(X[c].corr(X[COL_FRIO], method="spearman")),
        })
    return pd.DataFrame(filas).set_index("columna")


# ===========================================================================
# 4. Residuales y calidad del acumulado
# ===========================================================================
def calidad_acumulado(horario: pd.DataFrame, columnas: List[str]) -> pd.DataFrame:
    """% de lecturas horarias en que el acumulado baja (sin contar el reinicio de las 00:00)
    y % de lecturas negativas. Un medidor real nunca es negativo."""
    valores = horario[columnas]
    bajadas = valores.groupby(horario[COL_FECHA]).diff()[horario["HORA"] != "00:00"]
    return pd.DataFrame({
        "% horas con bajada": (bajadas < -1e-6).sum() / bajadas.notna().sum() * 100,
        "% lecturas negativas": (valores < 0).mean() * 100,
    })


# ===========================================================================
# 5. Renombradas
# ===========================================================================
def columnas_por_archivo(archivos: List[Tuple[str, Path]]) -> pd.DataFrame:
    """Matriz hoja/columna x archivo con True si el encabezado existe en ese Excel.

    Lee solo la primera fila de cada hoja (modo de solo lectura de openpyxl).
    """
    import openpyxl

    filas = []
    for etiqueta, ruta in archivos:
        libro = openpyxl.load_workbook(ruta, read_only=True, data_only=True)
        for hoja in libro.worksheets:
            encabezado = next(hoja.iter_rows(min_row=1, max_row=1, values_only=True), ())
            for nombre in encabezado:
                nombre = str(nombre).strip() if nombre is not None else ""
                if nombre and nombre.upper() not in ("DIA", "HORA"):
                    filas.append({"hoja": hoja.title.strip(), "columna": nombre, "archivo": etiqueta})
        libro.close()
    tabla = pd.DataFrame(filas).drop_duplicates()
    matriz = pd.crosstab([tabla["hoja"], tabla["columna"]], tabla["archivo"]).astype(bool)
    return matriz[[e for e, _ in archivos]]


def detectar_renombradas(X: pd.DataFrame, cfg: ConfigEDA) -> pd.DataFrame:
    """Pares de nombre casi igual con períodos complementarios (toda la serie)."""
    presente = X.notna()
    columnas = [c for c in X.columns if presente[c].any() and not presente[c].all()]
    candidatos = []
    for i, a in enumerate(columnas):
        for b in columnas[i + 1:]:
            if _hoja(a) != _hoja(b):
                continue
            if (presente[a] & presente[b]).sum() > cfg.max_solape_renombradas:
                continue
            sim = difflib.SequenceMatcher(None, _normalizar_nombre(a), _normalizar_nombre(b)).ratio()
            if sim >= cfg.similitud_nombres:
                candidatos.append((sim, a, b))

    usadas, filas = set(), []
    for sim, a, b in sorted(candidatos, reverse=True):
        if a in usadas or b in usadas:
            continue
        usadas.update((a, b))
        vieja, nueva = sorted((a, b), key=lambda c: X[c].last_valid_index())
        ratio = X[nueva].dropna().head(30).median() / X[vieja].dropna().tail(30).median()
        continuidad = cfg.rango_ratio_nivel[0] <= ratio <= cfg.rango_ratio_nivel[1]
        if continuidad:
            control = "Nivel continuo"
        elif nueva in cfg.renombradas_confirmadas:
            control = "Salto de nivel: el grupo confirmó que es la misma medición"
        else:
            control = "Salto de nivel: a confirmar"
        filas.append({
            "vieja": vieja, "nueva": nueva, "similitud": round(sim, 3),
            "dias_solapados": int((presente[a] & presente[b]).sum()),
            "inicio_vieja": X[vieja].first_valid_index().date(),
            "fin_vieja": X[vieja].last_valid_index().date(),
            "inicio_nueva": X[nueva].first_valid_index().date(),
            "ratio_nivel_nueva_vs_vieja": round(float(ratio), 3),
            "control": control,
            "unificar": continuidad or nueva in cfg.renombradas_confirmadas,
        })
    return pd.DataFrame(filas, columns=["vieja", "nueva", "similitud", "dias_solapados", "inicio_vieja",
                                        "fin_vieja", "inicio_nueva", "ratio_nivel_nueva_vs_vieja",
                                        "control", "unificar"])


def unificar(X: pd.DataFrame, pares: Dict[str, str]) -> pd.DataFrame:
    """Completa la columna nueva con la vieja y descarta la vieja."""
    X = X.copy()
    for vieja, nueva in pares.items():
        X[nueva] = X[nueva].combine_first(X[vieja])
    return X.drop(columns=list(pares))


# ===========================================================================
# 6. Copias exactas
# ===========================================================================
def detectar_copias(X: pd.DataFrame, cfg: ConfigEDA) -> pd.DataFrame:
    """Columnas con los mismos valores y los mismos nulos en toda la serie.

    Las columnas sin variación (por ejemplo, siempre 0) no se comparan: coinciden entre
    sí sin ser la misma medición. Las descarta el preprocesamiento como constantes en Train.
    """
    def prioridad(c):
        if c in cfg.columnas_obligatorias:
            return 0
        return 1 if any(re.search(p, c) for p in cfg.hojas_preferidas) else 2

    redondeado = X.round(cfg.decimales_copias)
    con_variacion = [c for c in X.columns if redondeado[c].nunique(dropna=True) > 1]
    huellas = {c: pd.util.hash_pandas_object(redondeado[c], index=False).sum() for c in con_variacion}
    orden = sorted(con_variacion, key=lambda c: (prioridad(c), X.columns.get_loc(c)))
    representantes: Dict[int, List[str]] = {}
    filas = []
    for c in orden:
        original = next((r for r in representantes.get(huellas[c], [])
                         if redondeado[r].equals(redondeado[c])), None)
        if original is None:
            representantes.setdefault(huellas[c], []).append(c)
        elif X[c].notna().any():
            filas.append({"columna": c, "copia_de": original,
                          "hoja_descartada": _hoja(c), "hoja_conservada": _hoja(original)})
    return pd.DataFrame(filas, columns=["columna", "copia_de", "hoja_descartada", "hoja_conservada"])


# ===========================================================================
# Decisiones
# ===========================================================================
def decidir_columnas(datos: Dict[str, object], cfg: ConfigEDA) -> Dict[str, object]:
    """Aplica las seis reglas en orden y devuelve la tabla de decisiones y sus evidencias."""
    X: pd.DataFrame = datos["diario"]
    horario: pd.DataFrame = datos["horario"]
    columnas = list(X.columns)
    decision: Dict[str, Tuple[str, str]] = {}  # columna -> (categoría, motivo)

    for c, motivo in columnas_metadatos(columnas, cfg).items():
        decision[c] = ("metadatos", motivo)

    identidad = verificar_identidad_frio(X, horario, cfg)
    if not (identidad["r2"] > 0.999 and np.allclose(identidad["coeficientes"]["coeficiente"], 1, atol=0.02)):
        raise ValueError("Frio (Kw) ya no es la suma de sus componentes: revisar antes de descartarlos.")
    for c, grupo in componentes_frio(columnas, cfg).items():
        decision.setdefault(c, ("componentes_frio",
                                f"Componente del frío ({grupo}): Frio (Kw) es la suma exacta de 12 medidores"))

    proxies = verificar_proxies(X, cfg)
    for c, fila in proxies.iterrows():
        decision.setdefault(c, ("proxies_frio",
                                f"Contiene al frío (frío = {fila['frío / columna (mediana)']:.0%} de la columna, "
                                f"Spearman {fila['Spearman con frío de D']:.3f} con el frío de D)"))

    residuales = [c for c in cfg.columnas_residuales if c in columnas]
    calidad = calidad_acumulado(horario, residuales)
    for c in residuales:
        decision.setdefault(c, ("residuales",
                                f"Calculada por diferencia: {calidad.loc[c, '% lecturas negativas']:.1f} % de "
                                f"lecturas negativas y {calidad.loc[c, '% horas con bajada']:.1f} % de horas con bajada"))

    restantes = X[[c for c in columnas if c not in decision]]
    renombradas = detectar_renombradas(restantes, cfg)
    pares = {f["vieja"]: f["nueva"] for _, f in renombradas.iterrows() if f["unificar"]}
    for _, f in renombradas.iterrows():
        if f["unificar"]:
            decision[f["vieja"]] = ("renombradas", f"Misma medición que '{f['nueva']}' ({f['control']}); "
                                                   f"se une a ella")

    unificado = unificar(restantes, pares)
    copias = detectar_copias(unificado, cfg)
    for _, f in copias.iterrows():
        decision[f["columna"]] = ("copias", f"Copia exacta de '{f['copia_de']}' en toda la serie")

    tabla = pd.DataFrame([{
        "columna": c,
        "hoja": _hoja(c),
        "decision": ("unificar" if decision.get(c, ("",))[0] == "renombradas"
                     else "descartar" if c in decision else "conservar"),
        "categoria": decision.get(c, ("", ""))[0],
        "motivo": decision.get(c, ("", ""))[1],
        "unida_en": pares.get(c, ""),
    } for c in columnas])
    return {"tabla": tabla, "pares": pares, "identidad": identidad, "proxies": proxies,
            "calidad_residuales": calidad, "renombradas": renombradas, "copias": copias}


def ejecutar_eda(cfg: Optional[ConfigEDA] = None, datos: Optional[Dict[str, object]] = None,
                 mostrar: bool = True) -> Dict[str, object]:
    """Calcula las decisiones, la tabla con el motivo de cada columna y su resumen.
    No escribe archivos: el preprocesamiento las recalcula con `cargar_decisiones`."""
    cfg = cfg or ConfigEDA()
    datos = datos or cargar_datos_eda()
    res = decidir_columnas(datos, cfg)
    tabla = res["tabla"]

    identidad = {k: v for k, v in res["identidad"].items() if k not in ("coeficientes", "columnas")}
    identidad["coeficientes"] = res["identidad"]["coeficientes"]["coeficiente"].round(4).to_dict()
    resumen = {
        "config": asdict(cfg),
        "entrada": {"archivo": CLAVE_ENTRADA, "sha256": datos["sha256"],
                    "dias": int(len(datos["diario"])), "columnas": int(datos["diario"].shape[1])},
        "identidad_frio": identidad,
        "conteo_por_categoria": tabla.loc[tabla["categoria"] != "", "categoria"].value_counts().to_dict(),
        "columnas_salida": int((tabla["decision"] == "conservar").sum()),
        "descartar": tabla.loc[tabla["decision"] == "descartar", "columna"].tolist(),
        "unificar": res["pares"],
    }
    if not mostrar:
        return {**res, "resumen": resumen, "datos": datos}

    print("=" * 78)
    print("EDA: DECISIONES ESTRUCTURALES SOBRE LAS COLUMNAS")
    print("=" * 78)
    print(f"Entrada        : {resumen['entrada']['dias']} días x {resumen['entrada']['columnas']} columnas")
    print(f"Identidad frío : R² = {identidad['r2']:.6f} | {identidad['pct_horas_dif_menor_tolerancia']:.1%} "
          f"de las horas con |frío − suma| <= {cfg.tolerancia_identidad_kw:.0f} kW")
    for categoria, n in resumen["conteo_por_categoria"].items():
        print(f"  {categoria:<18}: {n}")
    print(f"Columnas que pasan al preprocesamiento: {resumen['columnas_salida']}")
    print("=" * 78)
    return {**res, "resumen": resumen, "datos": datos}


def cargar_decisiones(checksums_path: Path = af.DEFAULT_CHECKSUMS_JSON) -> Dict[str, object]:
    """Decisiones de la EDA sobre el dataset_trabajo.csv vigente (verifica su SHA-256):
    el resumen de `ejecutar_eda` más la tabla con el motivo de cada columna.

    Se recalculan en memoria en vez de leerse de un archivo: son deterministas, no
    dependen de `y` y tardan unos segundos."""
    eda = ejecutar_eda(datos=cargar_datos_eda(checksums_path=checksums_path), mostrar=False)
    return {**eda["resumen"], "tabla": eda["tabla"]}


# ###########################################################################
# ETAPA 2. PREPROCESAMIENTO AJUSTADO CON TRAIN
# ###########################################################################
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
    #    EDA (etapa 1) y se obtienen con cargar_decisiones().

    # 2. Duplicadas en Train: qué copia se conserva
    columnas_obligatorias: Tuple[str, ...] = (af.DEFAULT_TARGET_COL,)
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
    """Aplica las decisiones de la etapa 1 (EDA): descarta las columnas que la EDA excluye y
    completa cada columna renombrada con su nombre viejo.

    No estima nada: las decisiones dependen de la estructura de la planilla (nombres, hojas,
    identidades entre medidores), no de los datos de Train. El reporte trae la categoría y el
    motivo de cada columna, tomados de la tabla de decisiones de la EDA.
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

    `decisiones` son las de la EDA; si no se pasan, se calculan con `cargar_decisiones`."""
    decisiones = decisiones if decisiones is not None else cargar_decisiones()
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
        registrado = af._sha_registrado(json.load(f), "data/processed/dataset_trabajo.csv")
    sha = huella(input_csv)
    if registrado is None or sha != registrado:
        raise ValueError("dataset_trabajo.csv no coincide con el SHA-256 registrado en checksums.json.")

    horario, _ = af.load_cleaned_data(input_csv)
    diario, _ = af.aggregate_daily_data(horario)
    etiquetadas, _ = af.prepare_target(diario)
    *_, info = af.temporal_split(etiquetadas[[COL_FECHA, COL_FRIO, COL_Y]], *cfg.proporciones)
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


def registrar_huellas(rutas: Dict[str, Path],
                      checksums_path: Path = af.DEFAULT_CHECKSUMS_JSON) -> Dict[str, str]:
    """Agrega a data/checksums.json el SHA-256 de cada archivo (clave: ruta relativa a la raíz,
    o absoluta si el archivo está fuera del proyecto)."""
    huellas = {n: huella(r) for n, r in rutas.items()}
    with open(checksums_path, encoding="utf-8") as f:
        checksums = json.load(f)
    for n, r in rutas.items():
        ruta = Path(r).resolve()
        clave = ruta.relative_to(PROJECT_ROOT) if ruta.is_relative_to(PROJECT_ROOT) else ruta
        checksums[clave.as_posix()] = huellas[n]
    with open(checksums_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(checksums, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return huellas


def ejecutar_preprocesamiento(
    cfg: Optional[Config] = None,
    omitir: Tuple[str, ...] = (),
    input_csv: Path = af.DEFAULT_INPUT_CSV,
    checksums_path: Path = af.DEFAULT_CHECKSUMS_JSON,
    salida_datos: Path = DATA_DIR / "processed",
    decisiones: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Ajusta el pipeline con Train, exporta los CSV sin escalar a `salida_datos` y registra
    sus SHA-256 en checksums.json. Devuelve el pipeline (con el reporte de cada paso), los
    datos, la verificación de causalidad, el impacto por paso, el LOF y un resumen."""
    cfg = cfg or Config()
    X, y, cortes, sha_entrada = cargar_datos_diarios(Path(input_csv), Path(checksums_path), cfg)
    fin_train = cortes["train"][1]

    if decisiones is None:
        decisiones = cargar_decisiones(checksums_path=Path(checksums_path))
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

    # ---------------- Exportación (solo los datos; los reportes quedan en `pipeline`) ----------------
    salida_datos = Path(salida_datos)
    salida_datos.mkdir(parents=True, exist_ok=True)

    def guardar(df: pd.DataFrame, ruta: Path) -> Path:
        df.to_csv(ruta, index=True, encoding="utf-8", lineterminator="\n", date_format="%Y-%m-%d")
        return ruta

    exportados = {"historial_completo.csv": guardar(datos, salida_datos / "historial_completo.csv")}
    for nombre in ("train", "validation", "test"):
        parte = datos[datos["particion"] == nombre].drop(columns=["particion"])
        if nombre == "train":
            parte = parte.drop(index=excluidos)
        exportados[f"{nombre}.csv"] = guardar(parte, salida_datos / f"{nombre}.csv")
    huellas = registrar_huellas(exportados, Path(checksums_path))

    resumen = {
        "semilla": SEMILLA,
        "config": asdict(cfg),
        "pasos_omitidos": list(omitir),
        "entrada": {"archivo": str(Path(input_csv).relative_to(PROJECT_ROOT).as_posix()),
                    "sha256": sha_entrada, "dias": len(X), "columnas": X.shape[1]},
        "decisiones_eda": {"columnas_descartadas": len(decisiones["descartar"]),
                           "columnas_unificadas": len(decisiones["unificar"])},
        "cortes": {p: [str(a.date()), str(b.date())] for p, (a, b) in cortes.items()},
        "fin_ajuste": str(fin_train.date()),
        "pasos": {nombre: paso.resumen() for nombre, paso in pipeline.steps},
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
        "salida": {"columnas_predictoras": X_proc.shape[1], "sha256": huellas},
    }

    _imprimir_resumen(resumen, pipeline)
    return {"datos": datos, "pipeline": pipeline, "lof": lof, "anomalias": anomalias,
            "causalidad": causalidad, "impacto": impacto, "resumen": resumen}


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


# ###########################################################################
# ETAPA 3. VARIABLES CREADAS Y SELECCIÓN DE VARIABLES
# ###########################################################################
COL_BODEGA = "Hl Producido Bodega_Consolidado Produccion"
COL_ENVASADA = "Hl Cerveza Envasada_Consolidado Produccion"
COL_MOSTO = "Hl de Mosto_Consolidado Produccion"  # entrada a fermentación
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
CLAVE_HISTORIAL = "data/processed/historial_completo.csv"


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
    # Para recrearlos habría que conservar esas columnas en la etapa 1 (EDA) y usar, por ejemplo,
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
    # (comparar_variantes() de src/auxiliar_functions.py): quita ventanas del frío útiles y el ciclo anual.
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
                        checksums_path: Path = af.DEFAULT_CHECKSUMS_JSON) -> pd.DataFrame:
    """Lee el historial preprocesado y verifica su SHA-256 contra data/checksums.json."""
    ruta = Path(carpeta) / "historial_completo.csv"
    with open(checksums_path, encoding="utf-8") as f:
        esperado = af._sha_registrado(json.load(f), CLAVE_HISTORIAL)
    if huella(ruta) != esperado:
        raise ValueError("historial_completo.csv no coincide con el SHA-256 registrado en checksums.json. "
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


def calcular_seleccion(historial: pd.DataFrame, cfg: Optional[ConfigSeleccion] = None,
                       metodos: Optional[List[str]] = None) -> Dict[str, object]:
    """Etapa 3 completa sobre `historial`, sin imprimir ni escribir archivos: candidatas,
    filtro de redundancia, métodos, subconjuntos y comparación en Validation.

    Es determinista (semilla fija), así que `src/train_model.py` la recalcula para obtener
    los subconjuntos en vez de leerlos de un archivo."""
    cfg = cfg or ConfigSeleccion()
    metodos = metodos or list(METODOS)
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

    por_familia = pd.Series([familia_de(c) for c in X_tr.columns]).value_counts()
    elegidas_por_familia = {m: pd.Series([familia_de(c) for c in r.columnas], dtype=object)
                            .value_counts().to_dict() for m, r in resultados.items()}

    resumen = {
        "semilla": SEMILLA,
        "config": asdict(cfg),
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
    return {"resultados": resultados, "comparacion": tabla_final, "votos": votos,
            "subconjuntos": subconjuntos, "ganador": ganador, "alpha": alpha,
            "candidatas": candidatas, "creadas": creadas, "redundancia": redundancia,
            "X_train": X_tr, "y_train": y_tr, "X_val": X_val, "y_val": y_val, "resumen": resumen}


def ejecutar_seleccion(cfg: Optional[ConfigSeleccion] = None,
                       metodos: Optional[List[str]] = None) -> Dict[str, object]:
    """Etapa 3 sobre historial_completo.csv (SHA-256 verificado): calcula e imprime la selección."""
    sel = calcular_seleccion(cargar_preprocesado(), cfg, metodos)
    resumen = sel["resumen"]
    print("=" * 78)
    print("SELECCIÓN DE VARIABLES")
    print("=" * 78)
    print(f"Candidatas: {resumen['candidatas']} -> {resumen['candidatas_tras_filtro_redundancia']} tras el "
          f"filtro de redundancia | Train: {len(sel['X_train'])} filas | Validation: {len(sel['X_val'])} filas")
    print("  " + " | ".join(f"{f}: {n}" for f, n in resumen["candidatas_tras_filtro_por_familia"].items()))
    print(f"Alpha de Ridge (validación temporal interna): {sel['alpha']}")
    for m, r in sel["resultados"].items():
        print(f"  {m:<20}: {len(r.columnas):>3} variables")
    print(sel["comparacion"].round(1).to_string(index=False))
    print(f"Ganador: {sel['ganador']}")
    print("=" * 78)
    return sel


# ###########################################################################
# Línea de comandos
# ###########################################################################
def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 2: decisiones de la EDA, preprocesamiento y selección.")
    parser.add_argument("--etapas", nargs="*", choices=list(ETAPAS), default=list(ETAPAS),
                        help="Etapas a correr, en orden (por defecto, las tres).")
    parser.add_argument("--completo", action="store_true",
                        help="Corre antes la limpieza horaria de los Excel (verificando sus SHA-256).")
    parser.add_argument("--omitir", nargs="*", default=[], choices=list(NOMBRES_PASOS),
                        help="Pasos del preprocesamiento a omitir.")
    parser.add_argument("--metodo-outliers", choices=list(DETECTORES_UNIVARIADOS), default="iqr")
    parser.add_argument("--tratamiento-outliers", choices=["winsorizar", "nan", "ninguno"],
                        default="winsorizar")
    parser.add_argument("--max-nulos-train", type=float, default=0.30)
    parser.add_argument("--tratamiento-lof", choices=["reportar", "excluir_train"], default="reportar")
    parser.add_argument("--metodos", nargs="*", choices=list(METODOS), default=list(METODOS),
                        help="Métodos de selección de variables.")
    args = parser.parse_args()
    if args.completo:
        af.run_cleaning_pipeline()
    decisiones = None
    if "eda" in args.etapas:
        eda = ejecutar_eda()
        decisiones = {**eda["resumen"], "tabla": eda["tabla"]}  # el preprocesamiento no las recalcula
    if "preprocesamiento" in args.etapas:
        cfg = Config(metodo_outliers=args.metodo_outliers,
                     tratamiento_outliers=args.tratamiento_outliers,
                     max_nulos_train=args.max_nulos_train,
                     tratamiento_lof=args.tratamiento_lof)
        ejecutar_preprocesamiento(cfg, omitir=tuple(args.omitir), decisiones=decisiones)
    if "seleccion" in args.etapas:
        ejecutar_seleccion(metodos=args.metodos)


if __name__ == "__main__":
    main()
