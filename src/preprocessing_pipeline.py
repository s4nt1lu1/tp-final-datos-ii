"""Pipeline reproducible de la Fase 2.

Etapa 1 (Parte A): limpieza horaria de los Excel crudos -> dataset_trabajo.csv.
Partes B a E: limpieza de predictoras, features de planta, creación y selección
de variables (transformadores de sklearn ajustados solo en Train).
Etapa 2 (Parte F): alinea D+1, fija la división cronológica, ajusta la
preparación en Train y compara los subconjuntos en Validation antes de aplicar
el resultado al resto de la serie.

Uso desde la raíz del proyecto:
    python -m src.preprocessing_pipeline
    python -m src.preprocessing_pipeline --solo-limpieza
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
# Al ejecutar el .py directamente, Python empieza a buscar módulos en src/.
# Agregamos la raíz para que funcionen los mismos imports que con python -m.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.auxiliar_functions import (
    DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SEMILLA,
    huella,
    verificar_checksums,
)

PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_INPUT_CSV = PROCESSED_DIR / "dataset_trabajo.csv"
DEFAULT_CHECKSUMS_JSON = DATA_DIR / "checksums.json"
DEFAULT_LINEAGE_JSON = PROCESSED_DIR / "data_lineage.json"

DEFAULT_DATE_COL = "DIA"
DEFAULT_TIME_COL = "HORA"
DEFAULT_CLOSING_TIME = "23:59"
DEFAULT_TARGET_COL = "Frio (Kw)_Consolidado EE"
DEFAULT_TARGET_NAME = "y"


# ===========================================================================
# PARTE A. Limpieza horaria (antes src/correccion_fechas.py)
# Excel crudos -> data/processed/dataset_trabajo.csv
# ===========================================================================
HORAS_CANONICAS: List[str] = [f"{h:02d}:00" for h in range(24)] + ["23:59"]
HORAS_REGULARES: List[str] = HORAS_CANONICAS[:-1]  # '00:00' a '23:00'
COL_TARGET = "Frio (Kw)_Consolidado EE"
TOL = 1e-3
UMBRAL_MAD = 3.5

# Reglas estrictas de Etapa 5
MIN_HORAS_DIA = 24
MAX_FRIO_00HS = 100.0
UMBRAL_SALTO_P999 = 5153.0
EXIGIR_AL_MENOS_UN_CERO_FRIO = True
# Primer año de la serie. La calibración termina antes del corte de Train.
# src/preprocessing_pipeline.py (líneas 71-80)
# Primer año de la serie. La calibración termina antes del corte de Train.
FIN_CALIBRACION = "2021-06-30"


def get_default_excel_sources(raw_dir: Path = RAW_DATA_DIR) -> List[Tuple[str, Path]]:
    """Devuelve los archivos Excel ordenados estrictamente del más nuevo al más viejo."""
    return [
        ("2022_2023", raw_dir / "Totalizadores Planta 2022_2023.xlsx"),
        ("2021_2023", raw_dir / "Totalizadores Planta - 2021_2023.xlsx"),
        ("2020_2022", raw_dir / "Totalizadores Planta 2020_2022.xlsx"),
    ]


def normalizar_hora_base(serie_hora: pd.Series) -> pd.Series:
    """Quita prefijos de fecha (1899-12-30, 1900-01-01), decimales .0 y normaliza HH."""
    s = serie_hora.astype(str).str.strip()
    s = s.str.replace(r"^\d{4}-\d{2}-\d{2}\s+", "", regex=True)
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"^(\d):", r"0\1:", regex=True)
    return s


def auditar_dias_repetidos_consolidado_ee(
    archivos_excel: List[Tuple[str, Path]],
) -> pd.DataFrame:
    """Etapa 1: Evalúa en 'Consolidado EE' los días con >25 registros (bloques extra)."""
    registros: List[Dict[str, Any]] = []

    for etiqueta, ruta in archivos_excel:
        if not ruta.exists():
            continue
        xls = pd.ExcelFile(ruta)
        mapa_hojas = {h.strip(): h for h in xls.sheet_names}
        if "Consolidado EE" not in mapa_hojas:
            continue

        df = pd.read_excel(xls, sheet_name=mapa_hojas["Consolidado EE"])
        df.columns = df.columns.astype(str).str.strip()
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
        df = df.dropna(subset=["DIA", "HORA"]).copy()
        df["DIA"] = pd.to_datetime(df["DIA"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["DIA"]).copy()

        s_hora = normalizar_hora_base(df["HORA"])
        mask_235959 = s_hora.str.startswith("23:59:59")
        df = df[~mask_235959].copy()
        s_hora = s_hora[~mask_235959]
        df["HORA_CLEAN"] = s_hora.str.replace(r"^(\d{2}:\d{2}):\d{2}$", r"\1", regex=True)

        cols_num = [c for c in df.columns if c not in ("DIA", "HORA", "HORA_CLEAN")]
        for c in cols_num:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        es_nuevo_bloque = (df["HORA_CLEAN"] == "00:00") & (df["DIA"] == df["DIA"].shift(1))
        df["ID_BLOQUE"] = es_nuevo_bloque.groupby(df["DIA"]).cumsum()

        conteo_dia = df.groupby("DIA").size()
        dias_mas_25 = conteo_dia[conteo_dia > 25].index

        for d in dias_mas_25:
            sub = df[df["DIA"] == d]
            n_filas_dia = len(sub)
            n_bloques = sub["ID_BLOQUE"].nunique()

            b0 = (
                sub[sub["ID_BLOQUE"] == 0]
                .drop_duplicates(subset=["HORA_CLEAN"], keep="first")
                .set_index("HORA_CLEAN")[cols_num]
            )

            for b_id in sorted(sub["ID_BLOQUE"].unique()):
                if b_id == 0:
                    continue
                bx = (
                    sub[sub["ID_BLOQUE"] == b_id]
                    .drop_duplicates(subset=["HORA_CLEAN"], keep="first")
                    .set_index("HORA_CLEAN")[cols_num]
                )

                horas_comunes = b0.index.intersection(bx.index)
                b0_al = b0.loc[horas_comunes]
                bx_al = bx.loc[horas_comunes]

                arr0 = b0_al.to_numpy(dtype=float)
                arrx = bx_al.to_numpy(dtype=float)

                es_vacio_o_cero = bool(np.all(np.isnan(arrx) | (np.abs(arrx) < 1e-6)))
                misma_mascara_nan = np.array_equal(np.isnan(arr0), np.isnan(arrx))
                iguales_num = np.allclose(arr0, arrx, atol=1e-4, rtol=1e-5, equal_nan=True)
                es_copia_exacta = bool(misma_mascara_nan and iguales_num)

                diff_abs = np.abs(np.nan_to_num(arr0, nan=0.0) - np.nan_to_num(arrx, nan=0.0))
                max_diff = float(np.max(diff_abs)) if diff_abs.size > 0 else 0.0

                frio_b0_2359 = (
                    b0_al.loc["23:59", "Frio (Kw)"]
                    if ("23:59" in b0_al.index and "Frio (Kw)" in b0_al.columns)
                    else np.nan
                )
                frio_bx_2359 = (
                    bx_al.loc["23:59", "Frio (Kw)"]
                    if ("23:59" in bx_al.index and "Frio (Kw)" in bx_al.columns)
                    else np.nan
                )

                if es_copia_exacta:
                    clasificacion = "1. Copia exacta del Bloque 0"
                elif es_vacio_o_cero:
                    clasificacion = "2. Bloque fantasma en Cero / NaN (relleno fin de mes)"
                else:
                    clasificacion = "3. Datos DISTINTOS (valores cambian respecto al Bloque 0)"

                registros.append(
                    {
                        "Archivo": etiqueta,
                        "DIA": d.strftime("%Y-%m-%d"),
                        "Filas_Totales_Dia": n_filas_dia,
                        "Total_Bloques": n_bloques,
                        "Bloque_Evaluado": b_id,
                        "Filas_Bloque": len(bx),
                        "Es_Copia_Identica": es_copia_exacta,
                        "Clasificacion": clasificacion,
                        "Frio_2359_Bloque0": frio_b0_2359,
                        "Frio_2359_BloqueX": frio_bx_2359,
                        "Max_Dif_Abs_EE": max_diff,
                    }
                )

    return pd.DataFrame(registros)


def reparar_secuencia_horaria(h_list: List[str]) -> List[str]:
    """Corrige typos en la secuencia horaria de un día (ej. 16->15->16, 15->15->17, 09->08)."""
    h = list(h_list)
    n = len(h)
    if n < 3:
        return h

    # Paso A: Sándwich de 3 elementos
    for i in range(1, n - 1):
        prev_h, curr_h, next_h = h[i - 1], h[i], h[i + 1]
        if prev_h in HORAS_REGULARES and next_h in HORAS_REGULARES:
            idx_p = HORAS_REGULARES.index(prev_h)
            idx_n = HORAS_REGULARES.index(next_h)
            if idx_n - idx_p == 2:
                esperada = HORAS_REGULARES[idx_p + 1]
                if curr_h != esperada:
                    h[i] = esperada

    # Paso B: Ancla de 4 elementos
    for i in range(1, n - 2):
        h_prev, h_a, h_b, h_post = h[i - 1], h[i], h[i + 1], h[i + 2]
        if h_prev in HORAS_REGULARES and h_post in HORAS_REGULARES:
            idx_prev = HORAS_REGULARES.index(h_prev)
            idx_post = HORAS_REGULARES.index(h_post)
            if idx_post - idx_prev == 3:
                esp_a = HORAS_REGULARES[idx_prev + 1]
                esp_b = HORAS_REGULARES[idx_prev + 2]
                if h_a != esp_a or h_b != esp_b:
                    h[i] = esp_a
                    h[i + 1] = esp_b

    # Paso C: Alineación canónica en días completos de 25 filas
    if n == 25 and h[0] == "00:00" and h[-1] == "23:59":
        coincidencias = sum(1 for a, b in zip(h, HORAS_CANONICAS) if a == b)
        if coincidencias >= 22 and h != HORAS_CANONICAS:
            h = HORAS_CANONICAS.copy()

    return h


def limpiar_y_auditar_hoja(
    df_raw: pd.DataFrame,
    archivo: str,
    hoja: str,
    log_reparaciones_horarias: List[Dict[str, Any]],
    log_filas_extra_borradas: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Etapa 2 (por hoja): limpia DIA/HORA, elimina 23:59:59 y bloques >25, y corrige typos."""
    df = df_raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
    df = df.loc[:, ~df.columns.duplicated()]

    mapa_claves: Dict[str, str] = {}
    for c in df.columns:
        if c.upper() == "DIA":
            mapa_claves[c] = "DIA"
        elif c.upper() == "HORA":
            mapa_claves[c] = "HORA"
    df = df.rename(columns=mapa_claves)

    if "DIA" not in df.columns or "HORA" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["DIA", "HORA"]).copy()
    df["DIA"] = pd.to_datetime(df["DIA"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["DIA"]).copy()

    s_hora_raw = df["HORA"].astype(str).str.strip()

    # 1. Fecha incrustada en HORA (ej. 1900-01-01 HH:MM:SS)
    mask_fecha_en_hora = s_hora_raw.str.match(r"^\d{4}-\d{2}-\d{2}\s+") & ~s_hora_raw.str.startswith(
        "1899-12-30"
    )
    if mask_fecha_en_hora.any():
        for d in df.loc[mask_fecha_en_hora, "DIA"].unique():
            sub_h = s_hora_raw[mask_fecha_en_hora & (df["DIA"] == d)]
            log_reparaciones_horarias.append(
                {
                    "DIA": d.strftime("%Y-%m-%d"),
                    "Archivo": archivo,
                    "Hoja": hoja,
                    "Tipo_Reparacion": "1. Fecha incrustada en HORA (1900-01-01)",
                    "Filas_Reparadas": int(len(sub_h)),
                    "Detalle": f"Se limpió prefijo de fecha conservando la hora. Ejemplos: {sub_h.unique().tolist()[:2]}",
                }
            )

    s_hora = normalizar_hora_base(s_hora_raw)

    # 2. Borrado de registros 23:59:59
    mask_235959 = s_hora.str.startswith("23:59:59")
    if mask_235959.any():
        for d in df.loc[mask_235959, "DIA"].unique():
            n_borradas = int((mask_235959 & (df["DIA"] == d)).sum())
            log_filas_extra_borradas.append(
                {
                    "DIA": d.strftime("%Y-%m-%d"),
                    "Archivo": archivo,
                    "Hoja": hoja,
                    "Motivo": "Registro extra 23:59:59",
                    "Filas_Borradas": n_borradas,
                }
            )
        df = df[~mask_235959].copy()
        s_hora = s_hora[~mask_235959]

    df["HORA"] = s_hora.str.replace(r"^(\d{2}:\d{2}):\d{2}$", r"\1", regex=True)

    # 3. Eliminar bloques repetidos (>25 registros por día)
    id_fila_dia = df.groupby("DIA").cumcount()
    mask_bloques_extra = id_fila_dia >= 25
    if mask_bloques_extra.any():
        for d_orig, grp_extra in df[mask_bloques_extra].groupby("DIA"):
            log_filas_extra_borradas.append(
                {
                    "DIA": d_orig.strftime("%Y-%m-%d"),
                    "Archivo": archivo,
                    "Hoja": hoja,
                    "Motivo": "Bloque repetido (>25 filas en cierre de mes < 31 días)",
                    "Filas_Borradas": int(len(grp_extra)),
                }
            )
        df = df[~mask_bloques_extra].copy()

    # 4. Corregir typos de secuencia horaria
    horas_corregidas: List[str] = []
    for d, grupo in df.groupby("DIA", sort=False):
        h_orig = grupo["HORA"].tolist()
        h_nueva = reparar_secuencia_horaria(h_orig)
        if h_nueva != h_orig:
            cambios = [
                f"pos {idx} ({o} -> {n})" for idx, (o, n) in enumerate(zip(h_orig, h_nueva)) if o != n
            ]
            log_reparaciones_horarias.append(
                {
                    "DIA": d.strftime("%Y-%m-%d"),
                    "Archivo": archivo,
                    "Hoja": hoja,
                    "Tipo_Reparacion": "2. Typo en secuencia horaria",
                    "Filas_Reparadas": int(len(cambios)),
                    "Detalle": "; ".join(cambios),
                }
            )
        horas_corregidas.extend(h_nueva)

    df["HORA"] = horas_corregidas
    df = df.drop_duplicates(subset=["DIA", "HORA"], keep="first")
    return df


def unificar_archivos_excel(
    archivos_excel: List[Tuple[str, Path]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Etapa 2: Une horizontalmente todas las hojas y desempata priorizando el archivo más nuevo."""
    log_reparaciones_horarias: List[Dict[str, Any]] = []
    log_filas_extra_borradas: List[Dict[str, Any]] = []
    log_solapamiento_archivos: List[Dict[str, Any]] = []

    dfs_por_archivo: List[pd.DataFrame] = []
    dias_ya_asignados = set()

    for etiqueta, ruta in archivos_excel:
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró el archivo Excel requerido: {ruta}")

        xls = pd.ExcelFile(ruta)
        hojas_limpias = []

        for hoja in xls.sheet_names:
            hoja_clean = hoja.strip()
            df_raw = pd.read_excel(xls, sheet_name=hoja)
            df_h = limpiar_y_auditar_hoja(
                df_raw,
                etiqueta,
                hoja_clean,
                log_reparaciones_horarias,
                log_filas_extra_borradas,
            )

            if not df_h.empty:
                renombres = {
                    c: f"{c}_{hoja_clean}" for c in df_h.columns if c not in ("DIA", "HORA")
                }
                df_h = df_h.rename(columns=renombres).set_index(["DIA", "HORA"])
                hojas_limpias.append(df_h)

        if not hojas_limpias:
            continue

        df_arch = pd.concat(hojas_limpias, axis=1, join="outer").reset_index()
        df_arch["_archivo_origen"] = etiqueta

        dias_en_archivo = set(df_arch["DIA"].unique())
        dias_solapados = sorted(dias_en_archivo.intersection(dias_ya_asignados))
        for d_sol in dias_solapados:
            log_solapamiento_archivos.append(
                {
                    "DIA": d_sol.strftime("%Y-%m-%d"),
                    "Archivo_Descartado": etiqueta,
                    "Motivo": "Día ya presente en un archivo Excel más nuevo",
                }
            )

        df_arch_nuevo = df_arch[~df_arch["DIA"].isin(dias_ya_asignados)].copy()
        dias_ya_asignados.update(df_arch_nuevo["DIA"].unique())
        dfs_por_archivo.append(df_arch_nuevo)

    if not dfs_por_archivo:
        raise ValueError("No se pudieron cargar datos válidos desde los archivos Excel.")

    df_unificado = (
        pd.concat(dfs_por_archivo, ignore_index=True)
        .drop_duplicates(subset=["DIA", "HORA"], keep="first")
        .sort_values(["DIA", "HORA"])
        .reset_index(drop=True)
    )

    cols_datos = [c for c in df_unificado.columns if c not in ("DIA", "HORA", "_archivo_origen")]
    for c in cols_datos:
        df_unificado[c] = pd.to_numeric(df_unificado[c], errors="coerce")

    return (
        df_unificado,
        pd.DataFrame(log_reparaciones_horarias),
        pd.DataFrame(log_filas_extra_borradas),
        pd.DataFrame(log_solapamiento_archivos),
    )


def calcular_parametros_mad_2359(
    df_horario: pd.DataFrame,
    columnas: List[str],
    umbral_z: float = UMBRAL_MAD,
) -> pd.DataFrame:
    """Etapa 3: Calcula Mediana, MAD y límites de aceptación sobre cierres 23:59."""
    df_cierre = df_horario[df_horario["HORA"] == "23:59"]
    params: Dict[str, Dict[str, Any]] = {}

    for col in columnas:
        s = df_cierre[col].dropna()
        s_pos = s[s > 0]
        if len(s_pos) == 0:
            params[col] = {
                "Mediana_2359": np.nan,
                "MAD_2359": np.nan,
                "MAD_Escalado": np.nan,
                "Lim_Inf_MAD": np.nan,
                "Lim_Sup_MAD": np.nan,
                "N_Cierres_Positivos": 0,
            }
            continue

        mediana = float(s_pos.median())
        mad = float(np.median(np.abs(s_pos - mediana)))
        mad_esc = 1.4826 * mad if mad > 0 else float(s_pos.std())
        lim_inf = max(0.0, mediana - umbral_z * mad_esc)
        lim_sup = mediana + umbral_z * mad_esc

        params[col] = {
            "Mediana_2359": mediana,
            "MAD_2359": mad,
            "MAD_Escalado": mad_esc,
            "Lim_Inf_MAD": lim_inf,
            "Lim_Sup_MAD": lim_sup,
            "N_Cierres_Positivos": int(len(s_pos)),
        }

    return pd.DataFrame.from_dict(params, orient="index")


def evaluar_valor_mad(
    valor: float,
    fila_mad: pd.Series,
    umbral_z: float = UMBRAL_MAD,
) -> Tuple[float, bool, str]:
    """Evalúa un valor de cierre 23:59 contra los parámetros MAD de su variable."""
    if pd.isna(valor) or valor <= 0:
        return np.nan, False, "Rechazado (23:59 nulo o <= 0)"
    mad_esc = fila_mad["MAD_Escalado"]
    mediana = fila_mad["Mediana_2359"]
    if pd.isna(mad_esc) or mad_esc <= 0:
        return np.nan, False, "Sin referencia MAD"
    z_mad = abs(valor - mediana) / mad_esc
    pasa = bool(z_mad <= umbral_z)
    estado = (
        f"OK MAD (Z={z_mad:.2f} <= {umbral_z})"
        if pasa
        else f"Outlier MAD (Z={z_mad:.2f} > {umbral_z})"
    )
    return float(z_mad), pasa, estado


def evaluar_casos_especiales_y_reparar_pico_01(
    df_unificado: pd.DataFrame,
    df_mad_ee: pd.DataFrame,
    target_col: str = COL_TARGET,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Etapa 4: Evalúa con MAD inicios != 00:00 y picos gigantes a la 01:00, reparando estos últimos."""
    df_trabajo = df_unificado.copy()
    ref_mad_frio = df_mad_ee.loc[target_col]
    casos_inicio_y_pico01: List[Dict[str, Any]] = []

    for dia, grupo in df_trabajo.groupby("DIA", sort=True):
        grupo = grupo.sort_values("HORA")
        horas = grupo["HORA"].tolist()
        archivo_origen = grupo["_archivo_origen"].iloc[-1]
        primera_hora = horas[0] if len(horas) > 0 else "N/A"
        tiene_2359 = "23:59" in horas

        s_frio = grupo.set_index("HORA")[target_col].dropna()
        val_00 = s_frio.get("00:00", np.nan)
        val_01 = s_frio.get("01:00", np.nan)
        val_02 = s_frio.get("02:00", np.nan)
        val_2359 = s_frio.get("23:59", np.nan)

        no_arranca_00_filas = primera_hora != "00:00"
        horas_positivas = s_frio[s_frio > 0].index.tolist()
        primera_hora_positiva = horas_positivas[0] if len(horas_positivas) > 0 else "Sin datos >0"
        arranque_tardio_frio = (
            not no_arranca_00_filas
            and pd.notna(val_00)
            and val_00 == 0
            and pd.notna(val_01)
            and val_01 == 0
            and len(horas_positivas) > 0
            and primera_hora_positiva not in ("00:00", "01:00")
        )
        pico_gigante_01 = pd.notna(val_01) and pd.notna(val_02) and (val_01 > val_02 + TOL)
        sin_cero_frio = bool(len(s_frio) > 0 and (s_frio.abs() > TOL).all())

        if no_arranca_00_filas or arranque_tardio_frio or pico_gigante_01 or sin_cero_frio:
            z_mad, pasa_mad, estado_mad = evaluar_valor_mad(val_2359, ref_mad_frio)
            s_desde_02 = s_frio.loc[s_frio.index >= "02:00"]
            monotono_desde_02 = (
                bool((s_desde_02.diff().dropna() >= -TOL).all()) if len(s_desde_02) >= 2 else False
            )

            etiquetas_caso = []
            if no_arranca_00_filas:
                etiquetas_caso.append(f"No arranca a las 00:00 (inicia {primera_hora})")
            if arranque_tardio_frio:
                etiquetas_caso.append(
                    f"Frio (Kw) inicia acumulación a las {primera_hora_positiva}"
                )
            if pico_gigante_01:
                etiquetas_caso.append(f"Dato gigante a la 01:00 ({val_01:,.1f} kW)")
            if sin_cero_frio:
                etiquetas_caso.append(
                    f"Sin reinicio: ningún 0 en Frio (Kw) (mín: {s_frio.min():,.1f} kW)"
                )

            if not tiene_2359:
                accion = "Eliminar día (sin registro a las 23:59)"
            elif not pasa_mad:
                accion = f"Eliminar día (cierre 23:59 rechazado por MAD: {estado_mad})"
            elif pico_gigante_01 and monotono_desde_02:
                idx_01 = grupo[grupo["HORA"] == "01:00"].index
                nuevo_01 = val_00 if (pd.notna(val_00) and val_00 <= val_02) else 0.0
                df_trabajo.loc[idx_01, target_col] = nuevo_01
                accion = f"Pico 01:00 reparado a {nuevo_01:.1f} kW y día conservado (MAD 23:59 OK)"
            elif pico_gigante_01:
                accion = "Conservar (pico 01:00 con caída adicional, cierre 23:59 OK por MAD)"
            elif monotono_desde_02:
                accion = "Conservar (23:59 validado por MAD y curva acumulativa)"
            else:
                accion = "Conservar (23:59 validado por MAD pese a caída acumulativa)"

            casos_inicio_y_pico01.append(
                {
                    "DIA": dia.strftime("%Y-%m-%d"),
                    "Archivo_Origen": archivo_origen,
                    "Caso_Detectado": " + ".join(etiquetas_caso),
                    "Primera_Hora_Tabla": primera_hora,
                    "Primera_Hora_Frio_Positivo": primera_hora_positiva,
                    "Frio_00hs": val_00,
                    "Frio_01hs_Original": val_01,
                    "Frio_02hs": val_02,
                    "Frio_2359hs": val_2359,
                    "Mediana_Ref_2359": ref_mad_frio["Mediana_2359"],
                    "Z_Score_MAD_2359": z_mad,
                    "Pasa_MAD_2359": pasa_mad,
                    "Monotono_02_a_2359": monotono_desde_02,
                    "Decision": accion,
                }
            )

    return df_trabajo, pd.DataFrame(casos_inicio_y_pico01)


def filtrar_dias_invalidos(
    df_trabajo: pd.DataFrame,
    df_mad_ee: pd.DataFrame,
    df_casos_especiales_mad: pd.DataFrame,
    target_col: str = COL_TARGET,
    umbral_salto: float = UMBRAL_SALTO_P999,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Etapa 5: Aplica filtros estructurales, de reinicio, fallas de medidor y MAD selectivo."""
    dias_borrados: List[Dict[str, Any]] = []
    dias_anomalos_conservados_mad: List[Dict[str, Any]] = []
    ref_mad_frio = df_mad_ee.loc[target_col]

    dias_pico01_reparados = set()
    if not df_casos_especiales_mad.empty:
        mask_rep = df_casos_especiales_mad["Decision"].str.contains(
            "reparado", case=False, na=False
        )
        dias_pico01_reparados = set(df_casos_especiales_mad.loc[mask_rep, "DIA"].tolist())

    for dia, grupo in df_trabajo.groupby("DIA", sort=True):
        dia_str = dia.strftime("%Y-%m-%d")
        grupo = grupo.sort_values("HORA")
        archivo_origen = grupo["_archivo_origen"].iloc[-1]
        horas = grupo["HORA"].tolist()
        n_horas = len(horas)
        primera_hora = horas[0] if n_horas > 0 else "N/A"
        tiene_2359 = "23:59" in horas

        s_frio = grupo.set_index("HORA")[target_col].dropna()
        val_00 = s_frio.get("00:00", np.nan)
        val_01 = s_frio.get("01:00", np.nan)
        val_2359 = s_frio.get("23:59", np.nan)

        tiene_al_menos_un_cero = bool(len(s_frio) > 0 and (s_frio.abs() <= TOL).any())
        min_frio_dia = float(s_frio.min()) if len(s_frio) > 0 else np.nan

        if not tiene_2359:
            dias_borrados.append(
                {
                    "DIA": dia_str,
                    "Archivo_Origen": archivo_origen,
                    "Etapa_Filtro": "1. Menor cantidad de horas (sin 23:59)",
                    "Filas_Dia": n_horas,
                    "Frio_2359_kW": val_2359,
                    "Z_Score_MAD_2359": np.nan,
                    "Motivo_Detallado": f"Día incompleto ({n_horas} hs): carece del registro de cierre a las 23:59",
                }
            )
            continue

        if pd.isna(val_2359) or not (0 < val_2359 <= 100_000):
            dias_borrados.append(
                {
                    "DIA": dia_str,
                    "Archivo_Origen": archivo_origen,
                    "Etapa_Filtro": "1. Sin dato válido en Frio (Kw) a las 23:59",
                    "Filas_Dia": n_horas,
                    "Frio_2359_kW": val_2359,
                    "Z_Score_MAD_2359": np.nan,
                    "Motivo_Detallado": f"Frio (Kw) a las 23:59 es NaN o fuera de rango físico (valor={val_2359})",
                }
            )
            continue

        n_lecturas_frio = len(s_frio)
        if n_horas < MIN_HORAS_DIA or n_lecturas_frio < MIN_HORAS_DIA:
            dias_borrados.append(
                {
                    "DIA": dia_str,
                    "Archivo_Origen": archivo_origen,
                    "Etapa_Filtro": "1. Menor cantidad de horas (día incompleto < 24 hs)",
                    "Filas_Dia": n_lecturas_frio,
                    "Frio_2359_kW": val_2359,
                    "Z_Score_MAD_2359": np.nan,
                    "Motivo_Detallado": f"El día tiene {n_lecturas_frio} lecturas válidas en Frio (Kw) (< {MIN_HORAS_DIA} requeridas)",
                }
            )
            continue

        if (
            pd.isna(val_00)
            or val_00 > MAX_FRIO_00HS
            or (EXIGIR_AL_MENOS_UN_CERO_FRIO and not tiene_al_menos_un_cero)
        ):
            z_mad, _, _ = evaluar_valor_mad(val_2359, ref_mad_frio)
            dias_borrados.append(
                {
                    "DIA": dia_str,
                    "Archivo_Origen": archivo_origen,
                    "Etapa_Filtro": "2. Sin reinicio del totalizador a medianoche (00:00)",
                    "Filas_Dia": n_horas,
                    "Frio_2359_kW": val_2359,
                    "Z_Score_MAD_2359": z_mad,
                    "Motivo_Detallado": f"El totalizador no reinició a las 00:00 (00:00={val_00} kW, mínimo={min_frio_dia:,.1f} kW)",
                }
            )
            continue

        diffs_frio = s_frio.diff().dropna()
        horas_congeladas = int((diffs_frio.abs() <= 10).sum())
        if horas_congeladas >= 10 and (diffs_frio < 0).any():
            z_mad, _, _ = evaluar_valor_mad(val_2359, ref_mad_frio)
            dias_borrados.append(
                {
                    "DIA": dia_str,
                    "Archivo_Origen": archivo_origen,
                    "Etapa_Filtro": "3. Falla de medidor (contador congelado)",
                    "Filas_Dia": n_horas,
                    "Frio_2359_kW": val_2359,
                    "Z_Score_MAD_2359": z_mad,
                    "Motivo_Detallado": f"Medidor congelado durante {horas_congeladas} hs (|inc|<=10 kW) con retroceso (cierre={val_2359:,.1f} kW)",
                }
            )
            continue

        if val_2359 > 40_000 and len(diffs_frio) > 0 and diffs_frio.max() > umbral_salto:
            z_mad, _, _ = evaluar_valor_mad(val_2359, ref_mad_frio)
            dias_borrados.append(
                {
                    "DIA": dia_str,
                    "Archivo_Origen": archivo_origen,
                    "Etapa_Filtro": "3. Falla de medidor (salto horario alto y cierre > 40.000 kW)",
                    "Filas_Dia": n_horas,
                    "Frio_2359_kW": val_2359,
                    "Z_Score_MAD_2359": z_mad,
                    "Motivo_Detallado": f"Salto de {diffs_frio.max():,.1f} kW (> P99,9={umbral_salto:,.0f} kW) y cierre de {val_2359:,.1f} kW",
                }
            )
            continue

        caidas = diffs_frio[diffs_frio < -TOL]
        hay_violacion_acumulativa = len(caidas) > 0
        no_arranca_00 = primera_hora != "00:00"
        horas_pos = s_frio[s_frio > 0].index.tolist()
        primera_hora_pos = horas_pos[0] if len(horas_pos) > 0 else "Sin datos >0"
        arranque_tardio = (
            not no_arranca_00
            and pd.notna(val_00)
            and val_00 == 0
            and pd.notna(val_01)
            and val_01 == 0
            and len(horas_pos) > 0
            and primera_hora_pos not in ("00:00", "01:00")
        )
        fue_reparado_01 = dia_str in dias_pico01_reparados

        motivos_anomalia = []
        if no_arranca_00:
            motivos_anomalia.append(f"No arranca a las 00:00 (inicia {primera_hora})")
        if arranque_tardio:
            motivos_anomalia.append(
                f"Arranque tardío en Frio (Kw) (acumula desde {primera_hora_pos})"
            )
        if fue_reparado_01:
            motivos_anomalia.append("Pico gigante a la 01:00 reparado en Etapa 4")
        if not tiene_al_menos_un_cero:
            motivos_anomalia.append(f"Sin valor 0 en Frio (Kw) (mín={min_frio_dia:,.1f} kW)")
        if hay_violacion_acumulativa:
            h_peor = caidas.idxmin()
            mag_peor = float(caidas.min())
            motivos_anomalia.append(
                f"Violación acumulativa: caída de {mag_peor:,.1f} kW a las {h_peor}"
            )

        if motivos_anomalia:
            z_mad, pasa_mad, estado_mad = evaluar_valor_mad(val_2359, ref_mad_frio)
            if pasa_mad:
                dias_anomalos_conservados_mad.append(
                    {
                        "DIA": dia_str,
                        "Archivo_Origen": archivo_origen,
                        "Anomalia_Detectada": " | ".join(motivos_anomalia),
                        "Primera_Hora": primera_hora,
                        "Filas_Dia": n_horas,
                        "Frio_2359_kW": val_2359,
                        "Z_Score_MAD_2359": z_mad,
                        "Estado_MAD_2359": estado_mad,
                        "Decision": "CONSERVADO (violación/anomalía con cierre 23:59 OK según MAD)",
                    }
                )
            else:
                dias_borrados.append(
                    {
                        "DIA": dia_str,
                        "Archivo_Origen": archivo_origen,
                        "Etapa_Filtro": "3. Anomalía acumulativa/inicio que falla MAD a las 23:59",
                        "Filas_Dia": n_horas,
                        "Frio_2359_kW": val_2359,
                        "Z_Score_MAD_2359": z_mad,
                        "Motivo_Detallado": f"{' | '.join(motivos_anomalia)} -> Rechazado por MAD ({estado_mad})",
                    }
                )

    df_dias_borrados = pd.DataFrame(dias_borrados)
    df_anomalos_conservados = pd.DataFrame(dias_anomalos_conservados_mad)

    set_borrados = set(df_dias_borrados["DIA"].tolist()) if not df_dias_borrados.empty else set()
    mask_borrar = df_trabajo["DIA"].dt.strftime("%Y-%m-%d").isin(set_borrados)
    df_dataset_final = df_trabajo[~mask_borrar].copy().reset_index(drop=True)

    return df_dataset_final, df_dias_borrados, df_anomalos_conservados


def exportar_y_versionar(
    df_dataset_final: pd.DataFrame,
    df_dias_borrados: pd.DataFrame,
    data_dir: Path = DATA_DIR,
    project_root: Path = PROJECT_ROOT,
    expected_checksum: Optional[str] = None,
    update_checksums_file: bool = True,
) -> Dict[str, Any]:
    """Etapa 7: Exporta `dataset_trabajo.csv`, compara SHA-256 previo y actualiza `checksums.json`."""
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    ruta_salida_csv = processed_dir / "dataset_trabajo.csv"
    ruta_borrados_csv = processed_dir / "tabla_dias_borrados.csv"

    cols_aux = [c for c in ["_archivo_origen"] if c in df_dataset_final.columns]
    df_export_csv = df_dataset_final.drop(columns=cols_aux).copy()
    df_export_csv["DIA"] = pd.to_datetime(df_export_csv["DIA"]).dt.strftime("%Y-%m-%d")
    df_export_csv.to_csv(ruta_salida_csv, index=False, encoding="utf-8", lineterminator="\n")

    if not df_dias_borrados.empty:
        df_dias_borrados.sort_values("DIA").to_csv(
            ruta_borrados_csv, index=False, encoding="utf-8", lineterminator="\n"
        )

    checksum_csv = huella(ruta_salida_csv)
    checksum_borrados = None
    if not df_dias_borrados.empty:
        checksum_borrados = huella(ruta_borrados_csv)
    checksum_previo_json: Optional[str] = None
    clave_rel = ruta_salida_csv.resolve().relative_to(project_root.resolve()).as_posix()

    for ruta_chk in [data_dir / "checksums.json", project_root / "checksums.json"]:
        if ruta_chk.exists():
            with open(ruta_chk, "r", encoding="utf-8") as f:
                datos_chk = json.load(f)

            if checksum_previo_json is None:
                if isinstance(datos_chk.get("datasets_procesados"), dict):
                    checksum_previo_json = datos_chk["datasets_procesados"].get("dataset_trabajo.csv")
                else:
                    checksum_previo_json = datos_chk.get(clave_rel) or datos_chk.get("dataset_trabajo.csv")

            if update_checksums_file:
                if isinstance(datos_chk.get("datasets_procesados"), dict):
                    datos_chk["datasets_procesados"]["dataset_trabajo.csv"] = checksum_csv
                    if checksum_borrados is not None:
                        datos_chk["datasets_procesados"]["tabla_dias_borrados.csv"] = checksum_borrados
                else:
                    datos_chk[clave_rel] = checksum_csv
                    if checksum_borrados is not None:
                        datos_chk["tabla_dias_borrados.csv"] = checksum_borrados
                with open(ruta_chk, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(datos_chk, f, indent=2, ensure_ascii=False)
                    f.write("\n")

    referencia = expected_checksum.strip().lower() if expected_checksum else (
        checksum_previo_json.strip().lower() if checksum_previo_json else None
    )
    coincide_checksum = (checksum_csv.lower() == referencia) if referencia else None

    return {
        "ruta_salida_csv": ruta_salida_csv,
        "ruta_borrados_csv": ruta_borrados_csv if not df_dias_borrados.empty else None,
        "tamano_mb": ruta_salida_csv.stat().st_size / (1024 * 1024),
        "n_filas": int(df_export_csv.shape[0]),
        "n_cols": int(df_export_csv.shape[1]),
        "n_dias_validos": int(df_export_csv["DIA"].nunique()),
        "fecha_min": str(df_export_csv["DIA"].min()),
        "fecha_max": str(df_export_csv["DIA"].max()),
        "n_dias_borrados": int(len(df_dias_borrados)),
        "checksum_sha256": checksum_csv,
        "checksum_previo_json": checksum_previo_json,
        "checksum_esperado": expected_checksum,
        "coincide_checksum": coincide_checksum,
    }

def calibrar_limpieza(df_unificado: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    """Parámetros de control de la limpieza, estimados solo con el primer año."""
    cols_datos = [c for c in df_unificado.columns if c not in ("DIA", "HORA", "_archivo_origen")]
    cols_ee = [c for c in cols_datos if c.endswith("_Consolidado EE")]
    calibracion = df_unificado[df_unificado["DIA"] <= FIN_CALIBRACION]
    df_mad_ee = calcular_parametros_mad_2359(calibracion, cols_ee, umbral_z=UMBRAL_MAD)
    saltos = calibracion.groupby("DIA")[COL_TARGET].diff()
    saltos_validos = saltos[saltos.between(0, 1_000_000, inclusive="left")]
    if saltos_validos.empty:
        raise ValueError(f"Faltan incrementos horarios para calibrar hasta {FIN_CALIBRACION}.")
    percentil_salto = float(saltos_validos.quantile(0.999))
    umbral_salto = float(np.ceil(percentil_salto / 100) * 100)
    return df_mad_ee, percentil_salto, umbral_salto

def run_cleaning_pipeline(
    raw_dir: Path = RAW_DATA_DIR,
    data_dir: Path = DATA_DIR,
    verify_raw_checksums: bool = True,
    save_diagnostic_plot: bool = False,
    expected_checksum: Optional[str] = None,
    update_checksums_file: bool = True,
) -> Dict[str, Any]:
    """Ejecuta las etapas 0 a 7 de limpieza horaria y genera `dataset_trabajo.csv`."""
    if verify_raw_checksums:
        verificar_checksums()

    archivos_excel = get_default_excel_sources(raw_dir)
    df_unificado, df_rep, df_extra, df_solap = unificar_archivos_excel(archivos_excel)

    cols_datos = [c for c in df_unificado.columns if c not in ("DIA", "HORA", "_archivo_origen")]
    cols_ee = [c for c in cols_datos if c.endswith("_Consolidado EE")]

    # Los parámetros de control se estiman en el primer año. Los cierres
    # de Validation y Test solo se inspeccionan con esas reglas ya fijadas.
    calibracion = df_unificado[df_unificado["DIA"] <= FIN_CALIBRACION]
    df_mad_ee = calcular_parametros_mad_2359(calibracion, cols_ee, umbral_z=UMBRAL_MAD)
    saltos = calibracion.groupby("DIA")[COL_TARGET].diff()
    # El percentil excluye retrocesos y los errores del orden de los millones.
    saltos_validos = saltos[saltos.between(0, 1_000_000, inclusive="left")]
    if saltos_validos.empty:
        raise ValueError(f"Faltan incrementos horarios para calibrar hasta {FIN_CALIBRACION}.")
    percentil_salto = float(saltos_validos.quantile(0.999))
    # La centena superior evita tratar diferencias de pocos kW como evidencia
    # de falla cuando el percentil se estima con un solo año de observaciones.
    umbral_salto = float(np.ceil(percentil_salto / 100) * 100)
    df_trabajo, df_casos_especiales = evaluar_casos_especiales_y_reparar_pico_01(
        df_unificado, df_mad_ee, target_col=COL_TARGET
    )
    df_dataset_final, df_dias_borrados, df_anomalos_conservados = filtrar_dias_invalidos(
        df_trabajo, df_mad_ee, df_casos_especiales, target_col=COL_TARGET, umbral_salto=umbral_salto
    )

    export_info = exportar_y_versionar(
        df_dataset_final=df_dataset_final,
        df_dias_borrados=df_dias_borrados,
        data_dir=data_dir,
        project_root=PROJECT_ROOT,
        expected_checksum=expected_checksum,
        update_checksums_file=update_checksums_file,
    )

    if save_diagnostic_plot:
        _guardar_grafico_diagnostico(
            df_trabajo=df_trabajo,
            df_mad_ee=df_mad_ee,
            df_anomalos_conservados=df_anomalos_conservados,
            df_dias_borrados=df_dias_borrados,
            output_path=data_dir / "processed" / "diagnostico_mad_limpieza.png",
        )

    print("=" * 80)
    print("LIMPIEZA HORARIA COMPLETADA — dataset_trabajo.csv")
    print("=" * 80)
    print(f"Archivo exportado     : {export_info['ruta_salida_csv']}")
    print(f"Tamaño en disco       : {export_info['tamano_mb']:.2f} MB")
    print(f"Dimensiones           : {export_info['n_filas']:,} filas x {export_info['n_cols']} columnas")
    print(
        f"Días válidos          : {export_info['n_dias_validos']:,} "
        f"({export_info['fecha_min']} a {export_info['fecha_max']})"
    )
    print(f"Días borrados         : {export_info['n_dias_borrados']}")
    print(f"Calibración hasta     : {FIN_CALIBRACION} | P99,9 saltos = {percentil_salto:.2f} kW")
    print(f"Umbral de salto       : {umbral_salto:.0f} kW (centena superior)")
    print(f"Anómalos OK MAD       : {len(df_anomalos_conservados)}")
    print(f"Reparaciones horarias : {len(df_rep)} | Solapados: {len(df_solap)} | Extra: {len(df_extra)}")
    print(f"SHA-256 generado      : {export_info['checksum_sha256']}")
    if export_info["checksum_previo_json"]:
        print(f"SHA-256 previo (json) : {export_info['checksum_previo_json']}")
    if export_info["checksum_esperado"]:
        print(f"SHA-256 esperado      : {export_info['checksum_esperado']}")
    if export_info["coincide_checksum"] is True:
        print("Comparación SHA-256   : COINCIDE EXACTAMENTE")
    elif export_info["coincide_checksum"] is False:
        print("Comparación SHA-256   : DIFIERE")
    else:
        print("Comparación SHA-256   : Sin hash previo en checksums.json para comparar")
    print("=" * 80)

    return {
        "df_dataset_final": df_dataset_final,
        "df_dias_borrados": df_dias_borrados,
        "df_anomalos_conservados": df_anomalos_conservados,
        "export_info": export_info,
    }


def _guardar_grafico_diagnostico(
    df_trabajo: pd.DataFrame,
    df_mad_ee: pd.DataFrame,
    df_anomalos_conservados: pd.DataFrame,
    df_dias_borrados: pd.DataFrame,
    output_path: Path,
) -> None:
    """Genera y guarda el gráfico diagnóstico de Etapa 6 sin bloquear ejecución headless."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator

    ref_mad_frio = df_mad_ee.loc[COL_TARGET]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    cierres_frio = df_trabajo[df_trabajo["HORA"] == "23:59"][COL_TARGET].dropna()
    cierres_rango_plot = cierres_frio[(cierres_frio >= 0) & (cierres_frio <= 100_000)]

    sns.histplot(
        cierres_rango_plot, bins=40, kde=True, color="#2b5c8f", ax=axes[0], label="Cierres 23:59 (Frio kW)"
    )
    axes[0].axvline(
        ref_mad_frio["Mediana_2359"],
        color="#d95f02",
        linestyle="--",
        linewidth=2,
        label=f"Mediana: {ref_mad_frio['Mediana_2359']:,.0f} kW",
    )
    axes[0].axvline(
        ref_mad_frio["Lim_Inf_MAD"],
        color="#e7298a",
        linestyle=":",
        linewidth=2,
        label=f"Límite Inf MAD: {ref_mad_frio['Lim_Inf_MAD']:,.0f} kW",
    )
    axes[0].axvline(
        ref_mad_frio["Lim_Sup_MAD"],
        color="#e7298a",
        linestyle="-.",
        linewidth=2,
        label=f"Límite Sup MAD ({UMBRAL_MAD} MAD): {ref_mad_frio['Lim_Sup_MAD']:,.0f} kW",
    )
    if not df_anomalos_conservados.empty:
        axes[0].scatter(
            df_anomalos_conservados["Frio_2359_kW"],
            np.zeros(len(df_anomalos_conservados)),
            color="#1b9e77",
            marker="D",
            s=42,
            zorder=5,
            label=f"Anomalías OK MAD conservadas ({len(df_anomalos_conservados)} días)",
        )
    axes[0].set_title(
        "Verificación MAD de 'Frio (Kw)' a las 23:59 (solo para casos anómalos)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Frio (Kw) acumulado a las 23:59")
    axes[0].set_ylabel("Frecuencia de días")
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[0].grid(alpha=0.25)

    if not df_dias_borrados.empty:
        conteo_plot = df_dias_borrados.groupby(["Archivo_Origen", "Etapa_Filtro"], as_index=False).agg(
            Dias=("DIA", "count")
        )
        sns.barplot(
            data=conteo_plot,
            x="Archivo_Origen",
            y="Dias",
            hue="Etapa_Filtro",
            palette="Set2",
            ax=axes[1],
        )
        axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].set_title("Días Borrados por Archivo Excel y Motivo", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Archivo Excel de Origen")
        axes[1].set_ylabel("Cantidad de Días Eliminados")
        axes[1].legend(title="Motivo", fontsize=8, title_fontsize=9, loc="upper right")
        axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# ===========================================================================
# PARTE B. Limpieza de predictoras (Sección 2.1) (antes src/limpieza_predictoras.py)
# LimpiadorExtremos e ImputadorTemporal: umbrales y medianas se aprenden
# en fit() SOLO con Train; transform() no recalcula nada (sin fuga del futuro).
# ===========================================================================

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

    1. Con un índice de fechas se rellena hacia adelante hasta `limite` días calendario.
       Con un índice común, el límite se cuenta en filas.
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
        if isinstance(X.index, pd.DatetimeIndex) and not X.empty:
            if not X.index.is_monotonic_increasing or X.index.has_duplicates:
                raise ValueError("El índice diario tiene que estar ordenado y sin fechas repetidas.")
            # Los días sin fila también cuentan dentro de los tres días de arrastre.
            calendario = pd.date_range(X.index.min(), X.index.max(), freq="D")
            X_ffill = X.reindex(calendario).ffill(limit=self.limite).reindex(X.index)
        else:
            X_ffill = X.ffill(limit=self.limite)
        self.imputados_ffill_ = nulos_antes - X_ffill.isna().sum()
        self.imputados_mediana_ = X_ffill.isna().sum()
        return X_ffill.fillna(self.medianas_)

    def get_feature_names_out(self, input_features=None):
        return list(self.columnas_)


# ===========================================================================
# PARTE C. Features de planta (Sección 2.2 B) (antes src/features_planta.py)
# FeaturesPlanta: ratios entre áreas, eficiencia contra la Meta e interacciones
# con el clima. Usa solo datos del día D o anteriores.
# ===========================================================================

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

    def __init__(
        self,
        col_fecha: str = "DIA",
        columnas_clima: tuple = ("temp_media_c", "radiacion_roll7"),
        incluir_agua_cond: bool = True,
    ):
        self.col_fecha = col_fecha
        self.columnas_clima = columnas_clima
        self.incluir_agua_cond = incluir_agua_cond

    def _verificar_columnas(self, X: pd.DataFrame) -> None:
        requeridas = [self.col_fecha] + self._columnas_a_limpiar()
        requeridas += [COLUMNAS_BASE["meta_frio"]] + list(self.columnas_clima)
        if self.incluir_agua_cond:
            requeridas.append(COLUMNAS_BASE["agua_cond"])
        faltantes = [c for c in requeridas if c not in X.columns]
        if faltantes:
            raise ValueError(f"FeaturesPlanta: faltan columnas en los datos de entrada: {faltantes}")

    def _columnas_a_limpiar(self) -> List[str]:
        columnas = list(COLUMNAS_A_LIMPIAR)
        if not self.incluir_agua_cond:
            columnas.remove(COLUMNAS_BASE["agua_planta"])
        if not self.columnas_clima:
            columnas.remove(COLUMNAS_BASE["hl_bodega"])
        return columnas

    def fit(self, X: pd.DataFrame, y=None):
        self._verificar_columnas(X)
        # Topes de imposibles y medianas aprendidos en train
        self.limpieza_ = Pipeline([
            ("extremos", LimpiadorExtremos(factor=5.0, cuantil=0.99)),
            ("imputacion", ImputadorTemporal(limite=3)),
        ]).fit(X[self._columnas_a_limpiar()])
        # Meta típica de cada mes del año (las metas son mensuales y cambian con la estación)
        meses = pd.to_datetime(X[self.col_fecha]).dt.month
        meta = X[COLUMNAS_BASE["meta_frio"]]
        meta = meta.where(meta > 0)
        self.meta_por_mes_ = meta.groupby(meses).median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._verificar_columnas(X)
        X = X.copy()
        base = self.limpieza_.transform(X[self._columnas_a_limpiar()])
        c = COLUMNAS_BASE
        nuevas = pd.DataFrame(index=X.index)

        # 1) Ratios entre áreas (proporciones del mismo día D)
        nuevas["ratio_frio_servicios"] = _ratio(base[c["frio"]], base[c["servicios"]])
        nuevas["ratio_bodega_elaboracion"] = _ratio(base[c["bodega"]], base[c["elaboracion"]])
        if self.incluir_agua_cond:
            agua_cond = X[c["agua_cond"]].where(X[c["agua_cond"]] >= 0)
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
        nombres = ["ratio_frio_servicios", "ratio_bodega_elaboracion"]
        if self.incluir_agua_cond:
            nombres.append("ratio_aguacond_aguaplanta")
        nombres += ["meta_frio_imputada", "eficiencia_frio_vs_meta"]
        return nombres + [f"bodega_x_{c}" for c in self.columnas_clima]


# ===========================================================================
# PARTE D. Creación de variables (Secciones 2.1 y 2.2) (antes src/feature_engineering.py)
# PreparadorFeatures guarda umbrales y medianas de Train y los aplica a otros períodos.
# ===========================================================================

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


# ===========================================================================
# PARTE E. Selección de variables (Sección 2.3) (antes src/seleccion_variables.py)
# Selectores ajustados en Train; la comparación de subconjuntos se hace en Validation.
# ===========================================================================

ALPHA_LASSO = 100
TOPE_BOSQUE = 12
ALPHA_RIDGE = 10


def seleccionar_variables(X_train, y_train, X_val, y_val):
    """Devuelve las columnas del mejor subconjunto y la evidencia de la decisión."""
    if X_train.index.max() >= X_val.index.min():
        raise ValueError("Validation debe empezar después de Train.")
    if not X_train.index.equals(y_train.index) or not X_val.index.equals(y_val.index):
        raise ValueError("Las fechas de X e y deben coincidir en cada partición.")

    # LASSO penaliza los coeficientes y puede llevarlos exactamente a cero.
    # El escalador se ajusta junto con LASSO usando las mismas filas de Train.
    lasso = make_pipeline(
        StandardScaler(),
        Lasso(alpha=ALPHA_LASSO, max_iter=30000, random_state=SEMILLA),
    )
    lasso.fit(X_train, y_train)
    coeficientes = pd.Series(lasso.named_steps["lasso"].coef_, index=X_train.columns)

    # El bosque ordena variables por reducción de error cuadrático en sus árboles.
    bosque = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=6,
        n_jobs=1,
        random_state=SEMILLA,
    )
    bosque.fit(X_train, y_train)
    importancia = pd.Series(bosque.feature_importances_, index=X_train.columns)
    top_bosque = importancia.sort_values(ascending=False, kind="stable").head(TOPE_BOSQUE)

    metodos = pd.DataFrame({
        "coeficiente_lasso": coeficientes,
        "importancia_bosque": importancia,
        "seleccion_lasso": coeficientes != 0,
        "seleccion_bosque": importancia.index.isin(top_bosque.index),
    })
    metodos["interseccion"] = metodos["seleccion_lasso"] & metodos["seleccion_bosque"]

    # El cierre de D es la referencia común. Se incluye en todas las alternativas
    # porque está disponible al pronosticar y permite comparar contra persistencia.
    subconjuntos = {"Todas": X_train.columns.tolist()}
    for nombre, columna in [
        ("LASSO", "seleccion_lasso"),
        ("Random Forest", "seleccion_bosque"),
        ("Intersección", "interseccion"),
    ]:
        seleccion = metodos[columna] | (metodos.index == "frio_hoy_d")
        subconjuntos[nombre] = metodos.index[seleccion].tolist()

    # Usamos el mismo estimador y los mismos parámetros para aislar el efecto
    # del subconjunto. La única comparación de rendimiento se hace en Validation.
    resultados = []
    for nombre, columnas in subconjuntos.items():
        modelo = make_pipeline(
            StandardScaler(),
            Ridge(alpha=ALPHA_RIDGE, random_state=SEMILLA),
        )
        modelo.fit(X_train[columnas], y_train)
        prediccion = modelo.predict(X_val[columnas])
        resultados.append({
            "subconjunto": nombre,
            "columnas": len(columnas),
            "mae_validation": mean_absolute_error(y_val, prediccion),
        })

    # En caso de empate a 0,01 kW se prefiere el subconjunto más corto.
    comparacion = pd.DataFrame(resultados)
    comparacion["mae_validation"] = comparacion["mae_validation"].round(2)
    comparacion = comparacion.sort_values(
        ["mae_validation", "columnas", "subconjunto"],
    ).reset_index(drop=True)
    ganador = comparacion.loc[0, "subconjunto"]
    return subconjuntos[ganador], metodos, comparacion


# ===========================================================================
# PARTE F. Preprocesamiento diario, target D+1, split y versionado
# ===========================================================================


def load_cleaned_data(
    filepath: Path | str = DEFAULT_INPUT_CSV,
    date_col: str = DEFAULT_DATE_COL,
    time_col: str = DEFAULT_TIME_COL,
    target_col: str = DEFAULT_TARGET_COL,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """ Carga el dataset horario generado por la Parte A (limpieza horaria)

    Args:
        filepath: Ruta al CSV (`dataset_trabajo.csv`).
        date_col: Nombre de la columna de fecha.
        time_col: Nombre de la columna de hora.
        target_col: Nombre de la columna objetivo en el cierre horario.

    Returns:
        Tupla `(df, info)` con el DataFrame validado y metadatos iniciales.

    Raises:
        FileNotFoundError: Si el archivo de entrada no existe.
        ValueError: Si faltan columnas obligatorias o no hay registros válidos.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset horario limpio en {path}. "
            "Generalo con python -m src.preprocessing_pipeline --solo-limpieza."
        )

    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    if df.empty:
        raise ValueError(f"El archivo de entrada está vacío: {path}")

    required_cols = {date_col, time_col, target_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias en {path.name}: {sorted(missing)}"
        )

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    invalid_dates = int(df[date_col].isna().sum())
    if invalid_dates > 0:
        raise ValueError(
            f"Se encontraron {invalid_dates} filas con fechas inválidas en '{date_col}'."
        )

    df[time_col] = df[time_col].astype(str).str.strip()

    info: Dict[str, Any] = {
        "input_file": path.name,
        "initial_rows": int(len(df)),
        "initial_columns": int(df.shape[1]),
        "initial_unique_days": int(df[date_col].nunique()),
        "initial_date_min": df[date_col].min().strftime("%Y-%m-%d"),
        "initial_date_max": df[date_col].max().strftime("%Y-%m-%d"),
    }
    return df, info


def aggregate_daily_data(
    df: pd.DataFrame,
    date_col: str = DEFAULT_DATE_COL,
    time_col: str = DEFAULT_TIME_COL,
    closing_time: str = DEFAULT_CLOSING_TIME,
    target_col: str = DEFAULT_TARGET_COL,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Conservo solo el 23:59 de cada dia

    Args:
        df: DataFrame horario limpio.
        date_col: Nombre de la columna de fecha.
        time_col: Nombre de la columna de hora.
        closing_time: Horario de cierre diario a extraer (`"23:59"`).
        target_col: Columna target para verificar disponibilidad de valor.

    Returns:
        Tupla `(df_daily, report)` con el DataFrame diario y métricas de auditoría.

    Raises:
        ValueError: Si existen fechas duplicadas tras filtrar por `closing_time`
            o si no quedan registros válidos.
    """
    if df.empty:
        raise ValueError("El DataFrame horario de entrada está vacío.")

    df_sorted = df.sort_values([date_col, time_col]).reset_index(drop=True)
    all_days = pd.Index(df_sorted[date_col].unique()).sort_values()

    mask_closing = df_sorted[time_col] == closing_time
    df_closing = df_sorted.loc[mask_closing].copy()

    if df_closing.empty:
        raise ValueError(
            f"No se encontraron registros con {time_col} == '{closing_time}'."
        )

    duplicated_mask = df_closing[date_col].duplicated(keep=False)
    if duplicated_mask.any():
        dup_dates = (
            df_closing.loc[duplicated_mask, date_col]
            .dt.strftime("%Y-%m-%d")
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Se detectaron fechas duplicadas en el cierre {closing_time}: {dup_dates[:10]}"
        )

    days_with_closing = pd.Index(df_closing[date_col].unique())
    missing_closing_days = all_days.difference(days_with_closing)
    missing_closing_list = [d.strftime("%Y-%m-%d") for d in missing_closing_days]

    # Conservar todas las variables operacionales de la observación de cierre,
    # removiendo únicamente la columna horaria constante.
    df_daily = df_closing.drop(columns=[time_col]).sort_values(date_col).reset_index(drop=True)

    # Validar que el target no sea nulo en el cierre diario
    null_target_mask = df_daily[target_col].isna()
    null_target_days = (
        df_daily.loc[null_target_mask, date_col].dt.strftime("%Y-%m-%d").tolist()
    )

    report: Dict[str, Any] = {
        "initial_days": int(len(all_days)),
        "days_with_2359": int(len(df_daily)),
        "days_without_2359": int(len(missing_closing_list)),
        "missing_2359_dates": missing_closing_list,
        "days_with_null_target_at_2359": int(len(null_target_days)),
        "null_target_dates": null_target_days,
        "daily_columns": int(df_daily.shape[1]),
        "date_min": df_daily[date_col].min().strftime("%Y-%m-%d"),
        "date_max": df_daily[date_col].max().strftime("%Y-%m-%d"),
    }
    return df_daily, report


def prepare_target(
    df_daily: pd.DataFrame,
    date_col: str = DEFAULT_DATE_COL,
    target_col: str = DEFAULT_TARGET_COL,
    target_name: str = DEFAULT_TARGET_NAME,
    drop_invalid_target: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Alinea temporalmente el DataFrame diario y construye el target del día D+1.

    Evita desplazamientos posicionales erróneos cuando faltan días calendario:
    asocia cada día `D` exclusivamente con el valor de `target_col` del día
    calendario `D + 1` (mediante indexado continuo o búsqueda exacta por fecha).

    Args:
        df_daily: DataFrame diario ordenado o desordenado con columna `date_col`.
        date_col: Nombre de la columna de fecha.
        target_col: Nombre de la columna base del consumo diario (`Frio (Kw)_Consolidado EE`).
        target_name: Nombre de la columna target resultante (`"y"`).
        drop_invalid_target: Si es True, descarta filas sin target D+1 válido.

    Returns:
        Tupla `(df_aligned, report)` con el DataFrame alineado y detalle de descartes.

    Raises:
        ValueError: Si falta `date_col` o `target_col`, o si hay fechas duplicadas.
    """
    if df_daily.empty:
        raise ValueError("El DataFrame diario está vacío.")
    for col in (date_col, target_col):
        if col not in df_daily.columns:
            raise ValueError(f"La columna requerida '{col}' no existe en el DataFrame diario.")

    df_work = df_daily.copy()
    df_work[date_col] = pd.to_datetime(df_work[date_col]).dt.normalize()
    if df_work[date_col].isna().any():
        raise ValueError("El DataFrame diario contiene fechas sin valor.")
    df_work = df_work.sort_values(date_col).reset_index(drop=True)

    if df_work[date_col].duplicated().any():
        raise ValueError("El DataFrame diario contiene fechas duplicadas.")

    # Alineación estricta por fecha calendario D -> D+1:
    # Creamos una serie indexada por fecha y consultamos D + 1 día.
    cierre = pd.to_numeric(df_work[target_col], errors="coerce")
    cierre_valido = np.isfinite(cierre) & (cierre >= 0)
    df_work[target_col] = cierre.where(cierre_valido)
    target_series = df_work.set_index(date_col)[target_col]
    next_calendar_dates = df_work[date_col] + pd.Timedelta(days=1)
    cierre_manana = next_calendar_dates.map(target_series)
    df_work[target_name] = cierre_manana.where(cierre_valido)

    # Auditoría de discontinuidades y targets faltantes
    is_next_day_present = next_calendar_dates.isin(target_series.index)
    is_target_valid = df_work[target_name].notna()

    discontinuity_dates = (
        df_work.loc[~is_next_day_present, date_col].dt.strftime("%Y-%m-%d").tolist()
    )
    missing_value_only_dates = (
        df_work.loc[is_next_day_present & cierre_manana.isna(), date_col]
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )
    discarded_dates = (
        df_work.loc[~is_target_valid, date_col].dt.strftime("%Y-%m-%d").tolist()
    )
    # Si mañana tiene cierre pero hoy es inválido, la fila tampoco sirve de ejemplo.
    invalid_origin_only = int((cierre_manana.notna() & ~cierre_valido).sum())

    n_before = int(len(df_work))
    if drop_invalid_target:
        df_out = df_work.loc[is_target_valid].copy().reset_index(drop=True)
    else:
        df_out = df_work

    n_after = int(len(df_out))
    if n_after == 0:
        raise ValueError("No quedaron observaciones con target válido tras la alineación D+1.")

    report: Dict[str, Any] = {
        "target_column_source": target_col,
        "target_column_output": target_name,
        "alignment_method": "calendar_date_D_plus_1_exact_lookup",
        "observations_before_target_filter": n_before,
        "valid_target_observations": int(is_target_valid.sum()),
        "discarded_observations": len(discarded_dates),
        "discarded_due_to_missing_next_day_or_last_day": len(discontinuity_dates),
        "discarded_due_to_nan_in_next_day_target": len(missing_value_only_dates),
        "discarded_due_to_invalid_origin_only": invalid_origin_only,
        "discarded_dates": discarded_dates,
        "date_min": df_out[date_col].min().strftime("%Y-%m-%d"),
        "date_max": df_out[date_col].max().strftime("%Y-%m-%d"),
    }
    return df_out, report


def temporal_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    date_col: str = DEFAULT_DATE_COL,
    target_col: str = DEFAULT_TARGET_NAME,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    Dict[str, Any],
]:
    """Divide cronológicamente el dataset en Train, Validation y Test sin shuffle.

    Args:
        df: DataFrame diario con features y target alineado.
        train_size: Proporción para entrenamiento (por defecto 0.70).
        val_size: Proporción para validación (por defecto 0.15).
        test_size: Proporción para prueba (por defecto 0.15).
        date_col: Nombre de la columna de fecha para ordenar cronológicamente.
        target_col: Nombre de la columna target (`"y"`).
        feature_cols: Lista explícita de columnas predictoras. Si es None, usa
            todas las columnas excepto `date_col` y `target_col`.

    Returns:
        Tupla `(X_train, y_train, X_val, y_val, X_test, y_test, split_info)`.
        `split_info` incluye además los DataFrames completos (`train_df`, `val_df`,
        `test_df`) y las métricas de tamaño, shapes, fechas y proporciones.

    Raises:
        ValueError: Si las proporciones no suman 1.0, si faltan columnas o si
            algún conjunto queda vacío.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0, atol=1e-6):
        raise ValueError(
            f"Las proporciones deben sumar 1.0 (recibido: {train_size + val_size + test_size:.6f})."
        )
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("train_size, val_size y test_size deben ser positivos.")
    if df.empty:
        raise ValueError("No se puede dividir un DataFrame vacío.")
    for col in (date_col, target_col):
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}' para realizar el split.")

    df_sorted = df.copy()
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col]).dt.normalize()
    if df_sorted[date_col].isna().any() or df_sorted[date_col].duplicated().any():
        raise ValueError("La división necesita fechas válidas y únicas.")
    if not np.isfinite(pd.to_numeric(df_sorted[target_col], errors="coerce")).all():
        raise ValueError("Antes de dividir, descartá las filas sin etiqueta válida de D+1.")
    df_sorted = df_sorted.sort_values(date_col).reset_index(drop=True)

    n_total = len(df_sorted)
    n_train = int(round(n_total * train_size))
    n_val = int(round(n_total * val_size))
    n_test = n_total - n_train - n_val

    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"El dataset ({n_total} filas) es insuficiente para generar 3 particiones no vacías."
        )

    train_df = df_sorted.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[n_train + n_val :].copy().reset_index(drop=True)

    if feature_cols is None:
        selected_features = [c for c in df_sorted.columns if c not in (date_col, target_col)]
    else:
        missing_feats = set(feature_cols) - set(df_sorted.columns)
        if missing_feats:
            raise ValueError(f"Features seleccionadas inexistentes: {sorted(missing_feats)}")
        selected_features = list(feature_cols)

    if not selected_features:
        raise ValueError("No hay columnas de features disponibles para construir X.")

    X_train = train_df[selected_features].copy()
    y_train = train_df[target_col].copy()
    X_val = val_df[selected_features].copy()
    y_val = val_df[target_col].copy()
    X_test = test_df[selected_features].copy()
    y_test = test_df[target_col].copy()

    def _build_split_meta(
        sub_df: pd.DataFrame, X_sub: pd.DataFrame, y_sub: pd.Series
    ) -> Dict[str, Any]:
        return {
            "observations": int(len(sub_df)),
            "n_features": int(X_sub.shape[1]),
            "X_shape": [int(X_sub.shape[0]), int(X_sub.shape[1])],
            "y_shape": [int(y_sub.shape[0])],
            "date_start": sub_df[date_col].min().strftime("%Y-%m-%d"),
            "date_end": sub_df[date_col].max().strftime("%Y-%m-%d"),
            "proportion": round(len(sub_df) / n_total, 6),
        }

    split_info: Dict[str, Any] = {
        "total_observations": int(n_total),
        "feature_columns": selected_features,
        "train": _build_split_meta(train_df, X_train, y_train),
        "validation": _build_split_meta(val_df, X_val, y_val),
        "test": _build_split_meta(test_df, X_test, y_test),
        "dataframes": {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
        },
    }
    return X_train, y_train, X_val, y_val, X_test, y_test, split_info


def save_processed_datasets(
    df_daily_base: pd.DataFrame,
    df_final: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path | str = PROCESSED_DIR,
    date_col: str = DEFAULT_DATE_COL,
) -> Dict[str, Path]:
    """Exporta los 5 conjuntos CSV sin escalar en `data/processed/`.

    Utiliza `index=False`, `lineterminator="\\n"` y formato de fecha `YYYY-MM-DD`.

    Args:
        df_daily_base: Dataset diario base de cierres 23:59 (sin target D+1).
        df_final: Dataset diario alineado con features disponibles y target `y`.
        train_df: Partición de entrenamiento sin escalar.
        val_df: Partición de validación sin escalar.
        test_df: Partición de prueba sin escalar.
        output_dir: Carpeta de destino (`data/processed`).
        date_col: Nombre de la columna de fecha.

    Returns:
        Diccionario `{nombre_archivo: Path}` con las rutas exportadas.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    datasets = {
        "dataset_diario_base.csv": df_daily_base,
        "dataset_final.csv": df_final,
        "train.csv": train_df,
        "validation.csv": val_df,
        "test.csv": test_df,
    }

    exported_paths: Dict[str, Path] = {}
    for filename, df_item in datasets.items():
        if df_item.empty:
            raise ValueError(f"No se puede exportar '{filename}' porque está vacío.")

        df_export = df_item.copy()
        if date_col in df_export.columns:
            df_export[date_col] = pd.to_datetime(df_export[date_col]).dt.strftime("%Y-%m-%d")

        dest = out_path / filename
        df_export.to_csv(dest, index=False, encoding="utf-8", lineterminator="\n")
        exported_paths[filename] = dest

    return exported_paths


def generate_checksum(filepath: Path | str, block_size: int = 1048576) -> str:
    """Calcula el hash SHA-256 de un archivo leyéndolo por bloques.

    Args:
        filepath: Ruta al archivo a verificar.
        block_size: Tamaño de lectura por bloque en bytes (por defecto 1 MiB).

    Returns:
        Cadena hexadecimal del digest SHA-256.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    return huella(Path(filepath), bloque=block_size)


def verificar_hashes_exportados(exported_paths, registro):
    """Compara los CSV de esta corrida con las huellas de referencia."""
    hashes = {}
    for nombre, ruta in exported_paths.items():
        if ruta is None or not Path(ruta).is_file():
            raise FileNotFoundError(f"La corrida no generó el archivo esperado: {nombre}.")
        clave = f"data/processed/{nombre}"
        esperado = registro.get(clave, registro.get(nombre))
        if esperado is None:
            raise ValueError(f"No hay un SHA-256 de referencia para {nombre}.")
        calculado = generate_checksum(ruta)
        if calculado != esperado:
            raise ValueError(
                f"SHA-256 distinto para {nombre}. Esperado: {esperado}. Calculado: {calculado}."
            )
        hashes[nombre] = calculado
    return hashes


def update_checksums(
    exported_paths: Dict[str, Path],
    checksums_path: Path | str = DEFAULT_CHECKSUMS_JSON,
    project_root: Path | str = PROJECT_ROOT,
) -> Dict[str, str]:
    """Actualiza `data/checksums.json` preservando las entradas existentes.

    Compatible tanto con el esquema plano usado en `data/checksums.json`
    (claves relativas tipo `"data/processed/..."`) como con un esquema agrupado
    `{"archivos_originales": ..., "datasets_procesados": ...}` si ya existiera.

    Args:
        exported_paths: Mapeo `{nombre_archivo: Path}` de los CSV exportados.
        checksums_path: Ruta al archivo `checksums.json`.
        project_root: Raíz del proyecto para construir rutas relativas POSIX.

    Returns:
        Diccionario `{nombre_archivo: sha256}` con los checksums generados.
    """
    chk_path = Path(checksums_path)
    root_path = Path(project_root).resolve()
    chk_path.parent.mkdir(parents=True, exist_ok=True)

    existing_data: Dict[str, Any] = {}
    if chk_path.exists():
        with open(chk_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)

    generated: Dict[str, str] = {}
    uses_nested_schema = isinstance(existing_data.get("datasets_procesados"), dict) or isinstance(
        existing_data.get("archivos_originales"), dict
    )

    if uses_nested_schema:
        existing_data.setdefault("datasets_procesados", {})

    for filename, file_path in exported_paths.items():
        digest = generate_checksum(file_path)
        generated[filename] = digest

        if uses_nested_schema:
            existing_data["datasets_procesados"][filename] = digest
        else:
            try:
                rel_key = file_path.resolve().relative_to(root_path).as_posix()
            except ValueError:
                rel_key = filename
            existing_data[rel_key] = digest

    with open(chk_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return generated


def build_scaler_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Construye y ajusta un `Pipeline` con `StandardScaler` usando solo `X_train`.

    El escalador ajustado debe utilizarse luego para transformar `X_val` y
    `X_test`, evitando fuga de datos (data leakage). No modifica ni guarda
    versiones escaladas en `data/processed/`.

    Args:
        X_train: DataFrame de features de entrenamiento.

    Returns:
        Instancia ajustada de `sklearn.pipeline.Pipeline`.

    Raises:
        ValueError: Si `X_train` está vacío o contiene columnas no numéricas.
    """
    if X_train.empty:
        raise ValueError("X_train está vacío; no se puede ajustar StandardScaler.")

    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"X_train contiene columnas no numéricas incompatibles con StandardScaler: {non_numeric}"
        )

    pipeline = Pipeline([("scaler", StandardScaler())])
    pipeline.fit(X_train)
    return pipeline


def generate_data_lineage(
    lineage_path: Path | str,
    input_info: Dict[str, Any],
    daily_report: Dict[str, Any],
    target_report: Dict[str, Any],
    split_info: Dict[str, Any],
    df_daily_base: pd.DataFrame,
    df_final: pd.DataFrame,
    checksums: Dict[str, str],
    preparation_report: Dict[str, Any],
) -> Path:
    """Genera `data/processed/data_lineage.json` con trazabilidad completa calculada.

    Args:
        lineage_path: Ruta de salida del JSON de linaje.
        input_info: Metadatos del archivo de entrada.
        daily_report: Métricas de extracción del cierre 23:59.
        target_report: Métricas de alineación D+1.
        split_info: Dimensiones, shapes, rangos y proporciones de Train/Val/Test.
        df_daily_base: DataFrame diario base.
        df_final: DataFrame final exportado.
        checksums: Hashes SHA-256 de los archivos generados.
        preparation_report: Reglas de preparación y evidencia de la selección.

    Returns:
        Ruta del archivo `data_lineage.json` generado.
    """
    out_path = Path(lineage_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lineage_payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).date().isoformat(),
        "script": "src/preprocessing_pipeline.py",
        "seed": SEMILLA,
        "pipeline_status": {
            "stage": "2.4_final",
            "feature_engineering_applied": True,
            "feature_selection_applied": True,
            "dataset_final_es_definitivo": True,
        },
        "input": input_info,
        "daily_extraction": {
            "criterion": DEFAULT_CLOSING_TIME,
            "initial_days": daily_report["initial_days"],
            "filtered_days_without_2359": daily_report["days_without_2359"],
            "missing_2359_dates": daily_report["missing_2359_dates"],
            "days_with_null_target_at_2359": daily_report["days_with_null_target_at_2359"],
            "final_daily_days": daily_report["days_with_2359"],
            "date_min": daily_report["date_min"],
            "date_max": daily_report["date_max"],
        },
        "target_alignment": target_report,
        "preparation": preparation_report,
        "splits": {
            "train": split_info["train"],
            "validation": split_info["validation"],
            "test": split_info["test"],
        },
        "variables": {
            "dataset_diario_base": list(df_daily_base.columns),
            "dataset_final": list(df_final.columns),
            "feature_columns": split_info["feature_columns"],
        },
        "checksums_sha256": checksums,
    }

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(lineage_payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return out_path


def run_pipeline(
    input_csv: Path | str = DEFAULT_INPUT_CSV,
    output_dir: Path | str = PROCESSED_DIR,
    checksums_path: Path | str = DEFAULT_CHECKSUMS_JSON,
    lineage_path: Path | str = DEFAULT_LINEAGE_JSON,
    *,
    verificar_hashes: bool = False,
) -> Dict[str, Any]:
    """Ejecuta las secciones 2.1 a 2.4 a partir de los archivos verificados."""
    if not Path(input_csv).exists():
        raise FileNotFoundError("Generá dataset_trabajo.csv con python -m src.preprocessing_pipeline --solo-limpieza.")
    with open(checksums_path, encoding="utf-8") as archivo:
        registro = json.load(archivo)
    # src/preprocessing_pipeline.py (líneas 858-864)
# src/preprocessing_pipeline.py (líneas 858-864)
    entrada_sha256 = generate_checksum(input_csv)
    if entrada_sha256 != registro["data/processed/dataset_trabajo.csv"]:
        raise ValueError("El SHA-256 de dataset_trabajo.csv difiere del registrado.")

    clima_sha256 = None
    clave_clima = "data/reference/clima_tecate_openmeteo.csv"
    if verificar_hashes or (CLIMA_CSV.is_file() and clave_clima in registro):
        clima_sha256 = generate_checksum(CLIMA_CSV)
        if clima_sha256 != registro.get(clave_clima):
            raise ValueError("El SHA-256 del clima de Tecate difiere del registrado.")

    # 1. Fijamos una sola población supervisada y un solo corte por fechas.
    df_hourly, input_info = load_cleaned_data(input_csv)
    input_info["sha256"] = entrada_sha256
    df_daily_base, daily_report = aggregate_daily_data(df_hourly)
    etiquetadas, target_report = prepare_target(df_daily_base)
    columnas_base = [DEFAULT_DATE_COL, DEFAULT_TARGET_COL, DEFAULT_TARGET_NAME]
    division_inicial = temporal_split(etiquetadas[columnas_base])
    cortes = division_inicial[-1]["dataframes"]
    train = cortes["train_df"].set_index(DEFAULT_DATE_COL)
    validation = cortes["val_df"].set_index(DEFAULT_DATE_COL)
    y_train = train[DEFAULT_TARGET_NAME]
    y_val = validation[DEFAULT_TARGET_NAME]

    # 2. Fit con fechas de Train. El historial incluye cierres sin y porque
    # contribuyen a ventanas posteriores, pero esos días no estiman parámetros.
    hasta_train = df_daily_base[DEFAULT_DATE_COL] <= train.index.max()
    hasta_val = df_daily_base[DEFAULT_DATE_COL] <= validation.index.max()
    preparador = PreparadorFeatures().fit(df_daily_base.loc[hasta_train], y_train)
    candidatas = preparador.transform(df_daily_base.loc[hasta_val])
    X_train_candidatas = candidatas.loc[train.index]
    X_val_candidatas = candidatas.loc[validation.index]
    for nombre, X in [("Train", X_train_candidatas), ("Validation", X_val_candidatas)]:
        columnas_invalidas = X.columns[~np.isfinite(X).all()].tolist()
        if columnas_invalidas:
            raise ValueError(f"Predictoras sin valor finito en {nombre}: {columnas_invalidas}")

    # 3. Los dos selectores se ajustan con Train. Validation elige el subconjunto.
    columnas, metodos, comparacion = seleccionar_variables(
        X_train_candidatas, y_train, X_val_candidatas, y_val,
    )

    # 4. Con la lista cerrada transformamos el historial completo. Asociamos
    # las features con las etiquetas ya construidas, por fecha de origen D.
    X_completo = preparador.transform(df_daily_base)
    etiquetas = etiquetadas.set_index(DEFAULT_DATE_COL)[DEFAULT_TARGET_NAME]
    df_final = X_completo.loc[etiquetas.index, columnas].copy()
    if not np.isfinite(df_final).all().all():
        faltantes = df_final.columns[~np.isfinite(df_final).all()].tolist()
        raise ValueError(f"Predictoras finales sin valor finito: {faltantes}")
    df_final[DEFAULT_TARGET_NAME] = etiquetas
    df_final = df_final.reset_index()
    X_train, y_train, X_val, y_val, X_test, y_test, split_info = temporal_split(df_final)

    # 5. El escalador aprende de Train. Exportamos los datos originales sin escalar.
    scaler_pipeline = build_scaler_pipeline(X_train)
    split_dfs = split_info["dataframes"]
    exported_paths = save_processed_datasets(
        df_daily_base=df_daily_base,
        df_final=df_final,
        train_df=split_dfs["train_df"],
        val_df=split_dfs["val_df"],
        test_df=split_dfs["test_df"],
        output_dir=output_dir,
    )
    # En auditoría se compara con el registro leído al empezar la corrida.
    if verificar_hashes:
        checksums = verificar_hashes_exportados(exported_paths, registro)
    else:
        checksums = update_checksums(exported_paths, checksums_path)

    preparation_report = {
        "hourly_cleaning": {
            "script": "src/preprocessing_pipeline.py (Parte A)",
            "calibration_through": FIN_CALIBRACION,
            "min_hourly_readings": MIN_HORAS_DIA,
            "mad_threshold": UMBRAL_MAD,
            "increment_threshold": "P99,9 de incrementos de calibración en [0, 1.000.000), redondeado a la centena superior",
            "handoff_sha256": entrada_sha256,
        },
        "fit_cutoff_D": split_info["train"]["date_end"],
        "first_validation_D": split_info["validation"]["date_start"],
        "forecast_time": "Cierre de D. El último y de Train ya está disponible al emitir el primer pronóstico de Validation.",
        "predictor_cleaning": {
            "script": "src/preprocessing_pipeline.py (Parte B)",
            "max_missing_train": 0.20,
            "min_abs_spearman_y_train": 0.40,
            "redundancy_abs_spearman": 0.90,
            "same_day_frio_proxy_abs_spearman": 0.98,
            "outliers": "Negativos o mayores a 5 × P99 de Train",
            "imputation": "Hasta tres días calendario hacia adelante y mediana de Train",
            "operativas": preparador.operativas_,
        },
        "feature_engineering": {
            "scripts": ["src/preprocessing_pipeline.py (Partes C y D)"],
            "rolling_days_and_min_obs": {"mean_3": 2, "mean_7": 4, "mean_14": 7, "std_7": 4},
            "lags_D": [1, 2, 7],
            "ratio_frio_servicios": "Intervalo [0, 1], imputación usando Train si es inválido",
            "missing_meta": "Mediana de metas positivas del mismo mes en Train",
            "weather_snapshot_sha256": clima_sha256,
            "weather_use": "Diagnóstico 2.2. Su disponibilidad al cierre de D no está documentada.",
            "candidatas": candidatas.columns.tolist(),
        },
        "selection": {
            "script": "src/preprocessing_pipeline.py (Parte E)",
            "fit_partition": "Train",
            "scoring_partition": "Validation",
            "lasso_alpha": ALPHA_LASSO,
            "rf_top": TOPE_BOSQUE,
            "ridge_alpha": ALPHA_RIDGE,
            "comparison": "MAE de Ridge en Validation con desempate por menos columnas",
            "validation_results": comparacion.to_dict(orient="records"),
        },
        "scaling": "StandardScaler ajustado en X_train. CSV exportados sin escalar.",
    }
    lineage_file = generate_data_lineage(
        lineage_path=lineage_path,
        input_info=input_info,
        daily_report=daily_report,
        target_report=target_report,
        split_info=split_info,
        df_daily_base=df_daily_base,
        df_final=df_final,
        checksums=checksums,
        preparation_report=preparation_report,
    )

    summary = {
        "input_info": input_info,
        "daily_report": daily_report,
        "target_report": target_report,
        "split_info": {k: v for k, v in split_info.items() if k != "dataframes"},
        "exported_paths": {k: str(v) for k, v in exported_paths.items()},
        "checksums": checksums,
        "lineage_file": str(lineage_file),
    }
    _print_execution_summary(summary)
    return {
        "df_daily_base": df_daily_base,
        "df_final": df_final,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler_pipeline": scaler_pipeline,
        "preparador": preparador,
        "X_train_candidatas": X_train_candidatas,
        "y_train_por_fecha": train[DEFAULT_TARGET_NAME],
        "evidencia_selectores": metodos,
        "comparacion_subconjuntos": comparacion,
        "summary": summary,
    }


def _print_execution_summary(summary: Dict[str, Any]) -> None:
    """Imprime un resumen estructurado de la ejecución del pipeline."""
    daily = summary["daily_report"]
    target = summary["target_report"]
    splits = summary["split_info"]
    checksums = summary["checksums"]

    print("=" * 80)
    print("SECCIÓN 2.4 — RESUMEN DE PREPARACIÓN Y VERSIONADO DE DATOS")
    print("=" * 80)
    print(f"Días iniciales                  : {daily['initial_days']}")
    print(f"Días con registro 23:59         : {daily['days_with_2359']}")
    print(f"Días descartados (sin 23:59)    : {daily['days_without_2359']}")
    print(
        f"Observaciones descartadas (D+1) : {target['discarded_observations']} "
        f"(discontinuidad/último día={target['discarded_due_to_missing_next_day_or_last_day']}, "
        f"cierre inválido D+1={target['discarded_due_to_nan_in_next_day_target']}, "
        f"solo cierre D inválido={target['discarded_due_to_invalid_origin_only']})"
    )
    print(f"Observaciones con target válido : {target['valid_target_observations']}")
    print("-" * 80)
    for name in ("train", "validation", "test"):
        s = splits[name]
        print(
            f"{name.upper():<10} | n={s['observations']:>4} ({s['proportion']*100:>5.2f}%) | "
            f"X={s['X_shape']} | y={s['y_shape']} | "
            f"{s['date_start']} -> {s['date_end']}"
        )
    print("-" * 80)
    print("Checksums SHA-256 generados:")
    for fname, digest in checksums.items():
        print(f"  {fname:<25}: {digest}")
    print(f"Linaje guardado en {Path(summary['lineage_file']).name}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Limpieza horaria + preprocesamiento diario (Fase 2)."
    )
    parser.add_argument("--solo-limpieza", action="store_true",
                        help="Ejecuta solo la Parte A (genera dataset_trabajo.csv).")
    parser.add_argument("--skip-raw-checksums", action="store_true",
                        help="Omite la verificación inicial de checksums de los crudos.")
    parser.add_argument("--save-plot", action="store_true",
                        help="Guarda data/processed/diagnostico_mad_limpieza.png.")
    parser.add_argument("--expected-checksum", type=str, default=None,
                        help="SHA-256 de referencia para dataset_trabajo.csv.")
    parser.add_argument("--verificar-hashes", "--no-update-checksums", action="store_true",
                        help="Compara los CSV con data/checksums.json y conserva el registro.")
    args = parser.parse_args()

    if args.verificar_hashes:
        with DEFAULT_CHECKSUMS_JSON.open(encoding="utf-8") as archivo:
            registro = json.load(archivo)

    limpieza = run_cleaning_pipeline(
        verify_raw_checksums=not args.skip_raw_checksums,
        save_diagnostic_plot=args.save_plot,
        expected_checksum=args.expected_checksum,
        update_checksums_file=not args.verificar_hashes,
    )
    if args.verificar_hashes:
        info = limpieza["export_info"]
        verificar_hashes_exportados({
            "dataset_trabajo.csv": info["ruta_salida_csv"],
            "tabla_dias_borrados.csv": info["ruta_borrados_csv"],
        }, registro)
        if args.expected_checksum and not info["coincide_checksum"]:
            raise ValueError("El SHA-256 de dataset_trabajo.csv difiere del indicado en --expected-checksum.")
    if not args.solo_limpieza:
        run_pipeline(verificar_hashes=args.verificar_hashes)


if __name__ == "__main__":
    main()