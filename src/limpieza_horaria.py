"""Limpieza horaria de los Excel crudos (Fase 1.1 y 2.1) -> dataset_trabajo.csv.

Primera etapa del pipeline. Une los tres libros, repara la secuencia horaria de cada hoja,
calibra con el primer año (hasta `FIN_CALIBRACION`, antes del corte de Train) los
umbrales del cierre 23:59 y descarta los días inválidos:

    * días con menos de 24 lecturas horarias;
    * totales imposibles (saltos por encima del P99,9 calibrado, del orden del millón);
    * días en que el totalizador del frío no se reinició a medianoche.

Exporta `data/processed/dataset_trabajo.csv` (horario, sin escalar), la tabla de días
borrados y sus SHA-256 en `data/checksums.json`. También expone las funciones que
reutilizan el resto de las etapas y `src/predict.py`: la lectura del dataset horario, el
cierre diario de las 23:59, el objetivo D+1 y la división cronológica 70/15/15.

Uso desde la raíz del proyecto:
    python -m src.limpieza_horaria
    python -m src.limpieza_horaria --verificar-hashes   # compara con data/checksums.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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
    huella,
    verificar_checksums,
)

PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_INPUT_CSV = PROCESSED_DIR / "dataset_trabajo.csv"
DEFAULT_CHECKSUMS_JSON = DATA_DIR / "checksums.json"

DEFAULT_DATE_COL = "DIA"
DEFAULT_TIME_COL = "HORA"
DEFAULT_CLOSING_TIME = "23:59"
DEFAULT_TARGET_COL = "Frio (Kw)_Consolidado EE"
DEFAULT_TARGET_NAME = "y"


# ===========================================================================
# Limpieza horaria
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


def _sha_registrado(registro: Dict[str, Any], clave_rel: str) -> Optional[str]:
    """Busca el SHA-256 de un archivo en checksums.json (esquema plano o agrupado).

    Esquema plano: clave relativa POSIX (``data/processed/x.csv``) o solo el nombre.
    Esquema agrupado: ``{"datasets_procesados": {"x.csv": ...}}``.
    """
    nombre = Path(clave_rel).name
    agrupado = registro.get("datasets_procesados")
    if isinstance(agrupado, dict) and nombre in agrupado:
        return agrupado[nombre]
    return registro.get(clave_rel) or registro.get(nombre)


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
    clave_rel = ruta_salida_csv.resolve().relative_to(project_root.resolve()).as_posix()

    # Un único registro de hashes: data/checksums.json (el mismo que usa
    # verificar_checksums y run_pipeline).
    ruta_chk = data_dir / "checksums.json"
    datos_chk: Dict[str, Any] = {}
    if ruta_chk.exists():
        with open(ruta_chk, "r", encoding="utf-8") as f:
            datos_chk = json.load(f)
    checksum_previo_json = _sha_registrado(datos_chk, clave_rel)

    # Un hash esperado explícito es una verificación: si no coincide, se corta
    # antes de registrar el nuevo hash como válido.
    if expected_checksum and checksum_csv.lower() != expected_checksum.strip().lower():
        raise ValueError(
            "El SHA-256 de dataset_trabajo.csv no coincide con el esperado.\n"
            f"  generado: {checksum_csv}\n  esperado: {expected_checksum.strip()}"
        )

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
    """Parámetros de control de la limpieza, estimados solo con el primer año.

    Los cierres de Validation y Test solo se inspeccionan con estas reglas ya
    fijadas. Única implementación: la usan run_cleaning_pipeline y train_model.
    """
    cols_datos = [c for c in df_unificado.columns if c not in ("DIA", "HORA", "_archivo_origen")]
    cols_ee = [c for c in cols_datos if c.endswith("_Consolidado EE")]
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

    # Misma calibración que usa train_model (una sola implementación).
    df_mad_ee, percentil_salto, umbral_salto = calibrar_limpieza(df_unificado)
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
# Cierre diario, objetivo D+1 y división cronológica
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Limpieza horaria de los Excel crudos.")
    parser.add_argument("--skip-raw-checksums", action="store_true",
                        help="Omite la verificación inicial de checksums de los crudos.")
    parser.add_argument("--save-plot", action="store_true",
                        help="Guarda data/processed/diagnostico_mad_limpieza.png.")
    parser.add_argument("--expected-checksum", type=str, default=None,
                        help="SHA-256 de referencia para dataset_trabajo.csv.")
    parser.add_argument("--verificar-hashes", action="store_true",
                        help="Compara los CSV con data/checksums.json y conserva el registro.")
    args = parser.parse_args()

    registro = None
    if args.verificar_hashes:
        with DEFAULT_CHECKSUMS_JSON.open(encoding="utf-8") as archivo:
            registro = json.load(archivo)

    limpieza = run_cleaning_pipeline(
        verify_raw_checksums=not args.skip_raw_checksums,
        save_diagnostic_plot=args.save_plot,
        expected_checksum=args.expected_checksum,
        update_checksums_file=not args.verificar_hashes,
    )
    if registro is not None:
        info = limpieza["export_info"]
        verificar_hashes_exportados({
            "dataset_trabajo.csv": info["ruta_salida_csv"],
            "tabla_dias_borrados.csv": info["ruta_borrados_csv"],
        }, registro)
        print("SHA-256 de dataset_trabajo.csv y tabla_dias_borrados.csv: coinciden con data/checksums.json")
    if args.expected_checksum and not limpieza["export_info"]["coincide_checksum"]:
        raise ValueError("El SHA-256 de dataset_trabajo.csv difiere del indicado en --expected-checksum.")


if __name__ == "__main__":
    main()
