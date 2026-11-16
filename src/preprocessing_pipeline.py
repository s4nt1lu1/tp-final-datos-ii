"""Pipeline reproducible de la Fase 2.

Parte del CSV horario limpio de `src/correccion_fechas.py`. Alinea D+1, fija
la división cronológica, ajusta la preparación en Train y compara los
subconjuntos en Validation antes de aplicar el resultado al resto de la serie.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.auxiliar_functions import DATA_DIR, SEMILLA, huella
from src.correccion_fechas import FIN_CALIBRACION, MIN_HORAS_DIA, UMBRAL_MAD
from src.feature_engineering import CLIMA_CSV, PreparadorFeatures
from src.seleccion_variables import ALPHA_LASSO, ALPHA_RIDGE, TOPE_BOSQUE, seleccionar_variables

PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_INPUT_CSV = PROCESSED_DIR / "dataset_trabajo.csv"
DEFAULT_CHECKSUMS_JSON = DATA_DIR / "checksums.json"
DEFAULT_LINEAGE_JSON = PROCESSED_DIR / "data_lineage.json"

DEFAULT_DATE_COL = "DIA"
DEFAULT_TIME_COL = "HORA"
DEFAULT_CLOSING_TIME = "23:59"
DEFAULT_TARGET_COL = "Frio (Kw)_Consolidado EE"
DEFAULT_TARGET_NAME = "y"


def load_cleaned_data(
    filepath: Path | str = DEFAULT_INPUT_CSV,
    date_col: str = DEFAULT_DATE_COL,
    time_col: str = DEFAULT_TIME_COL,
    target_col: str = DEFAULT_TARGET_COL,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """ Cargo el dataset que hice en correccion_fechas.py

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
            "Generalo con python -m src.correccion_fechas."
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
) -> Dict[str, Any]:
    """Ejecuta las secciones 2.1 a 2.4 a partir de los archivos verificados."""
    if not Path(input_csv).exists():
        raise FileNotFoundError("Generá dataset_trabajo.csv con python -m src.correccion_fechas.")
    with open(checksums_path, encoding="utf-8") as archivo:
        registro = json.load(archivo)
    entrada_sha256 = generate_checksum(input_csv)
    clima_sha256 = generate_checksum(CLIMA_CSV)
    if entrada_sha256 != registro["data/processed/dataset_trabajo.csv"]:
        raise ValueError("El SHA-256 de dataset_trabajo.csv difiere del registrado.")
    if clima_sha256 != registro["data/reference/clima_tecate_openmeteo.csv"]:
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
    checksums = update_checksums(exported_paths, checksums_path)

    preparation_report = {
        "hourly_cleaning": {
            "script": "src/correccion_fechas.py",
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
            "script": "src/limpieza_predictoras.py",
            "max_missing_train": 0.20,
            "min_abs_spearman_y_train": 0.40,
            "redundancy_abs_spearman": 0.90,
            "same_day_frio_proxy_abs_spearman": 0.98,
            "outliers": "Negativos o mayores a 5 × P99 de Train",
            "imputation": "Hasta tres días calendario hacia adelante y mediana de Train",
            "operativas": preparador.operativas_,
        },
        "feature_engineering": {
            "scripts": ["src/feature_engineering.py", "src/features_planta.py"],
            "rolling_days_and_min_obs": {"mean_3": 2, "mean_7": 4, "mean_14": 7, "std_7": 4},
            "lags_D": [1, 2, 7],
            "ratio_frio_servicios": "Intervalo [0, 1], imputación usando Train si es inválido",
            "missing_meta": "Mediana de metas positivas del mismo mes en Train",
            "weather_snapshot_sha256": clima_sha256,
            "weather_use": "Diagnóstico 2.2. Su disponibilidad al cierre de D no está documentada.",
            "candidatas": candidatas.columns.tolist(),
        },
        "selection": {
            "script": "src/seleccion_variables.py",
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


if __name__ == "__main__":
    run_pipeline()
