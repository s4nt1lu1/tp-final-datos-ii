"""Fase 4. Pronóstico del consumo de frío (kW) del día siguiente a partir de un Excel nuevo.

Uso desde la raíz del proyecto:
    python -m src.predict "data/raw/Totalizadores Planta de Cerveza 2024_2025.xlsx"
    python -m src.predict archivo.xlsx --salida results/predicciones.csv
    python -m src.predict archivo.xlsx --historial archivo_anterior.xlsx --desde 2024-07-01

Cadena completa (todo se reutiliza de los pipelines de entrenamiento, nada se reimplementa):
    Excel -> limpieza horaria de src/auxiliar_functions.py (con la calibración del primer año de
             entrenamiento) -> cierre diario 23:59
          -> pipeline de src/preprocessing_pipeline.py (decisiones de la EDA y pasos ajustados
             en Train) -> candidatas de src/preprocessing_pipeline.py (variables creadas ajustadas
             en Train) -> columnas elegidas -> modelo

El modelo, el pipeline ajustado y la calibración se guardan juntos en el .pkl que indica
`models/model_registry.json`. Las hojas y columnas del Excel se buscan por nombre.

La predicción es DIARIA: la fila con fecha D contiene el consumo estimado del día D+1.
"""

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:  # permite `python src/predict.py` aunque los módulos importen `src.`
    sys.path.insert(0, str(RAIZ))

import joblib
import numpy as np
import pandas as pd

from src import auxiliar_functions as af
from src import preprocessing_pipeline as pp
from src.auxiliar_functions import huella

REGISTRY = RAIZ / "models" / "model_registry.json"
SALIDA_DEFECTO = RAIZ / "results" / "predicciones.csv"
CLAVES_BUNDLE = ("modelo", "columnas", "pipeline", "columnas_entrada", "cfg_seleccion",
                 "variables_creadas", "requeridas", "medianas_train", "df_mad_ee", "umbral_salto")


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def _version_key(version):
    """'v1.10.0' -> (1, 10, 0). Se compara como números, no como texto."""
    try:
        return tuple(int(parte) for parte in str(version).lstrip("vV").split("."))
    except ValueError as error:
        raise ValueError(f"Versión inválida en el registry: '{version}'") from error


def load_model(version="latest"):
    """Carga desde el registry el bundle del modelo con su preprocesamiento.

    'latest' elige la versión semántica más alta.
    """
    if not REGISTRY.is_file():
        raise FileNotFoundError(f"No existe el registro de modelos: {REGISTRY}")
    try:
        registro = json.loads(REGISTRY.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{REGISTRY.name} no es un JSON válido: {error}") from error
    if not registro:
        raise ValueError(f"{REGISTRY.name} está vacío: entrená un modelo con src/train_model.py.")

    if version == "latest":
        version = max(registro, key=_version_key)
    if version not in registro:
        raise KeyError(f"La versión '{version}' no está en el registry. Disponibles: {sorted(registro)}")

    entrada = registro[version]
    ruta = RAIZ / entrada["path"]
    if not ruta.is_file():
        raise FileNotFoundError(f"El registry apunta a {ruta}, pero el archivo no existe.")
    if "sha256" in entrada and huella(ruta) != entrada["sha256"]:
        raise ValueError(f"El SHA-256 de {ruta.name} no coincide con el del registry: archivo alterado.")

    try:
        bundle = joblib.load(ruta)
    except (ModuleNotFoundError, AttributeError, ImportError) as error:
        # El .pkl guarda referencias a las clases del código con que se entrenó.
        commit = str(entrada.get("git_commit", ""))[:7]
        raise ValueError(
            f"La versión {version} se entrenó con código que ya no está en esta rama ({error}). "
            f"Para usarla, hacé checkout del tag {version} o del commit {commit} del registro."
        ) from error
    if not isinstance(bundle, dict) or bundle.get("tipo") != "v2":
        raise ValueError(f"{ruta.name} no es un modelo del pipeline actual. Para usarlo, hacé checkout "
                         f"del tag {version}.")
    faltan = [clave for clave in CLAVES_BUNDLE if clave not in bundle]
    if faltan:
        raise ValueError(f"El archivo {ruta.name} no tiene las claves esperadas: {faltan}")
    return bundle


# ---------------------------------------------------------------------------
# Validación del Excel de entrada (por nombre)
# ---------------------------------------------------------------------------
def _separar_hoja(columna):
    """'Frio (Kw)_Consolidado EE' -> ('Frio (Kw)', 'Consolidado EE'). Misma convención que la Fase 2."""
    nombre, _, hoja = columna.rpartition("_")
    return nombre, hoja


def _validar_hojas(ruta, columnas):
    """Comprueba, antes de cualquier cálculo pesado, que estén todas las hojas necesarias."""
    try:
        presentes = {hoja.strip() for hoja in pd.ExcelFile(ruta).sheet_names}
    except Exception as error:
        raise ValueError(f"No se pudo leer '{ruta.name}' como Excel (.xlsx): {error}") from error
    necesarias = sorted({_separar_hoja(c)[1] for c in columnas})
    faltan = [hoja for hoja in necesarias if hoja not in presentes]
    if faltan:
        raise ValueError(
            f"Faltan hojas en '{ruta.name}': {faltan}. Hojas presentes: {sorted(presentes)}"
        )


def _validar_columnas(df, columnas, ruta):
    """Lista TODAS las columnas faltantes, indicando la hoja de cada una."""
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        detalle = "; ".join(f"'{n}' en la hoja '{h}'" for n, h in map(_separar_hoja, faltan))
        raise ValueError(f"Faltan columnas en '{ruta.name}': {detalle}")


# ---------------------------------------------------------------------------
# Preprocesamiento (reutiliza las fases 1 y 2)
# ---------------------------------------------------------------------------
def _leer_excel(filepath, requeridas):
    """Lee un Excel y valida por nombre las hojas y las mediciones necesarias."""
    ruta = Path(filepath)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
    if ruta.suffix.lower() not in (".xlsx", ".xlsm"):
        raise ValueError(f"Se esperaba un archivo Excel (.xlsx), no '{ruta.suffix}'.")

    _validar_hojas(ruta, requeridas)
    tabla, *_ = af.unificar_archivos_excel([("entrada", ruta)])
    _validar_columnas(tabla, requeridas, ruta)
    return tabla


def _jornadas_limpias(ruta, bundle, requeridas, historial=None, desde=None):
    """Lee el Excel (y el historial previo, si lo hay), aplica la limpieza horaria con la
    calibración guardada y devuelve los cierres diarios y la primera fecha a pronosticar."""
    entrada = _leer_excel(ruta, requeridas)
    primer_dia = entrada["DIA"].min()
    inicio = primer_dia
    if desde is not None:
        inicio = max(primer_dia, pd.to_datetime(desde, format="%Y-%m-%d", errors="raise"))

    # Distinguimos fechas ausentes del Excel y fechas descartadas después de leerlo.
    calendario = pd.date_range(inicio, entrada["DIA"].max(), freq="D")
    fechas_ausentes = calendario.difference(entrada["DIA"])
    if len(fechas_ausentes) > 0:
        fechas = ", ".join(fechas_ausentes.strftime("%Y-%m-%d"))
        print(f"AVISO: faltan filas de estas fechas en '{ruta.name}': {fechas}.", file=sys.stderr)

    # El otro Excel solo aporta pasado. Las fechas nuevas se toman del Excel principal.
    df_unificado = entrada
    if historial is not None:
        previo = _leer_excel(historial, requeridas)
        previo = previo.loc[previo["DIA"] < primer_dia]
        df_unificado = pd.concat([previo, entrada], ignore_index=True)

    # Aplicamos la calibración guardada al entrenar, también sobre el historial previo.
    df_mad_ee = bundle["df_mad_ee"]
    umbral_salto = bundle["umbral_salto"]
    df_trabajo, df_casos = af.evaluar_casos_especiales_y_reparar_pico_01(
        df_unificado, df_mad_ee, target_col=af.COL_TARGET
    )
    df_horario, dias_descartados, _ = af.filtrar_dias_invalidos(
        df_trabajo, df_mad_ee, df_casos, target_col=af.COL_TARGET, umbral_salto=umbral_salto
    )
    for dia in dias_descartados.itertuples(index=False):
        if pd.Timestamp(dia.DIA) >= inicio:
            print(f"AVISO: se omitió {dia.DIA}. {dia.Motivo_Detallado}", file=sys.stderr)

    if df_horario.empty:
        raise ValueError(f"Ningún día de '{ruta.name}' pasó la limpieza (incompletos o inválidos).")
    df_horario = df_horario.drop(columns=["_archivo_origen"], errors="ignore")

    # Cierre diario 23:59.
    df_diario, _ = af.aggregate_daily_data(df_horario)
    return df_diario, inicio


def _avisar_contexto(df_diario, primer_pronostico, dias):
    """Avisa si faltan cierres previos para las ventanas más largas."""
    inicio_contexto = primer_pronostico - pd.Timedelta(days=dias)
    cierres_previos = df_diario["DIA"].between(inicio_contexto, primer_pronostico, inclusive="left")
    cantidad = int(cierres_previos.sum())
    if cantidad < dias:
        print(
            f"AVISO: hay {cantidad} cierres válidos en los {dias} días previos al primer pronóstico "
            f"({primer_pronostico.date()}). Las primeras ventanas pueden estar incompletas.",
            file=sys.stderr,
        )


def load_and_preprocess(filepath, bundle, *, historial=None, desde=None):
    """Devuelve las predictoras y las fechas D del Excel que se quiere pronosticar.

    Reutiliza el pipeline del preprocesamiento y las variables creadas de la selección, ambos
    ajustados con Train y guardados en el bundle.
    """
    ruta = Path(filepath)
    df_diario, inicio = _jornadas_limpias(ruta, bundle, bundle["requeridas"], historial, desde)

    # Mismas columnas de entrada que al entrenar. Las que no trae el Excel y el modelo no
    # necesita quedan vacías y las completa la imputación con lo aprendido en Train.
    X_dia = df_diario.set_index("DIA").sort_index().apply(pd.to_numeric, errors="coerce")
    ausentes = [c for c in bundle["columnas_entrada"] if c not in X_dia.columns]
    X_dia = X_dia.reindex(columns=bundle["columnas_entrada"])
    if ausentes:
        print(f"AVISO: '{ruta.name}' no trae {len(ausentes)} columnas del entrenamiento que el modelo "
              "no usa; se completan con la imputación aprendida en Train.", file=sys.stderr)

    X_proc = bundle["pipeline"].transform(X_dia)
    directas = pp.candidatas_directas(X_proc, bundle["cfg_seleccion"])
    candidatas = pd.concat([directas, bundle["variables_creadas"].transform(directas)], axis=1)
    candidatas = candidatas.loc[candidatas.index >= inicio]
    if candidatas.empty:
        raise ValueError(f"No hay jornadas válidas en '{ruta.name}' desde {inicio.date()}.")
    _avisar_contexto(df_diario, candidatas.index.min(), bundle.get("dias_contexto", 28))

    faltan = [c for c in bundle["columnas"] if c not in candidatas.columns]
    if faltan:
        raise ValueError(f"El preprocesamiento no generó las columnas que espera el modelo: {faltan}")
    X = candidatas[bundle["columnas"]]  # mismo orden que en el entrenamiento
    incompletas = X.isna().any(axis=1)
    if incompletas.any():
        print(f"AVISO: {int(incompletas.sum())} días sin historia suficiente para todas las ventanas "
              f"(desde {X.index[incompletas].min().date()}); se completan con las medianas de Train.",
              file=sys.stderr)
        X = X.fillna(bundle["medianas_train"])

    no_finitas = X.columns[~np.isfinite(X).all()].tolist()
    if no_finitas:
        raise ValueError(f"Predictoras sin valor finito tras el preprocesamiento: {no_finitas}")
    return X, pd.DatetimeIndex(X.index)


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
def predict_consumption(filepath, version="latest", *, historial=None, desde=None):
    """Una fila por día D; 'prediccion_frio_kw' es el consumo estimado de D+1."""
    bundle = load_model(version)
    X, fechas = load_and_preprocess(filepath, bundle, historial=historial, desde=desde)
    predicciones = bundle["modelo"].predict(X)
    return pd.DataFrame({
        "fecha": fechas.strftime("%Y-%m-%d"),
        "prediccion_frio_kw": predicciones,
    })


def main():
    parser = argparse.ArgumentParser(description="Predice el consumo de frío (kW) del día siguiente.")
    parser.add_argument("archivo", help="Excel con los totalizadores horarios.")
    parser.add_argument("--version", default="latest", help="Versión del registry (por defecto, la más alta).")
    parser.add_argument("--salida", type=Path, default=SALIDA_DEFECTO, help="CSV de salida.")
    parser.add_argument("--historial", type=Path, help="Excel con días anteriores al archivo principal.")
    parser.add_argument("--desde", help="Primera fecha D que se quiere exportar (AAAA-MM-DD).")
    argumentos = parser.parse_args()

    try:
        resultado = predict_consumption(
            argumentos.archivo, argumentos.version,
            historial=argumentos.historial, desde=argumentos.desde,
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        sys.exit(f"ERROR: {error}")
    except Exception as error:  # cualquier otro fallo también termina con un mensaje legible
        sys.exit(f"ERROR inesperado ({type(error).__name__}): {error}")

    argumentos.salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(argumentos.salida, index=False)
    print(f"{len(resultado)} predicciones generadas en {argumentos.salida}")


if __name__ == "__main__":
    main()
