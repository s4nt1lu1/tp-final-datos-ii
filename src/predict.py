"""Fase 4. Pronóstico del consumo de frío (kW) del día siguiente a partir de un Excel nuevo.

Uso desde la raíz del proyecto:
    python src/predict.py "data/Totalizadores Planta 2024_2025.xlsx"
    python src/predict.py archivo.xlsx --version v1.0.0 --salida results/predicciones.csv

Cadena completa (todo se reutiliza de src/preprocessing_pipeline.py, nada se reimplementa):
    Excel -> limpieza horaria (Parte A, con la calibración del primer año de entrenamiento)
          -> cierre diario 23:59 -> PreparadorFeatures.transform (ajustado en Train)
          -> columnas elegidas en Validation -> modelo del registro

La predicción es DIARIA: la fila con fecha D contiene el consumo estimado del día D+1.
"""

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:  # Permite `python src/predict.py` aunque los módulos importen `src.`
    sys.path.insert(0, str(RAIZ))

import joblib
import numpy as np
import pandas as pd

from src import preprocessing_pipeline as pp
from src.auxiliar_functions import huella

REGISTRY = RAIZ / "models" / "model_registry.json"
SALIDA_DEFECTO = RAIZ / "results" / "predicciones.csv"
CLAVES_BUNDLE = ("modelo", "preparador", "columnas", "df_mad_ee", "umbral_salto")
DIAS_DE_CONTEXTO = 16

# Cada derivada de planta usa estas mediciones. Las variables de calendario
# salen de DIA y las ventanas de Frío salen del cierre del totalizador.
INSUMOS_DE_PLANTA = {
    "ratio_frio_servicios": ("frio", "servicios"),
    "ratio_bodega_elaboracion": ("bodega", "elaboracion"),
    "eficiencia_frio_vs_meta": ("ee_frio_hl", "meta_frio"),
    "bodega_x_calor_estacional": ("hl_bodega",),
}
VARIABLES_DE_FRIO_Y_CALENDARIO = {
    "frio_hoy_d", "frio_lag_1", "frio_lag_2", "frio_lag_7", "frio_diff_1",
    "frio_roll_mean_3", "frio_roll_mean_7", "frio_roll_mean_14",
    "frio_roll_std_7", "mes_numero", "calor_estacional",
}


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
    """Carga el bundle {'modelo', 'preparador', 'columnas', 'version'} desde el registry.

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

    bundle = joblib.load(ruta)
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


def _columnas_requeridas(preparador, columnas_modelo):
    """Identifica qué mediciones hacen falta para las predictoras elegidas."""
    necesarias = {pp.COL_TARGET}  # También se usa para validar y limpiar cada jornada.
    for columna in columnas_modelo:
        if columna in preparador.operativas_:
            necesarias.add(columna)
        elif columna in INSUMOS_DE_PLANTA:
            for insumo in INSUMOS_DE_PLANTA[columna]:
                necesarias.add(pp.COLUMNAS_BASE[insumo])
        elif columna not in VARIABLES_DE_FRIO_Y_CALENDARIO:
            raise ValueError(f"No están definidos los insumos de la predictora '{columna}'.")
    return sorted(necesarias)


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
# Preprocesamiento (reutiliza la Fase 2)
# ---------------------------------------------------------------------------

def _leer_excel(filepath, requeridas):
    """Lee un Excel y valida por nombre las hojas y las mediciones necesarias."""
    ruta = Path(filepath)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
    if ruta.suffix.lower() not in (".xlsx", ".xlsm"):
        raise ValueError(f"Se esperaba un archivo Excel (.xlsx), no '{ruta.suffix}'.")

    _validar_hojas(ruta, requeridas)
    tabla, *_ = pp.unificar_archivos_excel([("entrada", ruta)])
    _validar_columnas(tabla, requeridas, ruta)
    return tabla


def load_and_preprocess(filepath, bundle, *, historial=None, desde=None):
    """Devuelve las predictoras y fechas D del Excel que se quiere pronosticar."""
    ruta = Path(filepath)
    preparador = bundle["preparador"]
    requeridas = _columnas_requeridas(preparador, bundle["columnas"])
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
    df_trabajo, df_casos = pp.evaluar_casos_especiales_y_reparar_pico_01(
        df_unificado, df_mad_ee, target_col=pp.COL_TARGET
    )
    df_horario, dias_descartados, _ = pp.filtrar_dias_invalidos(
        df_trabajo, df_mad_ee, df_casos, target_col=pp.COL_TARGET, umbral_salto=umbral_salto
    )
    for dia in dias_descartados.itertuples(index=False):
        if pd.Timestamp(dia.DIA) >= inicio:
            print(f"AVISO: se omitió {dia.DIA}. {dia.Motivo_Detallado}", file=sys.stderr)

    if df_horario.empty:
        raise ValueError(f"Ningún día de '{ruta.name}' pasó la limpieza (incompletos o inválidos).")
    df_horario = df_horario.drop(columns=["_archivo_origen"], errors="ignore")

    # Cierre diario 23:59 y features con el preparador ajustado en Train.
    df_diario, _ = pp.aggregate_daily_data(df_horario)

    # El preparador guardado calcula las candidatas de Fase 2. Completamos
    # con NaN solamente las columnas auxiliares ausentes, tras haber validado
    # los insumos de todas las predictoras elegidas y limpiado las jornadas.
    auxiliares = set(preparador.operativas_)
    auxiliares.update(preparador.planta_._columnas_a_limpiar())
    auxiliares.add(pp.COLUMNAS_BASE["meta_frio"])
    faltantes_auxiliares = sorted(auxiliares.difference(df_diario.columns))
    df_diario = df_diario.reindex(columns=df_diario.columns.tolist() + faltantes_auxiliares)
    X_todo = preparador.transform(df_diario)

    # El contexto participa en las ventanas y la imputación. Solo emitimos
    # pronósticos del archivo principal a partir de la fecha solicitada.
    X_todo = X_todo.loc[X_todo.index >= inicio]
    if X_todo.empty:
        raise ValueError(f"No hay jornadas válidas en '{ruta.name}' desde {inicio.date()}.")
    primer_pronostico = X_todo.index.min()
    inicio_contexto = primer_pronostico - pd.Timedelta(days=DIAS_DE_CONTEXTO)
    cierres_previos = df_diario["DIA"].between(inicio_contexto, primer_pronostico, inclusive="left")
    cantidad_contexto = int(cierres_previos.sum())
    if cantidad_contexto < DIAS_DE_CONTEXTO:
        print(
            f"AVISO: hay {cantidad_contexto} cierres válidos en los {DIAS_DE_CONTEXTO} días "
            f"previos al primer pronóstico ({primer_pronostico.date()}). "
            "Las primeras ventanas pueden estar incompletas.",
            file=sys.stderr,
        )

    faltan = [c for c in bundle["columnas"] if c not in X_todo.columns]
    if faltan:
        raise ValueError(f"El preparador no generó las columnas que espera el modelo: {faltan}")
    X = X_todo[bundle["columnas"]]  # Mismo orden que en el entrenamiento.

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
    except Exception as error:  # Cualquier otro fallo también termina con un mensaje legible.
        sys.exit(f"ERROR inesperado ({type(error).__name__}): {error}")

    argumentos.salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(argumentos.salida, index=False)
    print(f"{len(resultado)} predicciones generadas en {argumentos.salida}")


if __name__ == "__main__":
    main()