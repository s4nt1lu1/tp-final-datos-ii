"""Decisiones estructurales sobre las columnas (EDA).

Toma el dataset horario limpio (`data/processed/dataset_trabajo.csv`) y decide,
antes de cualquier ajuste estadístico, qué columnas no deben llegar al
preprocesamiento y cuáles son la misma medición con otro nombre:

    1. metadatos           : identificadores, columnas sin encabezado y la hora del último dato.
    2. componentes_frio    : `Frio (Kw)` es la suma exacta de 7 compresores Mycom, 3 grupos de
                             condensadores y 2 bombas de glicol. Sus componentes no agregan
                             información sobre el frío de D, y su reparto cambia porque los
                             compresores rotan.
    3. proxies_frio        : medidores que contienen al frío (Sala de Máquinas, Servicios,
                             Planta y KW Gral Planta).
    4. residuales          : columnas calculadas por diferencia (total menos áreas). Tienen
                             valores negativos, imposibles en un medidor.
    5. renombradas         : la misma medición con otro nombre según el Excel. Se unen y se
                             conserva el nombre más reciente.
    6. copias              : columnas idénticas en toda la serie (la misma medición en una hoja
                             Consolidado y en una Totalizadores). Se conserva una.

Por qué estas decisiones se toman acá y no en el preprocesamiento: dependen de la
estructura de la planilla (nombres, hojas, identidades contables entre medidores),
no de valores del objetivo `y` ni de parámetros estimados. Ninguna usa `y`, así que
pueden tomarse con la serie completa sin fuga de información. Las decisiones que
dependen de estadísticos (nulos, outliers, imputación, columnas constantes) quedan
para el preprocesamiento, que se ajusta solo con Train.

Salida en `results/eda/`:
    decisiones_columnas.csv : una fila por columna con su decisión y su motivo.
    resumen.json            : configuración, SHA-256 de la entrada y las listas que lee
                              `src/preprocessing_pipeline.py`.

Uso desde la raíz del proyecto:
    python -m src.eda
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from src import limpieza_horaria as lh
from src.auxiliar_functions import RESULTS_DIR, huella

COL_FECHA = lh.DEFAULT_DATE_COL
COL_FRIO = lh.DEFAULT_TARGET_COL
CARPETA_SALIDA = RESULTS_DIR / "eda"
RUTA_RESUMEN = CARPETA_SALIDA / "resumen.json"
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
def cargar_datos(input_csv: Path = lh.DEFAULT_INPUT_CSV,
                 checksums_path: Path = lh.DEFAULT_CHECKSUMS_JSON) -> Dict[str, object]:
    """Horario numérico y cierres diarios (23:59) de `dataset_trabajo.csv`, con su SHA-256 verificado."""
    with open(checksums_path, encoding="utf-8") as f:
        registrado = lh._sha_registrado(json.load(f), CLAVE_ENTRADA)
    sha = huella(input_csv)
    if registrado is None or sha != registrado:
        raise ValueError("dataset_trabajo.csv no coincide con el SHA-256 registrado en checksums.json.")
    horario, _ = lh.load_cleaned_data(input_csv)
    diario, _ = lh.aggregate_daily_data(horario)
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


def ejecutar(cfg: Optional[ConfigEDA] = None, salida: Path = CARPETA_SALIDA,
             datos: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Calcula las decisiones y exporta la tabla y el resumen que lee el preprocesamiento."""
    cfg = cfg or ConfigEDA()
    datos = datos or cargar_datos()
    res = decidir_columnas(datos, cfg)
    tabla = res["tabla"]

    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    ruta_tabla = salida / "decisiones_columnas.csv"
    tabla.to_csv(ruta_tabla, index=False, encoding="utf-8", lineterminator="\n")

    identidad = {k: v for k, v in res["identidad"].items() if k not in ("coeficientes", "columnas")}
    identidad["coeficientes"] = res["identidad"]["coeficientes"]["coeficiente"].round(4).to_dict()
    resumen = {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "src/eda.py",
        "config": asdict(cfg),
        "entrada": {"archivo": CLAVE_ENTRADA, "sha256": datos["sha256"],
                    "dias": int(len(datos["diario"])), "columnas": int(datos["diario"].shape[1])},
        "identidad_frio": identidad,
        "conteo_por_categoria": tabla.loc[tabla["categoria"] != "", "categoria"].value_counts().to_dict(),
        "columnas_salida": int((tabla["decision"] == "conservar").sum()),
        "descartar": tabla.loc[tabla["decision"] == "descartar", "columna"].tolist(),
        "unificar": res["pares"],
        "sha256_decisiones": huella(ruta_tabla),
    }
    with open(salida / "resumen.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")

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


def cargar_decisiones(ruta: Path = RUTA_RESUMEN,
                      checksums_path: Path = lh.DEFAULT_CHECKSUMS_JSON) -> Dict[str, object]:
    """Lee las decisiones de la EDA y verifica que correspondan al dataset_trabajo.csv vigente."""
    if not Path(ruta).exists():
        raise FileNotFoundError(f"No existe {ruta}. Correr antes: python -m src.eda")
    with open(ruta, encoding="utf-8") as f:
        resumen = json.load(f)
    with open(checksums_path, encoding="utf-8") as f:
        registrado = lh._sha_registrado(json.load(f), CLAVE_ENTRADA)
    if resumen["entrada"]["sha256"] != registrado:
        raise ValueError("Las decisiones de la EDA se tomaron con otra versión de dataset_trabajo.csv. "
                         "Volver a correr: python -m src.eda")
    ruta_tabla = Path(ruta).parent / "decisiones_columnas.csv"
    if huella(ruta_tabla) != resumen["sha256_decisiones"]:
        raise ValueError(f"{ruta_tabla.name} cambió después de generar el resumen de la EDA.")
    resumen["tabla"] = pd.read_csv(ruta_tabla, keep_default_na=False)
    return resumen


def main() -> None:
    ejecutar()


if __name__ == "__main__":
    main()
