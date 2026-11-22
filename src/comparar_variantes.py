"""Comparación de variantes del pipeline, solo en Validation (Test no se lee).

Variantes:
    * filtro de redundancia en la selección: desactivado o con umbral 0,90;
    * qué aprende el modelo: nivel de y, y − frío de D, y − media de 7 cierres, y / media de 7 cierres.

Para cada filtro se repite la selección (`src/seleccion_variables.py`) y la búsqueda de Optuna
(`src/train_model.py`) dejando que Optuna elija el objetivo con la validación interna en Train.
Después se evalúa en Validation:
    * el ganador de cada modelo (como en el modelado);
    * la mejor configuración de cada par modelo–objetivo (diagnóstico, no cambia la elección).

No escribe en `results/seleccion/`, `results/modelado/` ni en el log de experimentos.

Salida en `results/comparacion_variantes/`:
    comparacion_mejoras.csv      ganador de cada modelo en cada variante de filtro
    comparacion_por_objetivo.csv mejor configuración de cada par modelo–objetivo

Uso desde la raíz del proyecto:
    python -m src.comparar_variantes
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src import train_model as tm
from src import seleccion_variables as sv
from src.auxiliar_functions import RESULTS_DIR

SALIDA = RESULTS_DIR / "comparacion_variantes"
FILTROS = (None, 0.90)


def ejecutar(salida: Path = SALIDA) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    historial = sv.cargar_preprocesado()
    filas_comp, filas_obj = [], []

    for filtro in FILTROS:
        inicio = time.time()
        cfg_sel = sv.ConfigSeleccion(umbral_redundancia=filtro)
        candidatas = sv.construir_candidatas(historial, cfg_sel)
        d = sv.particiones(historial, candidatas)
        X_tr, y_tr, _ = d["train"]
        X_val, y_val, _ = d["validation"]
        columnas, _ = sv.filtrar_redundantes(X_tr, y_tr, cfg_sel)
        _, _, subconjuntos, _ = sv.seleccionar(X_tr[columnas], y_tr, X_val[columnas], y_val, cfg_sel)
        etiqueta = "sin filtro" if filtro is None else f"filtro {filtro}"

        cfg = tm.ConfigModelado(objetivos=tuple(tm.OBJETIVOS), registrar_log=False)
        estudios = {m: tm.optimizar(m, X_tr, y_tr, subconjuntos, cfg) for m in cfg.modelos}
        tabla, _, _ = tm.evaluar_en_validation(estudios, subconjuntos, X_tr, y_tr, X_val, y_val, None, cfg)
        for _, f in tabla.iterrows():
            filas_comp.append({"filtro": etiqueta, "modelo": f["modelo"], "conjunto": f["conjunto"],
                               "objetivo": json.loads(f["hiperparametros"]).get("objetivo_y"),
                               "n_columnas": f["n_columnas"], "mae_interno": f["mae_interno"],
                               "mae_val": f["mae"], "sesgo_val": f["sesgo"]})

        trials = tm.tabla_trials(estudios)
        for (modelo, objetivo), grupo in trials.groupby(["modelo", "objetivo_y"]):
            mejor = grupo.loc[grupo["mae_interno"].idxmin()]
            params = json.loads(mejor["hiperparametros"]) | ({"subsample_freq": 1} if modelo == "LightGBM" else {})
            cols = tm.columnas_de(subconjuntos[mejor["conjunto"]], objetivo)
            pred = tm.crear_estimador(modelo, params).fit(X_tr[cols], y_tr).predict(X_val[cols])
            filas_obj.append({"filtro": etiqueta, "modelo": modelo, "objetivo": objetivo, "trials": len(grupo),
                              "mae_interno": mejor["mae_interno"], "mae_val": mean_absolute_error(y_val, pred),
                              "sesgo_val": float(np.mean(pred - y_val))})
        print(f"[{etiqueta}] listo en {time.time() - inicio:.0f} s", flush=True)

    pd.DataFrame(filas_comp).to_csv(salida / "comparacion_mejoras.csv", index=False, lineterminator="\n")
    pd.DataFrame(filas_obj).to_csv(salida / "comparacion_por_objetivo.csv", index=False, lineterminator="\n")
    print(f"Resultados en {salida}")


if __name__ == "__main__":
    ejecutar()
