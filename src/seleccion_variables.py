"""Sección 2.3. Ajuste de selectores en Train y comparación en Validation."""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.auxiliar_functions import SEMILLA


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
