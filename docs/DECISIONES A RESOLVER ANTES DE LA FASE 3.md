# Decisiones a resolver antes de la Fase 3

## Punto de partida

La Fase 2 deja un conjunto reproducible de **12 predictoras**, elegido por MAE en Validation. El CSV local y el linaje corresponden a ese conjunto. Las decisiones de abajo se pueden resolver en Validation antes de evaluar el modelo final en Test.

## 1. Cuántas medias móviles de frío incluir

El conjunto actual incluye medias de **3, 7 y 14 días calendario**, calculadas con cierres hasta el día de emisión `D`. En Train, la correlación de Spearman fue **0,950** entre 3 y 7 días, **0,923** entre 3 y 14, y **0,974** entre 7 y 14.

Para medir el aporte de cada ventana mantuvimos fijas las otras nueve predictoras y reentrenamos Ridge y Random Forest en Train. Conservamos el corte y el orden original de las columnas. Ridge usó `StandardScaler` ajustado en Train y `alpha=10`. El bosque usó 200 árboles, profundidad máxima 8, al menos 6 observaciones por hoja y semilla 42.

| Ventanas de frío | MAE de Ridge en Validation (kW) | MAE de Random Forest en Validation (kW) |
|---|---:|---:|
| Ninguna | 2.604,39 | 2.843,52 |
| 3 días | 2.497,38 | 2.800,16 |
| 7 días | 2.242,90 | 2.535,60 |
| 14 días | 2.193,52 | 2.440,55 |
| 3 y 7 días | 2.241,08 | 2.512,53 |
| 3 y 14 días | 2.161,15 | **2.407,12** |
| 7 y 14 días | 2.166,88 | 2.454,82 |
| 3, 7 y 14 días | **2.149,43** | 2.455,47 |

**Decisión pendiente.** Si se conserva una sola media, la de 14 días rindió mejor con ambos estimadores. Las ventanas de **3 y 14 días** son la alternativa de dos columnas a comparar con el conjunto actual. Quitar la de siete días empeoró Ridge en 11,72 kW y mejoró Random Forest en 48,35 kW en Validation. El resultado depende del estimador. La media de siete días se calcula también como baseline obligatorio, independientemente de que el modelo la reciba como predictora.

## 2. Si se incorpora radiación solar de Open-Meteo

Lio calculó por separado las variables del historial de consumo de frío y las derivadas meteorológicas de Tecate. La temperatura y la radiación histórica se conservaron en el diagnóstico del notebook. En Train, `radiacion_roll7` obtuvo **Spearman 0,772** con el consumo de frío de `D+1`. La media usa la radiación de hasta el propio día `D`, incluido su acumulado diario.

La radiación y la temperatura quedaron fuera del conjunto candidato de 2.3 por una decisión conservadora de disponibilidad. La copia histórica indica a qué fecha corresponde cada observación. La hora a la que pudo consultarse queda sin documentar. Para un pronóstico emitido al cierre de `D`, el valor que use el modelo tiene que estar disponible en ese momento. La consigna recomienda Open-Meteo como posible fuente. [Documentación histórica de Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api)

La selección vigente incluye `bodega_x_calor_estacional`, que combina producción de Bodega con un índice calculado a partir del mes. Ese índice se conoce al cierre porque depende del calendario.

**Decisión pendiente.** Identificar una fuente meteorológica y una hora de consulta reproducibles para el histórico y para la predicción futura. Si se cumple esa condición, comparar en Validation el conjunto vigente, el conjunto con `radiacion_roll7` añadida y una variante que evalúe reemplazar la interacción estacional por la señal de radiación. Usar las mismas fechas y la misma configuración de entrenamiento en cada comparación. La correlación de 0,772 fundamenta la prueba. El aporte adicional al MAE sigue pendiente de medir.
