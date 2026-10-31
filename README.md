# Laboratorio de Datos II - Trabajo Práctico Final

Proyecto de la materia **Laboratorio de Datos II**, perteneciente a la carrera de **Ingeniería en Inteligencia Artificial** de la **Universidad Nacional del Litoral (UNL)**.

El objetivo de este proyecto es desarrollar un pipeline de Machine Learning completo, reproducible y auditable, para predecir el consumo energético de un sistema de refrigeración (`Frio (kW)`) del día siguiente ($D+1$) en una planta cervecera, a partir de registros acumulados durante el periodo (2020–2023).

---

## Requisitos y Entorno

- **Versión de Python:** 3.12 (fijada mediante `.python-version`)
- **Gestor de entorno y paquetes:** `uv`
- **Semilla global de reproducibilidad:** `SEMILLA = 42`

### Reproducción del Entorno Virtual

Para garantizar la reproducibilidad determinista del entorno aislado, ejecutar en la raíz del proyecto:

#### 1. Crear el entorno virtual con Python 3.12
```bash
uv venv
```
*(En Windows activar mediante `.venv\Scripts\activate`; en Linux/macOS con `source .venv/bin/activate`)*.

#### 2. Instalar dependencias exactas
```bash
uv pip install -r requirements.txt
```

---

## Gestión de Datos y Verificación de Integridad

Los archivos Excel crudos están excluidos del repositorio Git mediante `.gitignore` y deben ubicarse en la carpeta `data/raw/`:
- `Totalizadores Planta 2020_2022.xlsx`
- `Totalizadores Planta - 2021_2023.xlsx`
- `Totalizadores Planta 2022_2023.xlsx`

Para auditar la integridad de los archivos locales contra las sumas de verificación (`data/checksums.json`), ejecutar:
```bash
python -c "from src.auxiliar_functions import verificar_checksums; verificar_checksums()"
```

---

## Estructura del Proyecto

```text
tp_final_datos_2/
├── data/
│   ├── raw/                           # [IGNORADO EN GIT] Excels crudos
│   ├── processed/
│   │   └── data_lineage.json          # Trazabilidad de transformaciones y reglas aplicadas
│   └── checksums.json                 # Huellas SHA-256 de archivos
│   └── model_registry.json            #  Registro local
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── auxiliar_functions.py          # funciones auxiliares
├── results/
├── .gitignore                         # Reglas de exclusión d
├── .python-version                    # Fijación de Python 3.12 para uv
├── requirements.txt                   # Dependencias congeladas con uv pip freeze
└── README.md                          # Guia

---