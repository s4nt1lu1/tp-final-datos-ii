import hashlib
import json
from pathlib import Path
from typing import Dict

# Anclaje determinista respecto a la raíz del repositorio
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

SEMILLA = 42

def huella(ruta: Path, algoritmo: str = "sha256", bloque: int = 1 << 20) -> str:
    """Devuelve el resumen hexadecimal del archivo, leyéndolo de a 1 MiB."""
    h = hashlib.new(algoritmo)
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()

def generar_checksums_crudos() -> None:
    """Calcula y persiste en data/checksums.json las huellas de los datos crudos."""
    archivos_crudos = [
        "Totalizadores Planta 2020_2022.xlsx",
        "Totalizadores Planta - 2021_2023.xlsx",
        "Totalizadores Planta 2022_2023.xlsx",
    ]
    registro: Dict[str, str] = {}
    for nombre in archivos_crudos:
        ruta = RAW_DATA_DIR / nombre
        if not ruta.exists():
            raise FileNotFoundError(f"Falta el archivo requerido en data/raw/: {nombre}")
        registro[nombre] = huella(ruta, algoritmo="sha256")

    ruta_salida = DATA_DIR / "checksums.json"
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=4)
    print(f"Checksums generados exitosamente en: {ruta_salida}")

def verificar_checksums(archivo_referencia: Path = DATA_DIR / "checksums.json") -> bool:
    """Valida la integridad de data/raw/ contra data/checksums.json."""
    if not archivo_referencia.exists():
        raise FileNotFoundError(f"No existe el archivo de sumas: {archivo_referencia}")

    with open(archivo_referencia, "r", encoding="utf-8") as f:
        referencias = json.load(f)

    for nombre, hash_esperado in referencias.items():
        ruta = RAW_DATA_DIR / nombre
        if not ruta.exists():
            raise FileNotFoundError(f"Archivo faltante en data/raw/: {nombre}")
        hash_actual = huella(ruta, algoritmo="sha256")
        if hash_actual != hash_esperado:
            raise ValueError(
                f"Discrepancia en {nombre}.\nLocal:    {hash_actual}\nEsperado: {hash_esperado}"
            )
    print("Verificación de integridad completada: todos los archivos coinciden.")
    return True

if __name__ == "__main__":
    generar_checksums_crudos()