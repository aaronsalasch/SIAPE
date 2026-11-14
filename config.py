"""Configuración global compartida por todos los módulos del proyecto SIAPE.

SIAPE = Sistema Inteligente de Análisis, Predicción y trazabilidad Energética.
Todas las rutas son relativas a la raíz del proyecto, para que funcione igual en
Windows (D:\\Documentos\\ICD) y en Linux.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

# ---- Rutas ----
DIR_DATA_RAW = RAIZ / "data" / "raw"
DIR_DATA_PROC = RAIZ / "data" / "processed"
DIR_MODELOS = RAIZ / "models"
DIR_OUTPUTS = RAIZ / "outputs"
DIR_FIGURAS = DIR_OUTPUTS / "figuras"
DIR_REPORTES = DIR_OUTPUTS / "reportes"
DIR_EVIDENCIAS = DIR_OUTPUTS / "evidencias"
DIR_BLOCKCHAIN = RAIZ / "blockchain_data"

CSV_CONSUMO_RAW = DIR_DATA_RAW / "consumo_energetico_raw.csv"
CSV_REPORTES_RAW = DIR_DATA_RAW / "reportes_usuarios_raw.csv"
CSV_CONSUMO_PROC = DIR_DATA_PROC / "consumo_energetico_procesado.csv"
CSV_REPORTES_PROC = DIR_DATA_PROC / "reportes_procesados.csv"

MODELO_CONSUMO = DIR_MODELOS / "modelo_consumo.joblib"
METRICAS_ML = DIR_MODELOS / "metricas_modelos.json"
MODELO_KMEANS = DIR_MODELOS / "kmeans_perfiles.joblib"
MODELO_SENTIMIENTO = DIR_MODELOS / "modelo_sentimiento.joblib"
METRICAS_PNL = DIR_MODELOS / "metricas_pnl.json"

CADENA_JSON = DIR_BLOCKCHAIN / "cadena.json"

# ---- Reproducibilidad ----
SEMILLA = 42

# ---- Periodo simulado ----
FECHA_INICIO = "2026-01-01 00:00:00"
DIAS_SIMULADOS = 182          # ~6 meses, frecuencia horaria
FRACCION_TEST = 0.20          # último 20 % del periodo (división temporal)

# ---- Instalaciones simuladas (Veracruz, México) ----
# base_kwh: consumo medio por hora aproximado; superficie en m2.
INSTALACIONES = [
    {"id": "RES-001", "tipo": "residencial", "ciudad": "Veracruz",      "superficie_m2": 90,   "ocupantes": 4,   "base_kwh": 0.9},
    {"id": "RES-002", "tipo": "residencial", "ciudad": "Boca del Río",  "superficie_m2": 140,  "ocupantes": 5,   "base_kwh": 1.4},
    {"id": "RES-003", "tipo": "residencial", "ciudad": "Xalapa",        "superficie_m2": 75,   "ocupantes": 3,   "base_kwh": 0.6},
    {"id": "COM-001", "tipo": "comercial",   "ciudad": "Veracruz",      "superficie_m2": 300,  "ocupantes": 15,  "base_kwh": 8.0},
    {"id": "COM-002", "tipo": "comercial",   "ciudad": "Córdoba",       "superficie_m2": 220,  "ocupantes": 10,  "base_kwh": 5.5},
    {"id": "COM-003", "tipo": "comercial",   "ciudad": "Boca del Río",  "superficie_m2": 500,  "ocupantes": 30,  "base_kwh": 14.0},
    {"id": "IND-001", "tipo": "industrial",  "ciudad": "Coatzacoalcos", "superficie_m2": 2500, "ocupantes": 80,  "base_kwh": 95.0},
    {"id": "IND-002", "tipo": "industrial",  "ciudad": "Orizaba",       "superficie_m2": 1800, "ocupantes": 60,  "base_kwh": 70.0},
    {"id": "EDU-001", "tipo": "educativa",   "ciudad": "Veracruz",      "superficie_m2": 4000, "ocupantes": 600, "base_kwh": 40.0},
    {"id": "EDU-002", "tipo": "educativa",   "ciudad": "Xalapa",        "superficie_m2": 2600, "ocupantes": 350, "base_kwh": 26.0},
]

TIPOS_INSTALACION = ["residencial", "comercial", "industrial", "educativa"]
CIUDADES = sorted({i["ciudad"] for i in INSTALACIONES})

# Dispositivos/equipos que aparecen en reportes y que el NER debe reconocer
DISPOSITIVOS = [
    "aire acondicionado", "minisplit", "refrigerador", "transformador", "medidor",
    "compresor", "bomba de agua", "luminarias", "calentador", "motor", "inversor",
    "servidor", "elevador", "ventilador", "lavadora", "horno",
]
ORGANIZACIONES = ["CFE", "Comisión Federal de Electricidad", "PROFECO", "CENACE", "TecNM", "ITV"]

# Dificultad de prueba de trabajo de la blockchain (nº de ceros iniciales del hash)
DIFICULTAD_POW = 4

for _d in (DIR_DATA_RAW, DIR_DATA_PROC, DIR_MODELOS, DIR_FIGURAS, DIR_REPORTES, DIR_EVIDENCIAS, DIR_BLOCKCHAIN):
    _d.mkdir(parents=True, exist_ok=True)
