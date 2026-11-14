# SIAPE: Sistema Inteligente de Análisis, Predicción y Trazabilidad del Consumo Energético

Proyecto final de *Inteligencia Artificial en la Ciencia de Datos* (Instituto Tecnológico de Veracruz).

SIAPE es un prototipo que analiza el consumo eléctrico horario de 10 instalaciones de Veracruz (datos simulados), predice el consumo con Random Forest, agrupa perfiles diarios con K-Means, analiza 4,000 reportes de usuarios con PNL (sentimiento, entidades y generación de texto), registra el hash SHA-256 de los resultados en una Blockchain privada y presenta todo en un dashboard Flask.

## Integrantes

- Nestor Alexander Acosta Andrade: procesamiento de datos y Machine Learning
- Aaron Salas Chavez: Procesamiento de Lenguaje Natural
- Juan Alberto Santos Espinoza: Blockchain, integración y dashboard

## Requisitos

- Python 3.10 o superior (probado con 3.13)
- Dependencias en `requirements.txt` (pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, joblib, flask, python-docx)
- No requiere GPU ni conexión a internet para ejecutarse (la gráfica interactiva del pronóstico usa Chart.js y, sin red, se muestra una tabla de respaldo)

## Instalación

Windows:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Linux / macOS:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

```
python pipeline.py            # flujo completo (datos, ML, PNL, reporte, Blockchain, verificación)
python app/app.py             # dashboard en http://127.0.0.1:5000
```

Opciones de `pipeline.py`: `--datos`, `--ml`, `--pnl`, `--reporte`, `--registrar`, `--blockchain-demo`, `--verificar` (sin opciones equivale a `--todo`).

## Pruebas

```
python -m unittest tests.test_app tests.test_blockchain
```

Son 24 pruebas. Las de la aplicación requieren haber ejecutado antes `python pipeline.py`.

## Estructura

```
config.py        configuración (rutas, semilla, PoW)
pipeline.py      orquestador
app/             dashboard Flask (app.py, templates, static)
src/data/        generación y preparación de datos
src/ml/          modelo de consumo, K-Means, predicción
src/nlp/         procesamiento de lenguaje natural
src/blockchain/  cadena de bloques y prueba de alteración
src/reporte/     reporte automático
data/            datasets originales (raw) y procesados
models/          modelos (.joblib) y métricas (.json)
outputs/         evidencias, figuras y reportes
blockchain_data/ cadena.json
tests/           pruebas automatizadas
docs/            documentación
```

## Documentación

- Documento técnico completo: [docs/Documento_Tecnico_SIAPE.docx](docs/Documento_Tecnico_SIAPE.docx)
- [01 Análisis del problema](docs/01_analisis_del_problema.md)
- [02 Dataset](docs/02_dataset.md)
- [03 Modelo de Machine Learning](docs/03_modelo_ml.md)
- [04 Procesamiento de Lenguaje Natural](docs/04_pnl.md)
- [05 Blockchain](docs/05_blockchain.md)
- [06 Arquitectura](docs/06_arquitectura.md)
- [07 Conclusiones](docs/07_conclusiones.md)

Los datos son simulados con fines académicos; las métricas no representan desempeño con datos reales.
