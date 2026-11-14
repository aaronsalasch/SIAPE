# PLAN DE DESARROLLO — SIAPE
**Sistema Inteligente de Análisis, Predicción y Trazabilidad Energética**
Proyecto final de "IA en la ciencia de datos" (ISC, TecNM – Instituto Tecnológico de Veracruz).
Tema: *Sistema inteligente para análisis y optimización del consumo energético*.

## Problema delimitado
Un operador de eficiencia energética en el estado de Veracruz administra 10 instalaciones
(residenciales, comerciales, industriales y educativas; ver `config.INSTALACIONES`).
Necesita: (1) **predecir el consumo eléctrico horario (kWh)** a partir de hora, día,
temperatura, humedad, tipo de instalación y consumo histórico; (2) **agrupar perfiles de
consumo** para detectar comportamientos similares/anómalos; (3) **analizar los reportes
de texto de los usuarios** (quejas, fallas, comentarios) para medir sentimiento y
extraer entidades (equipos, lugares, fechas, códigos de instalación, organizaciones);
(4) **generar un reporte automático** y (5) **registrar el hash del reporte y de las
predicciones en una Blockchain** para garantizar que nadie altere los resultados que
sustentan decisiones (p. ej. auditorías o ajustes tarifarios).

## Restricciones técnicas (OBLIGATORIAS)
- Python 3.10+ (aquí 3.13). **Solo** estas librerías: pandas, numpy, scikit-learn,
  scipy, matplotlib, seaborn, joblib, flask, jinja2 y la biblioteca estándar.
  **No hay nltk, spaCy, streamlit ni plotly** y no se puede instalar nada (sin red).
- Todas las rutas vienen de `config.py` (importar `import config` añadiendo la raíz al
  `sys.path` si hace falta: `sys.path.insert(0, str(Path(__file__).resolve().parents[N]))`).
- Semilla `config.SEMILLA` en todo lo aleatorio. Todo debe ser reproducible.
- Código, comentarios, docstrings, nombres de columnas y salidas **en español**.
- Cada script ejecutable con `python -m src.<paquete>.<modulo>` desde la raíz **y** importable.
- Matplotlib con backend "Agg". Figuras PNG a `config.DIR_FIGURAS`, dpi 110.
- Cada módulo deja **evidencia de ejecución** (log de texto con métricas/ejemplos) en
  `config.DIR_EVIDENCIAS` (exigido por los entregables 3, 4 y 5).

## Estructura del proyecto
```
ICD/
  config.py  PLAN.md  README.md  requirements.txt  pipeline.py
  data/raw/        consumo_energetico_raw.csv, reportes_usuarios_raw.csv
  data/processed/  consumo_energetico_procesado.csv, reportes_procesados.csv
  src/data/        generar_datos.py, generar_reportes.py, preparar_datos.py
  src/ml/          entrenar_modelo.py, clustering.py, predecir.py
  src/nlp/         procesamiento_texto.py, sentimiento.py, entidades.py, generacion.py, analizador.py
  src/blockchain/  blockchain.py, demo_alteracion.py
  src/reporte/     generar_reporte.py
  app/             app.py, templates/, static/
  models/  outputs/{figuras,reportes,evidencias}/  blockchain_data/  docs/
```

## CONTRATOS ENTRE MÓDULOS

### A. Datos de consumo (`src/data/generar_datos.py`, `src/data/preparar_datos.py`)
**Raw** `config.CSV_CONSUMO_RAW` — una fila por instalación y hora (10 × 182 × 24 ≈ 43,680 + duplicados).
Columnas: `id_registro, id_instalacion, tipo_instalacion, ciudad, fecha_hora (YYYY-MM-DD HH:MM:SS),
dia_semana (texto: lunes..domingo), es_festivo (0/1), temperatura_c, humedad_pct, superficie_m2,
num_ocupantes, consumo_kwh`.
Simulación realista y DOCUMENTADA: clima de Veracruz (estacionalidad ene→jun, ciclo diario,
ciudad: Xalapa/Orizaba más frescas), perfil horario por tipo (residencial picos mañana/noche;
comercial 9–21 h; industrial turnos y caída domingos; educativa L–V 7–21 h y vacaciones de
Semana Santa 2026: 30-mar a 10-abr), efecto de temperatura (aire acondicionado > 26 °C),
festivos mexicanos 2026, ruido gaussiano, tendencia leve.
Problemas inyectados a propósito (y documentados con conteos): ~2 % nulos en temperatura,
humedad y consumo; ~1 % filas duplicadas; ~0.5 % atípicos (consumo negativo, picos ×8,
temperatura 99.9); inconsistencias de texto en `tipo_instalacion` ("Residencial", " comercial ")
y en `dia_semana` (mayúsculas).
**Procesado** `config.CSV_CONSUMO_PROC`: limpio, ordenado por `id_instalacion, fecha_hora`, con
columnas raw limpias + `hora, dia_semana_num (0=lunes), mes, dia_anio, es_fin_semana,
hora_sin, hora_cos, mes_sin, mes_cos, temp_x_hora_pico, consumo_lag_1h, consumo_lag_24h,
consumo_lag_168h, consumo_media_movil_24h, conjunto ("entrenamiento"/"prueba")`.
La división es TEMPORAL: último `config.FRACCION_TEST` del periodo = prueba.
`preparar_datos.py` documenta en evidencia: nulos antes/después por columna y estrategia
(interpolación temporal por instalación), duplicados eliminados, atípicos detectados
(IQR por instalación/hora + reglas físicas) y tratamiento, normalización (StandardScaler dentro
del Pipeline del modelo), codificación (OneHot de tipo_instalacion/ciudad), selección de
características (correlación + importancia), variables creadas. Debe exponer
`preparar_consumo() -> pd.DataFrame` y escribir `outputs/evidencias/preparacion_datos.txt`
y `outputs/evidencias/resumen_preparacion.json`.

### B. Machine Learning (`src/ml/`)
- `entrenar_modelo.py`: regresión supervisada, objetivo `consumo_kwh`. Comparar al menos
  Regresión Lineal, Árbol de Decisión, KNN y Random Forest (y opcionalmente SVR sobre muestra)
  con `Pipeline(ColumnTransformer(StandardScaler + OneHotEncoder), modelo)`. Validación cruzada
  temporal (`TimeSeriesSplit`) o búsqueda ligera de hiperparámetros para RF. Métricas MAE, RMSE,
  R², MAPE (sobre consumo > 0.1). Guardar el mejor en `config.MODELO_CONSUMO` (joblib, el
  Pipeline completo) y `config.METRICAS_ML` con:
  `{"seleccionado": str, "tipo_aprendizaje": "supervisado - regresión", "objetivo": "consumo_kwh",
    "caracteristicas": [..], "hiperparametros": {..}, "modelos": {nombre: {"mae","rmse","r2","mape","tiempo_s"}},
    "metricas_por_tipo": {tipo: {"mae","r2"}}, "importancias": {feature: valor}, "n_entrenamiento", "n_prueba",
    "fecha_entrenamiento"}`.
  Guardar `outputs/predicciones_test.csv` (`id_instalacion, fecha_hora, tipo_instalacion, real, predicho, error`).
  Figuras: `comparacion_modelos.png, real_vs_predicho.png, importancia_variables.png,
  residuos.png, serie_prueba_<id>.png` (2–3 instalaciones), `eda_consumo_hora_tipo.png,
  eda_temp_vs_consumo.png, eda_correlacion.png`.
- `clustering.py`: no supervisado, K-Means sobre perfiles diarios (vector de 24 h normalizado
  por instalación-día). Elegir k con codo + silhouette (k=2..8). Guardar `config.MODELO_KMEANS`
  y `models/clusters_resumen.json` (`k, silhouette, inercias, centroides, distribucion por tipo,
  dias_atipicos (top 20 más lejanos al centroide)`), figuras `kmeans_codo_silhouette.png`,
  `kmeans_centroides.png`.
- `predecir.py`: API para el dashboard:
  `predecir_consumo(id_instalacion: str, fecha_hora: str, temperatura_c: float, humedad_pct: float, es_festivo: int = 0) -> dict`
  → `{"id_instalacion","fecha_hora","prediccion_kwh","intervalo": [lo, hi], "modelo", "entradas": {...}}`.
  Los lags se obtienen del histórico procesado (mismo id, misma hora del último día/semana
  disponible) con fallback a promedios por instalación y hora. Intervalo con percentiles de
  los árboles del RF (o ± RMSE si no es RF). Cachear modelo y datos.
  También `pronostico_24h(id_instalacion, fecha: str, temperaturas: list|None) -> list[dict]`.

### C. PNL (`src/data/generar_reportes.py`, `src/nlp/`)
- `generar_reportes.py`: `config.CSV_REPORTES_RAW` con ~4,000 reportes en español simulados
  por plantillas combinatorias ricas (muchas variantes, sinónimos, errores ortográficos leves,
  mayúsculas, emojis ocasionales, signos), columnas
  `id_reporte, id_instalacion, fecha_hora, canal (app/correo/llamada/portal), texto, sentimiento_etiqueta (positivo/negativo/neutral)`.
  Los textos mencionan entidades: dispositivos (`config.DISPOSITIVOS`), ciudades (`config.CIUDADES`),
  organizaciones (`config.ORGANIZACIONES`), fechas ("12 de marzo", "15/03/2026", "ayer"),
  horas ("a las 18:30"), códigos de instalación (`RES-001`), folios (`FOL-123456`), montos ($1,250) y kWh ("350 kWh").
  Incluir ~10 % de casos ambiguos/mezclados y algo de ruido de etiqueta (~3 %) para que el
  clasificador no sea trivial; documentar el procedimiento. Algunos reportes deben correlacionar
  con eventos del dataset de consumo (p. ej. picos) — opcional.
- `procesamiento_texto.py`: normalización (minúsculas, acentos opcional, URLs/emojis),
  tokenización con regex, lista de stopwords en español propia (≥ 200 palabras, sin quitar
  negaciones "no", "nunca", "sin"), stemmer ligero en español implementado a mano (sufijos
  tipo Snowball simplificado) y lematizador por diccionario pequeño de verbos/plurales frecuentes.
  Función `preprocesar(texto) -> dict{"normalizado","tokens","sin_stopwords","stems","lemas"}`.
- `sentimiento.py`: representación TF-IDF (unigramas+bigramas) con el preprocesamiento anterior;
  comparar Naive Bayes multinomial, Regresión Logística y SVM lineal; elegir el mejor por F1-macro;
  matriz de confusión (`pnl_matriz_confusion.png`), reporte de clasificación; guardar
  `config.MODELO_SENTIMIENTO` y `config.METRICAS_PNL`
  (`{"seleccionado","modelos":{nombre:{"accuracy","f1_macro","precision_macro","recall_macro"}},"reporte_clasificacion","n_entrenamiento","n_prueba","terminos_top_por_clase"}`).
  Función `predecir_sentimiento(texto) -> {"sentimiento","probabilidades":{...}}`.
- `entidades.py`: NER híbrido (diccionarios/gazetteers + expresiones regulares), tipos:
  `DISPOSITIVO, LUGAR, ORGANIZACION, FECHA, HORA, CODIGO_INSTALACION, FOLIO, MONTO, ENERGIA, PERSONA`
  (personas por patrón "Sr./Sra./Ing./Lic. + Nombre"). `extraer_entidades(texto) -> list[{"texto","tipo","inicio","fin"}]`.
  Evaluar precisión/recall contra las entidades que el generador insertó (el generador debe
  guardar `entidades_esperadas` en JSON en el CSV raw para esto).
- `generacion.py`: NLG basada en plantillas con selección de frases según los datos
  (no LLM): `generar_resumen(datos: dict) -> str` que redacta un párrafo/reporte en español
  a partir de métricas del modelo, predicciones, clusters y resultados de PNL (ver D.reporte).
  Incluir también `resumir_reportes(df) -> str`.
- `analizador.py`: `analizar_texto(texto) -> {"preprocesado": {...}, "sentimiento": {...}, "entidades": [...]}`
  y `procesar_reportes() -> pd.DataFrame` que escribe `config.CSV_REPORTES_PROC`
  (`id_reporte, id_instalacion, fecha_hora, canal, texto, texto_limpio, sentimiento_etiqueta,
  sentimiento_predicho, confianza, entidades (JSON), n_entidades`), figuras
  `pnl_distribucion_sentimiento.png, pnl_entidades_frecuentes.png, pnl_terminos_tfidf.png`
  y evidencia `outputs/evidencias/pnl_ejemplos.txt` con ≥ 10 ejemplos entrada→salida.

### D. Blockchain (`src/blockchain/`)
- `blockchain.py`: blockchain privada/permisionada simulada, persistida en `config.CADENA_JSON`.
  `class Bloque`: `indice, marca_tiempo (ISO), datos (dict), hash_anterior, nonce, hash`;
  hash = SHA-256 del JSON canónico (`sort_keys=True, ensure_ascii=False`) de todos los campos
  excepto `hash`. Prueba de trabajo con `config.DIFICULTAD_POW` ceros. Bloque génesis.
  `class Blockchain(ruta=config.CADENA_JSON, dificultad=config.DIFICULTAD_POW)`:
  `agregar_registro(tipo: str, nombre: str, hash_documento: str, metadatos: dict|None=None) -> Bloque`,
  `registrar_archivo(ruta, tipo, metadatos=None) -> Bloque`,
  `registrar_contenido(contenido: str|bytes|dict, tipo, nombre, metadatos=None) -> Bloque`,
  `validar_cadena() -> {"valida": bool, "errores": [..], "n_bloques": int}` (recalcula hashes,
  enlaces y PoW), `buscar_por_hash(h) -> Bloque|None`, `buscar_por_nombre(nombre) -> list`,
  `verificar_archivo(ruta) -> dict`, `verificar_contenido(contenido, nombre=None) -> dict`
  → `{"integro": bool, "hash_calculado", "hash_registrado"|None, "bloque": indice|None, "mensaje"}`,
  `a_lista() -> list[dict]`, `guardar()`, `cargar()`.
  Funciones sueltas: `sha256_bytes(b)`, `sha256_archivo(ruta)` (por bloques), `sha256_json(obj)`.
  `datos` del bloque: `{"tipo": "reporte"|"prediccion"|"modelo"|"dataset"|"metricas", "nombre", "hash_documento", "metadatos": {...}}`.
- `demo_alteracion.py`: demostración reproducible (para el entregable 5): registra un
  documento de ejemplo, lo verifica (íntegro), lo modifica un carácter, lo verifica (modificado),
  luego altera un bloque dentro de una COPIA de la cadena y muestra que `validar_cadena` lo
  detecta. Escribe `outputs/evidencias/blockchain_prueba_alteracion.txt`. No debe corromper la
  cadena real (usa una ruta temporal).

### E. Integración (oleada 2): `src/reporte/generar_reporte.py`, `pipeline.py`, `app/`
- `generar_reporte.py`: `generar_reporte() -> dict` construye `outputs/reportes/reporte_<AAAAMMDD_HHMMSS>.json`
  y `.txt` (texto NLG) con: registros analizados, métricas ML, resumen de predicciones,
  clusters, resumen PNL; calcula SHA-256 y lo registra en la blockchain (tipo "reporte").
- `pipeline.py`: CLI `python pipeline.py [--todo|--datos|--ml|--pnl|--reporte|--blockchain-demo]`
  ejecuta en orden: generar datos → preparar → entrenar → clustering → generar reportes texto →
  PNL → reporte+hash+blockchain (también registra dataset procesado y modelo) → demo alteración.
- `app/app.py` (Flask, `python app/app.py` → http://127.0.0.1:5000): páginas Resumen (KPIs:
  registros, instalaciones, consumo total, R², sentimiento), Datos/EDA, Modelo ML, Predicción
  (formulario → predicción → botón "Registrar en Blockchain"), PNL (analizar texto libre +
  entidades resaltadas), Blockchain (explorador de bloques, validar cadena, verificar archivo
  subido o texto pegado), Reportes (generar/descargar). Chart.js por CDN con fallback a PNG.
  Comprensible para un usuario no técnico.

### F. Documentación (oleada 3): `README.md`, `docs/*.md`, documento técnico `.docx`.
