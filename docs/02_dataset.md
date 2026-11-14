# Entregable 2. Dataset

El proyecto utiliza **dos datasets simulados**:

1. **Consumo energético horario** de 10 instalaciones (datos estructurados, serie de tiempo).
2. **Reportes de usuarios** en español (datos no estructurados, texto).

Fuentes de las cifras: `outputs/evidencias/generacion_datos.txt`, `generacion_reportes.txt`, `preparacion_datos.txt`, `resumen_preparacion.json`, y los CSV de `data/`.

## 1. Origen de los datos

### 1.1 Por qué simulación

No se dispone de mediciones reales de instalaciones eléctricas ni de reportes reales de usuarios, y los datos reales de consumo suelen ser confidenciales o tener datos personales. La simulación permite:

- Conocer el mecanismo generador de los datos y, con ello, interpretar los resultados.
- Inyectar a propósito problemas de calidad (nulos, duplicados, atípicos, inconsistencias de texto) con conteos exactos para comprobar que la preparación los detecta.
- Reproducibilidad total: todo el azar usa `numpy.random.default_rng(config.SEMILLA)` (consumo) y `random` con la misma semilla 42 (reportes).
- Cumplir el requisito de más de 5,000 registros.

Por ser sintéticos, **no representan consumo real** y las métricas deben leerse con esa reserva.

### 1.2 Procedimiento de generación del consumo (`src/data/generar_datos.py`)

Periodo: `config.DIAS_SIMULADOS = 182` días desde `config.FECHA_INICIO = 2026-01-01 00:00:00`, frecuencia horaria: 4,368 horas por instalación x 10 instalaciones = **43,680 registros verdaderos**. (El último registro es el 1 de julio de 2026, 23:00; el docstring del script menciona "30-jun" pero el rango real de los datos llega al 1-jul.)

1. **Clima por ciudad** (`simular_clima`). Temperatura = tendencia estacional (crece linealmente de la media de enero a la de junio de cada ciudad; Xalapa, Orizaba y Córdoba son más frescas que la costa) + ciclo diario coseno (máximo a las 15 h) + anomalía diaria AR(1) (rho 0.7, sigma 1.8 °C) + ruido horario + frentes fríos ("nortes") de enero a marzo (6 % de probabilidad diaria, 2 días, -4 °C). La humedad disminuye con la temperatura, se recorta a [25, 100] % y sube 5 puntos desde mayo. Cada instalación suma ruido de sensor (0.3 °C y 1.5 %).
2. **Perfil horario por tipo** (`PERFILES`). Vectores de 24 factores para días de trabajo y de descanso: residencial con picos de 6 a 8 h y de 19 a 22 h; comercial abierto de 9 a 21 h; industrial con tres turnos (sábado 80 %, domingo 45 %); educativa de lunes a viernes de 7 a 21 h, con consumo de ~20 % en fines de semana, festivos y Semana Santa 2026 (30 de marzo a 10 de abril).
3. **Consumo base.** `consumo = base_kwh x perfil_hora x factor_día x tendencia x eficiencia_instalación + climatización`. Tendencia de +0.03 % por día; eficiencia por instalación ~ N(1, 0.03); climatización (aire acondicionado) = `base_kwh x coef_tipo x max(0, T-26) x perfil_hora`, con coeficientes `COEF_AIRE` (residencial 0.07, comercial 0.05, educativa 0.04, industrial 0.015 por °C).
4. **Eventos operativos.** Cada instalación tiene 3 días aleatorios de paro/ausencia (consumo x 0.35). Son anomalías reales, no errores (30 días en total).
5. **Ruido.** Factor diario multiplicativo N(1, 0.04) y ruido horario AR(1) (rho 0.6, sigma 0.05); consumo mínimo de 2 % del base.
6. **Inyección de problemas de calidad** (`inyectar_problemas`), con conteos exactos en `generacion_datos.txt`:

| Problema inyectado | Conteo |
|---|---|
| Nulos en `temperatura_c`, `humedad_pct`, `consumo_kwh` (2 % cada una) | 874 por columna |
| Atípicos: consumo negativo | 72 |
| Atípicos: pico de consumo x 8 | 72 |
| Atípicos: temperatura 99.9 °C (falla de sensor) | 74 |
| Atípicos totales | 218 |
| Inconsistencias de texto en `tipo_instalacion` (4 %) | 1,747 |
| Inconsistencias de texto en `dia_semana` (6 %) | 2,621 |
| Filas duplicadas (1 %, mismo `id_registro`) | 437 |
| Filas finales (43,680 + 437) | **44,117** |

### 1.3 Procedimiento de generación de reportes (`src/data/generar_reportes.py`)

Simulación por **plantillas combinatorias**:

1. Para cada reporte se elige una instalación de `config.INSTALACIONES`, un canal (app 35 %, correo 20 %, llamada 20 %, portal 25 %) y una fecha-hora dentro del periodo.
2. Se decide el sentimiento verdadero (positivo ~32 %, negativo ~38 %, neutral ~30 %).
3. El texto se arma con cláusulas de bancos por sentimiento; cada cláusula tiene alternativas `[a|b|c]` (sinónimos) y ranuras (`{disp} {lugar} {org} {fecha} {hora} {cod} {folio} {monto} {kwh} {persona}`) coherentes con el tipo de instalación (una planta industrial habla de transformadores y compresores; montos y kWh con magnitudes acordes).
4. Se añaden 0 a 3 oraciones de detalle, apertura según el canal y a veces un cierre.
5. Cada entidad insertada se guarda en `entidades_esperadas` (JSON) para evaluar el NER.
6. **Ruido y casos difíciles:** ~10 % de reportes mixtos o ambiguos (405 = 10.1 %), ~3 % de etiquetas cambiadas a propósito (116 = 2.9 %), negaciones, emojis que no siempre coinciden con el sentimiento (~15 %), errores ortográficos leves, mayúsculas y signos repetidos; ~3 % de dispositivos con alias coloquial ("refri", "clima") y ~1.5 % de ciudades con error de dedo.

Resultado (`generacion_reportes.txt`): 4,000 reportes, 100 % de textos únicos, periodo 2026-01-01 06:27 a 2026-07-01 21:00. Etiquetas: negativo 1,531, positivo 1,343, neutral 1,126. Canales: app 1,357, portal 1,030, correo 830, llamada 783.

## 2. Condiciones de uso

- Datos 100 % sintéticos generados por el equipo con código propio (licencia del proyecto: uso académico).
- No contienen personas reales, direcciones, medidores reales ni información confidencial. Los nombres que aparecen en los reportes ("Sr. Héctor Ramírez", etc.) son ficticios.
- Las organizaciones mencionadas (CFE, PROFECO, CENACE, TecNM, ITV) son nombres públicos usados solo como entidades de ejemplo; los reportes no representan comunicaciones reales con ellas.
- Pueden regenerarse idénticos ejecutando `python -m src.data.generar_datos` y `python -m src.data.generar_reportes` (semilla 42).

## 3. Archivos del dataset

| Archivo | Descripción | Filas x columnas |
|---|---|---|
| `data/raw/consumo_energetico_raw.csv` | Consumo crudo con problemas de calidad | 44,117 x 12 |
| `data/processed/consumo_energetico_procesado.csv` | Consumo limpio con variables creadas y división | 42,000 x 27 |
| `data/raw/reportes_usuarios_raw.csv` | Reportes crudos con entidades esperadas | 4,000 x 7 |
| `data/processed/reportes_procesados.csv` | Reportes con texto limpio, sentimiento y entidades | 4,000 x 11 |
| `outputs/predicciones_test.csv` | Real vs predicho en prueba (8,740 filas) | 8,740 x 6 |

El procesado tiene 42,000 filas = 43,680 - 1,680 (se descarta la primera semana de cada instalación, que no tiene 168 h de historia para el lag semanal). Rango: 2026-01-08 00:00 a 2026-07-01 23:00. Por tipo: residencial 12,600; comercial 12,600; industrial 8,400; educativa 8,400.

## 4. Diccionario de variables

### 4.1 Consumo, dataset crudo (`consumo_energetico_raw.csv`)

| Nombre | Tipo | Descripción | Unidad |
|---|---|---|---|
| `id_registro` | texto | Identificador del registro (REG-000001...) | - |
| `id_instalacion` | texto | Identificador de la instalación (RES-001, COM-002, IND-001, EDU-001...) | - |
| `tipo_instalacion` | categórica (texto) | residencial, comercial, industrial, educativa (con inconsistencias de mayúsculas y espacios en crudo) | - |
| `ciudad` | categórica (texto) | Ciudad de la instalación (Veracruz, Boca del Río, Xalapa, Córdoba, Coatzacoalcos, Orizaba) | - |
| `fecha_hora` | fecha-hora (texto `YYYY-MM-DD HH:MM:SS`) | Marca de tiempo horaria | - |
| `dia_semana` | categórica (texto) | Día de la semana (con inconsistencias de mayúsculas en crudo) | - |
| `es_festivo` | entero binario | 1 si es festivo mexicano 2026 | 0/1 |
| `temperatura_c` | decimal | Temperatura ambiente medida | °C |
| `humedad_pct` | decimal | Humedad relativa | % |
| `superficie_m2` | entero | Superficie de la instalación | m² |
| `num_ocupantes` | entero | Número de ocupantes habituales | personas |
| `consumo_kwh` | decimal | Consumo eléctrico en la hora (**variable objetivo**) | kWh |

### 4.2 Consumo, dataset procesado (`consumo_energetico_procesado.csv`)

Contiene las 12 columnas anteriores (ya limpias, `fecha_hora` como fecha-hora y `dia_semana` verificado contra la fecha) y 15 más:

| Nombre | Tipo | Descripción | Unidad |
|---|---|---|---|
| `hora` | entero (0-23) | Hora del día | h |
| `dia_semana_num` | entero (0-6) | Día de la semana, 0 = lunes | - |
| `mes` | entero (1-7) | Mes | - |
| `dia_anio` | entero | Día del año | - |
| `es_fin_semana` | entero binario | 1 si sábado o domingo | 0/1 |
| `hora_sin`, `hora_cos` | decimal | Codificación cíclica de la hora (periodo 24) | adimensional |
| `mes_sin`, `mes_cos` | decimal | Codificación cíclica del mes (periodo 12) | adimensional |
| `temp_x_hora_pico` | decimal | `temperatura_c` x 1 si la hora está entre 18 y 21, si no 0 (interacción calor-demanda alta) | °C |
| `consumo_lag_1h` | decimal | Consumo de la misma instalación 1 hora antes | kWh |
| `consumo_lag_24h` | decimal | Consumo 24 horas antes | kWh |
| `consumo_lag_168h` | decimal | Consumo 168 horas (una semana) antes | kWh |
| `consumo_media_movil_24h` | decimal | Media de las 24 horas previas (sin incluir la actual) | kWh |
| `conjunto` | categórica | `entrenamiento` o `prueba` | - |

### 4.3 Reportes, dataset crudo (`reportes_usuarios_raw.csv`)

| Nombre | Tipo | Descripción |
|---|---|---|
| `id_reporte` | texto | REP-00001... |
| `id_instalacion` | texto | Instalación que origina el reporte |
| `fecha_hora` | fecha-hora (texto) | Momento del reporte |
| `canal` | categórica | app, correo, llamada, portal |
| `texto` | texto | Reporte en español |
| `sentimiento_etiqueta` | categórica | positivo, negativo, neutral (etiqueta del generador, con ~3 % de ruido) |
| `entidades_esperadas` | texto (JSON) | Lista de `{"texto","tipo"}` insertadas por el generador (referencia para evaluar el NER) |

### 4.4 Reportes, dataset procesado (`reportes_procesados.csv`)

Conserva `id_reporte, id_instalacion, fecha_hora, canal, texto, sentimiento_etiqueta` y agrega:

| Nombre | Tipo | Descripción |
|---|---|---|
| `texto_limpio` | texto | Texto normalizado (minúsculas, entidades numéricas enmascaradas, emojis a marcadores) |
| `sentimiento_predicho` | categórica | Predicción del clasificador |
| `confianza` | decimal [0,1] | Probabilidad de la clase predicha |
| `entidades` | texto (JSON) | Entidades extraídas con `texto, tipo, inicio, fin` |
| `n_entidades` | entero | Número de entidades extraídas (11,322 en total) |

## 5. Tipos de datos

- **Numéricas continuas:** `temperatura_c`, `humedad_pct`, `consumo_kwh`, lags, media móvil, variables cíclicas.
- **Numéricas discretas:** `hora`, `mes`, `dia_anio`, `dia_semana_num`, `superficie_m2`, `num_ocupantes`.
- **Binarias:** `es_festivo`, `es_fin_semana`.
- **Categóricas nominales:** `tipo_instalacion`, `ciudad`, `id_instalacion`, `canal`, `conjunto`.
- **Fecha-hora:** `fecha_hora`.
- **Texto:** `texto`, `texto_limpio`; **JSON en texto:** `entidades_esperadas`, `entidades`.

Estadísticos del consumo procesado por tipo (kWh; calculados de `consumo_energetico_procesado.csv`):

| Tipo | Media | Desv. est. | Mín | Máx |
|---|---|---|---|---|
| residencial | 1.08 | 0.63 | 0.11 | 4.40 |
| comercial | 10.05 | 8.06 | 0.56 | 40.20 |
| industrial | 76.38 | 26.62 | 18.32 | 142.33 |
| educativa | 24.88 | 22.72 | 1.91 | 101.02 |

En el crudo (sin corregir) había mínimos negativos (p. ej. -114.20 kWh industrial) y máximos de hasta 922.73 kWh, producto de los atípicos inyectados.

## 6. Preparación de datos (`src/data/preparar_datos.py`)

Evidencia: `outputs/evidencias/preparacion_datos.txt` y `resumen_preparacion.json`. El orden de los pasos evita fuga de información.

### 6.1 Valores faltantes
- Antes: `temperatura_c` 885 (2.01 %), `humedad_pct` 881 (2.00 %), `consumo_kwh` 881 (2.00 %).
- Estrategia: **interpolación lineal en el tiempo por instalación** (la serie es horaria y suave: se interpola entre la hora válida anterior y la siguiente de la misma instalación). Se imputaron, incluyendo los atípicos convertidos en nulos: 948 temperaturas, 874 humedades y 1,057 consumos.
- Después: 0 nulos en todas las columnas.

### 6.2 Registros duplicados
- 437 filas idénticas eliminadas (44,117 a 43,680). Llave (instalación, hora) repetida: 0. Horas faltantes respecto al calendario esperado: 0.

### 6.3 Valores atípicos
- Reglas físicas: temperatura fuera de [-5, 50] °C: 74; humedad fuera de [0, 100] %: 0; consumo negativo: 72.
- Regla estadística IQR: consumo > Q3 + 3 x IQR por instalación, hora y tipo de día: 111 picos extremos.
- Total: 257 valores. **Tratamiento:** se marcan como nulos y se imputan (6.1). No se eliminan filas, para conservar la continuidad horaria necesaria para los lags.
- Nota: los 111 picos IQR incluyen los 72 picos x 8 inyectados y otros valores extremos que la regla marca; las cifras de inyección (218 atípicos) y de detección (257) no coinciden exactamente porque la regla IQR es estadística y las reglas físicas se aplican por separado.

### 6.4 Normalización de texto
- `tipo_instalacion`: 24 variantes distintas a 4 (strip + minúsculas), y se verifica contra el catálogo de `config.INSTALACIONES` (0 inconsistencias tras limpiar).
- `dia_semana`: 21 variantes a 7, recalculado desde la fecha (fuente de verdad).

### 6.5 Transformaciones y creación de variables
- Calendario: `hora, dia_semana_num, mes, dia_anio, es_fin_semana`.
- Cíclicas: `hora_sin/cos` (periodo 24) y `mes_sin/cos` (periodo 12), para que la hora 23 quede cerca de la 0.
- Interacción: `temp_x_hora_pico`.
- Históricas (solo valores pasados, `shift` antes de `rolling`): `consumo_lag_1h`, `consumo_lag_24h`, `consumo_lag_168h`, `consumo_media_movil_24h`.
- `agregar_variables_calendario()` se comparte con `src/ml/predecir.py` para que entrenamiento e inferencia usen las mismas definiciones.

### 6.6 Codificación de variables categóricas
- `tipo_instalacion` y `ciudad` con `OneHotEncoder(handle_unknown="ignore")`, dentro del `Pipeline` del modelo.

### 6.7 Normalización / estandarización
- `StandardScaler` sobre las 19 variables numéricas, **dentro del `Pipeline`** y ajustado solo con entrenamiento (por eso el CSV procesado conserva las unidades originales). Los modelos de árboles no lo requieren, pero sí Regresión Lineal, KNN y SVR.

### 6.8 Selección de características
- Se analizó la correlación de Pearson de cada variable con el consumo relativo (en entrenamiento): mayores magnitudes en `hora_cos` (-0.463), `hora` (0.401), `temperatura_c` (0.385), `humedad_pct` (-0.376), `hora_sin` (-0.375), `consumo_lag_1h` (0.325), `consumo_lag_168h` (0.279).
- Pares muy correlacionados (|r| > 0.95): `mes~dia_anio` (0.98), `mes~mes_cos` (1.00), `dia_anio~mes_cos` (0.98).
- **Decisión:** se conservan las 21 características (19 numéricas + 2 categóricas), porque los modelos de árboles toleran redundancia; la importancia final se reporta en `03_modelo_ml.md`.

### 6.9 División de los datos
- **División temporal** (no aleatoria), para simular un uso real en que se predice el futuro: corte en 2026-05-26 14:00. Entrenamiento: 33,260 filas (2026-01-08 a 2026-05-26 13:00). Prueba: 8,740 filas (2026-05-26 14:00 a 2026-07-01 23:00), es decir 20 % del periodo (`config.FRACCION_TEST`).
- Para la selección de hiperparámetros, validación cruzada `TimeSeriesSplit` de 3 cortes sobre el entrenamiento.
- **Anti-fuga:** lags solo con horas anteriores; escalado y codificación ajustados solo con entrenamiento.

### 6.10 Preparación de los reportes
El texto se normaliza, tokeniza, filtra y se lleva a raíces en `src/nlp/procesamiento_texto.py` (ver `04_pnl.md`). La división de los reportes es 80 % / 20 % estratificada por sentimiento: 3,200 de entrenamiento y 800 de prueba, semilla 42.
