"""Generación de lenguaje natural (NLG) basada en plantillas con selección condicionada.

No usa ningún LLM: cada frase se elige con reglas ``si-entonces`` según los valores de los
datos (calificar el R², alertar si hay muchos reportes negativos, recomendar acciones según
los dispositivos más mencionados, etc.).  Es determinista: los mismos datos producen el mismo
texto, lo que permite calcular un hash estable del reporte (blockchain).

Funciones públicas:
  * ``generar_resumen(datos: dict) -> str``
  * ``resumir_reportes(df) -> str``
"""
import json
from collections import Counter

# Recomendaciones de optimización por dispositivo (clave = nombre en config.DISPOSITIVOS)
RECOMENDACIONES = {
    "aire acondicionado": "ajustar los termostatos a 24-25 °C, dar mantenimiento a filtros y programar apagado automático fuera de horario",
    "minisplit": "limpiar filtros cada mes, revisar fugas de refrigerante y evaluar equipos con tecnología inverter",
    "refrigerador": "revisar sellos de puertas, evitar sobrecarga y verificar la temperatura de operación del compresor",
    "transformador": "medir el factor de potencia, revisar cargas desbalanceadas y programar mantenimiento termográfico",
    "medidor": "calibrar o sustituir medidores con lecturas dudosas y considerar medición inteligente para detectar errores de facturación",
    "compresor": "detectar fugas de aire comprimido, instalar variadores de frecuencia y mantener la presión mínima necesaria",
    "bomba de agua": "operar la bomba en horarios de menor tarifa y revisar el dimensionamiento y los sellos del equipo",
    "luminarias": "sustituir luminarias por tecnología LED y automatizar el encendido con sensores de presencia",
    "calentador": "aislar tuberías, ajustar la temperatura de consigna y valorar calentadores solares",
    "motor": "revisar alineación y lubricación, y considerar motores de alta eficiencia con variadores",
    "inversor": "verificar la configuración del inversor y su ventilación para evitar pérdidas por sobrecalentamiento",
    "servidor": "consolidar cargas, mejorar el enfriamiento del cuarto de equipos y apagar servidores sin uso",
    "elevador": "revisar el sistema de frenado regenerativo y programar mantenimiento preventivo",
    "ventilador": "limpiar aspas, revisar rodamientos y regular la velocidad según la ocupación",
    "lavadora": "usar ciclos en frío y cargas completas, y evitar el uso en horas pico",
    "horno": "mejorar el aislamiento, evitar aperturas innecesarias y concentrar los ciclos en horarios de menor demanda",
}


# ---------------------------------------------------------------------------
# Utilidades de formato y de lectura tolerante de los datos
# ---------------------------------------------------------------------------
def _num(x, decimales=2):
    """Formatea un número al estilo mexicano (1,234.56); devuelve 'N/D' si no es numérico."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "N/D"
    if v != v:  # NaN
        return "N/D"
    return f"{v:,.{decimales}f}"


def _es_num(x):
    try:
        return float(x) == float(x)
    except (TypeError, ValueError):
        return False


def _pares(obj, n=None):
    """Convierte dict / lista de pares / lista de dicts en [(clave, valor)] ordenada por valor."""
    if not obj:
        return []
    if isinstance(obj, dict):
        pares = list(obj.items())
    else:
        pares = []
        for e in obj:
            if isinstance(e, dict):
                clave = e.get("texto") or e.get("nombre") or e.get("termino") or e.get("tipo")
                val = e.get("n") or e.get("conteo") or e.get("frecuencia") or e.get("cantidad") or 1
                pares.append((clave, val))
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                pares.append((e[0], e[1]))
            else:
                pares.append((e, 1))
    pares = [(str(k), v) for k, v in pares if k is not None and _es_num(v)]
    pares.sort(key=lambda p: -float(p[1]))
    return pares[:n] if n else pares


def _lista_es(items):
    """Une elementos con comas y 'y': ['a','b','c'] -> 'a, b y c'."""
    items = [str(i) for i in items]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " y " + items[-1]


def _periodo_texto(p):
    if isinstance(p, str):
        return p
    if isinstance(p, dict):
        ini, fin = p.get("inicio") or p.get("desde"), p.get("fin") or p.get("hasta")
        return f"{ini} a {fin}" if ini and fin else ""
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        return f"{p[0]} a {p[1]}"
    return ""


def _energia(kwh):
    """Expresa kWh con una unidad cómoda (kWh, MWh, GWh)."""
    v = float(kwh)
    if abs(v) >= 1e6:
        return f"{_num(v / 1e6)} GWh ({_num(v, 0)} kWh)"
    if abs(v) >= 1e4:
        return f"{_num(v / 1e3, 1)} MWh ({_num(v, 0)} kWh)"
    return f"{_num(v)} kWh"


def _fracciones(dist):
    """Normaliza una distribución de conteos o porcentajes a fracciones que suman 1."""
    pares = {k.lower(): float(v) for k, v in _pares(dist)}
    total = sum(pares.values())
    return {k: v / total for k, v in pares.items()} if total > 0 else {}


def _calificar_r2(r2):
    if r2 >= 0.95:
        return "excelente", "explica casi toda la variabilidad del consumo"
    if r2 >= 0.85:
        return "muy bueno", "explica la gran mayoría de la variabilidad del consumo"
    if r2 >= 0.70:
        return "aceptable", "captura la tendencia general, aunque deja variabilidad sin explicar"
    if r2 >= 0.50:
        return "moderado", "solo explica parte de la variabilidad y debe usarse con cautela"
    return "limitado", "explica poca variabilidad, por lo que sus predicciones son de referencia"


def _calificar_silhouette(s):
    if s >= 0.50:
        return "una separación clara entre los grupos"
    if s >= 0.25:
        return "una separación razonable, con cierto traslape entre perfiles"
    return "grupos con bastante traslape, por lo que conviene interpretarlos con cautela"


# ---------------------------------------------------------------------------
# Párrafos
# ---------------------------------------------------------------------------
def _p_alcance(d):
    n, ni = d.get("n_registros"), d.get("n_instalaciones")
    periodo, total = _periodo_texto(d.get("periodo")), d.get("consumo_total_kwh")
    frases = []
    if _es_num(n) and _es_num(ni):
        frases.append(f"Este reporte automático resume el análisis de {_num(n, 0)} registros horarios de consumo eléctrico "
                      f"correspondientes a {_num(ni, 0)} instalaciones")
    elif _es_num(n):
        frases.append(f"Este reporte automático resume el análisis de {_num(n, 0)} registros horarios de consumo eléctrico")
    else:
        frases.append("Este reporte automático resume el análisis del consumo eléctrico de las instalaciones monitoreadas")
    frases[0] += f", durante el periodo {periodo}." if periodo else "."
    if _es_num(total):
        t = f"En conjunto, el consumo acumulado fue de {_energia(total)}"
        if _es_num(n) and float(n) > 0:
            t += f", lo que equivale a un promedio de {_num(float(total) / float(n))} kWh por registro horario"
        frases.append(t + ".")
    por_tipo = _pares(d.get("consumo_por_tipo"))
    if por_tipo:
        tot = sum(float(v) for _, v in por_tipo)
        k, v = por_tipo[0]
        frases.append(f"Por tipo de instalación, el mayor consumo corresponde al sector {k} ({_num(100 * float(v) / tot, 1)} % del total)"
                      + (f", seguido de {_lista_es(f'{a} ({_num(100 * float(b) / tot, 1)} %)' for a, b in por_tipo[1:3])}." if len(por_tipo) > 1 else "."))
        if len(por_tipo) > 1 and float(v) / tot > 0.6:
            frases.append(f"Esta concentración sugiere que las acciones de eficiencia en el sector {k} tendrán el mayor impacto global.")
    return " ".join(frases)


def _p_modelo(d):
    m = d.get("modelo") or {}
    if not m:
        return ""
    frases = []
    nombre = m.get("nombre")
    r2 = m.get("r2")
    intro = f"Para anticipar la demanda se entrenó un modelo de regresión supervisada ({nombre})" if nombre \
        else "Para anticipar la demanda se entrenó un modelo de regresión supervisada"
    if _es_num(r2):
        cal, expl = _calificar_r2(float(r2))
        frases.append(f"{intro}, que alcanzó un coeficiente de determinación R² de {_num(r2, 3)}; se trata de un ajuste {cal}, ya que {expl}.")
    else:
        frases.append(intro + ".")
    err = []
    if _es_num(m.get("mae")):
        err.append(f"un error absoluto medio (MAE) de {_num(m['mae'])} kWh")
    if _es_num(m.get("rmse")):
        err.append(f"una raíz del error cuadrático medio (RMSE) de {_num(m['rmse'])} kWh")
    if err:
        frases.append("En el conjunto de prueba se observó " + " y ".join(err) + ".")
    if _es_num(m.get("mae")) and _es_num(m.get("rmse")) and float(m["mae"]) > 0 and float(m["rmse"]) / float(m["mae"]) > 2.5:
        frases.append("Dado que el RMSE es notablemente mayor que el MAE, existen errores puntuales grandes, probablemente asociados a picos atípicos de consumo.")
    if _es_num(r2) and float(r2) < 0.70:
        frases.append("Por lo tanto, se recomienda incorporar más variables explicativas o más historia antes de basar decisiones operativas en el modelo.")
    return " ".join(frases)


def _p_prediccion(d):
    p = d.get("prediccion") or {}
    if not p:
        return ""
    frases = []
    valor = p.get("prediccion_kwh")
    inst, fecha = p.get("id_instalacion"), p.get("fecha_hora")
    if _es_num(valor):
        donde = f" para la instalación {inst}" if inst else ""
        cuando = f" el {fecha}" if fecha else ""
        frases.append(f"Como ejemplo de uso, el modelo estima un consumo de {_num(valor)} kWh{donde}{cuando}.")
        iv = p.get("intervalo")
        if isinstance(iv, (list, tuple)) and len(iv) == 2 and _es_num(iv[0]) and _es_num(iv[1]):
            ancho = float(iv[1]) - float(iv[0])
            rel = ancho / float(valor) if float(valor) else 0
            frases.append(f"El intervalo de predicción va de {_num(iv[0])} a {_num(iv[1])} kWh; "
                          + ("al ser estrecho, indica una estimación estable." if rel < 0.3
                             else "al ser amplio, refleja una incertidumbre considerable en esa hora."))
    for clave, etiqueta in (("total_24h_kwh", "consumo proyectado para las próximas 24 horas"),
                            ("pico_kwh", "pico de demanda previsto")):
        if _es_num(p.get(clave)):
            extra = f" (hora {p['hora_pico']})" if clave == "pico_kwh" and p.get("hora_pico") is not None else ""
            frases.append(f"El {etiqueta} es de {_num(p[clave])} kWh{extra}.")
    return " ".join(frases)


def _p_clusters(d):
    c = d.get("clusters") or {}
    if not c:
        return ""
    k, s = c.get("k"), c.get("silhouette")
    if not _es_num(k):
        return ""
    t = f"Con técnicas de aprendizaje no supervisado (K-Means) se identificaron {_num(k, 0)} perfiles típicos de consumo diario"
    if _es_num(s):
        t += f", con un coeficiente de silhouette de {_num(s, 3)}, lo que indica {_calificar_silhouette(float(s))}"
    t += "."
    atip = c.get("dias_atipicos") or c.get("n_atipicos")
    if isinstance(atip, (list, tuple)) and atip:
        t += f" Además, se señalaron {len(atip)} días atípicos que conviene revisar por posibles fallas o usos anómalos."
    elif _es_num(atip) and float(atip) > 0:
        t += f" Además, se señalaron {_num(atip, 0)} días atípicos que conviene revisar por posibles fallas o usos anómalos."
    return t


def _p_pnl(d):
    p = d.get("pnl") or {}
    if not p:
        return ""
    frases = []
    n = p.get("n_reportes")
    dist = _fracciones(p.get("distribucion_sentimiento"))
    intro = f"El análisis de lenguaje natural procesó {_num(n, 0)} reportes de usuarios" if _es_num(n) \
        else "El análisis de lenguaje natural procesó los reportes de usuarios"
    if dist:
        neg, pos, neu = dist.get("negativo", 0), dist.get("positivo", 0), dist.get("neutral", 0)
        frases.append(f"{intro}: {_num(100 * neg, 1)} % son negativos, {_num(100 * pos, 1)} % positivos y {_num(100 * neu, 1)} % neutrales.")
        if neg > 0.40:
            frases.append(f"ALERTA: el porcentaje de reportes negativos ({_num(100 * neg, 1)} %) supera el umbral de 40 %, "
                          "por lo que se recomienda priorizar la atención de las quejas y revisar las instalaciones con más incidencias.")
        elif neg > 0.25:
            frases.append("El nivel de insatisfacción es moderado; aunque no rebasa el umbral de alerta (40 %), conviene dar seguimiento a las quejas recurrentes.")
        elif pos >= neg:
            frases.append("En general predomina una percepción favorable del servicio, con pocas quejas.")
    else:
        frases.append(intro + ".")
    disp = _pares(p.get("dispositivos_top"), 5)
    if disp:
        frases.append("Los equipos más mencionados fueron " + _lista_es(f"{k} ({_num(v, 0)})" for k, v in disp[:4]) + ".")
    ents = _pares(p.get("entidades_top"), 5)
    if ents:
        frases.append("Entre las entidades más frecuentes destacan " + _lista_es(f"{k} ({_num(v, 0)})" for k, v in ents[:4]) + ".")
    return " ".join(frases)


def _p_recomendaciones(d):
    p = d.get("pnl") or {}
    disp = _pares(p.get("dispositivos_top"), 3)
    dist = _fracciones(p.get("distribucion_sentimiento"))
    recs = []
    for k, _ in disp:
        r = RECOMENDACIONES.get(k.lower())
        if r:
            recs.append(f"en {k}, {r}")
    frases = []
    if recs:
        frases.append("Con base en los equipos que más aparecen en los reportes, se recomienda: " + "; ".join(recs) + ".")
    m = d.get("modelo") or {}
    por_tipo = _pares(d.get("consumo_por_tipo"))
    if por_tipo:
        frases.append(f"Asimismo, se sugiere enfocar las auditorías energéticas primero en el sector {por_tipo[0][0]}, que concentra la mayor demanda.")
    if dist.get("negativo", 0) > 0.40:
        frases.append("Dado el alto porcentaje de quejas, también se aconseja establecer un tiempo máximo de respuesta y dar seguimiento a cada folio.")
    if _es_num(m.get("r2")) and float(m["r2"]) >= 0.85:
        frases.append("Dado que el modelo predictivo es confiable, puede emplearse para programar mantenimientos y desplazar cargas a horarios de menor demanda.")
    if not frases:
        return ""
    return " ".join(frases)


def _p_cierre(d):
    return ("Los resultados y predicciones de este reporte pueden registrarse en la cadena de bloques para garantizar "
            "su integridad: cualquier alteración posterior modificaría su huella SHA-256 y sería detectada en la verificación.")


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def generar_resumen(datos):
    """Redacta un reporte en español (varios párrafos) a partir de un diccionario de datos.

    Todas las claves son OPCIONALES; si falta alguna, su párrafo o frase simplemente se omite.
    Claves utilizadas:
      * ``n_registros`` (int), ``n_instalaciones`` (int), ``periodo`` (str, o [inicio, fin],
        o {"inicio","fin"}), ``consumo_total_kwh`` (float).
      * ``consumo_por_tipo`` (dict tipo -> kWh).
      * ``modelo``: {"nombre", "r2", "mae", "rmse"}  (califica el R² y reporta errores).
      * ``prediccion``: {"id_instalacion", "fecha_hora", "prediccion_kwh", "intervalo": [lo, hi],
        y opcionalmente "total_24h_kwh", "pico_kwh", "hora_pico"}.
      * ``clusters``: {"k", "silhouette", y opcionalmente "dias_atipicos" (lista o número)}.
      * ``pnl``: {"n_reportes", "distribucion_sentimiento" (dict etiqueta -> conteo o proporción),
        "entidades_top" y "dispositivos_top" (dict texto -> conteo, o lista de pares)}.
        Si el % negativo supera 40 % se emite una ALERTA; los dispositivos más mencionados
        generan recomendaciones de optimización.
    Devuelve el texto con párrafos separados por una línea en blanco.
    """
    d = datos or {}
    funciones = (_p_alcance, _p_modelo, _p_prediccion, _p_clusters, _p_pnl, _p_recomendaciones, _p_cierre)
    parrafos = []
    for f in funciones:
        try:
            t = f(d)
        except (TypeError, ValueError, KeyError, AttributeError):   # datos con formato inesperado
            t = ""
        if t:
            parrafos.append(t)
    return "\n\n".join(parrafos)


def resumir_reportes(df):
    """Resume en un párrafo un DataFrame de reportes procesados.

    Usa (si existen) las columnas ``sentimiento_predicho`` o ``sentimiento_etiqueta``,
    ``canal``, ``id_instalacion`` y ``entidades`` (JSON con lista de {"texto","tipo"}).
    """
    if df is None or len(df) == 0:
        return "No hay reportes de usuarios para resumir."
    n = len(df)
    col = "sentimiento_predicho" if "sentimiento_predicho" in df.columns else (
        "sentimiento_etiqueta" if "sentimiento_etiqueta" in df.columns else None)
    frases = [f"Se analizaron {n:,} reportes de usuarios."]
    dist = {}
    if col:
        conteo = df[col].value_counts()
        dist = {k: v / n for k, v in conteo.items()}
        frases.append("El sentimiento se distribuye así: " + _lista_es(f"{_num(100 * v, 1)} % {k}" for k, v in dist.items()) + ".")
        if dist.get("negativo", 0) > 0.40:
            frases.append("Es una proporción de quejas elevada (más de 40 %) que requiere atención prioritaria.")
        if "id_instalacion" in df.columns:
            neg = df[df[col] == "negativo"]["id_instalacion"].value_counts()
            if len(neg):
                frases.append(f"La instalación con más reportes negativos es {neg.index[0]} ({neg.iloc[0]} casos).")
    if "canal" in df.columns:
        canal = df["canal"].value_counts()
        frases.append(f"El canal más utilizado fue '{canal.index[0]}' ({_num(100 * canal.iloc[0] / n, 1)} %).")
    if "entidades" in df.columns:
        cont = Counter()
        for js in df["entidades"].dropna():
            try:
                for e in json.loads(js):
                    if e.get("tipo") == "DISPOSITIVO":
                        cont[e["texto"].lower()] += 1
            except (ValueError, TypeError):
                continue
        if cont:
            frases.append("Los equipos más mencionados son " + _lista_es(f"{k} ({v})" for k, v in cont.most_common(3)) + ".")
    return " ".join(frases)


if __name__ == "__main__":
    ejemplo = {
        "n_registros": 43680, "n_instalaciones": 10, "periodo": "2026-01-01 a 2026-06-30",
        "consumo_total_kwh": 8_540_000.5,
        "consumo_por_tipo": {"industrial": 6_100_000, "educativa": 1_500_000, "comercial": 800_000, "residencial": 140_000},
        "modelo": {"nombre": "Random Forest", "r2": 0.93, "mae": 4.1, "rmse": 9.8},
        "prediccion": {"id_instalacion": "COM-001", "fecha_hora": "2026-06-20 15:00", "prediccion_kwh": 12.4, "intervalo": [10.1, 15.2]},
        "clusters": {"k": 3, "silhouette": 0.41},
        "pnl": {"n_reportes": 4000, "distribucion_sentimiento": {"negativo": 1521, "positivo": 1368, "neutral": 1111},
                "entidades_top": {"CFE": 800, "Veracruz": 500}, "dispositivos_top": {"aire acondicionado": 300, "medidor": 250}},
    }
    print(generar_resumen(ejemplo))
    print("\n--- Datos vacíos ---")
    print(repr(generar_resumen({})))
    print(generar_resumen({"modelo": {"r2": 0.4}, "pnl": {"distribucion_sentimiento": {"negativo": 0.5}}}))
