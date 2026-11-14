"""Reconocimiento de entidades nombradas (NER) HÍBRIDO: diccionarios + expresiones regulares.

Tipos: DISPOSITIVO, LUGAR, ORGANIZACION, FECHA, HORA, CODIGO_INSTALACION, FOLIO, MONTO,
ENERGIA, PERSONA.

Cómo funciona (explicable en 4 pasos):
  1. Se crea una copia "plegada" del texto (minúsculas y sin acentos, MISMA longitud), de
     modo que los diccionarios (gazetteers) coincidan aunque el usuario escriba
     "cordoba" o "CFE" en minúsculas, y las posiciones sirvan para el texto original.
  2. Diccionarios -> DISPOSITIVO (config.DISPOSITIVOS, con plurales), LUGAR
     (config.CIUDADES) y ORGANIZACION (config.ORGANIZACIONES).
  3. Expresiones regulares -> FECHA, HORA, CODIGO_INSTALACION, FOLIO, MONTO, ENERGIA y
     PERSONA (título "Sr./Sra./Ing./Lic." + nombre con mayúscula).
  4. Resolución de traslapes: si dos coincidencias se solapan se conserva la más larga
     (p. ej. "Comisión Federal de Electricidad" frente a otras posibles coincidencias).

Limitaciones conocidas (se ven en la evaluación): no reconoce alias coloquiales
("refri", "clima"), ni ciudades fuera de config.CIUDADES, ni fechas muy informales.

Evaluación: ``evaluar_ner()`` compara contra ``entidades_esperadas`` del CSV generado.
Ejecución: ``python -m src.nlp.entidades``.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

TIPOS = ["DISPOSITIVO", "LUGAR", "ORGANIZACION", "FECHA", "HORA", "CODIGO_INSTALACION",
         "FOLIO", "MONTO", "ENERGIA", "PERSONA"]

_MAPA_PLEGADO = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuaeiouu")


def plegar(texto):
    """Minúsculas y sin acentos conservando la longitud (para mapear posiciones)."""
    salida = []
    for c in texto:
        c2 = c.translate(_MAPA_PLEGADO).lower()
        salida.append(c2 if len(c2) == 1 else c)
    return "".join(salida)


# ---------------------------------------------------------------------------
# Diccionarios
# ---------------------------------------------------------------------------
def _patron_dispositivo(nombre):
    """Patrón para un dispositivo admitiendo plural (-s/-es) y variantes simples."""
    n = plegar(nombre)
    if n == "minisplit":
        return r"mini ?splits?"
    palabras = n.split()
    partes = []
    for p in palabras:
        if p in ("de", "del", "la", "el"):
            partes.append(p)
        elif p.endswith(("a", "o", "e")):
            partes.append(re.escape(p) + r"s?")
        elif p.endswith("s"):
            partes.append(re.escape(p))
        else:
            partes.append(re.escape(p) + r"(?:es)?")
    return r"\s+".join(partes)


def _union(patrones):
    return "|".join(sorted(patrones, key=len, reverse=True))


_LIM_IZQ, _LIM_DER = r"(?<![a-z0-9])", r"(?![a-z0-9])"

RE_DISPOSITIVO = re.compile(_LIM_IZQ + "(?:" + _union(_patron_dispositivo(d) for d in config.DISPOSITIVOS) + ")" + _LIM_DER)
RE_LUGAR = re.compile(_LIM_IZQ + "(?:" + _union(re.escape(plegar(c)) for c in config.CIUDADES) + ")" + _LIM_DER)
RE_ORGANIZACION = re.compile(_LIM_IZQ + "(?:" + _union(re.escape(plegar(o)) for o in config.ORGANIZACIONES) + ")" + _LIM_DER)

# ---------------------------------------------------------------------------
# Expresiones regulares (se aplican sobre el texto plegado, salvo PERSONA)
# ---------------------------------------------------------------------------
_MESES = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre"
RE_FECHA = re.compile(
    rf"(?<![a-z0-9/])(?:\d{{1,2}} de (?:{_MESES})(?: de \d{{4}})?|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|ayer|hoy|anteayer|antier)(?![a-z0-9/])")
RE_HORA = re.compile(
    r"(?<![a-z0-9:])(?:a las \d{1,2}:\d{2}(?: ?(?:hrs?|horas|h))?(?![a-z0-9])|\d{1,2}:\d{2} ?(?:hrs?|horas|h)(?![a-z0-9])|\d{1,2}:\d{2}(?![a-z0-9:]))")
RE_CODIGO = re.compile(r"(?<![a-z0-9-])(?:res|com|ind|edu)-\d{3}(?![a-z0-9])")
RE_FOLIO = re.compile(r"(?<![a-z0-9-])fol-\d{4,8}(?![a-z0-9])")
RE_MONTO = re.compile(r"\$ ?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?")
RE_ENERGIA = re.compile(r"(?<![a-z0-9,.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)? ?kwh(?![a-z0-9])")
# PERSONA se aplica al texto ORIGINAL porque necesita las mayúsculas del nombre
RE_PERSONA = re.compile(
    r"(?<![A-Za-zÁÉÍÓÚáéíóúñÑ])(?i:sra?\.?|ing\.?|lic\.?)\s+"
    r"[A-ZÁÉÍÓÚÑ][a-záéíóúüñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúüñ]+){0,2}")

_REGLAS = [
    ("ORGANIZACION", RE_ORGANIZACION, True),
    ("LUGAR", RE_LUGAR, True),
    ("DISPOSITIVO", RE_DISPOSITIVO, True),
    ("FECHA", RE_FECHA, True),
    ("HORA", RE_HORA, True),
    ("CODIGO_INSTALACION", RE_CODIGO, True),
    ("FOLIO", RE_FOLIO, True),
    ("MONTO", RE_MONTO, True),
    ("ENERGIA", RE_ENERGIA, True),
    ("PERSONA", RE_PERSONA, False),   # False = se aplica al texto original
]


def extraer_entidades(texto):
    """Extrae entidades de ``texto``.

    Devuelve una lista de ``{"texto", "tipo", "inicio", "fin"}`` ordenada por posición.
    ``inicio``/``fin`` son índices sobre el texto original (``texto[inicio:fin]``).
    """
    if not isinstance(texto, str) or not texto:
        return []
    plegado = plegar(texto)
    candidatos = []
    for tipo, patron, sobre_plegado in _REGLAS:
        fuente = plegado if sobre_plegado else texto
        for m in patron.finditer(fuente):
            candidatos.append((m.start(), m.end(), tipo))
    # Resolución de traslapes: primero las más largas, luego las más a la izquierda
    candidatos.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
    elegidas, ocupado = [], []
    for ini, fin, tipo in candidatos:
        if any(ini < f and fin > i for i, f in ocupado):
            continue
        ocupado.append((ini, fin))
        elegidas.append((ini, fin, tipo))
    elegidas.sort()
    return [{"texto": texto[i:f], "tipo": t, "inicio": i, "fin": f} for i, f, t in elegidas]


# ---------------------------------------------------------------------------
# Evaluación contra las entidades esperadas
# ---------------------------------------------------------------------------
def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluar_ner(df=None, guardar=True):
    """Calcula precisión, recall y F1 por tipo de entidad.

    Una entidad cuenta como acierto si coinciden (texto en minúsculas, tipo) dentro del
    mismo reporte (se comparan multiconjuntos, así que las repeticiones cuentan).
    Devuelve ``{"por_tipo": {tipo: {precision, recall, f1, soporte, tp, fp, fn}},
    "global": {precision, recall, f1}, "errores": [...]}``.
    Si ``guardar`` es True, agrega la clave ``"ner"`` a ``config.METRICAS_PNL``.
    """
    import pandas as pd
    if df is None:
        df = pd.read_csv(config.CSV_REPORTES_RAW)
    tp, fp, fn = Counter(), Counter(), Counter()
    errores = []
    for _, fila in df.iterrows():
        esperadas = Counter((e["texto"].lower(), e["tipo"]) for e in json.loads(fila["entidades_esperadas"]))
        halladas = Counter((e["texto"].lower(), e["tipo"]) for e in extraer_entidades(fila["texto"]))
        for clave in set(esperadas) | set(halladas):
            ok = min(esperadas[clave], halladas[clave])
            tp[clave[1]] += ok
            fn[clave[1]] += esperadas[clave] - ok
            fp[clave[1]] += halladas[clave] - ok
            if esperadas[clave] != halladas[clave] and len(errores) < 400:
                errores.append({"id_reporte": fila["id_reporte"], "clave": clave,
                                "esperadas": esperadas[clave], "halladas": halladas[clave]})
    por_tipo = {}
    for t in TIPOS:
        p, r, f = _prf(tp[t], fp[t], fn[t])
        por_tipo[t] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                       "soporte": tp[t] + fn[t], "tp": tp[t], "fp": fp[t], "fn": fn[t]}
    p, r, f = _prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    resultado = {"por_tipo": por_tipo,
                 "global": {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)},
                 "errores": errores}
    if guardar:
        ruta = config.METRICAS_PNL
        datos = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
        datos["ner"] = {t: {k: v[k] for k in ("precision", "recall", "f1")} for t, v in por_tipo.items()}
        datos["ner_global"] = resultado["global"]
        ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultado


def tabla_ner(resultado):
    """Texto con la tabla de métricas del NER (para evidencias)."""
    lineas = [f"{'Tipo':<20}{'Precisión':>10}{'Recall':>9}{'F1':>8}{'Soporte':>9}"]
    for t, m in resultado["por_tipo"].items():
        lineas.append(f"{t:<20}{m['precision']:>10.3f}{m['recall']:>9.3f}{m['f1']:>8.3f}{m['soporte']:>9}")
    g = resultado["global"]
    lineas.append(f"{'GLOBAL (micro)':<20}{g['precision']:>10.3f}{g['recall']:>9.3f}{g['f1']:>8.3f}")
    return "\n".join(lineas)


def main():
    resultado = evaluar_ner()
    print(tabla_ner(resultado))
    print("\nEjemplos de errores (esperadas vs. halladas):")
    for e in resultado["errores"][:10]:
        print(" ", e)
    ejemplo = "El Ing. Pérez reportó que el minisplit de COM-002 en Córdoba consumió 350 kWh, cobro $1,250 ayer a las 18:30 (FOL-123456) a CFE."
    print("\nPrueba:", ejemplo)
    for e in extraer_entidades(ejemplo):
        print("  ", e)


if __name__ == "__main__":
    main()
