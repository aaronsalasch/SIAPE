"""Dashboard web de SIAPE (Flask).

Ejecución:  python app/app.py   (desde la raíz o desde app/)  ->  http://127.0.0.1:5000

Páginas: Inicio, Datos, Modelo ML, Clusters, Predicción, PNL, Blockchain y Reportes.
Si faltan modelos o datos, cada página muestra un aviso ("ejecuta python pipeline.py") en
lugar de fallar.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
from datetime import datetime
from functools import wraps
from pathlib import Path

# Permite ejecutar desde la raíz (python app/app.py) o desde app/ (python app.py)
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402

import pandas as pd  # noqa: E402
from flask import (Flask, abort, jsonify, redirect, render_template, request,  # noqa: E402
                   send_from_directory, url_for)
from markupsafe import Markup, escape  # noqa: E402
from werkzeug.utils import secure_filename  # noqa: E402

from src.blockchain.blockchain import Blockchain, sha256_json  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024      # permite verificar hasta el modelo (joblib)
app.config["JSON_AS_ASCII"] = False

# ----------------------------------------------------------------------------
# Utilidades: estado de archivos, caché por fecha de modificación, cadena global
# ----------------------------------------------------------------------------
ARCHIVOS_CLAVE = {
    "datos": [config.CSV_CONSUMO_PROC, config.DIR_EVIDENCIAS / "resumen_preparacion.json"],
    "modelo": [config.MODELO_CONSUMO, config.METRICAS_ML],
    "clusters": [config.DIR_MODELOS / "clusters_resumen.json"],
    "pnl": [config.MODELO_SENTIMIENTO, config.METRICAS_PNL, config.CSV_REPORTES_PROC],
}
_CACHE: dict = {}
_CANDADO_CACHE = threading.Lock()


def _cacheado(ruta: Path, lector):
    """Lee un archivo y lo guarda en memoria hasta que cambie su fecha de modificación."""
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    mt = ruta.stat().st_mtime
    with _CANDADO_CACHE:
        previo = _CACHE.get(str(ruta))
        if previo and previo[0] == mt:
            return previo[1]
    valor = lector(ruta)
    with _CANDADO_CACHE:
        _CACHE[str(ruta)] = (mt, valor)
    return valor


def leer_json(ruta):
    return _cacheado(ruta, lambda r: json.loads(Path(r).read_text(encoding="utf-8")))


def leer_csv(ruta, **kw):
    return _cacheado(ruta, lambda r: pd.read_csv(r, **kw))


def faltantes(*claves) -> list[str]:
    """Nombres de archivos que faltan para las secciones indicadas."""
    falta = []
    for c in claves:
        for r in ARCHIVOS_CLAVE[c]:
            if not Path(r).exists():
                falta.append(Path(r).name)
    return falta


def requiere(*claves):
    """Decorador: si faltan archivos, muestra el aviso 'ejecuta python pipeline.py'."""
    def deco(f):
        @wraps(f)
        def interno(*a, **kw):
            falta = faltantes(*claves)
            if falta:
                return render_template("aviso.html", falta=falta, pagina_activa=request.endpoint)
            return f(*a, **kw)
        return interno
    return deco


_CADENA = None
_MTIME_CADENA = None
_CANDADO_CADENA = threading.RLock()


def get_cadena() -> Blockchain:
    """Cadena única compartida por todas las peticiones; se recarga si otro proceso
    (p. ej. pipeline.py) modificó el archivo."""
    global _CADENA, _MTIME_CADENA
    with _CANDADO_CADENA:
        mt = config.CADENA_JSON.stat().st_mtime if config.CADENA_JSON.exists() else None
        if _CADENA is None:
            _CADENA = Blockchain()
        elif mt != _MTIME_CADENA:
            _CADENA.cargar()
        _MTIME_CADENA = config.CADENA_JSON.stat().st_mtime if config.CADENA_JSON.exists() else None
        return _CADENA


def _marcar_cadena_actualizada():
    global _MTIME_CADENA
    _MTIME_CADENA = config.CADENA_JSON.stat().st_mtime if config.CADENA_JSON.exists() else None


# ----------------------------------------------------------------------------
# Datos derivados (con caché)
# ----------------------------------------------------------------------------
def resumen_consumo() -> dict | None:
    """Totales del dataset procesado: registros, consumo total, por tipo y serie diaria."""
    def _calc(ruta):
        df = pd.read_csv(ruta, usecols=["id_instalacion", "tipo_instalacion", "fecha_hora", "consumo_kwh"])
        df["dia"] = df["fecha_hora"].str[:10]
        diario = df.groupby(["dia", "tipo_instalacion"])["consumo_kwh"].sum().unstack(fill_value=0)
        return {
            "n_registros": int(len(df)), "n_instalaciones": int(df["id_instalacion"].nunique()),
            "consumo_total": float(df["consumo_kwh"].sum()),
            "por_tipo": {k: float(v) for k, v in df.groupby("tipo_instalacion")["consumo_kwh"].sum().sort_values(ascending=False).items()},
            "inicio": df["fecha_hora"].min(), "fin": df["fecha_hora"].max(),
            "ultima_fecha": df["fecha_hora"].max()[:10],
            "dias": list(diario.index),
            "diario": {c: [round(float(x), 1) for x in diario[c]] for c in diario.columns},
        }
    return _cacheado(config.CSV_CONSUMO_PROC, _calc)


def figuras_existentes(prefijos=()) -> list[str]:
    if not config.DIR_FIGURAS.exists():
        return []
    return sorted(p.name for p in config.DIR_FIGURAS.glob("*.png") if not prefijos or p.name.startswith(tuple(prefijos)))


TITULOS_FIGURAS = {
    "eda_consumo_hora_tipo.png": "Consumo medio por hora del día y tipo de instalación",
    "eda_temp_vs_consumo.png": "Temperatura frente a consumo",
    "eda_correlacion.png": "Correlación entre variables",
    "comparacion_modelos.png": "Comparación de modelos (error y R²)",
    "real_vs_predicho.png": "Consumo real frente a predicho",
    "importancia_variables.png": "Importancia de las variables",
    "residuos.png": "Distribución de los errores (residuos)",
    "kmeans_codo_silhouette.png": "Elección de k: codo y silhouette",
    "kmeans_centroides.png": "Perfil horario de cada grupo",
    "pnl_matriz_confusion.png": "Matriz de confusión del sentimiento",
    "pnl_comparacion_clasificadores.png": "Comparación de clasificadores de sentimiento",
    "pnl_distribucion_sentimiento.png": "Distribución del sentimiento",
    "pnl_entidades_frecuentes.png": "Entidades más frecuentes",
    "pnl_terminos_tfidf.png": "Términos que más influyen en cada sentimiento",
}


def galeria(nombres) -> list[dict]:
    existentes = set(figuras_existentes())
    return [{"archivo": n, "titulo": TITULOS_FIGURAS.get(n, n)} for n in nombres if n in existentes]


@app.template_filter("num")
def filtro_num(v, dec=0):
    try:
        return f"{float(v):,.{int(dec)}f}"
    except (TypeError, ValueError):
        return "N/D"


@app.template_filter("hora_txt")
def filtro_hora_txt(fecha_hora):
    return str(fecha_hora)[11:16]


@app.template_filter("cor_sent")
def filtro_cor_sent(s):
    return {"positivo": "#2a9d8f", "negativo": "#e76f51", "neutral": "#8d99ae"}.get(s, "#999999")


@app.template_filter("pct")
def filtro_pct(v, dec=1):
    try:
        return f"{100 * float(v):.{int(dec)}f} %"
    except (TypeError, ValueError):
        return "N/D"


@app.context_processor
def contexto_global():
    return {"anio": datetime.now().year, "pagina_activa": request.endpoint,
            "hay_faltantes": bool(faltantes("datos", "modelo", "clusters", "pnl"))}


# ----------------------------------------------------------------------------
# Páginas
# ----------------------------------------------------------------------------
@app.route("/")
def inicio():
    kpi = {}
    res = resumen_consumo()
    met = leer_json(config.METRICAS_ML)
    pnl = leer_json(config.METRICAS_PNL)
    proc = leer_csv(config.CSV_REPORTES_PROC)
    clus = leer_json(config.DIR_MODELOS / "clusters_resumen.json")
    if res:
        kpi.update(registros=res["n_registros"], instalaciones=res["n_instalaciones"], consumo_total=res["consumo_total"],
                   por_tipo=res["por_tipo"], periodo=(res["inicio"][:10], res["fin"][:10]))
    if met:
        m = met["modelos"][met["seleccionado"]]
        kpi.update(modelo=met["seleccionado"], r2=m["r2"], mae=m["mae"], rmse=m["rmse"])
    if proc is not None:
        dist = proc["sentimiento_predicho"].value_counts(normalize=True).to_dict()
        kpi.update(n_reportes=len(proc), sentimiento=dist)
    if pnl:
        kpi["f1_pnl"] = pnl["modelos"][pnl["seleccionado"]]["f1_macro"]
    if clus:
        kpi.update(k=clus["k"], silhouette=clus["silhouette"])
    cadena = get_cadena()
    kpi["bloques"] = len(cadena.cadena)
    return render_template("inicio.html", kpi=kpi, faltan=faltantes("datos", "modelo", "clusters", "pnl"))


@app.route("/datos")
@requiere("datos")
def datos():
    prep = leer_json(config.DIR_EVIDENCIAS / "resumen_preparacion.json")
    res = resumen_consumo()
    proc_rep = leer_csv(config.CSV_REPORTES_PROC)
    tam = {"filas_raw": prep.get("filas_raw"), "filas_proc": prep.get("filas_procesadas"),
           "columnas_proc": prep.get("columnas_procesadas"), "reportes": None if proc_rep is None else len(proc_rep)}
    nulos_antes = prep.get("nulos_antes", {})
    nulos_desp = prep.get("nulos_despues", {})
    columnas_nulos = list(nulos_desp.keys())
    figs = galeria(["eda_consumo_hora_tipo.png", "eda_temp_vs_consumo.png", "eda_correlacion.png"])
    corr = sorted((prep.get("correlacion_con_consumo_relativo") or {}).items(), key=lambda kv: -abs(kv[1]))[:10]
    return render_template("datos.html", prep=prep, res=res, tam=tam, nulos_antes=nulos_antes, nulos_desp=nulos_desp,
                           columnas_nulos=columnas_nulos, figs=figs, corr=corr)


@app.route("/modelo")
@requiere("modelo")
def modelo():
    met = leer_json(config.METRICAS_ML)
    modelos = met["modelos"]
    orden = sorted(modelos.items(), key=lambda kv: kv[1]["rmse"])
    imp = sorted(met.get("importancias", {}).items(), key=lambda kv: -kv[1])
    figs = galeria(["comparacion_modelos.png", "real_vs_predicho.png", "importancia_variables.png", "residuos.png"]
                   + [f for f in figuras_existentes(("serie_prueba_",))])
    return render_template("modelo.html", met=met, orden=orden, imp=imp[:12], figs=figs,
                           sel=met["seleccionado"], imp_lag=met.get("importancias", {}).get("consumo_lag_1h"))


@app.route("/clusters")
@requiere("clusters")
def clusters():
    c = leer_json(config.DIR_MODELOS / "clusters_resumen.json")
    ks = sorted(c["silhouettes"], key=int)
    figs = galeria(["kmeans_codo_silhouette.png", "kmeans_centroides.png"])
    return render_template("clusters.html", c=c, ks=ks, figs=figs, sil=[round(c["silhouettes"][k], 3) for k in ks])


# ---- Predicción ----
def _inst_lista():
    return config.INSTALACIONES


@app.route("/prediccion", methods=["GET", "POST"])
@requiere("modelo", "datos")
def prediccion():
    from src.ml.predecir import pronostico_24h, predecir_consumo
    res = resumen_consumo()
    fecha_def = str((pd.Timestamp(res["ultima_fecha"]) + pd.Timedelta(days=1)).date())
    form = {"id_instalacion": "RES-001", "fecha_hora": fecha_def + "T20:00", "temperatura_c": "30", "humedad_pct": "75",
            "es_festivo": "", "temp_todo_dia": ""}
    resultado = pronostico = error = None
    if request.method == "POST":
        form.update({k: request.form.get(k, "") for k in form})
        try:
            temp = float(form["temperatura_c"])
            hum = float(form["humedad_pct"])
            if not (-10 <= temp <= 55):
                raise ValueError("La temperatura debe estar entre -10 y 55 °C.")
            if not (0 <= hum <= 100):
                raise ValueError("La humedad debe estar entre 0 y 100 %.")
            fest = 1 if form["es_festivo"] else 0
            fh = form["fecha_hora"].replace("T", " ")
            if len(fh) == 10:
                fh += " 12:00"
            pd.Timestamp(fh)                                   # valida el formato
            accion = request.form.get("accion", "predecir")
            if accion == "pronostico":
                temps = [temp] * 24 if form["temp_todo_dia"] else None
                filas = pronostico_24h(form["id_instalacion"], fh[:10], temps)
                pronostico = {"id_instalacion": form["id_instalacion"], "fecha": fh[:10], "filas": filas,
                              "total": round(sum(f["prediccion_kwh"] for f in filas), 2),
                              "pico": max(filas, key=lambda f: f["prediccion_kwh"])}
            else:
                resultado = predecir_consumo(form["id_instalacion"], fh, temp, hum, fest)
        except (ValueError, KeyError) as e:
            error = "No se pudo calcular: %s" % e
    return render_template("prediccion.html", instalaciones=_inst_lista(), form=form, resultado=resultado,
                           pronostico=pronostico, error=error)


@app.route("/api/registrar_prediccion", methods=["POST"])
@requiere("modelo", "datos")
def api_registrar_prediccion():
    """Recalcula la predicción en el servidor (no se confía en lo que envía el navegador),
    la registra como dict en la blockchain (tipo "prediccion") y devuelve el bloque."""
    from src.ml.predecir import predecir_consumo
    d = request.get_json(silent=True) or request.form
    try:
        fest = 1 if str(d.get("es_festivo", "")).lower() in ("1", "true", "on", "si") else 0
        fh = str(d["fecha_hora"]).replace("T", " ")
        r = predecir_consumo(d["id_instalacion"], fh, float(d["temperatura_c"]), float(d["humedad_pct"]), fest)
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"ok": False, "error": "Datos de la predicción inválidos: %s" % e}), 400
    cadena = get_cadena()
    nombre = "prediccion_%s_%s" % (r["id_instalacion"], r["fecha_hora"].replace(" ", "_").replace(":", ""))
    with _CANDADO_CADENA:
        b = cadena.registrar_contenido(r, "prediccion", nombre, {
            "prediccion_kwh": r["prediccion_kwh"], "intervalo": r["intervalo"], "modelo": r["modelo"]})
        _marcar_cadena_actualizada()
    # Texto JSON con el formato de Python (los números como 30.0 se conservan: así, al pegarlo
    # en "Verificar", se obtiene exactamente el mismo hash que se registró).
    return jsonify({"ok": True, "bloque": b.a_dict(), "contenido": r, "hash_documento": sha256_json(r),
                    "contenido_texto": json.dumps(r, ensure_ascii=False, indent=2)})


# ---- PNL ----
COLORES_ENTIDAD = {"DISPOSITIVO": "ent-disp", "LUGAR": "ent-lugar", "ORGANIZACION": "ent-org", "FECHA": "ent-fecha",
                   "HORA": "ent-hora", "CODIGO_INSTALACION": "ent-cod", "FOLIO": "ent-folio", "MONTO": "ent-monto",
                   "ENERGIA": "ent-energia", "PERSONA": "ent-persona"}
NOMBRES_ENTIDAD = {"DISPOSITIVO": "Equipo", "LUGAR": "Lugar", "ORGANIZACION": "Organización", "FECHA": "Fecha",
                   "HORA": "Hora", "CODIGO_INSTALACION": "Código de instalación", "FOLIO": "Folio", "MONTO": "Monto",
                   "ENERGIA": "Energía", "PERSONA": "Persona"}


def resaltar_entidades(texto: str, entidades: list) -> Markup:
    """Devuelve el texto como HTML seguro con cada entidad envuelta en <mark> de su color."""
    salida, pos = [], 0
    for e in sorted(entidades, key=lambda x: (x["inicio"], -x["fin"])):
        if e["inicio"] < pos:                 # se omiten solapamientos
            continue
        salida.append(str(escape(texto[pos:e["inicio"]])))
        salida.append('<mark class="ent %s" title="%s">%s<small>%s</small></mark>' % (
            COLORES_ENTIDAD.get(e["tipo"], "ent-otro"), escape(NOMBRES_ENTIDAD.get(e["tipo"], e["tipo"])),
            escape(texto[e["inicio"]:e["fin"]]), escape(NOMBRES_ENTIDAD.get(e["tipo"], e["tipo"]))))
        pos = e["fin"]
    salida.append(str(escape(texto[pos:])))
    return Markup("".join(salida))


EJEMPLOS_PNL = [
    "El transformador de IND-001 hizo un ruido horrible ayer a las 22:15 y CFE no responde, folio FOL-445566. Pésimo servicio!!",
    "Gracias al Ing. Carlos Ruiz por reparar el minisplit en Xalapa, ahora funciona de maravilla.",
    "Solicito el duplicado del recibo de $1,250 correspondiente a 350 kWh del 15/03/2026.",
]


@app.route("/pnl", methods=["GET", "POST"])
@requiere("pnl")
def pnl():
    from src.nlp.analizador import analizar_texto
    texto = request.form.get("texto", "").strip() if request.method == "POST" else ""
    analisis = resaltado = error = None
    if request.method == "POST":
        if not texto:
            error = "Escribe un texto para analizar."
        elif len(texto) > 3000:
            error = "El texto es demasiado largo (máximo 3,000 caracteres)."
        else:
            analisis = analizar_texto(texto)
            resaltado = resaltar_entidades(texto, analisis["entidades"])
    # ---- tabla de reportes recientes con filtros ----
    df = leer_csv(config.CSV_REPORTES_PROC)
    f_sent = request.args.get("sentimiento", "")
    f_inst = request.args.get("instalacion", "")
    f_q = request.args.get("q", "").strip()
    vista = df
    if f_sent:
        vista = vista[vista["sentimiento_predicho"] == f_sent]
    if f_inst:
        vista = vista[vista["id_instalacion"] == f_inst]
    if f_q:
        vista = vista[vista["texto"].str.contains(f_q, case=False, regex=False, na=False)]
    total_filtrado = len(vista)
    vista = vista.sort_values("fecha_hora", ascending=False).head(25)
    filas = vista[["id_reporte", "id_instalacion", "fecha_hora", "canal", "texto", "sentimiento_predicho", "confianza", "n_entidades"]].to_dict("records")
    met = leer_json(config.METRICAS_PNL)
    dist = df["sentimiento_predicho"].value_counts().to_dict()
    return render_template("pnl.html", texto=texto, analisis=analisis, resaltado=resaltado, error=error,
                           colores=COLORES_ENTIDAD, nombres=NOMBRES_ENTIDAD, ejemplos=EJEMPLOS_PNL,
                           filas=filas, total_filtrado=total_filtrado, total=len(df), f_sent=f_sent, f_inst=f_inst, f_q=f_q,
                           instalaciones=[i["id"] for i in config.INSTALACIONES], met=met, dist=dist,
                           figs=galeria(["pnl_distribucion_sentimiento.png", "pnl_entidades_frecuentes.png",
                                         "pnl_terminos_tfidf.png", "pnl_matriz_confusion.png", "pnl_comparacion_clasificadores.png"]))


# ---- Blockchain ----
def _parece_json(t: str):
    t = t.strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            v = json.loads(t)
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            return None
    return None


@app.route("/blockchain", methods=["GET", "POST"])
def blockchain():
    return _pagina_blockchain()


def _pagina_blockchain(verificacion=None, busqueda=None):
    cadena = get_cadena()
    bloques = cadena.a_lista()
    todos = request.args.get("todos") == "1"
    h = request.args.get("hash", "").strip()
    if h and busqueda is None:
        b = cadena.buscar_por_hash(h)
        busqueda = {"hash": h, "bloque": b.a_dict() if b else None}
    tipos = {}
    for b in bloques:
        tipos[b["datos"]["tipo"]] = tipos.get(b["datos"]["tipo"], 0) + 1
    mostrados = list(reversed(bloques)) if todos else list(reversed(bloques))[:30]
    return render_template("blockchain.html", bloques=mostrados, n_total=len(bloques), todos=todos, tipos=tipos,
                           verificacion=verificacion, busqueda=busqueda, dificultad=cadena.dificultad)


@app.route("/blockchain/validar", methods=["POST"])
def blockchain_validar():
    return jsonify(get_cadena().validar_cadena())


@app.route("/blockchain/verificar", methods=["POST"])
def blockchain_verificar():
    """Verifica un archivo subido (con su nombre original) o un texto/JSON pegado."""
    cadena = get_cadena()
    archivo = request.files.get("archivo")
    texto = request.form.get("texto", "")
    nombre_texto = request.form.get("nombre", "").strip() or None
    v = {"origen": None, "resultado": None, "error": None}
    if archivo and archivo.filename:
        nombre = secure_filename(archivo.filename) or archivo.filename
        with tempfile.TemporaryDirectory(prefix="siape_ver_") as tmp:
            ruta = Path(tmp) / "subido.bin"
            archivo.save(ruta)
            v["resultado"] = cadena.verificar_archivo(ruta, nombre=archivo.filename)
        v["origen"] = "Archivo: %s" % archivo.filename
    elif texto.strip():
        como_json = _parece_json(texto)
        v["resultado"] = cadena.verificar_contenido(como_json if como_json is not None else texto, nombre_texto)
        v["origen"] = "Texto pegado (%s)" % ("interpretado como JSON" if como_json is not None else "texto plano")
    else:
        v["error"] = "Sube un archivo o pega un texto para verificar."
    if request.headers.get("Accept") == "application/json":
        return jsonify(v)
    return _pagina_blockchain(verificacion=v)


@app.route("/blockchain/demo", methods=["POST"])
def blockchain_demo():
    """Demostración en vivo en una cadena TEMPORAL (la cadena real no se toca)."""
    pasos = []
    with tempfile.TemporaryDirectory(prefix="siape_demo_vivo_") as tmp:
        ruta = Path(tmp) / "cadena_demo.json"
        c = Blockchain(ruta)
        doc = "Reporte de ejemplo: consumo total 1,250 kWh en la instalación RES-001."
        b = c.registrar_contenido(doc, "reporte", "ejemplo.txt")
        c.registrar_contenido("Segundo documento de ejemplo", "reporte", "ejemplo2.txt")
        pasos.append({"titulo": "1. Se registra un documento en la cadena temporal",
                      "detalle": "Bloque #%d con hash %s..." % (b.indice, b.datos["hash_documento"][:20]), "ok": True})
        r = c.verificar_contenido(doc, "ejemplo.txt")
        pasos.append({"titulo": "2. Se verifica el documento original", "detalle": r["mensaje"], "ok": r["integro"]})
        r = c.verificar_contenido(doc.replace("1,250", "1,150"), "ejemplo.txt")
        pasos.append({"titulo": "3. Se cambia un número del documento (1,250 por 1,150) y se verifica",
                      "detalle": r["mensaje"], "ok": not r["integro"]})
        v = c.validar_cadena()
        pasos.append({"titulo": "4. Se valida la cadena sin tocarla", "detalle": "Cadena %s (%d bloques)." % ("válida" if v["valida"] else "inválida", v["n_bloques"]),
                      "ok": v["valida"]})
        # Alteración del bloque 1 directamente en el archivo temporal
        contenido = json.loads(ruta.read_text(encoding="utf-8"))
        contenido["bloques"][1]["datos"]["metadatos"] = {"alterado": "alguien cambió este bloque"}
        ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=2), encoding="utf-8")
        v2 = Blockchain(ruta).validar_cadena()
        pasos.append({"titulo": "5. Se altera el bloque #1 dentro del archivo y se valida de nuevo",
                      "detalle": "Cadena %s. %s" % ("válida" if v2["valida"] else "INVÁLIDA", " | ".join(v2["errores"][:3])),
                      "ok": not v2["valida"]})
    return jsonify({"pasos": pasos, "todo_correcto": all(p["ok"] for p in pasos)})


# ---- Reportes ----
def _listar_reportes() -> list[dict]:
    cadena = get_cadena()
    salida = []
    for p in sorted(config.DIR_REPORTES.glob("reporte_*.json"), reverse=True):
        ver = cadena.verificar_archivo(p)
        meta = {}
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
            meta = {"fecha": c.get("fecha_generacion"), "r2": c.get("modelo", {}).get("r2"),
                    "n_registros": c.get("datos", {}).get("n_registros")}
        except (json.JSONDecodeError, OSError):
            pass
        salida.append({"nombre": p.name, "txt": p.with_suffix(".txt").name if p.with_suffix(".txt").exists() else None,
                       "tam_kb": round(p.stat().st_size / 1024, 1), "ver": ver, **meta})
    return salida


@app.route("/reportes")
def reportes():
    return render_template("reportes.html", reportes=_listar_reportes(),
                           faltan=faltantes("datos", "modelo", "clusters", "pnl"))


@app.route("/reportes/generar", methods=["POST"])
@requiere("datos", "modelo", "clusters", "pnl")
def reportes_generar():
    from src.reporte.generar_reporte import generar_reporte
    with _CANDADO_CADENA:
        r = generar_reporte(registrar=True, cadena=get_cadena())
        _marcar_cadena_actualizada()
    return redirect(url_for("reporte_ver", nombre=r["nombre"], nuevo=1))


def _ruta_reporte(nombre: str) -> Path:
    nombre = secure_filename(nombre)
    ruta = config.DIR_REPORTES / nombre
    if not nombre.startswith("reporte_") or not ruta.exists():
        abort(404)
    return ruta


@app.route("/reportes/<nombre>")
def reporte_ver(nombre):
    ruta = _ruta_reporte(nombre)
    if ruta.suffix != ".json":
        abort(404)
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    ver = get_cadena().verificar_archivo(ruta)
    return render_template("reporte_ver.html", nombre=ruta.name, c=contenido, ver=ver,
                           txt=ruta.with_suffix(".txt").name, nuevo=request.args.get("nuevo") == "1")


@app.route("/reportes/descargar/<nombre>")
def reporte_descargar(nombre):
    ruta = _ruta_reporte(nombre)
    return send_from_directory(config.DIR_REPORTES, ruta.name, as_attachment=True)


# ---- Figuras ----
@app.route("/figuras/<nombre>")
def figura(nombre):
    if not nombre.lower().endswith(".png"):
        abort(404)
    return send_from_directory(config.DIR_FIGURAS, secure_filename(nombre))


@app.errorhandler(404)
def no_encontrado(e):
    return render_template("aviso.html", falta=[], mensaje="La página que buscas no existe.", pagina_activa=None), 404


if __name__ == "__main__":
    print("SIAPE dashboard en http://127.0.0.1:5000  (Ctrl+C para detener)")
    app.run(host="127.0.0.1", port=5000, debug=False)
