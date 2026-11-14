"""Analizador integral de texto: preprocesamiento + sentimiento + entidades.

Ejecución: ``python -m src.nlp.analizador``  (procesa todos los reportes y deja evidencias).

Funciones públicas:
  * ``analizar_texto(texto) -> dict``  (para el dashboard y pruebas con frases nuevas)
  * ``procesar_reportes() -> pd.DataFrame``  (escribe ``config.CSV_REPORTES_PROC``, figuras
    y ``outputs/evidencias/pnl_ejemplos.txt``)
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.nlp.entidades import extraer_entidades  # noqa: E402
from src.nlp.procesamiento_texto import preprocesar  # noqa: E402
from src.nlp.sentimiento import predecir_sentimiento, predecir_sentimiento_lote  # noqa: E402

COLORES = {"positivo": "#2a9d8f", "negativo": "#e76f51", "neutral": "#8d99ae"}


def analizar_texto(texto):
    """Analiza un texto libre.

    Devuelve ``{"preprocesado": {normalizado, tokens, sin_stopwords, stems, lemas},
    "sentimiento": {"sentimiento", "probabilidades"}, "entidades": [{"texto","tipo","inicio","fin"}]}``.
    """
    return {
        "preprocesado": preprocesar(texto),
        "sentimiento": predecir_sentimiento(texto),
        "entidades": extraer_entidades(texto),
    }


def _leer_raw():
    if not config.CSV_REPORTES_RAW.exists():
        from src.data.generar_reportes import generar_reportes
        generar_reportes()
    return pd.read_csv(config.CSV_REPORTES_RAW)


def procesar_reportes():
    """Procesa todos los reportes raw y escribe ``config.CSV_REPORTES_PROC``."""
    raw = _leer_raw()
    textos = raw["texto"].fillna("").tolist()
    etiquetas, proba, _ = predecir_sentimiento_lote(textos)
    ents = [extraer_entidades(t) for t in textos]
    proc = pd.DataFrame({
        "id_reporte": raw["id_reporte"],
        "id_instalacion": raw["id_instalacion"],
        "fecha_hora": raw["fecha_hora"],
        "canal": raw["canal"],
        "texto": raw["texto"],
        "texto_limpio": [preprocesar(t)["normalizado"] for t in textos],
        "sentimiento_etiqueta": raw["sentimiento_etiqueta"],
        "sentimiento_predicho": etiquetas,
        "confianza": proba.max(axis=1).round(4),
        "entidades": [json.dumps([{k: e[k] for k in ("texto", "tipo", "inicio", "fin")} for e in lista],
                                 ensure_ascii=False) for lista in ents],
        "n_entidades": [len(lista) for lista in ents],
    })
    config.DIR_DATA_PROC.mkdir(parents=True, exist_ok=True)
    proc.to_csv(config.CSV_REPORTES_PROC, index=False, encoding="utf-8")
    _figuras(proc)
    _evidencia_ejemplos(proc)
    return proc


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def resumen_pnl(proc):
    """Diccionario con el resumen PNL (útil para ``generacion.generar_resumen``)."""
    cont_ent, cont_disp = Counter(), Counter()
    for js in proc["entidades"]:
        for e in json.loads(js):
            cont_ent[e["texto"]] += 1 if e["tipo"] in ("LUGAR", "ORGANIZACION") else 0
            if e["tipo"] == "DISPOSITIVO":
                cont_disp[e["texto"].lower()] += 1
    return {"n_reportes": int(len(proc)),
            "distribucion_sentimiento": proc["sentimiento_predicho"].value_counts().to_dict(),
            "entidades_top": {k: v for k, v in cont_ent.most_common(10) if v},
            "dispositivos_top": dict(cont_disp.most_common(10))}


def _figuras(proc):
    # 1) Distribución de sentimiento: etiqueta vs. predicho y por canal
    orden = ["positivo", "neutral", "negativo"]
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
    tabla = pd.DataFrame({"etiqueta": proc["sentimiento_etiqueta"].value_counts(),
                          "predicho": proc["sentimiento_predicho"].value_counts()}).reindex(orden)
    tabla.plot(kind="bar", ax=ejes[0], color=["#264653", "#e9c46a"], rot=0)
    ejes[0].set_title("Sentimiento: etiqueta vs. predicción")
    ejes[0].set_ylabel("Reportes")
    cruz = pd.crosstab(proc["canal"], proc["sentimiento_predicho"], normalize="index")[orden] * 100
    cruz.plot(kind="bar", stacked=True, ax=ejes[1], color=[COLORES[o] for o in orden], rot=0)
    ejes[1].set_title("Sentimiento predicho por canal (%)")
    ejes[1].set_ylabel("%")
    ejes[1].legend(title="")
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / "pnl_distribucion_sentimiento.png", dpi=110)
    plt.close(fig)

    # 2) Entidades frecuentes: por tipo y top textos
    por_tipo, textos = Counter(), Counter()
    for js in proc["entidades"]:
        for e in json.loads(js):
            por_tipo[e["tipo"]] += 1
            if e["tipo"] in ("DISPOSITIVO", "LUGAR", "ORGANIZACION"):
                textos[(e["texto"].lower() if e["tipo"] == "DISPOSITIVO" else e["texto"], e["tipo"])] += 1
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.8))
    t = pd.Series(por_tipo).sort_values()
    ejes[0].barh(t.index, t.values, color="#457b9d")
    ejes[0].set_title("Entidades detectadas por tipo")
    top = textos.most_common(12)[::-1]
    pal = {"DISPOSITIVO": "#e76f51", "LUGAR": "#2a9d8f", "ORGANIZACION": "#e9c46a"}
    ejes[1].barh([k[0] for k, _ in top], [v for _, v in top], color=[pal[k[1]] for k, _ in top])
    ejes[1].set_title("Entidades más frecuentes (dispositivos, lugares, organizaciones)")
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / "pnl_entidades_frecuentes.png", dpi=110)
    plt.close(fig)

    # 3) Términos TF-IDF/peso más influyentes por clase
    ruta = config.METRICAS_PNL
    terminos = json.loads(ruta.read_text(encoding="utf-8")).get("terminos_top_por_clase", {}) if ruta.exists() else {}
    if terminos:
        fig, ejes = plt.subplots(1, 3, figsize=(14, 4.8))
        for ax, clase in zip(ejes, ["positivo", "neutral", "negativo"]):
            ts = terminos.get(clase, [])[::-1]
            ax.barh([x["termino"] for x in ts], [x["peso"] for x in ts], color=COLORES[clase])
            ax.set_title(f"Términos clave: {clase}")
            ax.set_xlabel("Peso (regresión logística sobre TF-IDF)")
        fig.tight_layout()
        fig.savefig(config.DIR_FIGURAS / "pnl_terminos_tfidf.png", dpi=110)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Evidencia de ejemplos
# ---------------------------------------------------------------------------
FRASES_NUEVAS = [
    "El transformador de IND-001 hizo un ruido horrible ayer a las 22:15 y CFE no responde, folio FOL-445566. Pésimo servicio!!",
    "Gracias al Ing. Carlos Ruiz por reparar el minisplit en Xalapa, ahora funciona de maravilla 👍",
    "Solicito el duplicado del recibo de $1,250 correspondiente a 350 kWh del 15/03/2026.",
    "El servicio fue rápido, pero el aire acondicionado sigue sin enfriar en Boca del Río",
    "No tengo ninguna queja, el compresor opera sin fallas desde el mantenimiento",
]


def _bloque(i, texto, etiqueta=None):
    a = analizar_texto(texto)
    p, s = a["preprocesado"], a["sentimiento"]
    L = [f"EJEMPLO {i}", f"  ENTRADA        : {texto}"]
    if etiqueta:
        L.append(f"  Etiqueta real  : {etiqueta}")
    L += [f"  Normalizado    : {p['normalizado']}",
          f"  Tokens         : {p['tokens']}",
          f"  Sin stopwords  : {p['sin_stopwords']}",
          f"  Stems          : {p['stems']}",
          f"  Lemas          : {p['lemas']}",
          f"  SENTIMIENTO    : {s['sentimiento']}  {s['probabilidades']}",
          "  ENTIDADES      : " + ("; ".join(f"[{e['tipo']}] {e['texto']}" for e in a["entidades"]) or "(ninguna)"), ""]
    return L


def _evidencia_ejemplos(proc):
    L = ["EVIDENCIA - EJEMPLOS DEL ANALIZADOR DE TEXTO (entrada -> salida)", "=" * 72,
         f"Reportes procesados: {len(proc)} | acierto sobre todo el corpus (incluye datos de entrenamiento, "
         f"por eso es mayor que el de prueba): {(proc['sentimiento_etiqueta'] == proc['sentimiento_predicho']).mean():.3f}",
         "", "A) Reportes del corpus (6 positivos/negativos/neutrales al azar + 2 errores del clasificador)", ""]
    muestra = pd.concat([proc[proc["sentimiento_etiqueta"] == s].sample(2, random_state=config.SEMILLA)
                         for s in ("positivo", "negativo", "neutral")])
    errores = proc[proc["sentimiento_etiqueta"] != proc["sentimiento_predicho"]].head(2)
    n = 0
    for _, f in pd.concat([muestra, errores]).iterrows():
        n += 1
        L += _bloque(n, f["texto"], f["sentimiento_etiqueta"])
    L += ["B) Frases nuevas (no están en el corpus)", ""]
    for t in FRASES_NUEVAS:
        n += 1
        L += _bloque(n, t)
    (config.DIR_EVIDENCIAS / "pnl_ejemplos.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    proc = procesar_reportes()
    print(f"Reportes procesados: {len(proc)} -> {config.CSV_REPORTES_PROC}")
    print(proc["sentimiento_predicho"].value_counts().to_string())
    print("Entidades por reporte (media):", round(proc["n_entidades"].mean(), 2))
    from src.nlp.generacion import resumir_reportes
    print(resumir_reportes(proc))
    for t in FRASES_NUEVAS[:3]:
        a = analizar_texto(t)
        print("\n>", t, "\n ", a["sentimiento"]["sentimiento"], [(e["texto"], e["tipo"]) for e in a["entidades"]])


if __name__ == "__main__":
    main()
