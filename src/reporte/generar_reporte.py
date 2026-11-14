"""Generador del reporte automático de SIAPE (consolida ML + clusters + PNL + predicciones).

Flujo:
  1. Reúne las cifras de los demás módulos (datos procesados, métricas del modelo,
     clusters, resumen de PNL y el pronóstico de 24 h de dos instalaciones).
  2. Redacta el texto en español con la NLG de ``src.nlp.generacion`` (campo ``texto_nlg``).
  3. Escribe ``outputs/reportes/reporte_<AAAAMMDD_HHMMSS>.json`` y ``.txt``.
  4. Calcula el SHA-256 del archivo .json y lo registra en la blockchain (tipo "reporte").
  5. Deja evidencia en ``outputs/evidencias/reporte_registro.txt``.

Ejecución: ``python -m src.reporte.generar_reporte``
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

import pandas as pd  # noqa: E402

from src.blockchain.blockchain import Blockchain, sha256_archivo  # noqa: E402
from src.ml.predecir import pronostico_24h  # noqa: E402
from src.nlp.generacion import generar_resumen  # noqa: E402

# Instalaciones "representativas": una de bajo consumo y una de alto consumo.
INSTALACIONES_REPRESENTATIVAS = ["RES-001", "IND-001"]


def _leer_json(ruta: Path) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def _resumen_datos() -> dict:
    """Cifras globales del dataset procesado."""
    df = pd.read_csv(config.CSV_CONSUMO_PROC, usecols=["id_instalacion", "tipo_instalacion", "fecha_hora", "consumo_kwh"])
    por_tipo = df.groupby("tipo_instalacion")["consumo_kwh"].sum().sort_values(ascending=False)
    return {
        "n_registros": int(len(df)),
        "n_instalaciones": int(df["id_instalacion"].nunique()),
        "periodo": {"inicio": str(df["fecha_hora"].min()), "fin": str(df["fecha_hora"].max())},
        "consumo_total_kwh": round(float(df["consumo_kwh"].sum()), 2),
        "consumo_por_tipo": {k: round(float(v), 2) for k, v in por_tipo.items()},
        "ultima_fecha": str(df["fecha_hora"].max())[:10],
    }


def _resumen_pronostico(id_inst: str, fecha: str) -> dict:
    """Pronóstico de 24 h de una instalación, resumido (total, pico y valores por hora)."""
    filas = pronostico_24h(id_inst, fecha)
    kwh = [f["prediccion_kwh"] for f in filas]
    i_pico = max(range(24), key=lambda i: kwh[i])
    tipo = next(i["tipo"] for i in config.INSTALACIONES if i["id"] == id_inst)
    return {
        "id_instalacion": id_inst,
        "tipo": tipo,
        "fecha": fecha,
        "fecha_hora": filas[i_pico]["fecha_hora"],       # hora del pico (para el texto NLG)
        "prediccion_kwh": filas[i_pico]["prediccion_kwh"],
        "intervalo": filas[i_pico]["intervalo"],
        "total_24h_kwh": round(sum(kwh), 3),
        "pico_kwh": kwh[i_pico],
        "hora_pico": "%02d:00" % i_pico,
        "minimo_kwh": min(kwh),
        "horas": [{"hora": i, "kwh": kwh[i], "intervalo": filas[i]["intervalo"]} for i in range(24)],
    }


def construir_contenido() -> dict:
    """Arma el diccionario completo del reporte (sin escribirlo a disco)."""
    from src.nlp.analizador import resumen_pnl

    datos = _resumen_datos()
    met = _leer_json(config.METRICAS_ML)
    sel = met["seleccionado"]
    m = met["modelos"][sel]
    clus = _leer_json(config.DIR_MODELOS / "clusters_resumen.json")
    met_pnl = _leer_json(config.METRICAS_PNL)
    proc = pd.read_csv(config.CSV_REPORTES_PROC)
    pnl = resumen_pnl(proc)

    # Pronóstico para el día siguiente al último dato histórico
    fecha_pron = str((pd.Timestamp(datos["ultima_fecha"]) + pd.Timedelta(days=1)).date())
    pronosticos = [_resumen_pronostico(i, fecha_pron) for i in INSTALACIONES_REPRESENTATIVAS]

    importancias = sorted(met.get("importancias", {}).items(), key=lambda kv: -kv[1])[:8]
    modelo = {
        "nombre": sel, "r2": m["r2"], "mae": m["mae"], "rmse": m["rmse"], "mape": m.get("mape"),
        "rmse_cv": m.get("rmse_cv"), "criterio_seleccion": met.get("criterio_seleccion"),
        "cobertura_intervalo_90": met.get("cobertura_intervalo_90"),
        "hiperparametros": met.get("hiperparametros"),
        "metricas_por_tipo": met.get("metricas_por_tipo"),
        "linea_base_24h": met["modelos"].get("Línea base (hace 24 h)"),
        "importancias_top": {k: round(v, 4) for k, v in importancias},
        "n_entrenamiento": met.get("n_entrenamiento"), "n_prueba": met.get("n_prueba"),
    }
    clusters = {
        "k": clus["k"], "silhouette": clus["silhouette"], "tamanos": clus.get("tamanos"),
        "descripciones": clus.get("descripciones"),
        "dias_atipicos": clus.get("dias_atipicos", [])[:5],
        "n_dias_atipicos": len(clus.get("dias_atipicos", [])),
    }
    pnl_completo = dict(pnl)
    pnl_completo.update({"modelo": met_pnl.get("seleccionado"),
                         "f1_macro": (met_pnl.get("modelos", {}).get(met_pnl.get("seleccionado"), {}) or {}).get("f1_macro")})

    # ---- Texto NLG: datos globales + pronóstico de la 1.ª instalación ----
    p1, p2 = pronosticos[0], pronosticos[1]
    entrada_nlg = {
        "n_registros": datos["n_registros"], "n_instalaciones": datos["n_instalaciones"],
        "periodo": datos["periodo"], "consumo_total_kwh": datos["consumo_total_kwh"],
        "consumo_por_tipo": datos["consumo_por_tipo"],
        "modelo": {"nombre": sel, "r2": m["r2"], "mae": m["mae"], "rmse": m["rmse"]},
        "prediccion": {k: p1[k] for k in ("id_instalacion", "fecha_hora", "prediccion_kwh", "intervalo",
                                          "total_24h_kwh", "pico_kwh", "hora_pico")},
        "clusters": {"k": clus["k"], "silhouette": clus["silhouette"], "dias_atipicos": clus.get("dias_atipicos", [])},
        "pnl": pnl,
    }
    texto = generar_resumen(entrada_nlg)
    # Párrafo de predicción de la 2.ª instalación: se reutiliza la NLG y se toma solo el párrafo
    # que empieza con "Como ejemplo de uso" (el resto -alcance genérico y cierre- se descarta).
    extra = generar_resumen({"prediccion": {k: p2[k] for k in ("id_instalacion", "fecha_hora", "prediccion_kwh", "intervalo",
                                                                "total_24h_kwh", "pico_kwh", "hora_pico")}})
    parrafo2 = next((p for p in extra.split("\n\n") if p.startswith("Como ejemplo de uso")), "")
    parrafo2 = parrafo2.replace("Como ejemplo de uso, el modelo estima", "Para una instalación de mayor tamaño, el modelo estima", 1)
    partes = texto.split("\n\n")
    idx = next((i for i, p in enumerate(partes) if p.startswith("Como ejemplo de uso")), None)
    nota = ("Nota sobre el rango de predicción: es el rango entre los percentiles 5 y 95 de las predicciones de los "
            "árboles del Random Forest; en la prueba contuvo el valor real el %.1f %% de las veces, por lo que debe leerse "
            "como un rango orientativo y no como un intervalo de confianza del 90 %%."
            % (100 * (met.get("cobertura_intervalo_90") or 0)))
    pos = (idx + 1) if idx is not None else len(partes)
    partes[pos:pos] = [t for t in (parrafo2, nota) if t]
    texto = "\n\n".join(partes)

    return {
        "titulo": "Reporte automático SIAPE - análisis y predicción de consumo energético",
        "fecha_generacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datos": datos,
        "modelo": modelo,
        "predicciones": pronosticos,
        "clusters": clusters,
        "pnl": pnl_completo,
        "texto_nlg": texto,
    }


def generar_reporte(registrar: bool = True, cadena: Blockchain | None = None) -> dict:
    """Genera el reporte (.json y .txt) y, si ``registrar``, registra su hash en la blockchain.

    ``cadena``: instancia de Blockchain a usar (la app Flask pasa la suya para no tener dos
    copias en memoria); si es None se abre la cadena por omisión.
    Devuelve {ruta_json, ruta_txt, nombre, hash, bloque, hash_bloque, registrado, texto_nlg, contenido}.
    """
    contenido = construir_contenido()
    sello = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.DIR_REPORTES.mkdir(parents=True, exist_ok=True)
    ruta_json = config.DIR_REPORTES / ("reporte_%s.json" % sello)
    ruta_txt = config.DIR_REPORTES / ("reporte_%s.txt" % sello)
    # Si se genera dos veces en el mismo segundo, se evita sobrescribir un reporte ya registrado
    n = 1
    while ruta_json.exists():
        n += 1
        ruta_json = config.DIR_REPORTES / ("reporte_%s_%d.json" % (sello, n))
        ruta_txt = config.DIR_REPORTES / ("reporte_%s_%d.txt" % (sello, n))

    ruta_json.write_text(json.dumps(contenido, ensure_ascii=False, indent=2), encoding="utf-8")
    d = contenido["datos"]
    encabezado = ["=" * 72, contenido["titulo"].upper(), "Generado: " + contenido["fecha_generacion"],
                  "Periodo analizado: %s a %s" % (d["periodo"]["inicio"], d["periodo"]["fin"]), "=" * 72, ""]
    ruta_txt.write_text("\n".join(encabezado) + contenido["texto_nlg"] + "\n", encoding="utf-8")

    hash_doc = sha256_archivo(ruta_json)       # hash del ARCHIVO .json tal como quedó en disco
    res = {"ruta_json": str(ruta_json), "ruta_txt": str(ruta_txt), "nombre": ruta_json.name,
           "hash": hash_doc, "bloque": None, "hash_bloque": None, "registrado": False,
           "texto_nlg": contenido["texto_nlg"], "contenido": contenido}
    if registrar:
        cadena = cadena or Blockchain()
        bloque = cadena.registrar_archivo(ruta_json, "reporte", {
            "modelo": contenido["modelo"]["nombre"], "r2": round(contenido["modelo"]["r2"], 4),
            "n_registros": d["n_registros"], "archivo_txt": ruta_txt.name})
        res.update(bloque=bloque.indice, hash_bloque=bloque.hash, registrado=True)
        val = cadena.validar_cadena()
        _evidencia(res, val, bloque)
    return res


def _evidencia(res: dict, val: dict, bloque) -> None:
    """Escribe outputs/evidencias/reporte_registro.txt con el detalle del registro."""
    L = ["EVIDENCIA - GENERACIÓN Y REGISTRO DEL REPORTE EN LA BLOCKCHAIN", "=" * 72,
         "Fecha: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         "Reporte JSON      : " + res["nombre"],
         "Reporte TXT       : " + Path(res["ruta_txt"]).name,
         "SHA-256 (archivo) : " + res["hash"],
         "Bloque registrado : #%d (tipo=%s)" % (bloque.indice, bloque.datos["tipo"]),
         "Hash del bloque   : " + bloque.hash,
         "Hash anterior     : " + bloque.hash_anterior,
         "Nonce (PoW)       : %d" % bloque.nonce,
         "Marca de tiempo   : " + bloque.marca_tiempo,
         "Validación cadena : %s (%d bloques, %d errores)" % ("VÁLIDA" if val["valida"] else "INVÁLIDA", val["n_bloques"], len(val["errores"])),
         "", "Cómo verificarlo: python pipeline.py --verificar   (recalcula el SHA-256 y lo compara con el bloque)",
         "", "-" * 72, "TEXTO GENERADO (NLG)", "-" * 72, res["texto_nlg"], ""]
    (config.DIR_EVIDENCIAS / "reporte_registro.txt").write_text("\n".join(L), encoding="utf-8")


def main() -> dict:
    res = generar_reporte(registrar=True)
    print("Reporte JSON :", res["ruta_json"])
    print("Reporte TXT  :", res["ruta_txt"])
    print("SHA-256      :", res["hash"])
    print("Bloque       : #%s" % res["bloque"])
    return res


if __name__ == "__main__":
    main()
