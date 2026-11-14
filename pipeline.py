"""Orquestador de SIAPE: ejecuta el flujo completo o etapas sueltas.

Uso (desde la raíz del proyecto):
    python pipeline.py               # = --todo
    python pipeline.py --todo        # datos -> ML -> PNL -> registro de artefactos -> reporte -> demo -> verificación
    python pipeline.py --datos       # generar y preparar datos de consumo
    python pipeline.py --ml          # entrenar modelo de regresión + clustering
    python pipeline.py --pnl         # generar reportes de texto, entrenar sentimiento y procesarlos
    python pipeline.py --registrar   # registrar dataset, modelo y métricas en la blockchain
    python pipeline.py --reporte     # generar el reporte y registrar su hash en la blockchain
    python pipeline.py --blockchain-demo   # demostración de detección de alteraciones
    python pipeline.py --verificar   # valida la cadena y verifica reportes y artefactos registrados

Es idempotente: si se repite, los archivos se regeneran y la cadena simplemente crece
(nunca se borra ni se reescribe un bloque anterior).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

from src.blockchain.blockchain import Blockchain  # noqa: E402

_TIEMPOS: list[tuple[str, float]] = []


def _etapa(titulo: str, funcion, *args, **kwargs):
    """Ejecuta una etapa mostrando encabezado y tiempo; guarda el tiempo para el resumen."""
    print("\n" + "=" * 72)
    print(">>> %s" % titulo)
    print("=" * 72, flush=True)
    t0 = time.perf_counter()
    resultado = funcion(*args, **kwargs)
    dt = time.perf_counter() - t0
    _TIEMPOS.append((titulo, dt))
    print("--- %s terminado en %.1f s" % (titulo, dt), flush=True)
    return resultado


# ---------------------------------------------------------------------------
# Etapas
# ---------------------------------------------------------------------------
def etapa_datos():
    from src.data import generar_datos
    from src.data.preparar_datos import preparar_consumo
    _etapa("1a. Generar datos de consumo (simulación)", generar_datos.main)
    _etapa("1b. Preparar datos (limpieza, variables, división temporal)", preparar_consumo)


def etapa_ml():
    from src.ml import clustering, entrenar_modelo
    _etapa("2a. Entrenar y comparar modelos de regresión", entrenar_modelo.main)
    _etapa("2b. Clustering K-Means de perfiles diarios", clustering.main)


def etapa_pnl():
    from src.data import generar_reportes
    from src.nlp import sentimiento
    from src.nlp.analizador import procesar_reportes
    _etapa("3a. Generar reportes de usuarios (texto)", generar_reportes.main)
    _etapa("3b. Entrenar clasificador de sentimiento y evaluar NER", sentimiento.entrenar)
    _etapa("3c. Procesar reportes (sentimiento + entidades)", procesar_reportes)


def registrar_artefactos(cadena: Blockchain) -> list:
    """Registra en la blockchain el dataset procesado, el modelo y las métricas."""
    def _reg():
        bloques = []
        for ruta, tipo in ((config.CSV_CONSUMO_PROC, "dataset"), (config.MODELO_CONSUMO, "modelo"),
                           (config.METRICAS_ML, "metricas")):
            if not Path(ruta).exists():
                print("  (omitido, no existe: %s)" % ruta)
                continue
            b = cadena.registrar_archivo(ruta, tipo)
            print("  bloque #%d  tipo=%-8s %s  sha256=%s..." % (b.indice, tipo, b.datos["nombre"], b.datos["hash_documento"][:16]))
            bloques.append(b)
        return bloques
    return _etapa("4. Registrar dataset, modelo y métricas en la blockchain", _reg)


def etapa_reporte(cadena: Blockchain):
    from src.reporte.generar_reporte import generar_reporte

    def _rep():
        r = generar_reporte(registrar=True, cadena=cadena)
        print("  Reporte: %s\n  SHA-256: %s\n  Bloque : #%s" % (r["nombre"], r["hash"], r["bloque"]))
        return r
    return _etapa("5. Generar reporte automático y registrar su hash", _rep)


def etapa_demo():
    from src.blockchain.demo_alteracion import ejecutar_demo
    ok = _etapa("6. Demostración de detección de alteraciones", ejecutar_demo)
    print("  Resultado de la demostración: %s" % ("TODAS LAS ALTERACIONES DETECTADAS" if ok else "REVISAR (falló alguna comprobación)"))
    return ok


def verificar(cadena: Blockchain | None = None) -> bool:
    """Valida la cadena y verifica cada reporte y artefacto contra ella. Devuelve True si todo está bien."""
    cadena = cadena or Blockchain()

    def _ver():
        todo_ok = True
        val = cadena.validar_cadena()
        print("  Cadena: %s (%d bloques)" % ("VÁLIDA" if val["valida"] else "INVÁLIDA", val["n_bloques"]))
        for e in val["errores"]:
            print("    - " + e)
        todo_ok &= val["valida"]
        archivos = sorted(config.DIR_REPORTES.glob("reporte_*.json")) + [
            p for p in (config.CSV_CONSUMO_PROC, config.MODELO_CONSUMO, config.METRICAS_ML) if Path(p).exists()]
        for ruta in archivos:
            r = cadena.verificar_archivo(ruta)
            print("  [%s] %s -> %s" % ("OK" if r["integro"] else "XX", Path(ruta).name, r["mensaje"]))
            todo_ok &= r["integro"]
        return todo_ok
    ok = _etapa("Verificación de integridad", _ver)
    print("  RESULTADO: %s" % ("TODO ÍNTEGRO" if ok else "SE ENCONTRARON PROBLEMAS"))
    return ok


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pipeline de SIAPE (por defecto ejecuta --todo).")
    for flag, ayuda in (("--todo", "flujo completo (por defecto)"), ("--datos", "generar y preparar datos"),
                        ("--ml", "entrenar modelo y clustering"), ("--pnl", "reportes de texto, sentimiento y NER"),
                        ("--reporte", "reporte automático + hash en blockchain"),
                        ("--blockchain-demo", "demostración de alteración"),
                        ("--registrar", "registrar dataset, modelo y métricas en la blockchain"),
                        ("--verificar", "validar la cadena y verificar reportes y artefactos")):
        ap.add_argument(flag, action="store_true", help=ayuda)
    a = ap.parse_args(argv)
    if not any(vars(a).values()):
        a.todo = True

    t0 = time.perf_counter()
    cadena = None
    if a.todo or a.datos:
        etapa_datos()
    if a.todo or a.ml:
        etapa_ml()
    if a.todo or a.pnl:
        etapa_pnl()
    if a.todo or a.registrar or a.reporte:
        cadena = Blockchain()
    if a.todo or a.registrar:
        registrar_artefactos(cadena)
    if a.todo or a.reporte:
        etapa_reporte(cadena)
    demo_ok = True
    if a.todo or a.blockchain_demo:
        demo_ok = etapa_demo()
    ver_ok = True
    if a.todo or a.verificar:
        ver_ok = verificar(Blockchain())

    print("\n" + "#" * 72)
    print("RESUMEN FINAL")
    print("#" * 72)
    for titulo, dt in _TIEMPOS:
        print("  %-62s %7.1f s" % (titulo, dt))
    print("  %-62s %7.1f s" % ("TOTAL", time.perf_counter() - t0))
    cad = Blockchain()
    print("  Bloques en la cadena: %d | cadena válida: %s" % (len(cad.cadena), cad.validar_cadena()["valida"]))
    print("  Para ver el dashboard: python app/app.py  ->  http://127.0.0.1:5000")
    return 0 if (demo_ok and ver_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
