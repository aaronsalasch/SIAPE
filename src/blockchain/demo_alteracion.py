"""Demostración reproducible de detección de alteraciones en la blockchain de SIAPE.

Se trabaja SIEMPRE en un directorio temporal: no se crea ni modifica la cadena real
(``config.CADENA_JSON``). Genera ``outputs/evidencias/blockchain_prueba_alteracion.txt``.

Ejecutar:  python -m src.blockchain.demo_alteracion
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.blockchain.blockchain import Blockchain, Bloque, sha256_archivo, sha256_json  # noqa: E402

SALIDA = config.DIR_EVIDENCIAS / "blockchain_prueba_alteracion.txt"


class _Registro:
    """Acumula las líneas de la evidencia y cuenta pruebas correctas/fallidas."""

    def __init__(self):
        self.lineas = []
        self.pruebas = []

    def linea(self, txt=""):
        self.lineas.append(txt)

    def paso(self, n, titulo):
        self.linea("")
        self.linea("PASO %d: %s" % (n, titulo))
        self.linea("-" * 70)

    def comparar(self, descripcion, esperado, obtenido):
        ok = esperado == obtenido
        self.pruebas.append(ok)
        self.linea("  Esperado : %s" % (esperado,))
        self.linea("  Obtenido : %s" % (obtenido,))
        self.linea("  Resultado: %s  (%s)" % ("CORRECTO" if ok else "FALLÓ", descripcion))


def _leer(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def _escribir(ruta: Path, contenido: dict) -> None:
    ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=2), encoding="utf-8")


def ejecutar_demo(ruta_salida=SALIDA) -> bool:
    """Corre la demostración, escribe la evidencia y devuelve True si todo salió como se esperaba."""
    r = _Registro()
    real = config.CADENA_JSON
    hash_real_antes = sha256_archivo(real) if real.exists() else None
    existia_real = real.exists()

    r.linea("=" * 70)
    r.linea("SIAPE - PRUEBA DE DETECCIÓN DE ALTERACIONES EN LA BLOCKCHAIN")
    r.linea("Fecha de ejecución: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    r.linea("Dificultad PoW: %d ceros iniciales | Cadena de prueba en directorio temporal" % config.DIFICULTAD_POW)
    r.linea("=" * 70)

    with tempfile.TemporaryDirectory(prefix="siape_demo_") as tmp:
        tmp = Path(tmp)
        ruta_cadena = tmp / "cadena_demo.json"

        # PASO 1
        r.paso(1, "Crear una cadena nueva (temporal) con su bloque génesis")
        cadena = Blockchain(ruta_cadena)
        g = cadena.cadena[0]
        r.linea("  Bloque 0 (génesis) hash: %s (nonce=%d)" % (g.hash, g.nonce))
        r.comparar("la cadena inicia con 1 bloque", 1, len(cadena.cadena))

        # PASO 2
        r.paso(2, "Registrar un documento de ejemplo (reporte) en la cadena")
        doc = tmp / "reporte_demo.txt"
        doc.write_text("REPORTE DE CONSUMO ENERGÉTICO\nInstalación COM-001: 1250 kWh en marzo.\n"
                       "Predicción próximas 24 h: 52.3 kWh.\n", encoding="utf-8")
        b1 = cadena.registrar_archivo(doc, "reporte", {"origen": "demo"})
        r.linea("  Documento: %s" % doc.name)
        r.linea("  SHA-256 del documento: %s" % b1.datos["hash_documento"])
        r.linea("  Bloque %d: hash=%s" % (b1.indice, b1.hash))
        r.linea("             hash_anterior=%s" % b1.hash_anterior)
        r.comparar("el bloque enlaza con el génesis", g.hash, b1.hash_anterior)

        # Dos bloques más para poder probar enlaces entre bloques
        pred = {"id_instalacion": "COM-001", "prediccion_kwh": 52.3, "modelo": "RandomForest"}
        b2 = cadena.registrar_contenido(pred, "prediccion", "prediccion_COM-001")
        b3 = cadena.registrar_contenido("métricas: R2=0.97", "metricas", "metricas_modelo")
        r.linea("  También se registraron el bloque %d (predicción, dict) y el %d (métricas, texto)."
                % (b2.indice, b3.indice))

        # PASO 3
        r.paso(3, "Verificar el documento SIN modificar (debe ser íntegro)")
        v = cadena.verificar_archivo(doc)
        r.linea("  Hash calculado : %s" % v["hash_calculado"])
        r.linea("  Hash registrado: %s" % v["hash_registrado"])
        r.linea("  Mensaje: %s" % v["mensaje"])
        r.comparar("documento intacto", True, v["integro"])

        # PASO 4
        r.paso(4, "Modificar UN solo carácter del documento (1250 -> 1260) y verificar")
        original = doc.read_text(encoding="utf-8")
        modificado = original.replace("1250", "1260", 1)
        diferencias = sum(1 for a, c in zip(original, modificado) if a != c)
        doc.write_text(modificado, encoding="utf-8")
        r.linea("  Caracteres distintos: %d" % diferencias)
        v = cadena.verificar_archivo(doc)
        r.linea("  Hash calculado : %s" % v["hash_calculado"])
        r.linea("  Hash registrado: %s" % v["hash_registrado"])
        r.linea("  Mensaje: %s" % v["mensaje"])
        r.comparar("un carácter cambia todo el hash", False, v["integro"])

        # PASO 5
        r.paso(5, "Verificar contenido en memoria (dict de predicción): íntegro y alterado")
        v_ok = cadena.verificar_contenido(pred, "prediccion_COM-001")
        pred_mod = dict(pred, prediccion_kwh=52.4)
        v_mal = cadena.verificar_contenido(pred_mod, "prediccion_COM-001")
        r.linea("  Original : %s" % v_ok["mensaje"])
        r.linea("  Alterado (52.3 -> 52.4): %s" % v_mal["mensaje"])
        r.comparar("(original, alterado)", (True, False), (v_ok["integro"], v_mal["integro"]))

        # PASO 6
        r.paso(6, "Validar la cadena sin alteraciones")
        val = cadena.validar_cadena()
        r.linea("  Bloques: %d | errores: %s" % (val["n_bloques"], val["errores"] or "ninguno"))
        r.comparar("cadena válida", True, val["valida"])

        # PASO 7
        r.paso(7, "Alterar el bloque 1 dentro de una COPIA de la cadena (cambiar el hash_documento)")
        copia1 = tmp / "copia_alterada.json"
        shutil.copy(ruta_cadena, copia1)
        datos = _leer(copia1)
        falso = sha256_json("documento falsificado")
        datos["bloques"][1]["datos"]["hash_documento"] = falso
        _escribir(copia1, datos)
        r.linea("  hash_documento original del bloque 1: %s" % b1.datos["hash_documento"])
        r.linea("  hash_documento alterado             : %s" % falso)
        val = Blockchain(copia1).validar_cadena()
        for e in val["errores"]:
            r.linea("  Detectado: %s" % e)
        r.comparar("cadena inválida con error en el bloque 1",
                   (False, True),
                   (val["valida"], any(e.startswith("Bloque 1:") and "alterado" in e for e in val["errores"])))

        # PASO 8
        r.paso(8, "Enlace roto: el atacante recalcula y re-mina el bloque 1 para ocultar el cambio")
        copia2 = tmp / "copia_enlace_roto.json"
        shutil.copy(ruta_cadena, copia2)
        datos = _leer(copia2)
        datos["bloques"][1]["datos"]["hash_documento"] = falso
        bloque_forjado = Bloque.desde_dict(datos["bloques"][1])
        bloque_forjado.minar(config.DIFICULTAD_POW)   # hash y PoW de nuevo válidos
        datos["bloques"][1] = bloque_forjado.a_dict()
        _escribir(copia2, datos)
        r.linea("  Hash original del bloque 1 : %s" % b1.hash)
        r.linea("  Hash forjado del bloque 1  : %s" % bloque_forjado.hash)
        val = Blockchain(copia2).validar_cadena()
        for e in val["errores"]:
            r.linea("  Detectado: %s" % e)
        r.comparar("el bloque 1 ya es consistente, pero el enlace del bloque 2 se rompe",
                   (False, True),
                   (val["valida"], any(e.startswith("Bloque 2:") and "enlace roto" in e for e in val["errores"])))

        # PASO 9
        r.paso(9, "Comprobar que la cadena original de la demo sigue válida y la real no se tocó")
        val = Blockchain(ruta_cadena).validar_cadena()
        r.comparar("cadena de la demo intacta", True, val["valida"])
        hash_real_despues = sha256_archivo(real) if real.exists() else None
        r.linea("  Cadena real (%s) existía antes: %s" % (real, existia_real))
        r.comparar("cadena real sin cambios (hash del archivo)", hash_real_antes, hash_real_despues)

    r.linea("")
    r.linea("=" * 70)
    ok = all(r.pruebas)
    r.linea("RESUMEN: %d/%d comprobaciones correctas -> %s"
            % (sum(r.pruebas), len(r.pruebas), "TODAS LAS ALTERACIONES FUERON DETECTADAS" if ok else "HAY FALLOS"))
    r.linea("=" * 70)

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text("\n".join(r.lineas) + "\n", encoding="utf-8")
    print("\n".join(r.lineas))
    print("\nEvidencia guardada en:", ruta_salida)
    return ok


if __name__ == "__main__":
    sys.exit(0 if ejecutar_demo() else 1)
