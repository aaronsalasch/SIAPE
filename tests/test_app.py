"""Pruebas de integración del dashboard Flask (todas las rutas).

Requiere haber ejecutado antes `python pipeline.py`. La cadena real se respalda antes de
las pruebas y se restaura al final, para que los registros de prueba no la ensucien.
Ejecutar desde la raíz:  python -m unittest tests.test_app -v
"""
import io
import json
import shutil
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "app"))

import config  # noqa: E402


class PruebasDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.respaldo = config.CADENA_JSON.with_suffix(".respaldo_tests.json")
        if config.CADENA_JSON.exists():
            shutil.copy2(config.CADENA_JSON, cls.respaldo)
        cls.reportes_previos = set(config.DIR_REPORTES.glob("reporte_*"))
        # La generación de un reporte reescribe esta evidencia: se respalda para restaurarla
        cls.evidencia = config.DIR_EVIDENCIAS / "reporte_registro.txt"
        cls.evidencia_previa = cls.evidencia.read_bytes() if cls.evidencia.exists() else None
        import app as modulo_app
        cls.modulo = modulo_app
        modulo_app.app.config["TESTING"] = True
        cls.cliente = modulo_app.app.test_client()

    @classmethod
    def tearDownClass(cls):
        # Restaurar cadena y borrar los reportes creados por las pruebas
        if cls.respaldo.exists():
            shutil.move(cls.respaldo, config.CADENA_JSON)
        for p in set(config.DIR_REPORTES.glob("reporte_*")) - cls.reportes_previos:
            p.unlink()
        if cls.evidencia_previa is not None:
            cls.evidencia.write_bytes(cls.evidencia_previa)

    def _get_ok(self, ruta):
        r = self.cliente.get(ruta)
        self.assertEqual(r.status_code, 200, f"{ruta} -> {r.status_code}")
        return r

    def test_paginas_get(self):
        for ruta in ["/", "/datos", "/modelo", "/clusters", "/prediccion", "/pnl", "/blockchain",
                     "/blockchain?todos=1", "/reportes"]:
            with self.subTest(ruta=ruta):
                r = self._get_ok(ruta)
                self.assertIn(b"SIAPE", r.data)

    def test_figura(self):
        r = self._get_ok("/figuras/comparacion_modelos.png")
        self.assertEqual(r.mimetype, "image/png")

    def test_prediccion_y_pronostico(self):
        datos = {"id_instalacion": "COM-001", "fecha_hora": "2026-07-02T13:00", "temperatura_c": "31",
                 "humedad_pct": "70", "accion": "predecir"}
        r = self.cliente.post("/prediccion", data=datos)
        self.assertEqual(r.status_code, 200)
        self.assertIn("kWh".encode(), r.data)
        datos["accion"] = "pronostico"
        r = self.cliente.post("/prediccion", data=datos)
        self.assertEqual(r.status_code, 200)

    def test_prediccion_invalida(self):
        r = self.cliente.post("/prediccion", data={"id_instalacion": "RES-001", "fecha_hora": "2026-07-02T13:00",
                                                    "temperatura_c": "200", "humedad_pct": "70"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("No se pudo calcular".encode(), r.data)

    def test_registrar_prediccion_y_verificar(self):
        r = self.cliente.post("/api/registrar_prediccion", json={
            "id_instalacion": "RES-002", "fecha_hora": "2026-07-02 20:00", "temperatura_c": 30, "humedad_pct": 75})
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertTrue(d["ok"])
        # Verificar el mismo contenido (íntegro) y una versión alterada (modificado)
        nombre = d["bloque"]["datos"]["nombre"]
        r = self.cliente.post("/blockchain/verificar", data={"texto": d["contenido_texto"], "nombre": nombre},
                              headers={"Accept": "application/json"})
        self.assertTrue(r.get_json()["resultado"]["integro"])
        alterado = dict(d["contenido"], prediccion_kwh=d["contenido"]["prediccion_kwh"] + 1)
        r = self.cliente.post("/blockchain/verificar", data={"texto": json.dumps(alterado), "nombre": nombre},
                              headers={"Accept": "application/json"})
        self.assertFalse(r.get_json()["resultado"]["integro"])

    def test_pnl(self):
        texto = "El minisplit de RES-001 en Veracruz falla desde ayer a las 18:30, pésimo servicio de CFE."
        r = self.cliente.post("/pnl", data={"texto": texto})
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"ent-disp", r.data)   # entidad DISPOSITIVO resaltada

    def test_validar_cadena(self):
        r = self.cliente.post("/blockchain/validar")
        self.assertTrue(r.get_json()["valida"])

    def test_verificar_archivo_integro_y_alterado(self):
        reporte = sorted(config.DIR_REPORTES.glob("reporte_*.json"))[0]
        contenido = reporte.read_bytes()
        r = self.cliente.post("/blockchain/verificar", headers={"Accept": "application/json"},
                              data={"archivo": (io.BytesIO(contenido), reporte.name)},
                              content_type="multipart/form-data")
        self.assertTrue(r.get_json()["resultado"]["integro"])
        r = self.cliente.post("/blockchain/verificar", headers={"Accept": "application/json"},
                              data={"archivo": (io.BytesIO(contenido + b" "), reporte.name)},
                              content_type="multipart/form-data")
        self.assertFalse(r.get_json()["resultado"]["integro"])

    def test_demo_alteracion(self):
        r = self.cliente.post("/blockchain/demo")
        self.assertTrue(r.get_json()["todo_correcto"])

    def test_generar_y_ver_reporte(self):
        r = self.cliente.post("/reportes/generar")
        self.assertEqual(r.status_code, 302)
        r = self.cliente.get(r.headers["Location"])
        self.assertEqual(r.status_code, 200)
        nombre = sorted(config.DIR_REPORTES.glob("reporte_*.json"))[-1].name
        self._get_ok("/reportes/descargar/" + nombre)


if __name__ == "__main__":
    unittest.main(verbosity=2)
