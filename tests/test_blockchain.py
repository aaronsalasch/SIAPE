"""Pruebas rápidas de src/blockchain/blockchain.py (unittest, solo stdlib)."""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.blockchain.blockchain import (  # noqa: E402
    Blockchain, Bloque, sha256_archivo, sha256_bytes, sha256_json)

DIF = 2  # dificultad baja para que los tests sean rápidos


class PruebasBlockchain(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.ruta = self.dir / "cadena.json"
        self.bc = Blockchain(self.ruta, dificultad=DIF)

    def tearDown(self):
        self._tmp.cleanup()

    def _alterar_json(self, fn):
        d = json.loads(self.ruta.read_text(encoding="utf-8"))
        fn(d["bloques"])
        self.ruta.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        return Blockchain(self.ruta, dificultad=DIF)

    def test_hashes(self):
        self.assertEqual(sha256_bytes(b"abc"),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertEqual(sha256_json({"a": 1, "b": 2}), sha256_json({"b": 2, "a": 1}))
        f = self.dir / "x.txt"
        f.write_bytes(b"abc")
        self.assertEqual(sha256_archivo(f), sha256_bytes(b"abc"))

    def test_genesis_y_persistencia(self):
        self.assertTrue(self.ruta.exists())
        self.assertEqual(len(self.bc.cadena), 1)
        self.assertTrue(self.bc.cadena[0].hash.startswith("0" * DIF))
        self.assertFalse(self.ruta.with_name("cadena.json.tmp").exists())
        self.bc.registrar_contenido("hola", "reporte", "r.txt")
        otra = Blockchain(self.ruta, dificultad=DIF)
        self.assertEqual(otra.a_lista(), self.bc.a_lista())

    def test_bloque_y_pow(self):
        b = self.bc.registrar_contenido({"k": "ñ"}, "prediccion", "p")
        self.assertEqual(b.hash, b.calcular_hash())
        self.assertTrue(b.hash.startswith("0" * DIF))
        self.assertEqual(b.indice, 1)
        self.assertEqual(b.hash_anterior, self.bc.cadena[0].hash)
        self.assertEqual(set(b.datos), {"tipo", "nombre", "hash_documento", "metadatos"})

    def test_verificar_contenido(self):
        self.bc.registrar_contenido("texto áé", "reporte", "doc")
        self.assertTrue(self.bc.verificar_contenido("texto áé", "doc")["integro"])
        v = self.bc.verificar_contenido("texto áè", "doc")
        self.assertFalse(v["integro"])
        self.assertEqual(v["bloque"], 1)
        self.assertIsNotNone(v["hash_registrado"])
        self.bc.registrar_contenido({"b": 1, "a": 2}, "prediccion", "dic")
        self.assertTrue(self.bc.verificar_contenido({"a": 2, "b": 1}, "dic")["integro"])
        nr = self.bc.verificar_contenido("nada", "otro")
        self.assertFalse(nr["integro"])
        self.assertIsNone(nr["hash_registrado"])
        self.assertIsNone(nr["bloque"])

    def test_mismo_nombre_varias_versiones(self):
        self.bc.registrar_contenido("v1", "reporte", "doc")
        self.bc.registrar_contenido("v2", "reporte", "doc")
        self.assertEqual(len(self.bc.buscar_por_nombre("doc")), 2)
        self.assertTrue(self.bc.verificar_contenido("v2", "doc")["integro"])
        self.assertTrue(self.bc.verificar_contenido("v1", "doc")["integro"])  # coincide con un bloque
        v = self.bc.verificar_contenido("v3", "doc")
        self.assertFalse(v["integro"])
        self.assertEqual(v["bloque"], 2)  # se compara con el más reciente

    def test_archivo(self):
        f = self.dir / "r.txt"
        f.write_text("consumo 1250", encoding="utf-8")
        b = self.bc.registrar_archivo(f, "reporte", {"x": 1})
        self.assertEqual(b.datos["nombre"], "r.txt")
        self.assertEqual(b.datos["metadatos"]["x"], 1)
        self.assertTrue(self.bc.verificar_archivo(f)["integro"])
        f.write_text("consumo 1260", encoding="utf-8")
        self.assertFalse(self.bc.verificar_archivo(f)["integro"])

    def test_buscar_por_hash(self):
        b = self.bc.registrar_contenido("a", "reporte", "a.txt")
        self.assertIs(self.bc.buscar_por_hash(b.hash), b)
        self.assertIs(self.bc.buscar_por_hash(b.datos["hash_documento"]), b)
        self.assertIsNone(self.bc.buscar_por_hash("f" * 64))

    def test_cadena_valida(self):
        for i in range(3):
            self.bc.registrar_contenido("d%d" % i, "reporte", "d%d" % i)
        r = self.bc.validar_cadena()
        self.assertTrue(r["valida"])
        self.assertEqual((r["errores"], r["n_bloques"]), ([], 4))

    def test_detecta_dato_alterado(self):
        self.bc.registrar_contenido("a", "reporte", "a")
        self.bc.registrar_contenido("b", "reporte", "b")
        bc2 = self._alterar_json(lambda bl: bl[1]["datos"].__setitem__("nombre", "falso"))
        r = bc2.validar_cadena()
        self.assertFalse(r["valida"])
        self.assertTrue(any("Bloque 1" in e and "alterado" in e for e in r["errores"]))

    def test_detecta_enlace_roto(self):
        self.bc.registrar_contenido("a", "reporte", "a")
        self.bc.registrar_contenido("b", "reporte", "b")

        def forjar(bl):
            bl[1]["datos"]["nombre"] = "falso"
            b = Bloque.desde_dict(bl[1])
            b.minar(DIF)
            bl[1] = b.a_dict()
        r = self._alterar_json(forjar).validar_cadena()
        self.assertFalse(r["valida"])
        self.assertEqual(len(r["errores"]), 1)
        self.assertIn("Bloque 2: enlace roto", r["errores"][0])

    def test_detecta_pow_invalida(self):
        self.bc.registrar_contenido("a", "reporte", "a")

        def sin_pow(bl):
            b = Bloque.desde_dict(bl[1])
            n = 0
            while True:  # hash correcto pero sin ceros iniciales
                b.nonce = n
                b.hash = b.calcular_hash()
                if not b.hash.startswith("0" * DIF):
                    break
                n += 1
            bl[1] = b.a_dict()
        r = self._alterar_json(sin_pow).validar_cadena()
        self.assertFalse(r["valida"])
        self.assertTrue(any("prueba de trabajo" in e for e in r["errores"]))

    def test_detecta_indices(self):
        self.bc.registrar_contenido("a", "reporte", "a")

        def indice(bl):
            bl[1]["indice"] = 5
        r = self._alterar_json(indice).validar_cadena()
        self.assertTrue(any("índice no consecutivo" in e for e in r["errores"]))

    def test_archivo_danado_no_se_sobrescribe(self):
        self.ruta.write_text("{no es json", encoding="utf-8")
        with self.assertRaises(ValueError):
            Blockchain(self.ruta, dificultad=DIF)
        self.assertEqual(self.ruta.read_text(encoding="utf-8"), "{no es json")

    def test_hilos(self):
        def tarea(i):
            self.bc.registrar_contenido("t%d" % i, "reporte", "t%d" % i)
        hilos = [threading.Thread(target=tarea, args=(i,)) for i in range(8)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join()
        self.assertEqual(len(self.bc.cadena), 9)
        self.assertTrue(self.bc.validar_cadena()["valida"])
        self.assertTrue(Blockchain(self.ruta, dificultad=DIF).validar_cadena()["valida"])


if __name__ == "__main__":
    unittest.main()
