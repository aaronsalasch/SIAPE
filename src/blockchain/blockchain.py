"""Blockchain didáctica de SIAPE para garantizar la integridad de reportes y predicciones.

TIPO DE BLOCKCHAIN
------------------
Es una blockchain **privada y permisionada simulada**:
  * Privada/permisionada: solo la institución (el operador de SIAPE) escribe en la cadena;
    cualquiera con acceso al archivo JSON puede *verificar* pero no hay red pública.
  * Un solo nodo validador: no hay consenso distribuido; el propio sistema valida y agrega
    bloques (por eso se guarda en un archivo JSON local, ``config.CADENA_JSON``).
  * Prueba de trabajo (PoW) como ejemplo didáctico: cada bloque se "mina" hasta que su hash
    empieza con ``config.DIFICULTAD_POW`` ceros. En un sistema real de un solo nodo la PoW no
    aporta seguridad adicional; aquí se incluye para mostrar el concepto y encarecer
    la reescritura de la cadena.

ESTRUCTURA DEL BLOQUE
---------------------
    indice         número consecutivo (0 = génesis)
    marca_tiempo   fecha/hora ISO 8601 (UTC) de creación
    datos          {"tipo", "nombre", "hash_documento", "metadatos"}
    hash_anterior  hash del bloque previo (el génesis usa 64 ceros)
    nonce          número que se va probando hasta cumplir la PoW
    hash           SHA-256 del bloque

CÁLCULO DEL HASH
----------------
hash = SHA-256 del JSON *canónico* (``sort_keys=True, ensure_ascii=False``, sin espacios
sobrantes, codificado en UTF-8) de todos los campos del bloque EXCEPTO ``hash``.
Como el hash incluye ``hash_anterior``, cambiar cualquier bloque rompe el enlace con el siguiente:
eso es lo que vuelve la cadena "a prueba de alteraciones".

¿POR QUÉ SE GUARDA EL HASH Y NO EL DOCUMENTO?
---------------------------------------------
  * Un hash SHA-256 es una "huella digital" de 64 caracteres: cualquier cambio de un solo
    carácter en el documento produce un hash totalmente distinto.
  * No se puede reconstruir el documento a partir del hash, así que la cadena no expone
    información sensible (privacidad) y se mantiene pequeña aunque el documento pese megabytes.
  * Para auditar, se vuelve a calcular el hash del documento actual y se compara con el
    registrado: si coincide, no fue modificado desde que se registró.

Solo usa la biblioteca estándar. Se puede ejecutar una mini demo con
``python -m src.blockchain.blockchain``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Permite "import config" tanto al ejecutar con -m como al importar el módulo.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

HASH_CERO = "0" * 64          # hash_anterior del bloque génesis
TAM_BLOQUE_LECTURA = 1024 * 1024  # 1 MiB por lectura al hashear archivos


# ----------------------------------------------------------------------------
# Funciones de hash
# ----------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    """SHA-256 (hexadecimal, 64 caracteres) de una secuencia de bytes."""
    return hashlib.sha256(b).hexdigest()


def sha256_archivo(ruta) -> str:
    """SHA-256 de un archivo, leído por bloques de 1 MiB (no carga todo en memoria)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while True:
            trozo = f.read(TAM_BLOQUE_LECTURA)
            if not trozo:
                break
            h.update(trozo)
    return h.hexdigest()


def _json_canonico(obj) -> str:
    """JSON canónico: claves ordenadas, sin espacios sobrantes y sin escapar acentos."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_json(obj) -> str:
    """SHA-256 del JSON canónico de un objeto (dos dicts iguales dan el mismo hash
    sin importar el orden de sus claves)."""
    return sha256_bytes(_json_canonico(obj).encode("utf-8"))


def _hash_contenido(contenido) -> str:
    """Hash de un contenido en memoria: dict -> JSON canónico, str -> UTF-8, bytes -> directo."""
    if isinstance(contenido, dict):
        return sha256_json(contenido)
    if isinstance(contenido, str):
        return sha256_bytes(contenido.encode("utf-8"))
    if isinstance(contenido, (bytes, bytearray)):
        return sha256_bytes(bytes(contenido))
    raise TypeError("El contenido debe ser str, bytes o dict, no %s" % type(contenido).__name__)


def _cumple_pow(hash_bloque: str, dificultad: int) -> bool:
    """True si el hash empieza con `dificultad` ceros."""
    return hash_bloque.startswith("0" * dificultad)


# ----------------------------------------------------------------------------
# Bloque
# ----------------------------------------------------------------------------
class Bloque:
    """Un bloque de la cadena. Ver la documentación del módulo para su estructura."""

    def __init__(self, indice: int, marca_tiempo: str, datos: dict,
                 hash_anterior: str, nonce: int = 0, hash: str = ""):
        self.indice = indice
        self.marca_tiempo = marca_tiempo
        self.datos = datos
        self.hash_anterior = hash_anterior
        self.nonce = nonce
        self.hash = hash

    def _campos_sin_hash(self) -> dict:
        """Todos los campos que entran al hash (todos menos `hash`)."""
        return {
            "indice": self.indice,
            "marca_tiempo": self.marca_tiempo,
            "datos": self.datos,
            "hash_anterior": self.hash_anterior,
            "nonce": self.nonce,
        }

    def calcular_hash(self) -> str:
        """Recalcula el hash a partir del contenido actual del bloque."""
        return sha256_json(self._campos_sin_hash())

    def minar(self, dificultad: int) -> None:
        """Prueba de trabajo: incrementa `nonce` hasta que el hash tenga `dificultad` ceros."""
        self.nonce = 0
        self.hash = self.calcular_hash()
        while not _cumple_pow(self.hash, dificultad):
            self.nonce += 1
            self.hash = self.calcular_hash()

    def a_dict(self) -> dict:
        """Copia del bloque como diccionario (listo para JSON)."""
        d = copy.deepcopy(self._campos_sin_hash())
        d["hash"] = self.hash
        return d

    @classmethod
    def desde_dict(cls, d: dict) -> "Bloque":
        """Reconstruye un bloque desde un diccionario (p. ej. leído del JSON)."""
        return cls(d["indice"], d["marca_tiempo"], d["datos"], d["hash_anterior"],
                   d.get("nonce", 0), d.get("hash", ""))

    def __repr__(self) -> str:
        return "Bloque(indice=%d, tipo=%r, nombre=%r, hash=%s...)" % (
            self.indice, self.datos.get("tipo"), self.datos.get("nombre"), self.hash[:12])


# ----------------------------------------------------------------------------
# Blockchain
# ----------------------------------------------------------------------------
def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Blockchain:
    """Cadena de bloques persistida en un archivo JSON.

    Es segura entre hilos (la usa Flask): un candado reentrante protege lecturas y
    escrituras de la lista de bloques y del archivo. (No protege contra varios
    *procesos* escribiendo a la vez; con un solo nodo validador no hace falta.)
    Si el archivo no existe, se crea con el bloque génesis.
    """

    def __init__(self, ruta=config.CADENA_JSON, dificultad: int = config.DIFICULTAD_POW):
        self.ruta = Path(ruta)
        self.dificultad = dificultad
        self.cadena: list[Bloque] = []
        # RLock (reentrante) porque unos métodos públicos llaman a otros.
        self._candado = threading.RLock()
        self.cargar()

    # ---------------- persistencia ----------------
    def _crear_genesis(self) -> Bloque:
        datos = {
            "tipo": "genesis",
            "nombre": "Bloque génesis SIAPE",
            "hash_documento": HASH_CERO,
            "metadatos": {"descripcion": "Inicio de la cadena de trazabilidad energética SIAPE"},
        }
        b = Bloque(0, _ahora(), datos, HASH_CERO)
        b.minar(self.dificultad)
        return b

    def guardar(self) -> None:
        """Guarda la cadena de forma atómica: escribe a un .tmp y lo renombra, así un
        corte de energía no deja el JSON a medias."""
        with self._candado:
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            contenido = {"dificultad": self.dificultad,
                         "bloques": [b.a_dict() for b in self.cadena]}
            tmp = self.ruta.with_name(self.ruta.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(contenido, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.ruta)  # renombrado atómico

    def cargar(self) -> None:
        """Lee la cadena del JSON. Si no existe, crea el génesis y la guarda.
        NO valida la cadena (para eso está `validar_cadena`), así se pueden auditar
        archivos alterados."""
        with self._candado:
            if not self.ruta.exists():
                self.cadena = [self._crear_genesis()]
                self.guardar()
                return
            try:
                with open(self.ruta, "r", encoding="utf-8") as f:
                    contenido = json.load(f)
                lista = contenido["bloques"] if isinstance(contenido, dict) else contenido
                self.cadena = [Bloque.desde_dict(d) for d in lista]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                # No se sobrescribe: el archivo dañado puede ser evidencia.
                raise ValueError("El archivo de cadena %s está dañado: %s" % (self.ruta, e))

    # ---------------- registro ----------------
    def agregar_registro(self, tipo: str, nombre: str, hash_documento: str,
                         metadatos: dict | None = None) -> Bloque:
        """Mina y agrega un bloque con el hash de un documento (no el documento)."""
        with self._candado:
            ultimo = self.cadena[-1]
            datos = {
                "tipo": tipo,
                "nombre": nombre,
                "hash_documento": hash_documento,
                "metadatos": dict(metadatos) if metadatos else {},
            }
            bloque = Bloque(ultimo.indice + 1, _ahora(), datos, ultimo.hash)
            bloque.minar(self.dificultad)
            self.cadena.append(bloque)
            self.guardar()
            return bloque

    def registrar_archivo(self, ruta, tipo: str, metadatos: dict | None = None) -> Bloque:
        """Calcula el SHA-256 de un archivo y lo registra (nombre = nombre del archivo)."""
        ruta = Path(ruta)
        meta = {"tamano_bytes": ruta.stat().st_size}
        if metadatos:
            meta.update(metadatos)
        return self.agregar_registro(tipo, ruta.name, sha256_archivo(ruta), meta)

    def registrar_contenido(self, contenido, tipo: str, nombre: str,
                            metadatos: dict | None = None) -> Bloque:
        """Registra el hash de un contenido en memoria (str, bytes o dict)."""
        return self.agregar_registro(tipo, nombre, _hash_contenido(contenido), metadatos)

    # ---------------- validación ----------------
    def validar_cadena(self) -> dict:
        """Recalcula hashes, enlaces, PoW e índices de todos los bloques.

        Devuelve {"valida": bool, "errores": [mensajes en español], "n_bloques": int}.
        """
        with self._candado:
            errores = []
            for pos, b in enumerate(self.cadena):
                if b.indice != pos:
                    errores.append("Bloque %d: índice no consecutivo (dice %r, debería ser %d)."
                                   % (pos, b.indice, pos))
                recalculado = b.calcular_hash()
                if recalculado != b.hash:
                    errores.append("Bloque %d: el hash guardado no coincide con su contenido "
                                   "(guardado %s..., recalculado %s...): el bloque fue alterado."
                                   % (pos, b.hash[:12], recalculado[:12]))
                if not _cumple_pow(b.hash, self.dificultad):
                    errores.append("Bloque %d: prueba de trabajo inválida (el hash no empieza "
                                   "con %d ceros)." % (pos, self.dificultad))
                if pos == 0:
                    if b.hash_anterior != HASH_CERO:
                        errores.append("Bloque 0: el génesis debe tener hash_anterior de 64 ceros.")
                elif b.hash_anterior != self.cadena[pos - 1].hash:
                    errores.append("Bloque %d: enlace roto, hash_anterior no coincide con el "
                                   "hash del bloque %d." % (pos, pos - 1))
            if not self.cadena:
                errores.append("La cadena está vacía (no tiene bloque génesis).")
            return {"valida": not errores, "errores": errores, "n_bloques": len(self.cadena)}

    # ---------------- búsqueda ----------------
    def buscar_por_hash(self, h: str) -> Bloque | None:
        """Bloque más reciente cuyo `hash` (del bloque) o `hash_documento` sea `h`."""
        with self._candado:
            for b in reversed(self.cadena):
                if b.hash == h or b.datos.get("hash_documento") == h:
                    return b
            return None

    def buscar_por_nombre(self, nombre: str) -> list:
        """Todos los bloques con ese nombre de documento, del más antiguo al más reciente."""
        with self._candado:
            return [b for b in self.cadena if b.datos.get("nombre") == nombre]

    # ---------------- verificación ----------------
    def _verificar_hash(self, h: str, nombre: str | None) -> dict:
        """Lógica común: compara un hash calculado contra la cadena."""
        with self._candado:
            res = {"integro": False, "hash_calculado": h, "hash_registrado": None,
                   "bloque": None, "mensaje": ""}
            coincidencias = [b for b in self.cadena
                             if b.datos.get("hash_documento") == h and b.datos.get("tipo") != "genesis"]
            por_nombre = self.buscar_por_nombre(nombre) if nombre else []
            por_nombre = [b for b in por_nombre if b.datos.get("tipo") != "genesis"]

            if coincidencias:
                # Se prefiere un bloque con el mismo nombre; si no, el más reciente.
                elegido = next((b for b in reversed(coincidencias) if b in por_nombre),
                               coincidencias[-1])
                res.update(integro=True, hash_registrado=h, bloque=elegido.indice)
                msg = "ÍNTEGRO: el hash coincide con el registrado en el bloque %d." % elegido.indice
                if por_nombre and elegido is not por_nombre[-1]:
                    msg += (" Atención: corresponde a una versión anterior; el registro más "
                            "reciente de '%s' es el bloque %d." % (nombre, por_nombre[-1].indice))
                res["mensaje"] = msg
            elif por_nombre:
                ultimo = por_nombre[-1]
                res.update(hash_registrado=ultimo.datos.get("hash_documento"), bloque=ultimo.indice)
                res["mensaje"] = ("MODIFICADO: el hash actual no coincide con el registrado para "
                                  "'%s' en el bloque %d." % (nombre, ultimo.indice))
            else:
                res["mensaje"] = ("NO REGISTRADO: no existe en la cadena ningún bloque con ese hash"
                                  + (" ni con el nombre '%s'." % nombre if nombre else "."))
            return res

    def verificar_archivo(self, ruta, nombre: str | None = None) -> dict:
        """Verifica un archivo contra la cadena. `nombre` es opcional: por omisión se usa el
        nombre del archivo (útil cuando se sube una copia con nombre temporal)."""
        ruta = Path(ruta)
        return self._verificar_hash(sha256_archivo(ruta), nombre or ruta.name)

    def verificar_contenido(self, contenido, nombre: str | None = None) -> dict:
        """Verifica un contenido en memoria (dict -> JSON canónico, str -> UTF-8, bytes).

        Con `nombre`, si hay varios bloques con ese nombre se compara con el más reciente
        y se reporta "MODIFICADO" si no coincide. Si el hash coincide con cualquier bloque
        es "ÍNTEGRO".
        """
        return self._verificar_hash(_hash_contenido(contenido), nombre)

    # ---------------- utilidades ----------------
    def a_lista(self) -> list:
        """Cadena completa como lista de diccionarios (copias)."""
        with self._candado:
            return [b.a_dict() for b in self.cadena]


# ----------------------------------------------------------------------------
# Mini demo: python -m src.blockchain.blockchain
# ----------------------------------------------------------------------------
def _mini_demo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cadena = Blockchain(Path(tmp) / "cadena_demo.json")
        print("Cadena creada con génesis:", cadena.cadena[0])
        t0 = time.perf_counter()
        b = cadena.registrar_contenido("Reporte de ejemplo: consumo total 1,250 kWh",
                                       "reporte", "reporte_demo.txt", {"autor": "demo"})
        print("Bloque minado en %.2f s (nonce=%d):" % (time.perf_counter() - t0, b.nonce), b)
        print("Verificar texto original :", cadena.verificar_contenido(
            "Reporte de ejemplo: consumo total 1,250 kWh", "reporte_demo.txt")["mensaje"])
        print("Verificar texto alterado :", cadena.verificar_contenido(
            "Reporte de ejemplo: consumo total 1,260 kWh", "reporte_demo.txt")["mensaje"])
        print("Validación:", cadena.validar_cadena())


if __name__ == "__main__":
    _mini_demo()
