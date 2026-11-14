"""Generador de reportes de usuarios (texto en español) para el módulo de PNL de SIAPE.

Ejecución:  ``python -m src.data.generar_reportes``  (desde la raíz del proyecto)

Salida: ``config.CSV_REPORTES_RAW`` con ~4,000 reportes simulados y las columnas
``id_reporte, id_instalacion, fecha_hora, canal, texto, sentimiento_etiqueta,
entidades_esperadas``.  ``entidades_esperadas`` es una lista JSON de objetos
``{"texto": ..., "tipo": ...}`` con las entidades que el generador insertó; sirve para
evaluar el NER (precisión/recall/F1) en ``src/nlp/entidades.py``.

PROCEDIMIENTO DE GENERACIÓN (simulación por plantillas combinatorias)
---------------------------------------------------------------------
1. Para cada reporte se elige una instalación de ``config.INSTALACIONES`` (así el
   ``id_instalacion`` y la ciudad son coherentes), un canal (app / correo / llamada /
   portal) y una fecha-hora dentro del periodo ``FECHA_INICIO`` + ``DIAS_SIMULADOS``.
2. Se decide el sentimiento "verdadero" de la plantilla (positivo ~32 %, negativo ~38 %,
   neutral ~30 % entre los reportes "puros").
3. El texto se arma con *cláusulas* tomadas de bancos por sentimiento.  Cada cláusula es
   una plantilla con:
     * alternativas ``[a|b|c]`` (sinónimos y variantes: se elige una al azar), y
     * ranuras ``{disp} {lugar} {org} {fecha} {hora} {cod} {folio} {monto} {kwh}
       {persona}`` que se rellenan con entidades.  Las ranuras admiten artículo:
       ``{el:disp}`` -> "el/la/los/las", ``{del:..}``, ``{un:..}``, ``{al:..}``.
   Las ranuras son coherentes con la instalación (p. ej. una planta industrial habla de
   transformadores y compresores; el código mencionado casi siempre es el suyo; los
   montos y kWh tienen magnitudes acordes al tipo de instalación).
4. Se añaden 0-3 oraciones de detalle (código, folio, fecha, hora, lectura en kWh, persona
   que atendió, monto), una apertura según el canal y, a veces, un cierre.
5. Cada entidad insertada se guarda con su texto y su tipo (``entidades_esperadas``).

RUIDO Y CASOS DIFÍCILES (para que el clasificador no sea trivial)
-----------------------------------------------------------------
* ~10 % de los reportes son MIXTOS/AMBIGUOS: combinan una cláusula positiva con una
  negativa (o neutral con alguna de ellas) con conectores como "pero", "aunque", "sin
  embargo".  La etiqueta sigue la cláusula posterior al conector en ~80 % de los casos y
  la primera en ~20 %, como haría un anotador humano dudoso.
* ~3 % de ruido de etiqueta: se cambia la etiqueta por otra al azar (error de anotación).
* Negaciones ("no funciona", "sin fallas", "no tengo ninguna queja"), "gracias" en los
  tres sentimientos, y emojis que ~15 % de las veces no coinciden con el sentimiento.
* Errores ortográficos leves (letras omitidas/intercambiadas/dobles, acentos faltantes),
  textos en minúsculas, palabras en MAYÚSCULAS de énfasis y signos repetidos.  El ruido
  solo se aplica al texto "plano"; las entidades se conservan íntegras, salvo casos
  intencionales: ~4 % de las ciudades sin acento/minúsculas, ~1.5 % con error de dedo y ~3 % de dispositivos con
  alias coloquial ("refri", "clima") que el NER por diccionario no conoce (genera falsos
  negativos realistas).
* Semilla fija ``config.SEMILLA``: la generación es reproducible.
"""
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

import pandas as pd  # noqa: E402

N_REPORTES = 4000
P_MIXTO = 0.10          # proporción de casos ambiguos/mezclados
P_RUIDO_ETIQUETA = 0.03  # proporción de etiquetas cambiadas a propósito

CANALES = ["app", "correo", "llamada", "portal"]
PESOS_CANAL = [0.35, 0.20, 0.20, 0.25]
SENTIMIENTOS = ["positivo", "negativo", "neutral"]
PESOS_SENT = [0.32, 0.38, 0.30]

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

# Dispositivos más probables según el tipo de instalación
DISP_POR_TIPO = {
    "residencial": ["minisplit", "refrigerador", "lavadora", "calentador", "aire acondicionado",
                    "ventilador", "luminarias", "medidor", "horno"],
    "comercial": ["aire acondicionado", "refrigerador", "luminarias", "elevador", "servidor",
                  "medidor", "inversor", "minisplit", "horno"],
    "industrial": ["transformador", "compresor", "motor", "bomba de agua", "horno", "inversor",
                   "medidor", "ventilador"],
    "educativa": ["aire acondicionado", "servidor", "luminarias", "ventilador", "elevador",
                  "bomba de agua", "medidor", "minisplit"],
}
FEMENINOS = {"bomba de agua", "lavadora", "luminarias"}
PLURALES = {"luminarias"}
ALIAS = {"refrigerador": "refri", "aire acondicionado": "clima"}

# Magnitudes por tipo: (rango monto en pesos, rango kWh)
MAGNITUDES = {
    "residencial": ((300, 2500), (80, 900)),
    "comercial": ((3000, 25000), (800, 8000)),
    "industrial": ((40000, 300000), (20000, 180000)),
    "educativa": ((8000, 90000), (4000, 40000)),
}

NOMBRES = ["Carlos", "María", "José", "Ana", "Luis", "Martha", "Jorge", "Laura", "Miguel", "Patricia",
           "Fernando", "Rosa", "Ricardo", "Elena", "Alberto", "Sofía", "Héctor", "Gabriela"]
APELLIDOS = ["Hernández", "García", "Martínez", "López", "Pérez", "Ruiz", "Gómez", "Sánchez", "Ramírez",
             "Flores", "Torres", "Domínguez", "Castillo", "Mendoza", "Rosales", "Cruz", "Vázquez", "Ortiz"]
TITULOS = ["Sr.", "Sra.", "Ing.", "Lic."]

# ---------------------------------------------------------------------------
# BANCOS DE CLÁUSULAS (alternativas entre [ ] y ranuras entre { })
# ---------------------------------------------------------------------------
NEGATIVAS = [
    "{el:disp} [falla|se descompone|se apaga|se sobrecalienta|se traba] [constantemente|cada rato|todos los días|otra vez|sin parar]",
    "[pésimo|muy mal|terrible|deplorable|malísimo] [servicio|trato|la atención] de {org}",
    "el recibo llegó [carísimo|con un cobro excesivo|con un monto injusto] de {monto}",
    "[hubo|tuvimos|sigue habiendo] un apagón [en {lugar}|de varias horas|sin aviso previo|que dañó todo]",
    "{el:disp} hace un ruido [extraño|muy fuerte|molesto|horrible] y huele a quemado",
    "nadie de {org} [ha venido|contesta|resuelve nada|da la cara] y ya van [tres|cuatro|varios|muchos] días",
    "[la baja de voltaje|las variaciones de voltaje|un pico de voltaje] [dañaron|quemaron|averiaron|echaron a perder] {el:disp}",
    "el consumo [se disparó|subió muchísimo|aumentó sin justificación|está por las nubes] hasta {kwh}",
    "estoy [muy molesto|harto|decepcionado|inconforme|indignado] con [la lentitud|el mal servicio|la falta de respuesta|el abandono]",
    "el técnico [nunca llegó|llegó tarde y no resolvió nada|dejó el problema peor|ni siquiera revisó nada]",
    "no [funciona|sirve|enciende|arranca|responde] {el:disp} [desde el lunes|desde hace una semana|después de la tormenta|desde la última visita]",
    "[es inaceptable|es una vergüenza|es un abuso|es injusto] que [cobren|nos cobren] {monto} [sin explicación|por un consumo que no corresponde|y no aclaren nada]",
    "la fuga de corriente en {lugar} es [peligrosa|un riesgo serio|muy grave], [urgente|favor de atender ya|exijo atención inmediata]",
    "{el:disp} [está dañado|está descompuesto|quedó inservible|sigue averiado] y [nadie responde|no hay refacciones|no lo quieren arreglar]",
    "los cortes de luz en {lugar} son [constantes|frecuentes|interminables] y [afectan nuestra operación|dañan los equipos|nos causan pérdidas]",
    "[me cobraron|nos cobraron|nos facturaron] [dos veces|de más|doble] {monto} y no [corrigen|aclaran|devuelven] el error",
    "la tarifa [injusta|abusiva|impagable] de {org} [nos tiene al límite|es insostenible|no tiene justificación]",
    "[se quemó|explotó|tronó|se botó] {el:disp} y [perdimos equipo|hubo riesgo de incendio|quedamos sin servicio]",
    "[sigo esperando|llevo semanas esperando|seguimos esperando] respuesta de {org} [y nada|sin solución|y no hacen caso]",
    "{el:disp} consume [demasiado|muchísima energía|de más] y [nadie lo revisa|no sabemos por qué|nos sale carísimo]",
    "[mala experiencia|qué desastre|fue un fiasco], [no lo recomiendo|no vuelvo a llamar|todo salió mal]",
    "no quedé [conforme|satisfecho|contento] con [la reparación|la respuesta|el mantenimiento] de {el:disp}",
    "{el:disp} [marca de más|marca lecturas absurdas|está descalibrado] y {org} [no quiere revisarlo|ignora la queja]",
    "[sigue la falla|persiste el problema|el problema continúa] en {el:disp} [a pesar de la visita|aunque ya lo reportamos|y es peor cada vez]",
    "[cero|nula|pésima] [atención|respuesta|comunicación] de {org}, [qué mal|qué lástima|muy decepcionante]",
    "[otra vez|de nuevo] se fue la luz en {lugar} y [no avisaron|nadie explica nada|estamos hartos]",
]

POSITIVAS = [
    "{el:disp} [funciona|trabaja|opera] [perfecto|de maravilla|excelente|muy bien|de lujo] [desde que lo revisaron|después del mantenimiento|desde la reparación|otra vez]",
    "[excelente|muy buen|buen|magnífico|impecable] [servicio|trabajo|soporte|trato] de {org}",
    "[muy satisfecho|contento|agradecido|encantado|feliz] con [la atención recibida|la rapidez|el resultado|la solución]",
    "el técnico [llegó puntual|fue muy amable|resolvió todo rápido|solucionó el problema el mismo día|explicó todo con paciencia]",
    "[logramos ahorrar|redujimos el consumo a|bajó el consumo a] {kwh} [este mes|gracias al ajuste|después de la revisión]",
    "el recibo [salió más barato|bajó bastante|llegó correcto|vino mucho mejor] [esta vez|este periodo]: solo {monto}",
    "[gracias|muchas gracias] a {persona} por [su amabilidad|su ayuda|su rapidez|resolverlo]",
    "{el:disp} nuevo [ahorra mucha energía|es muy eficiente|funciona sin ruido|es una maravilla]",
    "[todo funciona bien|todo quedó resuelto|todo está en orden|sin ningún problema] en {lugar} [desde la visita|ahora|gracias al apoyo]",
    "[recomiendo|felicito|aplaudo] a {org} por [su rapidez|la solución|el seguimiento|su profesionalismo]",
    "el servicio eléctrico en {lugar} [está estable|ha mejorado mucho|es confiable|funciona de maravilla]",
    "[quedé|quedamos] [muy conformes|satisfechos|muy contentos|encantados] con [la reparación|el mantenimiento|la instalación] de {del:disp}",
    "la app [es muy útil|está muy clara|nos ayuda a controlar el consumo|es fácil de usar]",
    "[sin fallas|sin cortes|sin apagones|sin quejas] [desde el mantenimiento|en todo el mes|en las últimas semanas|desde la visita]",
    "[buen|excelente|gran] [ahorro|resultado|avance]: {kwh} [menos que el mes pasado|menos de consumo|de reducción]",
    "{el:disp} quedó [como nuevo|funcionando al cien|impecable|perfecto]",
    "no [tengo|hay] [ninguna queja|ningún problema|nada que reportar] con {el:disp}, [todo bien|excelente|funciona de maravilla]",
    "[nos resolvieron|me resolvieron] el problema [rápido|en menos de una hora|sin costo] y [con mucha amabilidad|sin complicaciones]",
    "[ahora sí|por fin] [bajó|disminuyó|se normalizó] el recibo a {monto}, [qué alivio|muy agradecido|excelente noticia]",
    "[gran] [labor|apoyo|atención] de {persona}, [muy profesional|nos ayudó muchísimo|lo recomiendo]",
]

NEUTRALES = [
    "[solicito|quisiera|requiero|necesito] información sobre [mi consumo|la tarifa|el contrato|el proceso de aclaración|los horarios de facturación]",
    "[reporto|registro|envío] la lectura {del:disp}: {kwh}",
    "[quisiera programar|favor de agendar|solicito|requiero] una visita técnica para revisar {el:disp}",
    "[consulto|pregunto] cuál es el horario de atención de {org} en {lugar}",
    "[se realizó|se hizo|se efectuó] la revisión de rutina {del:disp} en {lugar}",
    "[adjunto|envío] [la fotografía|el comprobante|el recibo|la evidencia] [correspondiente|del periodo|de la visita]",
    "[necesito|requiero|solicito] un [duplicado|comprobante|estado de cuenta|desglose] del recibo",
    "[solicito|requiero] [cotización|información] para [instalar|sustituir|cambiar] {un:disp}",
    "el consumo registrado fue de {kwh} [en el periodo|durante el mes|según el último corte]",
    "el cargo del recibo es de {monto}, [lo comento para su conocimiento|para dar seguimiento|solo informo]",
    "[se solicita|favor de|quedo al pendiente de] [confirmar|verificar|informar] el estatus del folio {folio}",
    "[el personal|la cuadrilla] de {org} [realizó|hará|programó] [una inspección|trabajos de mantenimiento|una revisión] en {lugar}",
    "[comento que|informo que|aviso que] [se instaló|se movió|se reubicó|se cambió] {el:disp} en el área de {lugar}",
    "[tengo una duda|consulta|pregunta] sobre [la facturación|el horario de la tarifa|los kilowatts-hora facturados|el cambio de titular]",
    "[solicito|requiero] [constancia|copia|confirmación] de [pago|la lectura|la visita] [del mes|del periodo|del trámite]",
    "[se reporta|queda registrado] [el mantenimiento|el cambio|el ajuste] {del:disp} [para su archivo|según lo acordado|como parte del programa]",
    "[quisiera saber|me gustaría saber|necesito saber] si {org} [tiene programas de ahorro|ofrece auditorías energéticas|atiende en fin de semana]",
    "{el:disp} [se encuentra|está] [en operación normal|pendiente de revisión|en el área de mantenimiento] [según el registro|por ahora|hasta nuevo aviso]",
]

# Detalles (oraciones cortas con entidades) comunes a cualquier sentimiento
DETALLES = [
    "[Mi código de instalación es|Instalación|Código:|Hablo de la instalación] {cod}",
    "[Folio de seguimiento:|Folio:|Mi folio es|Número de reporte:] {folio}",
    "[Ocurrió|Sucedió|Lo noté|Pasó|Fue] {el:fecha}",
    "[Ocurrió|Sucedió|Fue|Se presentó] {el:fecha} {hora}",
    "[Lectura actual:|Última lectura:|El contador marca] {kwh}",
    "[Me atendió|Atendió|Me ayudó|Mi contacto fue] {persona}",
    "[Zona:|Ubicación:|Es en|Estamos en] {lugar}",
    "[Les escribo desde|Reporto desde|Escribo desde] {lugar}",
    "[Fecha del evento:|Desde] {el:fecha}",
    "[Hora aproximada:|Hora del evento:|Registrado] {hora}",
    "[Referencia:|Para su seguimiento,] {cod} con el folio {folio}",
]
DETALLES_NEG = [
    "[Me cobraron|Pagué|Me llegó un cargo de] {monto} [de más|y no es justo|sin explicación]",
    "[Llevo esperando desde|El problema empezó] {el:fecha}",
    "[Les reporto otra vez el folio|Ya levanté el reporte] {folio}",
]
DETALLES_POS = [
    "[Pagué solo|El recibo fue de|Me cobraron] {monto}",
    "[Les dejo mi folio|Mi folio es] {folio} [por si lo necesitan|para felicitar al equipo]",
]

APERTURAS = {
    "app": ["", "", "", "Reporte desde la app: ", "Hola, ", "Buenas, ", "Alerta: "],
    "correo": ["Estimados: ", "Buen día, ", "Buenas tardes, ", "Estimado personal: ", "A quien corresponda: ", "Saludos cordiales. "],
    "llamada": ["Llamada recibida: ", "Usuario reporta por teléfono: ", "Cliente indica que ", "Se registra llamada: ", "Reporte telefónico: "],
    "portal": ["", "Comentario: ", "Queja/sugerencia: ", "Hola, ", "Asunto: ", "Reporte: "],
}
CIERRES = {
    "positivo": ["¡Gracias!", "Muchas gracias.", "Sigan así.", "Excelente trabajo.", "Gracias por todo.", "Quedo agradecido."],
    "negativo": ["Espero una respuesta pronta.", "Exijo una solución.", "Gracias.", "Ya no sé a quién acudir.", "Urge atención.", "Favor de responder."],
    "neutral": ["Quedo atento.", "Gracias.", "Saludos.", "Quedo al pendiente.", "Muchas gracias de antemano.", "Saludos cordiales."],
}
EMOJIS = {
    "positivo": ["😊", "👍", "🙏", "😀", "✅", "👏"],
    "negativo": ["😡", "😠", "👎", "😤", "😞", "⚠️"],
    "neutral": ["📎", "📌", "🙂", "📄"],
}
CONECTORES = [", pero ", ", aunque ", ", sin embargo ", "; no obstante, ", ", y aun así ", ", eso sí, "]
NEGACIONES = {"nunca", "jamás", "nada", "ningún", "ninguna", "tampoco"}

_RE_ALT = re.compile(r"\[([^\[\]]+)\]")
_RE_RANURA = re.compile(r"\{(?:(el|del|un|al):)?(\w+)\}")
_ACENTOS = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")


# ---------------------------------------------------------------------------
# Contexto de generación de un reporte: produce el valor de cada ranura
# ---------------------------------------------------------------------------
class Contexto:
    """Guarda la instalación, la fecha y los valores ya elegidos para las ranuras.

    Una misma ranura (p. ej. ``{cod}``) tiene el mismo valor dentro de un reporte; para
    tener dos valores distintos se usan nombres con número (``{disp2}``, ``{fecha2}``).
    """

    def __init__(self, rng, inst, fecha_reporte):
        self.rng = rng
        self.inst = inst
        self.fecha = fecha_reporte
        self.cache = {}

    def valor(self, nombre):
        if nombre not in self.cache:
            base = re.sub(r"\d+$", "", nombre)
            self.cache[nombre] = getattr(self, "_" + base)(nombre)
        return self.cache[nombre]

    # --- generadores por tipo de entidad ---
    def _disp(self, nombre):
        r = self.rng
        opciones = DISP_POR_TIPO[self.inst["tipo"]]
        usados = {v["canon"] for v in self.cache.values() if v.get("canon")}
        libres = [d for d in opciones if d not in usados] or opciones
        canon = r.choice(libres)
        texto = canon
        if canon in ALIAS and r.random() < 0.03:
            texto = ALIAS[canon]                    # alias coloquial: el NER no lo conoce
        elif canon == "minisplit" and r.random() < 0.08:
            texto = "mini split"
        return {"texto": texto, "tipo": "DISPOSITIVO", "canon": canon,
                "fem": canon in FEMENINOS, "plural": canon in PLURALES}

    def _lugar(self, nombre):
        r = self.rng
        ciudad = self.inst["ciudad"] if r.random() < 0.85 else r.choice(config.CIUDADES)
        texto = ciudad
        if r.random() < 0.04:                       # variante informal
            texto = r.choice([ciudad.lower(), ciudad.translate(_ACENTOS)])
        elif r.random() < 0.015:                    # error de dedo: el diccionario no lo reconoce
            i = r.randrange(2, len(ciudad) - 1)
            texto = ciudad[:i] + ciudad[i + 1:]
        return {"texto": texto, "tipo": "LUGAR"}

    def _org(self, nombre):
        r = self.rng
        if self.inst["tipo"] == "educativa" and r.random() < 0.10:
            texto = r.choice(["TecNM", "ITV"])
        else:
            texto = r.choices(["CFE", "Comisión Federal de Electricidad", "PROFECO", "CENACE"],
                              weights=[0.55, 0.25, 0.14, 0.06])[0]
        return {"texto": texto, "tipo": "ORGANIZACION"}

    def _fecha(self, nombre):
        r = self.rng
        d = self.fecha - timedelta(days=r.randint(0, 20))
        d = max(d, datetime.fromisoformat(config.FECHA_INICIO))
        modo = r.choices(["texto", "num", "rel"], weights=[0.45, 0.30, 0.25])[0]
        if modo == "texto":
            texto = f"{d.day} de {MESES[d.month - 1]}"
            if r.random() < 0.15:
                texto += f" de {d.year}"
            return {"texto": texto, "tipo": "FECHA", "relativa": False}
        if modo == "num":
            return {"texto": f"{d.day:02d}/{d.month:02d}/{d.year}", "tipo": "FECHA", "relativa": False}
        return {"texto": r.choice(["ayer", "hoy", "anteayer"]), "tipo": "FECHA", "relativa": True}

    def _hora(self, nombre):
        r = self.rng
        h, m = r.randint(0, 23), r.choice([0, 5, 10, 15, 20, 30, 40, 45, 50])
        if r.random() < 0.70:
            return {"texto": f"a las {h}:{m:02d}", "tipo": "HORA"}
        return {"texto": f"{h:02d}:{m:02d} hrs", "tipo": "HORA"}

    def _cod(self, nombre):
        r = self.rng
        cod = self.inst["id"] if r.random() < 0.88 else r.choice(config.INSTALACIONES)["id"]
        return {"texto": cod, "tipo": "CODIGO_INSTALACION"}

    def _folio(self, nombre):
        return {"texto": f"FOL-{self.rng.randint(100000, 999999)}", "tipo": "FOLIO"}

    def _monto(self, nombre):
        r = self.rng
        lo, hi = MAGNITUDES[self.inst["tipo"]][0]
        v = r.randint(lo, hi)
        v = round(v, -1) if v > 100 else v
        texto = f"${v:,}"
        if r.random() < 0.2:
            texto += f".{r.choice(['50', '20', '75', '90'])}"
        return {"texto": texto, "tipo": "MONTO"}

    def _kwh(self, nombre):
        r = self.rng
        lo, hi = MAGNITUDES[self.inst["tipo"]][1]
        v = r.randint(lo, hi)
        unidad = "kWh" if r.random() < 0.9 else "kwh"
        return {"texto": f"{v:,} {unidad}", "tipo": "ENERGIA"}

    def _persona(self, nombre):
        r = self.rng
        titulo = r.choice(TITULOS)
        if r.random() < 0.5:
            nom = r.choice(APELLIDOS)
        else:
            nom = f"{r.choice(NOMBRES)} {r.choice(APELLIDOS)}"
        return {"texto": f"{titulo} {nom}", "tipo": "PERSONA"}


def _articulo(art, valor):
    """Devuelve el artículo (con espacio) para la ranura; cadena vacía si no aplica."""
    if valor["tipo"] == "FECHA":
        return "el " if (art in ("el",) and not valor.get("relativa")) else ""
    fem, plural = valor.get("fem", False), valor.get("plural", False)
    tabla = {
        "el": ("el", "la", "los", "las"),
        "del": ("del", "de la", "de los", "de las"),
        "un": ("un", "una", "unos", "unas"),
        "al": ("al", "a la", "a los", "a las"),
    }[art]
    idx = (2 if plural else 0) + (1 if fem else 0)
    return tabla[idx] + " "


def llenar(plantilla, ctx):
    """Convierte una plantilla en una lista de partes ``(texto, tipo_entidad | None)``."""
    t = _RE_ALT.sub(lambda m: ctx.rng.choice(m.group(1).split("|")), plantilla)
    t = re.sub(r"\s{2,}", " ", t).strip()
    partes, pos = [], 0
    for m in _RE_RANURA.finditer(t):
        if m.start() > pos:
            partes.append((t[pos:m.start()], None))
        art, nombre = m.groups()
        v = ctx.valor(nombre)
        if art:
            a = _articulo(art, v)
            if a:
                partes.append((a, None))
        partes.append((v["texto"], v["tipo"]))
        pos = m.end()
    if pos < len(t):
        partes.append((t[pos:], None))
    return partes


def _capitalizar(partes):
    """Pone mayúscula inicial a la primera parte si es texto plano."""
    if partes and partes[0][1] is None and partes[0][0]:
        s = partes[0][0]
        partes = [(s[0].upper() + s[1:], None)] + partes[1:]
    return partes


# ---------------------------------------------------------------------------
# Ruido sobre el texto plano (las entidades no se tocan)
# ---------------------------------------------------------------------------
def _typo(palabra, rng):
    op = rng.randrange(4)
    if op == 0 and len(palabra) > 5:                         # omitir una letra
        i = rng.randrange(1, len(palabra) - 1)
        return palabra[:i] + palabra[i + 1:]
    if op == 1 and len(palabra) > 5:                         # intercambiar dos letras
        i = rng.randrange(1, len(palabra) - 2)
        return palabra[:i] + palabra[i + 1] + palabra[i] + palabra[i + 2:]
    if op == 2:                                              # duplicar una letra
        i = rng.randrange(1, len(palabra) - 1)
        return palabra[:i] + palabra[i] + palabra[i:]
    return palabra.translate(_ACENTOS)                       # quitar acentos


def _ruido_plano(texto, rng, sin_acentos, minusculas, p_typo=0.035):
    if minusculas:
        texto = texto.lower()
    if sin_acentos:
        texto = texto.translate(_ACENTOS)

    def cambiar(m):
        p = m.group(0)
        if p.lower() in NEGACIONES or rng.random() > p_typo:
            return p
        return _typo(p, rng)

    return re.sub(r"[A-Za-záéíóúñÁÉÍÓÚÑ]{5,}", cambiar, texto)


def aplicar_ruido(partes, rng, etiqueta, canal):
    """Aplica ruido ortográfico/de formato a las partes planas y arma el texto final."""
    sin_acentos = rng.random() < 0.06
    minusculas = rng.random() < 0.08
    enfasis = etiqueta == "negativo" and rng.random() < 0.04
    idx_enfasis = None
    if enfasis:
        planas = [i for i, (s, t) in enumerate(partes) if t is None and len(s) > 6]
        idx_enfasis = rng.choice(planas) if planas else None
    salida = []
    for i, (s, t) in enumerate(partes):
        if t is None:
            s = _ruido_plano(s, rng, sin_acentos, minusculas)
            if i == idx_enfasis:
                s = s.upper()
        salida.append(s)
    texto = "".join(salida).strip()
    # signos repetidos y emojis ocasionales (a veces no coinciden con el sentimiento)
    if etiqueta == "negativo" and rng.random() < 0.15:
        texto = texto.rstrip(".") + rng.choice(["!!", "!!!", "!"])
    elif etiqueta == "positivo" and rng.random() < 0.20:
        texto = texto.rstrip(".") + rng.choice(["!", "!!", "!"])
    p_emoji = 0.25 if canal == "app" else 0.04
    if rng.random() < p_emoji:
        e_sent = etiqueta if rng.random() < 0.85 else rng.choice(SENTIMIENTOS)
        texto += " " + rng.choice(EMOJIS[e_sent])
    return texto


# ---------------------------------------------------------------------------
# Construcción de un reporte
# ---------------------------------------------------------------------------
def _entidades(partes):
    return [{"texto": s, "tipo": t} for s, t in partes if t is not None]


def _oracion(partes):
    """Capitaliza y cierra con punto una oración (lista de partes)."""
    partes = _capitalizar(partes)
    ult_s, ult_t = partes[-1]
    if ult_t is None and ult_s.rstrip().endswith((".", "!", "?")):
        return partes
    return partes + [(".", None)]


def construir_reporte(rng, fecha_base, dias):
    inst = rng.choice(config.INSTALACIONES)
    canal = rng.choices(CANALES, weights=PESOS_CANAL)[0]
    dia = rng.randrange(dias)
    hora = rng.choices(range(24), weights=[1] * 6 + [3] * 3 + [5] * 9 + [4] * 4 + [2] * 2)[0]
    fecha = fecha_base + timedelta(days=dia, hours=hora, minutes=rng.randrange(60))
    ctx = Contexto(rng, inst, fecha)
    sentencias = []
    mixto = rng.random() < P_MIXTO

    apertura = rng.choice(APERTURAS[canal])
    if apertura:
        sentencias_pre = [(apertura, None)]
    else:
        sentencias_pre = []

    if mixto:
        s1, s2 = rng.choice([("positivo", "negativo"), ("negativo", "positivo"), ("neutral", "negativo"),
                             ("neutral", "positivo"), ("positivo", "negativo"), ("negativo", "positivo")])
        bancos = {"positivo": POSITIVAS, "negativo": NEGATIVAS, "neutral": NEUTRALES}
        c1 = llenar(rng.choice(bancos[s1]), ctx)
        c2 = llenar(rng.choice(bancos[s2]), ctx)
        if rng.random() < 0.7:
            cuerpo = _capitalizar(c1) + [(rng.choice(CONECTORES), None)] + c2 + [(".", None)]
        else:
            cuerpo = [("Aunque ", None)] + c1 + [(", ", None)] + c2 + [(".", None)]
        etiqueta = s2 if rng.random() < 0.8 else s1
        sentencias.append(cuerpo)
    else:
        etiqueta = rng.choices(SENTIMIENTOS, weights=PESOS_SENT)[0]
        banco = {"positivo": POSITIVAS, "negativo": NEGATIVAS, "neutral": NEUTRALES}[etiqueta]
        k = 1 if rng.random() < 0.55 else 2
        for c in rng.sample(banco, k):
            sentencias.append(_oracion(llenar(c, ctx)))

    # oraciones de detalle con entidades
    n_det = rng.choices([0, 1, 2, 3], weights=[0.2, 0.4, 0.28, 0.12])[0]
    pool = list(DETALLES)
    if etiqueta == "negativo" and rng.random() < 0.4:
        pool += DETALLES_NEG * 3
    if etiqueta == "positivo" and rng.random() < 0.4:
        pool += DETALLES_POS * 3
    elegidos = []
    for _ in range(n_det):
        d = rng.choice(pool)
        if d not in elegidos:
            elegidos.append(d)
    for d in elegidos:
        sentencias.append(_oracion(llenar(d, ctx)))

    # cierre (opcional)
    if rng.random() < 0.40:
        sentencias.append([(rng.choice(CIERRES[etiqueta if rng.random() < 0.85 else "neutral"]), None)])
    # firma en correos
    firma = []
    if canal == "correo" and rng.random() < 0.35:
        firma = [("Atentamente, ", None)] + llenar("{persona}", Contexto(rng, inst, fecha))

    partes = list(sentencias_pre)
    for i, s in enumerate(sentencias):
        if i > 0:
            partes.append((" ", None))
        # tras "Cliente indica que " la primera cláusula va en minúscula
        if i == 0 and apertura.endswith("que ") and s[0][1] is None:
            s = [(s[0][0][0].lower() + s[0][0][1:], None)] + s[1:]
        partes.extend(s)
    if firma:
        partes.append((" ", None))
        partes.extend(firma)

    # etiqueta con ruido de anotación
    ruido = False
    if rng.random() < P_RUIDO_ETIQUETA:
        etiqueta = rng.choice([s for s in SENTIMIENTOS if s != etiqueta])
        ruido = True

    texto = aplicar_ruido(partes, rng, etiqueta, canal)
    return {
        "id_instalacion": inst["id"],
        "fecha_hora": fecha.strftime("%Y-%m-%d %H:%M:%S"),
        "canal": canal,
        "texto": texto,
        "sentimiento_etiqueta": etiqueta,
        "entidades_esperadas": json.dumps(_entidades(partes), ensure_ascii=False),
        "_mixto": mixto,
        "_ruido": ruido,
    }


def generar_reportes(n=N_REPORTES, guardar=True):
    """Genera ``n`` reportes simulados y (opcionalmente) escribe el CSV raw y la evidencia."""
    rng = random.Random(config.SEMILLA)
    fecha_base = datetime.fromisoformat(config.FECHA_INICIO)
    filas = [construir_reporte(rng, fecha_base, config.DIAS_SIMULADOS) for _ in range(n)]
    df = pd.DataFrame(filas).sort_values("fecha_hora", kind="stable").reset_index(drop=True)
    df.insert(0, "id_reporte", [f"REP-{i + 1:05d}" for i in range(len(df))])
    if guardar:
        config.DIR_DATA_RAW.mkdir(parents=True, exist_ok=True)
        cols = ["id_reporte", "id_instalacion", "fecha_hora", "canal", "texto",
                "sentimiento_etiqueta", "entidades_esperadas"]
        df[cols].to_csv(config.CSV_REPORTES_RAW, index=False, encoding="utf-8")
        _escribir_evidencia(df)
    return df


def _escribir_evidencia(df):
    ents = Counter()
    for js in df["entidades_esperadas"]:
        for e in json.loads(js):
            ents[e["tipo"]] += 1
    lineas = [
        "EVIDENCIA - GENERACIÓN DE REPORTES DE USUARIOS (SIAPE)",
        "=" * 60,
        f"Reportes generados: {len(df)}",
        f"Periodo: {df['fecha_hora'].min()} a {df['fecha_hora'].max()}",
        f"Textos únicos: {df['texto'].nunique()} ({df['texto'].nunique() / len(df):.1%})",
        f"Casos mixtos/ambiguos: {int(df['_mixto'].sum())} ({df['_mixto'].mean():.1%})",
        f"Etiquetas con ruido deliberado: {int(df['_ruido'].sum())} ({df['_ruido'].mean():.1%})",
        "",
        "Distribución de sentimiento (etiqueta):",
        df["sentimiento_etiqueta"].value_counts().to_string(),
        "",
        "Distribución por canal:",
        df["canal"].value_counts().to_string(),
        "",
        "Reportes por instalación:",
        df["id_instalacion"].value_counts().sort_index().to_string(),
        "",
        "Entidades esperadas insertadas por tipo:",
        "\n".join(f"  {t}: {n}" for t, n in ents.most_common()),
        "",
        "Ejemplos:",
    ]
    for _, f in df.sample(8, random_state=config.SEMILLA).iterrows():
        lineas.append(f"[{f['id_reporte']}|{f['id_instalacion']}|{f['canal']}|{f['sentimiento_etiqueta']}] {f['texto']}")
    (config.DIR_EVIDENCIAS / "generacion_reportes.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main():
    df = generar_reportes()
    print(f"Reportes generados: {len(df)} -> {config.CSV_REPORTES_RAW}")
    print(df["sentimiento_etiqueta"].value_counts().to_string())
    for t in df["texto"].head(8):
        print("-", t)


if __name__ == "__main__":
    main()
