"""Preprocesamiento de texto en español, implementado desde cero (sin nltk ni spaCy).

Etapas (cada una se puede explicar por separado):
  1. NORMALIZACIÓN: minúsculas, URLs y correos fuera, emojis convertidos en marcadores
     ("emojipositivo"/"emojinegativo"), folios/códigos/montos/horas/fechas numéricas
     enmascarados (son entidades, no aportan al sentimiento), espacios limpios.
  2. TOKENIZACIÓN con expresión regular (palabras con acentos, ñ y ü).
  3. STOPWORDS: lista propia (> 200 palabras) que CONSERVA las negaciones
     ("no", "nunca", "sin", "ni", "tampoco", "nada", "ningún"...) porque cambian el sentido.
  4. STEMMING: stemmer ligero por sufijos, inspirado en Snowball español (simplificado).
  5. LEMATIZACIÓN: diccionario de verbos frecuentes conjugados + reglas simples de plural.

Uso principal: ``preprocesar(texto) -> dict`` con las claves
``normalizado, tokens, sin_stopwords, stems, lemas``.
"""
import re
import unicodedata

# ---------------------------------------------------------------------------
# 1. Normalización
# ---------------------------------------------------------------------------
_RE_URL = re.compile(r"(?:https?://|www\.)\S+")
_RE_CORREO = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_RE_FOLIO = re.compile(r"\bfol-\d{4,8}\b", re.I)
_RE_CODIGO = re.compile(r"\b(?:res|com|ind|edu)-\d{3}\b", re.I)
_RE_MONTO = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\$\s?\d+(?:\.\d{1,2})?")
_RE_ENERGIA = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?kwh\b|\b\d+(?:\.\d+)?\s?kwh\b", re.I)
_RE_HORA = re.compile(r"\b\d{1,2}:\d{2}(?:\s?(?:hrs?|h|horas))?\b", re.I)
_RE_FECHA_NUM = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_RE_NUMERO = re.compile(r"\d+")
# Emojis y símbolos: se clasifican en positivos/negativos; el resto se descarta
_EMOJI_POS = set("😊😀😃😄😁😍🥰👍👏🙏✅💪🎉😎🙂❤")
_EMOJI_NEG = set("😡😠🤬👎😤😞😩😢😭⚠🔥💥❌😒😫")
_RE_SIMBOLOS = re.compile("[\U0001F300-\U0001FAFF☀-➿️‍]")

_RE_TOKEN = re.compile(r"[a-záéíóúüñ]+")

_TABLA_SIN_ACENTO = str.maketrans("áéíóúü", "aeiouu")


def quitar_acentos(texto):
    """Quita acentos y diéresis conservando la ñ."""
    return texto.translate(_TABLA_SIN_ACENTO)


def _marcar_emoji(m):
    c = m.group(0)
    if c in _EMOJI_POS:
        return " emojipositivo "
    if c in _EMOJI_NEG:
        return " emojinegativo "
    return " "


def normalizar(texto, sin_acentos=False):
    """Limpia el texto: minúsculas, sin URLs/correos, emojis -> marcadores, entidades numéricas fuera."""
    if not isinstance(texto, str):
        texto = "" if texto is None else str(texto)
    t = unicodedata.normalize("NFC", texto).lower()
    t = _RE_URL.sub(" ", t)
    t = _RE_CORREO.sub(" ", t)
    t = _RE_FOLIO.sub(" ", t)
    t = _RE_CODIGO.sub(" ", t)
    t = _RE_MONTO.sub(" ", t)
    t = _RE_ENERGIA.sub(" ", t)
    t = _RE_HORA.sub(" ", t)
    t = _RE_FECHA_NUM.sub(" ", t)
    t = _RE_SIMBOLOS.sub(_marcar_emoji, t)
    t = _RE_NUMERO.sub(" ", t)
    if sin_acentos:
        t = quitar_acentos(t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------
# 2. Tokenización
# ---------------------------------------------------------------------------
def tokenizar(texto_normalizado):
    """Divide el texto normalizado en palabras (solo letras, con acentos/ñ)."""
    return _RE_TOKEN.findall(texto_normalizado)


# ---------------------------------------------------------------------------
# 3. Stopwords (sin negaciones)
# ---------------------------------------------------------------------------
_STOP_TEXTO = """
a al algo algunas algunos ante antes aquel aquella aquellas aquello aquellos aqui asi aun aunque
cada casi como con contra cual cuales cuando cuanto de del desde donde dos durante e el ella ellas
ello ellos en entre era erais eran eras eres es esa esas ese eso esos esta estaba estaban estado
estais estamos estan estar estas este esto estos estoy fue fueron fui fuimos ha habia habian haber
habra han has hasta hay haya he hemos hube la las le les lo los me mi mia mias mio mios mis mucho
muchos nos nosotras nosotros nuestra nuestras nuestro nuestros o os otra otras otro otros para
pero poco por porque que quien quienes se sea sean ser sera si sido siendo sobre sois somos son soy
su sus suya suyas suyo suyos tambien tanto te tendra tenemos tener tengo ti tiene tienen todo todos
tu tus tuya tuyo un una unas uno unos vosotros vuestra vuestro y ya yo
ahi ahora alla alli allá alguien alguna alguno bajo bastante cierto cierta con conmigo contigo
cuyo cuya demas dentro despues detras dicho hacia hacer hace hacen hizo luego mas menos mismo misma
mismos mismas mientras muy nuestro otro parte pues quiza segun siempre sino solo sólo sus tal tales
tan tras usted ustedes vez veces vamos van vaya voy vi vio
enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre
lunes martes miercoles jueves viernes sabado domingo
asimismo ademas entonces etc dia dias tarde tardes noche noches
hola saludos estimados estimado cordiales atentamente
fui fuiste fueron seria serian estaria estarian estuvo estuvieron estuve esten este
tuvo tuvieron tuve habria hubo hubiera hubiese tenia tenian tendria tienes
podemos puede pueden puedo podria podrian debe deben debo
nuestras quisiera quiero queremos quieren
mio tuyo suyo propio propia propios propias
dicha dichos dichas dicho eso esas estos aquellos
unas unos cuyos
sera seran fueron fuese fuera fueran seamos
eres somos
ayer hoy anteayer antier folio codigo
"""
# Palabras que NUNCA deben eliminarse (cambian el sentimiento)
NEGACIONES = {"no", "nunca", "jamás", "jamas", "sin", "ni", "tampoco", "nada", "ningún", "ninguno",
              "ninguna", "nadie", "ningun"}
# Se conservan palabras con carga emocional: "gracias", "buen/buena", "pero", intensificadores
STOPWORDS = (set(_STOP_TEXTO.split()) | {quitar_acentos(w) for w in _STOP_TEXTO.split()}) - NEGACIONES
STOPWORDS.discard("pero")   # el contraste ("pero", "aunque") ayuda a detectar casos mixtos
STOPWORDS.discard("muy")    # los intensificadores sí aportan al sentimiento
STOPWORDS.discard("mas")
STOPWORDS.discard("más")
STOPWORDS.discard("menos")
STOPWORDS.discard("poco")
STOPWORDS.discard("mucho")
STOPWORDS.discard("sólo")
STOPWORDS.discard("solo")   # "solo $850" aparece como señal positiva


def quitar_stopwords(tokens):
    """Elimina stopwords conservando las negaciones."""
    return [t for t in tokens if t not in STOPWORDS]


# ---------------------------------------------------------------------------
# 4. Stemmer ligero (sufijos tipo Snowball simplificado)
# ---------------------------------------------------------------------------
# Sufijos derivativos (se prueban del más largo al más corto, una sola vez)
_SUF_DERIV = sorted([
    "amientos", "imientos", "amiento", "imiento", "aciones", "uciones", "adoras", "adores",
    "ancias", "encias", "logías", "idades", "amente", "mente", "ación", "ución", "adora", "ador",
    "ancia", "encia", "idad", "ables", "ibles", "able", "ible", "istas", "ista", "osos", "osas",
    "oso", "osa", "ivos", "ivas", "ivo", "iva", "ismo", "ico", "ica", "icos", "icas",
], key=len, reverse=True)
# Sufijos verbales
_SUF_VERBO = sorted([
    "aremos", "eremos", "iremos", "ábamos", "abamos", "iéramos", "ando", "iendo", "yendo",
    "aron", "ieron", "arán", "erán", "irán", "aban", "ían", "aste", "iste", "amos", "emos", "imos",
    "aba", "ías", "ado", "ido", "ada", "ida", "ados", "idos", "adas", "idas", "ará", "erá", "irá",
    "ar", "er", "ir", "an", "en", "ió", "ía", "ó", "é", "as", "es", "áis", "éis",
], key=len, reverse=True)
# Terminaciones residuales (género / número)
_SUF_RESID = ["os", "as", "es", "a", "o", "e", "s"]

_ARTICULOS_FUERA = {"no", "sin", "ni"}   # negaciones: no se stemizan


def stem(palabra):
    """Devuelve la raíz aproximada de una palabra en español.

    Pasos: (1) quitar acentos, (2) sufijo derivativo, (3) sufijo verbal,
    (4) terminación de género/número. Siempre se deja una raíz de >= 3 letras.
    Ejemplos: funcionamiento -> funcion, fallas -> fall, descompuesto -> descompuest.
    """
    p = quitar_acentos(palabra.lower())
    if palabra in NEGACIONES or p in _ARTICULOS_FUERA or len(p) <= 3:
        return p
    for suf in _SUF_DERIV:
        s = quitar_acentos(suf)
        if p.endswith(s) and len(p) - len(s) >= 3:
            p = p[: -len(s)]
            break
    else:
        for suf in _SUF_VERBO:
            s = quitar_acentos(suf)
            if p.endswith(s) and len(p) - len(s) >= 3:
                p = p[: -len(s)]
                break
    for suf in _SUF_RESID:
        if p.endswith(suf) and len(p) - len(suf) >= 3:
            p = p[: -len(suf)]
            break
    return p


# ---------------------------------------------------------------------------
# 5. Lematizador por diccionario + reglas de plural
# ---------------------------------------------------------------------------
def _conjugar_regular(verbo):
    """Genera formas regulares (presente, pretérito, imperfecto, gerundio, participio)."""
    raiz, term = verbo[:-2], verbo[-2:]
    if term == "ar":
        formas = ["o", "as", "a", "amos", "an", "é", "aste", "ó", "aron", "aba", "aban", "ando", "ado",
                  "ada", "ados", "adas", "e", "en"]
    elif term == "er":
        formas = ["o", "es", "e", "emos", "en", "í", "iste", "ió", "ieron", "ía", "ían", "iendo", "ido",
                  "ida", "idos", "idas", "a", "an"]
    else:
        formas = ["o", "es", "e", "imos", "en", "í", "iste", "ió", "ieron", "ía", "ían", "iendo", "ido",
                  "ida", "idos", "idas", "a", "an"]
    return {raiz + f for f in formas}


_VERBOS_REGULARES = [
    "funcionar", "fallar", "apagar", "encender", "cobrar", "revisar", "resolver", "atender", "reportar",
    "solicitar", "consumir", "reparar", "instalar", "cambiar", "medir", "pagar", "llegar", "esperar",
    "contestar", "responder", "arreglar", "ahorrar", "reducir", "aumentar", "bajar", "subir", "dañar",
    "quemar", "sobrecalentar", "descomponer", "informar", "consultar", "enviar", "adjuntar", "agendar",
    "programar", "verificar", "confirmar", "recomendar", "agradecer", "quejar", "reclamar", "marcar",
    "facturar", "aclarar", "corregir", "devolver", "tardar", "llamar", "escribir", "registrar",
    "mejorar", "empeorar", "normalizar", "visitar", "inspeccionar", "mantener", "realizar", "efectuar",
    "disparar", "trabajar", "operar", "sirvir", "ayudar", "explicar", "felicitar", "aplaudir", "controlar",
    "usar", "necesitar", "requerir", "persistir", "continuar", "seguir", "arrancar", "trabar", "explotar",
]
_VERBOS_REGULARES = [v for v in _VERBOS_REGULARES if v != "sirvir"]

# Formas irregulares frecuentes
_IRREGULARES = {
    "es": "ser", "son": "ser", "fue": "ser", "fueron": "ser", "era": "ser", "sido": "ser", "soy": "ser",
    "está": "estar", "están": "estar", "estaba": "estar", "estuvo": "estar", "estoy": "estar",
    "tiene": "tener", "tienen": "tener", "tengo": "tener", "tuvo": "tener", "tuvimos": "tener",
    "tenía": "tener", "hace": "hacer", "hacen": "hacer", "hizo": "hacer", "hecho": "hacer",
    "hay": "haber", "hubo": "haber", "había": "haber", "ha": "haber", "han": "haber",
    "va": "ir", "van": "ir", "voy": "ir", "fui": "ir", "sigue": "seguir", "siguen": "seguir",
    "sirve": "servir", "sirven": "servir", "sirvió": "servir", "huele": "oler", "pide": "pedir",
    "pidió": "pedir", "quedó": "quedar", "quedé": "quedar", "quedamos": "quedar", "puede": "poder",
    "pueden": "poder", "pudo": "poder", "quiero": "querer", "quiere": "querer", "dijo": "decir",
    "dice": "decir", "dio": "dar", "da": "dar", "dan": "dar", "vino": "venir", "viene": "venir",
    "vienen": "venir", "puso": "poner", "pone": "poner", "ponen": "poner", "salió": "salir",
    "sale": "salir", "salen": "salir", "perdimos": "perder", "pierde": "perder", "pierden": "perder",
    "enciende": "encender", "enciendo": "encender", "prende": "prender", "tronó": "tronar",
    "bajó": "bajar", "subió": "subir", "cobraron": "cobrar", "cobró": "cobrar", "nos": "nosotros",
    "mejor": "bueno", "peor": "malo", "mejores": "bueno", "peores": "malo", "buena": "bueno",
    "buenos": "bueno", "buenas": "bueno", "mala": "malo", "malos": "malo", "malas": "malo",
    "carísimo": "caro", "carísima": "caro", "malísimo": "malo", "luces": "luz", "voltajes": "voltaje",
    "tardó": "tardar", "resolvieron": "resolver", "resolvió": "resolver", "solucionó": "solucionar",
    "solucionaron": "solucionar", "recibí": "recibir", "recibimos": "recibir", "recibo": "recibo",
}
_PLURALES_IRREG = {"luces": "luz", "veces": "vez", "lápices": "lápiz", "meses": "mes", "paises": "país",
                   "kilowatts": "kilowatt", "horas": "hora", "días": "día", "dias": "día"}
_NO_PLURAL = {"lunes", "martes", "miércoles", "jueves", "viernes", "crisis", "mes", "más", "menos",
              "gas", "dios", "paris", "análisis", "tras", "antes", "después", "gracias", "pues", "es",
              "ademas", "además", "mas", "tres", "dos", "nos", "los", "las", "les", "ves", "vez",
              "tras", "ahorros"}   # ahorros se resuelve con regla; el resto no cambia

_LEMAS = {}
for _v in _VERBOS_REGULARES:
    _LEMAS[_v] = _v
    for _f in _conjugar_regular(_v):
        _LEMAS.setdefault(_f, _v)
_LEMAS.update(_IRREGULARES)
_LEMAS.update(_PLURALES_IRREG)
_LEMAS.pop("a", None)


def lematizar(palabra):
    """Devuelve el lema: diccionario (verbos/irregulares) y, si no está, regla de plural."""
    p = palabra.lower()
    if p in _LEMAS:
        return _LEMAS[p]
    if p in NEGACIONES or p in _NO_PLURAL or len(p) <= 3:
        return p
    if p.endswith("mos"):
        return p                    # verbos en 1.ª persona del plural (logramos)
    if p.endswith(("bles", "gles")):
        return p[:-1]               # interminables -> interminable
    if p.endswith("ciones"):
        return p[:-6] + "ción"
    if p.endswith("iones"):
        return p[:-2]
    if p.endswith("ces"):
        return p[:-3] + "z"
    if p.endswith(("res", "les", "nes", "des", "jes", "ses", "xes")) and len(p) > 4:
        return p[:-2]               # transformadores -> transformador, apagones -> apagon
    if p.endswith("s") and not p.endswith("ss") and len(p) > 4:
        return p[:-1]               # fallas -> falla, luminarias -> luminaria
    return p


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------
def preprocesar(texto):
    """Ejecuta todo el pipeline y devuelve un diccionario.

    Claves: ``normalizado`` (str), ``tokens`` (list), ``sin_stopwords`` (list),
    ``stems`` (list de raíces de las palabras sin stopwords) y ``lemas`` (list de lemas).
    """
    normalizado = normalizar(texto)
    tokens = tokenizar(normalizado)
    sin_stop = quitar_stopwords(tokens)
    return {
        "normalizado": normalizado,
        "tokens": tokens,
        "sin_stopwords": sin_stop,
        "stems": [stem(t) for t in sin_stop],
        "lemas": [lematizar(t) for t in sin_stop],
    }


def texto_para_modelo(texto):
    """Cadena de stems separada por espacios: la entrada del vectorizador TF-IDF."""
    return " ".join(preprocesar(texto)["stems"])


if __name__ == "__main__":
    ejemplo = "El refrigerador NO funciona desde ayer 😡. Folio FOL-123456, cobraron $1,250!"
    import json
    print(json.dumps(preprocesar(ejemplo), ensure_ascii=False, indent=2))
    print("Stopwords:", len(STOPWORDS))
