"""Generación del conjunto de datos SIAPE: consumo eléctrico horario simulado.

Ejecución:  python -m src.data.generar_datos      (desde la raíz del proyecto)

=====================================================================================
POR QUÉ SIMULAR
=====================================================================================
No se cuenta con mediciones reales de las 10 instalaciones, así que se construye un
simulador físico-estadístico sencillo, con parámetros explícitos, que reproduce los
patrones que un operador de eficiencia energética observaría en el estado de Veracruz.
El resultado es un CSV "crudo" (``config.CSV_CONSUMO_RAW``) con problemas de calidad
inyectados a propósito, que luego ``preparar_datos.py`` debe detectar y corregir.

=====================================================================================
PROCEDIMIENTO DE GENERACIÓN (paso a paso)
=====================================================================================
Periodo: ``config.DIAS_SIMULADOS`` (182) días desde ``config.FECHA_INICIO``
(1-ene-2026 a 30-jun-2026), frecuencia horaria: 4,368 horas por instalación y
10 instalaciones => 43,680 registros "verdaderos".

1) CLIMA POR CIUDAD (``simular_clima``)
   Temperatura(t) = T_estacional(día) + A * cos(2π (hora - 15) / 24) + anomalía_diaria
                    + ruido_horario + efecto_frente_frío
   * T_estacional: crece linealmente de la media de enero a la media de junio de la
     ciudad (ver ``CLIMA_CIUDAD``; Xalapa/Orizaba/Córdoba son más frescas que la costa).
   * A: amplitud del ciclo diario (máximo a las 15 h, mínimo a las 3 h).
   * anomalía_diaria: proceso AR(1) (rho=0.7, sigma=1.8 °C) que representa que unos días
     son más cálidos que otros.
   * frentes fríos ("nortes"): entre enero y marzo, con probabilidad 6 % por día empieza
     un frente de 2 días que baja la temperatura 4 °C.
   Humedad relativa(t) = hum_base - 1.5 * (T - T_media_ciudad) + ruido AR(1), recortada a
   [25, 100] %; +5 puntos desde mayo (inicio de temporada húmeda).
   Cada instalación mide con su propio sensor: se suma ruido de sensor N(0, 0.3 °C) y
   N(0, 1.5 %) al clima de su ciudad.

2) PERFIL HORARIO POR TIPO (``PERFILES``)
   Cada tipo tiene un vector de 24 factores para días "de trabajo" (normalizado a
   media 1) y otro para días "de descanso" (normalizado con la media del de trabajo):
   * residencial: picos 6-8 h y 19-22 h; fin de semana más tarde y más alto de día.
   * comercial: abierto 9-21 h con picos al mediodía y 17-19 h; cerrado = refrigeración.
   * industrial: tres turnos (6-14, 14-22, 22-6) con el turno nocturno más bajo;
     sábado 80 % y domingo 45 % de la actividad.
   * educativa: lunes a viernes 7-21 h; fines de semana, festivos y vacaciones de
     Semana Santa 2026 (30-mar a 10-abr) el consumo cae a ~20 % (solo base).

3) CONSUMO BASE
   consumo = base_kwh * perfil_hora * factor_día * tendencia * eficiencia_instalación
             + componente_climatización
   * factor_día: coeficientes por tipo para sábado/domingo y festivos (``FACTOR_DIA``).
   * tendencia: +0.03 % por día (alrededor de +5.5 % en el semestre).
   * eficiencia_instalación ~ N(1, 0.03), fija por instalación.
   * componente_climatización (aire acondicionado): base_kwh * coef_tipo * max(0, T-26)
     * perfil_hora, con una pequeña corrección por humedad alta (>70 %) cuando T>26 °C.
     Los coeficientes están en ``COEF_AIRE`` (residencial 0.07 /°C, comercial 0.05,
     educativa 0.04, industrial 0.015).

4) EVENTOS OPERATIVOS (anomalías reales, NO errores de captura)
   Cada instalación tiene 3 días aleatorios de "paro/ausencia" (consumo ×0.35 todo el
   día). Sirven para que el clustering encuentre días atípicos.

5) RUIDO
   * factor aleatorio diario multiplicativo N(1, 0.04) (unos días se consume más).
   * ruido horario AR(1) (rho=0.6, sigma=0.05) multiplicativo: la desviación de una hora
     se parece a la de la hora anterior, igual que en la realidad.
   El consumo se recorta a un mínimo de 2 % del consumo base (nunca es negativo).

6) INYECCIÓN DE PROBLEMAS DE CALIDAD (``inyectar_problemas``), en este orden:
   a) Nulos: 2 % de las filas en ``temperatura_c``, 2 % en ``humedad_pct``, 2 % en
      ``consumo_kwh`` (columnas independientes).
   b) Atípicos: 0.5 % de las filas, repartidas en tres tipos:
        - consumo negativo (se cambia el signo),
        - pico de consumo (valor × 8),
        - temperatura 99.9 (falla de sensor).
   c) Inconsistencias de texto: 4 % de ``tipo_instalacion`` ("Residencial", " comercial ",
      "INDUSTRIAL"...) y 6 % de ``dia_semana`` ("LUNES", "Martes").
   d) Duplicados: 1 % de filas se copian exactamente (mismo ``id_registro``).
   Se guardan los conteos exactos en ``outputs/evidencias/generacion_datos.txt``.

Todo el azar usa ``numpy.random.default_rng(config.SEMILLA)``: la ejecución es
reproducible.

Supuestos y limitaciones: los datos son sintéticos; los parámetros se eligieron por
verosimilitud (no por calibración con mediciones); las ciudades comparten el clima entre
sus instalaciones; no se modelan apagones ni tarifas.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

# --------------------------------------------------------------------------------------
# Parámetros del simulador
# --------------------------------------------------------------------------------------
DIAS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

# Festivos / días no laborables de México 2026 dentro del periodo simulado.
FESTIVOS_2026 = {
    "2026-01-01",  # Año Nuevo
    "2026-02-02",  # Día de la Constitución (primer lunes de febrero)
    "2026-03-16",  # Natalicio de Benito Juárez (tercer lunes de marzo)
    "2026-04-02",  # Jueves Santo
    "2026-04-03",  # Viernes Santo
    "2026-05-01",  # Día del Trabajo
}
VACACIONES_INICIO = "2026-03-30"   # Semana Santa / Pascua 2026 (educativas)
VACACIONES_FIN = "2026-04-10"

# (temp. media enero, temp. media junio, amplitud diaria, humedad base)
CLIMA_CIUDAD = {
    "Veracruz":      (22.0, 28.5, 4.0, 78.0),
    "Boca del Río":  (22.0, 28.5, 4.0, 78.0),
    "Xalapa":        (14.5, 20.0, 5.0, 80.0),
    "Córdoba":       (19.5, 26.0, 6.0, 72.0),
    "Orizaba":       (15.5, 21.0, 5.5, 74.0),
    "Coatzacoalcos": (23.0, 29.0, 4.0, 80.0),
}

# Factores horarios (24 valores). Se normalizan dentro de ``simular_instalacion``.
PERFILES = {
    "residencial": {
        "trabajo": [0.55, 0.50, 0.48, 0.47, 0.50, 0.70, 1.20, 1.40, 1.10, 0.80, 0.75, 0.75,
                    0.80, 0.90, 0.85, 0.80, 0.85, 1.00, 1.20, 1.70, 1.90, 1.80, 1.40, 0.90],
        "descanso": [0.60, 0.52, 0.50, 0.48, 0.48, 0.50, 0.60, 0.80, 1.10, 1.30, 1.20, 1.10,
                     1.10, 1.10, 1.00, 1.00, 1.00, 1.10, 1.30, 1.60, 1.80, 1.70, 1.30, 0.90],
    },
    "comercial": {
        "trabajo": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.35, 0.60, 1.00, 1.10, 1.20,
                    1.30, 1.30, 1.15, 1.10, 1.15, 1.25, 1.25, 1.15, 1.00, 0.70, 0.40, 0.30],
        "descanso": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.35, 0.60, 1.00, 1.10, 1.20,
                     1.30, 1.30, 1.15, 1.10, 1.15, 1.25, 1.25, 1.15, 1.00, 0.70, 0.40, 0.30],
    },
    "industrial": {
        "trabajo": [0.70, 0.70, 0.70, 0.70, 0.72, 0.85, 1.15, 1.15, 1.15, 1.15, 1.15, 1.15,
                    1.15, 1.15, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 0.80, 0.70],
        "descanso": [0.70, 0.70, 0.70, 0.70, 0.72, 0.85, 1.15, 1.15, 1.15, 1.15, 1.15, 1.15,
                     1.15, 1.15, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 0.80, 0.70],
    },
    "educativa": {
        "trabajo": [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.40, 0.80, 1.30, 1.30, 1.30, 1.30,
                    1.30, 1.30, 1.00, 1.15, 1.15, 1.15, 1.00, 1.00, 1.00, 0.60, 0.25, 0.20],
        "descanso": [0.18] * 24,
    },
}

# Factor de actividad según el día (sábado, domingo, festivo)
FACTOR_DIA = {
    "residencial": {"sabado": 1.05, "domingo": 1.10, "festivo": 1.10},
    "comercial":   {"sabado": 1.10, "domingo": 0.90, "festivo": 0.85},
    "industrial":  {"sabado": 0.80, "domingo": 0.45, "festivo": 0.50},
    "educativa":   {"sabado": 1.00, "domingo": 1.00, "festivo": 1.00},  # ya cubierto por perfil
}
COEF_AIRE = {"residencial": 0.07, "comercial": 0.05, "educativa": 0.04, "industrial": 0.015}

TENDENCIA_DIARIA = 0.0003
SIGMA_EFICIENCIA = 0.03
SIGMA_DIA = 0.04
RHO_RUIDO, SIGMA_RUIDO = 0.6, 0.05
PAROS_POR_INSTALACION = 3
FACTOR_PARO = 0.35

# Inyección de problemas
PROP_NULOS = 0.02
PROP_ATIPICOS = 0.005
PROP_TEXTO_TIPO = 0.04
PROP_TEXTO_DIA = 0.06
PROP_DUPLICADOS = 0.01


def _ar1(rng, n: int, rho: float, sigma: float) -> np.ndarray:
    """Serie AR(1) estacionaria con desviación estándar ``sigma``."""
    eps = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), n)
    return lfilter([1.0], [1.0, -rho], eps)


# --------------------------------------------------------------------------------------
# 1) Clima
# --------------------------------------------------------------------------------------
def simular_clima(ciudad: str, fechas: pd.DatetimeIndex, rng) -> pd.DataFrame:
    """Simula temperatura (°C) y humedad (%) horarias para una ciudad."""
    t_ene, t_jun, amplitud, hum_base = CLIMA_CIUDAD[ciudad]
    n = len(fechas)
    n_dias = n // 24
    dia = np.arange(n) // 24
    hora = np.arange(n) % 24

    estacional = t_ene + (t_jun - t_ene) * dia / (n_dias - 1)
    ciclo = amplitud * np.cos(2 * np.pi * (hora - 15) / 24)
    anomalia = np.repeat(_ar1(rng, n_dias, 0.7, 1.8), 24)

    # Frentes fríos (nortes) de enero a marzo
    frente = np.zeros(n_dias)
    d = 0
    while d < n_dias:
        if d < 90 and rng.random() < 0.06:
            frente[d:d + 2] = -4.0
            d += 2
        else:
            d += 1
    frente = np.repeat(frente, 24)

    temp = estacional + ciclo + anomalia + frente + rng.normal(0, 0.4, n)

    t_media = (t_ene + t_jun) / 2
    hum = hum_base - 1.5 * (temp - t_media) + _ar1(rng, n, 0.9, 4.0)
    hum = hum + np.where(dia >= 120, 5.0, 0.0)       # desde mayo: temporada húmeda
    return pd.DataFrame({"temperatura_c": temp, "humedad_pct": np.clip(hum, 25, 100)})


# --------------------------------------------------------------------------------------
# 2-5) Consumo de una instalación
# --------------------------------------------------------------------------------------
def simular_instalacion(inst: dict, fechas: pd.DatetimeIndex, clima: pd.DataFrame, rng) -> pd.DataFrame:
    """Simula el consumo horario de una instalación y devuelve un DataFrame con sus columnas."""
    tipo = inst["tipo"]
    n = len(fechas)
    n_dias = n // 24
    hora = fechas.hour.to_numpy()
    dia_sem = fechas.dayofweek.to_numpy()
    fecha_str = fechas.strftime("%Y-%m-%d")
    es_festivo = np.isin(fecha_str, list(FESTIVOS_2026)).astype(int)
    en_vacaciones = (fecha_str >= VACACIONES_INICIO) & (fecha_str <= VACACIONES_FIN)

    # Sensor propio de la instalación sobre el clima de su ciudad
    temp = clima["temperatura_c"].to_numpy() + rng.normal(0, 0.3, n)
    hum = np.clip(clima["humedad_pct"].to_numpy() + rng.normal(0, 1.5, n), 25, 100)

    # Perfil horario: "trabajo" o "descanso"
    # Ambos perfiles se dividen entre la media del perfil de trabajo, de modo que un día
    # de descanso de una escuela (0.18) quede MUY por debajo de un día lectivo.
    ref = np.mean(PERFILES[tipo]["trabajo"])
    p_trab = (np.asarray(PERFILES[tipo]["trabajo"]) / ref)[hora]
    p_desc = (np.asarray(PERFILES[tipo]["descanso"]) / ref)[hora]
    fin_semana = dia_sem >= 5
    if tipo == "educativa":
        cerrado = fin_semana | (es_festivo == 1) | en_vacaciones
        perfil = np.where(cerrado, p_desc, p_trab)
    else:
        perfil = np.where(fin_semana | (es_festivo == 1), p_desc, p_trab)

    # Factor de actividad del día
    f = FACTOR_DIA[tipo]
    factor = np.ones(n)
    factor = np.where(dia_sem == 5, f["sabado"], factor)
    factor = np.where(dia_sem == 6, f["domingo"], factor)
    factor = np.where((es_festivo == 1) & (dia_sem < 5), f["festivo"], factor)

    # Tendencia leve, eficiencia propia y factor aleatorio de cada día
    tendencia = 1 + TENDENCIA_DIARIA * (np.arange(n) // 24)
    eficiencia = rng.normal(1.0, SIGMA_EFICIENCIA)
    factor_dia_rnd = np.repeat(rng.normal(1.0, SIGMA_DIA, n_dias), 24)

    base = inst["base_kwh"]
    consumo_base = base * perfil * factor * tendencia * eficiencia

    # Climatización: aire acondicionado por encima de 26 °C, proporcional a la ocupación
    exceso = np.maximum(0, temp - 26)
    corr_hum = 1 + 0.004 * np.maximum(0, hum - 70) * (temp > 26)
    climatizacion = base * COEF_AIRE[tipo] * exceso * corr_hum * perfil * factor
    consumo = consumo_base + climatizacion

    # Eventos operativos: 3 días de paro/ausencia (laborables, al azar)
    dias_laborables = np.where(pd.Series(fechas[::24].dayofweek < 5).to_numpy())[0]
    dias_paro = rng.choice(dias_laborables, PAROS_POR_INSTALACION, replace=False)
    mult_paro = np.ones(n_dias)
    mult_paro[dias_paro] = FACTOR_PARO
    consumo = consumo * np.repeat(mult_paro, 24)

    # Ruido: factor diario + AR(1) horario multiplicativo
    ruido = 1 + _ar1(rng, n, RHO_RUIDO, SIGMA_RUIDO)
    consumo = consumo * factor_dia_rnd * ruido
    consumo = np.maximum(consumo, 0.02 * base)

    return pd.DataFrame({
        "id_instalacion": inst["id"],
        "tipo_instalacion": tipo,
        "ciudad": inst["ciudad"],
        "fecha_hora": fechas,
        "es_festivo": es_festivo,
        "temperatura_c": np.round(temp, 1),
        "humedad_pct": np.round(hum, 1),
        "superficie_m2": inst["superficie_m2"],
        "num_ocupantes": inst["ocupantes"],
        "consumo_kwh": np.round(consumo, 3),
        "_dias_paro": len(dias_paro),
    })


# --------------------------------------------------------------------------------------
# 6) Problemas de calidad
# --------------------------------------------------------------------------------------
def inyectar_problemas(df: pd.DataFrame, rng) -> tuple[pd.DataFrame, dict]:
    """Inyecta nulos, atípicos, inconsistencias de texto y duplicados. Devuelve (df, conteos)."""
    df = df.copy()
    n = len(df)
    conteos: dict = {"filas_simuladas": n}

    # a) Nulos independientes por columna
    n_nulos = int(round(PROP_NULOS * n))
    filas_nulas = {}
    for col in ["temperatura_c", "humedad_pct", "consumo_kwh"]:
        idx = rng.choice(n, n_nulos, replace=False)
        df.loc[df.index[idx], col] = np.nan
        filas_nulas[col] = set(idx.tolist())
        conteos[f"nulos_{col}"] = n_nulos

    # b) Atípicos: 0.5 % de las filas, en tres tipos, sobre valores no nulos
    n_at = int(round(PROP_ATIPICOS * n))
    n_neg = n_at // 3
    n_pico = n_at // 3
    n_temp = n_at - n_neg - n_pico
    usadas: set = set()

    def elegir(cantidad: int, col: str) -> np.ndarray:
        candidatos = np.array(sorted(set(range(n)) - filas_nulas[col] - usadas))
        elegidas = rng.choice(candidatos, cantidad, replace=False)
        usadas.update(elegidas.tolist())
        return elegidas

    i_neg = elegir(n_neg, "consumo_kwh")
    df.iloc[i_neg, df.columns.get_loc("consumo_kwh")] *= -1
    i_pico = elegir(n_pico, "consumo_kwh")
    df.iloc[i_pico, df.columns.get_loc("consumo_kwh")] *= 8
    i_temp = elegir(n_temp, "temperatura_c")
    df.iloc[i_temp, df.columns.get_loc("temperatura_c")] = 99.9
    conteos["atipicos_consumo_negativo"] = n_neg
    conteos["atipicos_pico_x8"] = n_pico
    conteos["atipicos_temperatura_99.9"] = n_temp
    conteos["atipicos_total"] = n_at

    # dia_semana (texto) y variantes de texto
    dias = np.array(DIAS_ES, dtype=object)[df["fecha_hora"].dt.dayofweek.to_numpy()]
    df["dia_semana"] = dias
    df["tipo_instalacion"] = df["tipo_instalacion"].astype(object)

    # c) Inconsistencias de texto
    n_tipo = int(round(PROP_TEXTO_TIPO * n))
    idx_tipo = rng.choice(n, n_tipo, replace=False)
    originales = df["tipo_instalacion"].to_numpy(dtype=object).copy()
    for i in idx_tipo:
        v = originales[i]
        originales[i] = rng.choice([v.capitalize(), f" {v} ", v.upper(), f"{v} ", f" {v.capitalize()}"])
    df["tipo_instalacion"] = originales
    conteos["texto_tipo_instalacion_inconsistente"] = n_tipo

    n_dia = int(round(PROP_TEXTO_DIA * n))
    idx_dia = rng.choice(n, n_dia, replace=False)
    dias = df["dia_semana"].to_numpy(dtype=object).copy()
    for i in idx_dia:
        v = dias[i]
        dias[i] = v.upper() if rng.random() < 0.5 else v.capitalize()
    df["dia_semana"] = dias
    conteos["texto_dia_semana_inconsistente"] = n_dia

    # d) Duplicados exactos (se colocan justo después de su original)
    df["_orden"] = np.arange(n, dtype=float)
    n_dup = int(round(PROP_DUPLICADOS * n))
    copias = df.iloc[rng.choice(n, n_dup, replace=False)].copy()
    copias["_orden"] += 0.5
    df = pd.concat([df, copias]).sort_values("_orden", kind="stable").drop(columns="_orden")
    conteos["filas_duplicadas"] = n_dup
    conteos["filas_finales"] = len(df)
    return df.reset_index(drop=True), conteos


def generar_consumo() -> pd.DataFrame:
    """Simula las 10 instalaciones, inyecta problemas y devuelve el DataFrame crudo."""
    rng = np.random.default_rng(config.SEMILLA)
    fechas = pd.date_range(config.FECHA_INICIO, periods=config.DIAS_SIMULADOS * 24, freq="h")

    # Clima compartido por ciudad (se generan en orden alfabético: reproducible)
    clima = {c: simular_clima(c, fechas, rng) for c in sorted({i["ciudad"] for i in config.INSTALACIONES})}

    partes = [simular_instalacion(inst, fechas, clima[inst["ciudad"]], rng) for inst in config.INSTALACIONES]
    df = pd.concat(partes, ignore_index=True)
    n_paros = int(df.groupby("id_instalacion")["_dias_paro"].first().sum())
    df = df.drop(columns="_dias_paro")
    df.insert(0, "id_registro", [f"REG-{i:06d}" for i in range(1, len(df) + 1)])

    df, conteos = inyectar_problemas(df, rng)
    conteos["eventos_paro_ausencia_(dias, no son errores)"] = n_paros
    df.attrs["conteos"] = conteos
    return df


COLUMNAS_RAW = ["id_registro", "id_instalacion", "tipo_instalacion", "ciudad", "fecha_hora", "dia_semana",
                "es_festivo", "temperatura_c", "humedad_pct", "superficie_m2", "num_ocupantes", "consumo_kwh"]


def main() -> pd.DataFrame:
    df = generar_consumo()
    conteos = df.attrs["conteos"]
    df = df[COLUMNAS_RAW].copy()
    df["fecha_hora"] = df["fecha_hora"].dt.strftime("%Y-%m-%d %H:%M:%S")
    config.CSV_CONSUMO_RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CSV_CONSUMO_RAW, index=False, encoding="utf-8")

    lineas = [
        "EVIDENCIA - GENERACIÓN DE DATOS DE CONSUMO (SIAPE)",
        f"Fecha de ejecución: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Semilla: {config.SEMILLA}",
        f"Periodo: {config.FECHA_INICIO} + {config.DIAS_SIMULADOS} días (frecuencia horaria)",
        f"Instalaciones: {len(config.INSTALACIONES)}  |  Archivo: {config.CSV_CONSUMO_RAW}",
        "",
        "Conteos de problemas inyectados a propósito:",
    ]
    lineas += [f"  - {k}: {v}" for k, v in conteos.items()]
    lineas += ["", "Muestra del archivo crudo (primeras 5 filas):", df.head().to_string(index=False), "",
               "Estadísticos del consumo por tipo (kWh, sin corregir):",
               df.groupby(df["tipo_instalacion"].str.strip().str.lower())["consumo_kwh"].describe().round(2).to_string()]
    texto = "\n".join(lineas)
    print(texto)
    (config.DIR_EVIDENCIAS / "generacion_datos.txt").write_text(texto + "\n", encoding="utf-8")
    return df


if __name__ == "__main__":
    main()
