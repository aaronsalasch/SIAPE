"""Limpieza y preparación de datos de consumo (CRISP-DM: preparación de los datos).

Ejecución:  python -m src.data.preparar_datos
Uso como módulo:  ``from src.data.preparar_datos import preparar_consumo``

Entrada : ``config.CSV_CONSUMO_RAW``  (datos crudos con problemas de calidad)
Salidas : ``config.CSV_CONSUMO_PROC``, ``outputs/evidencias/preparacion_datos.txt`` y
          ``outputs/evidencias/resumen_preparacion.json``

Pasos (en este orden; el orden importa para NO tener fuga de información):
 1. Carga y diagnóstico inicial (nulos y tipos).
 2. Normalización de texto (``tipo_instalacion`` y ``dia_semana``).
 3. Eliminación de duplicados (filas idénticas y llave instalación+hora repetida).
 4. Detección de atípicos: reglas físicas + IQR (picos) por instalación/hora/tipo de día.
    Los atípicos se convierten en nulos.
 5. Imputación: interpolación temporal por instalación (nulos originales + atípicos).
 6. Ingeniería de características (variables de calendario, cíclicas, interacción).
    Los *lags* y la media móvil se calculan DESPUÉS de limpiar y solo con valores
    pasados (``shift`` antes de ``rolling``), así ninguna fila "ve" su propio valor ni
    el futuro.
 7. División TEMPORAL entrenamiento/prueba: el último ``config.FRACCION_TEST`` del
    periodo es prueba (nunca se mezclan filas al azar).
 8. Análisis de correlación para selección de características.

Nota sobre la normalización: el escalado (StandardScaler) y la codificación (OneHot) se
hacen dentro del ``Pipeline`` del modelo (ver ``src/ml/entrenar_modelo.py``) y se ajustan
únicamente con el conjunto de entrenamiento; por eso aquí no se escalan los datos.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

DIAS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
HORA_PICO = (18, 21)             # horario de demanda alta (18:00 a 21:59)
FACTOR_IQR = 3.0                 # valla "extrema": Q3 + 3*IQR
TEMP_MIN, TEMP_MAX = -5.0, 50.0  # límites físicos razonables para Veracruz (°C)

COLUMNAS_NUEVAS = ["hora", "dia_semana_num", "mes", "dia_anio", "es_fin_semana", "hora_sin", "hora_cos",
                   "mes_sin", "mes_cos", "temp_x_hora_pico", "consumo_lag_1h", "consumo_lag_24h",
                   "consumo_lag_168h", "consumo_media_movil_24h", "conjunto"]
COLUMNAS_BASE = ["id_registro", "id_instalacion", "tipo_instalacion", "ciudad", "fecha_hora", "dia_semana",
                 "es_festivo", "temperatura_c", "humedad_pct", "superficie_m2", "num_ocupantes", "consumo_kwh"]
COLUMNAS_PROC = COLUMNAS_BASE + COLUMNAS_NUEVAS


def agregar_variables_calendario(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las variables que dependen solo de la fecha/hora y la temperatura.

    Se comparte con ``src/ml/predecir.py`` para que entrenamiento e inferencia usen
    exactamente las mismas definiciones.
    """
    f = df["fecha_hora"]
    df["hora"] = f.dt.hour
    df["dia_semana_num"] = f.dt.dayofweek                      # 0 = lunes
    df["mes"] = f.dt.month
    df["dia_anio"] = f.dt.dayofyear
    df["es_fin_semana"] = (df["dia_semana_num"] >= 5).astype(int)
    # Codificación cíclica: la hora 23 queda "cerca" de la 0 (igual diciembre y enero)
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    es_pico = df["hora"].between(*HORA_PICO).astype(int)
    df["temp_x_hora_pico"] = df["temperatura_c"] * es_pico     # calor en la hora de mayor demanda
    return df


def _contar_nulos(df: pd.DataFrame, columnas) -> dict:
    return {c: int(df[c].isna().sum()) for c in columnas}


def preparar_consumo() -> pd.DataFrame:
    """Ejecuta todo el proceso de limpieza y devuelve el DataFrame procesado (también lo guarda)."""
    log: list[str] = []
    resumen: dict = {"fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    def p(texto: str = "") -> None:
        log.append(texto)

    cols_num = ["temperatura_c", "humedad_pct", "consumo_kwh"]

    # ------------------------------------------------------------------ 1. Carga
    df = pd.read_csv(config.CSV_CONSUMO_RAW, encoding="utf-8")
    resumen["filas_raw"] = int(len(df))
    nulos_antes = _contar_nulos(df, df.columns)
    p("EVIDENCIA - PREPARACIÓN DE DATOS DE CONSUMO (SIAPE)")
    p(f"Fecha de ejecución: {resumen['fecha_ejecucion']}")
    p(f"Entrada: {config.CSV_CONSUMO_RAW}  ({len(df)} filas, {df.shape[1]} columnas)")
    p("\n1) NULOS ANTES DE LA LIMPIEZA (por columna)")
    for c, v in nulos_antes.items():
        if v:
            p(f"   {c:<18}{v:>6}  ({100 * v / len(df):.2f} %)")

    # ------------------------------------------------------------------ 2. Texto
    tipo_antes = df["tipo_instalacion"].nunique()
    dia_antes = df["dia_semana"].nunique()
    df["tipo_instalacion"] = df["tipo_instalacion"].str.strip().str.lower()
    df["dia_semana"] = df["dia_semana"].str.strip().str.lower()
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], format="%Y-%m-%d %H:%M:%S")
    # El día de la semana se verifica contra la fecha (fuente de verdad)
    dia_real = df["fecha_hora"].dt.dayofweek.map(dict(enumerate(DIAS_ES)))
    dia_incons = int((df["dia_semana"] != dia_real).sum())
    df["dia_semana"] = dia_real
    # El tipo se verifica contra el catálogo de instalaciones de config.py
    tipo_real = df["id_instalacion"].map({i["id"]: i["tipo"] for i in config.INSTALACIONES})
    tipo_incons = int((df["tipo_instalacion"] != tipo_real).sum())
    df["tipo_instalacion"] = tipo_real
    p("\n2) NORMALIZACIÓN DE TEXTO")
    p(f"   tipo_instalacion: {tipo_antes} variantes distintas -> {df['tipo_instalacion'].nunique()} "
      f"(strip + minúsculas); {tipo_incons} inconsistencias con el catálogo tras limpiar")
    p(f"   dia_semana: {dia_antes} variantes distintas -> {df['dia_semana'].nunique()} "
      f"(strip + minúsculas); {dia_incons} discrepancias con la fecha tras limpiar")

    # ------------------------------------------------------------------ 3. Duplicados
    n0 = len(df)
    dup_exactos = int(df.duplicated().sum())
    df = df.drop_duplicates()
    dup_llave = int(df.duplicated(["id_instalacion", "fecha_hora"]).sum())
    df = df.drop_duplicates(["id_instalacion", "fecha_hora"], keep="first")
    df = df.sort_values(["id_instalacion", "fecha_hora"]).reset_index(drop=True)
    resumen["duplicados_eliminados"] = dup_exactos + dup_llave
    p("\n3) DUPLICADOS")
    p(f"   Filas idénticas eliminadas: {dup_exactos}; llave (instalación, hora) repetida: {dup_llave}")
    p(f"   Filas: {n0} -> {len(df)}")
    horas_esperadas = config.DIAS_SIMULADOS * 24
    faltantes = int(sum(horas_esperadas - g for g in df.groupby("id_instalacion").size()))
    p(f"   Horas faltantes respecto al calendario esperado: {faltantes}")

    # ------------------------------------------------------------------ 4. Atípicos
    p("\n4) ATÍPICOS (reglas físicas + IQR por instalación/hora)")
    mask_temp = (df["temperatura_c"] > TEMP_MAX) | (df["temperatura_c"] < TEMP_MIN)
    mask_hum = (df["humedad_pct"] > 100) | (df["humedad_pct"] < 0)
    mask_neg = df["consumo_kwh"] < 0
    p(f"   Regla física temperatura fuera de [{TEMP_MIN}, {TEMP_MAX}] °C: {int(mask_temp.sum())}")
    p(f"   Regla física humedad fuera de [0, 100] %: {int(mask_hum.sum())}")
    p(f"   Regla física consumo negativo: {int(mask_neg.sum())}")

    # IQR por instalación, hora del día y tipo de día (laboral / no laborable: fin de semana
    # o festivo), para no confundir la caída normal de los domingos con un atípico. Se
    # calcula sobre los valores no negativos (los negativos ya se detectaron con la regla
    # física). Solo se tratan los PICOS (valla superior): una caída fuerte puede ser un paro
    # o ausencia real y no un error de captura.
    no_laborable = (df["fecha_hora"].dt.dayofweek >= 5) | (df["es_festivo"] == 1)
    clave = [df["id_instalacion"], df["fecha_hora"].dt.hour, no_laborable]
    valido = df["consumo_kwh"].where(~mask_neg)
    g = valido.groupby(clave)
    q1, q3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    mask_iqr = valido > q3 + FACTOR_IQR * iqr
    p(f"   Regla IQR (valor > Q3 + {FACTOR_IQR:g}*IQR) por instalación/hora/tipo de día: "
      f"{int(mask_iqr.sum())} picos extremos de consumo")
    atipicos_temp = int(mask_temp.sum() + mask_hum.sum())
    atipicos_cons = int((mask_neg | mask_iqr).sum())
    resumen["atipicos"] = {"temperatura_fuera_de_rango": int(mask_temp.sum()),
                           "humedad_fuera_de_rango": int(mask_hum.sum()),
                           "consumo_negativo": int(mask_neg.sum()),
                           "consumo_pico_iqr": int(mask_iqr.sum()),
                           "total_detectados": atipicos_temp + atipicos_cons}
    p(f"   Tratamiento: se marcan como nulos (NaN) y se imputan en el paso 5. Total: "
      f"{resumen['atipicos']['total_detectados']} valores.")
    df.loc[mask_temp, "temperatura_c"] = np.nan
    df.loc[mask_hum, "humedad_pct"] = np.nan
    df.loc[mask_neg | mask_iqr, "consumo_kwh"] = np.nan

    # ------------------------------------------------------------------ 5. Imputación
    nulos_para_imputar = _contar_nulos(df, cols_num)
    p("\n5) IMPUTACIÓN (interpolación lineal en el tiempo, por instalación)")
    p("   Estrategia: la serie es horaria y suave, por lo que se interpola entre la hora válida")
    p("   anterior y la siguiente de la MISMA instalación. Nulos a imputar (originales + atípicos):")
    for c, v in nulos_para_imputar.items():
        p(f"   {c:<18}{v:>6}")
    df = df.set_index("fecha_hora")
    for c in cols_num:
        df[c] = (df.groupby("id_instalacion")[c]
                   .transform(lambda s: s.interpolate(method="time", limit_direction="both")))
    df = df.reset_index()
    df["consumo_kwh"] = df["consumo_kwh"].clip(lower=0).round(3)
    df["temperatura_c"] = df["temperatura_c"].round(1)
    df["humedad_pct"] = df["humedad_pct"].round(1)
    nulos_despues = _contar_nulos(df, COLUMNAS_BASE)
    p("   Nulos DESPUÉS de imputar:")
    for c in cols_num:
        p(f"   {c:<18}{nulos_despues[c]:>6}")
    resumen["nulos_antes"] = {c: v for c, v in nulos_antes.items() if v}
    resumen["nulos_despues"] = nulos_despues
    resumen["estrategia_nulos"] = "interpolación temporal lineal por instalación"

    # ------------------------------------------------------------------ 6. Variables
    df = agregar_variables_calendario(df)
    g = df.groupby("id_instalacion")["consumo_kwh"]
    df["consumo_lag_1h"] = g.shift(1)
    df["consumo_lag_24h"] = g.shift(24)
    df["consumo_lag_168h"] = g.shift(168)
    # Media móvil de las 24 horas ANTERIORES (shift(1) evita incluir la hora actual)
    df["consumo_media_movil_24h"] = (g.transform(lambda s: s.shift(1).rolling(24, min_periods=24).mean()))
    antes_lags = len(df)

    # ------------------------------------------------------------------ 7. División temporal
    horas_unicas = np.sort(df["fecha_hora"].unique())
    corte = horas_unicas[int(len(horas_unicas) * (1 - config.FRACCION_TEST))]
    df["conjunto"] = np.where(df["fecha_hora"] >= corte, "prueba", "entrenamiento")
    df = df.dropna(subset=["consumo_lag_168h", "consumo_media_movil_24h"]).reset_index(drop=True)
    sin_historial = antes_lags - len(df)
    df["fecha_hora_txt"] = df["fecha_hora"].dt.strftime("%Y-%m-%d %H:%M:%S")

    p("\n6) INGENIERÍA DE CARACTERÍSTICAS (variables creadas)")
    p("   Calendario: hora, dia_semana_num (0=lunes), mes, dia_anio, es_fin_semana")
    p("   Cíclicas: hora_sin/hora_cos (periodo 24), mes_sin/mes_cos (periodo 12)")
    p(f"   Interacción: temp_x_hora_pico = temperatura_c * (hora entre {HORA_PICO[0]} y {HORA_PICO[1]})")
    p("   Históricas (solo valores pasados): consumo_lag_1h, consumo_lag_24h, consumo_lag_168h,")
    p("   consumo_media_movil_24h (media de las 24 h previas, sin incluir la actual)")
    p(f"   Filas descartadas por no tener 168 h de historia (primera semana de cada instalación): {sin_historial}")

    p("\n7) DIVISIÓN TEMPORAL")
    p(f"   Corte: {pd.Timestamp(corte)}  (último {config.FRACCION_TEST:.0%} del periodo = prueba)")
    cuenta = df["conjunto"].value_counts()
    ini_pr, fin_ent = df.loc[df["conjunto"] == "prueba", "fecha_hora"].min(), df.loc[df["conjunto"] == "entrenamiento", "fecha_hora"].max()
    p(f"   entrenamiento: {cuenta['entrenamiento']} filas (hasta {fin_ent})")
    p(f"   prueba: {cuenta['prueba']} filas (desde {ini_pr})")
    p("   Garantía anti-fuga: todos los lags usan solo horas anteriores; el escalado/codificación")
    p("   se ajustan dentro del Pipeline solo con entrenamiento.")

    # ------------------------------------------------------------------ 8. Selección
    ent = df[df["conjunto"] == "entrenamiento"]
    numericas = [c for c in COLUMNAS_PROC if c not in
                 ("id_registro", "id_instalacion", "tipo_instalacion", "ciudad", "fecha_hora", "dia_semana", "conjunto")
                 and c != "consumo_kwh"]
    # Correlación con el consumo RELATIVO a la media de cada instalación (para que el
    # tamaño de la instalación no domine el análisis)
    rel = ent["consumo_kwh"] / ent.groupby("id_instalacion")["consumo_kwh"].transform("mean")
    corr_obj = ent[numericas].corrwith(rel).sort_values(key=np.abs, ascending=False)
    corr_mat = ent[numericas].corr().abs()
    pares = [(a, b, float(corr_mat.loc[a, b])) for i, a in enumerate(numericas) for b in numericas[i + 1:]
             if corr_mat.loc[a, b] > 0.95]
    p("\n8) SELECCIÓN DE CARACTERÍSTICAS (correlación de Pearson con el consumo relativo, entrenamiento)")
    for c, v in corr_obj.items():
        p(f"   {c:<26}{v:>8.3f}")
    p("   Pares muy correlacionados (|r| > 0.95): " + (", ".join(f"{a}~{b} ({r:.2f})" for a, b, r in pares) or "ninguno"))
    p("   Decisión: se conservan todas (los modelos de árboles toleran redundancia; la")
    p("   importancia de variables del Random Forest se reporta en entrenamiento_modelo.txt).")
    p("   Categóricas (tipo_instalacion, ciudad) se codifican con OneHotEncoder dentro del Pipeline.")
    p("   Escalado: StandardScaler dentro del Pipeline (numéricas), ajustado solo con entrenamiento.")

    # ------------------------------------------------------------------ Salida
    df["fecha_hora"] = df["fecha_hora_txt"]
    df = df[COLUMNAS_PROC]
    config.CSV_CONSUMO_PROC.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CSV_CONSUMO_PROC, index=False, encoding="utf-8")
    p(f"\nSalida: {config.CSV_CONSUMO_PROC}  ({len(df)} filas, {df.shape[1]} columnas)")
    p(f"Nulos totales en el procesado: {int(df.isna().sum().sum())}")

    resumen.update({
        "filas_procesadas": int(len(df)), "columnas_procesadas": int(df.shape[1]),
        "filas_descartadas_sin_historia": int(sin_historial),
        "transformaciones": ["strip+minúsculas en texto", "día de la semana recalculado desde la fecha",
                             "eliminación de duplicados", "atípicos -> NaN (reglas físicas + IQR)",
                             "interpolación temporal por instalación", "variables de calendario y cíclicas",
                             "lags 1/24/168 h y media móvil 24 h (solo pasado)", "división temporal"],
        "variables_creadas": COLUMNAS_NUEVAS[:-1],
        "division": {"corte": str(pd.Timestamp(corte)), "fraccion_test": config.FRACCION_TEST,
                     "n_entrenamiento": int(cuenta["entrenamiento"]), "n_prueba": int(cuenta["prueba"])},
        "correlacion_con_consumo_relativo": {k: round(float(v), 4) for k, v in corr_obj.items()},
        "pares_muy_correlacionados": [[a, b, round(r, 3)] for a, b, r in pares],
    })
    texto = "\n".join(log)
    print(texto)
    (config.DIR_EVIDENCIAS / "preparacion_datos.txt").write_text(texto + "\n", encoding="utf-8")
    (config.DIR_EVIDENCIAS / "resumen_preparacion.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return df


if __name__ == "__main__":
    preparar_consumo()
