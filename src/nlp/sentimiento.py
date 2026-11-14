"""Clasificación de sentimiento (positivo / negativo / neutral) de reportes de usuarios.

Ejecución: ``python -m src.nlp.sentimiento``

Pipeline:
  texto -> preprocesar (normalizar, tokenizar, quitar stopwords sin negaciones, stemming)
        -> TF-IDF con unigramas + bigramas (los bigramas capturan "no funciona", "sin fallas")
        -> clasificador.
Se comparan tres clasificadores: Naive Bayes multinomial, Regresión Logística y SVM lineal
(calibrado para obtener probabilidades).  El mejor se elige por F1-macro promedio en
validación cruzada (5 pliegues) sobre el conjunto de entrenamiento; las métricas reportadas
se calculan en el conjunto de prueba (20 %, estratificado), que no se usa para elegir.

El modelo guardado recibe texto YA preprocesado (cadena de stems); ``predecir_sentimiento``
se encarga de preprocesar, de modo que el archivo joblib no depende de funciones propias.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from src.nlp.procesamiento_texto import texto_para_modelo  # noqa: E402

CLASES = ["negativo", "neutral", "positivo"]
_CACHE = {}


def _vectorizador():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")


def _candidatos():
    """Los tres clasificadores a comparar, cada uno dentro de un Pipeline con su TF-IDF."""
    return {
        "Naive Bayes multinomial": Pipeline([("tfidf", _vectorizador()), ("clf", MultinomialNB(alpha=0.3))]),
        "Regresión Logística": Pipeline([("tfidf", _vectorizador()),
                                         ("clf", LogisticRegression(C=5.0, max_iter=2000, random_state=config.SEMILLA))]),
        "SVM lineal": Pipeline([("tfidf", _vectorizador()),
                                ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, random_state=config.SEMILLA), cv=3))]),
    }


def cargar_datos():
    """Lee los reportes raw (los genera si no existen) y devuelve (df, textos_preprocesados)."""
    if not config.CSV_REPORTES_RAW.exists():
        from src.data.generar_reportes import generar_reportes
        generar_reportes()
    df = pd.read_csv(config.CSV_REPORTES_RAW)
    return df, [texto_para_modelo(t) for t in df["texto"]]


def terminos_top(df_X, y, n=12):
    """Términos con mayor peso por clase según una Regresión Logística auxiliar."""
    vec = _vectorizador()
    X = vec.fit_transform(df_X)
    lr = LogisticRegression(C=5.0, max_iter=2000, random_state=config.SEMILLA).fit(X, y)
    nombres = np.array(vec.get_feature_names_out())
    top = {}
    for i, clase in enumerate(lr.classes_):
        orden = np.argsort(lr.coef_[i])[::-1][:n]
        top[clase] = [{"termino": nombres[j], "peso": round(float(lr.coef_[i][j]), 3)} for j in orden]
    return top


def entrenar():
    """Entrena y compara los clasificadores. Devuelve (metricas, texto_evidencia)."""
    from src.nlp.entidades import evaluar_ner, tabla_ner
    df, X = cargar_datos()
    y = df["sentimiento_etiqueta"].values
    idx = np.arange(len(df))
    i_tr, i_te = train_test_split(idx, test_size=0.20, stratify=y, random_state=config.SEMILLA)
    X_tr, X_te = [X[i] for i in i_tr], [X[i] for i in i_te]
    y_tr, y_te = y[i_tr], y[i_te]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEMILLA)
    resultados, modelos, preds = {}, {}, {}
    for nombre, pipe in _candidatos().items():
        cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro").mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        resultados[nombre] = {
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "f1_macro": round(f1_score(y_te, pred, average="macro"), 4),
            "precision_macro": round(precision_score(y_te, pred, average="macro"), 4),
            "recall_macro": round(recall_score(y_te, pred, average="macro"), 4),
            "f1_macro_cv": round(float(cv_f1), 4),
        }
        modelos[nombre], preds[nombre] = pipe, pred

    seleccionado = max(resultados, key=lambda n: resultados[n]["f1_macro_cv"])
    pred = preds[seleccionado]
    reporte = classification_report(y_te, pred, labels=CLASES, output_dict=True, zero_division=0)
    reporte_txt = classification_report(y_te, pred, labels=CLASES, digits=3, zero_division=0)
    matriz = confusion_matrix(y_te, pred, labels=CLASES)

    # Se guarda el modelo seleccionado, reentrenado con todos los datos
    final = _candidatos()[seleccionado].fit(X, y)
    joblib.dump({"pipeline": final, "clases": list(final.classes_), "nombre": seleccionado},
                config.MODELO_SENTIMIENTO)

    previas = json.loads(config.METRICAS_PNL.read_text(encoding="utf-8")) if config.METRICAS_PNL.exists() else {}
    metricas = {
        "seleccionado": seleccionado,
        "modelos": resultados,
        "reporte_clasificacion": reporte,
        "matriz_confusion": {"clases": CLASES, "valores": matriz.tolist()},
        "n_entrenamiento": int(len(i_tr)),
        "n_prueba": int(len(i_te)),
        "terminos_top_por_clase": terminos_top(X_tr, y_tr),
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
    }
    for k in ("ner", "ner_global"):
        if k in previas:
            metricas[k] = previas[k]
    config.METRICAS_PNL.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")

    _figuras(matriz, resultados, seleccionado)

    ner = evaluar_ner()       # agrega "ner" a METRICAS_PNL
    _evidencia(df, resultados, seleccionado, reporte_txt, matriz, metricas, ner, tabla_ner(ner), len(i_tr), len(i_te))
    return metricas


def _figuras(matriz, resultados, seleccionado):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", xticklabels=CLASES, yticklabels=CLASES, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión - {seleccionado}")
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / "pnl_matriz_confusion.png", dpi=110)
    plt.close(fig)

    nombres = list(resultados)
    x = np.arange(len(nombres))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, [resultados[n]["accuracy"] for n in nombres], 0.4, label="Accuracy")
    ax.bar(x + 0.2, [resultados[n]["f1_macro"] for n in nombres], 0.4, label="F1-macro")
    ax.set_xticks(x)
    ax.set_xticklabels(nombres)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Comparación de clasificadores de sentimiento (prueba)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / "pnl_comparacion_clasificadores.png", dpi=110)
    plt.close(fig)


def _evidencia(df, resultados, seleccionado, reporte_txt, matriz, metricas, ner, tabla, n_tr, n_te):
    L = ["EVIDENCIA - ENTRENAMIENTO PNL (SENTIMIENTO + NER)", "=" * 64,
         f"Reportes: {len(df)} | entrenamiento: {n_tr} | prueba: {n_te} (estratificado, semilla {config.SEMILLA})",
         "Representación: TF-IDF unigramas+bigramas sobre stems (sin stopwords, con negaciones).",
         "", "1. COMPARACIÓN DE CLASIFICADORES (conjunto de prueba)",
         f"{'Modelo':<26}{'Accuracy':>9}{'F1-macro':>10}{'Prec.':>8}{'Recall':>8}{'F1 CV(5)':>10}"]
    for n, m in resultados.items():
        L.append(f"{n:<26}{m['accuracy']:>9.4f}{m['f1_macro']:>10.4f}{m['precision_macro']:>8.4f}"
                 f"{m['recall_macro']:>8.4f}{m['f1_macro_cv']:>10.4f}")
    L += ["", f"Seleccionado (mayor F1-macro en validación cruzada): {seleccionado}", "",
          "2. REPORTE DE CLASIFICACIÓN (modelo seleccionado, prueba)", reporte_txt,
          "Matriz de confusión (filas=real, columnas=predicho; orden: " + ", ".join(CLASES) + ")",
          str(matriz), "",
          "Nota: la accuracy no es 1.0 porque ~10 % de los reportes son mixtos/ambiguos y ~3 % de las",
          "etiquetas tienen ruido deliberado (ver docstring de src/data/generar_reportes.py).", "",
          "3. TÉRMINOS MÁS INFLUYENTES POR CLASE (Regresión Logística)"]
    for clase, ts in metricas["terminos_top_por_clase"].items():
        L.append(f"  {clase}: " + ", ".join(f"{t['termino']} ({t['peso']})" for t in ts))
    L += ["", "4. NER (diccionarios + regex) contra entidades_esperadas del generador",
          "(un acierto = mismo texto en minúsculas y mismo tipo dentro del reporte)", tabla, "",
          "Limitación: los alias coloquiales ('refri', 'clima') y errores de dedo en ciudades no están",
          "en los diccionarios, por eso el recall de DISPOSITIVO y LUGAR no es 1.0. Al ser datos",
          "simulados con patrones regulares, el NER sobre texto real sería menos perfecto."]
    (config.DIR_EVIDENCIAS / "pnl_entrenamiento.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def _cargar_modelo():
    if "m" not in _CACHE:
        if not config.MODELO_SENTIMIENTO.exists():
            entrenar()
        _CACHE["m"] = joblib.load(config.MODELO_SENTIMIENTO)
    return _CACHE["m"]


def predecir_sentimiento(texto):
    """Predice el sentimiento de un texto.

    Devuelve ``{"sentimiento": str, "probabilidades": {"negativo": p, "neutral": p, "positivo": p}}``.
    """
    m = _cargar_modelo()
    pipe = m["pipeline"]
    proba = pipe.predict_proba([texto_para_modelo(texto)])[0]
    probs = {c: round(float(p), 4) for c, p in zip(m["clases"], proba)}
    return {"sentimiento": max(probs, key=probs.get), "probabilidades": probs}


def predecir_sentimiento_lote(textos):
    """Versión vectorizada: devuelve (etiquetas, matriz de probabilidades, clases)."""
    m = _cargar_modelo()
    proba = m["pipeline"].predict_proba([texto_para_modelo(t) for t in textos])
    etiquetas = [m["clases"][i] for i in proba.argmax(axis=1)]
    return etiquetas, proba, m["clases"]


def main():
    metricas = entrenar()
    print("Clasificador seleccionado:", metricas["seleccionado"])
    for n, m in metricas["modelos"].items():
        print(f"  {n:<26} acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} f1_cv={m['f1_macro_cv']:.4f}")
    for frase in ["El refrigerador no funciona, pésimo servicio de CFE",
                  "Excelente atención, todo funciona perfecto",
                  "Solicito información sobre mi recibo"]:
        print(frase, "->", predecir_sentimiento(frase))


if __name__ == "__main__":
    main()
