"""Análisis: limpieza, perfiles (clustering), modelo de participación y escenarios."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split

SEED = 7
ELECCIONES = (2018, 2021, 2024)
K_RANGO = range(6, 11)  # k candidatos; el silhouette se reporta como referencia
K_FINAL = 9  # decisión de diseño: 9 perfiles accionables (rejilla de 3 × 3)

# ---------------------------------------------------------------------------
# Especificación de variables (compartida con el simulador en JavaScript)
# ---------------------------------------------------------------------------


def _f(nombre, col, kind="num", transform="raw", level=None, grupo=None):
    return dict(name=nombre, var=col, kind=kind, transform=transform, level=level, grupo=grupo or col)


MODELO_SPEC = [
    _f("edad", "edad", grupo="edad"),
    _f("edad²", "edad", transform="sq", grupo="edad"),
    _f("escolaridad", "escolaridad_anios"),
    _f("ingreso (log)", "ingreso_mensual_usd", transform="log"),
    _f("interés político", "interes_politico"),
    _f("confianza institucional", "confianza_instituciones"),
    _f("distancia a la casilla (log)", "distancia_casilla_km", transform="log1p"),
    _f("horas en redes", "horas_redes_dia"),
    _f("contacto de campaña", "contacto_campana"),
    _f("organización civil", "participa_org_civil"),
    _f("mujer", "sexo", kind="dummy", level="Mujer"),
    _f("zona semiurbana", "zona", kind="dummy", level="semiurbana"),
    _f("zona rural", "zona", kind="dummy", level="rural"),
    _f("región Bajío", "region", kind="dummy", level="Bajío"),
    _f("región Centro", "region", kind="dummy", level="Centro"),
    _f("región Golfo", "region", kind="dummy", level="Golfo"),
    _f("región Sur", "region", kind="dummy", level="Sur"),
]

CLUSTER_SPEC = [
    _f("edad", "edad"),
    _f("escolaridad", "escolaridad_anios"),
    _f("ingreso", "ingreso_mensual_usd", transform="log"),
    _f("interes", "interes_politico"),
    _f("confianza", "confianza_instituciones"),
    _f("distancia", "distancia_casilla_km", transform="log1p"),
    _f("redes", "horas_redes_dia"),
    _f("contacto", "contacto_campana"),
    _f("org", "participa_org_civil"),
    _f("historial", "tasa_participacion"),
    _f("zona_rural", "zona", kind="dummy", level="rural"),
]

NOMBRE_VARIABLE = {
    "edad": "Edad", "escolaridad_anios": "Escolaridad", "ingreso_mensual_usd": "Ingreso",
    "interes_politico": "Interés político", "confianza_instituciones": "Confianza institucional",
    "distancia_casilla_km": "Distancia a la casilla", "horas_redes_dia": "Horas en redes",
    "contacto_campana": "Contacto de campañas", "participa_org_civil": "Organización civil",
    "sexo": "Sexo", "zona": "Zona", "region": "Región",
}


def _transformar(x, t):
    if t == "raw":
        return x
    if t == "sq":
        return x ** 2
    if t == "log":
        return np.log(x)
    if t == "log1p":
        return np.log1p(x)
    raise ValueError(t)


def matriz(df, spec):
    cols = []
    for f in spec:
        if f["kind"] == "dummy":
            v = (df[f["var"]] == f["level"]).to_numpy(dtype=float)
        else:
            v = _transformar(df[f["var"]].to_numpy(dtype=float), f["transform"])
        cols.append(v)
    return np.column_stack(cols)


def _estandarizar(spec, X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return [dict(f, mean=float(m), std=float(s)) for f, m, s in zip(spec, mu, sd)], (X - mu) / sd


# ---------------------------------------------------------------------------
# Carga y limpieza
# ---------------------------------------------------------------------------

def cargar(ruta):
    df = pd.read_csv(ruta)
    n_ing, n_conf = int(df["ingreso_mensual_usd"].isna().sum()), int(df["confianza_instituciones"].isna().sum())
    df["ingreso_mensual_usd"] = df.groupby("zona")["ingreso_mensual_usd"].transform(lambda s: s.fillna(s.median()))
    df["confianza_instituciones"] = (
        df.groupby("zona")["confianza_instituciones"].transform(lambda s: s.fillna(s.median())).round().astype(int))
    votos = df[[f"voto_{a}" for a in ELECCIONES]]
    df["n_elegible"] = votos.notna().sum(axis=1)
    df["tasa_participacion"] = votos.mean(axis=1)
    assert df["n_elegible"].min() >= 1 and df[["edad", "ingreso_mensual_usd"]].notna().all().all()

    df["edad_grupo"] = pd.cut(df["edad"], [17, 24, 34, 44, 54, 64, 74, 200],
                              labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"])
    df["nivel_educativo"] = pd.cut(df["escolaridad_anios"], [-1, 6, 9, 12, 16, 40],
                                   labels=["Primaria o menos", "Secundaria", "Media superior", "Superior", "Posgrado"])
    df["ingreso_quintil"] = pd.qcut(df["ingreso_mensual_usd"], 5,
                                    labels=["Q1 (más bajo)", "Q2", "Q3", "Q4", "Q5 (más alto)"])
    df["distancia_grupo"] = pd.cut(df["distancia_casilla_km"], [0, 1, 3, 6, 10, 1000],
                                   labels=["≤1 km", "1–3 km", "3–6 km", "6–10 km", ">10 km"])
    return df, dict(ingreso_imputado=n_ing, confianza_imputada=n_conf)


# ---------------------------------------------------------------------------
# Modelo de participación (regresión logística interpretable)
# ---------------------------------------------------------------------------

def ajustar_modelo(df):
    spec, Xs = _estandarizar(MODELO_SPEC, matriz(df, MODELO_SPEC))
    clf = LogisticRegression(C=10, max_iter=2000).fit(Xs, df["voto_2024"].to_numpy(dtype=int))
    for f, c in zip(spec, clf.coef_[0]):
        f["coef"] = float(c)
    return dict(intercept=float(clf.intercept_[0]), features=spec)


def predecir(modelo, df):
    X = matriz(df, modelo["features"])
    mu = np.array([f["mean"] for f in modelo["features"]])
    sd = np.array([f["std"] for f in modelo["features"]])
    coef = np.array([f["coef"] for f in modelo["features"]])
    z = modelo["intercept"] + ((X - mu) / sd) @ coef
    return 1 / (1 + np.exp(-z))


def evaluar_modelo(df):
    y = df["voto_2024"].to_numpy(dtype=int)
    tr, te = train_test_split(np.arange(len(df)), test_size=0.25, random_state=SEED, stratify=y)
    d_tr, d_te = df.iloc[tr], df.iloc[te]
    modelo = ajustar_modelo(d_tr)
    p_te = predecir(modelo, d_te)
    auc = roc_auc_score(y[te], p_te)

    hgb = HistGradientBoostingClassifier(random_state=SEED).fit(matriz(d_tr, MODELO_SPEC), y[tr])
    auc_hgb = roc_auc_score(y[te], hgb.predict_proba(matriz(d_te, MODELO_SPEC))[:, 1])

    # Calibración por deciles de probabilidad
    dec = pd.qcut(p_te, 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"pred": p_te, "real": y[te], "dec": dec}).groupby("dec").mean()
    err_cal = float((cal["pred"] - cal["real"]).abs().max())

    # Importancia por permutación a nivel de variable original
    rng = np.random.default_rng(SEED)
    imp = {}
    for var in NOMBRE_VARIABLE:
        caidas = []
        for _ in range(10):
            d_perm = d_te.copy()
            d_perm[var] = rng.permutation(d_perm[var].to_numpy())
            caidas.append(auc - roc_auc_score(y[te], predecir(modelo, d_perm)))
        imp[var] = float(np.mean(caidas))
    importancia = pd.Series(imp).sort_values(ascending=False)
    return dict(auc=float(auc), auc_hgb=float(auc_hgb), err_calibracion=err_cal,
                calibracion=cal.reset_index(drop=True), importancia=importancia)


# ---------------------------------------------------------------------------
# Segmentación en perfiles
# ---------------------------------------------------------------------------

def segmentar(df, verdad=None):
    spec, X = _estandarizar(CLUSTER_SPEC, matriz(df, CLUSTER_SPEC))
    sils = {}
    modelos = {}
    for k in K_RANGO:
        km = KMeans(k, n_init=10, random_state=SEED).fit(X)
        modelos[k] = km
        sils[k] = float(silhouette_score(X, km.labels_, sample_size=6000, random_state=SEED))
    k = K_FINAL
    km = modelos[k]

    # Estabilidad: re-ajusta con submuestras del 80 % y compara contra la partición final
    rng = np.random.default_rng(SEED)
    aris = []
    for s in range(5):
        idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
        sub = KMeans(k, n_init=5, random_state=100 + s).fit(X[idx])
        aris.append(adjusted_rand_score(km.labels_, sub.predict(X)))

    # Numeración: perfil 1 = mayor participación en 2024
    tasa24 = pd.Series(df["voto_2024"].to_numpy(dtype=float)).groupby(km.labels_).mean()
    orden = list(tasa24.sort_values(ascending=False).index)
    mapa = {c: i + 1 for i, c in enumerate(orden)}
    perfil = np.array([mapa[c] for c in km.labels_])
    centroides = km.cluster_centers_[orden]

    ari_verdad = None
    if verdad is not None:
        ari_verdad = float(adjusted_rand_score(verdad, perfil))

    pca = PCA(n_components=2, random_state=SEED).fit(X)
    return dict(k=k, spec=spec, X=X, perfil=perfil, centroides=centroides, silhouette=sils,
                estabilidad=float(np.mean(aris)), ari_verdad=ari_verdad, pca=pca,
                pca_xy=pca.transform(X), pca_var=pca.explained_variance_ratio_)


# Frases para nombrar perfiles según el rasgo más distintivo (z de centroide)
FRASES = {
    ("interes", 1): "muy interesados en política", ("interes", -1): "poco interesados en política",
    ("confianza", 1): "con alta confianza institucional", ("confianza", -1): "desconfiados de las instituciones",
    ("distancia", 1): "lejos de su casilla", ("distancia", -1): "cerca de su casilla",
    ("redes", 1): "hiperconectados", ("redes", -1): "poco conectados a internet",
    ("contacto", 1): "contactados por campañas", ("contacto", -1): "sin contacto de campañas",
    ("org", 1): "organizados en la sociedad civil",
    ("escolaridad", 1): "con estudios superiores", ("escolaridad", -1): "con baja escolaridad",
    ("ingreso", 1): "de ingresos altos", ("ingreso", -1): "de ingresos bajos",
    ("zona_rural", 1): "de zonas rurales",
}


def _nombre_edad(edad):
    if edad < 30:
        return "Jóvenes"
    if edad < 50:
        return "Adultos"
    if edad < 65:
        return "Adultos maduros"
    return "Personas mayores"


def nombrar_perfiles(z, edades):
    """Nombres únicos a partir del rasgo (o los dos rasgos) más distintivo de cada centroide."""
    idx = {f["name"]: i for i, f in enumerate(CLUSTER_SPEC)}
    nombres = []
    for c in range(len(z)):
        cand = []
        for (var, signo), frase in FRASES.items():
            val = z[c][idx[var]] * signo
            if val > 0.35:
                cand.append((val, frase))
        cand.sort(reverse=True)
        frases = [f for _, f in cand] or ["con rasgos intermedios"]
        nom = f"{_nombre_edad(edades[c])} {frases[0]}"
        if nom in nombres and len(frases) > 1:
            nom = f"{_nombre_edad(edades[c])} {frases[0]} y {frases[1]}"
        n = 2
        base = nom
        while nom in nombres:
            nom, n = f"{base} ({n})", n + 1
        nombres.append(nom)
    return nombres


def perfilar(df, seg):
    df = df.assign(perfil=seg["perfil"])
    z = seg["centroides"]
    k = seg["k"]
    g = df.groupby("perfil")
    media = g.agg(edad=("edad", "mean"), escolaridad=("escolaridad_anios", "mean"),
                  ingreso=("ingreso_mensual_usd", "median"), interes=("interes_politico", "mean"),
                  confianza=("confianza_instituciones", "mean"), distancia=("distancia_casilla_km", "median"),
                  redes=("horas_redes_dia", "mean"), contacto=("contacto_campana", "mean"),
                  org=("participa_org_civil", "mean"), historial=("tasa_participacion", "mean"),
                  v2018=("voto_2018", "mean"), v2021=("voto_2021", "mean"), v2024=("voto_2024", "mean"),
                  n=("id", "count"))
    zonas = pd.crosstab(df["perfil"], df["zona"], normalize="index")
    ocup = df.groupby("perfil")["ocupacion"].agg(lambda s: s.value_counts().index[0])
    nombres = nombrar_perfiles(z, media["edad"].to_numpy())
    prom = float(df["voto_2024"].mean())
    tasa_global = float(df["tasa_participacion"].mean())

    perfiles = []
    for c in range(k):
        m = media.iloc[c]
        pid = c + 1
        barreras, fortalezas = [], []
        if m["distancia"] >= 4: barreras.append("Lejos de la casilla")
        if m["confianza"] < 2.5: barreras.append("Baja confianza institucional")
        if m["interes"] < 2.7: barreras.append("Bajo interés político")
        if m["contacto"] < 0.25: barreras.append("Poco contacto de campañas")
        if m["redes"] < 1.5: barreras.append("Poco acceso digital")
        if m["escolaridad"] < 8: barreras.append("Baja escolaridad")
        if m["edad"] < 30: barreras.append("Primer voto / juventud")
        if m["ingreso"] < 250: barreras.append("Ingresos bajos")
        if m["interes"] >= 3.8: fortalezas.append("Alto interés político")
        if m["confianza"] >= 3.5: fortalezas.append("Alta confianza institucional")
        if m["org"] >= 0.5: fortalezas.append("Vida organizada")
        if m["contacto"] >= 0.5: fortalezas.append("Muy contactados")
        if m["redes"] >= 4: fortalezas.append("Muy conectados")
        if m["historial"] >= tasa_global + 0.1: fortalezas.append("Hábito de votar")
        nivel = ("alta" if m["historial"] >= tasa_global + 0.10
                 else "baja" if m["historial"] <= tasa_global - 0.10 else "media")
        perfiles.append(dict(
            id=pid, nombre=nombres[c], n=int(m["n"]), share=float(m["n"] / len(df)),
            v2018=float(m["v2018"]), v2021=float(m["v2021"]), v2024=float(m["v2024"]),
            brecha=float(m["v2024"] - prom), nivel=nivel,
            edad=float(m["edad"]), escolaridad=float(m["escolaridad"]), ingreso=float(m["ingreso"]),
            interes=float(m["interes"]), confianza=float(m["confianza"]), distancia=float(m["distancia"]),
            redes=float(m["redes"]), contacto=float(m["contacto"]), org=float(m["org"]),
            historial=float(m["historial"]), rural=float(zonas.loc[pid].get("rural", 0.0)),
            ocupacion=str(ocup.loc[pid]), barreras=barreras, fortalezas=fortalezas,
            z={f["name"]: float(z[c][i]) for i, f in enumerate(CLUSTER_SPEC)}))
    return df, perfiles


# ---------------------------------------------------------------------------
# Escenarios de intervención
# ---------------------------------------------------------------------------

LEVERS = [
    dict(id="distancia", nombre="Acercar la casilla", detalle="Distancia a la casilla ≤ 1 km",
         f=lambda d: d.assign(distancia_casilla_km=d["distancia_casilla_km"].clip(upper=1.0))),
    dict(id="contacto", nombre="Contacto personalizado", detalle="Todas las personas reciben contacto directo",
         f=lambda d: d.assign(contacto_campana=1)),
    dict(id="interes", nombre="Educación cívica", detalle="+0.5 puntos de interés político (máx. 5)",
         f=lambda d: d.assign(interes_politico=(d["interes_politico"] + 0.5).clip(upper=5))),
    dict(id="confianza", nombre="Confianza institucional", detalle="+0.5 puntos de confianza (máx. 5)",
         f=lambda d: d.assign(confianza_instituciones=(d["confianza_instituciones"] + 0.5).clip(upper=5))),
    dict(id="org", nombre="Vinculación comunitaria", detalle="Participación en una organización civil",
         f=lambda d: d.assign(participa_org_civil=1)),
]


def accion_recomendada(lever_id, p):
    if lever_id == "distancia":
        if p["edad"] >= 60:
            return "Traslado asistido y casillas especiales cerca del domicilio"
        if p["rural"] > 0.4:
            return "Transporte comunitario gratuito el día de la elección y casillas móviles"
        return "Más casillas de proximidad y voto anticipado en puntos cercanos"
    if lever_id == "contacto":
        if p["redes"] >= 3:
            return "Recordatorios personalizados por WhatsApp/SMS con lugar y horario de su casilla"
        return "Visitas domiciliarias y llamadas con la ubicación de su casilla"
    if lever_id == "interes":
        if p["redes"] >= 3:
            return "Contenido corto en redes con creadores locales y dinámicas de educación cívica"
        return "Radio comunitaria y talleres presenciales de educación cívica"
    if lever_id == "confianza":
        return "Transparencia del conteo, observación ciudadana y rendición de cuentas visibles"
    return "Alianzas con organizaciones, colectivos y asociaciones locales para movilizar"


def escenarios(df, modelo, k):
    base = predecir(modelo, df)
    perf = df["perfil"].to_numpy()
    matriz_pp, glob = [], []
    cols = []
    for lv in LEVERS:
        delta = predecir(modelo, lv["f"](df)) - base
        cols.append(pd.Series(delta).groupby(perf).mean().reindex(range(1, k + 1)).to_numpy() * 100)
        glob.append(float(delta.mean() * 100))
    matriz_pp = np.column_stack(cols)  # perfil x palanca, puntos porcentuales con cobertura 100 %
    return dict(matriz=matriz_pp, global_pp=glob, prob_base=base)


# ---------------------------------------------------------------------------
# Hallazgos descriptivos
# ---------------------------------------------------------------------------

def hallazgos(df, perfiles):
    t = lambda m: float(df.loc[m, "voto_2024"].mean())
    out = []
    por_edad = df.groupby("edad_grupo", observed=True)["voto_2024"].mean()
    lo, hi = por_edad.idxmin(), por_edad.idxmax()
    out.append(dict(valor=f"{(por_edad.max() - por_edad.min()) * 100:.0f} pp", titulo="Brecha generacional",
                    texto=f"El grupo de {lo} años participó {por_edad.min():.0%} frente a {por_edad.max():.0%} "
                          f"del grupo de {hi}."))
    d_cerca, d_lejos = t(df["distancia_casilla_km"] <= 1), t(df["distancia_casilla_km"] > 10)
    out.append(dict(valor=f"{(d_cerca - d_lejos) * 100:.0f} pp", titulo="Costo de la distancia",
                    texto=f"Votó {d_cerca:.0%} de quienes viven a ≤1 km de su casilla, contra {d_lejos:.0%} "
                          f"de quienes viven a más de 10 km."))
    c1, c0 = t(df["contacto_campana"] == 1), t(df["contacto_campana"] == 0)
    out.append(dict(valor=f"{(c1 - c0) * 100:.0f} pp", titulo="Efecto del contacto",
                    texto=f"Participó {c1:.0%} de quienes fueron contactados por campañas y {c0:.0%} de quienes no."))
    e_lo, e_hi = t(df["nivel_educativo"] == "Primaria o menos"), t(df["nivel_educativo"].isin(["Superior", "Posgrado"]))
    out.append(dict(valor=f"{(e_hi - e_lo) * 100:+.0f} pp", titulo="Educación y voto",
                    texto=f"Con estudios superiores o posgrado votó {e_hi:.0%}; con primaria o menos, {e_lo:.0%}."))
    ambos = df[(df["voto_2018"].notna()) & (df["voto_2021"].notna())]
    h1 = ambos.loc[(ambos["voto_2018"] == 1) & (ambos["voto_2021"] == 1), "voto_2024"].mean()
    h0 = ambos.loc[(ambos["voto_2018"] == 0) & (ambos["voto_2021"] == 0), "voto_2024"].mean()
    out.append(dict(valor=f"{h1:.0%} vs {h0:.0%}", titulo="El hábito pesa",
                    texto="Participación 2024 de quienes votaron en 2018 y 2021 frente a quienes no votaron en ninguna."))
    mejor, peor = perfiles[0], perfiles[-1]
    out.append(dict(valor=f"{(mejor['v2024'] - peor['v2024']) * 100:.0f} pp", titulo="Brecha entre perfiles",
                    texto=f"Perfil 1 ({mejor['v2024']:.0%}) frente al perfil {peor['id']} ({peor['v2024']:.0%})."))
    return out


def tabla_cruzada(df, filas, cols):
    return (df.pivot_table(index=filas, columns=cols, values="voto_2024", aggfunc="mean", observed=True) * 100)


def ejecutar(ruta_csv, ruta_verdad=None):
    df, imputados = cargar(ruta_csv)
    verdad = None
    if ruta_verdad and Path(ruta_verdad).exists():
        v = pd.read_csv(ruta_verdad)
        assert (v["id"].to_numpy() == df["id"].to_numpy()).all()
        verdad = v["arquetipo"].to_numpy()
    seg = segmentar(df, verdad)
    df, perfiles = perfilar(df, seg)
    ev = evaluar_modelo(df)
    modelo = ajustar_modelo(df)
    df["prob_participacion"] = predecir(modelo, df)
    esc = escenarios(df, modelo, seg["k"])

    for i, p in enumerate(perfiles):
        orden = np.argsort(-esc["matriz"][i])[:3]
        p["estrategias"] = [dict(id=LEVERS[j]["id"], nombre=LEVERS[j]["nombre"], pp=float(esc["matriz"][i][j]),
                                 accion=accion_recomendada(LEVERS[j]["id"], p)) for j in orden]
    return dict(df=df, imputados=imputados, seg=seg, perfiles=perfiles, ev=ev, modelo=modelo, esc=esc,
                hallazgos=hallazgos(df, perfiles))
