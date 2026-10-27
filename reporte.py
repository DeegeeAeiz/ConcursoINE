"""Construye el tablero interactivo (HTML autocontenido) con los resultados del análisis."""
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

import analisis as A
from generar_datos import REGIONES, ZONAS

RAIZ = Path(__file__).parent
FUENTE = '"Plus Jakarta Sans", "Segoe UI Variable", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif'
TINTA, TINTA2, MUTED, LINEA = "#3b1f5c", "#594a70", "#6b5f80", "#ddd5e8"
ACENTO, POS, NEG = "#6250d6", "#2a78d6", "#d95926"
# Categórica fija (validada): un color por perfil, en orden y sin reciclar
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#5f6b7a"]
SECUENCIAL = [[0, "#cde2fb"], [0.3, "#86b6ef"], [0.6, "#2a78d6"], [1, "#0d366b"]]
DIVERGENTE = [[0, NEG], [0.5, "#f3f1f0"], [1, "#256abf"]]
CFG = dict(displayModeBar=False, responsive=True)


def _tinta_sobre(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#2b1646" if lum > 0.42 else "#ffffff"


def _html(fig, h, div_id=None):
    return fig.to_html(full_html=False, include_plotlyjs=False, config=CFG, default_height=f"{h}px",
                       default_width="100%", div_id=div_id)


def _base(fig, h, l=48, r=16, t=8, b=44, legend=False):
    fig.update_layout(
        height=h, margin=dict(l=l, r=r, t=t, b=b), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FUENTE, size=13, color=TINTA2), showlegend=legend,
        hoverlabel=dict(bgcolor="#fff", bordercolor=LINEA, font=dict(family=FUENTE, size=13, color=TINTA)),
        legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=11)),
    )
    fig.update_xaxes(linecolor=LINEA, showgrid=False, ticks="")
    fig.update_yaxes(gridcolor=LINEA, zeroline=False, ticks="")
    return fig


def _tasa_por(df, col):
    g = df.groupby(col, observed=True)["voto_2024"].agg(["mean", "count"])
    return g.index.astype(str).tolist(), (g["mean"] * 100).tolist(), g["count"].tolist()


def _envolver(txt):
    """Parte etiquetas largas en dos líneas por el espacio más cercano al centro."""
    if len(txt) <= 11 or " " not in txt:
        return txt
    cortes = [i for i, c in enumerate(txt) if c == " "]
    i = min(cortes, key=lambda j: abs(j - len(txt) / 2))
    return txt[:i] + "<br>" + txt[i + 1:]


def fig_barras(etiquetas, valores, ns, prom, h=290, rot=0):
    fig = go.Figure(go.Bar(
        x=[_envolver(e) for e in etiquetas], y=valores, customdata=ns, marker=dict(color=ACENTO, cornerradius=4),
        text=[f"{v:.0f}%" for v in valores], textposition="outside", cliponaxis=False, textfont=dict(color=TINTA),
        hovertemplate="%{x}<br><b>%{y:.1f}%</b> votó en 2024<br>n = %{customdata:,}<extra></extra>"))
    fig.add_hline(y=prom, line=dict(color=MUTED, dash="dot", width=1.5))
    _base(fig, h, l=40, t=14)
    fig.update_yaxes(range=[0, 112], tickvals=[0, 25, 50, 75, 100], ticksuffix="%")
    fig.update_xaxes(tickangle=rot)
    return fig


def fig_perfiles_participacion(perfiles, prom):
    et = [f"P{p['id']} · {p['nombre']}" for p in perfiles]
    fig = go.Figure(go.Bar(
        y=et, x=[p["v2024"] * 100 for p in perfiles], orientation="h",
        marker=dict(color=[COLORES[i] for i in range(len(perfiles))], cornerradius=4),
        text=[f"{p['v2024']:.0%}" for p in perfiles], textposition="outside", cliponaxis=False,
        textfont=dict(color=TINTA), customdata=[p["n"] for p in perfiles],
        hovertemplate="%{y}<br><b>%{x:.1f}%</b> votó en 2024<br>%{customdata:,} personas<extra></extra>"))
    fig.add_vline(x=prom * 100, line=dict(color=MUTED, dash="dot", width=1.5))
    _base(fig, 60 + 46 * len(perfiles), l=300, r=40)
    fig.update_xaxes(range=[0, 108], ticksuffix="%", showgrid=True, gridcolor=LINEA)
    fig.update_yaxes(autorange="reversed", showgrid=False, automargin=True)
    return fig


def fig_tendencia(perfiles):
    fig = go.Figure()
    for i, p in enumerate(perfiles):
        fig.add_trace(go.Scatter(
            x=["2018", "2021", "2024"], y=[p["v2018"] * 100, p["v2021"] * 100, p["v2024"] * 100],
            mode="lines+markers", name=f"P{p['id']}", line=dict(color=COLORES[i], width=2.5),
            marker=dict(size=9, color=COLORES[i], line=dict(color="#fff", width=2)),
            hovertemplate=f"<b>P{p['id']} · {html.escape(p['nombre'])}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>"))
    _base(fig, 380, l=44, legend=True)
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    fig.update_layout(legend=dict(orientation="h", y=-0.12, x=0, font=dict(size=12)))
    return fig


def fig_pca(R, n=3500):
    seg, df = R["seg"], R["df"]
    rng = np.random.default_rng(1)
    idx = rng.choice(len(df), n, replace=False)
    xy, perfil = seg["pca_xy"][idx], seg["perfil"][idx]
    fig = go.Figure()
    for i, p in enumerate(R["perfiles"]):
        m = perfil == p["id"]
        fig.add_trace(go.Scatter(
            x=xy[m, 0], y=xy[m, 1], mode="markers", name=f"P{p['id']}",
            marker=dict(size=6, color=COLORES[i], opacity=0.55),
            hovertemplate=f"<b>P{p['id']} · {html.escape(p['nombre'])}</b><extra></extra>"))
    v = seg["pca_var"]
    _base(fig, 380, l=44, legend=True)
    fig.update_xaxes(title=dict(text=f"Componente 1 ({v[0]:.0%} de la varianza)", font=dict(size=12)),
                     zeroline=False, showgrid=True, gridcolor=LINEA)
    fig.update_yaxes(title=dict(text=f"Componente 2 ({v[1]:.0%})", font=dict(size=12)))
    fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=12), itemsizing="constant"))
    return fig


RASGOS = [("edad", "Edad", lambda p: f"{p['edad']:.0f}"),
          ("escolaridad", "Escolaridad", lambda p: f"{p['escolaridad']:.1f} a"),
          ("ingreso", "Ingreso", lambda p: f"${p['ingreso']:,.0f}"),
          ("interes", "Interés político", lambda p: f"{p['interes']:.1f}"),
          ("confianza", "Confianza inst.", lambda p: f"{p['confianza']:.1f}"),
          ("distancia", "Dist. a casilla", lambda p: f"{p['distancia']:.1f} km"),
          ("redes", "Horas en redes", lambda p: f"{p['redes']:.1f} h"),
          ("contacto", "Contacto camp.", lambda p: f"{p['contacto']:.0%}"),
          ("org", "Org. civil", lambda p: f"{p['org']:.0%}"),
          ("historial", "Historial de voto", lambda p: f"{p['historial']:.0%}"),
          ("zona_rural", "Zona rural", lambda p: f"{p['rural']:.0%}")]


def fig_rasgos(perfiles):
    z = [[np.clip(p["z"][k], -2, 2) for k, _, _ in RASGOS] for p in perfiles]
    txt = [[f(p) for _, _, f in RASGOS] for p in perfiles]
    y = [f"P{p['id']} · {p['nombre']}" for p in perfiles]
    fig = go.Figure(go.Heatmap(
        z=z, x=[n for _, n, _ in RASGOS], y=y, text=txt, texttemplate="%{text}", colorscale=DIVERGENTE,
        zmin=-2, zmax=2, zmid=0, xgap=3, ygap=3,
        colorbar=dict(title=dict(text="vs. promedio", side="right", font=dict(size=12)), thickness=12, len=0.9,
                      tickvals=[-2, 0, 2], ticktext=["menor", "promedio", "mayor"]),
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>"))
    _base(fig, 120 + 44 * len(perfiles), l=300, r=10, b=70)
    fig.update_yaxes(autorange="reversed", automargin=True, showgrid=False)
    fig.update_xaxes(side="bottom", tickangle=-25)
    return fig


def fig_silhouette(seg):
    ks = list(seg["silhouette"])
    fig = go.Figure(go.Bar(
        x=[f"k = {k}" for k in ks], y=[seg["silhouette"][k] for k in ks],
        marker=dict(color=[ACENTO if k == seg["k"] else "#c5bef2" for k in ks], cornerradius=4),
        text=[f"{seg['silhouette'][k]:.3f}" for k in ks], textposition="outside", textfont=dict(color=TINTA),
        hovertemplate="%{x}<br>silhouette %{y:.3f}<extra></extra>"))
    _base(fig, 230, l=40, t=14)
    fig.update_yaxes(range=[0, max(seg["silhouette"].values()) * 1.25])
    return fig


def fig_importancia(imp):
    imp = imp.clip(lower=0).sort_values()
    fig = go.Figure(go.Bar(
        y=[A.NOMBRE_VARIABLE[v] for v in imp.index], x=imp.values * 100, orientation="h",
        marker=dict(color=ACENTO, cornerradius=4), text=[f"{v * 100:.1f}" for v in imp.values],
        textposition="outside", cliponaxis=False, textfont=dict(color=TINTA),
        hovertemplate="%{y}<br>Caída del AUC al barajar: %{x:.2f} puntos<extra></extra>"))
    _base(fig, 380, l=150, r=40)
    fig.update_xaxes(showgrid=True, gridcolor=LINEA, range=[0, imp.max() * 100 * 1.15])
    return fig


def fig_calor(tabla, ns, h, l=130, rot=0):
    z = tabla.where(ns >= 30)
    txt = z.map(lambda v: "" if pd.isna(v) else f"{v:.0f}%")
    fig = go.Figure(go.Heatmap(
        z=z.values, x=[str(c) for c in z.columns], y=[str(i) for i in z.index], colorscale=SECUENCIAL,
        zmin=0, zmax=100, xgap=3, ygap=3, text=txt.values, texttemplate="%{text}", showscale=False,
        customdata=ns.values,
        hovertemplate="%{y} · %{x}<br><b>%{z:.1f}%</b> votó en 2024<br>n = %{customdata:,}<extra></extra>"))
    _base(fig, h, l=l, r=10, b=50)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickangle=rot)
    return fig


def _cruce(df, filas, cols):
    t = df.pivot_table(index=filas, columns=cols, values="voto_2024", aggfunc="mean", observed=True) * 100
    n = df.pivot_table(index=filas, columns=cols, values="voto_2024", aggfunc="count", observed=True).fillna(0)
    return t, n


# ---------------------------------------------------------------------------

def _tarjeta_perfil(p, i):
    color = COLORES[i]
    brecha = p["brecha"] * 100
    clase = "delta-pos" if brecha >= 0 else "delta-neg"
    datos = [("Edad", f"{p['edad']:.0f} años"), ("Escolaridad", f"{p['escolaridad']:.1f} años"),
             ("Ingreso/mes", f"${p['ingreso']:,.0f}"), ("Distancia", f"{p['distancia']:.1f} km"),
             ("Interés (1-5)", f"{p['interes']:.1f}"), ("Confianza (1-5)", f"{p['confianza']:.1f}"),
             ("Redes", f"{p['redes']:.1f} h/día"), ("Contacto camp.", f"{p['contacto']:.0%}"),
             ("Org. civil", f"{p['org']:.0%}"), ("Zona rural", f"{p['rural']:.0%}")]
    dl = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in datos)
    tags = "".join(f'<span class="tag bar">{html.escape(b)}</span>' for b in p["barreras"]) or \
        '<span class="tag fort">Sin barreras relevantes</span>'
    fort = "".join(f'<span class="tag fort">{html.escape(f)}</span>' for f in p["fortalezas"])
    est = "".join(
        f'<li><b>{html.escape(e["nombre"])}</b> <em data-pp="{e["pp"]:.2f}">hasta +{e["pp"]:.1f} pp</em><br>'
        f'{html.escape(e["accion"])}</li>' for e in p["estrategias"])
    return f"""
<article class="perfil" style="--c:{color}">
  <header><span class="pid" style="color:{_tinta_sobre(color)}">P{p['id']}</span><h3>{html.escape(p['nombre'])}</h3></header>
  <div class="turnout"><strong>{p['v2024']:.0%}</strong><span>votó en 2024 · <span class="{clase}">{brecha:+.0f} pp</span> vs. promedio</span></div>
  <div class="sizebar"><i style="width:{p['share'] * 100:.1f}%"></i></div>
  <small>{p['share']:.1%} de la población · {p['n']:,} personas · ocupación más común: {html.escape(p['ocupacion'].lower())}</small>
  <dl class="datos">{dl}</dl>
  <h4>Barreras</h4><div class="tags">{tags}</div>
  {f'<h4>Fortalezas</h4><div class="tags">{fort}</div>' if fort else ''}
  <h4>Estrategias con mayor impacto estimado</h4>
  <ul class="estrat">{est}</ul>
</article>"""


def _tipicos(R):
    df = R["df"]
    out = []
    for p in R["perfiles"]:
        d = df[df["perfil"] == p["id"]]
        out.append(dict(
            edad=int(round(d["edad"].median())), sexo=d["sexo"].mode().iloc[0], region=d["region"].mode().iloc[0],
            zona=d["zona"].mode().iloc[0], escolaridad_anios=int(round(d["escolaridad_anios"].median())),
            ingreso_mensual_usd=int(round(d["ingreso_mensual_usd"].median(), -1)),
            interes_politico=int(round(p["interes"])), confianza_instituciones=int(round(p["confianza"])),
            distancia_casilla_km=round(float(d["distancia_casilla_km"].median()), 1),
            horas_redes_dia=round(p["redes"], 1), contacto_campana=int(round(p["contacto"])),
            participa_org_civil=int(round(p["org"])), tasa_participacion=round(p["historial"], 2)))
    return out


def _controles_simulador(t):
    def rango(var, etiqueta, lo, hi, paso, val):
        return (f'<label>{etiqueta}<span class="fila"><input type="range" data-var="{var}" min="{lo}" max="{hi}" '
                f'step="{paso}" value="{val}"><output id="o-{var}">{val}</output></span></label>')

    def sel(var, etiqueta, opciones, val):
        ops = "".join(f'<option value="{o}"{" selected" if o == val else ""}>{o.capitalize()}</option>' for o in opciones)
        return f'<label>{etiqueta}<select data-var="{var}">{ops}</select></label>'

    def chk(var, etiqueta):
        return f'<label class="chk"><input type="checkbox" data-var="{var}">{etiqueta}</label>'

    return "".join([
        rango("edad", "Edad", 18, 90, 1, t["edad"]),
        sel("sexo", "Sexo", ["Mujer", "Hombre"], t["sexo"]),
        sel("region", "Región", REGIONES, t["region"]),
        sel("zona", "Zona de residencia", ZONAS, t["zona"]),
        rango("escolaridad_anios", "Años de escolaridad", 0, 20, 1, t["escolaridad_anios"]),
        rango("ingreso_mensual_usd", "Ingreso mensual (USD)", 60, 6000, 10, t["ingreso_mensual_usd"]),
        rango("interes_politico", "Interés político (1-5)", 1, 5, 1, t["interes_politico"]),
        rango("confianza_instituciones", "Confianza en instituciones (1-5)", 1, 5, 1, t["confianza_instituciones"]),
        rango("distancia_casilla_km", "Distancia a la casilla (km)", 0.1, 30, 0.1, t["distancia_casilla_km"]),
        rango("horas_redes_dia", "Horas diarias en redes", 0, 12, 0.1, t["horas_redes_dia"]),
        chk("contacto_campana", "Ha sido contactado por campañas"),
        chk("participa_org_civil", "Participa en una organización civil"),
        rango("tasa_participacion", "Historial de voto (solo para asignar perfil)", 0, 100, 1,
              int(round(t["tasa_participacion"] * 100))),
    ])


def _cinta_hero(perfiles, prom):
    """Cinta SVG: ancho = tamaño del perfil, altura = participación 2024."""
    k, ancho, hueco, base, alto = len(perfiles), 1000, 8, 214, 185
    util = ancho - hueco * (k - 1)
    total = sum(p["share"] for p in perfiles)
    x, barras = 0.0, []
    for i, p in enumerate(perfiles):
        w = p["share"] / total * util
        h = p["v2024"] * alto
        cx = x + w / 2
        nombre = html.escape(p["nombre"])
        barras.append(
            f'<g><rect class="rb" style="animation-delay:{i * 90}ms" x="{x:.1f}" y="{base - h:.1f}" width="{w:.1f}" '
            f'height="{h:.1f}" rx="12" fill="{COLORES[i]}"><title>P{p["id"]} · {nombre}: {p["v2024"]:.0%} votó en 2024 · '
            f'{p["share"]:.1%} de la población</title></rect>'
            f'<text x="{cx:.1f}" y="{base - h - 10:.1f}" text-anchor="middle" fill="#fff" stroke="#3a1d63" stroke-width="6" stroke-linejoin="round" paint-order="stroke" font-size="21" font-weight="800">{p["v2024"]:.0%}</text>'
            f'<text x="{cx:.1f}" y="{base + 24}" text-anchor="middle" fill="#e3d9fa" font-size="15" font-weight="700">P{p["id"]}</text></g>')
        x += w + hueco
    y_prom = base - prom * alto
    linea = (f'<line x1="0" x2="{ancho}" y1="{y_prom:.1f}" y2="{y_prom:.1f}" stroke="#fff" stroke-opacity=".75" '
             f'stroke-width="1.5" stroke-dasharray="3 6"/>'
             f'<text x="{ancho}" y="{y_prom - 9:.1f}" text-anchor="end" fill="#fff" font-size="14" font-weight="700">'
             f'Promedio {prom:.0%}</text>')
    return (f'<svg viewBox="0 0 {ancho} 252" role="img" aria-label="Participación 2024 por perfil: ancho proporcional al '
            f'tamaño del perfil, altura proporcional a la participación">{"".join(barras)}{linea}</svg>')


def _json(obj):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    return json.dumps(obj, ensure_ascii=False, default=conv).replace("</", "<\\/")


def construir(R, ruta_salida):
    df, perfiles, seg, ev, esc = R["df"], R["perfiles"], R["seg"], R["ev"], R["esc"]
    prom = float(df["voto_2024"].mean())
    n = len(df)
    k = seg["k"]

    # --- Panorama
    lab, val, ns = _tasa_por(df.assign(e=df["voto_2024"]), "edad_grupo")
    graficas = {
        "Participación por elección": (
            "Entre quienes eran elegibles (≥ 18 años) en cada fecha.",
            fig_barras(["2018", "2021", "2024"],
                       [df[f"voto_{a}"].mean() * 100 for a in A.ELECCIONES],
                       [int(df[f"voto_{a}"].notna().sum()) for a in A.ELECCIONES], prom * 100)),
        "Por grupo de edad": ("Los jóvenes votan mucho menos que los mayores.", fig_barras(lab, val, ns, prom * 100)),
        "Por nivel educativo": ("", fig_barras(*_tasa_por(df, "nivel_educativo"), prom * 100)),
        "Por quintil de ingreso": ("", fig_barras(*_tasa_por(df, "ingreso_quintil"), prom * 100)),
        "Por distancia a la casilla": ("Cada kilómetro de traslado cuesta votos.",
                                       fig_barras(*_tasa_por(df, "distancia_grupo"), prom * 100)),
        "Por región": ("", fig_barras(*_tasa_por(df, "region"), prom * 100)),
    }
    panorama = "".join(
        f'<div class="card"><h3>{t}</h3><div class="sub">{s or "Participación 2024 (%)"}</div>{_html(f, 290)}</div>'
        for t, (s, f) in graficas.items())

    hallazgos = "".join(
        f'<div class="hallazgo"><div class="v">{h["valor"]}</div><div class="t">{h["titulo"]}</div>'
        f'<p>{h["texto"]}</p></div>' for h in R["hallazgos"])

    tarjetas = "".join(_tarjeta_perfil(p, i) for i, p in enumerate(perfiles))

    # --- Factores
    t_edu, n_edu = _cruce(df, "nivel_educativo", "edad_grupo")
    t_reg, n_reg = _cruce(df, "region", "zona")
    t_reg = t_reg[[z for z in ZONAS if z in t_reg.columns]]
    n_reg = n_reg[t_reg.columns]

    tipicos = _tipicos(R)
    inicial = len(perfiles) - 1
    presets = "".join(
        f'<button style="--c:{COLORES[i]}" data-tipico="{i}">P{p["id"]}</button>' for i, p in enumerate(perfiles))

    nombre_var = dict(A.NOMBRE_VARIABLE)
    datos_js = dict(
        modelo=R["modelo"],
        cluster=dict(features=seg["spec"], centroides=seg["centroides"]),
        perfiles=[dict(id=p["id"], nombre=p["nombre"], color=COLORES[i], share=p["share"], n=p["n"], v2024=p["v2024"])
                  for i, p in enumerate(perfiles)],
        palancas=[dict(id=lv["id"], nombre=lv["nombre"], detalle=lv["detalle"]) for lv in A.LEVERS],
        matriz=esc["matriz"], global_pp=esc["global_pp"], promedio=prom, nombreVariable=nombre_var,
        tipicos=tipicos, tipicoInicial=inicial,
        estilo=dict(fuente=FUENTE, tinta=TINTA, tinta2=TINTA2, linea=LINEA, acento=ACENTO, pos=POS, neg=NEG,
                    secuencial=SECUENCIAL))

    detalle_palancas = "".join(f"<li><b>{lv['nombre']}:</b> {lv['detalle']}.</li>" for lv in A.LEVERS)
    gap = (perfiles[0]["v2024"] - perfiles[-1]["v2024"]) * 100
    css = (RAIZ / "static" / "estilo.css").read_text(encoding="utf-8")
    js = (RAIZ / "static" / "app.js").read_text(encoding="utf-8")
    sil_txt = ", ".join(f"k={kk}: {v:.3f}" for kk, v in seg["silhouette"].items())
    imp_ing, imp_conf = R["imputados"]["ingreso_imputado"], R["imputados"]["confianza_imputada"]

    pagina = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Perfiles ciudadanos de participación</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>{css}</style>
<script>{get_plotlyjs()}</script></head><body>
<nav class="nav"><div class="wrap">
<span class="marca"><svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true"><circle cx="11" cy="11" r="10" fill="none" stroke="#c9b8ff" stroke-width="2"/><circle cx="11" cy="11" r="5.5" fill="none" stroke="#ffb38a" stroke-width="2"/><circle cx="11" cy="11" r="1.8" fill="#fff"/></svg>Perfiles ciudadanos</span>
<a href="#panorama">Panorama</a><a href="#perfiles">Perfiles</a><a href="#factores">Factores</a>
<a href="#estrategias">Estrategias</a><a href="#simulador">Simulador</a><a href="#metodo">Metodología</a></div></nav>

<header class="hero"><div class="wrap">
  <span class="eyebrow"><i></i>Hackatón · Participación electoral</span>
  <h1>Perfiles ciudadanos<br>de <span class="grad">participación</span></h1>
  <p class="lead">Identificamos patrones de comportamiento electoral y construimos {k} perfiles que explican
  cómo participan distintos grupos de la población y qué estrategias podrían incentivar una mayor participación.</p>
  <div class="cta"><a class="btn primario" href="#perfiles">Explorar los perfiles</a>
  <a class="btn secundario" href="#simulador">Probar el simulador</a></div>
  <div class="cinta">
    <div class="titulo">La población en {k} perfiles</div>
    <div class="nota">Ancho = tamaño del perfil · Altura = participación en 2024 · Datos sintéticos (no son personas reales)</div>
    {_cinta_hero(perfiles, prom)}
  </div>
  <div class="kpis">
    <div class="kpi"><div class="v">{n:,}</div><div class="l">ciudadanos analizados</div></div>
    <div class="kpi"><div class="v">{prom:.0%}</div><div class="l">participación en 2024</div></div>
    <div class="kpi"><div class="v">{k}</div><div class="l">perfiles de participación</div></div>
    <div class="kpi"><div class="v">{gap:.0f} pp</div><div class="l">brecha entre el perfil más y menos participativo</div></div>
  </div>
</div></header>

<main class="wrap">
<section id="panorama">
  <div class="sec-head"><div class="num">01 · Panorama</div><h2>¿Quién vota y quién no?</h2>
  <p>La participación no se reparte de forma pareja: edad, educación, distancia y contacto con las campañas
  marcan diferencias enormes. La línea punteada es el promedio general ({prom:.0%}).</p></div>
  <div class="hallazgos">{hallazgos}</div>
  <div class="grid2">{panorama}</div>
</section>

<section id="perfiles">
  <div class="sec-head"><div class="num">02 · Perfiles</div><h2>{k} perfiles ciudadanos</h2>
  <p>Agrupamos a la población con K-Means según sus características y su historial de voto. Los perfiles
  están ordenados de mayor a menor participación en 2024.</p></div>
  <div class="perfiles">{tarjetas}</div>
  <div class="stack">
    <div class="card"><h3>Participación 2024 por perfil</h3><div class="sub">Línea punteada: promedio general.</div>{_html(fig_perfiles_participacion(perfiles, prom), 60 + 46 * len(perfiles), 'g-perfiles')}</div>
    <div class="card"><h3>¿Qué distingue a cada perfil?</h3><div class="sub">Color: desviaciones respecto al promedio de toda la población. Texto: valor típico del perfil.</div><div class="hscroll"><div>{_html(fig_rasgos(perfiles), 120 + 44 * len(perfiles), "g-rasgos")}</div></div></div>
    <div class="grid2">
      <div class="card"><h3>Evolución por elección</h3><div class="sub">Entre quienes eran elegibles en cada fecha.</div>{_html(fig_tendencia(perfiles), 380)}</div>
      <div class="card"><h3>Mapa de perfiles</h3><div class="sub">Las {len(seg['spec'])} variables proyectadas en 2 dimensiones (muestra de 3,500 personas).</div>{_html(fig_pca(R), 380)}</div>
    </div>
  </div>
</section>

<section id="factores">
  <div class="sec-head"><div class="num">03 · Factores</div><h2>¿Qué impulsa la participación?</h2>
  <p>Un modelo de regresión logística (AUC {ev['auc']:.2f} en datos no vistos) estima la probabilidad de votar en
  2024 a partir de las características de la persona. Medimos cuánto empeora al barajar cada variable.</p></div>
  <div class="grid2">
    <div class="card"><h3>Importancia de cada factor</h3><div class="sub">Caída del AUC (×100) al barajar la variable. Más alto = más determinante.</div>{_html(fig_importancia(ev['importancia']), 380)}</div>
    <div class="card"><h3>Edad × nivel educativo</h3><div class="sub">Participación 2024 (%). Se omiten celdas con menos de 30 personas.</div>{_html(fig_calor(t_edu, n_edu, 380), 380)}</div>
  </div>
  <div class="card" style="margin-top:16px"><h3>Región × tipo de zona</h3><div class="sub">Participación 2024 (%).</div>{_html(fig_calor(t_reg, n_reg, 260, l=90), 260)}</div>
</section>

<section id="estrategias">
  <div class="sec-head"><div class="num">04 · Estrategias</div><h2>¿Dónde conviene intervenir?</h2>
  <p>Usamos el modelo para simular cinco palancas de política pública y estimar cuántos puntos de participación
  aportaría cada una en cada perfil. Ajusta la <b>cobertura</b>: qué fracción de las personas alcanza realmente la acción.</p></div>
  <div class="control"><label for="cobertura">Cobertura de la acción</label>
    <input id="cobertura" type="range" min="5" max="100" step="5" value="40"><output id="cobertura-val">40%</output></div>
  <div class="callout" id="kpi-palanca"></div>
  <div class="card"><h3>Puntos de participación ganados</h3><div class="sub">Por perfil y palanca (puntos porcentuales de participación en 2024).</div><div id="mapa-estrategias" class="chart"></div></div>
  <div class="grid2" style="margin-top:16px">
    <div class="card"><h3>Alcance de cada palanca</h3><div class="sub">Votos adicionales por cada 10,000 ciudadanos (toda la población).</div><div id="barras-palancas" class="chart"></div></div>
    <div class="card"><h3>Supuestos de cada palanca</h3><div class="sub">Cambio simulado en las personas alcanzadas.</div>
      <ul style="margin:0;padding-left:18px">{detalle_palancas}</ul>
      <p class="aviso" style="margin-top:14px">Los puntos se obtienen como la diferencia entre la probabilidad predicha con y sin la acción, promediada en cada perfil y multiplicada por la cobertura.</p></div>
  </div>
  <p class="aviso">Estimaciones del modelo: reflejan asociaciones observadas en estos datos, no efectos causales garantizados.
  Un piloto con grupo de control debería validarlas antes de escalar.</p>
</section>

<section id="simulador">
  <div class="sec-head"><div class="num">05 · Simulador</div><h2>Simulador de ciudadano</h2>
  <p>Describe a una persona (o parte de un perfil típico) y mira su probabilidad de votar, a qué perfil se parece
  y qué acción la movilizaría más.</p></div>
  <div class="sim">
    <div class="card form"><div class="presets"><span style="font-size:.8rem;font-weight:700;color:var(--muted);align-self:center">Perfil típico:</span>{presets}</div>
      {_controles_simulador(tipicos[inicial])}</div>
    <div class="stack">
      <div class="card resultado">
        <div style="display:flex;gap:18px;align-items:flex-end;flex-wrap:wrap"><div><div class="sub">Probabilidad estimada de votar</div><div class="big" id="sim-prob">–</div></div>
          <div style="padding-bottom:8px"><span class="chip" id="sim-perfil">–</span><div class="sub" id="sim-perfil-txt" style="margin-top:6px"></div></div></div>
        <div class="medidor"><i id="sim-medidor"></i></div><div class="sub" id="sim-vs"></div></div>
      <div class="card"><h3>¿Qué empuja su decisión?</h3><div class="sub">Contribución de cada factor frente a una persona promedio (escala logit).</div><div id="sim-contrib" class="chart"></div></div>
      <div class="card"><h3>Si aplicamos una acción…</h3><div class="sub">Aumento de su probabilidad de votar (puntos porcentuales).</div><ul class="palancas-sim" id="sim-palancas"></ul></div>
    </div>
  </div>
</section>

<section id="metodo">
  <div class="sec-head"><div class="num">06 · Metodología</div><h2>Cómo se hizo y qué límites tiene</h2></div>
  <div class="metodo">
    <div class="card"><h3>Datos</h3><ul>
      <li><b>{n:,} personas sintéticas</b> con variables demográficas, actitudinales y de acceso (edad, escolaridad, ingreso,
      zona, interés político, confianza, distancia a la casilla, redes, contacto de campañas, organización civil) y voto en 2018, 2021 y 2024.</li>
      <li>Generadas con 9 grupos latentes y un modelo logístico de voto con hábito individual; la participación se calibró
      a 63 % (2018), 52 % (2021) y 61 % (2024). Quienes tenían &lt; 18 años en una elección no son elegibles (sin dato).</li>
      <li>Datos faltantes: {imp_ing:,} ingresos y {imp_conf:,} valores de confianza se imputaron con la mediana de su tipo de zona.</li></ul></div>
    <div class="card"><h3>Perfiles</h3><ul>
      <li>K-Means sobre 11 variables estandarizadas (ingreso y distancia en escala logarítmica) e historial de voto.</li>
      <li>Se fijó k = {k} por diseño (una rejilla de perfiles accionable); como referencia, el <i>silhouette</i> para cada k candidato es: {sil_txt}. Valores bajos y similares indican que los grupos se solapan y forman un continuo, no islas separadas.</li>
      <li><b>Estabilidad:</b> al reajustar con submuestras del 80 %, la partición se conserva (ARI medio = {seg['estabilidad']:.2f}).</li>
      <li>Frente a los grupos latentes con los que se generaron los datos, la recuperación es ARI = {seg['ari_verdad']:.2f}:
      los perfiles encontrados reflejan los grupos reales sin ser idénticos.</li>
      <li>Los nombres se asignan automáticamente a partir del rasgo más distintivo de cada centroide.</li></ul>
      <div style="margin-top:8px">{_html(fig_silhouette(seg), 230)}</div></div>
    <div class="card"><h3>Modelo de participación</h3><ul>
      <li>Regresión logística sobre {len(R['modelo']['features'])} términos (edad y edad², escolaridad, ingreso, interés, confianza,
      distancia, redes, contacto, organización, sexo, zona y región), entrenada con 75 % y evaluada con 25 % de los datos.</li>
      <li><b>AUC = {ev['auc']:.3f}</b>; un gradient boosting obtiene {ev['auc_hgb']:.3f}, así que el modelo lineal no pierde información relevante.</li>
      <li>Calibración: el error máximo por decil de probabilidad es de {ev['err_calibracion'] * 100:.1f} pp.</li>
      <li>El historial de voto no entra al modelo (para medir los factores estructurales, no el hábito).</li></ul></div>
    <div class="card"><h3>Límites y reproducibilidad</h3><ul>
      <li>Los resultados dependen del proceso generador sintético; con datos reales deben revalidarse.</li>
      <li>Las simulaciones de estrategias son asociaciones del modelo, no causalidad; se recomienda un piloto con grupo de control.</li>
      <li>Reproducir: <code>python main.py</code> (semillas fijas). Archivos en <code>salida/</code>:
      <code>reporte.html</code>, <code>perfiles_resumen.csv</code>, <code>ciudadanos_con_perfil.csv</code>.</li>
      <li>El simulador usa el mismo modelo y los mismos centroides que Python (verificado con <code>verificar.py</code>).</li></ul></div>
  </div>
</section>
</main>
<footer><strong>Perfiles ciudadanos de participación</strong><br>Datos sintéticos · Python (pandas, scikit-learn, Plotly)</footer>
<script>const DATOS = {_json(datos_js)};</script>
<script>{js}</script>
</body></html>"""
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(pagina, encoding="utf-8")
    return datos_js
