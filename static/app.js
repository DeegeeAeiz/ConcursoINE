/* Lógica del tablero: modelo logístico, asignación de perfil, palancas y gráficas interactivas.
   Las funciones puras se exportan para poder verificarlas contra Python con Node. */

function transformar(x, t) {
  switch (t) {
    case "raw": return x;
    case "sq": return x * x;
    case "log": return Math.log(x);
    case "log1p": return Math.log1p(x);
  }
  throw new Error("transformación desconocida: " + t);
}

function valorFeature(f, rec) {
  if (f.kind === "dummy") return rec[f.var] === f.level ? 1 : 0;
  return transformar(Number(rec[f.var]), f.transform);
}

function zscore(f, rec) {
  return (valorFeature(f, rec) - f.mean) / f.std;
}

function logit(modelo, rec) {
  let z = modelo.intercept;
  for (const f of modelo.features) z += f.coef * zscore(f, rec);
  return z;
}

function sigmoide(z) { return 1 / (1 + Math.exp(-z)); }

function probabilidad(modelo, rec) { return sigmoide(logit(modelo, rec)); }

function contribuciones(modelo, rec) {
  const out = {};
  for (const f of modelo.features) out[f.grupo] = (out[f.grupo] || 0) + f.coef * zscore(f, rec);
  return out;
}

function perfilMasCercano(cluster, rec) {
  const x = cluster.features.map((f) => zscore(f, rec));
  let mejor = 0, dmin = Infinity;
  cluster.centroides.forEach((c, i) => {
    let d = 0;
    for (let j = 0; j < x.length; j++) d += (x[j] - c[j]) ** 2;
    if (d < dmin) { dmin = d; mejor = i; }
  });
  return mejor;
}

const PALANCAS = {
  distancia: (r) => ({ ...r, distancia_casilla_km: Math.min(Number(r.distancia_casilla_km), 1) }),
  contacto: (r) => ({ ...r, contacto_campana: 1 }),
  interes: (r) => ({ ...r, interes_politico: Math.min(Number(r.interes_politico) + 0.5, 5) }),
  confianza: (r) => ({ ...r, confianza_instituciones: Math.min(Number(r.confianza_instituciones) + 0.5, 5) }),
  org: (r) => ({ ...r, participa_org_civil: 1 }),
};

if (typeof module !== "undefined" && module.exports) {
  module.exports = { transformar, valorFeature, logit, probabilidad, contribuciones, perfilMasCercano, PALANCAS };
}

/* ------------------------------------------------------------------ UI */
function iniciarUI(D) {
  const pct = (x, d = 0) => (x * 100).toFixed(d) + "%";
  const cfg = { displayModeBar: false, responsive: true };
  const base = (extra) => Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: D.estilo.fuente, size: 13, color: D.estilo.tinta2 },
    hoverlabel: { bgcolor: "#fff", bordercolor: D.estilo.linea, font: { color: D.estilo.tinta, size: 13 } },
    margin: { l: 10, r: 20, t: 8, b: 40 },
  }, extra);
  const corto = (s, n = 44) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
  const movil = () => window.innerWidth < 700;

  /* --- Estrategias: cobertura ajustable --- */
  const slider = document.getElementById("cobertura");
  const etiqueta = document.getElementById("cobertura-val");
  const maxCelda = Math.max(...D.matriz.flat());
  function dibujarEstrategias() {
    const cov = slider.value / 100;
    etiqueta.textContent = slider.value + "%";
    const y = D.perfiles.map((p) => (movil() ? `P${p.id}` : `P${p.id} · ${corto(p.nombre)}`));
    const x = D.palancas.map((p) => p.nombre);
    Plotly.react("mapa-estrategias", [{
      type: "heatmap", z: D.matriz.map((r) => r.map((v) => v * cov)), x, y,
      colorscale: D.estilo.secuencial, zmin: 0, zmax: maxCelda, xgap: 3, ygap: 3, showscale: false,
      texttemplate: "+%{z:.1f}", hovertemplate: "<b>%{y}</b><br>%{x}<br>+%{z:.2f} puntos de participación<extra></extra>",
    }], base({ margin: { l: movil() ? 36 : 300, r: 10, t: 8, b: 70 }, height: 90 + 40 * D.perfiles.length,
      xaxis: { side: "bottom", tickangle: 0, automargin: true }, yaxis: { autorange: "reversed", automargin: true } }), cfg);

    const votos = D.global_pp.map((v) => v * 100 * cov); // pp -> votos por cada 10 000
    const orden = votos.map((v, i) => i).sort((a, b) => votos[b] - votos[a]);
    Plotly.react("barras-palancas", [{
      type: "bar", orientation: "h", x: orden.map((i) => votos[i]), y: orden.map((i) => D.palancas[i].nombre),
      marker: { color: D.estilo.acento, cornerradius: 4 }, text: orden.map((i) => Math.round(votos[i]).toLocaleString("es-MX")),
      textposition: "outside", cliponaxis: false,
      hovertemplate: "%{y}<br>%{x:,.0f} votos adicionales por cada 10 000 ciudadanos<extra></extra>",
    }], base({ height: 300, margin: { l: 170, r: 50, t: 8, b: 40 },
      xaxis: { gridcolor: D.estilo.linea, zeroline: false, range: [0, Math.max(...votos) * 1.2 + 1] },
      yaxis: { autorange: "reversed", automargin: true } }), cfg);
    const mejor = orden[0];
    document.getElementById("kpi-palanca").innerHTML =
      `<strong>${D.palancas[mejor].nombre}</strong> es la palanca con mayor alcance: ` +
      `<strong>+${Math.round(votos[mejor]).toLocaleString("es-MX")}</strong> votos por cada 10,000 ciudadanos ` +
      `(+${(D.global_pp[mejor] * cov).toFixed(1)} puntos de participación) con una cobertura del ${slider.value}%.`;
    document.querySelectorAll("[data-pp]").forEach((el) => {
      el.textContent = "hasta +" + Number(el.dataset.pp).toFixed(1) + " pp";
    });
  }
  slider.addEventListener("input", dibujarEstrategias);
  dibujarEstrategias();

  /* En pantallas angostas los perfiles se rotulan solo con su número (P1, P2…) */
  function compactar() {
    const largo = D.perfiles.map((p) => `P${p.id} · ${p.nombre}`);
    const corto = D.perfiles.map((p) => `P${p.id}`);
    for (const id of ["g-perfiles", "g-rasgos"]) {
      const el = document.getElementById(id);
      if (el && el.data) Plotly.relayout(el, { "yaxis.tickvals": largo, "yaxis.ticktext": movil() ? corto : largo, "margin.l": movil() ? 36 : 300 });
    }
  }
  let anchoPrevio = window.innerWidth < 700;
  window.addEventListener("resize", () => {
    if ((window.innerWidth < 700) !== anchoPrevio) { anchoPrevio = window.innerWidth < 700; compactar(); dibujarEstrategias(); }
  });
  compactar();

  /* --- Simulador de ciudadano --- */
  const campos = Array.from(document.querySelectorAll("[data-var]"));
  function leer() {
    const rec = {};
    for (const el of campos) {
      if (el.type === "checkbox") rec[el.dataset.var] = el.checked ? 1 : 0;
      else if (el.tagName === "SELECT") rec[el.dataset.var] = el.value;
      else rec[el.dataset.var] = Number(el.value);
    }
    rec.tasa_participacion = rec.tasa_participacion / 100;
    return rec;
  }
  function escribir(rec) {
    for (const el of campos) {
      const v = el.dataset.var === "tasa_participacion" ? rec[el.dataset.var] * 100 : rec[el.dataset.var];
      if (el.type === "checkbox") el.checked = !!v; else el.value = v;
    }
    actualizarSim();
  }
  function actualizarSim() {
    for (const el of campos) {
      const out = document.getElementById("o-" + el.dataset.var);
      if (out && el.type !== "checkbox") out.textContent = el.dataset.var === "tasa_participacion" ? el.value + "%" : el.value;
    }
    const rec = leer();
    const p = probabilidad(D.modelo, rec);
    document.getElementById("sim-prob").textContent = pct(p);
    document.getElementById("sim-medidor").style.width = pct(p);
    const dif = (p - D.promedio) * 100;
    document.getElementById("sim-vs").textContent =
      `${dif >= 0 ? "+" : ""}${dif.toFixed(0)} pp frente al promedio general (${pct(D.promedio)})`;

    const pf = D.perfiles[perfilMasCercano(D.cluster, rec)];
    const chip = document.getElementById("sim-perfil");
    chip.textContent = `P${pf.id} · ${pf.nombre}`;
    chip.style.setProperty("--c", pf.color);
    document.getElementById("sim-perfil-txt").textContent = `Participación de este perfil en 2024: ${pct(pf.v2024)}`;

    const contrib = contribuciones(D.modelo, rec);
    const grupos = Object.keys(contrib).sort((a, b) => Math.abs(contrib[a]) - Math.abs(contrib[b]));
    Plotly.react("sim-contrib", [{
      type: "bar", orientation: "h", y: grupos.map((g) => D.nombreVariable[g] || g), x: grupos.map((g) => contrib[g]),
      marker: { color: grupos.map((g) => (contrib[g] >= 0 ? D.estilo.pos : D.estilo.neg)), cornerradius: 3 },
      hovertemplate: "%{y}: %{x:+.2f}<extra></extra>",
    }], base({ height: 380, margin: { l: 150, r: 20, t: 8, b: 46 },
      xaxis: { gridcolor: D.estilo.linea, zerolinecolor: D.estilo.tinta2, title: { text: "← empuja a abstenerse · empuja a votar →", font: { size: 12 } } },
      yaxis: { automargin: true } }), cfg);

    const res = D.palancas.map((pl) => ({ pl, d: (probabilidad(D.modelo, PALANCAS[pl.id](rec)) - p) * 100 }))
      .sort((a, b) => b.d - a.d);
    document.getElementById("sim-palancas").innerHTML = res.map((r) =>
      `<li><span>${r.pl.nombre}</span><i style="--w:${Math.min(100, r.d * 5)}%"></i><b>${r.d >= 0.05 ? "+" + r.d.toFixed(1) : "0.0"} pp</b></li>`).join("");
  }
  campos.forEach((el) => el.addEventListener("input", actualizarSim));
  document.querySelectorAll("[data-tipico]").forEach((b) =>
    b.addEventListener("click", () => escribir(D.tipicos[Number(b.dataset.tipico)])));
  escribir(D.tipicos[D.tipicoInicial]);
}

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", () => iniciarUI(DATOS));
}
