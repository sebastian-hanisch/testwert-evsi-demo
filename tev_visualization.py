"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.

Beschriftungen im Baum stehen als Anmerkungen (`add_annotation`) mit hellem Hintergrund, nicht als Text-Spuren (DEMO-PLAYBOOK §3)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import tev_constants as C
import tev_evaluation as ev
import tev_scenario as S


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      legend=dict(orientation="h", yanchor="top", y=-0.28))
    return lock_axes(fig)


# --- Experiment 1: der Baum ---------------------------------------------------------------------------------------------------------------------

def build_tree(a: ev.TestAnalysis):
    """Entscheidungsbaum (Ereignisknoten): Testergebnis -> Wahl -> Erwartungswert je Alternative, mit den Zahlen je Zweig."""
    fig = go.Figure()
    nodes_chance, nodes_dec = [(0.0, 0.0)], []
    lines = []
    notes = []
    labels = ("+", "−")
    ys = (1.6, -1.6)
    for row, yc in zip(a.rows, ys):
        lines.append(((0.0, 0.0), (1.0, yc)))
        nodes_dec.append((1.0, yc))
        post = "—" if row["posterior"] is None else ev.de(row["posterior"], 2)
        notes.append(dict(x=0.5, y=yc / 2 + (0.18 if yc > 0 else -0.18), text=f"Ergebnis {labels[row['y']]}: P = {ev.de(row['p_y'], 2)}", bg=True))
        notes.append(dict(x=1.0, y=yc + (0.55 if yc > 0 else -0.55), text=f"P(hoch | {labels[row['y']]}) = {post}", bg=True))
        for k, (name, dy) in enumerate((("G", 0.45), ("K", -0.45))):
            ye = yc + dy
            lines.append(((1.0, yc), (2.0, ye)))
            val = row["ev"][name]
            chosen = row["choice"] == name
            txt = f"{name}: " + ("—" if val is None else f"{ev.de(val, 1)} GE")
            notes.append(dict(x=2.08, y=ye, text=("<b>" + txt + " ✓</b>") if chosen else txt, xanchor="left", bg=False, color=C.COLOR_MAIN if chosen else None))
    for (x0, y0), (x1, y1) in lines:
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color="rgba(120,120,120,0.8)", width=2), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[p[0] for p in nodes_chance], y=[p[1] for p in nodes_chance], mode="markers", name="Ereignisknoten (Testergebnis)",
                             marker=dict(size=16, symbol="circle", color=C.COLOR_ALT), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[p[0] for p in nodes_dec], y=[p[1] for p in nodes_dec], mode="markers", name="Entscheidungsknoten (Wahl)",
                             marker=dict(size=16, symbol="square", color=C.COLOR_MAIN), hoverinfo="skip"))
    notes.append(dict(x=-0.05, y=0.0, text=f"Wert mit Test<br><b>{ev.de(a.ev_with, 1)} GE</b><br>(EV0 = {ev.de(a.ev0, 1)} GE)", xanchor="right", bg=False))
    for n in notes:
        fig.add_annotation(x=n["x"], y=n["y"], text=n["text"], showarrow=False, xanchor=n.get("xanchor", "center"), font=dict(size=12, color="#222222" if n.get("bg") else n.get("color")),
                           bgcolor="rgba(255,255,255,0.88)" if n.get("bg") else None)
    fig.update_xaxes(visible=False, range=[-1.2, 3.2])
    fig.update_yaxes(visible=False, range=[-2.7, 2.7])
    return _base(fig, height=380)


# --- Experiment 2: wertlos? ---------------------------------------------------------------------------------------------------------------------

def build_evsi_curve(z: ev.ZeroAnalysis):
    fig = go.Figure()
    fig.add_vrect(x0=0.5, x1=max(0.5, z.q0), fillcolor="rgba(192,80,77,0.12)", line_width=0, annotation_text="wertlos", annotation_position="top left")
    fig.add_trace(go.Scatter(x=list(z.q_grid), y=list(z.evsi_q), mode="lines+markers", line=dict(color=C.COLOR_MAIN, width=3), marker=dict(size=5), name="EVSI(q)",
                             hovertemplate="q = %{x:.2f}: EVSI %{y:.1f} GE<extra></extra>"))
    fig.add_hline(y=z.evpi, line=dict(color=C.COLOR_ALT, dash="dash"), annotation_text=f"EVPI = {ev.de(z.evpi, 1)} GE", annotation_position="bottom right")
    fig.update_xaxes(title_text="Trefferquote q (se = sp = q)", range=[0.495, 1.005])
    fig.update_yaxes(title_text="EVSI (GE)", range=[-2, z.evpi * 1.12 + 1])
    return _base(fig, legend=False)


def build_q0_over_p(z: ev.ZeroAnalysis):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(z.p_curve), y=list(z.q0_curve), mode="lines", line=dict(color=C.COLOR_MAIN, width=3), name="Wertlos-Schwelle q0(p)",
                             hovertemplate="Vorab %{x:.2f}: wertlos bis q = %{y:.3f}<extra></extra>"))
    fig.add_vline(x=ev.P_STAR, line=dict(color=C.COLOR_ALT, dash="dash"), annotation_text="p = 13/30", annotation_position="top")
    fig.add_trace(go.Scatter(x=[z.p], y=[z.q0], mode="markers", marker=dict(size=12, color=C.COLOR_WARN), name="gewählter Vorab-Wert", hoverinfo="skip"))
    fig.update_xaxes(title_text="Vorab P(hoch)", range=[0, 1])
    fig.update_yaxes(title_text="Wertlos bis zur Trefferquote", range=[0.49, 1.0])
    return _base(fig)


def build_heatmap(z: ev.ZeroAnalysis):
    """EVSI/EVPI über Sensitivität (senkrecht) und Spezifität (waagerecht); grau: EVSI = 0; gestrichelt: se + sp = 1 (Test ohne Aussage)."""
    xs = list(ev.SE_GRID)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=xs, y=xs, z=z.grid, zmin=0, zmax=1, colorscale="Viridis", colorbar=dict(title=dict(text="EVSI/EVPI", side="top"), nticks=4),
                             hovertemplate="sp %{x:.2f}, se %{y:.2f}: EVSI/EVPI %{z:.3f}<extra></extra>", name="EVSI/EVPI"))
    mask = np.where(z.zero_mask, 1.0, np.nan)
    fig.add_trace(go.Heatmap(x=xs, y=xs, z=mask, zmin=0, zmax=1, colorscale=[[0, "rgba(160,160,160,0.85)"], [1, "rgba(160,160,160,0.85)"]], showscale=False,
                             hovertemplate="sp %{x:.2f}, se %{y:.2f}: EVSI = 0<extra></extra>", name="EVSI = 0"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[1, 0], mode="lines", line=dict(color="white", dash="dash", width=2), name="se + sp = 1 (ohne Aussage)", hoverinfo="skip"))
    fig.update_xaxes(title_text="Spezifität sp", range=[-0.005, 1.005])
    fig.update_yaxes(title_text="Sensitivität se", range=[-0.005, 1.005])
    return _base(fig, height=420)


# --- Experiment 3: Schwellenkurve ---------------------------------------------------------------------------------------------------------------

def build_threshold_curve(ca: ev.CostAnalysis):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(ca.c_curve), y=list(ca.q_curve), mode="lines", line=dict(color=C.COLOR_MAIN, width=3), name="Mindest-Trefferquote q*(c)",
                             connectgaps=False, hovertemplate="c = %{x:.1f} GE: q* = %{y:.3f}<extra></extra>"))
    fig.add_vline(x=ca.evpi, line=dict(color=C.COLOR_WARN, dash="dash"), annotation_text=f"ab c = EVPI = {ev.de(ca.evpi, 1)} GE: nie", annotation_position="top left")
    if ca.q_star is not None:
        fig.add_trace(go.Scatter(x=[ca.c], y=[ca.q_star], mode="markers", marker=dict(size=12, color=C.COLOR_WARN), name="gewählte Testkosten", hoverinfo="skip"))
    fig.update_xaxes(title_text="Testkosten c (GE)", range=[0, C.C_MAX + 2])
    fig.update_yaxes(title_text="Mindest-Trefferquote q*", range=[0.49, 1.01])
    return _base(fig)


# --- Experiment 4: mehrere Tests ----------------------------------------------------------------------------------------------------------------

def build_evsi_n(m: ev.MultiAnalysis):
    ns = list(range(1, len(m.values)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=m.values[1:], mode="lines+markers", line=dict(color=C.COLOR_MAIN, width=3), marker=dict(size=5), name="EVSI(n)",
                             hovertemplate="n = %{x}: EVSI %{y:.2f} GE<extra></extra>"))
    fig.add_hline(y=m.evpi, line=dict(color=C.COLOR_ALT, dash="dash"), annotation_text=f"EVPI = {ev.de(m.evpi, 1)} GE", annotation_position="bottom right")
    fig.update_xaxes(title_text="Zahl der Tests n", range=[0, len(m.values)])
    fig.update_yaxes(title_text="EVSI (GE)", range=[0, m.evpi * 1.1 + 1])
    return _base(fig, legend=False)


def build_increments(m: ev.MultiAnalysis):
    ns = list(range(1, len(m.values)))
    cols = [C.COLOR_WARN if n in m.rises else C.COLOR_MAIN for n in ns]
    fig = go.Figure(go.Bar(x=ns, y=m.inc[1:], marker_color=cols, name="Zuwachs", hovertemplate="n = %{x}: Zuwachs %{y:.2f} GE<extra></extra>"))
    if m.c > 0:
        fig.add_hline(y=m.c, line=dict(color="gray", dash="dot"), annotation_text=f"Kosten c = {ev.de(m.c, 1)} GE", annotation_position="top right")
    fig.update_xaxes(title_text="Zahl der Tests n (rot: der Zuwachs steigt gegenüber dem Vorgänger)", range=[0, len(m.values)])
    fig.update_yaxes(title_text="Zuwachs je Test (GE)")
    return _base(fig, legend=False)


def build_net(m: ev.MultiAnalysis):
    ns = list(range(0, len(m.values)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=m.net, mode="lines+markers", line=dict(color=C.COLOR_MAIN, width=3), marker=dict(size=5), name="Nettowert EVSI(n) − c·n",
                             hovertemplate="n = %{x}: Nettowert %{y:.2f} GE<extra></extra>"))
    fig.add_trace(go.Scatter(x=[m.n_star], y=[m.net[m.n_star]], mode="markers", marker=dict(size=14, color=C.COLOR_WARN, symbol="star"), name=f"Optimum n* = {m.n_star}", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[m.n_rule], y=[m.net[m.n_rule]], mode="markers", marker=dict(size=12, color="#e69f00", symbol="diamond"), name=f"Abbruchregel stoppt bei n = {m.n_rule}", hoverinfo="skip"))
    fig.update_xaxes(title_text="Zahl der Tests n", range=[-1, len(m.values)])
    fig.update_yaxes(title_text="Nettowert (GE)")
    return _base(fig)


# --- Experiment 5: die Falle --------------------------------------------------------------------------------------------------------------------

def build_trap_bars(t: ev.TrapAnalysis):
    labels = ["wahr (EVSI)", "Korrelation × EVPI", "Korrelation² × EVPI", "Trefferquote × EVPI"]
    vals = [t.true_value, t.corr, t.corr2, t.rate]
    cols = [C.COLOR_MAIN, C.COLOR_ALT, C.COLOR_ALT, C.COLOR_ALT]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=cols, text=[ev.de(v, 1) for v in vals], textposition="outside", hoverinfo="skip"))
    fig.add_hline(y=t.evpi, line=dict(color="gray", dash="dash"), annotation_text=f"EVPI = {ev.de(t.evpi, 1)} GE", annotation_position="top right")
    fig.update_yaxes(title_text="GE", range=[0, max(vals + [t.evpi]) * 1.18 + 1])
    return _base(fig, legend=False)


def build_trap_curves(t: ev.TrapAnalysis):
    prior = S.prior_of(t.p)
    q = ev.Q_GRID
    true, phi, rules = ev._rules_on_grid(prior, t.evpi)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(q), y=list(true), mode="lines", line=dict(color=C.COLOR_MAIN, width=3), name="wahr (EVSI)"))
    fig.add_trace(go.Scatter(x=list(q), y=list(rules["corr"]), mode="lines", line=dict(color=C.COLOR_ALT, width=2, dash="dash"), name="Korrelation × EVPI"))
    fig.add_trace(go.Scatter(x=list(q), y=list(rules["corr2"]), mode="lines", line=dict(color="#e69f00", width=2, dash="dot"), name="Korrelation² × EVPI"))
    fig.add_trace(go.Scatter(x=[t.q], y=[t.true_value], mode="markers", marker=dict(size=12, color=C.COLOR_WARN), name="gewählte Trefferquote", hoverinfo="skip"))
    fig.update_xaxes(title_text="Trefferquote q", range=[0.495, 1.005])
    fig.update_yaxes(title_text="Wert (GE)", range=[-2, t.evpi * 1.05 + 1])
    return _base(fig)


def build_pair_heatmap(t: ev.TrapAnalysis):
    """Gemeinsam minus Summe der Einzelwerte für zwei Tests (Trefferquoten qa waagerecht, qb senkrecht); blau: weniger als die Summe, rot: mehr."""
    xs = list(ev.Q_GRID)
    lim = float(max(abs(t.pair_diff.min()), abs(t.pair_diff.max()), 1e-9))
    fig = go.Figure(go.Heatmap(x=xs, y=xs, z=t.pair_diff.T, zmin=-lim, zmax=lim, colorscale="RdBu_r", zmid=0,
                               colorbar=dict(title=dict(text="gemeinsam − Summe (GE)", side="top"), nticks=5),
                               hovertemplate="qa %{x:.2f}, qb %{y:.2f}: gemeinsam − Summe %{z:.2f} GE<extra></extra>"))
    fig.add_trace(go.Scatter(x=[t.qa], y=[t.qb], mode="markers", marker=dict(size=12, color="black", symbol="x"), name="gewähltes Paar", hoverinfo="skip"))
    fig.update_xaxes(title_text="Trefferquote Test A", range=[0.495, 1.005])
    fig.update_yaxes(title_text="Trefferquote Test B", range=[0.495, 1.005])
    return _base(fig, height=420)
