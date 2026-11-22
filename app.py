"""Ein Test und sein Wert – Bayes-Update und EVSI - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Zahlenbeispiel (Kapazitätsfall, ausgedacht, Geldeinheiten GE): vor der Wahl zwischen G und K steht ein Test mit bekannter Trefferquote. Die Seite rechnet
Bayes-Update, EVSI und die Schwellen und zeigt fünf Experimente; Aufbau nach DEMO-PLAYBOOK §2 (Schnellstart in der Seitenleiste).

Lauffähig mit: streamlit run app.py
"""

import time

import pandas as pd
import streamlit as st

import tev_evaluation as ev
import tev_scenario as S
from tev_layout import dunkles_design, kopf_css
from tev_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    SETTING_SPECS,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from tev_visualization import (
    build_evsi_curve,
    build_evsi_n,
    build_heatmap,
    build_increments,
    build_net,
    build_pair_heatmap,
    build_q0_over_p,
    build_threshold_curve,
    build_tree,
    build_trap_bars,
    build_trap_curves,
)

st.set_page_config(page_title="Ein Test und sein Wert – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _test(p, se, sp, c):
    return ev.analyse_test(ev.Settings(p=p, se=se, sp=sp, c=c))


@st.cache_data(show_spinner=False)
def _zero(p):
    return ev.analyse_zero(p)


@st.cache_data(show_spinner=False)
def _cost(p, c):
    return ev.analyse_cost(p, c)


@st.cache_data(show_spinner=False)
def _multi(p, q, n, c):
    return ev.analyse_multi(p, q, n, c)


@st.cache_data(show_spinner=False)
def _worthless(p):
    return ev.worthless_table(p)


@st.cache_data(show_spinner=False)
def _trap(p, q, qa, qb):
    return ev.analyse_trap(p, q, qa, qb)


@st.cache_data(show_spinner=False)
def _random(seed, m):
    t0 = time.perf_counter()
    stats = S.problem_stats(seed, m)
    return stats, time.perf_counter() - t0


def _table(rows, columns):
    st.dataframe(pd.DataFrame(rows, columns=columns).astype(str), hide_index=True, width="stretch")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in tev_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧪 Ein Test und sein Wert – Bayes-Update und EVSI")
st.markdown(
    """
**Ein Test vor einer Wahl kostet etwas und sagt nicht immer die Wahrheit: was ist er wert?** Das Zahlenbeispiel ist **ausgedacht** (Geldeinheiten GE, ein Modell und kein Fall aus der Praxis):
man wählt zwischen **G** (groß bauen: +200 GE bei hoher, −100 GE bei niedriger Nachfrage) und **K** (klein bauen oder mieten: sicher +30 GE); ohne Information gilt P(hoch) = 1/2.
Vor die Wahl tritt ein **Test** mit bekannter Trefferquote. Die Demo rechnet das **Bayes-Update**, den **Wert der Stichprobeninformation (EVSI)** und die Schwellen, ab denen sich ein Test lohnt,
und zeigt, warum sich dieser Wert **weder aus der Trefferquote noch aus „Korrelation × EVPI“ ablesen** lässt und warum sich die Werte mehrerer Tests nicht addieren. Alles gilt für den
Erwartungswert (linearer Nutzen); die Trefferquote ist eine **Eingabe**, kein Befund über einen echten Test.

Weiter unten: fünf Experimente, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. „Entscheidungsbaum“ meint hier den Entscheidungsbaum mit
Ereignisknoten, nicht den Baum aus Daten ([`cart-demo`](https://github.com/sebastian-hanisch/cart-demo)).
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/testwert-evsi-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Stellen Sie sich vor, ein Hafen soll groß oder klein gebaut werden, und niemand weiß, wie hoch die Nachfrage wird. Ein Gutachten mit bekannter Trefferquote kostet etwas. Es ist nur dann etwas wert, wenn die Meldung die Wahl ändern kann: Würde man nach jeder Meldung dasselbe bauen, bringt es nichts, wie genau es auch trifft. Die Demo rechnet nach, wie sich die Wahrscheinlichkeiten nach der Meldung ändern (**Bayes-Update**), und bewertet den Hinweis mit dem **Wert der Stichprobeninformation (EVSI)**. Sie zeigt, ab welcher Trefferquote sich ein Test bei Kosten lohnt, und zählt, wann zwei Hinweise zusammen weniger, mehr oder gleich viel bringen wie die Summe ihrer Einzelwerte. Alle Zahlen sind ausgedacht, es ist ein Rechenmodell und kein echter Betrieb.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/testwert-evsi-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert ein Test im Baum", expanded=True):
    st.markdown(
        """
1. **Ohne Test** hat G den Erwartungswert EV(G) = P · 200 − (1 − P) · 100 und K den Wert 30; die bessere Wahl gibt **EV0**. Ein Hellseher, der den Zustand vorher kennt, wählt in jedem Zustand die bessere Alternative:
   der Mehrwert gegenüber EV0 ist das **EVPI** (Wert der vollkommenen Information, englisch Expected Value of Perfect Information) und die Obergrenze für alles, was ein Test bringen kann.
2. **Der Test** liefert ein Ergebnis + oder −. Die **Sensitivität** se ist P(+ | hoch), die **Spezifität** sp ist P(− | niedrig); sind beide gleich, heißt der Wert **Trefferquote q**. Im Baum steht vorn ein
   **Ereignisknoten „Testergebnis“**, danach der **Entscheidungsknoten „Wahl“**.
3. **Bayes umgedreht:** aus Vorab und Testeigenschaften folgen P(+), P(−) und die Posterior-Werte P(hoch | +) und P(hoch | −).
4. **Rollback:** nach jedem Ergebnis die bessere Alternative wählen; der Wert mit Test ist der Mittelwert dieser Werte mit den Gewichten P(+) und P(−). **EVSI** = Wert mit Test − EV0, und
   **ENBS** = EVSI − c ist der Nettonutzen bei Testkosten c.
5. **Wann der Test nichts bringt:** wählt man nach + und nach − dieselbe Alternative, ändert der Test nichts, und EVSI = 0. Das ist keine Frage der Trefferquote allein, sondern ob ein Ergebnis die Wahl kippen kann.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Vorab-Wissen**")
    p = st.slider("Vorab P(hoch)", *bounds("p_slider"), step=0.001, format="%.3f", key="p_slider",
                  help="Wahrscheinlichkeit hoher Nachfrage vor dem Test. Bei 13/30 = 0,433 sind G und K gleich viel wert; darüber wird G gewählt, darunter K.")
    st.markdown("**Der Test (Experiment 1)**")
    se = st.slider("Sensitivität se = P(+ | hoch)", *bounds("se_slider"), step=0.01, key="se_slider", help="Wie oft der Test bei hoher Nachfrage „+“ zeigt.")
    sp = st.slider("Spezifität sp = P(− | niedrig)", *bounds("sp_slider"), step=0.01, key="sp_slider", help="Wie oft der Test bei niedriger Nachfrage „−“ zeigt.")
    st.markdown("**Kosten**")
    c = st.slider("Testkosten c (GE)", *bounds("c_slider"), step=0.5, key="c_slider",
                  help="Preis eines Tests in GE (Experimente 1, 3 und 4; in Experiment 4 je Test). Das EVPI liegt bei Vorab 1/2 bei 65 GE: ab dort lohnt kein Test.")

sync_query_params({key: st.session_state[key] for key in SETTING_SPECS})

prior_p, se_v, sp_v, c_v = float(p), float(se), float(sp), float(c)

# --- Experiment 1: von Hand ----------------------------------------------------------------------------------------------------------------------

a = _test(prior_p, se_v, sp_v, c_v)
st.markdown("## 🎯 Experiment 1: Von Hand – Test, Bayes-Update, Rollback")
st.caption("Ein Test vor der Wahl zwischen G und K. Vorab, Sensitivität, Spezifität und Testkosten stehen in der Seitenleiste. Alle Werte in GE (ausgedachtes Zahlenbeispiel).")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("EV0 (ohne Test)", f"{ev.de(a.ev0, 1)} GE", delta=f"Wahl {a.best0}", delta_color="off", help="Wert der besten Wahl ohne Information.")
m2.metric("EVPI (Hellseher)", f"{ev.de(a.evpi, 1)} GE", delta="Obergrenze für jeden Test", delta_color="off", help="Wert der vollkommenen Information bei linearem Nutzen.")
m3.metric("EVSI (dieser Test)", f"{ev.de(a.evsi, 1)} GE", delta=f"Wert mit Test {ev.de(a.ev_with, 1)} GE", delta_color="off", help="Wert mit Test minus EV0.")
m4.metric("EVSI / EVPI", "–" if a.ratio is None else ev.pct(a.ratio), delta="Nenner: EVPI", delta_color="off", help="Anteil am EVPI. Der Nenner steht immer daneben.")
m5.metric("ENBS = EVSI − c", f"{ev.de(a.enbs, 1, sign=True)} GE", delta=f"bei Kosten {ev.de(a.c, 1)} GE", delta_color="off", help="Erwarteter Nettonutzen des Tests (nach Testkosten).")
st.plotly_chart(build_tree(a), width="stretch", key="tree_chart")
st.caption("Entscheidungsbaum (Ereignisknoten): links der Ereignisknoten „Testergebnis“, in der Mitte die Entscheidungsknoten, rechts der Erwartungswert je Alternative nach dem Ergebnis; "
           "✓ markiert die gewählte Alternative.")

tl, tr = st.columns(2)
with tl:
    st.markdown("**Verbundwahrscheinlichkeiten P(Zustand, Ergebnis)**")
    _table(
        [["hoch", ev.de(a.joint[0][0], 3), ev.de(a.joint[1][0], 3), ev.de(a.joint[0][0] + a.joint[1][0], 3)],
         ["niedrig", ev.de(a.joint[0][1], 3), ev.de(a.joint[1][1], 3), ev.de(a.joint[0][1] + a.joint[1][1], 3)],
         ["Summe P(y)", ev.de(a.rows[0]["p_y"], 3), ev.de(a.rows[1]["p_y"], 3), ev.de(a.rows[0]["p_y"] + a.rows[1]["p_y"], 3)]],
        ["Zustand", "Ergebnis +", "Ergebnis −", "Summe"],
    )
with tr:
    st.markdown("**Posterior, Wahl und Beitrag zum Wert mit Test**")
    rows = []
    for r, lab in zip(a.rows, ("+", "−")):
        rows.append([lab, ev.de(r["p_y"], 3), "–" if r["posterior"] is None else ev.de(r["posterior"], 3),
                     "–" if r["ev"]["G"] is None else ev.de(r["ev"]["G"], 1), "–" if r["ev"]["K"] is None else ev.de(r["ev"]["K"], 1), r["choice"],
                     ev.de(r["joint_value"], 1)])
    _table(rows, ["Ergebnis", "P(y)", "P(hoch | y)", "EV(G | y)", "EV(K | y)", "Wahl", "P(y) · bester Wert"])
st.caption(f"Zweite Rechnung, Verlust statt Gewinn: der erwartete **Verlust mit Test** (ex-post-Mehrkosten gegen einen Hellseher, in diesem Fall gleich dem Opportunitätsverlust) beträgt "
           f"{ev.de(a.loss_with, 1)} GE, ohne Test {ev.de(a.loss_without, 1)} GE; die Differenz ist das EVSI ({ev.de(a.loss_without - a.loss_with, 1)} GE). "
           "„Regret“ heißt hier den Mehraufwand gegen einen Hellseher, nicht das Savage-Kriterium.")
kind, text = ev.verdict_test(a)
getattr(st, kind)(text)

# --- Kernfragen ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("---")
st.markdown("## ❓ Kernfragen")
cost_a = _cost(prior_p, c_v)
zero_a = _zero(prior_p)
multi_a = _multi(prior_p, float(st.session_state["q4_slider"]), int(st.session_state["n_slider"]), c_v)
trap_a = _trap(prior_p, float(st.session_state["q5_slider"]), float(st.session_state["qa_slider"]), float(st.session_state["qb_slider"]))
st.markdown("**1. Was ist ein Test wert, und wann ändert er die Wahl nie?** (Experimente 1 und 2)")
st.info(ev.verdict_zero(zero_a)[1])
st.markdown("**2. Ab welcher Trefferquote lohnt er bei Kosten?** (Experiment 3)")
getattr(st, ev.verdict_cost(cost_a)[0])(ev.verdict_cost(cost_a)[1])
st.markdown("**3. Wie rechnet man mehrere Tests zusammen, und warum geht das nicht durch Addieren oder durch „Korrelation × EVPI“?** (Experimente 4 und 5)")
getattr(st, ev.verdict_multi(multi_a)[0])(ev.verdict_multi(multi_a)[1])
getattr(st, ev.verdict_trap(trap_a)[0])(ev.verdict_trap(trap_a)[1])

st.markdown("---")

# --- Experiment 2: wann wertlos? --------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wann ist ein Test wertlos?")
st.caption("Vorab steht in der Seitenleiste. Links der Wert eines symmetrischen Tests (se = sp = q) über der Trefferquote, rechts die Wertlos-Schwelle über dem Vorab. "
           "Zähleinheit der Zählungen: Rasterpunkt (se und sp in Schritten von 0,01, q von 0,50 bis 1,00 in Schritten von 0,01).")
cl, cr = st.columns(2)
with cl:
    st.plotly_chart(build_evsi_curve(zero_a), width="stretch", key="evsi_curve")
with cr:
    st.plotly_chart(build_q0_over_p(zero_a), width="stretch", key="q0_curve")
st.markdown("**Sensitivität gegen Spezifität: EVSI/EVPI (grau: EVSI = 0)**")
st.plotly_chart(build_heatmap(zero_a), width="stretch", key="heatmap")
kind, text = ev.verdict_zero(zero_a)
getattr(st, kind)(text)

with st.expander("Gegenprobe über Zufallsprobleme"):
    st.markdown(
        "Zufällige Probleme mit zwei Alternativen und zwei Zuständen: Auszahlungen ganzzahlig gleichverteilt in [−100, 200], Vorab gleichverteilt in [0,05; 0,95], "
        "Sensitivität und Spezifität gleichverteilt in [0, 1] (Zähleinheit Zufallsproblem, Zufallsstrom SplitMix64). Gezählt wird, wie oft EVSI = 0 ist und ob die Kette "
        "EVPI ≥ EVSI ≥ 0 hält. **Die Verteilung der Probleme bestimmt die Zahlen**: andere Verteilungen geben andere Anteile."
    )
    ra, rb, rc = st.columns([2, 2, 1])
    with ra:
        rz_seed = st.number_input("Zufalls-Seed der Probleme", *bounds("rz_seed_input"), step=1, key="rz_seed_input")
    with rb:
        rz_m = st.slider("Zahl der Zufallsprobleme", *bounds("rz_m_slider"), step=1000, key="rz_m_slider")
    with rc:
        st.write("")
        st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=randomize_seed)
    with st.spinner(f"Rechne {int(rz_m)} Zufallsprobleme …"):
        stats, seconds = _random(int(rz_seed), int(rz_m))
    kind, text = ev.verdict_random(stats)
    getattr(st, kind)(text)
    st.caption(f"Rechenzeit der Gegenprobe: {ev.de(seconds, 2)} s (beim ersten Aufruf dieser Einstellung).")

st.markdown("---")

# --- Experiment 3: ab welcher Trefferquote lohnt? ---------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Ab welcher Trefferquote lohnt der Test?")
st.caption("Vorab und Testkosten c stehen in der Seitenleiste. Gesucht ist die kleinste Trefferquote q*, bei der EVSI die Kosten übersteigt (ENBS > 0).")
st.plotly_chart(build_threshold_curve(cost_a), width="stretch", key="threshold_curve")
tl, tr = st.columns(2)
with tl:
    st.markdown(f"**Asymmetrischer Test: Mindest-Spezifität bei Testkosten {ev.de(c_v, 1)} GE**")
    _table([[ev.de(se_, 2), "nie" if sp_ is None else ev.de(sp_, 4), ev.de(max_, 1)] for se_, sp_, max_ in cost_a.spec_rows],
           ["Sensitivität se", "Mindest-Spezifität", "EVSI bei sp = 1 (GE)"])
with tr:
    st.markdown(f"**Je Vorab: Wertlos-Schwelle und Mindest-Trefferquote bei Kosten {ev.de(c_v, 1)} GE**")
    _table([[ev.de(pp, 2), ev.de(e, 1), ev.de(q0, 4), "nie" if qs is None else ev.de(qs, 4)] for pp, e, q0, qs in cost_a.prior_rows],
           ["Vorab", "EVPI (GE)", "wertlos für 1/2 ≤ q ≤ q0", "Mindest-Trefferquote q*"])
kind, text = ev.verdict_cost(cost_a)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4: mehrere Tests -------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Mehrere Tests, und wann hört man auf?")
st.caption("n unabhängige gleichartige Tests mit Trefferquote q, ausgewertet über die Zahl der positiven Ergebnisse; Testkosten c (Seitenleiste) je Test, Vorab ebenfalls dort. "
           "Zähleinheit: Testzahl n (1 bis 60).")
r1, r2 = st.columns(2)
with r1:
    q4 = st.slider("Trefferquote q je Test", *bounds("q4_slider"), step=0.01, key="q4_slider", help="Sensitivität = Spezifität jedes einzelnen Tests.")
with r2:
    n4 = st.slider("Zahl der Tests n (markiert in den Tabellen)", *bounds("n_slider"), step=1, key="n_slider", help="Wie viele Tests zusammen ausgewertet werden.")
m = _multi(prior_p, float(q4), int(n4), c_v)
l1, l2 = st.columns(2)
with l1:
    st.plotly_chart(build_evsi_n(m), width="stretch", key="evsi_n")
with l2:
    st.plotly_chart(build_increments(m), width="stretch", key="increments")
st.plotly_chart(build_net(m), width="stretch", key="net_chart")
st.markdown(f"**Gewählte Testzahl n = {int(n4)}:** EVSI = {ev.de(m.values[int(n4)], 2)} GE ({ev.pct(m.ratio_n)} des EVPI von {ev.de(m.evpi, 1)} GE), Nettowert {ev.de(m.net[int(n4)], 2)} GE.")
if not m.measured_cell:
    st.caption("Hinweis: Diese Kombination aus Vorab und Trefferquote gehört nicht zu den 16 Zellen (Vorab 0,40 / 0,45 / 0,50 / 0,60 mal q = 0,6 / 0,7 / 0,8 / 0,9), in denen die Aussagen der README über die Form von EVSI(n) gezählt sind.")
st.markdown("**Einzeln wertlos, zusammen nicht: ein Test mit Trefferquote 0,55**")
_table([[k, ev.de(v, 3)] for k, v in _worthless(prior_p)], ["Zahl der Tests n", "EVSI (GE)"])
kind, text = ev.verdict_multi(m)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 5: die Falle -----------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Den Wert aus der Trefferquote ablesen?")
st.caption("Die Faustregeln „Korrelation × EVPI“ und „Korrelation² × EVPI“ gegen den wahren Wert; darunter zwei verschiedene Tests, einzeln und gemeinsam. Korrelation heißt hier der Phi-Koeffizient "
           "von Zustand und Testergebnis (bei Vorab 1/2 und symmetrischem Test gleich 2q − 1), nicht eine Korrelation in einem Normalmodell. Zähleinheit: Rasterpunkt (q) und Testpaar.")
f1, f2, f3 = st.columns(3)
with f1:
    q5 = st.slider("Trefferquote q des Tests", *bounds("q5_slider"), step=0.01, key="q5_slider", help="Test für den Vergleich der Faustregeln mit dem wahren Wert.")
with f2:
    qa = st.slider("Test A: Trefferquote", *bounds("qa_slider"), step=0.01, key="qa_slider", help="Erster Test des Paares.")
with f3:
    qb = st.slider("Test B: Trefferquote", *bounds("qb_slider"), step=0.01, key="qb_slider", help="Zweiter Test des Paares; bedingt unabhängig von A.")
t = _trap(prior_p, float(q5), float(qa), float(qb))
g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(build_trap_bars(t), width="stretch", key="trap_bars")
with g2:
    st.plotly_chart(build_trap_curves(t), width="stretch", key="trap_curves")
st.markdown(f"**Die drei Regeln auf dem Raster q = 0,50 bis 1,00 (51 Punkte), Vorab {ev.de(prior_p, 3)}, EVPI {ev.de(t.evpi, 1)} GE**")
_table(
    [["Korrelation · EVPI (bei Vorab 1/2: (2q − 1) · EVPI)", ev.de(t.rules["corr"][0], 2, sign=True), ev.de(t.rules["corr"][1], 2), ev.pct(abs(t.rules["corr"][0]) / t.evpi) if t.evpi > 0 else "–", f"{t.rules['corr'][2]} von 51"],
     ["Korrelation² · EVPI", ev.de(t.rules["corr2"][0], 2, sign=True), ev.de(t.rules["corr2"][1], 2), ev.pct(abs(t.rules["corr2"][0]) / t.evpi) if t.evpi > 0 else "–", f"{t.rules['corr2'][2]} von 51"],
     ["q · EVPI (Trefferquote × EVPI)", ev.de(t.rules["rate"][0], 2, sign=True), ev.de(t.rules["rate"][1], 2), ev.pct(abs(t.rules["rate"][0]) / t.evpi) if t.evpi > 0 else "–", f"{t.rules['rate'][2]} von 51"]],
    ["Regel", "größter Fehler (GE)", "bei q =", "in % von EVPI (Nenner: EVPI)", "Punkte mit Fehler ≤ 5 % von EVPI"],
)
st.markdown("**Zwei Tests, 51 × 51 Raster (Zähleinheit Testpaar): gemeinsam gegen Summe der Einzelwerte**")
st.plotly_chart(build_pair_heatmap(t), width="stretch", key="pair_heatmap")
pc = t.pair_counts
_table(
    [[ev.de_int(pc["pairs"]), ev.de_int(pc["sub"]), ev.de_int(pc["super"]), ev.de_int(pc["equal"]), ev.de_int(pc["equal_with_zero_test"])],
     [ev.de_int(pc["both_pos"]), ev.de_int(pc["both_pos_sub"]), ev.de_int(pc["both_pos_super"]), ev.de_int(pc["both_pos_equal"]), "–"]],
    ["Paare", "gemeinsam weniger als die Summe", "gemeinsam mehr als die Summe", "gemeinsam gleich der Summe", "von den gleichen: mit einem wertlosen Test"],
)
st.caption(f"Erste Zeile: alle {ev.de_int(pc['pairs'])} Paare; zweite Zeile: nur Paare, in denen beide Tests einzeln etwas wert sind ({ev.de_int(pc['both_pos'])} Paare). "
           f"Beide einzeln wertlos: {pc['both_zero']} Paare, davon gemeinsam wertvoll {pc['both_zero_joint_pos']}.")
st.markdown("**Andere Vorab-Werte: Korrelation × EVPI gegen den wahren Wert (je 51 Rasterpunkte)**")
_table(
    [[ev.de(pp, 2), ev.de(e, 1), ev.de(q0, 4), f"{bw} von {tot}", f"{up} von 51", ev.de(mx, 1)] for pp, e, q0, bw, tot, up, mx in t.priors_rows],
    ["Vorab", "EVPI (GE)", "wertlos für 1/2 ≤ q ≤ q0", "wahrer Wert zwischen den Regeln (nur q > q0)", "Korrelation · EVPI nicht unter dem wahren Wert", "größter Fehler (GE)"],
)
kind, text = ev.verdict_trap(t)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wo es weitergeht |
|---|---|---|
| **Linearer Nutzen** (Erwartungswert in GE) | EVPI ≥ EVSI ≥ 0 und die Zerlegung gelten so nur bei linearem Nutzen oder in der Zahlungsbereitschafts-Definition; unter Risikoaversion kann EVSI exakt 0 werden, obwohl er für den Erwartungswert positiv ist. | [`information-risikoaversion-demo`](https://github.com/sebastian-hanisch/information-risikoaversion-demo) |
| **Vorab und Trefferquote sind bekannte Zahlen** | Sind sie nur geschätzt, verschieben Fehler in beiden die Werte und die Schwellen; hier nicht gerechnet. | [`kipp-schwelle-demo`](https://github.com/sebastian-hanisch/kipp-schwelle-demo) (Wahrscheinlichkeit unbekannt) |
| **Zwei Zustände, zwei Alternativen** | Mehr Zustände, Alternativen oder Testergebnisse sind nicht gerechnet; der Rollback ist derselbe, die Zählungen dieser Seite gelten dort nicht. | – |
| **Mehrere Tests gleichartig, symmetrisch, bedingt unabhängig** | Tests mit gemeinsamer Fehlerursache bringen vermutlich weniger (hier nicht gerechnet und nicht gemessen). Die Aussagen über die Form von EVSI(n) sind nur für 16 Zellen gezählt. | – |
| **Kosten je Test, alle Tests vorab** | Fixkosten und die Wahl des Abbruchzeitpunkts nach den Ergebnissen (sequentielle Tests) sind nicht gerechnet. | [`baum-strategie-demo`](https://github.com/sebastian-hanisch/baum-strategie-demo) (Strategien im Baum) |
| **EVSI ist exakt** (diskreter Test) | Bei stetigen Parametern wird EVSI zum Schätzproblem (Erwartung in der Erwartung); hier wird nichts geschätzt. | `geschachtelte-erwartungen-demo` (Monte-Carlo-Demos) |
| **Das Zahlenbeispiel ist ausgedacht** | Alle Zahlen sind Eingaben eines Modells; für einen echten Fall müssen Auszahlungen, Vorab und Trefferquote belegt sein. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/testwert-evsi-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind zwei Zustände (hoch, niedrig) mit Vorab $p$ und $1-p$, Auszahlungen $u(a,s)$ in GE, ein Test mit Sensitivität und Spezifität und Testkosten $c$. Gesucht ist der Wert $EVSI$ des Tests und daraus, ob er sich lohnt.

| Symbol | Bedeutung |
|---|---|
| $s \in \{\text{hoch}, \text{niedrig}\}$ | Zustand (Nachfrage) |
| $a \in \{G, K\}$ | Alternative: G groß bauen, K klein bauen oder mieten |
| $p = P(\text{hoch})$ | Vorab-Wahrscheinlichkeit der hohen Nachfrage vor dem Test |
| $u(a,s)$ | Auszahlung in GE (Ertrag, nicht Kosten) |
| $y \in \{+,-\}$ | Testergebnis |
| $se$, $sp$, $q$ | Sensitivität $P(+\mid\text{hoch})$, Spezifität $P(-\mid\text{niedrig})$; Trefferquote $q$, wenn $se = sp = q$ |
| $EV_0$, $EVPI$, $EVSI$ | Wert ohne Information, mit vollkommener Information, mit dem Test (Differenzen zu $EV_0$) |
| $c$, $ENBS$ | Kosten je Test in GE, Nettonutzen $EVSI - c$ |
| $n$, $k$ | Zahl der Tests, Zahl der positiven Ergebnisse |
| $p^*$, $q_0$, $q^*(c)$ | Schwelle der Wahl, Wertlos-Schwelle, Mindest-Trefferquote bei Kosten $c$ |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/testwert-evsi-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* zwei Zustände (hoch, niedrig) mit Vorab $p$ und $1-p$, Auszahlungen $u(a,s)$ in GE, ein Test mit Sensitivität und Spezifität, Testkosten $c$. *Gesucht:* der Wert $EVSI$ des Tests und,
daraus, ob er sich lohnt. Das Modell ist vollständig für diesen Fall (zwei Zustände, zwei Alternativen, ein Test mit zwei Ergebnissen; die Erweiterung auf mehrere Tests steht unten); was nicht
gerechnet ist, steht im Abschnitt 🚧.

| Symbol | Bedeutung |
|---|---|
| $s \in \{\text{hoch}, \text{niedrig}\}$ | Zustand (Nachfrage) |
| $a \in \{G, K\}$ | Alternative: G groß bauen, K klein bauen oder mieten |
| $p = P(\text{hoch})$ | Vorab-Wahrscheinlichkeit (Prior), $p_{\text{hoch}} = p$, $p_{\text{niedrig}} = 1-p$ |
| $u(a,s)$ | Auszahlung in GE: $u(G,\text{hoch}) = 200$, $u(G,\text{niedrig}) = -100$, $u(K,\cdot) = 30$ |
| $y \in \{+,-\}$ | Testergebnis |
| $se$, $sp$ | Sensitivität $P(+\mid\text{hoch})$, Spezifität $P(-\mid\text{niedrig})$; $q$ = Trefferquote, wenn $se = sp = q$ |
| $EV_0$, $EVPI$, $EVSI$ | Wert ohne Information, mit vollkommener Information, mit dem Test (Differenzen zu $EV_0$) |
| $c$, $ENBS$ | Kosten je Test (GE), Nettonutzen $EVSI - c$ |
| $n$, $k$ | Zahl der Tests, Zahl der positiven Ergebnisse |
| $p^*$, $q_0$, $q^*(c)$ | Schwelle der Wahl, Wertlos-Schwelle, Mindest-Trefferquote bei Kosten $c$ |

**Ohne Test.**
$$EV_0 = \max_a \sum_s p_s\,u(a,s), \qquad EV(G) = 300\,p - 100, \quad EV(K) = 30.$$
$$p^* = \tfrac{13}{30} \quad (300\,p^* - 100 = 30): \text{ für } p \ge p^* \text{ wählt man } G\text{, darunter } K.$$
$$EVPI = \sum_s p_s \max_a u(a,s) - EV_0.$$

**Mit Test (Bayes umgedreht, Rollback).** Verbundwahrscheinlichkeit und Posterior:
$$P(s,y) = p_s\,P(y\mid s), \qquad P(y) = \sum_s P(s,y), \qquad P(\text{hoch}\mid y) = \frac{P(\text{hoch},y)}{P(y)}.$$
$$EVSI = \sum_y \max_a \sum_s P(s,y)\,u(a,s) - EV_0, \qquad ENBS = EVSI - c.$$
Zweiter Rechenweg über den Verlust: $EVPI - EVSI$ ist der erwartete Opportunitätsverlust $\sum_y \sum_s P(s,y)\,\bigl(\max_{a'} u(a',s) - u(a_y,s)\bigr)$ der gewählten Regel $y \mapsto a_y$.

**Vorab 1/2, symmetrischer Test.** Hier ist $P(+) = \tfrac12$ für jedes $q$, und für $q \ge \tfrac12$ gilt
$$EVSI(q) = \max\bigl(0,\; 150\,q - 85\bigr), \qquad q_0 = \tfrac{17}{30}, \qquad q^*(c) = \frac{85 + c}{150} \;\; (c < 65).$$
Für $q < \tfrac12$ wird spiegelbildlich gelesen, $EVSI(q) = EVSI(1-q)$: der Test ist nur für $\tfrac{13}{30} \le q \le \tfrac{17}{30}$ wertlos (zum Beispiel $EVSI(0{,}1) = 50$ GE).
Allgemein ist $q_0(p)$ die größte Trefferquote, bei der der Posterior nach $+$ und nach $-$ auf derselben Seite von $p^*$ liegt.

**Mehrere Tests.** $n$ gleichartige, bedingt unabhängige symmetrische Tests, ausgewertet über die Zahl $k$ der positiven Ergebnisse:
$$P(k\mid \text{hoch}) = \binom{n}{k} q^k (1-q)^{n-k}, \qquad P(k\mid \text{niedrig}) = \binom{n}{k} (1-q)^k q^{n-k}.$$
Der Zuwachs je Test ist $EVSI(n) - EVSI(n-1)$; die Abbruchregel kauft weiter, solange er größer als $c$ ist, das Optimum maximiert $EVSI(n) - c\,n$.

**Faustregel „Korrelation × EVPI“.** Phi-Koeffizient von Zustand und Ergebnis:
$$\varphi = \frac{P(\text{hoch},+) - p\,P(+)}{\sqrt{p\,(1-p)\,P(+)\,(1-P(+))}} \;=\; 2q - 1 \quad (p = \tfrac12,\; se = sp = q).$$
Verglichen werden $\varphi \cdot EVPI$ und $\varphi^2 \cdot EVPI$ mit $EVSI$.

**Literatur** (Gleichungen und Befunde sinngemäß, in eigenen Worten): Raiffa und Schlaifer (1961), *Applied Statistical Decision Theory*, Rollback Abschnitt 1.2.1, Wert der Information Kapitel 4
(Gleichungen 4-29 bis 4-36; additive Nutzen Abschnitt 4.2); Howard und Matheson (2005), „Influence Diagrams“, *Decision Analysis* 2 (3), 127–143, Abschnitte 6 und 7 (Baum umdrehen) und 15;
Bickel (2008), „The Relationship Between Perfect and Imperfect Information in a Two-Action Risk-Sensitive Problem“, *Decision Analysis* 5 (3), 116–128, Einleitung. Die Zahlen dieser Seite sind
eigene Nachrechnung in diesem Modell; die Zahlen aus Howard und Matheson sind nicht nachgerechnet.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Entscheidungsanalyse: Information, Risiko, Grenzen](https://sebastianhanisch.net/konzepte-entscheidungsanalyse.html)."
)
