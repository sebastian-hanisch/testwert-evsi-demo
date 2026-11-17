"""Kennzahlen, Zählungen und Verdikte der fünf Experimente.

Alle Rechnungen laufen über den Kern `tev_algorithm.py` in Gleitkomma; Zählungen auf Rastern nutzen die Toleranz `A.ZERO_TOL` (die Tests vergleichen sie mit
der exakten Rechnung in Brüchen). Ein Verdikt ist eine reine Funktion `verdict_*(Analyse) -> (Art, Text)`: die App ruft `getattr(st, Art)(Text)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import tev_algorithm as A
import tev_constants as C
import tev_scenario as S

TOL = A.ZERO_TOL
P_STAR = float(S.P_STAR)   # 13/30
Q_GRID = np.arange(50, 101) / 100.0                 # Trefferquoten 0,50 bis 1,00 (51 Rasterpunkte)
SE_GRID = np.arange(0, C.GRID_STEPS + 1) / 100.0    # Sensitivität und Spezifität 0 bis 1 (101 Rasterpunkte)


# --- Formatierung (deutsch: Dezimalkomma, echtes Minuszeichen) -------------------------------------------------------------------------------

def de(x: float, nd: int = 1, sign: bool = False) -> str:
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    if float(s) == 0.0:                     # „-0,0“ und „+0,0“ vermeiden: null hat kein Vorzeichen
        s = s.lstrip("+-")
    return s.replace(".", ",").replace("-", "−")


def pct(x: float, nd: int = 1) -> str:
    """Anteil 0 bis 1 als Prozentzahl mit Dezimalkomma, z. B. 0,538 -> „53,8 %“."""
    return de(100.0 * x, nd) + " %"


def de_int(n: int) -> str:
    """Ganzzahl mit Leerzeichen als Tausendertrenner ab 1 000: 10201 -> „10 201“."""
    return f"{int(n):,}".replace(",", " ")


def alt(name: str) -> str:
    return {"G": "G", "K": "K"}.get(name, name)


def tests(n: int) -> str:
    """„1 Test“, „3 Tests“."""
    return f"{n} Test" if n == 1 else f"{n} Tests"


@dataclass(frozen=True)
class Settings:
    p: float = C.DEFAULT_P
    se: float = C.DEFAULT_SE
    sp: float = C.DEFAULT_SP
    c: float = C.DEFAULT_C
    q4: float = C.DEFAULT_Q4
    n: int = C.DEFAULT_N
    q5: float = C.DEFAULT_Q5
    qa: float = C.DEFAULT_QPAIR
    qb: float = C.DEFAULT_QPAIR
    rz_seed: int = C.DEFAULT_RZ_SEED
    rz_m: int = C.DEFAULT_RZ_M


# --- Experiment 1: von Hand ----------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TestAnalysis:
    p: float
    se: float
    sp: float
    c: float
    ev: dict            # Erwartungswert je Alternative ohne Test
    ev0: float
    best0: str
    evpi: float
    rows: list          # Rollback je Ergebnis (+, -)
    ev_with: float
    evsi: float
    loss_with: float    # erwarteter Opportunitätsverlust mit Test = EVPI - EVSI
    loss_without: float  # ... ohne Test = EVPI
    ratio: float | None  # EVSI/EVPI, None bei EVPI = 0
    enbs: float
    worthless: bool
    joint: list         # [(P(hoch, y), P(niedrig, y))] je Ergebnis


def analyse_test(s: Settings) -> TestAnalysis:
    prior = S.prior_of(s.p)
    lik = A.lik_bin(s.se, s.sp)
    rows = A.rollback(prior, S.PAY, lik)
    e0, ep, es = A.ev0(prior, S.PAY), A.evpi(prior, S.PAY), A.evsi(prior, S.PAY, lik)
    return TestAnalysis(
        p=s.p, se=s.se, sp=s.sp, c=s.c, ev={k: A.ev_of(prior, a) for k, a in S.PAY.items()}, ev0=e0, best0=A.best_action(prior, S.PAY), evpi=ep,
        rows=rows, ev_with=e0 + es, evsi=es, loss_with=A.loss_with_test(prior, S.PAY, lik), loss_without=ep,
        ratio=None if ep <= TOL else es / ep, enbs=es - s.c, worthless=es <= TOL, joint=A.joint(prior, lik),
    )


def verdict_test(a: TestAnalysis) -> tuple[str, str]:
    """Verdikt zu Experiment 1: ändert der Test die Wahl, was ist er wert, lohnt er bei den Kosten?"""
    if a.evpi <= TOL:
        return "info", "Hier hat selbst vollkommene Information keinen Wert (EVPI = 0): eine Alternative ist in jedem Zustand mindestens so gut."
    if a.worthless:
        sees = " (se + sp > 1: er sieht nützlich aus)" if a.se + a.sp > 1 + 1e-12 else ""
        price = "bringt auch ohne Kosten nichts" if a.c <= TOL else f"lohnt auch zum Preis von {de(a.c, 1)} GE nicht"
        return "warning", (f"**Wertlos:** nach beiden Ergebnissen dieselbe Wahl ({alt(a.rows[0]['choice'])}), EVSI = 0 GE{sees}. Der Test ändert nichts, "
                           f"er {price} (ENBS = {de(a.enbs, 1, sign=True)} GE).")
    plus, minus = a.rows[0], a.rows[1]
    head = (f"Test mit Sensitivität {de(a.se, 2)} und Spezifität {de(a.sp, 2)}: nach + {alt(plus['choice'])}, nach − {alt(minus['choice'])}. "
            f"**EVSI = {de(a.evsi, 1)} GE** = {pct(a.ratio)} des EVPI von {de(a.evpi, 1)} GE (Wert ohne Test EV0 = {de(a.ev0, 1)} GE).")
    if a.enbs > TOL:
        return "success", head + f" Er lohnt, solange er weniger als {de(a.evsi, 1)} GE kostet; bei {de(a.c, 1)} GE ist ENBS = {de(a.enbs, 1, sign=True)} GE."
    return "warning", head + f" Bei Kosten von {de(a.c, 1)} GE lohnt er nicht: ENBS = {de(a.enbs, 1, sign=True)} GE (nicht kaufen)."


# --- Experiment 2: wann ist ein Test wertlos? ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ZeroAnalysis:
    p: float
    evpi: float
    q_grid: np.ndarray
    evsi_q: np.ndarray           # EVSI(q) auf Q_GRID (symmetrisch, se = sp = q)
    q0: float                    # größte wertlose Trefferquote (geschlossen)
    zero_q: int                  # Rasterpunkte q (von 51) mit EVSI = 0
    ratio_at: dict               # EVSI/EVPI bei q = 0,6 / 0,7 / 0,8 / 0,9 / 0,95
    grid: np.ndarray             # EVSI/EVPI über (se, sp), 101 x 101 (Zeilen se, Spalten sp)
    zero_mask: np.ndarray        # EVSI = 0
    zero_total: int
    n_points: int
    useful_total: int            # Punkte mit se + sp > 1
    zero_useful: int
    p_curve: np.ndarray
    q0_curve: np.ndarray


def analyse_zero(p: float) -> ZeroAnalysis:
    prior = S.prior_of(p)
    ep = A.evpi(prior, S.PAY)
    evsi_q = A.evsi_bin_array(prior, S.PAY, Q_GRID, Q_GRID)
    z = A.evsi_bin_array(prior, S.PAY, SE_GRID[:, None], SE_GRID[None, :])
    zero = z <= TOL
    idx = np.arange(C.GRID_STEPS + 1)
    useful = (idx[:, None] + idx[None, :]) > C.GRID_STEPS      # se + sp > 1 in ganzen Hundertsteln (kein Gleitkommavergleich)
    p_curve = np.arange(1, 100) / 100.0
    return ZeroAnalysis(
        p=p, evpi=ep, q_grid=Q_GRID, evsi_q=evsi_q, q0=float(A.q0_closed(p, P_STAR)), zero_q=int(np.sum(evsi_q <= TOL)),
        ratio_at={q: (float(A.evsi(prior, S.PAY, A.lik_bin(q, q))) / ep if ep > TOL else 0.0) for q in (0.6, 0.7, 0.8, 0.9, 0.95)},
        grid=(z / ep if ep > TOL else np.zeros_like(z)), zero_mask=zero, zero_total=int(zero.sum()), n_points=int(zero.size),
        useful_total=int(useful.sum()), zero_useful=int((zero & useful).sum()),
        p_curve=p_curve, q0_curve=np.array([float(A.q0_closed(float(x), P_STAR)) for x in p_curve]),
    )


def verdict_zero(a: ZeroAnalysis) -> tuple[str, str]:
    """Verdikt zu Experiment 2 (Vorab und Se/Sp-Raster)."""
    zero_share = a.zero_total / a.n_points
    useful_share = a.zero_useful / a.useful_total
    if abs(a.p - P_STAR) < 1e-9:
        head = (f"Vorab = 13/30 = {de(a.p, 4)}, der Schwelle der Wahl: jeder Test mit einer Trefferquote über 1/2 hat einen Wert über 0 "
                f"(wertlos nur bei q = {de(a.q0, 1)}). ")
    elif abs(a.p - 0.5) < 1e-12:
        head = (f"Vorab 1/2: ein symmetrischer Test ist von der Trefferquote q = 1/2 bis q = 17/30 = {de(a.q0, 3)} wertlos, darüber gilt "
                f"EVSI = 150 q − 85 GE (bei q = 1 das EVPI von {de(a.evpi, 0)} GE); für q < 1/2 wird spiegelbildlich gelesen, für q < 13/30 ist der Test nicht wertlos. ")
    else:
        head = (f"Vorab {de(a.p, 3)}: ein symmetrischer Test ist von der Trefferquote q = 1/2 bis q = {de(a.q0, 4)} wertlos "
                f"(auf dem Raster q = 0,50 bis 1,00 sind es {a.zero_q} von 51 Punkten); für q < 1/2 wird spiegelbildlich gelesen. ")
    tail = (f"Auf dem Raster aus Sensitivität und Spezifität (Zähleinheit Rasterpunkt, Schritt 0,01) hat EVSI an {de_int(a.zero_total)} von {de_int(a.n_points)} Punkten "
            f"({pct(zero_share)}) den Wert 0, darunter {de_int(a.zero_useful)} von {de_int(a.useful_total)} der Punkte mit se + sp > 1 ({pct(useful_share)}), "
            "also von Tests, die nützlich aussehen.")
    return "info", head + tail


# --- Experiment 3: ab welcher Trefferquote lohnt der Test? -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CostAnalysis:
    p: float
    c: float
    evpi: float
    q_star: float | None         # None: der vollkommene Test deckt die Kosten nicht (c >= EVPI)
    c_curve: np.ndarray
    q_curve: np.ndarray          # q*(c) auf c_curve (nan, wo c >= EVPI)
    spec_rows: list             # [(se, Mindest-Spezifität oder None, EVSI(se, 1))]
    prior_rows: list            # [(p, EVPI(p), q0(p), q*(c) oder None)] für die Vorab-Werte aus C.TRAP_PRIORS


def analyse_cost(p: float, c: float) -> CostAnalysis:
    prior = S.prior_of(p)
    ep = A.evpi(prior, S.PAY)
    q_grid = np.linspace(0.5, 1.0, 5001)
    ev_q = A.evsi_bin_array(prior, S.PAY, q_grid, q_grid)
    c_curve = np.linspace(0.0, C.C_MAX, 129)
    q_curve = np.full(c_curve.shape, np.nan)
    for i, cv in enumerate(c_curve):
        if cv < ep - TOL:
            q_curve[i] = float(np.interp(cv + 1e-12, ev_q, q_grid)) if ev_q[-1] > cv else np.nan   # ev_q ist nicht fallend
    rows = []
    for se in C.SPEC_SENSITIVITIES:
        rows.append((se, A.min_specificity(prior, S.PAY, se, c), float(A.evsi(prior, S.PAY, A.lik_bin(se, 1.0)))))
    prows = []
    for pp in C.TRAP_PRIORS:
        pr = S.prior_of(pp)
        prows.append((pp, float(A.evpi(pr, S.PAY)), float(A.q0_closed(pp, P_STAR)), A.q_star(pr, S.PAY, c)))
    return CostAnalysis(p=p, c=c, evpi=ep, q_star=A.q_star(prior, S.PAY, c), c_curve=c_curve, q_curve=q_curve, spec_rows=rows, prior_rows=prows)


def verdict_cost(a: CostAnalysis) -> tuple[str, str]:
    """Verdikt zu Experiment 3 (Mindest-Trefferquote bei Testkosten c)."""
    if a.evpi <= TOL:
        return "info", "Hier hat Information keinen Wert (EVPI = 0), also lohnt kein Test."
    share = a.c / a.evpi
    if a.q_star is None:
        return "warning", (f"Testkosten c = {de(a.c, 1)} GE ({pct(share)} des EVPI von {de(a.evpi, 1)} GE): **nie**. Auch ein fehlerfreier Test (q = 1) bringt höchstens "
                           f"das EVPI, kostet er so viel oder mehr, bleibt ENBS ≤ 0.")
    if a.c <= TOL:
        return "info", (f"Ohne Testkosten lohnt jeder Test mit EVSI > 0, also ab der Wertlos-Schwelle q = {de(a.q_star, 4)} (Vorab {de(a.p, 3)}); darunter ist er wertlos.")
    text = (f"Testkosten c = {de(a.c, 1)} GE ({pct(share)} des EVPI von {de(a.evpi, 1)} GE): ein symmetrischer Test lohnt ab der Trefferquote **$q^*$ = {de(a.q_star, 4)}** "
            f"(Vorab {de(a.p, 3)}).")
    if abs(a.p - 0.5) < 1e-12:
        text += " Bei Vorab 1/2 gilt $q^*(c) = (85 + c)/150$; ab c = 65 gibt es kein q mehr, das reicht."
    return "success", text


# --- Experiment 4: mehrere Tests ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MultiAnalysis:
    p: float
    q: float
    n: int
    c: float
    evpi: float
    values: list            # EVSI(0..60)
    inc: list               # Zuwachs je Test (None, EVSI(1) − EVSI(0), ...)
    rises: list             # Stellen n = 2..60, an denen der Zuwachs steigt
    n_places: int           # Zahl der Stellen n = 2..60 (59)
    odd_rises: int
    net: list               # EVSI(n) - c n
    n_star: int             # Optimum
    n_rule: int             # Abbruchregel
    first_pos: int | None   # kleinste Testzahl mit EVSI > 0
    concave_even: bool
    concave_odd: bool
    measured_cell: bool
    ratio_n: float          # EVSI(n)/EVPI für die gewählte Testzahl


def analyse_multi(p: float, q: float, n: int, c: float) -> MultiAnalysis:
    prior = S.prior_of(p)
    ep = A.evpi(prior, S.PAY)
    vals = A.evsi_series(prior, S.PAY, q, C.N_SERIES)
    inc = A.increments(vals)
    rises, places = A.count_rises(vals, TOL)
    net = [vals[k] - c * k for k in range(len(vals))]
    d2 = {par: [vals[k] - vals[k - 2] for k in range(par + 2, len(vals), 2)] for par in (0, 1)}
    conc = {par: all(d2[par][i] >= d2[par][i + 1] - TOL for i in range(len(d2[par]) - 1)) for par in (0, 1)}
    cell = any(abs(p - pp) < 1e-12 for pp in C.CELL_PRIORS) and any(abs(q - qq) < 1e-12 for qq in C.CELL_QS)
    return MultiAnalysis(
        p=p, q=q, n=n, c=c, evpi=ep, values=vals, inc=inc, rises=rises, n_places=places, odd_rises=sum(1 for r in rises if r % 2 == 1), net=net,
        n_star=A.best_n(vals, c, TOL), n_rule=A.rule_n(vals, c, TOL), first_pos=next((k for k in range(1, len(vals)) if vals[k] > TOL), None),
        concave_even=conc[0], concave_odd=conc[1], measured_cell=cell, ratio_n=(vals[n] / ep if ep > TOL else 0.0),
    )


def worthless_table(p: float, q: float = 0.55, ns=(1, 2, 3, 5, 10, 20, 50)) -> list:
    """EVSI(n) für einen einzeln wertlosen Test mit Trefferquote q (Standard 0,55) bei den Testzahlen ns."""
    prior = S.prior_of(p)
    vals = A.evsi_series(prior, S.PAY, q, max(ns))
    return [(k, float(vals[k])) for k in ns]


def verdict_multi(a: MultiAnalysis) -> tuple[str, str]:
    """Verdikt zu Experiment 4 (Zuwächse, Abbruchregel gegen Optimum)."""
    if a.evpi <= TOL:
        return "info", "Hier hat Information keinen Wert (EVPI = 0), auch mehrere Tests bringen nichts."
    incs = "; ".join(de(a.inc[k], 2) for k in range(1, 5))
    if a.first_pos is None:
        why = ("Ein Test ohne Aussage (q = 1/2) wird durch Wiederholen nicht besser." if abs(a.q - 0.5) < 1e-12
               else "Für die gewählten Werte ändert auch die Gesamtheit der Ergebnisse die Wahl nie.")
        return "warning", f"Trefferquote {de(a.q, 2)}: auch {C.N_SERIES} gleichartige Tests zusammen sind wertlos (EVSI = 0). {why}"
    if a.rises:
        shape = (f"Die ersten Zuwächse je Test: {incs} GE. Der Zuwachs steigt an {len(a.rises)} von {a.n_places} Stellen (n = 2 bis {C.N_SERIES}), davon {a.odd_rises} bei ungeraden n: "
                 "EVSI ist in der Testzahl nicht konkav.")
    else:
        shape = f"Die ersten Zuwächse je Test: {incs} GE. Hier fällt der Zuwachs über n = 2 bis {C.N_SERIES} durchgehend: EVSI ist in der Testzahl konkav."
    first = "" if a.first_pos == 1 else f" Ein einzelner Test ist wertlos, erst ab {a.first_pos} Tests ist EVSI > 0."
    if a.c <= TOL:
        cost = (f" Ohne Testkosten (c = 0) lohnt jeder weitere Test, solange er etwas bringt; bei {tests(a.n)} ist EVSI = {de(a.values[a.n], 2)} GE "
                f"({pct(a.ratio_n)} des EVPI von {de(a.evpi, 1)} GE).")
        return "info", shape + first + cost
    if a.n_rule == a.n_star:
        cost = (f" Bei Testkosten von {de(a.c, 1)} GE je Test stimmen die Abbruchregel und das Optimum überein: {tests(a.n_star)}, Nettowert {de(a.net[a.n_star], 1)} GE.")
        return "info", shape + first + cost
    cost = (f" **Bei {de(a.c, 1)} GE je Test hört die Abbruchregel nach {tests(a.n_rule)} auf (Nettowert {de(a.net[a.n_rule], 1)} GE), das Optimum liegt bei {tests(a.n_star)} "
            f"({de(a.net[a.n_star], 1)} GE).**")
    return "warning", shape + first + cost


# --- Experiment 5: die Falle -----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TrapAnalysis:
    p: float
    q: float
    qa: float
    qb: float
    evpi: float
    true_value: float
    phi: float
    corr: float                # Korrelation × EVPI
    corr2: float               # Korrelation² × EVPI
    rate: float                # Trefferquote × EVPI (wörtliche Lesart)
    rules: dict                # Regel -> (größter Fehler, q dort, Punkte mit |Fehler| <= 5 % von EVPI) auf dem Raster q = 0,50 bis 1,00
    corr_above: int            # Rasterpunkte (von 51), an denen Korrelation × EVPI nicht unter dem wahren Wert liegt
    pair_joint: float
    pair_a: float
    pair_b: float
    pair_counts: dict          # sub, super, equal über 51 x 51 Paare; both_pos_sub/eq/super; both_zero, both_zero_joint_pos
    pair_diff: np.ndarray      # gemeinsam minus Summe der Einzelwerte, 51 x 51
    priors_rows: list = field(default_factory=list)   # je Vorab: (p, EVPI, q0, zwischen, gesamt, nicht unter, größter Fehler)


def _rules_on_grid(prior, ep):
    true = A.evsi_bin_array(prior, S.PAY, Q_GRID, Q_GRID)
    phi = A.phi_array(prior, Q_GRID, Q_GRID)
    return true, phi, {"corr": phi * ep, "corr2": phi ** 2 * ep, "rate": Q_GRID * ep}


def analyse_trap(p: float, q: float, qa: float, qb: float) -> TrapAnalysis:
    prior = S.prior_of(p)
    ep = A.evpi(prior, S.PAY)
    true, phi, rules = _rules_on_grid(prior, ep)
    stats = {}
    for name, vals in rules.items():
        err = vals - true
        k = int(np.argmax(np.abs(err)))
        stats[name] = (float(err[k]), float(Q_GRID[k]), int(np.sum(np.abs(err) <= 0.05 * ep + TOL)))
    # gewählter Punkt
    t = float(A.evsi_bin_array(prior, S.PAY, q, q))
    ph = float(A.phi_array(prior, q, q))
    # Paare
    ev_single = A.evsi_bin_array(prior, S.PAY, Q_GRID, Q_GRID)
    joint = A.evsi_pair_array(prior, S.PAY, Q_GRID, Q_GRID)
    ssum = ev_single[:, None] + ev_single[None, :]
    diff = joint - ssum
    sub, sup = diff < -TOL, diff > TOL
    eq = ~(sub | sup)
    pos = (ev_single > TOL)[:, None] & (ev_single > TOL)[None, :]
    zero = (ev_single <= TOL)[:, None] & (ev_single <= TOL)[None, :]
    counts = {"pairs": int(diff.size), "sub": int(sub.sum()), "super": int(sup.sum()), "equal": int(eq.sum()),
              "both_pos": int(pos.sum()), "both_pos_sub": int((pos & sub).sum()), "both_pos_equal": int((pos & eq).sum()), "both_pos_super": int((pos & sup).sum()),
              "equal_with_zero_test": int((eq & ~pos).sum()), "both_zero": int(zero.sum()), "both_zero_joint_pos": int((zero & (joint > TOL)).sum())}
    pj = float(A.evsi_pair_array(prior, S.PAY, [qa], [qb])[0, 0])
    pa, pb = float(A.evsi_bin_array(prior, S.PAY, qa, qa)), float(A.evsi_bin_array(prior, S.PAY, qb, qb))
    rows = []
    for pp in C.TRAP_PRIORS:
        pr = S.prior_of(pp)
        e = A.evpi(pr, S.PAY)
        tr, ph_, rl = _rules_on_grid(pr, e)
        q0p = float(A.q0_closed(pp, P_STAR))
        above = Q_GRID > q0p + 1e-12
        between = above & (rl["corr2"] <= tr + 1e-9) & (tr <= rl["corr"] + 1e-9)
        rows.append((pp, float(e), q0p, int(between.sum()), int(above.sum()), int(np.sum(rl["corr"] >= tr - 1e-9)), float(np.max(np.abs(rl["corr"] - tr)))))
    return TrapAnalysis(
        p=p, q=q, qa=qa, qb=qb, evpi=ep, true_value=t, phi=ph, corr=ph * ep, corr2=ph ** 2 * ep, rate=q * ep, rules=stats,
        corr_above=int(np.sum(rules["corr"] >= true - TOL)), pair_joint=pj, pair_a=pa, pair_b=pb, pair_counts=counts, pair_diff=diff, priors_rows=rows,
    )


def verdict_trap(a: TrapAnalysis) -> tuple[str, str]:
    """Verdikt zu Experiment 5: wahrer Wert gegen die Faustregeln, Summe gegen gemeinsam."""
    if a.evpi <= TOL:
        return "info", "Hier hat Information keinen Wert (EVPI = 0)."
    a1 = (f"Trefferquote {de(a.q, 2)}: wahr {de(a.true_value, 1)} GE; Korrelation × EVPI = {de(a.corr, 1)} GE, Korrelation² × EVPI = {de(a.corr2, 1)} GE, "
          f"Trefferquote × EVPI = {de(a.rate, 1)} GE (Korrelation = Phi-Koeffizient von Zustand und Ergebnis = {de(a.phi, 3)}).")
    if a.corr_above == 51:
        above = (f" Auf dem Raster q = 0,50 bis 1,00 liegt Korrelation × EVPI an allen 51 Punkten nicht unter dem wahren Wert (Vorab {de(a.p, 3)}): "
                 "als Wert gelesen überschätzt sie, sie wirkt hier wie eine obere Schranke (Zählung auf dem Raster, kein Beweis).")
    else:
        above = (f" Auf dem Raster q = 0,50 bis 1,00 liegt Korrelation × EVPI an {a.corr_above} von 51 Punkten nicht unter dem wahren Wert (Vorab {de(a.p, 3)}): "
                 "hier ist sie keine obere Schranke.")
    summ = a.pair_a + a.pair_b
    if a.pair_joint < summ - TOL:
        rel = "weniger als"
    elif a.pair_joint > summ + TOL:
        rel = "mehr als"
    else:
        rel = "genau"
    pair = (f" Zwei Tests ({de(a.qa, 2)}; {de(a.qb, 2)}): einzeln {de(a.pair_a, 2)} und {de(a.pair_b, 2)} GE, gemeinsam {de(a.pair_joint, 2)} GE, also {rel} die Summe {de(summ, 2)} GE.")
    kind = "warning" if (abs(a.corr - a.true_value) > 0.05 * a.evpi) else "info"
    return kind, a1 + above + pair


# --- Zufallsproblem-Gegenprobe ---------------------------------------------------------------------------------------------------------------

def verdict_random(st: S.ProblemStats) -> tuple[str, str]:
    """Verdikt der Gegenprobe über Zufallsprobleme (Zähleinheit Zufallsproblem)."""
    if st.nondominated == 0:
        return "info", f"Alle {de_int(st.m)} Zufallsprobleme sind dominiert: jeder Test hat dort den Wert 0."
    share0 = st.zero_nd / st.nondominated
    share8 = st.zero_nd_q8 / st.nondominated
    ok = st.chain_violations == 0
    chain = (f"EVPI ≥ EVSI ≥ 0 gilt in allen {de_int(st.m)} Problemen (Toleranz 10⁻⁹ GE)." if ok else
             f"EVPI ≥ EVSI ≥ 0 verletzt in {de_int(st.chain_violations)} von {de_int(st.m)} Problemen (Toleranz 10⁻⁹ GE).")
    text = (f"{de_int(st.m)} Zufallsprobleme: {de_int(st.dominated)} dominiert (eine Alternative ist in beiden Zuständen mindestens so gut wie die andere, daher Wert 0 für jeden Test), {de_int(st.nondominated)} nicht dominiert. {chain} "
            f"An den nicht dominierten ist EVSI = 0 in {pct(share0)} (Test mit zufälligen se, sp) und {pct(share8)} (Test mit Trefferquote 0,8); "
            f"Median EVSI/EVPI {de(st.median_ratio, 3)}, Mittel {de(st.mean_ratio, 3)} (beide über die nicht dominierten Probleme). Die Bedingung „nach beiden Ergebnissen dieselbe Wahl“ stimmt in "
            f"{de_int(st.nondominated - st.cond_violations)} von {de_int(st.nondominated)} nicht dominierten Problemen mit EVSI = 0 überein.")
    return ("success" if ok and st.cond_violations == 0 else "warning"), text
