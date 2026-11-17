"""Kern: Wert eines Tests vor einer Wahl (EVSI) in einem Modell mit zwei Zuständen.

Modell: Zustand 0 = hoch, Zustand 1 = niedrig, Vorab-Wahrscheinlichkeiten `prior = (p, 1 - p)`; `pay[name] = (Auszahlung bei hoch, bei niedrig)`
(Ertrag, nicht Kosten), Bewertung nach dem Erwartungswert (linearer Nutzen). Ein Test hat Ergebnisse y mit der Likelihood-Matrix `lik[s][y] = P(y | s)`.
Alle Funktionen dieses Teils sind typgenerisch: mit `float` (App) oder mit `fractions.Fraction` (Tests, exakt), sie benutzen nur + - * / und max.

Rechenweg (Rollback über den umgedrehten Baum): je Ergebnis y die verbundenen Wahrscheinlichkeiten P(s, y) = P(s) * P(y | s), je Alternative der
verbundene Erwartungswert, davon das Maximum, über y summiert. EVSI = (Wert mit Test) - EV0.

Der zweite Teil (numpy) rechnet Raster vektorisiert, in Gleitkomma, mit Toleranz (`ZERO_TOL`); die Tests vergleichen ihn mit dem exakten Teil in Brüchen.
"""

from __future__ import annotations

from itertools import product
from math import comb

import numpy as np

ZERO_TOL = 1e-9   # EVSI <= ZERO_TOL gilt in Gleitkomma als null (Auszahlungen in GE, ganzzahlig; kleinster echter Wert auf den Rastern liegt weit darüber)


# --- Wert ohne Test ---------------------------------------------------------------------------------------------------------------------------

def ev_of(prior, a):
    """Erwartungswert einer Alternative a = (hoch, niedrig)."""
    return prior[0] * a[0] + prior[1] * a[1]


def best_action(prior, pay):
    """Name der Alternative mit dem größten Erwartungswert; bei Gleichstand die zuerst genannte."""
    best, best_v = None, None
    for name, a in pay.items():
        v = ev_of(prior, a)
        if best_v is None or v > best_v:
            best, best_v = name, v
    return best


def ev0(prior, pay):
    """Wert der besten Wahl ohne Information (EV0)."""
    return max(ev_of(prior, a) for a in pay.values())


def evpi(prior, pay):
    """Wert der vollkommenen Information: Erwartungswert der Wahl mit bekanntem Zustand minus EV0."""
    return prior[0] * max(a[0] for a in pay.values()) + prior[1] * max(a[1] for a in pay.values()) - ev0(prior, pay)


def pstar(pay):
    """Vorab-Wahrscheinlichkeit P(hoch), bei der zwei Alternativen gleich viel wert sind; None, wenn es keine gibt (parallel oder dominiert)."""
    (a, b) = list(pay.values())
    den = (a[0] - b[0]) - (a[1] - b[1])
    if den == 0:
        return None
    p = (b[1] - a[1]) / den
    return p if 0 < p < 1 else None


# --- Tests und ihr Wert -----------------------------------------------------------------------------------------------------------------------

def lik_bin(se, sp):
    """Test mit zwei Ergebnissen: Ergebnis 0 = positiv, 1 = negativ; se = P(+ | hoch), sp = P(- | niedrig)."""
    return [[se, 1 - se], [1 - sp, sp]]


def _powers(x, n):
    """[x^0, ..., x^n] durch wiederholte Multiplikation (kein `**`: die Gleitkomma-Potenz kommt aus der C-Bibliothek und kann in der letzten Stelle abweichen)."""
    out = [x * 0 + 1]
    for _ in range(n):
        out.append(out[-1] * x)
    return out


def lik_n(q, n):
    """n unabhängige gleichartige symmetrische Tests (Trefferquote q); Ergebnis = Zahl k der positiven Ergebnisse, k = 0..n."""
    pq, pr = _powers(q, n), _powers(1 - q, n)
    return [[comb(n, k) * pq[k] * pr[n - k] for k in range(n + 1)],
            [comb(n, k) * pr[k] * pq[n - k] for k in range(n + 1)]]


def lik_pair(l1, l2):
    """Zwei bedingt unabhängige Tests, gemeinsam ausgewertet (Ergebnis = Paar der Einzelergebnisse)."""
    return [[l1[s][a] * l2[s][b] for a in range(len(l1[s])) for b in range(len(l2[s]))] for s in range(2)]


def joint(prior, lik):
    """Verbundwahrscheinlichkeiten je Ergebnis y: Liste von (P(hoch, y), P(niedrig, y))."""
    return [(prior[0] * lik[0][y], prior[1] * lik[1][y]) for y in range(len(lik[0]))]


def ev_with(prior, pay, lik):
    """Wert der Wahl mit Test: je Ergebnis die beste Alternative auf den verbundenen Wahrscheinlichkeiten, über die Ergebnisse summiert."""
    total = 0
    for j0, j1 in joint(prior, lik):
        total += max(j0 * a[0] + j1 * a[1] for a in pay.values())
    return total


def evsi(prior, pay, lik):
    """Erwarteter Wert der Stichprobeninformation (hier: eines Tests): Wert mit Test minus EV0."""
    return ev_with(prior, pay, lik) - ev0(prior, pay)


def rollback(prior, pay, lik):
    """Der umgedrehte Baum: je Ergebnis P(y), Posterior P(hoch | y) (None bei P(y) = 0), Erwartungswert je Alternative und die gewählte Alternative."""
    rows = []
    for y, (j0, j1) in enumerate(joint(prior, lik)):
        py = j0 + j1
        post = None if py == 0 else j0 / py
        evs = {name: (ev_of((post, 1 - post), a) if post is not None else None) for name, a in pay.items()}
        # Wahl auf den verbundenen Wahrscheinlichkeiten (ohne Teilen), Gleichstand: die zuerst genannte Alternative
        pick, pick_v = None, None
        for name, a in pay.items():
            v = j0 * a[0] + j1 * a[1]
            if pick_v is None or v > pick_v:
                pick, pick_v = name, v
        rows.append({"y": y, "p_y": py, "posterior": post, "ev": evs, "choice": pick, "joint_value": pick_v})
    return rows


def loss_with_test(prior, pay, lik):
    """Zweiter Rechenweg: erwarteter Opportunitätsverlust mit Test (ex-post-Mehrkosten gegen einen Hellseher) = EVPI - EVSI."""
    best_hi, best_lo = max(a[0] for a in pay.values()), max(a[1] for a in pay.values())
    total = 0
    for row in rollback(prior, pay, lik):
        a = pay[row["choice"]]
        j0, j1 = joint(prior, lik)[row["y"]]
        total += j0 * (best_hi - a[0]) + j1 * (best_lo - a[1])
    return total


def evsi_rules(prior, pay, lik):
    """Orakel: alle Entscheidungsregeln y -> Alternative aufzählen (|A|^m Regeln), die beste minus EV0."""
    acts = list(pay.values())
    m = len(lik[0])
    best = None
    for rule in product(range(len(acts)), repeat=m):
        v = sum(prior[0] * lik[0][y] * acts[rule[y]][0] + prior[1] * lik[1][y] * acts[rule[y]][1] for y in range(m))
        best = v if best is None or v > best else best
    return best - ev0(prior, pay)


def q0_closed(p, p_star):
    """Größte Trefferquote q >= 1/2, bis zu der ein symmetrischer Test wertlos ist (Vorab p, Schwelle p_star der Wahl); 1 bei p = 0 oder 1."""
    if p <= 0 or p >= 1:
        return p * 0 + 1
    odds, ostar = p / (1 - p), p_star / (1 - p_star)
    return odds / (odds + ostar) if p >= p_star else ostar / (ostar + odds)


# --- Schwellen per Bisektion (EVSI ist konvex in q und sp; nicht fallend für q >= 1/2 bzw. sp >= 1 - se, Minimum 0) -----------------------------------------------------

def q_star(prior, pay, cost, iters=60):
    """Kleinste Trefferquote q eines symmetrischen Tests mit EVSI(q) > cost; None, wenn schon der vollkommene Test (EVPI) die Kosten nicht deckt.

    Bei cost = 0 ist das die Wertlos-Schwelle q0 (Infimum). Bisektion; EVSI(q) ist in q nicht fallend (konvex in q, Minimum 0 bei q = 1/2)."""
    if evpi(prior, pay) <= cost + ZERO_TOL:
        return None
    lo, hi = 0.5, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if evsi(prior, pay, lik_bin(mid, mid)) > cost + ZERO_TOL:
            hi = mid
        else:
            lo = mid
    return hi


def min_specificity(prior, pay, se, cost, iters=60):
    """Kleinste Spezifität sp, bei der ein Test mit Sensitivität se die Kosten übertrifft; None, wenn selbst sp = 1 nicht reicht.

    EVSI(se, sp) ist in sp auf [1 - se, 1] nicht fallend (konvex in sp, Wert 0 bei sp = 1 - se)."""
    if evsi(prior, pay, lik_bin(se, 1.0)) <= cost + ZERO_TOL:
        return None
    lo, hi = max(0.0, 1.0 - se), 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if evsi(prior, pay, lik_bin(se, mid)) > cost + ZERO_TOL:
            hi = mid
        else:
            lo = mid
    return hi


# --- mehrere Tests ----------------------------------------------------------------------------------------------------------------------------

def evsi_series(prior, pay, q, n_max):
    """[EVSI(0), ..., EVSI(n_max)] für n unabhängige symmetrische Tests mit Trefferquote q (EVSI(0) = 0)."""
    return [evsi(prior, pay, lik_n(q, n)) if n > 0 else prior[0] * 0 for n in range(n_max + 1)]


def increments(values):
    """Zuwachs je Test: [None, EVSI(1) − EVSI(0), EVSI(2) − EVSI(1), ...]."""
    return [None] + [values[n] - values[n - 1] for n in range(1, len(values))]


def count_rises(values, tol=0):
    """An wie vielen Stellen n = 2..n_max steigt der Zuwachs (Zuwachs(n) > Zuwachs(n-1) + tol)? Rückgabe (Stellen als Liste, Zahl der Stellen n = 2..n_max)."""
    inc = increments(values)
    rises = [n for n in range(2, len(values)) if inc[n] > inc[n - 1] + tol]
    return rises, len(values) - 2


def best_n(values, cost, tol=0):
    """Optimale Testzahl n* = argmax (EVSI(n) - cost * n); bei Gleichstand (Abstand <= tol) die kleinere Zahl."""
    net = [values[n] - cost * n for n in range(len(values))]
    top = max(net)
    return next(n for n in range(len(net)) if net[n] >= top - tol)


def rule_n(values, cost, tol=0):
    """Abbruchregel „kaufe weiter, solange der nächste Test mehr bringt als er kostet“: Testzahl, bei der sie stoppt."""
    inc = increments(values)
    n = 0
    while n < len(values) - 1 and inc[n + 1] > cost + tol:
        n += 1
    return n


# --- Raster in numpy (Gleitkomma) -------------------------------------------------------------------------------------------------------------

def evsi_bin_array(prior, pay, se, sp):
    """EVSI eines Tests mit zwei Ergebnissen für Felder se, sp (broadcast); Gleitkomma. Gleiche Rechnung wie `evsi` mit `lik_bin`."""
    se, sp = np.asarray(se, dtype=float), np.asarray(sp, dtype=float)
    j_hp, j_np = prior[0] * se, prior[1] * (1.0 - sp)
    j_hm, j_nm = prior[0] * (1.0 - se), prior[1] * sp
    plus = None
    minus = None
    for a in pay.values():
        v_plus, v_minus = j_hp * a[0] + j_np * a[1], j_hm * a[0] + j_nm * a[1]
        plus = v_plus if plus is None else np.maximum(plus, v_plus)
        minus = v_minus if minus is None else np.maximum(minus, v_minus)
    return plus + minus - ev0(prior, pay)


def evsi_pair_array(prior, pay, qa, qb):
    """EVSI zweier bedingt unabhängiger symmetrischer Tests (Trefferquoten qa, qb als Felder) für alle Paare: Feld der Form (len(qa), len(qb))."""
    qa = np.asarray(qa, dtype=float)[:, None]
    qb = np.asarray(qb, dtype=float)[None, :]
    total = 0.0
    for ya, yb in product((0, 1), repeat=2):
        la_h = qa if ya == 0 else 1.0 - qa           # P(Ergebnis ya | hoch): positiv mit q, negativ mit 1 - q
        la_n = 1.0 - qa if ya == 0 else qa           # P(Ergebnis ya | niedrig)
        lb_h = qb if yb == 0 else 1.0 - qb
        lb_n = 1.0 - qb if yb == 0 else qb
        j0, j1 = prior[0] * la_h * lb_h, prior[1] * la_n * lb_n
        best = None
        for a in pay.values():
            v = j0 * a[0] + j1 * a[1]
            best = v if best is None else np.maximum(best, v)
        total = total + best
    return total - ev0(prior, pay)


def phi_array(prior, se, sp):
    """Phi-Korrelation von Zustand und Testergebnis (binär, Felder se, sp); 0, wo eine Varianz null ist."""
    se, sp = np.asarray(se, dtype=float), np.asarray(sp, dtype=float)
    p0, p1 = prior
    p_plus = p0 * se + p1 * (1.0 - sp)
    cov = p0 * se - p0 * p_plus
    den = np.sqrt(p0 * p1 * p_plus * (1.0 - p_plus))
    return np.where(den > 0, cov / np.where(den > 0, den, 1.0), 0.0)


def phi_scalar(prior, se, sp):
    """Phi-Korrelation für einzelne Werte (float)."""
    return float(phi_array(prior, se, sp))
