"""Das Zahlenbeispiel (Kapazitätsfall) und die Zufallsprobleme der Gegenprobe.

Das Beispiel ist ausgedacht: die Auszahlungen sind Zahlen in Geldeinheiten (GE), kein Fall aus der Praxis und keine Messung. Es dient nur dazu, dass sich
jede Rechnung von Hand nachvollziehen lässt: G (groß bauen) bringt +200 GE bei hoher und -100 GE bei niedriger Nachfrage, K (klein bauen oder mieten)
sicher +30 GE; ohne Information gilt P(hoch) = 1/2.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import tev_algorithm as A
import tev_rng as R

PAY = {"G": (200.0, -100.0), "K": (30.0, 30.0)}                     # Gleitkomma (App)
PAY_EXACT = {"G": (Fraction(200), Fraction(-100)), "K": (Fraction(30), Fraction(30))}   # Brüche (Tests)
P_STAR = A.pstar(PAY_EXACT)                                         # 13/30: darüber G, darunter K
NAMES = {"G": "G (groß bauen)", "K": "K (klein bauen oder mieten)"}
STATE_NAMES = ("hoch", "niedrig")


def prior_of(p: float) -> tuple[float, float]:
    return (float(p), 1.0 - float(p))


# --- Zufallsprobleme (Gegenprobe zu den Aussagen über EVSI = 0) ---------------------------------------------------------------------------------

PAY_LO, PAY_HI = -100, 200          # Auszahlungen ganzzahlig gleichverteilt in [PAY_LO, PAY_HI]
PRIOR_LO, PRIOR_HI = 0.05, 0.95     # Vorab P(hoch) gleichverteilt in [PRIOR_LO, PRIOR_HI]


@dataclass(frozen=True)
class ProblemStats:
    m: int                  # Zahl der Zufallsprobleme (Zähleinheit)
    dominated: int          # Probleme, in denen eine Alternative in beiden Zuständen mindestens so gut ist (Wert 0 für jeden Test)
    nondominated: int
    zero_nd: int            # nicht dominierte Probleme mit EVSI = 0 (Tests mit zufälligem se, sp)
    zero_nd_q8: int         # nicht dominierte Probleme mit EVSI = 0 bei Trefferquote 0,8 (se = sp = 0,8)
    chain_violations: int   # Probleme mit EVSI < 0 oder EVSI > EVPI (Toleranz ZERO_TOL)
    cond_violations: int    # Probleme, in denen "beide Posterior auf derselben Seite der Schwelle" nicht mit "EVSI = 0" übereinstimmt (nur nicht dominierte)
    median_ratio: float     # Median von EVSI/EVPI über die nicht dominierten
    mean_ratio: float


def random_problems(seed: int, m: int) -> dict[str, np.ndarray]:
    """m Zufallsprobleme aus einem SplitMix64-Strom (Block 1: Vorab, Blöcke 2 bis 5: vier Auszahlungen, Block 6 und 7: se, sp)."""
    raw = R.splitmix_array(seed, 7 * m)
    u = (raw >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    ints = (raw[m:5 * m] % np.uint64(PAY_HI - PAY_LO + 1)).astype(np.int64) + PAY_LO
    return {
        "p": PRIOR_LO + (PRIOR_HI - PRIOR_LO) * u[0:m],
        "g_hi": ints[0:m].astype(float), "g_lo": ints[m:2 * m].astype(float),      # Alternative 1: Auszahlung bei hoch, bei niedrig
        "k_hi": ints[2 * m:3 * m].astype(float), "k_lo": ints[3 * m:4 * m].astype(float),   # Alternative 2
        "se": u[5 * m:6 * m], "sp": u[6 * m:7 * m],
    }


def _evsi_arrays(d, se, sp):
    p, q = d["p"], 1.0 - d["p"]
    j_hp, j_np, j_hm, j_nm = p * se, q * (1.0 - sp), p * (1.0 - se), q * sp
    plus = np.maximum(j_hp * d["g_hi"] + j_np * d["g_lo"], j_hp * d["k_hi"] + j_np * d["k_lo"])
    minus = np.maximum(j_hm * d["g_hi"] + j_nm * d["g_lo"], j_hm * d["k_hi"] + j_nm * d["k_lo"])
    ev0 = np.maximum(p * d["g_hi"] + q * d["g_lo"], p * d["k_hi"] + q * d["k_lo"])
    evpi = p * np.maximum(d["g_hi"], d["k_hi"]) + q * np.maximum(d["g_lo"], d["k_lo"]) - ev0
    return plus + minus - ev0, evpi


def problem_stats(seed: int, m: int) -> ProblemStats:
    """Zählungen über m Zufallsprobleme; reine Funktion von (seed, m)."""
    d = random_problems(seed, m)
    evsi, evpi = _evsi_arrays(d, d["se"], d["sp"])
    evsi8, _ = _evsi_arrays(d, np.full(m, 0.8), np.full(m, 0.8))
    dom = ((d["g_hi"] >= d["k_hi"]) & (d["g_lo"] >= d["k_lo"])) | ((d["k_hi"] >= d["g_hi"]) & (d["k_lo"] >= d["g_lo"]))
    nd = ~dom
    tol = A.ZERO_TOL
    chain = int(np.sum((evsi < -tol) | (evsi > evpi + tol)))
    # Bedingung "Wahl ändert sich nie": die beste Alternative nach + und nach - stimmt überein (Posterior auf derselben Seite der Schwelle)
    p, q, se, sp = d["p"], 1.0 - d["p"], d["se"], d["sp"]
    j_hp, j_np, j_hm, j_nm = p * se, q * (1.0 - sp), p * (1.0 - se), q * sp
    diff_plus = (j_hp * d["g_hi"] + j_np * d["g_lo"]) - (j_hp * d["k_hi"] + j_np * d["k_lo"])
    diff_minus = (j_hm * d["g_hi"] + j_nm * d["g_lo"]) - (j_hm * d["k_hi"] + j_nm * d["k_lo"])
    same_side = ((diff_plus >= -tol) & (diff_minus >= -tol)) | ((diff_plus <= tol) & (diff_minus <= tol))
    cond = int(np.sum(nd & (same_side != (evsi <= tol))))
    ratio = evsi[nd & (evpi > tol)] / evpi[nd & (evpi > tol)]
    return ProblemStats(
        m=m, dominated=int(dom.sum()), nondominated=int(nd.sum()),
        zero_nd=int(np.sum(nd & (evsi <= tol))), zero_nd_q8=int(np.sum(nd & (evsi8 <= tol))),
        chain_violations=chain, cond_violations=cond,
        median_ratio=float(np.median(ratio)) if ratio.size else 0.0, mean_ratio=float(ratio.mean()) if ratio.size else 0.0,
    )
