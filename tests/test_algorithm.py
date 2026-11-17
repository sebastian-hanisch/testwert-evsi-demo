"""Mini-Instanzen von Hand (DEMO-PLAYBOOK §1b, ORACLE-PLAYBOOK §5): jede Zahl steht hier als Rechnung, Brüche gegen Brüche.

Handrechnung Kapazitätsfall: EV(G) = 1/2 · 200 + 1/2 · (−100) = 50, EV(K) = 30, EV0 = 50; ein Hellseher wählt bei hoch G (200) und bei niedrig K (30):
1/2 · 200 + 1/2 · 30 = 115, also EVPI = 115 − 50 = 65. Test mit Trefferquote 4/5: P(+) = 1/2, Posterior 4/5 und 1/5, EV(G | +) = 4/5 · 200 − 1/5 · 100 = 140 > 30,
EV(G | −) = 1/5 · 200 − 4/5 · 100 = −40 < 30; Wert mit Test 1/2 · 140 + 1/2 · 30 = 85, EVSI = 35."""

from fractions import Fraction as F

import pytest
from conftest import CAP, HALF, INV, P_STAR

import tev_algorithm as A


def lb(q):
    return A.lik_bin(F(q), F(q))


# --- ohne Test ------------------------------------------------------------------------------------------------------------------------------------

def test_values_without_test():
    assert A.ev_of(HALF, CAP["G"]) == 50 == F(1, 2) * 200 + F(1, 2) * (-100)
    assert A.ev_of(HALF, CAP["K"]) == 30
    assert A.ev0(HALF, CAP) == 50 and A.best_action(HALF, CAP) == "G"
    assert A.evpi(HALF, CAP) == 65 == F(1, 2) * 200 + F(1, 2) * 30 - 50


def test_threshold_of_the_choice_is_13_over_30():
    assert A.pstar(CAP) == P_STAR == F(13, 30)
    assert 300 * P_STAR - 100 == 30                      # EV(G) = 300 p − 100 gleich EV(K) = 30
    assert A.best_action((F(12, 30), F(18, 30)), CAP) == "K" and A.best_action((F(14, 30), F(16, 30)), CAP) == "G"


# --- ein Test mit Trefferquote 4/5 ----------------------------------------------------------------------------------------------------------------

def test_test_with_hit_rate_4_5_rollback_by_hand():
    lik = lb(F(4, 5))
    assert A.joint(HALF, lik) == [(F(2, 5), F(1, 10)), (F(1, 10), F(2, 5))]          # (hoch, +) (niedrig, +) | (hoch, −) (niedrig, −)
    rows = A.rollback(HALF, CAP, lik)
    assert [r["p_y"] for r in rows] == [F(1, 2), F(1, 2)]
    assert [r["posterior"] for r in rows] == [F(4, 5), F(1, 5)]
    assert rows[0]["ev"] == {"G": F(140), "K": F(30)} and rows[1]["ev"] == {"G": F(-40), "K": F(30)}
    assert [r["choice"] for r in rows] == ["G", "K"]
    assert A.ev_with(HALF, CAP, lik) == 85 == F(1, 2) * 140 + F(1, 2) * 30
    assert A.evsi(HALF, CAP, lik) == 35
    assert A.evsi(HALF, CAP, lik) / A.evpi(HALF, CAP) == F(7, 13)


def test_loss_with_test_is_evpi_minus_evsi_by_hand():
    """Verlust: nach + wird G gewählt (Fehler bei niedrig: 1/10 · (30 − (−100)) = 13), nach − wird K gewählt (Fehler bei hoch: 1/10 · (200 − 30) = 17): 13 + 17 = 30 = 65 − 35."""
    lik = lb(F(4, 5))
    assert A.loss_with_test(HALF, CAP, lik) == F(1, 10) * 130 + F(1, 10) * 170 == 30 == 65 - 35


def test_test_without_content_and_perfect_test():
    assert A.evsi(HALF, CAP, A.lik_bin(F(7, 10), F(3, 10))) == 0                    # se + sp = 1: das Ergebnis sagt nichts
    assert A.evsi(HALF, CAP, A.lik_bin(F(1), F(1))) == A.evpi(HALF, CAP) == 65      # fehlerfreier Test


# --- Wertlos-Schwelle -----------------------------------------------------------------------------------------------------------------------------

def test_worthless_threshold_17_over_30_by_hand():
    """Nach − ist P(hoch) = 1 − q; G bleibt, solange 300 (1 − q) − 100 ≥ 30, also q ≤ 17/30."""
    assert 300 * (1 - F(17, 30)) - 100 == 30
    assert A.q0_closed(HALF[0], P_STAR) == F(17, 30)
    assert A.evsi(HALF, CAP, lb(F(17, 30))) == 0
    assert A.evsi(HALF, CAP, lb(F(17, 30) + F(1, 1000))) == F(3, 20)


def test_evsi_of_the_symmetric_test_is_150q_minus_85():
    for q, expected in ((F(6, 10), 5), (F(7, 10), 20), (F(9, 10), 50), (F(1), 65), (F(57, 100), F(1, 2))):
        assert A.evsi(HALF, CAP, lb(q)) == expected == 150 * q - 85
    assert [k for k in range(50, 101) if A.evsi(HALF, CAP, lb(F(k, 100))) == 0] == list(range(50, 57))     # 7 Rasterpunkte 0,50 bis 0,56


def test_threshold_curve_by_hand():
    """q*(c) = (85 + c)/150 bei Vorab 1/2: c = 5 gibt 3/5, 10 gibt 19/30, 20 gibt 7/10, 35 gibt 4/5, 50 gibt 9/10, 65 gibt 1."""
    for c, q in ((5, F(3, 5)), (10, F(19, 30)), (20, F(7, 10)), (35, F(4, 5)), (50, F(9, 10)), (65, F(1))):
        assert F(85 + c, 150) == q and A.evsi(HALF, CAP, lb(q)) == c


def test_threshold_by_bisection_equals_the_closed_form():
    for c in (1.0, 5.0, 10.0, 35.0, 64.0):
        assert A.q_star((0.5, 0.5), {k: tuple(map(float, v)) for k, v in CAP.items()}, c) == pytest.approx((85 + c) / 150, abs=1e-9)
    assert A.q_star((0.5, 0.5), {k: tuple(map(float, v)) for k, v in CAP.items()}, 65.0) is None


# --- einseitiger Test -----------------------------------------------------------------------------------------------------------------------------

def test_one_sided_test_se_1_sp_1_2():
    """se = 1, sp = 1/2: P(+) = 1/2 · 1 + 1/2 · 1/2 = 3/4; nach +: G (EV 3/4-gewichtet: 1/2 · 200 − 1/4 · 100 = 75 gegen 22,5), nach −: K (7,5 gegen −25); Wert mit Test 82,5, EVSI 32,5."""
    lik = A.lik_bin(F(1), F(1, 2))
    rows = A.rollback(HALF, CAP, lik)
    assert rows[0]["p_y"] == F(3, 4) and rows[1]["p_y"] == F(1, 4)
    assert [r["choice"] for r in rows] == ["G", "K"]
    assert rows[0]["joint_value"] == 75 and F(3, 4) * 30 == F(45, 2)
    assert rows[1]["joint_value"] == F(15, 2) == F(1, 4) * 30
    assert A.ev_with(HALF, CAP, lik) == F(165, 2)
    assert A.evsi(HALF, CAP, lik) == F(65, 2)


# --- mehrere Tests --------------------------------------------------------------------------------------------------------------------------------

def test_two_and_three_tests_with_hit_rate_0_7_by_hand():
    q = F(7, 10)
    lik2 = A.lik_n(q, 2)
    assert [lik2[0][k] * F(1, 2) + lik2[1][k] * F(1, 2) for k in range(3)] == [F(29, 100), F(21, 50), F(29, 100)]       # P(k = 0, 1, 2)
    rows = A.rollback(HALF, CAP, lik2)
    assert rows[2]["posterior"] == F(49, 58)                                                                           # (0,7)² gegen (0,3)²
    assert [r["choice"] for r in rows] == ["K", "G", "G"]                                                              # k = 0: K, k = 1 (Posterior 1/2): G, k = 2: G
    assert [r["joint_value"] for r in rows] == [F(87, 10), 21, F(89, 2)]                                               # 8,7 (K) · 21 (G) · 44,5 (G)
    assert A.ev_with(HALF, CAP, lik2) == F(371, 5) == F(742, 10)
    assert A.evsi(HALF, CAP, lik2) == F(121, 5)                                                                        # 24,2
    assert A.evsi(HALF, CAP, A.lik_n(q, 3)) == F(163, 5)                                                               # 32,6
    assert [A.evsi(HALF, CAP, A.lik_n(q, n)) for n in (1, 2, 3)] == [20, F(121, 5), F(163, 5)]
    # Zuwächse 20, 4,2, 8,4: der dritte Test bringt doppelt so viel wie der zweite
    assert F(163, 5) - F(121, 5) == F(42, 5) == 2 * (F(121, 5) - 20)


def test_decision_boundary_of_n_tests_is_k_at_least_half_n():
    q = F(7, 10)
    for n in range(1, 13):
        for k in range(n + 1):
            row = A.rollback(HALF, CAP, A.lik_n(q, n))[k]
            assert (row["choice"] == "G") == (2 * k >= n), (n, k)


def test_stopping_rule_against_optimum_at_cost_5_by_hand():
    """Nettowert EVSI(n) − 5 n: n = 1: 15, n = 2: 14,2, n = 3: 17,6; die Regel hört nach dem ersten Test auf (4,2 < 5), das Optimum liegt bei drei."""
    vals = A.evsi_series(HALF, CAP, F(7, 10), 6)
    assert [vals[n] - 5 * n for n in (1, 2, 3)] == [15, F(71, 5), F(88, 5)]
    assert A.rule_n(vals, F(5)) == 1 and A.best_n(vals, F(5)) == 3


# --- einzeln wertlose Tests -----------------------------------------------------------------------------------------------------------------------

def test_two_individually_worthless_tests_are_worth_something():
    """q = 0,55: ein Test wertlos; zwei Tests: nach k = 0 ist P(hoch) = 0,2025/0,505 = 81/202 = 0,401 < 13/30, die Wahl kippt zu K."""
    q = F(11, 20)
    assert A.evsi(HALF, CAP, lb(q)) == 0
    lik2 = A.lik_n(q, 2)
    rows = A.rollback(HALF, CAP, lik2)
    assert rows[0]["p_y"] == F(2525, 10000)                                                   # P(k = 0) = 1/2 · 0,45² + 1/2 · 0,55² = 0,2525
    assert rows[0]["posterior"] == F(81, 202) and F(81, 202) < P_STAR
    assert rows[0]["choice"] == "K"
    assert A.evsi(HALF, CAP, lik2) == F(49, 20)
    # zwei verschiedene wertlose Tests (0,55 und 0,56) zusammen: 3,19
    assert A.evsi(HALF, CAP, A.lik_pair(lb(F(11, 20)), lb(F(14, 25)))) == F(319, 100)


# --- zweites Zahlenbeispiel -----------------------------------------------------------------------------------------------------------------------

def test_second_example_investment_hand_values():
    """A +100/−60, B sicher +5, Vorab 9/20: EV(A) = 45 − 33 = 12; EVPI = 9/20 · 100 + 11/20 · 5 − 12 = 143/4; EVSI(se 9/10, sp 4/5) = 973/40; q0 = (9/11)/((9/11) + (13/19)) = 171/314."""
    prior = (F(9, 20), F(11, 20))
    assert A.ev0(prior, INV) == 12 and A.evpi(prior, INV) == F(143, 4)
    assert A.evsi(prior, INV, A.lik_bin(F(9, 10), F(4, 5))) == F(973, 40)
    ps = A.pstar(INV)
    assert ps == F(13, 32)
    odds, ostar = F(9, 11), ps / (1 - ps)
    assert ostar == F(13, 19) and A.q0_closed(prior[0], ps) == odds / (odds + ostar) == F(171, 314)
    assert A.evsi(prior, INV, lb(F(171, 314))) == 0 and A.evsi(prior, INV, lb(F(171, 314) + F(1, 1000))) > 0


# --- Phi und Falle --------------------------------------------------------------------------------------------------------------------------------

def test_phi_correlation_by_hand_and_the_rules_at_q_0_8():
    import numpy as np
    # Kovarianz P(hoch, +) − P(hoch) P(+) = 2/5 − 1/4 = 3/20, Varianzen je 1/4: Korrelation (3/20)/(1/4) = 3/5 = 2q − 1
    assert F(2, 5) - F(1, 2) * F(1, 2) == F(3, 20) and F(3, 20) / F(1, 4) == F(3, 5)
    assert float(A.phi_array((0.5, 0.5), 0.8, 0.8)) == pytest.approx(0.6, abs=1e-12)
    assert F(3, 5) * 65 == 39 and F(3, 5) ** 2 * 65 == F(117, 5) and A.evsi(HALF, CAP, lb(F(4, 5))) == 35
    assert 39 - 35 == 4 == 20 * (1 - F(4, 5))
    assert np.isfinite(float(A.phi_array((0.5, 0.5), 1.0, 0.0)))


def test_float_and_fraction_give_the_same_evsi_on_51_points():
    capf = {k: tuple(float(x) for x in v) for k, v in CAP.items()}
    for k in range(50, 101):
        q = F(k, 100)
        assert float(A.evsi(HALF, CAP, lb(q))) == pytest.approx(A.evsi((0.5, 0.5), capf, A.lik_bin(k / 100, k / 100)), abs=1e-10)
