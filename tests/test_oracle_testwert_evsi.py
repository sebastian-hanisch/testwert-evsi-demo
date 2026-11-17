"""Orakel-Tests: der Kern (Rollback über verbundene Wahrscheinlichkeiten) gegen andere Rechenwege (ORACLE-PLAYBOOK §2, §5, §6).

Orakel für EVSI: (a) Aufzählung aller Entscheidungsregeln, (b) der Baum mit Bayes-Posterior (mit Teilen), (c) EVPI minus Verlust mit Test, (d) ein lineares Programm über
randomisierte Entscheidungsregeln (scipy, HiGHS); für Mehrfachtests die Aufzählung aller 2^n Ergebnisfolgen; für Schwellen geschlossene Formeln und `scipy.optimize.brentq`;
für die numpy-Raster die exakte Rechnung in Brüchen. Literaturwerte mit Zahl gibt es nicht (die Zahlen von Howard und Matheson sind nicht nachgerechnet, siehe README):
geprüft werden die Gleichungen in der Form, in der die README sie nennt.
"""

from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest
from conftest import CAP, HALF, INV, P_STAR

import tev_algorithm as A
import tev_scenario as S

scipy_opt = pytest.importorskip("scipy.optimize")


# --- unabhängige Rechenwege (im Test geschrieben, nicht aus dem Kern) -------------------------------------------------------------------------------

def oracle_tree(prior, pay, lik):
    """Baum mit Bayes-Posterior: P(y) und P(hoch | y) mit Teilen, je Ergebnis die beste Alternative auf dem Posterior."""
    total = 0
    for y in range(len(lik[0])):
        py = prior[0] * lik[0][y] + prior[1] * lik[1][y]
        if py == 0:
            continue
        post = (prior[0] * lik[0][y] / py, prior[1] * lik[1][y] / py)
        total += py * max(post[0] * u[0] + post[1] * u[1] for u in pay.values())
    return total - max(prior[0] * u[0] + prior[1] * u[1] for u in pay.values())


def oracle_rules(prior, pay, lik):
    """Alle Entscheidungsregeln y -> Alternative aufzählen und den Erwartungswert jeder Regel ausrechnen."""
    acts = list(pay.values())
    best = None
    for rule in product(range(len(acts)), repeat=len(lik[0])):
        v = sum(prior[s] * lik[s][y] * acts[rule[y]][s] for s in (0, 1) for y in range(len(lik[0])))
        best = v if best is None or v > best else best
    return best - max(prior[0] * u[0] + prior[1] * u[1] for u in pay.values())


def oracle_lp(prior, pay, lik):
    """Lineares Programm über randomisierte Regeln: max sum_y sum_a x[y,a] * wert[y,a], sum_a x[y,a] = 1, x >= 0 (Gleitkomma, HiGHS)."""
    acts = list(pay.values())
    m, na = len(lik[0]), len(acts)
    c = np.zeros(m * na)
    for y in range(m):
        for a, u in enumerate(acts):
            c[y * na + a] = -float(prior[0] * lik[0][y] * u[0] + prior[1] * lik[1][y] * u[1])
    a_eq = np.zeros((m, m * na))
    for y in range(m):
        a_eq[y, y * na:(y + 1) * na] = 1.0
    res = scipy_opt.linprog(c, A_eq=a_eq, b_eq=np.ones(m), bounds=(0, None), method="highs")
    assert res.status == 0
    return -res.fun - float(max(prior[0] * u[0] + prior[1] * u[1] for u in pay.values()))


# --- EVSI auf Rastern: Formel gegen alle Orakel -------------------------------------------------------------------------------------------------

def test_formula_equals_rules_tree_and_loss_on_441_exact_points():
    for i in range(21):
        for j in range(21):
            lik = A.lik_bin(F(i, 20), F(j, 20))
            v = A.evsi(HALF, CAP, lik)
            assert v == oracle_rules(HALF, CAP, lik) == oracle_tree(HALF, CAP, lik) == A.evpi(HALF, CAP) - A.loss_with_test(HALF, CAP, lik), (i, j)
            assert 0 <= v <= 65


@pytest.mark.parametrize("prior", [(F(3, 10), F(7, 10)), (F(9, 20), F(11, 20)), (F(4, 5), F(1, 5))])
def test_formula_equals_rules_and_tree_for_other_priors_and_the_second_example(prior):
    for pay in (CAP, INV):
        for i in range(0, 11):
            for j in range(0, 11):
                lik = A.lik_bin(F(i, 10), F(j, 10))
                assert A.evsi(prior, pay, lik) == oracle_rules(prior, pay, lik) == oracle_tree(prior, pay, lik)


def test_linear_program_over_randomized_rules_gives_the_same_value():
    """Das LP findet den besten randomisierten Plan; sein Wert gleich dem Rollback (Toleranz 1e-9: HiGHS rechnet in Gleitkomma)."""
    capf = {k: tuple(float(x) for x in v) for k, v in CAP.items()}
    for prior in ((0.3, 0.7), (0.5, 0.5), (0.7, 0.3)):
        for i in range(0, 11):
            for j in range(0, 11):
                lik = A.lik_bin(i / 10, j / 10)
                assert oracle_lp(prior, capf, lik) == pytest.approx(A.evsi(prior, capf, lik), abs=1e-9), (prior, i, j)


def test_linear_program_for_n_tests_and_two_different_tests():
    capf = {k: tuple(float(x) for x in v) for k, v in CAP.items()}
    for n in (1, 2, 3, 5, 8):
        lik = A.lik_n(0.7, n)
        assert oracle_lp((0.5, 0.5), capf, lik) == pytest.approx(A.evsi((0.5, 0.5), capf, lik), abs=1e-9)
    lik = A.lik_pair(A.lik_bin(0.6, 0.6), A.lik_bin(0.9, 0.8))
    assert oracle_lp((0.5, 0.5), capf, lik) == pytest.approx(A.evsi((0.5, 0.5), capf, lik), abs=1e-9)


# --- Mehrfachtests: Aufzählung aller Ergebnisfolgen -----------------------------------------------------------------------------------------------

def test_counting_positives_equals_enumerating_all_2_to_the_n_sequences():
    q = F(7, 10)
    for n in range(1, 11):
        # jede Folge (r1, ..., rn) mit r in {+, −}: P(Folge | hoch) = q^(Zahl +) (1 − q)^(Zahl −), bei niedrig umgekehrt
        seq_hi, seq_lo = [], []
        for seq in product((0, 1), repeat=n):
            plus = seq.count(0)
            seq_hi.append(q ** plus * (1 - q) ** (n - plus))
            seq_lo.append((1 - q) ** plus * q ** (n - plus))
        assert A.evsi(HALF, CAP, [seq_hi, seq_lo]) == A.evsi(HALF, CAP, A.lik_n(q, n)), n


def test_pair_of_equal_tests_is_two_tests_and_pair_bounds_hold_on_a_grid():
    q = F(7, 10)
    assert A.evsi(HALF, CAP, A.lik_pair(A.lik_bin(q, q), A.lik_bin(q, q))) == A.evsi(HALF, CAP, A.lik_n(q, 2)) == F(121, 5)
    for qa in (F(1, 2), F(11, 20), F(3, 5), F(4, 5), F(1)):
        for qb in (F(1, 2), F(11, 20), F(7, 10), F(9, 10), F(1)):
            joint = A.evsi(HALF, CAP, A.lik_pair(A.lik_bin(qa, qa), A.lik_bin(qb, qb)))
            assert max(A.evsi(HALF, CAP, A.lik_bin(qa, qa)), A.evsi(HALF, CAP, A.lik_bin(qb, qb))) <= joint <= 65


# --- geschlossene Formeln gegen den Kern --------------------------------------------------------------------------------------------------------

def test_closed_form_150q_minus_85_on_101_points():
    for k in range(0, 101):
        q = F(k, 100)
        qq = q if q >= F(1, 2) else 1 - q                       # unter 1/2 spiegelbildlich gelesen
        assert A.evsi(HALF, CAP, A.lik_bin(q, q)) == max(F(0), 150 * qq - 85), k


def test_zero_threshold_q0_for_19_priors_exact():
    for k in range(1, 20):
        p = F(k, 20)
        pr = (p, 1 - p)
        q0 = A.q0_closed(p, P_STAR)
        lik = lambda q: A.lik_bin(q, q)                         # noqa: E731
        assert A.evsi(pr, CAP, lik(q0)) == 0 and A.evsi(pr, CAP, lik((F(1, 2) + q0) / 2)) == 0, p
        if p != P_STAR and q0 < F(9999, 10000):
            assert A.evsi(pr, CAP, lik(q0 + F(1, 10000))) > 0, p
    assert A.q0_closed(P_STAR, P_STAR) == F(1, 2)
    assert all(A.evsi((P_STAR, 1 - P_STAR), CAP, A.lik_bin(F(k, 100), F(k, 100))) > 0 for k in range(51, 100))


def test_threshold_curve_equals_brentq_and_the_formula():
    capf = {k: tuple(float(x) for x in v) for k, v in CAP.items()}
    for c in (0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 64):
        root = scipy_opt.brentq(lambda q, c=c: A.evsi((0.5, 0.5), capf, A.lik_bin(q, q)) - c, 0.5667, 1.0, xtol=1e-14)
        assert A.q_star((0.5, 0.5), capf, float(c)) == pytest.approx(root, abs=1e-9)
        assert root == pytest.approx((85 + c) / 150, abs=1e-9)


def test_min_specificity_equals_brentq():
    capf = {k: tuple(float(x) for x in v) for k, v in CAP.items()}
    for se in (0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
        root = scipy_opt.brentq(lambda sp, se=se: A.evsi((0.5, 0.5), capf, A.lik_bin(se, sp)) - 5.0, 1 - se + 1e-9, 1.0, xtol=1e-13)
        assert A.min_specificity((0.5, 0.5), capf, se, 5.0) == pytest.approx(root, abs=1e-9)
    # se = 0,6: selbst mit sp = 1 ist EVSI = 31 GE (von Hand: P(+) = 0,3, nach + sicher hoch, G bringt 60; nach − bleibt K mit 0,7 · 30 = 21; 60 + 21 − 50 = 31); bei Kosten 31 nie, knapp darunter fast sp = 1
    assert A.evsi((0.5, 0.5), capf, A.lik_bin(0.6, 1.0)) == pytest.approx(31.0, abs=1e-9)
    assert A.min_specificity((0.5, 0.5), capf, 0.6, 31.0) is None
    assert A.min_specificity((0.5, 0.5), capf, 0.6, 30.9) == pytest.approx(0.99, abs=0.02)


# --- numpy-Raster gegen die exakte Rechnung ----------------------------------------------------------------------------------------------------

def _exact_grid_zero_mask():
    zero = np.zeros((101, 101), dtype=bool)
    cond_violations = 0
    for i in range(101):
        for j in range(101):
            lik = A.lik_bin(F(i, 100), F(j, 100))
            v = A.evsi(HALF, CAP, lik)
            zero[i, j] = v == 0
            p_plus = HALF[0] * lik[0][0] + HALF[1] * lik[1][0]
            p_minus = 1 - p_plus
            posts = [HALF[0] * lik[0][0] / p_plus] if p_plus else []
            posts += [HALF[0] * lik[0][1] / p_minus] if p_minus else []
            same = all(x >= P_STAR for x in posts) or all(x <= P_STAR for x in posts)
            cond_violations += same != (v == 0)
    return zero, cond_violations


def test_numpy_grid_of_10201_points_equals_the_exact_zero_set():
    exact, cond_violations = _exact_grid_zero_mask()
    grid = np.arange(101) / 100.0
    z = A.evsi_bin_array((0.5, 0.5), S.PAY, grid[:, None], grid[None, :])
    assert np.array_equal(z <= A.ZERO_TOL, exact)
    assert int(exact.sum()) == 2383 and cond_violations == 0     # Bedingung „beide Posterior auf derselben Seite von 13/30“ = Nullmenge, 0 Fehlzuordnungen


def test_numpy_pair_grid_equals_the_exact_counts():
    qs = [F(k, 100) for k in range(50, 101)]
    single = [A.evsi(HALF, CAP, A.lik_bin(q, q)) for q in qs]
    sub = sup = eq = 0
    for ia, qa in enumerate(qs):
        for ib, qb in enumerate(qs):
            joint = A.evsi(HALF, CAP, A.lik_pair(A.lik_bin(qa, qa), A.lik_bin(qb, qb)))
            total = single[ia] + single[ib]
            sub += joint < total
            sup += joint > total
            eq += joint == total
    arr = A.evsi_pair_array((0.5, 0.5), S.PAY, np.arange(50, 101) / 100.0, np.arange(50, 101) / 100.0)
    one = A.evsi_bin_array((0.5, 0.5), S.PAY, np.arange(50, 101) / 100.0, np.arange(50, 101) / 100.0)
    diff = arr - (one[:, None] + one[None, :])
    tol = A.ZERO_TOL
    assert (int((diff < -tol).sum()), int((diff > tol).sum()), int((np.abs(diff) <= tol).sum())) == (sub, sup, eq) == (1915, 84, 602)


def test_phi_array_equals_the_definition_in_fractions():
    for p in (F(3, 10), F(1, 2), F(4, 5)):
        for q in (F(11, 20), F(7, 10), F(19, 20)):
            p_plus = p * q + (1 - p) * (1 - q)
            cov = p * q - p * p_plus
            expected = float(cov) / float(p * (1 - p) * p_plus * (1 - p_plus)) ** 0.5
            assert float(A.phi_array((float(p), float(1 - p)), float(q), float(q))) == pytest.approx(expected, abs=1e-12)


# --- angezeigte Kennzahlen -------------------------------------------------------------------------------------------------------------------

def test_displayed_figures_of_experiment_1_match_the_oracles():
    import tev_evaluation as ev
    for se, sp, p, c in ((0.8, 0.8, 0.5, 5.0), (1.0, 0.5, 0.5, 0.0), (0.9, 0.6, 0.3, 12.0), (0.57, 0.57, 0.5, 0.2)):
        a = ev.analyse_test(ev.Settings(p=p, se=se, sp=sp, c=c))
        prior, lik = (F(p), 1 - F(p)), A.lik_bin(F(se), F(sp))
        v = oracle_tree(prior, CAP, lik)
        assert a.evsi == pytest.approx(float(v), abs=1e-9)
        assert a.evpi == pytest.approx(float(A.evpi(prior, CAP)), abs=1e-9)
        assert a.enbs == pytest.approx(float(v) - c, abs=1e-9)
        assert a.loss_with == pytest.approx(float(A.evpi(prior, CAP) - v), abs=1e-9)
        for row in a.rows:
            if row["p_y"] > 0:
                assert row["posterior"] == pytest.approx(float(prior[0] * lik[0][row["y"]] / (prior[0] * lik[0][row["y"]] + prior[1] * lik[1][row["y"]])), abs=1e-9)
