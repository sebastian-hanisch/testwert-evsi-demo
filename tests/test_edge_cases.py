"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): Vorab 0 und 1, Gleichstand an der Schwelle, Tests ohne Aussage, P(Ergebnis) = 0, n = 0 und 1."""

from fractions import Fraction as F

import pytest
from conftest import CAP, HALF, P_STAR

import tev_algorithm as A
import tev_evaluation as ev
import tev_scenario as S


def test_prior_zero_or_one_makes_information_worthless():
    for prior in ((F(0), F(1)), (F(1), F(0))):
        assert A.evpi(prior, CAP) == 0
        for q in (F(1, 2), F(4, 5), F(1)):
            assert A.evsi(prior, CAP, A.lik_bin(q, q)) == 0
    assert A.q0_closed(F(0), P_STAR) == 1 and A.q0_closed(F(1), P_STAR) == 1


def test_prior_exactly_at_the_threshold_every_test_above_one_half_has_value():
    prior = (P_STAR, 1 - P_STAR)
    assert A.ev_of(prior, CAP["G"]) == A.ev_of(prior, CAP["K"]) == 30           # Gleichstand: beide Alternativen gleich viel wert
    assert A.best_action(prior, CAP) == "G"                                     # dokumentierte Regel: bei Gleichstand die zuerst genannte (G)
    assert A.evsi(prior, CAP, A.lik_bin(F(1, 2), F(1, 2))) == 0
    assert all(A.evsi(prior, CAP, A.lik_bin(F(k, 100), F(k, 100))) > 0 for k in (51, 60, 99))


def test_test_without_statement_se_sp_one_half():
    lik = A.lik_bin(F(1, 2), F(1, 2))
    assert A.evsi(HALF, CAP, lik) == 0
    rows = A.rollback(HALF, CAP, lik)
    assert [r["posterior"] for r in rows] == [F(1, 2), F(1, 2)]                 # der Posterior bleibt das Vorab


def test_hit_rate_below_one_half_is_read_the_other_way_round():
    for k in range(0, 50):
        assert A.evsi(HALF, CAP, A.lik_bin(F(k, 100), F(k, 100))) == A.evsi(HALF, CAP, A.lik_bin(F(100 - k, 100), F(100 - k, 100)))


def test_outcome_with_probability_zero_has_no_posterior_and_no_division_by_zero():
    lik = A.lik_bin(F(0), F(1))                                                 # se = 0, sp = 1: „+“ kommt nie vor
    rows = A.rollback(HALF, CAP, lik)
    assert rows[0]["p_y"] == 0 and rows[0]["posterior"] is None and rows[0]["ev"] == {"G": None, "K": None}
    assert rows[1]["p_y"] == 1 and rows[1]["posterior"] == F(1, 2)
    assert A.evsi(HALF, CAP, lik) == 0
    a = ev.analyse_test(ev.Settings(se=0.0, sp=1.0))
    assert a.worthless and a.ratio == 0
    assert ev.verdict_test(a)[0] == "warning"


def test_dominated_alternative_every_test_is_worthless():
    pay = {"G": (F(100), F(50)), "K": (F(30), F(30))}                          # G in beiden Zuständen besser
    assert A.evpi(HALF, pay) == 0
    assert A.evsi(HALF, pay, A.lik_bin(F(1), F(1))) == 0
    assert A.pstar(pay) is None


def test_equal_expected_values_of_both_alternatives():
    pay = {"G": (F(100), F(0)), "K": (F(50), F(50))}                            # bei p = 1/2 gleich (50)
    assert A.ev_of(HALF, pay["G"]) == A.ev_of(HALF, pay["K"])
    assert A.evpi(HALF, pay) == 25 and A.evsi(HALF, pay, A.lik_bin(F(9, 10), F(9, 10))) > 0


def test_zero_tests_and_one_test():
    assert A.evsi_series(HALF, CAP, F(7, 10), 0) == [0]
    vals = A.evsi_series(HALF, CAP, F(7, 10), 1)
    assert vals == [0, 20]
    assert A.lik_n(F(7, 10), 0) == [[1], [1]] and A.evsi(HALF, CAP, A.lik_n(F(7, 10), 0)) == 0   # kein Test: ein Ergebnis mit Wahrscheinlichkeit 1
    assert A.best_n([F(0)], F(5)) == 0 and A.rule_n([F(0)], F(5)) == 0


def test_threshold_functions_at_the_limits():
    capf = S.PAY
    prior = (0.5, 0.5)
    assert A.q_star(prior, capf, 65.0) is None and A.q_star(prior, capf, 1000.0) is None
    assert A.q_star(prior, capf, 0.0) == pytest.approx(17 / 30, abs=1e-9)         # Kosten 0: die Wertlos-Schwelle
    assert A.q_star(prior, capf, 64.999) == pytest.approx((85 + 64.999) / 150, abs=1e-9)
    assert A.min_specificity(prior, capf, 1.0, 64.9999) is not None and A.min_specificity(prior, capf, 1.0, 65.0) is None


def test_stopping_rule_and_optimum_with_a_tie_choose_the_smaller_number():
    """Bei c = 6,3 sind n = 1 und n = 3 gleich gut (Nettowert 13,7); beide Funktionen nehmen die kleinere Zahl."""
    vals = A.evsi_series(HALF, CAP, F(7, 10), 6)
    c = F(63, 10)
    assert vals[1] - c == vals[3] - 3 * c
    assert A.best_n(vals, c) == 1 and A.rule_n(vals, c) == 1


def test_count_rises_counts_places_two_to_n_max():
    assert A.count_rises([0, 1, 3, 4], 0) == ([2], 2)       # Zuwächse 1, 2, 1: steigt nur bei n = 2
    assert A.count_rises([0, 1, 2, 3], 0) == ([], 2)        # gleiche Zuwächse: kein Anstieg (Gleichstand zählt nicht)


def test_prior_bounds_of_the_app_slider_are_valid_probabilities():
    import tev_constants as C
    assert 0 < C.P_MIN < 1 and 0 < C.P_MAX < 1 and C.P_MIN < C.P_MAX
    for p in (C.P_MIN, C.P_MAX):
        a = ev.analyse_test(ev.Settings(p=p))
        assert a.evpi > 0 and a.evsi >= 0
        ev.analyse_zero(p)
        ev.analyse_cost(p, C.C_MAX)
        ev.analyse_multi(p, 1.0, C.N_MAX, 0.0)
        ev.analyse_trap(p, 0.5, 0.5, 0.5)
        ev.analyse_trap(p, 1.0, 1.0, 1.0)


def test_slider_extremes_of_every_experiment_compute():
    import tev_constants as C
    for se, sp in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)):
        a = ev.analyse_test(ev.Settings(se=se, sp=sp, c=C.C_MAX))
        assert a.evsi >= -1e-9 and ev.verdict_test(a)[1]
    for q in (0.5, 1.0):
        for n in (C.N_MIN, C.N_MAX):
            m = ev.analyse_multi(0.5, q, n, C.C_MAX)
            assert ev.verdict_multi(m)[1]
    assert ev.verdict_cost(ev.analyse_cost(0.5, C.C_MAX))[1] and ev.verdict_cost(ev.analyse_cost(0.5, 0.0))[1]
