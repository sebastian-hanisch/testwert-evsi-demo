"""Claims: jede Zahl der README (Tabelle „Befunde“, B1 bis B14) und jeder Schnellstart mit seiner erwarteten Verdikt-Zeile.

Exakte Zahlen stehen in Brüchen (`fractions`), Zählungen aus numpy und Zufallsströmen als Band; die Zuordnung „Zahl → Test“ steht in der README.
B1, B14 und die Handwerte stehen in `test_algorithm.py`, die exakten Raster in `test_oracle_testwert_evsi.py`; hier die Zahlen, die dort nicht schon als Rechnung stehen."""

from fractions import Fraction as F

import numpy as np
import pytest
from conftest import CAP, HALF, P_STAR

import tev_algorithm as A
import tev_constants as C
import tev_evaluation as ev
import tev_scenario as S

CAPF = S.PAY
TOL = A.ZERO_TOL


def lb(q):
    return A.lik_bin(F(q), F(q))


def r(x, nd):
    return round(float(x), nd)


class TestBefunde:
    def test_b2_threshold_17_over_30_and_seven_worthless_grid_points(self):
        assert all(A.evsi(HALF, CAP, lb(F(k, 100))) == max(F(0), 150 * F(k, 100) - 85) for k in range(50, 101))
        assert A.q0_closed(HALF[0], P_STAR) == F(17, 30) and r(F(17, 30), 3) == 0.567
        zero = [k for k in range(50, 101) if A.evsi(HALF, CAP, lb(F(k, 100))) == 0]
        assert len(zero) == 7 and zero[-1] == 56 and A.evsi(HALF, CAP, lb(F(57, 100))) == F(1, 2)
        # Bedingung des Satzes „wertlos“: nur für 13/30 ≤ q ≤ 17/30 (q < 1/2 spiegelbildlich gelesen), darunter nicht
        wertlos = [F(k, 300) for k in range(0, 301) if A.evsi(HALF, CAP, lb(F(k, 300))) == 0]
        assert (min(wertlos), max(wertlos)) == (F(13, 30), F(17, 30))
        assert A.evsi(HALF, CAP, lb(F(1, 10))) == 50 and A.evsi(HALF, CAP, lb(F(2, 5))) == 5 and A.evsi(HALF, CAP, lb(F(13, 30))) == 0
        assert all(A.evsi(HALF, CAP, lb(F(k, 100))) == A.evsi(HALF, CAP, lb(1 - F(k, 100))) for k in range(0, 101))
        z = ev.analyse_zero(0.5)
        assert z.zero_q == 7
        assert [ev.pct(z.ratio_at[q]) for q in (0.6, 0.7, 0.8, 0.9, 0.95)] == ["7,7 %", "30,8 %", "53,8 %", "76,9 %", "88,5 %"]

    def test_b3_threshold_for_each_prior(self):
        expected = {F(1, 20): 0.9356, F(1, 10): 0.8731, F(3, 10): 0.6408, F(2, 5): 0.5342, F(9, 20): 0.5169, F(1, 2): 0.5667, F(3, 5): 0.6623, F(4, 5): 0.8395, F(19, 20): 0.9613}
        for p, q0 in expected.items():
            assert r(A.q0_closed(p, P_STAR), 4) == q0, p
        assert A.q0_closed(P_STAR, P_STAR) == F(1, 2)
        assert all(A.evsi((P_STAR, 1 - P_STAR), CAP, lb(F(k, 100))) > 0 for k in range(51, 100))            # 49 Punkte

    def test_b4_grid_counts(self):
        z = ev.analyse_zero(0.5)
        assert (z.zero_total, z.n_points, z.useful_total, z.zero_useful) == (2383, 10201, 5050, 1141)
        assert ev.pct(z.zero_total / z.n_points) == "23,4 %" and ev.pct(z.zero_useful / z.useful_total) == "22,6 %"
        grid = np.arange(101) / 100.0
        v = A.evsi_bin_array((0.5, 0.5), CAPF, grid[:, None], grid[None, :])
        assert v.min() >= -TOL and v.max() <= 65.0 + TOL                                                   # EVPI >= EVSI >= 0 an allen 10 201 Punkten

    def test_b5_threshold_curve_asymmetric_and_per_prior(self):
        for c, q in ((1, 0.5733), (5, 0.6), (10, 0.6333), (20, 0.7), (40, 0.8333), (60, 0.9667), (64, 0.9933)):
            assert r(F(85 + c, 150), 4) == q
            assert A.q_star((0.5, 0.5), CAPF, float(c)) == pytest.approx((85 + c) / 150, abs=1e-9)
        assert A.q_star((0.5, 0.5), CAPF, 65.0) is None and ev.pct(5 / 65) == "7,7 %"
        spec = {se: (None if sp is None else round(sp, 4)) for se, sp, _ in ev.analyse_cost(0.5, 5.0).spec_rows}
        assert spec == {0.6: 0.6, 0.7: 0.4692, 0.8: 0.3385, 0.9: 0.2077, 0.95: 0.1423, 1.0: 0.0769}
        per_prior = {p: round(A.q_star((p, 1 - p), CAPF, 5.0), 4) for p in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8)}
        assert per_prior == {0.2: 0.7899, 0.3: 0.6761, 0.4: 0.5685, 0.5: 0.6, 0.6: 0.6948, 0.8: 0.8704}

    def test_b6_n_tests_values_and_shape(self):
        vals = A.evsi_series(HALF, CAP, F(7, 10), 60)
        assert [r(vals[n], 2) for n in (1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 60)] == [20.0, 24.2, 32.6, 35.25, 40.54, 42.39, 51.21, 60.43, 63.36, 64.39, 64.91]
        assert ev.pct(vals[60] / 65) == "99,9 %"
        inc = A.increments(vals)
        assert [r(inc[n], 1) for n in range(1, 4)] == [20.0, 4.2, 8.4] and r(inc[4], 2) == 2.65 and r(inc[4], 3) == 2.646
        rises, places = A.count_rises(vals, 0)
        assert (len(rises), places) == (29, 59) and all(n % 2 == 1 for n in rises)
        for par in (0, 1):                                                                                   # jede zweite Testzahl getrennt: konkav
            d2 = [vals[n] - vals[n - 2] for n in range(par + 2, 61, 2)]
            assert all(d2[i] >= d2[i + 1] for i in range(len(d2) - 1))
        for q in (F(6, 10), F(7, 10), F(8, 10), F(9, 10)):                                                   # „G genau bei k ≥ n/2“ gilt bei diesen Trefferquoten
            for n in range(1, 21):
                for k, row in enumerate(A.rollback(HALF, CAP, A.lik_n(q, n))):
                    assert (row["choice"] == "G") == (2 * k >= n), (q, n, k)
        row = A.rollback(HALF, CAP, A.lik_n(F(11, 20), 3))[1]                                                # ... bei q = 0,55 nicht: n = 3, k = 1
        assert row["posterior"] == F(9, 20) > P_STAR and row["choice"] == "G" and 2 * 1 < 3
        m = ev.analyse_multi(0.5, 0.7, 3, 0.0)                                                               # dieselbe Zählung in der App (Gleitkomma)
        assert (len(m.rises), m.n_places, m.odd_rises) == (29, 59, 29)

    def test_b7_sixteen_cells(self):
        rises_n, kinds = [], []
        for pp in (F(2, 5), F(9, 20), F(1, 2), F(3, 5)):
            for qq in (F(6, 10), F(7, 10), F(8, 10), F(9, 10)):
                vals = A.evsi_series((pp, 1 - pp), CAP, qq, 40)
                rises, places = A.count_rises(vals, 0)
                assert places == 39
                rises_n.append(len(rises))
                odd = sum(1 for x in rises if x % 2 == 1)
                kinds.append(None if not rises else "ungerade" if odd == len(rises) else "gerade" if odd == 0 else "gemischt")
                m = ev.analyse_multi(float(pp), float(qq), 3, 0.0)                                           # die App zählt bis n = 60: die Zelle ist dieselbe
                assert m.measured_cell
        assert sum(1 for x in rises_n if x > 0) == 15 and rises_n.count(0) == 1
        assert rises_n.count(19) == 14 and rises_n.count(20) == 1
        assert (kinds.count("ungerade"), kinds.count("gerade"), kinds.count("gemischt")) == (12, 2, 1)
        assert kinds[2 * 4 + 0] == "gerade" and kinds[3 * 4 + 1] == "gerade" and kinds[3 * 4 + 0] == "gemischt" and kinds[3 * 4 + 2] is None   # (0,5; 0,6), (0,6; 0,7), (0,6; 0,6), (0,6; 0,8)
        assert not ev.analyse_multi(0.6, 0.8, 3, 0.0).rises and ev.analyse_multi(0.6, 0.8, 3, 0.0).measured_cell
        assert not ev.analyse_multi(0.5, 0.75, 3, 0.0).measured_cell and not ev.analyse_multi(0.55, 0.7, 3, 0.0).measured_cell      # außerhalb der 16 Zellen

    def test_b8_individually_worthless_tests(self):
        vals = A.evsi_series(HALF, CAP, F(11, 20), 50)
        assert vals[1] == 0 and vals[2] == F(49, 20)
        assert [r(vals[n], 3) for n in (3, 5, 10, 20, 50)] == [3.069, 5.5, 10.553, 17.249, 29.829]
        assert F(81, 202) < P_STAR and r(F(81, 202), 3) == 0.401
        v52 = A.evsi_series(HALF, CAP, F(13, 25), 50)
        assert [n for n in range(1, 51) if v52[n] > 0][0] == 4 and r(v52[4], 4) == 0.2404 and r(v52[50], 3) == 8.541
        v0 = A.evsi_series(HALF, CAP, F(17, 30), 2)
        assert v0[1] == 0 and r(v0[2], 3) == 4.911
        assert all(v == 0 for v in A.evsi_series(HALF, CAP, F(1, 2), 60))                                     # ohne Aussage bleibt es 0

    def test_b9_pairs(self):
        t = ev.analyse_trap(0.5, 0.8, 0.7, 0.7)
        pc = t.pair_counts
        assert (pc["pairs"], pc["sub"], pc["super"], pc["equal"]) == (2601, 1915, 84, 602) and pc["equal_with_zero_test"] == 602
        assert (pc["both_pos"], pc["both_pos_sub"], pc["both_pos_super"], pc["both_pos_equal"]) == (1936, 1915, 21, 0)
        assert (pc["both_zero"], pc["both_zero_joint_pos"]) == (49, 21)
        qs = ev.Q_GRID
        single = A.evsi_bin_array((0.5, 0.5), CAPF, qs, qs)
        ia, ib = np.nonzero((t.pair_diff > TOL) & (single > TOL)[:, None] & (single > TOL)[None, :])
        assert len(ia) == 21 and qs[ia].min() == 0.57 and qs[ia].max() == 0.62 and qs[ib].min() == 0.57 and qs[ib].max() == 0.62
        for (qa, qb), (joint, total) in {(F(7, 10), F(7, 10)): (24.2, 40), (F(4, 5), F(4, 5)): (38.2, 70), (F(9, 10), F(7, 10)): (50, 70), (F(11, 20), F(11, 20)): (2.45, 0),
                                         (F(11, 20), F(14, 25)): (3.19, 0), (F(3, 5), F(3, 5)): (9.8, 10), (F(57, 100), F(57, 100)): (5.402, 1.0)}.items():
            j = A.evsi(HALF, CAP, A.lik_pair(lb(qa), lb(qb)))
            assert float(j) == pytest.approx(joint, abs=1e-9)
            assert float(A.evsi(HALF, CAP, lb(qa)) + A.evsi(HALF, CAP, lb(qb))) == pytest.approx(total, abs=1e-9)

    def test_b10_the_correlation_trap_at_prior_one_half(self):
        t = ev.analyse_trap(0.5, 0.8, 0.7, 0.7)
        assert (round(t.true_value, 6), round(t.corr, 6), round(t.corr2, 6), round(t.rate, 6)) == (35.0, 39.0, 23.4, 52.0)
        (e1, q1, n1), (e2, q2, n2), (e3, q3, n3) = t.rules["corr"], t.rules["corr2"], t.rules["rate"]
        assert (round(e1, 2), q1, n1) == (8.6, 0.57, 20) and (round(e2, 2), q2, n2) == (-11.63, 0.79, 15) and (round(e3, 2), q3, n3) == (36.55, 0.57, 4)
        assert [ev.pct(abs(e) / 65) for e in (e1, e2, e3)] == ["13,2 %", "17,9 %", "56,2 %"]
        assert t.corr_above == 51
        for k in range(57, 101):                                                                             # exakt: Fehler = 20 (1 − q) ab q = 17/30
            q = F(k, 100)
            assert (2 * q - 1) * 65 - A.evsi(HALF, CAP, lb(q)) == 20 * (1 - q)
        under = [k for k in range(57, 101) if (2 * F(k, 100) - 1) ** 2 * 65 <= A.evsi(HALF, CAP, lb(F(k, 100)))]
        assert len(under) == 43 and [k for k in range(57, 101) if k not in under] == [57]

    def test_b11_other_priors_of_the_trap(self):
        t = ev.analyse_trap(0.5, 0.8, 0.7, 0.7)
        rows = {pp: (between, total, not_under, round(mx, 1)) for pp, _, _, between, total, not_under, mx in t.priors_rows}
        assert rows == {0.3: (32, 36, 51, 13.2), 0.4: (47, 47, 51, 4.5), 0.45: (49, 49, 51, 2.4), 0.5: (43, 44, 51, 8.6), 0.6: (27, 34, 51, 16.3), 0.8: (1, 17, 51, 15.4)}
        assert sum(v[2] for v in rows.values()) == 306
        e45 = [row for row in t.priors_rows if row[0] == 0.45][0]
        assert round(e45[1], 1) == 71.5 and ev.pct(e45[6] / e45[1]) == "3,3 %"

    def test_b12_stopping_rule_against_optimum(self):
        cs = [F(k, 10) for k in range(1, 201)]
        counts = {}
        for qq in (F(6, 10), F(7, 10), F(8, 10), F(9, 10)):
            v = A.evsi_series(HALF, CAP, qq, 60)
            less = more = 0
            worst = (F(0), None)
            last = F(0)
            for c in cs:
                b, rule = A.best_n(v, c), A.rule_n(v, c)
                less += rule < b
                more += rule > b
                if rule != b:
                    last = c
                loss = (v[b] - c * b) - (v[rule] - c * rule)
                if loss > worst[0]:
                    worst = (loss, c)
            counts[float(qq)] = (less, more, round(float(worst[0]), 2), float(worst[1]), float(last))
        assert counts[0.6][:2] == (27, 0) and counts[0.7][:2] == (60, 0) and counts[0.8][:2] == (71, 0) and counts[0.9][:2] == (50, 0)
        assert counts[0.6][2:4] == (1.27, 1.4) and counts[0.8][2:4] == (8.51, 3.2) and counts[0.9][2:4] == (7.2, 1.8) and counts[0.7][2:4] == (4.2, 4.2)
        assert counts[0.7][4] == 6.2                                                                           # ab c = 6,3 stimmen Regel und Optimum überein
        v = A.evsi_series(HALF, CAP, F(7, 10), 60)
        for c, rule, best, loss in ((1, 11, 13, 0.377), (2, 5, 9, 1.641), (3, 3, 5, 1.938), (5, 1, 3, 2.6), (6, 1, 3, 0.6)):
            c = F(c)
            assert (A.rule_n(v, c), A.best_n(v, c)) == (rule, best)
            assert r((v[best] - c * best) - (v[rule] - c * rule), 3) == loss
        assert (r(v[3] - 15, 1), r(v[1] - 5, 1)) == (17.6, 15.0)
        assert A.rule_n(v, F(63, 10)) == 1 == A.best_n(v, F(63, 10)) and v[1] - F(63, 10) == v[3] - 3 * F(63, 10)   # Gleichstand bei c = 6,3: die kleinere Zahl
        vf = A.evsi_series((0.5, 0.5), CAPF, 0.7, 60)                                                         # Gleitkomma mit Toleranz: dieselbe Zählung
        assert sum(1 for k in range(1, 201) if A.rule_n(vf, k / 10, 1e-9) != A.best_n(vf, k / 10, 1e-9)) == 60

    def test_b13_random_problems(self):
        st = S.problem_stats(11001, 200000)
        assert st.chain_violations == 0 and st.cond_violations == 0 and st.median_ratio == 0.0
        assert st.dominated == 100675 and st.nondominated == 99325                                           # ganzzahlig, plattformfest
        assert abs(st.zero_nd - 65864) <= 2 and abs(st.zero_nd_q8 - 48261) <= 2
        assert round(st.mean_ratio, 3) == 0.112
        assert ev.pct(st.dominated / st.m) == "50,3 %" and ev.pct(st.zero_nd / st.nondominated) == "66,3 %" and ev.pct(st.zero_nd_q8 / st.nondominated) == "48,6 %"
        zero, dom, q8 = [], [], []
        for seed in range(11001, 11006):
            s = S.problem_stats(seed, 200000)
            assert s.chain_violations == 0 and s.cond_violations == 0
            zero.append(100 * s.zero_nd / s.nondominated)
            dom.append(100 * s.dominated / s.m)
            q8.append(100 * s.zero_nd_q8 / s.nondominated)
        assert (round(min(zero), 1), round(max(zero), 1)) == (66.0, 66.4)
        assert (round(min(dom), 1), round(max(dom), 1)) == (50.2, 50.4)
        assert (round(min(q8), 1), round(max(q8), 1)) == (48.1, 48.6)


# --- Schnellstart: jedes Szenario meldet die erwartete Verdikt-Zeile --------------------------------------------------------------------------------

def _verdicts(name):
    s = ev.Settings(**C.PRESETS[name])
    return {
        "test": ev.verdict_test(ev.analyse_test(s)),
        "zero": ev.verdict_zero(ev.analyse_zero(s.p)),
        "cost": ev.verdict_cost(ev.analyse_cost(s.p, s.c)),
        "multi": ev.verdict_multi(ev.analyse_multi(s.p, s.q4, s.n, s.c)),
        "trap": ev.verdict_trap(ev.analyse_trap(s.p, s.q5, s.qa, s.qb)),
    }


def test_preset_hand_example_reports_35_gu_and_7_13_of_evpi():
    v = _verdicts("Kapazitätsfall, Test 0,8 (von Hand)")
    assert v["test"][0] == "success" and "**EVSI = 35,0 GE** = 53,8 % des EVPI von 65,0 GE" in v["test"][1]
    assert "wahr 35,0 GE; Korrelation × EVPI = 39,0 GE" in v["trap"][1]


def test_preset_worthless_test_reports_worthless_and_no_effect():
    v = _verdicts("Wertloser Test (0,55)")
    assert v["test"][0] == "warning" and "**Wertlos:**" in v["test"][1] and "EVSI = 0 GE" in v["test"][1]
    assert "Ein einzelner Test ist wertlos, erst ab 2 Tests ist EVSI > 0." in v["multi"][1]
    assert ev.analyse_test(ev.Settings(**C.PRESETS["Wertloser Test (0,55)"])).evsi <= TOL


def test_preset_just_above_threshold_reports_half_a_gu():
    v = _verdicts("Knapp über der Schwelle (0,57)")
    assert v["test"][0] == "success" and "**EVSI = 0,5 GE** = 0,8 % des EVPI von 65,0 GE" in v["test"][1]
    assert "nach + G, nach − K" in v["test"][1]


def test_preset_one_sided_test_reports_32_5_gu():
    v = _verdicts("Einseitiger Test (se 1, sp 1/2)")
    assert "**EVSI = 32,5 GE** = 50,0 % des EVPI von 65,0 GE" in v["test"][1] and "nach + G, nach − K" in v["test"][1]


def test_preset_prior_at_the_threshold_reports_value_above_zero_for_a_weak_test():
    v = _verdicts("Vorab an der Schwelle (13/30)")
    a = ev.analyse_test(ev.Settings(**C.PRESETS["Vorab an der Schwelle (13/30)"]))
    assert not a.worthless and a.evsi > 0.1 and v["test"][0] == "success"
    assert "jeder Test mit einer Trefferquote über 1/2 hat einen Wert über 0" in v["zero"][1] and "Vorab = 13/30 = 0,4333" in v["zero"][1]
    exact = A.evsi((P_STAR, 1 - P_STAR), CAP, lb(F(52, 100)))
    assert a.evsi == pytest.approx(float(exact), abs=1e-9) and exact > 0


def test_preset_two_tests_reports_24_2_instead_of_40():
    v = _verdicts("Zwei Tests (0,7)")
    assert "bei 2 Tests ist EVSI = 24,20 GE" in v["multi"][1]
    assert "einzeln 20,00 und 20,00 GE, gemeinsam 24,20 GE, also weniger als die Summe 40,00 GE" in v["trap"][1]


def test_preset_test_cost_5_reports_rule_one_test_optimum_three():
    v = _verdicts("Testkosten 5 GE")
    assert v["multi"][0] == "warning"
    assert "hört die Abbruchregel nach 1 Test auf (Nettowert 15,0 GE), das Optimum liegt bei 3 Tests (17,6 GE)" in v["multi"][1]
    assert "$q^*$ = 0,6000" in v["cost"][1] and "7,7 % des EVPI" in v["cost"][1]
    assert "ENBS = +30,0 GE" in v["test"][1]                                                                  # 35 − 5


def test_preset_two_individually_worthless_tests_report_zero_individually_and_2_45_together():
    v = _verdicts("Zwei einzeln wertlose Tests (0,55)")
    assert "einzeln 0,00 und 0,00 GE, gemeinsam 2,45 GE, also mehr als die Summe 0,00 GE" in v["trap"][1]
    assert "bei 2 Tests ist EVSI = 2,45 GE" in v["multi"][1] and "Ein einzelner Test ist wertlos" in v["multi"][1]
    assert v["test"][0] == "warning" and "**Wertlos:**" in v["test"][1]


def test_no_preset_without_effect_reports_an_effect():
    """Ein Szenario, in dem der Test die Wahl nicht ändert, meldet in Experiment 1 keinen Wert; ein Szenario mit Wert meldet keinen Wertlos-Text."""
    for name, preset in C.PRESETS.items():
        a = ev.analyse_test(ev.Settings(**preset))
        text = ev.verdict_test(a)[1]
        assert (a.evsi <= TOL) == ("**Wertlos:**" in text), name
        assert (a.evsi > TOL) == ("**EVSI =" in text), name


def test_denominator_note_evpi_over_ev0_would_read_130_percent():
    """README, Modell: bei Erträgen ist EVPI/EV0 = 65/50 = 130 % und irreführend; die App nennt GE und den Anteil an EVPI mit dem Nenner."""
    a = ev.analyse_test(ev.Settings())
    assert a.evpi / a.ev0 == pytest.approx(1.3, abs=1e-12)
    assert a.ratio == pytest.approx(7 / 13, abs=1e-12)
