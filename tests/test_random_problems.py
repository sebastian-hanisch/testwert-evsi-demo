"""Zufallsprobleme der Gegenprobe: Aufbau aus dem Zufallsstrom, Vektorrechnung gegen den Einzelrechenweg des Kerns, Zählungen gegen eine Schleife."""

import numpy as np
import pytest

import tev_algorithm as A
import tev_rng as R
import tev_scenario as S


def _from_class_stream(seed, m):
    """Dieselben Probleme aus der Klasse (Block 1: Vorab, Blöcke 2 bis 5: vier Auszahlungen, Block 6 und 7: se, sp), Wert für Wert gezogen."""
    g = R.SplitMix64(seed)
    raw = [g.next_u64() for _ in range(7 * m)]
    out = []
    for i in range(m):
        p = S.PRIOR_LO + (S.PRIOR_HI - S.PRIOR_LO) * ((raw[i] >> 11) / float(1 << 53))
        pay = [int(raw[(1 + j) * m + i] % 301) - 100 for j in range(4)]
        se = (raw[5 * m + i] >> 11) / float(1 << 53)
        sp = (raw[6 * m + i] >> 11) / float(1 << 53)
        out.append((p, pay, se, sp))
    return out


def test_random_problems_equal_the_class_stream():
    d = S.random_problems(11001, 40)
    for i, (p, pay, se, sp) in enumerate(_from_class_stream(11001, 40)):
        assert d["p"][i] == p and d["se"][i] == se and d["sp"][i] == sp
        assert [d["g_hi"][i], d["g_lo"][i], d["k_hi"][i], d["k_lo"][i]] == pay


def test_ranges_of_the_random_problems():
    d = S.random_problems(5, 5000)
    assert d["p"].min() >= 0.05 and d["p"].max() < 0.95
    for k in ("g_hi", "g_lo", "k_hi", "k_lo"):
        assert d[k].min() >= -100 and d[k].max() <= 200 and np.all(d[k] == np.round(d[k]))
    assert d["se"].min() >= 0 and d["se"].max() < 1 and d["sp"].min() >= 0 and d["sp"].max() < 1


def test_vector_evsi_equals_the_core_per_problem():
    d = S.random_problems(11001, 300)
    evsi, evpi = S._evsi_arrays(d, d["se"], d["sp"])
    for i in range(300):
        prior = (float(d["p"][i]), 1 - float(d["p"][i]))
        pay = {"G": (float(d["g_hi"][i]), float(d["g_lo"][i])), "K": (float(d["k_hi"][i]), float(d["k_lo"][i]))}
        assert evsi[i] == pytest.approx(A.evsi(prior, pay, A.lik_bin(float(d["se"][i]), float(d["sp"][i]))), abs=1e-9)
        assert evpi[i] == pytest.approx(A.evpi(prior, pay), abs=1e-9)


def test_counts_equal_a_loop_over_the_problems():
    m = 400
    d = S.random_problems(77, m)
    st = S.problem_stats(77, m)
    dom = zero = zero8 = 0
    for i in range(m):
        prior = (float(d["p"][i]), 1 - float(d["p"][i]))
        pay = {"G": (float(d["g_hi"][i]), float(d["g_lo"][i])), "K": (float(d["k_hi"][i]), float(d["k_lo"][i]))}
        g, k = pay["G"], pay["K"]
        if (g[0] >= k[0] and g[1] >= k[1]) or (k[0] >= g[0] and k[1] >= g[1]):
            dom += 1
            continue
        zero += A.evsi(prior, pay, A.lik_bin(float(d["se"][i]), float(d["sp"][i]))) <= A.ZERO_TOL
        zero8 += A.evsi(prior, pay, A.lik_bin(0.8, 0.8)) <= A.ZERO_TOL
    assert (st.dominated, st.nondominated, st.zero_nd, st.zero_nd_q8) == (dom, m - dom, zero, zero8)
    assert st.chain_violations == 0 and st.cond_violations == 0


def test_dominated_problems_have_value_zero_for_every_test():
    d = S.random_problems(3, 2000)
    evsi, _ = S._evsi_arrays(d, d["se"], d["sp"])
    dom = ((d["g_hi"] >= d["k_hi"]) & (d["g_lo"] >= d["k_lo"])) | ((d["k_hi"] >= d["g_hi"]) & (d["k_lo"] >= d["g_lo"]))
    assert dom.sum() > 0 and np.all(evsi[dom] <= A.ZERO_TOL)


def test_same_seed_same_statistics_and_other_seed_other_statistics():
    assert S.problem_stats(11001, 1000) == S.problem_stats(11001, 1000)
    assert S.problem_stats(11001, 1000) != S.problem_stats(11002, 1000)
