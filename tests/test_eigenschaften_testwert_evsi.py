"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen über den Wert eines Tests, die ohne das richtige Ergebnis prüfbar sind.

Je Eigenschaft 40 Zufallsinstanzen aus dem Zufallsstrom SplitMix64 (feste Zufalls-Seeds), exakt in Brüchen. Jede Eigenschaft gilt für den Erwartungswert (linearer Nutzen)
mit zwei Zuständen; die Ordnung EVPI >= EVSI >= 0 gilt so nur dort."""

from fractions import Fraction as F

import tev_algorithm as A
import tev_rng as R

N = 40


def instance(seed):
    """Zufallsinstanz: Vorab k/20 (k = 1..19), zwei Alternativen mit ganzzahligen Auszahlungen in [-100, 200], Test mit se, sp = k/10."""
    g = R.SplitMix64(seed)
    p = F(1 + g.below(19), 20)
    pay = {"A": (F(g.below(301) - 100), F(g.below(301) - 100)), "B": (F(g.below(301) - 100), F(g.below(301) - 100))}
    se, sp = F(g.below(11), 10), F(g.below(11), 10)
    return (p, 1 - p), pay, se, sp


def test_evsi_is_between_zero_and_evpi():
    for seed in range(N):
        prior, pay, se, sp = instance(seed)
        v = A.evsi(prior, pay, A.lik_bin(se, sp))
        assert 0 <= v <= A.evpi(prior, pay), seed


def test_perfect_test_gives_evpi_and_test_without_content_gives_zero():
    for seed in range(N):
        prior, pay, se, sp = instance(seed)
        assert A.evsi(prior, pay, A.lik_bin(F(1), F(1))) == A.evpi(prior, pay)
        assert A.evsi(prior, pay, A.lik_bin(se, 1 - se)) == 0                      # se + sp = 1: das Ergebnis hängt nicht vom Zustand ab


def test_renaming_alternatives_and_outcomes_changes_nothing():
    for seed in range(N):
        prior, pay, se, sp = instance(seed)
        base = A.evsi(prior, pay, A.lik_bin(se, sp))
        assert A.evsi(prior, {"B": pay["B"], "A": pay["A"]}, A.lik_bin(se, sp)) == base               # Reihenfolge der Alternativen
        assert A.evsi(prior, pay, A.lik_bin(1 - se, 1 - sp)) == base                                   # Ergebnisse + und − vertauscht (se, sp) -> (1 − se, 1 − sp)


def test_swapping_the_two_states_changes_nothing():
    for seed in range(N):
        prior, pay, se, sp = instance(seed)
        swapped_pay = {k: (v[1], v[0]) for k, v in pay.items()}
        # Zustände getauscht und dabei die Ergebnisse umbenannt: P(+' | neu hoch) = P(− | niedrig) = sp, P(−' | neu niedrig) = P(+ | hoch) = se
        assert A.evsi((prior[1], prior[0]), swapped_pay, A.lik_bin(sp, se)) == A.evsi(prior, pay, A.lik_bin(se, sp))


def test_scaling_payoffs_scales_evsi_and_shifting_changes_nothing():
    for seed in range(N):
        prior, pay, se, sp = instance(seed)
        base = A.evsi(prior, pay, A.lik_bin(se, sp))
        scaled = {k: (3 * v[0], 3 * v[1]) for k, v in pay.items()}
        shifted = {k: (v[0] + 17, v[1] + 17) for k, v in pay.items()}
        assert A.evsi(prior, scaled, A.lik_bin(se, sp)) == 3 * base
        assert A.evsi(prior, shifted, A.lik_bin(se, sp)) == base


def test_adding_noise_to_the_test_never_raises_evsi():
    """Blackwell: dreht man das Ergebnis mit Wahrscheinlichkeit e um, ist der Test eine Vergröberung des alten und nie mehr wert."""
    for seed in range(N):
        prior, pay, se, sp = instance(seed)
        e = F(1 + R.SplitMix64(seed + 1000).below(4), 10)
        noisy = A.lik_bin(se * (1 - e) + (1 - se) * e, sp * (1 - e) + (1 - sp) * e)
        assert A.evsi(prior, pay, noisy) <= A.evsi(prior, pay, A.lik_bin(se, sp)), seed


def test_two_tests_together_are_worth_at_least_each_and_at_most_evpi():
    for seed in range(N):
        prior, pay, se, sp = instance(seed)
        qa, qb = F(5 + R.SplitMix64(seed).below(6), 10), F(5 + R.SplitMix64(seed + 7).below(6), 10)
        la, lb = A.lik_bin(qa, qa), A.lik_bin(qb, qb)
        joint = A.evsi(prior, pay, A.lik_pair(la, lb))
        assert max(A.evsi(prior, pay, la), A.evsi(prior, pay, lb)) <= joint <= A.evpi(prior, pay), seed


def test_evsi_does_not_fall_when_a_test_is_added():
    for seed in range(N):
        prior, pay, _, _ = instance(seed)
        q = F(5 + R.SplitMix64(seed + 3).below(6), 10)
        vals = A.evsi_series(prior, pay, q, 8)
        assert all(vals[n] >= vals[n - 1] for n in range(1, 9)), seed
        assert vals[0] == 0


def test_seed_gives_the_same_output():
    assert instance(5) == instance(5)
    assert [R.SplitMix64(9).below(10) for _ in range(3)] == [R.SplitMix64(9).below(10) for _ in range(3)]


def test_rollback_posteriors_are_probabilities_and_average_back_to_the_prior():
    for seed in range(N):
        prior, pay, se, sp = instance(seed)
        rows = A.rollback(prior, pay, A.lik_bin(se, sp))
        assert sum(r["p_y"] for r in rows) == 1
        assert sum(r["p_y"] * r["posterior"] for r in rows if r["posterior"] is not None) == prior[0]      # Mittel der Posterior = Vorab
        assert all(0 <= r["posterior"] <= 1 for r in rows if r["posterior"] is not None)


def test_more_cost_never_raises_the_optimal_number_of_tests():
    vals = A.evsi_series((F(1, 2), F(1, 2)), {"G": (F(200), F(-100)), "K": (F(30), F(30))}, F(7, 10), 20)
    chosen = [A.best_n(vals, F(c, 2)) for c in range(0, 60)]
    assert all(chosen[i] >= chosen[i + 1] for i in range(len(chosen) - 1))
