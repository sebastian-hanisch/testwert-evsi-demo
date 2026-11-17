"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text). Die Analysen kommen aus den Einstellungen, deren Werte von Hand nachgerechnet sind (EVSI 35 bei q = 0,8, 17/30, 24,2 bei zwei Tests mit 0,7);
Schwellen werden mit `dataclasses.replace` an einer Zeile verschoben."""

from dataclasses import replace

import tev_evaluation as ev
import tev_scenario as S


def _test(**kw):
    return ev.analyse_test(ev.Settings(**kw))


# --- Formatierung ---------------------------------------------------------------------------------------------------------------------------

def test_german_number_formats():
    assert ev.de(1.5, 2) == "1,50" and ev.de(-0.75, 1) == "−0,8" and ev.de(35.0, 1) == "35,0"
    assert ev.de(2.0, 1, sign=True) == "+2,0" and ev.de(-5.0, 1, sign=True) == "−5,0"
    assert ev.de(0.0, 1, sign=True) == "0,0" and ev.de(-0.00001, 1, sign=True) == "0,0"           # null hat kein Vorzeichen
    assert ev.pct(7 / 13) == "53,8 %" and ev.pct(0.226, 1) == "22,6 %"
    assert ev.tests(1) == "1 Test" and ev.tests(3) == "3 Tests"
    assert ev.de_int(999) == "999" and ev.de_int(1000) == "1 000" and ev.de_int(10201) == "10 201" and ev.de_int(200000) == "200 000"


# --- Experiment 1 -------------------------------------------------------------------------------------------------------------------------------

def test_verdict_test_names_choice_evsi_share_and_cost_threshold():
    kind, text = ev.verdict_test(_test())                                           # q = 0,8, c = 0
    assert kind == "success"
    for part in ("Sensitivität 0,80 und Spezifität 0,80", "nach + G, nach − K", "**EVSI = 35,0 GE** = 53,8 % des EVPI von 65,0 GE", "EV0 = 50,0 GE",
                 "solange er weniger als 35,0 GE kostet", "bei 0,0 GE ist ENBS = +35,0 GE"):
        assert part in text, part


def test_verdict_test_threshold_is_inclusive_at_cost_equal_to_evsi():
    assert ev.verdict_test(_test(c=34.5))[0] == "success"
    kind, text = ev.verdict_test(_test(c=35.0))                                     # ENBS genau 0: nicht kaufen
    assert kind == "warning" and "lohnt er nicht: ENBS = 0,0 GE (nicht kaufen)" in text
    kind, text = ev.verdict_test(_test(c=40.0))
    assert kind == "warning" and "Bei Kosten von 40,0 GE" in text and "ENBS = −5,0 GE" in text


def test_verdict_test_worthless_with_and_without_a_useful_looking_test():
    kind, text = ev.verdict_test(_test(se=0.55, sp=0.55))                           # se + sp = 1,1 > 1
    assert kind == "warning" and text.startswith("**Wertlos:**") and "dieselbe Wahl (G), EVSI = 0 GE" in text and "(se + sp > 1: er sieht nützlich aus)" in text
    assert "bringt auch ohne Kosten nichts" in text
    kind, text = ev.verdict_test(_test(se=0.5, sp=0.5, c=5.0))
    assert "sieht nützlich aus" not in text and "lohnt auch zum Preis von 5,0 GE nicht (ENBS = −5,0 GE)" in text


def test_verdict_test_one_sided_test_and_no_information_case():
    kind, text = ev.verdict_test(_test(se=1.0, sp=0.5))
    assert kind == "success" and "**EVSI = 32,5 GE** = 50,0 % des EVPI" in text
    a = replace(_test(), evpi=0.0, ratio=None)
    kind, text = ev.verdict_test(a)
    assert kind == "info" and "EVPI = 0" in text


# --- Experiment 2 -------------------------------------------------------------------------------------------------------------------------------

def test_verdict_zero_at_prior_one_half_names_threshold_formula_and_counts():
    kind, text = ev.verdict_zero(ev.analyse_zero(0.5))
    assert kind == "info"
    for part in ("q = 17/30 = 0,567 wertlos", "für q < 13/30 ist der Test nicht wertlos", "EVSI = 150 q − 85 GE", "das EVPI von 65 GE", "2 383 von 10 201 Punkten (23,4 %)", "1 141 von 5 050 der Punkte mit se + sp > 1 (22,6 %)"):
        assert part in text, part


def test_verdict_zero_for_another_prior_and_at_the_threshold_of_the_choice():
    text = ev.verdict_zero(ev.analyse_zero(0.45))[1]
    assert "Vorab 0,450" in text and "q = 0,5169 wertlos" in text and "2 von 51 Punkten" in text and "665 von 10 201" in text
    text = ev.verdict_zero(ev.analyse_zero(13 / 30))[1]
    assert "Vorab = 13/30 = 0,4333" in text and "jeder Test mit einer Trefferquote über 1/2 hat einen Wert über 0" in text and "101 von 10 201" in text and "0 von 5 050" in text


# --- Experiment 3 -------------------------------------------------------------------------------------------------------------------------------

def test_verdict_cost_gives_the_mindest_hit_rate_and_the_share_of_evpi():
    kind, text = ev.verdict_cost(ev.analyse_cost(0.5, 5.0))
    assert kind == "success" and "Testkosten c = 5,0 GE (7,7 % des EVPI von 65,0 GE)" in text and "$q^*$ = 0,6000" in text and "(85 + c)/150" in text
    assert "$q^*$ = 0,8000" in ev.verdict_cost(ev.analyse_cost(0.5, 35.0))[1]


def test_verdict_cost_never_when_cost_reaches_evpi_and_free_case():
    kind, text = ev.verdict_cost(ev.analyse_cost(0.5, 65.0))
    assert kind == "warning" and "**nie**" in text and "(100,0 % des EVPI von 65,0 GE)" in text
    assert ev.verdict_cost(ev.analyse_cost(0.5, 64.0))[0] == "success"            # knapp darunter gibt es noch eine Trefferquote
    text = ev.verdict_cost(ev.analyse_cost(0.8, 30.0))[1]                           # EVPI bei Vorab 0,8 ist 26
    assert "(115,4 % des EVPI von 26,0 GE)" in text and "**nie**" in text
    kind, text = ev.verdict_cost(ev.analyse_cost(0.5, 0.0))
    assert kind == "info" and "ab der Wertlos-Schwelle q = 0,5667" in text


# --- Experiment 4 -------------------------------------------------------------------------------------------------------------------------------

def test_verdict_multi_without_cost_names_increments_and_counts():
    kind, text = ev.verdict_multi(ev.analyse_multi(0.5, 0.7, 3, 0.0))
    assert kind == "info"
    for part in ("20,00; 4,20; 8,40; 2,65 GE", "29 von 59 Stellen (n = 2 bis 60), davon 29 bei ungeraden n", "nicht konkav", "bei 3 Tests ist EVSI = 32,60 GE (50,2 % des EVPI von 65,0 GE)"):
        assert part in text, part


def test_verdict_multi_stopping_rule_against_optimum_at_cost_5_and_at_cost_10():
    kind, text = ev.verdict_multi(ev.analyse_multi(0.5, 0.7, 3, 5.0))
    assert kind == "warning" and "hört die Abbruchregel nach 1 Test auf (Nettowert 15,0 GE)" in text and "das Optimum liegt bei 3 Tests (17,6 GE)" in text
    kind, text = ev.verdict_multi(ev.analyse_multi(0.5, 0.7, 3, 10.0))
    assert kind == "info" and "stimmen die Abbruchregel und das Optimum überein: 1 Test, Nettowert 10,0 GE" in text


def test_verdict_multi_individually_worthless_tests_and_the_concave_cell():
    text = ev.verdict_multi(ev.analyse_multi(0.5, 0.55, 1, 0.0))[1]
    assert "Ein einzelner Test ist wertlos, erst ab 2 Tests ist EVSI > 0." in text and "0,00; 2,45; 0,62; 1,82 GE" in text
    assert "erst ab 4 Tests ist EVSI > 0" in ev.verdict_multi(ev.analyse_multi(0.5, 0.52, 1, 0.0))[1]
    text = ev.verdict_multi(ev.analyse_multi(0.6, 0.8, 3, 0.0))[1]
    assert "fällt der Zuwachs über n = 2 bis 60 durchgehend" in text and "konkav" in text and "nicht konkav" not in text


def test_verdict_multi_test_without_content_stays_worthless():
    kind, text = ev.verdict_multi(ev.analyse_multi(0.5, 0.5, 3, 0.0))
    assert kind == "warning" and "auch 60 gleichartige Tests zusammen sind wertlos" in text and "Ein Test ohne Aussage (q = 1/2)" in text


# --- Experiment 5 -------------------------------------------------------------------------------------------------------------------------------

def test_verdict_trap_names_the_four_values_and_the_pair():
    kind, text = ev.verdict_trap(ev.analyse_trap(0.5, 0.8, 0.7, 0.7))
    assert kind == "warning"                                                         # 39 gegen 35: Abstand 4 GE ist mehr als 5 % von 65 (3,25)
    for part in ("wahr 35,0 GE", "Korrelation × EVPI = 39,0 GE", "Korrelation² × EVPI = 23,4 GE", "Trefferquote × EVPI = 52,0 GE", "= 0,600",
                 "an allen 51 Punkten nicht unter dem wahren Wert", "kein Beweis", "einzeln 20,00 und 20,00 GE, gemeinsam 24,20 GE, also weniger als die Summe 40,00 GE"):
        assert part in text, part


def test_verdict_trap_pair_relations_more_equal_and_less():
    assert "also mehr als die Summe 1,00 GE" in ev.verdict_trap(ev.analyse_trap(0.5, 0.8, 0.57, 0.57))[1]   # 5,4 gegen 1,0: gemeinsam mehr
    assert "also genau die Summe 20,00 GE" in ev.verdict_trap(ev.analyse_trap(0.5, 0.8, 0.7, 0.5))[1]      # ein Test ohne Aussage fügt nichts hinzu
    assert "gemeinsam 2,45 GE" in ev.verdict_trap(ev.analyse_trap(0.5, 0.8, 0.55, 0.55))[1]


def test_verdict_trap_boundary_of_the_warning_kind_is_5_percent_of_evpi():
    a = ev.analyse_trap(0.5, 0.8, 0.7, 0.7)
    assert ev.verdict_trap(replace(a, corr=a.true_value + 0.05 * a.evpi - 1e-6))[0] == "info"
    assert ev.verdict_trap(replace(a, corr=a.true_value + 0.05 * a.evpi + 1e-6))[0] == "warning"
    text = ev.verdict_trap(replace(a, corr_above=40))[1]
    assert "an 40 von 51 Punkten nicht unter dem wahren Wert" in text and "keine obere Schranke" in text


# --- Zufallsproblem-Gegenprobe ----------------------------------------------------------------------------------------------------------------

def _stats(**kw):
    base = dict(m=1000, dominated=500, nondominated=500, zero_nd=330, zero_nd_q8=240, chain_violations=0, cond_violations=0, median_ratio=0.0, mean_ratio=0.11)
    base.update(kw)
    return S.ProblemStats(**base)


def test_verdict_random_names_counts_shares_and_the_chain():
    kind, text = ev.verdict_random(_stats())
    assert kind == "success"
    for part in ("1 000 Zufallsprobleme: 500 dominiert (eine Alternative ist in beiden Zuständen mindestens so gut", "500 nicht dominiert", "(beide über die nicht dominierten Probleme)", "gilt in allen 1 000 Problemen", "EVSI = 0 in 66,0 %", "48,0 % (Test mit Trefferquote 0,8)",
                 "Median EVSI/EVPI 0,000, Mittel 0,110", "in 500 von 500 nicht dominierten Problemen"):
        assert part in text, part


def test_verdict_random_violations_and_all_dominated():
    kind, text = ev.verdict_random(_stats(chain_violations=3))
    assert kind == "warning" and "verletzt in 3 von 1 000" in text
    kind, text = ev.verdict_random(_stats(cond_violations=2))
    assert kind == "warning" and "in 498 von 500" in text
    kind, text = ev.verdict_random(_stats(dominated=1000, nondominated=0, zero_nd=0, zero_nd_q8=0))
    assert kind == "info" and "Alle 1 000 Zufallsprobleme sind dominiert" in text
