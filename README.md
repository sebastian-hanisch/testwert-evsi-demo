# Ein Test und sein Wert – Bayes-Update und EVSI – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-testwert-evsi-demo.streamlit.app/)** -->

Eine von mehreren Demos zur **Entscheidungsanalyse** (Entscheiden unter Unsicherheit) für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie nimmt eine Annahme heraus, die [`baum-strategie-demo`](https://github.com/sebastian-hanisch/baum-strategie-demo) noch macht: dass die Wahl jetzt und ohne weitere Information fällt. Vor die Wahl tritt ein **Test** mit bekannter Trefferquote, der etwas kostet; die Demo rechnet das **Bayes-Update im umgedrehten Baum**, den **Wert der Stichprobeninformation (EVSI)** und die Schwellen, ab denen sich ein Test lohnt, und zeigt in fünf Experimenten, wann ein Test wertlos ist, warum sich sein Wert weder aus der Trefferquote noch aus „Korrelation × EVPI“ ablesen lässt und warum sich die Werte mehrerer Tests nicht addieren.
**Das Zahlenbeispiel ist ausgedacht** (Kapazitätsfall, Geldeinheiten GE): G (groß bauen) bringt +200 GE bei hoher und −100 GE bei niedriger Nachfrage, K (klein bauen oder mieten) sicher +30 GE, ohne Information gilt P(hoch) = 1/2, bewertet wird der Erwartungswert. Es ist ein **Modell, kein Fall aus der Praxis**; die Trefferquote des Tests ist eine Eingabe und kein Befund über einen echten Test. „Entscheidungsbaum“ heißt hier der Entscheidungsbaum mit Ereignisknoten, nicht der Baum aus Daten ([`cart-demo`](https://github.com/sebastian-hanisch/cart-demo)); „Regret“ meint den ex-post-Mehraufwand gegen einen Hellseher (im Erwartungswert der Opportunitätsverlust), nicht das Savage-Kriterium.
Nachbarn: [`baum-strategie-demo`](https://github.com/sebastian-hanisch/baum-strategie-demo) (Rollback und Strategien im Baum, der Test ist dort noch nicht im Baum), [`kipp-schwelle-demo`](https://github.com/sebastian-hanisch/kipp-schwelle-demo) (Vorab und Trefferquote sind dort nicht bekannt), [`information-risikoaversion-demo`](https://github.com/sebastian-hanisch/information-risikoaversion-demo) (derselbe Test unter Risikoaversion), [`newsvendor-evpi-demo`](https://github.com/sebastian-hanisch/newsvendor-evpi-demo) (EVPI und VSS für ein Lagerproblem mit Szenarien, ohne verrauschte Hinweise; hier steht ein Ertrag, dort Kosten), `geschachtelte-erwartungen-demo` (EVSI als Schätzproblem bei stetigen Parametern; hier ist der Test diskret und EVSI exakt), [`test-power-demo`](https://github.com/sebastian-hanisch/test-power-demo) (statistischer Test und Stichprobenumfang, keine Geldrechnung) und [`kipppunkt-bifurkation-demo`](https://github.com/sebastian-hanisch/kipppunkt-bifurkation-demo) (Frühwarnung mit Alarmanteilen, ohne Entscheidungswert). Die README dieser Nachbarn wurde nicht in allen Fällen gelesen (`geschachtelte-erwartungen-demo` nicht); die Abgrenzung stützt sich dort auf die Kurzbeschreibung der Website.

## Warum dieses Problem

Ein Test lohnt sich nur, wenn sein Ergebnis die Wahl kippen kann. **Wann ist das so, ab welcher Trefferquote lohnt er bei Kosten, und wie rechnet man mehrere Tests zusammen?** Im Kapazitätsfall mit Vorab 1/2, symmetrischem Test (Sensitivität = Spezifität = Trefferquote q) und Erwartungswert in GE gilt: ein Test mit 1/2 ≤ q ≤ 17/30 = 0,567 ist **exakt wertlos** (für q < 13/30 ist er es nicht: q < 1/2 wird spiegelbildlich gelesen, EVSI(0,1) = 50 GE), darüber wächst sein Wert als 150 q − 85 GE bis zum EVPI von 65 GE; bei q = 0,8 sind es 35 GE, das sind 7/13 = 53,8 % des EVPI (Nenner: EVPI), nicht 80 % (= q) und nicht 60 % (Korrelation 2q − 1 als Anteil am EVPI). Die Falle: der Wert lässt sich aus der Trefferquote nicht ablesen. „Korrelation × EVPI“ gibt bei q = 0,8 den Wert 39 statt 35 GE und liegt an keinem der 51 Rasterpunkte q = 0,50 bis 1,00 unter dem wahren Wert (Vorab 1/2; für sechs Vorab-Werte an allen 306 Rasterpunkten gezählt, kein Beweis); und Informationswerte addieren sich nicht: zwei Tests mit q = 0,7 sind zusammen 24,2 GE wert statt 2 × 20, zwei einzeln wertlose Tests mit q = 0,55 zusammen 2,45 GE. Weil EVSI in der Testzahl nicht konkav wächst (Vorab 1/2, q = 0,7), hört die Regel „kaufe weiter, solange der nächste Test mehr bringt als er kostet“ bei 5 GE je Test nach einem Test auf, obwohl drei Tests 2,6 GE mehr Nettowert geben (17,6 gegen 15).
Alle Formeln und Befunde sind **eigene Nachrechnung in diesem Modell, bei diesen Parametern** (zwei Zustände, zwei Alternativen, Erwartungswert, Test mit bekannter Trefferquote, Wiederholungen bedingt unabhängig); Zähleinheiten sind Rasterpunkt, Testzahl, Testpaar und Zufallsproblem. Eine Neuheitsaussage gibt es nicht: Bayes-Update, EVPI, EVSI und die Ordnung EVPI ≥ EVSI ≥ 0 bei linearem Nutzen stehen in Lehrbüchern (siehe Quellen); neu ist hier nur die Vorführbarkeit mit Zählungen.

## Modell

**Zustände und Alternativen.** Zustand hoch oder niedrig mit Vorab p = P(hoch); Auszahlungen u(a, s) in GE (**Ertrag, nicht Kosten**): u(G, hoch) = 200, u(G, niedrig) = −100, u(K, ·) = 30. EV(G) = 300 p − 100, EV(K) = 30; die Schwelle der Wahl ist p\* = 13/30, darüber wird G gewählt, darunter K, bei Gleichstand G (das ändert keinen Wert).
**Ohne Test.** **EV0** = max_a Σ_s p_s u(a, s) (Wert der besten Wahl ohne Information). **EVPI** = Σ_s p_s max_a u(a, s) − EV0 (Wert der vollkommenen Information).
**Test.** Ergebnis + oder −; Sensitivität se = P(+ | hoch), Spezifität sp = P(− | niedrig); bei se = sp heißt der Wert **Trefferquote q** (symmetrischer Test; q < 1/2 wird spiegelbildlich gelesen). **EVSI** = Σ_y max_a Σ_s P(s, y) u(a, s) − EV0, gerechnet über die verbundenen Wahrscheinlichkeiten P(s, y) = P(s) P(y | s) (Rollback über den umgedrehten Baum; die Posterior-Werte P(hoch | y) stehen in der App daneben). **Testkosten c**, **ENBS = EVSI − c** (Begriff nach Kunst u. a.: Differenz aus EVSI und Studienkosten). Zweiter Rechenweg: der erwartete **Verlust mit Test** (Mehraufwand gegen einen Hellseher) ist EVPI − EVSI.
**Mehrere Tests.** n gleichartige, bedingt unabhängige symmetrische Tests, ausgewertet über die Zahl k der positiven Ergebnisse (hinreichende Statistik: k enthält alles, was die Folge der Ergebnisse über den Zustand sagt; gegen die Aufzählung aller 2ⁿ Folgen für n ≤ 10 geprüft); ohne Test (n = 0) ist EVSI(0) = 0; zwei verschiedene Tests als Paar. Die Binomialgewichte werden durch wiederholte Multiplikation gebildet, nicht mit `**` (die Gleitkomma-Potenz kommt aus der C-Bibliothek).
**Nenner.** Bei Erträgen stehen GE und der Anteil an EVPI, der Nenner steht immer daneben. EV0 = 50 GE ist hier der Bezug (RP). Als Anteil an RP wäre EVPI hier 130 % und irreführend; die Demo zeigt GE und den Anteil an EVPI.
**Alles gilt für linearen Nutzen** (Erwartungswert in GE): die Ordnung EVPI ≥ EVSI ≥ 0 und die Zerlegung stehen so nur dort oder in der Zahlungsbereitschafts-Definition. Unter Risikoaversion: [`information-risikoaversion-demo`](https://github.com/sebastian-hanisch/information-risikoaversion-demo).
Die Formeln stehen im Expander 📐 der App.

## Methodik

| Baustein | Wie er arbeitet | Grenze |
|---|---|---|
| **Rollback** (`tev_algorithm.py`: `ev_with`, `evsi`, `rollback`) | je Ergebnis die beste Alternative auf den verbundenen Wahrscheinlichkeiten, über die Ergebnisse summiert; typgenerisch (Gleitkomma in der App, `Fraction` in den Tests) | zwei Zustände; Alternativen und Ergebnisse beliebig viele |
| **Schwellen** (`q0_closed`, `q_star`, `min_specificity`) | q0(p) geschlossen aus Quote und Schwellenquote; q\*(c) und die Mindest-Spezifität per Bisektion (EVSI ist konvex in q und sp; nicht fallend für q ≥ 1/2 beziehungsweise sp ≥ 1 − se, Minimum 0) | bei Kosten 0 ist q\* das Infimum der Trefferquoten mit EVSI > 0 |
| **Mehrere Tests** (`lik_n`, `evsi_series`, `rule_n`, `best_n`) | Binomialgewichte, EVSI für n = 1 bis 60, Abbruchregel und Optimum n\* = argmax (EVSI(n) − c n), bei Gleichstand die kleinere Zahl | nur symmetrisch und bedingt unabhängig |
| **Raster in numpy** (`evsi_bin_array`, `evsi_pair_array`, `phi_array`) | 101 × 101 Punkte (se, sp) und 51 × 51 Paare (qa, qb), vektorisiert, Gleitkomma mit Toleranz 10⁻⁹ GE | die Tests vergleichen die Zählungen mit der exakten Rechnung in Brüchen |
| **Zufallsprobleme** (`tev_scenario.py`) | SplitMix64 (`tev_rng.py`): Vorab gleichverteilt in [0,05; 0,95], vier Auszahlungen (je Alternative und Zustand eine, auch u(K, ·) darf je Zustand verschieden sein) ganzzahlig gleichverteilt in [−100, 200], se und sp gleichverteilt in [0, 1] | die Verteilung bestimmt die Zahlen |

Nichts ist eingefroren: alle Rechnungen laufen in der App (Messung dieser README, nicht im Test: 50 000 Zufallsprobleme rund 0,02 s auf dem Entwicklungsrechner, Windows, Python 3.14, NumPy 2.5).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen im **Kapazitätsfall** (G +200/−100, K +30), Zähleinheit in der Zeile; „Test“ ist `tests/test_claims.py`, wenn nichts anderes steht. Exakte Zahlen sind in Brüchen gerechnet, Zählungen aus der numpy-Rechnung und aus Zufallsströmen als Band geprüft (Plattformhinweis unten).

| Nr. | Frage | Befund (Voraussetzung im Satz) | Test |
|---|---|---|---|
| B1 | Von Hand | Vorab 1/2: EV0 = 50 GE, EVPI = 65 GE; Test mit q = 0,8: P(+) = 1/2, Posterior 0,8 und 0,2, nach + G (140), nach − K (30), Wert mit Test 85, **EVSI = 35 GE = 7/13 = 53,8 % des EVPI**; Verlust mit Test 30 = 65 − 35 (Aufzählung aller 4 Regeln, Baum, Verlustweg und lineares Programm (LP) über randomisierte Regeln gleich) | `test_algorithm`, `test_oracle_testwert_evsi` (exakt) |
| B2 | Wertlos-Schwelle | Vorab 1/2, symmetrisch, 51 Punkte q = k/100 mit k = 50 bis 100: EVSI(q) = max(0, 150 q − 85) (für q < 1/2 spiegelbildlich); **q0 = 17/30 = 0,567**; von 51 Rasterpunkten (q = 0,50 bis 1,00) sind **7 wertlos** (q ≤ 0,56); EVSI(0,57) = 0,5 GE; EVSI/EVPI = 7,7 / 30,8 / 53,8 / 76,9 / 88,5 % bei q = 0,6 / 0,7 / 0,8 / 0,9 / 0,95 | `test_oracle` (exakt), `TestBefunde::test_b2` |
| B3 | Schwelle je Vorab | symmetrisch, 19 Vorab-Werte p = k/20, exakt: q0 = 0,9356 (p = 0,05), 0,8731 (0,10), 0,6408 (0,30), 0,5342 (0,40), 0,5169 (0,45), 0,5667 (0,50), 0,6623 (0,60), 0,8395 (0,80), 0,9613 (0,95); bei p = 13/30 ist q0 = 1/2, jeder Test mit q > 1/2 (q = 0,51 bis 0,99, 49 Punkte) hat einen Wert über 0. Der Test ist also vor allem dann wertlos, wenn das Vorab weit von der Schwelle 13/30 liegt | `test_oracle`, `TestBefunde::test_b3` |
| B4 | Se/Sp-Raster | Vorab 1/2, 101 × 101 = 10 201 Rasterpunkte (Schritt 0,01): EVPI ≥ EVSI ≥ 0 an allen (0 Verletzungen); die Bedingung „Posterior nach + und nach − auf derselben Seite von 13/30“ stimmt mit der Nullmenge von EVSI an allen Punkten überein (0 Fehlzuordnungen); EVSI = 0 an **2 383 von 10 201 Punkten (23,4 %)**, davon 1 141 von 5 050 mit se + sp > 1 (22,6 %), also bei Tests, die nützlich aussehen | `test_oracle` (exakt), `TestBefunde::test_b4` |
| B5 | Ab welcher Trefferquote lohnt der Test? | Vorab 1/2, symmetrisch: **q\*(c) = (85 + c)/150** für c < 65, gleich der Bisektion (11 Werte, Abweichung unter 10⁻⁹): c = 1 gibt 0,5733, 5 gibt 0,6000 (c = 7,7 % des EVPI von 65 GE), 10 gibt 0,6333, 20 gibt 0,7000, 40 gibt 0,8333, 60 gibt 0,9667, 64 gibt 0,9933; **ab c = 65 nie**. Asymmetrisch, c = 5: Mindest-Spezifität bei se = 0,6 / 0,7 / 0,8 / 0,9 / 0,95 / 1,0: 0,6000 / 0,4692 / 0,3385 / 0,2077 / 0,1423 / 0,0769. Je Vorab (c = 5): q\* = 0,7899 (p = 0,2), 0,6761 (0,3), 0,5685 (0,4), 0,6000 (0,5), 0,6948 (0,6), 0,8704 (0,8) | `test_oracle`, `TestBefunde::test_b5` |
| B6 | Mehrere Tests | Vorab 1/2, q = 0,7, n unabhängige Tests (Zähleinheit Testzahl): EVSI(n) = 20 / 24,2 / 32,6 / 35,25 / 40,54 / 42,39 GE für n = 1 bis 6, 51,21 (n = 10), 60,43 (20), 63,36 (30), 64,39 (40), **64,91 (60) = 99,9 % des EVPI**; die Zuwächse je Test sind 20 / 4,2 / 8,4 / 2,65 / … (der vierte ist 2,646); EVSI ist **nicht konkav**: der Zuwachs steigt an **29 von 59 Stellen** (n = 2 bis 60), alle bei ungeraden n; für jede zweite Testzahl getrennt ist EVSI bis n = 60 konkav. Bei Vorab 1/2 und q = 0,6 / 0,7 / 0,8 / 0,9 (n = 1 bis 20) wird G genau dann gewählt, wenn k ≥ n/2; bei q = 0,55 gilt das nicht (n = 3, k = 1: Posterior 0,45 > 13/30, also G, obwohl k < n/2) | `test_oracle`, `test_algorithm`, `TestBefunde::test_b6` |
| B7 | Gilt das für andere Zellen? | 16 Zellen (Vorab 0,4 / 0,45 / 0,5 / 0,6 mal q = 0,6 / 0,7 / 0,8 / 0,9, n = 2 bis 40, exakt): der Zuwachs steigt in **15 von 16** Zellen an (an 19 von 39 Stellen in 14 Zellen, an 20 in einer); in der Zelle (Vorab 0,6; q = 0,8) fällt er durchgehend (konkav). Die Stufen (Testzahlen, an denen der Zuwachs gegenüber dem vorigen steigt) liegen in 12 Zellen bei ungeraden n, in zwei bei geraden (Vorab 0,5, q = 0,6; Vorab 0,6, q = 0,7) und in einer gemischt (Vorab 0,6, q = 0,6). „Nicht konkav“ gilt also nicht überall, und „bei jedem ungeraden n“ ist keine allgemeine Regel | `TestBefunde::test_b7` |
| B8 | Einzeln wertlose Tests | Vorab 1/2: bei q = 0,55 ist EVSI(1) = 0, aber EVSI(2) = 2,45 GE (nach k = 0 ist P(hoch) = 81/202 = 0,401 < 13/30, die Wahl kippt zu K), EVSI(3) = 3,069, (5) = 5,500, (10) = 10,553, (20) = 17,249, (50) = 29,829; bei q = 0,52 erst ab n = 4 positiv (0,2404; n = 50: 8,541); bei q = 17/30 genau ist n = 1 wertlos und n = 2 gibt 4,911; **ein Test ohne Aussage (q = 1/2) bleibt für jedes n = 1 bis 60 wertlos**. „Die Summe einzeln wertloser Tests ist nicht wertlos“ gilt also nur bei q > 1/2 | `test_algorithm`, `TestBefunde::test_b8` |
| B9 | Nicht-Additivität | zwei bedingt unabhängige symmetrische Tests, Raster 51 × 51 = **2 601 Paare** (Zähleinheit Testpaar), Vorab 1/2: max(EVSI A, EVSI B) ≤ EVSI(A, B) ≤ EVPI in allen; gemeinsam **weniger als die Summe in 1 915 Paaren, mehr in 84, gleich in 602**, und diese 602 enthalten alle einen wertlosen Test. Von 1 936 Paaren mit zwei einzeln wertvollen Tests sind 1 915 sub-, 21 superadditiv (qa, qb zwischen 0,57 und 0,62) und 0 gleich; von 49 Paaren mit zwei einzeln wertlosen Tests sind 21 gemeinsam wertvoll. Beispiele (gemeinsam gegen Summe, GE): (0,7; 0,7) 24,2 gegen 40; (0,8; 0,8) 38,2 gegen 70; (0,9; 0,7) 50 gegen 70; (0,55; 0,55) 2,45 gegen 0; (0,55; 0,56) 3,19 gegen 0; (0,6; 0,6) 9,8 gegen 10; (0,57; 0,57) 5,40 gegen 1,00 | `test_oracle` (exakt), `TestBefunde::test_b9` |
| B10 | Die Falle „Korrelation × EVPI“ | Korrelation = Phi-Koeffizient von Zustand und Ergebnis (bei Vorab 1/2 und symmetrisch 2q − 1; eine Entsprechung, nicht Bickels ρ im Normalmodell). Vorab 1/2, 51 Rasterpunkte: größter Fehler von Korrelation × EVPI +8,60 GE bei q = 0,57 (13,2 % des EVPI von 65 GE), von Korrelation² × EVPI −11,63 GE bei q = 0,79 (17,9 %), von q × EVPI +36,55 GE bei q = 0,57 (56,2 %); Punkte mit Fehler ≤ 5 % des EVPI: 20 / 15 / 4 von 51. **Korrelation × EVPI liegt an keinem der 51 Punkte unter dem wahren Wert**, ab q = 17/30 um genau 20 (1 − q) darüber (44 Punkte, exakt); Korrelation² × EVPI liegt an 43 von 44 Punkten nicht über dem wahren Wert (an 42 streng darunter, bei q = 1,00 gleich; Ausnahme q = 0,57); bei q = 0,8: wahr 35, 39, 23,4, 52 GE | `test_algorithm`, `TestBefunde::test_b10` |
| B11 | Andere Vorab-Werte der Falle | 6 Vorab-Werte × 51 = 306 Rasterpunkte: Korrelation × EVPI nirgends unter dem wahren Wert (306 von 306, Toleranz 10⁻⁹ GE); „wahrer Wert zwischen den Regeln“ nur für q > q0(p) an 32 von 36 (p = 0,3), 47 von 47 (0,4), 49 von 49 (0,45), 43 von 44 (0,5), 27 von 34 (0,6) und **1 von 17 (0,8)**; größter Fehler 13,2 / 4,5 / 2,4 / 8,6 / 16,3 / 15,4 GE. **Negativbefund:** nahe der Schwelle (p = 0,45, EVPI 71,5 GE) liegt die Korrelationsregel höchstens 2,4 GE (3,3 %) daneben; die Falle ist keine Warnung vor einer immer groben Regel, sondern davor, sie als **Wert** statt als obere Schranke zu lesen, und das nur in diesen 306 Punkten (kein Beweis) | `TestBefunde::test_b11` |
| B12 | Weitere Tests kaufen? | Vorab 1/2, q = 0,7, n ≤ 60, Kosten c = 0,1 bis 20,0 je Test in Schritten von 0,1 (200 Werte, exakt): die Regel „kaufe weiter, solange der nächste Zuwachs größer als c ist“ weicht vom Optimum in **60 von 200** Werten ab (alle bei c ≤ 6,2; ab 6,3 stimmen beide überein, bei c = 6,3 ist es ein Gleichstand zwischen einem und drei Tests), und **immer nach unten** (sie kauft zu wenig, nie zu viel). Beispiele (c; Regel; Optimum; Nettoverlust in GE): 1,0; 11; 13; 0,377 — 2,0; 5; 9; 1,641 — 3,0; 3; 5; 1,938 — **5,0; 1; 3; 2,600 (Nettowert 15 gegen 17,6)** — 6,0; 1; 3; 0,600; größter Nettoverlust 4,2 GE bei c = 4,2 (Regel 1, Optimum 3). Andere Trefferquoten (Vorab 1/2): Abweichungen in 27 (q = 0,6), 71 (0,8) und 50 (0,9) von je 200 Werten, alle nach unten, größter Nettoverlust 1,27 / 8,51 / 7,20 GE (bei c = 1,4 / 3,2 / 1,8); nur diese vier Trefferquoten gemessen | `TestBefunde::test_b12` |
| B13 | Zufallsprobleme | Zähleinheit Zufallsproblem, Zufalls-Seed 11001, 200 000 Probleme (Aufbau siehe Methodik): EVPI ≥ EVSI ≥ 0 in allen (0 Verletzungen, Toleranz 10⁻⁹ GE); **100 675 (50,3 %) sind dominiert** (eine Alternative ist in beiden Zuständen mindestens so gut wie die andere, daher Wert 0 für jeden Test); an den 99 325 übrigen ist **EVSI = 0 in 66,3 %** (Test mit zufälligen se und sp), bei einem Test mit q = 0,8 in 48,6 %, Median EVSI/EVPI 0, Mittel 0,112 (beide über diese 99 325 nicht dominierten Probleme, Test mit zufälligen se und sp); die Bedingung „nach beiden Ergebnissen dieselbe Wahl“ gilt genau dann, wenn EVSI = 0 (0 Ausnahmen). Über fünf Zufalls-Seeds (11001 bis 11005, je 200 000) liegt der Anteil EVSI = 0 zwischen 66,0 und 66,4 %, der Anteil dominiert zwischen 50,2 und 50,4 % und der Anteil bei q = 0,8 zwischen 48,1 und 48,6 %; bei 1 000 Problemen streut er stärker (62,8 bis 69,3 % für zwei Zufalls-Seeds) | `TestBefunde::test_b13` |
| B14 | Zweites Zahlenbeispiel (nur in den Tests) | A +100/−60, B sicher +5, Vorab 9/20: EV(A) = 12, EVPI = 143/4, EVSI (se 9/10, sp 4/5) = 973/40, q0 = 171/314 | `test_algorithm::test_second_example_investment_hand_values` (exakt) |

### Befunde und Korrekturen gegenüber dem Plan

- **Alle Zahlen der früheren Messung am Kapazitätsfall haben sich im Bau reproduziert** (B1 bis B12 stimmen auf die genannten Stellen). Abweichend ist nur B13, weil die Demo einen anderen Zufallsstrom benutzt (SplitMix64 statt NumPy-Generator) und die Zufallsprobleme anders zugeordnet sind: früher 66,1 % EVSI = 0, 50,4 % dominiert, 48,6 % bei q = 0,8, Mittel 0,113; jetzt 66,3 %, 50,3 %, 48,6 %, 0,112. Die Größenordnung hält, die letzte Stelle gehört zur Zufallsfolge; darum stehen hier die Zahlen des Baus und ihre Streuung über fünf Zufalls-Seeds.
- **B7 und B10/B11 sind Begrenzungen der früheren Verallgemeinerungen**, nicht neue Befunde: „nicht konkav“ gilt in 15 von 16 Zellen, die Summe einzeln wertloser Tests ist nur bei q > 1/2 nicht wertlos, und die Korrelationsregel ist an 306 von 306 Punkten eine obere Schranke, aber nahe der Schwelle fast genau.
- **Korrektur einer Rundung:** EVSI(10) bei q = 0,55 ist 10,553 (exakt 10,5535), nicht 10,554 wie in der früheren Messung.
- **Nenner:** bei Erträgen GE und Anteil an EVPI (siehe Modell), nicht Prozent von EV0.
- **Oberfläche gegenüber dem Plan:** Experiment 2 und 3 zeigen die symmetrische und die asymmetrische Sicht gleichzeitig statt über einen Umschalter; Vorab, Sensitivität, Spezifität und Testkosten stehen als allgemeine Einstellungen in der Seitenleiste, die Regler der Experimente 4 und 5 im Hauptbereich; Experiment 4 und 5 haben je einen eigenen Regler für die Trefferquote. Die Nullmenge der Heatmap ist grau statt schraffiert (Plotly kann keine Schraffur über Heatmaps). Der Vorab-Regler hat die Schrittweite 0,001; 13/30 liegt nur über den Schnellstart exakt, am Regler ist es 0,433.
- **Nicht umgesetzt gegenüber dem Plan:** der Zufallsproblem-Lauf ist ein Expander in Experiment 2 mit Zufalls-Seed und 1 000 bis 50 000 Problemen (wie geplant); ein PDF-Export wurde nicht gebaut. Nichts wurde aus Platzgründen gestrichen: die Seite hat fünf Experimente, die Teilung in eine eigene Seite „Mehrere Tests“ war nicht nötig.

### Zahl → Test

Zahlen, die im Test **exakt** stehen (Brüche, kein Band): B1, B2 (EVSI(q) an 51 Punkten), B3 (q0 an 19 Vorab-Werten), B4 (Nullmenge und Bedingung an 10 201 Punkten), B6 (Bruchwerte, Entscheidungsgrenze), B8, B9 (Zählung 1 915 / 84 / 602 in Brüchen), B14. **Als Band** (Zählungen und Anteile aus numpy, Zufallsströmen und Gleitkomma): die Zählungen aus der numpy-Rechnung (Toleranz 10⁻⁹ GE; sie stimmen im Test exakt mit der Rechnung in Brüchen überein), die Anteile der Zufallsprobleme (± 1 Prozentpunkt, Zählungen ± 2), die Zahlen B5 mit der Bisektion (10⁻⁹), B10 bis B12 als Zählungen. **Messung dieser README, nicht im Test:** die Rechenzeiten (Methodik) und die Streuung bei 1 000 Zufallsproblemen (B13, letzter Satz).

## Quellen

Nur so zitiert, wie gelesen (Stufen der Quellenberichte: W gelesen, A Zusammenfassung, S Suchtreffer); Gleichungen und Befunde sinngemäß, in eigenen Worten.

- **Raiffa und Schlaifer (1961)**, *Applied Statistical Decision Theory* (W, Kopie ohne geklärte Weitergaberechte): Rollback als „averaging out and folding back“ (Abschnitt 1.2.1), Wert der Information, Gleichungen 4-29 bis 4-36, die Ordnung EVSI ≤ EVPI und die Voraussetzung additiver (linearer) Nutzen (Abschnitt 4.2). Die Beispielzahlen der Tabelle 1.1 wurden nicht gelesen.
- **Howard und Matheson (2005)**, „Influence Diagrams“, *Decision Analysis* 2 (3), 127–143 (W): „Baum umdrehen“ (Abschnitte 6 und 7), Wert der Hellsicht als Obergrenze des Wertes realer Information (Abschnitt 15.1), Beispiel mit dem Befund, dass sich Informationswerte nicht addieren (1,1 und 0,4 gegen 1,38 Mio. $ für beide). **Diese Zahlen sind nicht nachgerechnet** (die Eingaben stehen nur in Tabellen und Abbildungen); der superadditive Fall in B9 ist eigene Nachrechnung in diesem Modell.
- **Bickel (2008)**, „The Relationship Between Perfect and Imperfect Information in a Two-Action Risk-Sensitive Problem“, *Decision Analysis* 5 (3), 116–128 (W, Zusammenfassung und Einleitung): die Rechnungen „Wert = ρ × EVPI“ und „Wert = ρ² × EVPI“ sind nicht allgemein richtig und gelten nur unter bestimmten Bedingungen; dort ist ρ die Korrelation in einem Normalmodell, hier der Phi-Koeffizient binärer Größen. Samson u. a. (1989) mit „sub-, super- oder additiv“ nur über Bickel (W der Zitatstelle).
- **Kunst u. a.**, arXiv 1910.03368 (W, Zusammenfassung und Einleitung): ENBS als Differenz aus EVSI und Studienkosten; EVSI bei stetigen Parametern als Schätzproblem (Regression, Importance Sampling, Gaußsche Näherung, Moment-Matching), das hier nicht vorkommt.
- **Offen (Literaturwerte, kein Test):** Zahlen aus Raiffa/Schlaifer (Tabelle 1.1) und aus Howard/Matheson (Abschnitt 15) sind nicht nachgerechnet; die Gleichungen werden in der Form geprüft, in der sie oben stehen. Howard (1966) liegt nur als Suchtreffer vor und wird nicht zitiert.

## Ehrliche Grenzen

- **Ein Fall:** zwei Zustände, zwei Alternativen, Erwartungswert in GE. Drei Zustände und Tests mit mehr als zwei Ergebnissen sind nicht gerechnet; der Rollback ist derselbe, die Zählungen dieser README gelten dort nicht.
- **Ausgedachte Zahlen:** die Auszahlungen, das Vorab und die Trefferquoten sind Eingaben eines Modells. Ob echte Tests solche Trefferquoten haben und ob bei einem echten Test die Wiederholungen bedingt unabhängig sind (B6 bis B9, B12), ist hier nicht gezeigt.
- **Mehrere Tests nur symmetrisch und bedingt unabhängig;** abhängige Tests (gemeinsame Fehlerursache) sind nicht gerechnet; sie bringen vermutlich weniger als unabhängige (nicht gemessen). Die Form von EVSI(n) ist nur für den Kapazitätsfall, bis n = 60 (B6) beziehungsweise n = 40 (B7) und für die 16 Zellen gezählt; der Regler der App erlaubt andere Werte, dort ist sie nicht belegt (die App weist darauf hin).
- **Deutung nicht gemessen:** dass bei geraden n die Zelle k = n/2 die kleinste Zelle ist, in der G gewählt wird, ist rechnerisch (bei Vorab 1/2 und q = 0,7: G genau bei k ≥ n/2; bei q = 0,55 gilt das nicht), dass sie den kleinen Zuwachs bei geraden n **verursacht**, ist eine Deutung, die nicht getrennt gemessen wurde. Warum die Zelle (Vorab 0,6; q = 0,8) konkav bleibt, ist nicht untersucht.
- **Die Korrelationsregel** ist nur für den Phi-Koeffizienten binärer Tests gerechnet, nicht für Bickels Normalmodell; „obere Schranke“ ist eine Zählung an 306 Rasterpunkten, kein Beweis.
- **Kosten:** nur Kosten je Test; Fixkosten und die Wahl des Abbruchzeitpunkts nach den Ergebnissen (sequentielle Tests) sind nicht gerechnet. Die Aussage „die Abbruchregel kauft zu wenig“ ist für vier Trefferquoten (0,6 / 0,7 / 0,8 / 0,9) bei Vorab 1/2 gezählt, in keiner der 4 × 200 Zellen kauft sie zu viel.
- **Zufallsprobleme:** die Anteile in B13 gehören zu dieser Verteilung der Probleme und zu diesem Zufallsstrom; eine andere Verteilung gibt andere Anteile.
- **Plattform:** Zählungen aus numpy und aus Zufallsströmen sind als Band geprüft; unter Windows (Python 3.14, NumPy 2.5.3) gemessen. Der Linux-Lauf (Docker) ist grün (siehe Verifikation).
- **Nicht mit echtem Mausklick getestet:** die Oberfläche ist mit dem Test-Framework von Streamlit (`AppTest`) geprüft, nicht jeder Regler im Browser.

## Verifikation

- **Kern:** Handwerte in Brüchen (Kapazitätsfall und zweites Zahlenbeispiel), alle Tests der Methodik gegen die Hand (`tests/test_algorithm.py`).
- **Orakel** (`tests/test_oracle_testwert_evsi.py`, unabhängige Rechenwege): EVSI gegen (a) die Aufzählung aller Entscheidungsregeln, (b) den Baum mit Bayes-Posterior, (c) EVPI minus Verlust mit Test, alle exakt auf 441 Rasterpunkten (se, sp in Schritten von 1/20) und für drei weitere Vorab-Werte und das zweite Zahlenbeispiel; (d) ein lineares Programm über randomisierte Regeln (`scipy.optimize.linprog`, HiGHS, Toleranz 10⁻⁹); Mehrfachtests gegen die Aufzählung aller 2ⁿ Folgen (n ≤ 10); q0(p), EVSI(q) und q\*(c) gegen geschlossene Formeln und `scipy.optimize.brentq`; die numpy-Raster (10 201 Punkte, 2 601 Paare) gegen die exakte Rechnung in Brüchen; die angezeigten Kennzahlen von Experiment 1 gegen den Baum.
- **Eigenschaften** (`tests/test_eigenschaften_testwert_evsi.py`, je 40 Zufallsinstanzen, exakt): EVPI ≥ EVSI ≥ 0; vollkommener Test gibt EVPI, Test ohne Aussage 0; Umbenennen der Alternativen und der Ergebnisse, Tausch der Zustände; Skalieren und Verschieben der Auszahlungen; zusätzliches Rauschen im Test senkt EVSI nie (Blackwell); EVSI(A, B) zwischen dem Größeren der beiden und EVPI; EVSI nicht fallend in der Testzahl; Mittel der Posterior-Werte gleich dem Vorab.
- **Randfälle** (`tests/test_edge_cases.py`): Vorab 0 und 1, Gleichstand an der Schwelle, Tests ohne Aussage, Ergebnis mit Wahrscheinlichkeit 0, dominierte Alternative, n = 0 und 1, Reglerränder aller Experimente.
- **Zufallsproblem-Gegenprobe** (`tests/test_random_problems.py`) und **Zufallsgenerator** (`tests/test_rng.py`, Folge gleich der Klasse, bekannte Werte).
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede Zahl im Text an handgesetzten Werten; **Claims** (`tests/test_claims.py`): jede Zahl dieser README, je Schnellstart die erwartete Verdikt-Zeile (der Schnellstart „Wertloser Test (0,55)“ meldet „wertlos“, „Zwei einzeln wertlose Tests (0,55)“ meldet „einzeln 0,00 und 0,00 GE, gemeinsam 2,45 GE“).
- **Oberfläche:** Presets, Layout (Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Änderung eines Reglers hebt die Markierung auf), AppTest-Rauchtest, Footer, Laufzeit-Importe.
- **Testzahl:** 151 Tests (`pytest --collect-only -q tests`).
- **Linux-Lauf (Docker, `demo_linux_check.py`, 2026-11-17):** grün, alle 151 Tests bestanden (Zählungen aus numpy und Zufallsströmen werden als Band geprüft).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: fünf Experimente, Kernfragen, Grenzen, Formeln |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur für die Orakel-Tests) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): Streamlit, NumPy, pandas, Plotly |
| `tev_algorithm.py` | Kern: EV0, EVPI, EVSI, Rollback, Schwellen, Mehrfachtests, numpy-Raster |
| `tev_constants.py` | Reglergrenzen, Raster, Farben, Schnellstart-Szenarien |
| `tev_evaluation.py` | Analysen und Verdikte der fünf Experimente, Zahlenformate |
| `tev_layout.py` | Fester Kopf (Stylesheet) |
| `tev_presets.py` | Permalink (Begrenzen), Schnellstart in der Seitenleiste, aktives Preset hervorgehoben, Zufalls-Seed-Knopf |
| `tev_rng.py` | SplitMix64 (Kopie der geprüften Vorlage) mit vektorisierter Folge |
| `tev_scenario.py` | Zahlenbeispiel (ausgedacht) und Zufallsprobleme |
| `tev_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tests/` | Handwerte, Orakel, Eigenschaften, Randfälle, Zufallsprobleme, Zufallsgenerator, Verdikte, Claims, Presets, Layout, AppTest-Rauchtest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Risikoaversion:** der Wert eines Tests beim Exponentialnutzen (dort kann EVSI exakt 0 werden, obwohl er für den Erwartungswert positiv ist) steht in [`information-risikoaversion-demo`](https://github.com/sebastian-hanisch/information-risikoaversion-demo).
- **Unbekannte Wahrscheinlichkeiten** (Schätzfehler in Vorab und Trefferquote): [`kipp-schwelle-demo`](https://github.com/sebastian-hanisch/kipp-schwelle-demo).
- **Stetige Parameter** (EVSI als Schätzproblem): `geschachtelte-erwartungen-demo` (Monte-Carlo-Demos der Website, ohne Link: lokal nicht vorhanden, der GitHub-Name ist nicht geprüft). Verfahren zur Berechnung von EVSI aus einer Sensitivitätsanalyse (Regression, Moment-Matching) sind nicht nachgebaut.
- **Abhängige und sequentielle Tests**, Fixkosten, mehr als zwei Zustände oder Ergebnisse, die Wahl zwischen verschiedenartigen Tests.
- **Bickels Normalmodell** mit Exponentialnutzen; die Korrelationsregel ist nur für binäre Tests gerechnet.
- **PDF-Export** der Ergebnisse.
- Der Wert von Schätzung, Vorschau oder Prognose in anderen Gegenständen (etwa Prognose-Wert bei der Kaiplanung) ist nicht Gegenstand dieser Demo.
- Die Verfahrenswahl zwischen Baum, Programmierung und Szenario-LP („welche Information kauft man“ als Entscheidungsknoten) ist nicht Gegenstand dieser Demo; [`entscheiden-lernen-wahl-demo`](https://github.com/sebastian-hanisch/entscheiden-lernen-wahl-demo) und [`regret-statt-fehler-demo`](https://github.com/sebastian-hanisch/regret-statt-fehler-demo) nennen den Wert der Information nur als Brücke zu [`newsvendor-evpi-demo`](https://github.com/sebastian-hanisch/newsvendor-evpi-demo).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Entscheidungsanalyse: Information, Risiko, Grenzen](https://sebastianhanisch.net/konzepte-entscheidungsanalyse.html).
