"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Schnellstart-Szenarien (Presets).

Die Auszahlungen des Zahlenbeispiels stehen in `tev_scenario.py` (ausgedacht, in Geldeinheiten GE)."""

# Regler (alle Wahrscheinlichkeiten und Trefferquoten als Anteil 0 bis 1)
P_MIN, P_MAX, DEFAULT_P = 0.05, 0.95, 0.5            # Vorab P(hoch)
RATE_MIN, RATE_MAX = 0.0, 1.0                        # Sensitivität und Spezifität
DEFAULT_SE, DEFAULT_SP = 0.8, 0.8
C_MIN, C_MAX, DEFAULT_C = 0.0, 64.0, 0.0             # Testkosten in GE (EVPI des Zahlenbeispiels bei Vorab 1/2: 65 GE)
Q_MIN, Q_MAX, DEFAULT_Q4, DEFAULT_Q5, DEFAULT_QPAIR = 0.5, 1.0, 0.7, 0.8, 0.7   # Trefferquote q = Sensitivität = Spezifität (symmetrischer Test)
N_MIN, N_MAX, DEFAULT_N = 1, 60, 3                   # Zahl gleichartiger Tests
RZ_SEED_MAX, DEFAULT_RZ_SEED = 999999, 11001         # Zufalls-Seed der Zufallsprobleme
RZ_M_MIN, RZ_M_MAX, DEFAULT_RZ_M = 1000, 50000, 20000   # Zahl der Zufallsprobleme

# Raster der Zählungen (Zähleinheit: Rasterpunkt)
GRID_STEPS = 100                                     # se, sp = i/100 (101 mal 101 Punkte); q = k/100 (51 Punkte von 0,50 bis 1,00)
N_SERIES = 60                                        # Testzahlen 1 bis 60 werden gerechnet
TRAP_PRIORS = (0.3, 0.4, 0.45, 0.5, 0.6, 0.8)        # Vorab-Werte der Gegenprobe zur Faustregel „Korrelation mal EVPI“
CELL_PRIORS, CELL_QS = (0.4, 0.45, 0.5, 0.6), (0.6, 0.7, 0.8, 0.9)   # die 16 Zellen, in denen die Testzahl-Aussagen gezählt sind
SPEC_SENSITIVITIES = (0.6, 0.7, 0.8, 0.9, 0.95, 1.0)  # Zeilen der Tabelle der Mindest-Spezifität

COLOR_MAIN = "#2F6B65"
COLOR_ALT = "#4c78a8"
COLOR_WARN = "#c0504d"
COLOR_GREY = "rgba(150,150,150,0.55)"

# Schnellstart: jedes Szenario nennt alle Regler (Reihenfolge wie PRESET_KEYS in tev_presets.py).
_BASE = {"p": DEFAULT_P, "se": DEFAULT_SE, "sp": DEFAULT_SP, "c": DEFAULT_C, "q4": DEFAULT_Q4, "n": DEFAULT_N, "q5": DEFAULT_Q5, "qa": DEFAULT_QPAIR, "qb": DEFAULT_QPAIR,
         "rz_seed": DEFAULT_RZ_SEED, "rz_m": DEFAULT_RZ_M}


def _preset(**kw):
    out = dict(_BASE)
    out.update(kw)
    return out


PRESETS = {
    "Kapazitätsfall, Test 0,8 (von Hand)": _preset(),
    "Wertloser Test (0,55)": _preset(se=0.55, sp=0.55, q4=0.55, n=1, q5=0.55, qa=0.55, qb=0.55),
    "Knapp über der Schwelle (0,57)": _preset(se=0.57, sp=0.57, q4=0.57, n=1, q5=0.57, qa=0.57, qb=0.57),
    "Einseitiger Test (se 1, sp 1/2)": _preset(se=1.0, sp=0.5),
    "Vorab an der Schwelle (13/30)": _preset(p=13 / 30, se=0.52, sp=0.52, q4=0.52, n=1, q5=0.52, qa=0.52, qb=0.52),
    "Zwei Tests (0,7)": _preset(se=0.7, sp=0.7, q4=0.7, n=2, q5=0.7),
    "Testkosten 5 GE": _preset(c=5.0),
    "Zwei einzeln wertlose Tests (0,55)": _preset(se=0.55, sp=0.55, q4=0.55, n=2, q5=0.55, qa=0.55, qb=0.55),
}
PRESET_HELP = {
    "Kapazitätsfall, Test 0,8 (von Hand)": "Vorab 1/2, Test mit Trefferquote 0,8 (se = sp = 0,8), ohne Testkosten: das Beispiel, das sich von Hand nachrechnen lässt (EVSI 35 GE von höchstens 65 GE).",
    "Wertloser Test (0,55)": "Trefferquote 0,55: unter der Wertlos-Schwelle 17/30 = 0,567, die Wahl ändert sich nach keinem Ergebnis, EVSI = 0.",
    "Knapp über der Schwelle (0,57)": "Trefferquote 0,57: gerade über der Wertlos-Schwelle 17/30, der Test kippt die Wahl nach einem negativen Ergebnis, EVSI = 0,5 GE.",
    "Einseitiger Test (se 1, sp 1/2)": "Sensitivität 1, Spezifität 1/2: positiv sagt wenig, negativ sagt sicher „niedrig“; EVSI = 32,5 GE.",
    "Vorab an der Schwelle (13/30)": "Vorab P(hoch) = 13/30 = 0,4333, dem Punkt, an dem G und K gleich viel wert sind: schon ein Test mit Trefferquote 0,52 hat einen Wert über 0.",
    "Zwei Tests (0,7)": "Zwei gleichartige unabhängige Tests mit Trefferquote 0,7: gemeinsam 24,2 GE, nicht 2 mal 20.",
    "Testkosten 5 GE": "Jeder Test kostet 5 GE (Trefferquote 0,7): die Abbruchregel „kaufe weiter, solange der nächste Test mehr bringt als er kostet“ hört nach einem Test auf, das Optimum liegt bei drei.",
    "Zwei einzeln wertlose Tests (0,55)": "Zwei Tests mit Trefferquote 0,55, einzeln wertlos: zusammen 2,45 GE.",
}
