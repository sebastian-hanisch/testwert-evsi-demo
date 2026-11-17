"""Gemeinsame Fixtures: das Zahlenbeispiel in Brüchen, an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HALF = (F(1, 2), F(1, 2))
CAP = {"G": (F(200), F(-100)), "K": (F(30), F(30))}      # Kapazitätsfall (ausgedacht): G groß bauen, K klein bauen oder mieten
INV = {"A": (F(100), F(-60)), "B": (F(5), F(5))}         # zweites Zahlenbeispiel nur für die Tests (Handwerte 143/4, 973/40, 171/314)
P_STAR = F(13, 30)


@pytest.fixture
def cap():
    return CAP


@pytest.fixture
def half():
    return HALF
