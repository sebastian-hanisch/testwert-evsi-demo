"""Presets (Schnellstart): Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Reglerschlüssel; jedes Szenario rechnet."""

import pytest

import tev_constants as C
import tev_evaluation as ev
import tev_presets as P

STANDARD = "Kapazitätsfall, Test 0,8 (von Hand)"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings_and_is_first():
    assert next(iter(C.PRESETS)) == STANDARD
    assert ev.Settings(**C.PRESETS[STANDARD]) == ev.Settings()
    for key, state_key in P.PRESET_KEYS.items():
        assert C.PRESETS[STANDARD][key] == P.SETTING_SPECS[state_key].default


def test_setting_specs_cover_every_preset_key_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_no_two_presets_are_identical():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(values) == len(set(values))


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_computes_all_experiments(name):
    s = ev.Settings(**C.PRESETS[name])
    ev.verdict_test(ev.analyse_test(s))
    ev.verdict_zero(ev.analyse_zero(s.p))
    ev.verdict_cost(ev.analyse_cost(s.p, s.c))
    ev.verdict_multi(ev.analyse_multi(s.p, s.q4, s.n, s.c))
    ev.verdict_trap(ev.analyse_trap(s.p, s.q5, s.qa, s.qb))


def test_threshold_preset_is_exactly_13_over_30():
    assert C.PRESETS["Vorab an der Schwelle (13/30)"]["p"] == 13 / 30


def test_preset_texts_do_not_use_internal_words():
    internal = ("St" + "ück", "Li" + "nie", "Vor" + "messung", "Pha" + "se 0")             # Wörter der eigenen Planung gehören nicht in sichtbare Texte
    for text in list(C.PRESETS) + list(C.PRESET_HELP.values()):
        for word in internal:
            assert word not in text
