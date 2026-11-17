"""Zufallsgenerator: die vektorisierte Folge ist gleich der Folge der Klasse; bekannte Werte; Gleichverteilte und Ganzzahlen liegen in ihren Bereichen."""

import numpy as np

import tev_rng as R


def test_array_stream_equals_class_stream():
    g = R.SplitMix64(12345)
    assert [g.next_u64() for _ in range(50)] == [int(v) for v in R.splitmix_array(12345, 50)]
    g = R.SplitMix64(7)
    assert [g.random() for _ in range(50)] == list(R.uniform_array(7, 50))
    g = R.SplitMix64(11001)
    assert [g.below(301) for _ in range(50)] == list(R.below_array(11001, 50, 301))


def test_known_splitmix_values_for_seed_zero():
    """Referenzfolge von SplitMix64 mit Zustand 0 (Vigna): 0xe220a8397b1dcdaf, 0x6e789e6aa1b965f4."""
    g = R.SplitMix64(0)
    assert g.next_u64() == 0xE220A8397B1DCDAF
    assert g.next_u64() == 0x6E789E6AA1B965F4
    assert int(R.splitmix_array(0, 2)[1]) == 0x6E789E6AA1B965F4


def test_ranges_and_rough_uniformity():
    u = R.uniform_array(3, 20000)
    assert u.min() >= 0.0 and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.01
    k = R.below_array(3, 20000, 301)
    assert k.min() >= 0 and k.max() <= 300 and abs(k.mean() - 150.0) < 3.0


def test_large_seed_wraps_modulo_2_to_the_64():
    assert list(R.splitmix_array(2**64 + 5, 3)) == list(R.splitmix_array(5, 3))
    assert R.SplitMix64(2**64 + 5).next_u64() == int(R.splitmix_array(5, 1)[0])
    assert R.splitmix_array(5, 0).size == 0 and np.all(R.splitmix_array(5, 4)[:2] == R.splitmix_array(5, 2))
