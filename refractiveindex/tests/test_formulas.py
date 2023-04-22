import pytest

from refractiveindex.dispersion_formulas import n_air


def test_n_air():
    # comparison numbers from wolframalpha
    assert n_air(0.5) == pytest.approx(1.00027897)
    assert n_air(0.78, 23.0, 0.1) == pytest.approx(1.000263)
