from pathlib import Path

from numba import config

config.DISABLE_JIT = True  # disable njit so numba function get covered by coverage
import pytest

from refractiveindex.dispersion_formulas import n_air
from refractiveindex.main import RefractiveIndexLibrary


def test_n_air():
    # comparison numbers from wolframalpha
    assert n_air(0.5) == pytest.approx(1.00027377)
    assert n_air(0.78, 25.0, 0.1) == pytest.approx(1.000261856)


def test_n_various_glasses():
    print(Path.cwd().parent.joinpath("database/catalog-nk.yml"))
    db = RefractiveIndexLibrary(
        path_to_library=Path.cwd().parent.joinpath("database/catalog-nk.yml"),
        auto_upgrade=False,
    )
    bk7 = db.get_material("glass", "SCHOTT-BK", "N-BK7")
    assert bk7.get_n(0.5876) == pytest.approx(1.5168, 1e-5)
    assert bk7.get_k(0.5876) == pytest.approx(9.7525e-9)
    assert bk7.get_n(0.5460740) == pytest.approx(1.51872, 1e-5)


def test_n_at_temperature():
    # values from ZEMAX article
    # https://support.zemax.com/hc/en-us/articles/1500005576002-How-OpticStudio-calculates-refractive-index-at-arbitrary-temperatures-and-pressures
    n_bk7_ref = 1.51851533  # @20deg 1atm  @0.55014022mu
    n_bk7_30deg_2atm = 1.51814375  # @30deg 2 atm @0.55mu

    db = RefractiveIndexLibrary(
        path_to_library=Path.cwd().parent.joinpath("database/catalog-nk.yml"),
        auto_upgrade=False,
    )
    bk7 = db.get_material("glass", "SCHOTT-BK", "N-BK7")
    assert bk7.get_n_at_temperature(0.55014022, 20) == pytest.approx(n_bk7_ref)
    assert bk7.get_n_at_temperature(0.55, 30, P=0.202650) == pytest.approx(
        n_bk7_30deg_2atm
    )
    assert bk7.get_n_at_temperature(0.55, 50) == pytest.approx(
        1.5186118999
    )  # value from ZEMAX
    assert bk7.get_n_at_temperature(0.55, 0.0, P=0.0) == pytest.approx(
        1.5189176695
    )  # value from ZEMAX
    assert bk7.get_n_at_temperature(0.8, 10.0, P=0.202650) == pytest.approx(
        1.5103242657
    )  # value from ZEMAX
    assert bk7.get_n(0.55) == pytest.approx(bk7.get_n_at_temperature(0.55, 20))
