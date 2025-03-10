import numpy as np
import pytest

from pyindexrepo import RefractiveIndexLibrary
from pyindexrepo.dispersion_formulas import n_air


@pytest.fixture(scope="session")
def db():
    db = RefractiveIndexLibrary(force_upgrade=False, auto_upgrade=False)
    assert len(db.materials_yaml) > 0
    return db


def test_n_air():
    # comparison numbers from wolframalpha
    assert n_air(0.5) == pytest.approx(1.00027377)
    assert n_air(0.78, 25.0, 0.1) == pytest.approx(1.000261856)


def test_n_formula1(db):
    sap = db.get_material("3d", "crystals", "sapphire")
    assert sap.get_n(0.4926) == pytest.approx(1.7749069668851, 1e-5)


def test_n_formula2(db):
    bk7 = db.get_material("specs", "SCHOTT-optical", "N-BK7")
    assert bk7.get_n(0.5876) == pytest.approx(1.5168, 1e-5)
    assert bk7.get_k(0.5876) == pytest.approx(9.7525e-9)
    assert bk7.get_n(0.5460740) == pytest.approx(1.51872, 1e-5)


def test_n_formula3(db):
    bac4 = db.get_material("specs", "HOYA-optical", "BAC4")
    # test single float
    assert bac4.get_n(0.5792) == pytest.approx(1.5692758037963, 1e-5)
    # test array
    assert bac4.get_n(np.array((0.5792,))) == pytest.approx(1.5692758037963, 1e-5)


def test_n_formula4(db):
    nb = db.get_material("other", "Nb-RbTiOPO4", "Carvajal-γ")
    assert nb.get_n(0.9500) == pytest.approx(1.8811351002965, 1e-5)


def test_n_formula5(db):
    sio2 = db.get_material("3d", "liquids", "acetone")
    assert sio2.get_n(0.6108) == pytest.approx(
        1.358423145362, 1e-5
    )  # value via RefractiveIndex.info


def test_n_formula6(db):
    mat = db.get_material("other", "5PCH", "Wu-53.4C-e")
    assert mat.get_n(0.6640) == pytest.approx(1.5743755094386, 1e-5)


def test_n_formula7(db):
    mat = db.get_material("main", "Si", "Edwards")
    assert mat.get_n(8.078) == pytest.approx(3.4223271140184, 1e-5)


def test_n_formula8(db):
    mat = db.get_material("main", "TlCl", "Schroter")
    assert mat.get_n(0.5220) == pytest.approx(2.30210288728, 1e-5)


def test_n_formula9(db):
    mat = db.get_material("organic", "urea", "Rosker-e")
    assert mat.get_n(0.7256) == pytest.approx(1.5978342243166, 1e-5)


def test_n_at_temperature(db):
    # values from ZEMAX article
    # https://support.zemax.com/hc/en-us/articles/1500005576002-How-OpticStudio-calculates-refractive-index-at-arbitrary-temperatures-and-pressures
    n_bk7_ref = 1.51851533  # @20deg 1atm  @0.55014022mu
    n_bk7_30deg_2atm = 1.51814375  # @30deg 2 atm @0.55mu

    bk7 = db.get_material("specs", "SCHOTT-optical", "N-BK7")
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


def test_silver(db):
    silver = db.get_material("main", "Ag", "Ciesielski")
    assert silver.get_n(0.5) == pytest.approx(0.083000, 1e-5)
    assert silver.get_k(0.5) == pytest.approx(2.8180, 1e-5)

def test_different_datatypes(db):
    bk7 = db.get_material("specs", "SCHOTT-optical", "N-BK7")
    res = bk7.get_k(0.5)
    assert (
        isinstance(res, float)
        or isinstance(res, np.float64)
        or isinstance(res, np.float32)
        or isinstance(res, np.ndarray)
    )  # res can be a np.ndarray, because scipy.interpolate returns a np.ndarray

    assert isinstance(bk7.get_k([0.5, 0.6, 0.7]), np.ndarray)

    assert isinstance(bk7.get_k(np.linspace(0.5, 0.6, 100)), np.ndarray)
