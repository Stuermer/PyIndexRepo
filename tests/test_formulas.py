from pathlib import Path

import pytest

from refractiveindex.dispersion_formulas import n_air, relative_to_absolute
from refractiveindex.main import RefractiveIndexLibrary


def test_n_air():
    # comparison numbers from wolframalpha
    assert n_air(0.5) == pytest.approx(1.00027377)
    assert n_air(0.78, 25.0, 0.1) == pytest.approx(1.000261856)


def test_n_various_glasses():
    print(Path.cwd().parent.joinpath('database/library.yml'))
    db = RefractiveIndexLibrary(path_to_library=Path.cwd().parent.joinpath('database/library.yml'), auto_upgrade=False)
    bk7 = db.get_material('glass', 'SCHOTT-BK', 'N-BK7')
    assert bk7.get_n(0.5876) == pytest.approx(1.5168, 1E-5)
    assert bk7.get_k(0.5876) == pytest.approx(9.7525e-9)


def test_n_at_temperature():
    # @20 deg
    n_bk7_ref = 1.517220
    db = RefractiveIndexLibrary(path_to_library=Path.cwd().parent.joinpath('database/library.yml'), auto_upgrade=False)
    bk7 = db.get_material('glass', 'SCHOTT-BK', 'N-BK7')
    assert relative_to_absolute(bk7.get_n_at_temperature(0.5876, 20), 0.5876) == n_bk7_ref