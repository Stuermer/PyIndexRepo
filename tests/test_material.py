import numpy as np

from pyindexrepo import Material, TabulatedIndexData


def test_tabulated_n_material():
    m = Material(
        n=TabulatedIndexData(
            wl=[0.3, 0.5, 0.7],
            n_or_k=[1.5, 1.4, 1.3],
            interpolation_func="interp1d",
            bounds_error=True,
        )
    )
    assert 1.4 < m.get_n(0.4) < 1.5


def test_tabulated_k_material():
    m = Material(
        k=TabulatedIndexData(
            wl=[0.3, 0.5, 0.7],
            n_or_k=[1.5, 1.4, 1.3],
            interpolation_func="interp1d",
            bounds_error=True,
        )
    )
    assert 1.4 < m.get_k(0.4) < 1.5


def test_missing_data():
    m = Material()
    array_zero = np.array([0.0, 0.0, 0.0])
    assert np.array_equal(m.get_n([0.3, 0.5, 0.7]), array_zero)
    assert np.array_equal(m.get_k([0.3, 0.5, 0.7]), array_zero)
    n, k = m.get_nk([0.3, 0.5, 0.7])
    assert np.array_equal(n, array_zero)
    assert np.array_equal(k, array_zero)
