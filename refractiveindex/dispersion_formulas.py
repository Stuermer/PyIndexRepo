import numpy as np
from numba import njit


@njit(cache=True)
def formula_1_helper(wavelength, c1, c2):
    return c1 * (wavelength ** 2) / (wavelength ** 2 - c2 ** 2)


@njit(cache=True)
def formula_1(wavelength, coefficients):
    nsq = 1. + coefficients[0]
    for i in range(1, len(coefficients), 2):
        nsq = nsq + formula_1_helper(wavelength, coefficients[i], coefficients[i + 1])
    return np.sqrt(nsq)


@njit(cache=True)
def formula_2_helper(wavelength, c1, c2):
    return c1 * (wavelength ** 2) / (wavelength ** 2 - c2)


@njit(cache=True)
def formula_2(wavelength, coefficients):
    nsq = 1. + coefficients[0]
    for i in range(1, len(coefficients), 2):
        nsq = nsq + formula_2_helper(wavelength, coefficients[i], coefficients[i + 1])
    return np.sqrt(nsq)


@njit(cache=True)
def formula_3457_helper(wavelength, c1, c2):
    return c1 * wavelength ** c2


@njit(cache=True)
def formula_3(wavelength, coefficients):
    nsq = np.ones_like(wavelength) * coefficients[0]
    for i in range(1, len(coefficients), 2):
        nsq = nsq + formula_3457_helper(wavelength, coefficients[i], coefficients[i + 1])
    return np.sqrt(nsq)


@njit(cache=True)
def formula_4_helper1(wavelength, c1, c2, c3, c4):
    return c1 * wavelength ** c2 / (wavelength ** 2 - c3 ** c4)


@njit(cache=True)
def formula_4(wavelength, coefficients):
    nsq = coefficients[0]
    for i in range(1, min(8, len(coefficients)), 4):
        nsq = nsq + formula_4_helper1(wavelength, coefficients[i], coefficients[i + 1], coefficients[i + 2],
                                      coefficients[i + 3])
    if len(coefficients) > 9:
        for i in range(9, len(coefficients), 2):
            nsq = nsq + formula_3457_helper(wavelength, coefficients[i], coefficients[i + 1])
    return np.sqrt(nsq)


@njit(cache=True)
def formula_5(wavelength, coefficients):
    n = coefficients[0]
    for i in range(1, len(coefficients), 2):
        n = n + formula_3457_helper(wavelength, coefficients[i], coefficients[i + 1])
    return n


@njit(cache=True)
def formula_6_helper(wavelength, c1, c2):
    return c1 / (c2 - wavelength ** (-2))


@njit(cache=True)
def formula_6(wavelength, coefficients):
    n = 1 + coefficients[0]
    for i in range(1, len(coefficients), 2):
        n = n + formula_6_helper(wavelength, coefficients[i], coefficients[i + 1])
    return n


@njit(cache=True)
def formula_7_helper1(wavelength, c1, p):
    return c1 / (wavelength ** 2 - 0.028) ** p


@njit(cache=True)
def formula_7(wavelength, coefficients):
    n = coefficients[0]
    n = n + formula_7_helper1(wavelength, coefficients[1], 1)
    n = n + formula_7_helper1(wavelength, coefficients[2], 2)
    for i in range(3, len(coefficients)):
        n = n + formula_3457_helper(wavelength, coefficients[i], 2 * (i - 2))


@njit(cache=True)
def dn_absolute_temperature(n_abs_ref, wl, dT, D0, D1, D2, E0, E1, w_tk):
    """ Change of absolute refractive index with temperature

    Returns the change of the absolute refractive index for given wavelength and temperature

    Args:
        n_abs_ref: absolute refractive index at reference temperature
        wl: wavelength in vacuum [micron]
        dT: temperature difference between reference and actual temperature
        D0: constant depending on material
        D1: constant depending on material
        D2: constant depending on material
        E0: constant depending on material
        E1: constant depending on material
        w_tk: constant depending on material

    Returns:

    """
    return (n_abs_ref ** 2 - 1.) / (2. * n_abs_ref) * (
            D0 * dT + D1 * dT ** 2 + D2 * dT ** 3 + (E0 * dT + E1 * dT ** 2) / (wl ** 2 - w_tk ** 2))


# @njit(cache=True)
def n_air(wl, T=20.0, P=0.101325):
    n_air_ref = 1. + 1E-8 * (
            6432.8 + (2949810. * wl ** 2) / (146. * wl ** 2 - 1.) + (25540. * wl ** 2) / (41. * wl ** 2 - 1.))
    return 1. + (n_air_ref - 1.) / (1. + 3.4785E-3 * (T - 15)) * (P / 0.101325)


# @njit(cache=True)
def dn_dt_air(wl, T, P):
    return -0.00367 * (n_air(wl, T, P) - 1.) / (1. + 0.00367 * T)


# @njit(cache=True)
def absolute_to_relative(n_abs, wl, T=20.0, P=0.101325):
    """ Converts absolute refractive index to relative

    Args:
        n_abs: absolute refractive index
        wl: wavelength [micron]
        T: Temperature [Deg C]
        P: Pressure [MPa]

    Returns:

    """
    return n_abs / n_air(wl, T, P)


# @njit(cache=True)
def relative_to_absolute(n_rel, wl, T=20.0, P=0.101325):
    """ Converts relative refractive index to absolute

    Args:
        n_rel: relative refractive index
        wl: wavelength [micron]
        T: Temperature [Deg C]
        P: Pressure [MPa]

    Returns:

    """
    return n_rel * n_air(wl, T, P)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print(n_air(0.5))
    x = np.linspace(0.3, 1, 1000)
    plt.figure()
    plt.plot(x, n_air(x))
    plt.plot(x, n_air(x, 20, 0.101325))
    plt.plot(x, n_air(x, P=0.850))
    plt.show()
