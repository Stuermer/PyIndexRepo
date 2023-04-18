import numpy as np
from numba import njit


@njit(cache=True)
def formula_1_helper(wavelength, c1, c2):
    return c1 * (wavelength ** 2) / (wavelength ** 2 - c2 ** 2)


@njit(cache=True)
def formula_1(wavelength, coefficients):
    nsq = 1. + coefficients[0]
    for i in range(1, len(coefficients), 2):
        nsq += formula_1_helper(wavelength, coefficients[i], coefficients[i + 1])
    return np.sqrt(nsq)


@njit(cache=True)
def formula_2_helper(wavelength, c1, c2):
    return c1 * (wavelength ** 2) / (wavelength ** 2 - c2)


@njit(cache=True)
def formula_2(wavelength, coefficients):
    nsq = 1 + coefficients[0]
    for i in range(1, len(coefficients), 2):
        nsq += formula_2_helper(wavelength, coefficients[i], coefficients[i + 1])
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
        nsq += formula_4_helper1(wavelength, coefficients[i], coefficients[i + 1], coefficients[i + 2],
                                 coefficients[i + 3])
    if len(coefficients) > 9:
        for i in range(9, len(coefficients), 2):
            nsq += formula_3457_helper(wavelength, coefficients[i], coefficients[i + 1])
    return np.sqrt(nsq)


@njit(cache=True)
def formula_5(wavelength, coefficients):
    n = coefficients[0]
    for i in range(1, len(coefficients), 2):
        n += formula_3457_helper(wavelength, coefficients[i], coefficients[i + 1])
    return n


@njit(cache=True)
def formula_6_helper(wavelength, c1, c2):
    return c1 / (c2 - wavelength ** (-2))


@njit(cache=True)
def formula_6(wavelength, coefficients):
    n = 1 + coefficients[0]
    for i in range(1, len(coefficients), 2):
        n += formula_6_helper(wavelength, coefficients[i], coefficients[i + 1])
    return n


@njit(cache=True)
def formula_7_helper1(wavelength, c1, p):
    return c1 / (wavelength ** 2 - 0.028) ** p


@njit(cache=True)
def formula_7(wavelength, coefficients):
    n = coefficients[0]
    n += formula_7_helper1(wavelength, coefficients[1], 1)
    n += formula_7_helper1(wavelength, coefficients[2], 2)
    for i in range(3, len(coefficients)):
        n += formula_3457_helper(wavelength, coefficients[i], 2 * (i - 2))
