from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def formula_1_helper(wavelength, c1, c2):
    return c1 * (wavelength**2) / (wavelength**2 - c2**2)


@njit(cache=True)
def formula_1(wavelength, coefficients):
    """Sellmeier (preferred) dispersion formula"""
    nsq = 1.0 + coefficients[0]
    for i in range(1, len(coefficients), 2):
        nsq = nsq + formula_1_helper(wavelength, coefficients[i], coefficients[i + 1])
    return np.sqrt(nsq)


@njit(cache=True)
def formula_2_helper(wavelength, c1, c2):
    return c1 * (wavelength**2) / (wavelength**2 - c2)


@njit(cache=True)
def formula_2(wavelength, coefficients):
    """Sellmeier-2 dispersion formula"""
    nsq = 1.0 + coefficients[0]
    for i in range(1, len(coefficients), 2):
        nsq = nsq + formula_2_helper(wavelength, coefficients[i], coefficients[i + 1])
    return np.sqrt(nsq)


@njit(cache=True)
def formula_3457_helper(wavelength, c1, c2):
    return c1 * wavelength**c2


@njit(cache=True)
def formula_3(wavelength, coefficients):
    """Polynomial dispersion formula"""
    nsq = np.ones_like(wavelength) * coefficients[0]
    for i in range(1, len(coefficients), 2):
        nsq = nsq + formula_3457_helper(
            wavelength, coefficients[i], coefficients[i + 1]
        )
    return np.sqrt(nsq)


@njit(cache=True)
def formula_4_helper1(wavelength, c1, c2, c3, c4):
    return c1 * wavelength**c2 / (wavelength**2 - c3**c4)


@njit(cache=True)
def formula_4(wavelength, coefficients):
    """RefractiveIndex.INFO dispersion formula"""
    nsq = coefficients[0]
    for i in range(1, min(8, len(coefficients)), 4):
        nsq = nsq + formula_4_helper1(
            wavelength,
            coefficients[i],
            coefficients[i + 1],
            coefficients[i + 2],
            coefficients[i + 3],
        )
    if len(coefficients) > 9:
        for i in range(9, len(coefficients), 2):
            nsq = nsq + formula_3457_helper(
                wavelength, coefficients[i], coefficients[i + 1]
            )
    return np.sqrt(nsq)


@njit(cache=True)
def formula_5(wavelength, coefficients):
    """Cauchy dispersion formula"""
    n = coefficients[0]
    for i in range(1, len(coefficients), 2):
        n = n + formula_3457_helper(wavelength, coefficients[i], coefficients[i + 1])
    return n


@njit(cache=True)
def formula_6_helper(wavelength, c1, c2):
    return c1 / (c2 - wavelength ** (-2))


@njit(cache=True)
def formula_6(wavelength, coefficients):
    """Gases dispersion formula"""
    n = 1 + coefficients[0]
    for i in range(1, len(coefficients), 2):
        n = n + formula_6_helper(wavelength, coefficients[i], coefficients[i + 1])
    return n


@njit(cache=True)
def formula_7_helper1(wavelength, c1, p):
    return c1 / (wavelength**2 - 0.028) ** p


@njit(cache=True)
def formula_7(wavelength, coefficients):
    """Herzberger dispersion formula"""
    n = coefficients[0]
    n = n + formula_7_helper1(wavelength, coefficients[1], 1)
    n = n + formula_7_helper1(wavelength, coefficients[2], 2)
    for i in range(3, len(coefficients)):
        n = n + formula_3457_helper(wavelength, coefficients[i], 2 * (i - 2))


@njit(cache=True)
def dn_absolute_temperature(n_abs_ref, wl, dT, D0, D1, D2, E0, E1, w_tk):
    """dn/dT of absolute refractive index at certain Temperature

    Returns the temperature coefficient of the absolute refractive index for given wavelength and temperature
    Formula (2) of Schott TIE-19

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
    return (
        (n_abs_ref**2 - 1.0)
        / (2.0 * n_abs_ref)
        * (
            D0
            + 2.0 * D1 * dT
            + 3 * D2 * dT**2
            + ((E0 + 2 * E1 * dT) / (wl**2 - w_tk**2))
        )
    )


@njit(cache=True)
def delta_absolute_temperature(n_abs_ref, wl, dT, D0, D1, D2, E0, E1, w_tk):
    """deltaT of absolute refractive index at certain Temperature

    Returns the temperature coefficient of the absolute refractive index for given wavelength and temperature
    Formula (3) of Schott TIE-19

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
    return (
        (n_abs_ref**2 - 1.0)
        / (2.0 * n_abs_ref)
        * (
            D0 * dT
            + D1 * dT**2
            + D2 * dT**3
            + (E0 * dT + E1 * dT**2) / (wl**2 - w_tk**2)
        )
    )


@njit(cache=True)
def n_air(wl: float | np.ndarray, T: float = 20.0, P: float = 0.10133):
    """Refractive index of air

    Calculates the refractive index of air as described in Schott TIE 19:
    Formula (8) of TIE 19
    https://www.schott.com/en-gb/products/optical-glass-p1000267/downloads

    Args:
        wl: wavelength
        T: air temperature [°Celsius]
        P: air pressure [MPa]

    Returns:
        refractive index of air
    """
    n_air_ref = 1.0 + 1e-8 * (
        6432.8
        + (2949810.0 * wl**2) / (146.0 * wl**2 - 1.0)
        + (25540.0 * wl**2) / (41.0 * wl**2 - 1.0)
    )
    return 1.0 + (n_air_ref - 1.0) / (1.0 + 3.4785e-3 * (T - 15.0)) * (P / 0.10133)


@njit(cache=True)
def dn_dt_air(wl: float | np.ndarray, T: float, P: float):
    """Temperature coefficient dn/dT of air

    Calculates the temperature dependence of the refractive index of air as given in Schott's TIE 19:
    Formula (10) of TIE-19
    https://www.schott.com/en-gb/products/optical-glass-p1000267/downloads

    Args:
        wl: wavelength
        T: air temperature [°Celsius]
        P: air pressure [MPa]

    Returns:
        dn/dT of air
    """
    return -0.00367 * (n_air(wl, T, P) - 1.0) / (1.0 + 0.00367 * T)


@njit(cache=True)
def absolute_to_relative(n_abs, wl, T=20.0, P=0.10133):
    """Converts absolute refractive index to relative

    Formula (5) of TIE-19

    Args:
        n_abs: absolute refractive index.
        wl: wavelength [micron]
        T: Temperature [°Celsius] of absolute data
        P: Pressure [MPa]

    Returns:

    """
    return n_abs / n_air(wl, T, P)


@njit(cache=True)
def relative_to_absolute(n_rel, wl, T=20.0, P=0.10133):
    """Converts relative refractive index to absolute

    Reverse of Formula (5) of TIE-19

    Args:
        n_rel: relative refractive index
        wl: wavelength [micron]
        T: Temperature [°Celsius]
        P: Pressure [MPa]

    Returns:

    """
    return n_rel * n_air(wl, T, P)
