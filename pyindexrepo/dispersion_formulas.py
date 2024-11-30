from __future__ import annotations

import sys

import numpy as np
from numba import float64, float32
from numba import njit, config

if "pytest" in sys.modules:
    config.DISABLE_JIT = True  # disable njit so numba function get covered by coverage


@njit(
    [float64(float64, float64, float64), float64[:](float64[:], float64, float64)],
    cache=True,
)
def formula_1_helper(wavelength, c1, c2):
    return c1 * (wavelength**2) / (wavelength**2 - c2**2)


@njit(
    [
        float64(float64, float64[:]),
        float64[:](float64[:], float64[:]),
    ],
    cache=True,
)
def formula_1(wavelength, coefficients):
    r"""Sellmeier (preferred) dispersion formula

    The formula has the general form:


    $$
    {n}^{2}-1={C}_{1}+\frac{{C}_{2}{\lambda }^{2}}{{\lambda }^{2}-{C}_{3}^{2}}+\frac{{C}_{4}{\lambda }^{2}}{{\lambda }^{2}-{C}_{5}^{2}}+\frac{{C}_{6}{\lambda }^{2}}{{\lambda }^{2}-{C}_{7}^{2}}+\frac{{C}_{8}{\lambda }^{2}}{{\lambda }^{2}-{C}_{9}^{2}}+\frac{{C}_{10}{\lambda }^{2}}{{\lambda }^{2}-{C}_{11}^{2}}+\frac{{C}_{12}{\lambda }^{2}}{{\lambda }^{2}-{C}_{13}^{2}}+\frac{{C}_{14}{\lambda }^{2}}{{\lambda }^{2}-{C}_{15}^{2}}+\frac{{C}_{16}{\lambda }^{2}}{{\lambda }^{2}-{C}_{17}^{2}}
    $$


    Args:
        wavelength: wavelength
        coefficients: list of coefficients
    """
    nsq = np.ones_like(wavelength) + coefficients[0]
    for i in range(1, len(coefficients), 2):
        nsq = nsq + formula_1_helper(wavelength, coefficients[i], coefficients[i + 1])
    return np.sqrt(nsq)


@njit(
    [float64(float64, float64, float64), float64[:](float64[:], float64, float64)],
    cache=True,
)
def formula_2_helper(wavelength, c1, c2):
    return c1 * (wavelength**2) / (wavelength**2 - c2)


@njit(
    [
        float64(float64, float64[:]),
        float64[:](float64[:], float64[:]),
    ],
    cache=True,
)
def formula_2(wavelength, coefficients):
    r"""Sellmeier-2 dispersion formula.

    The formula has the general form:

    $$
    {n}^{2}-1={C}_{1}+\frac{{C}_{2}{\lambda }^{2}}{{\lambda }^{2}-{C}_{3}}+\frac{{C}_{4}{\lambda }^{2}}{{\lambda }^{2}-{C}_{5}}+\frac{{C}_{6}{\lambda }^{2}}{{\lambda }^{2}-{C}_{7}}+\frac{{C}_{8}{\lambda }^{2}}{{\lambda }^{2}-{C}_{9}}+\frac{{C}_{10}{\lambda }^{2}}{{\lambda }^{2}-{C}_{11}}+\frac{{C}_{12}{\lambda }^{2}}{{\lambda }^{2}-{C}_{13}}+\frac{{C}_{14}{\lambda }^{2}}{{\lambda }^{2}-{C}_{15}}+\frac{{C}_{16}{\lambda }^{2}}{{\lambda }^{2}-{C}_{17}}
    $$

    Args:
        wavelength: wavelength
        coefficients: list of coefficients
    """
    nsq = np.ones_like(wavelength) + coefficients[0]
    for i in range(1, len(coefficients), 2):
        nsq = nsq + formula_2_helper(wavelength, coefficients[i], coefficients[i + 1])
    return np.sqrt(nsq)


@njit(
    [float64(float64, float64, float64), float64[:](float64[:], float64, float64)],
    cache=True,
)
def formula_3457_helper(wavelength, c1, c2):
    return c1 * wavelength**c2


@njit(
    [
        float64(float64, float64[:]),
        float64[:](float64[:], float64[:]),
    ],
    cache=True,
)
def formula_3(wavelength, coefficients):
    r"""Polynomial dispersion formula.

    The formula has the general form:

    $$
    {n}^{2}={C}_{1}+{C}_{2}{\lambda }^{{C}_{3}}+{C}_{4}{\lambda }^{{C}_{5}}+{C}_{6}{\lambda }^{{C}_{7}}+{C}_{8}{\lambda }^{{C}_{9}}+{C}_{10}{\lambda }^{{C}_{11}}+{C}_{12}{\lambda }^{{C}_{13}}+{C}_{14}{\lambda }^{{C}_{15}}+{C}_{16}{\lambda }^{{C}_{17}}
    $$

    Args:
        wavelength: wavelength
        coefficients: list of coefficients
    """
    nsq = np.ones_like(wavelength) * coefficients[0]
    for i in range(1, len(coefficients), 2):
        nsq = nsq + formula_3457_helper(
            wavelength, coefficients[i], coefficients[i + 1]
        )
    return np.sqrt(nsq)


@njit(cache=True)
def formula_4_helper1(wavelength, c1, c2, c3, c4):
    return c1 * wavelength**c2 / (wavelength**2 - c3**c4)


@njit(
    [
        float64(float64, float64[:]),
        float64[:](float64[:], float64[:]),
    ],
    cache=True,
)
def formula_4(wavelength, coefficients):
    r"""RefractiveIndex.INFO dispersion formula

    The formula has the general form:

    $$
    {n}^{2}={C}_{1}+\frac{{C}_{2}{\lambda }^{{C}_{3}}}{{\lambda }^{2}-{C}_{4}^{{C}_{5}}}+\frac{{C}_{6}{\lambda }^{{C}_{7}}}{{\lambda }^{2}-{C}_{8}^{{C}_{9}}}+{C}_{10}{\lambda }^{{C}_{11}}+{C}_{12}{\lambda }^{{C}_{13}}+{C}_{14}{\lambda }^{{C}_{15}}+{C}_{16}{\lambda }^{{C}_{17}}
    $$

    Args:
        wavelength: wavelength
        coefficients: list of coefficients
    """
    nsq = np.zeros_like(wavelength) + coefficients[0]
    for i in range(1, min(9, len(coefficients)), 4):
        nsq = nsq + formula_4_helper1(
            wavelength,
            coefficients[i],
            coefficients[i + 1],
            coefficients[i + 2],
            coefficients[i + 3],
        )
    if len(coefficients) > 8:
        for i in range(9, len(coefficients), 2):
            nsq = nsq + formula_3457_helper(
                wavelength, coefficients[i], coefficients[i + 1]
            )
    return np.sqrt(nsq)


@njit(
    [
        float64(float64, float64[:]),
        float64[:](float64[:], float64[:]),
    ],
    cache=True,
)
def formula_5(wavelength, coefficients):
    r"""Cauchy dispersion formula

    The formula has the general form:

    $$
    n={C}_{1}+{C}_{2}{\lambda }^{{C}_{3}}+{C}_{4}{\lambda }^{{C}_{5}}+{C}_{6}{\lambda }^{{C}_{7}}+{C}_{8}{\lambda }^{{C}_{9}}+{C}_{10}{\lambda }^{{C}_{11}}
    $$

    Args:
        wavelength: wavelength
        coefficients: list of coefficients
    """
    n = np.zeros_like(wavelength) + coefficients[0]
    for i in range(1, len(coefficients), 2):
        n = n + formula_3457_helper(wavelength, coefficients[i], coefficients[i + 1])
    return n


@njit(
    [
        float64(float64, float64, float64),
        float64[:](float64[:], float64, float64),
        float32(float32, float32, float32),
        float32[:](float32[:], float32, float32),
    ],
    cache=True,
)
def formula_6_helper(wavelength, c1, c2):
    return c1 / (c2 - wavelength ** (-2))


@njit(
    [
        float64(float64, float64[:]),
        float64[:](float64[:], float64[:]),
    ],
    cache=True,
)
def formula_6(wavelength, coefficients):
    r"""Gases dispersion formula

    The formula has the general form:

    $$
    n-1={C}_{1}+\frac{{C}_{2}}{{C}_{3}-{\lambda }^{-2}}+\frac{{C}_{4}}{{C}_{5}-{\lambda }^{-2}}+\frac{{C}_{6}}{{C}_{7}-{\lambda }^{-2}}+\frac{{C}_{8}}{{C}_{9}-{\lambda }^{-2}}+\frac{{C}_{10}}{{C}_{11}-{\lambda }^{-2}}
    $$

    Args:
        wavelength: wavelength
        coefficients: list of coefficients

    """
    n = np.ones_like(wavelength) + coefficients[0]
    for i in range(1, len(coefficients), 2):
        n = n + formula_6_helper(wavelength, coefficients[i], coefficients[i + 1])
    return n


@njit(
    [float64(float64, float64, float64), float64[:](float64[:], float64, float64)],
    cache=True,
)
def formula_7_helper1(wavelength, c1, p):
    return c1 / (wavelength**2 - 0.028) ** p


@njit(
    [
        float64(float64, float64[:]),
        float64[:](float64[:], float64[:]),
    ],
    cache=True,
)
def formula_7(wavelength, coefficients):
    r"""Herzberger dispersion formula

    The formula has the general form:

    $$
    n={C}_{1}+\frac{{C}_{2}}{{\lambda }^{2}-0.028}+{C}_{3}{\left(\frac{1}{{\lambda }^{2}-0.028}\right)}^{2}+{C}_{4}{\lambda }^{2}+{C}_{5}{\lambda }^{4}+{C}_{6}{\lambda }^{6}
    $$

    Args:
        wavelength: wavelength
        coefficients: list of coefficients
    """
    n = np.zeros_like(wavelength) + coefficients[0]
    n = n + formula_7_helper1(wavelength, coefficients[1], 1)
    n = n + formula_7_helper1(wavelength, coefficients[2], 2)
    for i in range(3, len(coefficients)):
        n = n + formula_3457_helper(wavelength, coefficients[i], float64(2 * (i - 2)))
    return n


@njit(
    [
        float64(float64, float64[:]),
        float64[:](float64[:], float64[:]),
    ],
    cache=True,
)
def formula_8(wavelength, coefficients):
    r"""Formula 8 dispersion formula from refractiveindex.info

    The formula has the general form:

    $$
    \frac{{n}^{2}-1}{{n}^{2}+2}={C}_{1}+\frac{{C}_{2}{\lambda }^{2}}{{\lambda }^{2}-{C}_{3}}+{C}_{4}{\lambda }^{2}
    $$

    """
    return np.sqrt(
        (
            wavelength**2
            * (
                2 * coefficients[0]
                + 2 * coefficients[1]
                + 2 * coefficients[3] * wavelength**2
                + 1
            )
            - coefficients[2]
            * (2 * coefficients[0] + 2 * coefficients[3] * wavelength**2 + 1)
        )
        / (
            coefficients[2] * (coefficients[0] + coefficients[3] * wavelength**2 - 1)
            - wavelength**2
            * (coefficients[0] + coefficients[1] + coefficients[3] * wavelength ** 2 - 1)
        )
    )


@njit(
    [
        float64(float64, float64[:]),
        float64[:](float64[:], float64[:]),
    ],
    cache=True,
)
def formula_9(wavelength, coefficients):
    r"""Formula 9 dispersion formula from refractiveindex.info

    The formula has the general form:

    $$
    {n}^{2}={C}_{1}+\frac{{C}_{2}}{{\lambda }^{2}-{C}_{3}}+\frac{{C}_{4}(\lambda -{C}_{5})}{{(\lambda -{C}_{5})}^{2}+{C}_{6}}
    $$

    Args:
        wavelength: wavelength
        coefficients: list of coefficients
    """
    return np.sqrt(
        coefficients[0]
        + coefficients[1] / (wavelength**2 - coefficients[2])
        + coefficients[3]
        * (wavelength - coefficients[4])
        / ((wavelength - coefficients[4]) ** 2 + coefficients[5])
    )


@njit(cache=True)
def dn_absolute_temperature(
    n_abs_ref: float | np.ndarray,
    wl: float | np.ndarray,
    dT: float,
    D0: float,
    D1: float,
    D2: float,
    E0: float,
    E1: float,
    w_tk: float,
):
    """dn/dT of absolute refractive index at certain Temperature

    Returns the temperature coefficient of the absolute refractive index for given wavelength and temperature
    Formula (2) of [Schott TIE-19](https://www.schott.com/en-gb/products/optical-glass-p1000267/downloads)

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
def delta_absolute_temperature(
    n_abs_ref: float | np.ndarray,
    wl: float | np.ndarray,
    dT: float,
    D0: float,
    D1: float,
    D2: float,
    E0: float,
    E1: float,
    w_tk: float,
):
    """deltaT of absolute refractive index at certain Temperature

    Returns the temperature coefficient of the absolute refractive index for given wavelength and temperature
    Formula (3) of [Schott TIE-19](https://www.schott.com/en-gb/products/optical-glass-p1000267/downloads)

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

    Calculates the refractive index of air as described in
    [Schott TIE-19](https://www.schott.com/en-gb/products/optical-glass-p1000267/downloads) - Formula (8):

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

    Calculates the temperature dependence of the refractive index of air as given in
    [Schott TIE-19](https://www.schott.com/en-gb/products/optical-glass-p1000267/downloads) - Formula (10):

    Args:
        wl: wavelength
        T: air temperature [°Celsius]
        P: air pressure [MPa]

    Returns:
        dn/dT of air
    """
    return -0.00367 * (n_air(wl, T, P) - 1.0) / (1.0 + 0.00367 * T)


@njit(cache=True)
def absolute_to_relative(
    n_abs: float | np.ndarray,
    wl: float | np.ndarray,
    T: float = 20.0,
    P: float = 0.10133,
):
    """Converts absolute refractive index to relative

    Formula (5) of [Schott TIE-19](https://www.schott.com/en-gb/products/optical-glass-p1000267/downloads)

    Args:
        n_abs: absolute refractive index.
        wl: wavelength [micron]
        T: Temperature [°Celsius] of absolute data
        P: Pressure [MPa]

    Returns:

    """
    return n_abs / n_air(wl, T, P)


@njit(cache=True)
def relative_to_absolute(
    n_rel: float | np.ndarray,
    wl: float | np.ndarray,
    T: float = 20.0,
    P: float = 0.10133,
):
    """Converts relative refractive index to absolute

    Reverse of Formula (5) of [Schott TIE-19](https://www.schott.com/en-gb/products/optical-glass-p1000267/downloads)

    Args:
        n_rel: relative refractive index
        wl: wavelength [micron]
        T: Temperature [°Celsius]
        P: Pressure [MPa]

    Returns:

    """
    return n_rel * n_air(wl, T, P)


@njit(cache=True)
def n_absolute_with_given_dndt(
        n_rel: float | np.ndarray, wl: float | np.ndarray, dT: float, coefficient: float
):
    """Calculates absolute refractive index with given dn/dT

    Formula (6) of [Schott TIE-19](https://www.schott.com/en-gb/products/optical-glass-p1000267/downloads)

    Args:
        n_rel: relative refractive index
        wl: wavelength [micron]
        dT: temperature difference [°Celsius]
        coefficient: dn/dT

    Returns:

    """
    return n_rel + coefficient * dT
