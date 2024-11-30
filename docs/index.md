# Welcome to PyIndexRepo

## Overview

PyIndexRepo is a Python package that provides a simple interface
to the refractive index database at [refractiveindex.info](https://refractiveindex.info/).

There are two main classes in the package:

- [RefractiveIndexLibrary][pyindexrepo.main.RefractiveIndexLibrary]: This class provides access to the refractive index
  database. It allows you to search for materials e.g. by name or refractive index.
- [Material][pyindexrepo.main.Material]: This class represents a material with its refractive index data. It provides
  methods to calculate the refractive index at any given wavelength based on the available data.
  It is possible to use custom data for the calculation or to use one of the implemented dispersion formulas.

For details on the available methods, see the [API documentation](api.md).

## Usage

Here is a simple example of how to use the package:

```python
from pyindexrepo import RefractiveIndexLibrary

# Create a database object
db = RefractiveIndexLibrary()

# Search for a material
bk7 = db.search_material_by_page_name('N-BK7')  # returns a list of different BK7 glasses
```

Now, `bk7` is a list of `Material` objects, we can access further information about the material:

```python
for glass in bk7:
    print(glass)
```

Let's say we are interested in the first glass in the list, we can access the refractive index data:

```python
print(bk7.get_n(0.5875618))  # get the refractive index at 587.5618 nm
bk7.get_n(np.linspace(0.4, 0.8, 100))  # get the refractive index for a range of wavelengths
```

The Material package can be used to generate custom materials:
Let's say we want to create a custom material with certain sellmeier coefficients:

```python
from pyindexrepo import Material, FormulaIndexData
from pyindexrepo.dispersion_formulas import
    formula_2  # the formula_2 is one particular implementation of the sellmeier formula. Check the documentation for more options. 

# Create a custom material with sellmeier coefficients (B1, B2, B3, C1, C2, C3) = (1.5, 0.1, 0.2, 0.3, 0.4, 0.5)
custom_material = Material(FormulaIndexData(formula_2, [1.5, 0.1, 0.2, 0.3, 0.4, 0.5]))
print(custom_material.get_n(0.5875618))  # get the refractive index at 587.5618 nm
```

Note: The refractive index is calculated using the formula provided. The formula should be a function that takes the
wavelength and the coefficients as arguments and returns the refractive index.
It can be a custom function or one of the implemented dispersion formulas.

These formulas are implemented in the `pyindexrepo.dispersion_formulas` module and are speed-optimized using the numba
library.

## Installation

Installing the package is done via pip:

```bash
pip install pyindexrepo
```
