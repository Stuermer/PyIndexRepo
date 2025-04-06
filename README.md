[![PyPI - Version](https://img.shields.io/pypi/v/PyIndexRepo)](https://pypi.org/project/PyIndexRepo/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/PyIndexRepo)](https://pypi.org/project/PyIndexRepo/)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Stuermer/PyIndexRepo/tests.yaml?label=pytest)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Stuermer/PyIndexRepo/docs.yaml?label=mkdocs&link=https%3A%2F%2Fstuermer.github.io%2FPyIndexRepo%2F)](https://stuermer.github.io/PyIndexRepo/)
[![Coverage Status](https://coveralls.io/repos/github/Stuermer/PyIndexRepo/badge.svg?branch=master)](https://coveralls.io/github/Stuermer/PyIndexRepo?branch=master)


# PyIndexRepo

This package gives access to the refractive index data from [RefractiveIndex.info]().

The focus of this package is to provide a convenient interface to the data, and be
efficient in the calculation of (temperature-dependent) refractive indices.
Internally, the YAML data are converted to Python objects, for convenient access. Numba is used to speed up recursive calculations.


## Usage

#### Basics

There are multiple ways to access the data. The most basic way is to use the `RefractiveIndexLibrary` class:

```python
from pyindexrepo import RefractiveIndexLibrary

db = RefractiveIndexLibrary(auto_upgrade=True)
bk7 = db.search_material_by_page_name('N-BK7')[0]  # returns a list of different BK7 glasses
print(bk7.get_n(0.5875618))
```

When executed for the first time, the database from the
RefractiveIndex [Github Repo](https://github.com/polyanskiy/refractiveindex.info-database) will be downloaded and
converted to a python object. This process takes a few minutes. Consecutive calls will load the database object
from a local file (almost instantaneously).

Auto-upgrade of the library is supported, but switched off by default.

The search function will return a list of materials, as there are multiple materials with the same name.
You can also select the material by 'shelf', 'book' and 'page' as specified in the RefractiveIndex.info database.

```python
from pyindexrepo import RefractiveIndexLibrary

db = RefractiveIndexLibrary(auto_upgrade=True)
bk7 = db.get_material('specs', 'schott', 'N-BK7')
print(bk7.get_n(0.5875618))
```

For further information how to interact with the data, please refer to the [documentation](https://stuermer.github.io/PyIndexRepo/).
In particular, check the API of the `RefractiveIndexLibrary` class and the `Material` class.

#### Temperature data

When temperature data is available, the refractive index of a material can be queried at any temperature
within the valid temperature range:

```python
import numpy as np
from pyindexrepo import RefractiveIndexLibrary

db = RefractiveIndexLibrary(auto_upgrade=True)
bk7 = db.search_material_by_page_name('N-BK7')[0]  # returns a list of different BK7 glasses
wl = np.linspace(0.4, 0.7, 10000)
print(bk7.get_n_at_temperature(wl, temperature=30))
```

``` 
[1.53088657 1.53088257 1.53087857 ... 1.51309187 1.51309107 1.51309027]
```

## Installation

```bash
pip install pyindexrepo
```
