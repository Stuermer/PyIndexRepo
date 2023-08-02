from __future__ import annotations

import pickle
import warnings
import zipfile
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import requests as requests
import yaml
from yaml.scanner import ScannerError
from scipy.interpolate import interp1d
from refractiveindex import dispersion_formulas

TemperatureRange = namedtuple('TemperatureRange', ['min', 'max'])


@dataclass
class ThermalDispersion:
    formula_type: str | None = None
    coefficients: np.ndarray | None = None
    delta_n_abs: callable = field(init=False)

    def __post_init__(self):
        if self.formula_type == 'Schott formula':
            self.delta_n_abs = getattr(dispersion_formulas, 'dn_absolute_temperature')


@dataclass
class ThermalExpansion:
    temperature_range: TemperatureRange
    coefficient: float


@dataclass
class Specs:
    n_is_absolute: bool | None = None
    wavelength_is_vacuum: bool | None = None
    temperature: float | None = None
    thermal_dispersion: ThermalDispersion | None = None
    nd: float | None = None
    Vd: float | None = None
    glass_code: float | None = None
    glass_status: str | None = None
    density: float | None = None
    thermal_expansion: List[ThermalExpansion] | None = None
    climatic_resistance: float | None = None
    stain_resistance: float | None = None
    acid_resistance: float | None = None
    alkali_resistance: float | None = None
    phosphate_resistance: float | None = None

    @staticmethod
    def read_specs_from_yaml(specs_dict):

        # Parse ThermalDispersion objects
        thermal_dispersion_list = []
        if 'thermal_dispersion' in specs_dict:
            for td_dict in specs_dict['thermal_dispersion']:
                td = ThermalDispersion(formula_type=td_dict['type'],
                                       coefficients=np.array([float(val) for val in td_dict['coefficients'].split()],
                                                             dtype=float))
                thermal_dispersion_list.append(td)
        if len(thermal_dispersion_list) > 1:
            warnings.warn(
                'There are multiple values given for the thermal dispersion. This is not supported at the moment.')

        # Parse TemperatureRange objects
        temperature_range_list = []
        if 'thermal_expansion' in specs_dict:
            for tr_dict in specs_dict['thermal_expansion']:
                tr = TemperatureRange(tr_dict['temperature_range'][0], tr_dict['temperature_range'][1])
                temperature_range_list.append(tr)
        temperature = specs_dict.get('temperature')
        if temperature is not None:
            temperature = float(temperature.split()[0])
        # Create Specs dataclass instance
        specs = Specs(
            n_is_absolute=specs_dict.get('n_is_absolute'),
            wavelength_is_vacuum=specs_dict.get('wavelength_is_vacuum'),
            temperature=temperature,
            thermal_dispersion=thermal_dispersion_list[0] if thermal_dispersion_list else None,
            nd=specs_dict.get('nd'),
            Vd=specs_dict.get('Vd'),
            glass_code=specs_dict.get('glass_code'),
            glass_status=specs_dict.get('glass_status'),
            density=float(specs_dict.get('density').replace(' g/cm<sup>3</sup>', '')),
            thermal_expansion=[ThermalExpansion(temperature_range=tr, coefficient=te_dict['coefficient']) for
                               tr, te_dict in zip(temperature_range_list, specs_dict['thermal_expansion'])],
            climatic_resistance=specs_dict.get('climatic_resistance'),
            stain_resistance=specs_dict.get('stain_resistance'),
            acid_resistance=specs_dict.get('acid_resistance'),
            alkali_resistance=specs_dict.get('alkali_resistance'),
            phosphate_resistance=specs_dict.get('phosphate_resistance')
        )

        return specs

    def get_coefficient_of_thermal_expansion(self, temperature: float) -> float:
        """Returns the coefficient of thermal expansion for a given temperature."""
        if self.thermal_expansion is not None:
            # sort by total temperature range and return coefficient for smallest range
            self.thermal_expansion.sort(key=lambda exp: exp.temperature_range.max - exp.temperature_range.min)
            for expansion in self.thermal_expansion:
                if expansion.temperature_range.min <= temperature <= expansion.temperature_range.max:
                    return expansion.coefficient
            # if temperature is outside any temperature range, return the value for the closest temperature range
            self.thermal_expansion.sort(
                key=lambda exp, t: min(abs(t - exp.temperature_range.max), abs(t - exp.temperature_range.min)))

            warnings.warn("Temperature is outside any temperature range, returning closest value as coefficient")
            return self.thermal_expansion[0].coefficient
        else:
            warnings.warn("No thermal expansion data. Returning 0.0")
            return 0.0


@dataclass
class Material:
    n: TabulatedIndexData | FormulaIndexData | None = field(default=None)
    k: TabulatedIndexData | FormulaIndexData | None = field(default=None)
    specs: Specs | None = field(default=None)

    def get_n(self, wavelength):
        if self.n is None:
            print('WARNING: no n data, returning 0')
            return np.zeros_like(wavelength)
        return self.n.get_n_or_k(wavelength)

    def get_k(self, wavelength):
        if self.k is None:
            print('WARNING: no k data, returning 0')
            return np.zeros_like(wavelength)
        return self.k.get_n_or_k(wavelength)

    def get_nk(self, wavelength):
        return self.get_n(wavelength), self.get_k(wavelength)

    def get_n_at_temperature(self, wavelength, temperature, P=0.101325):
        assert self.specs.thermal_expansion is not None, 'There is no thermal dispersion formula available ' \
                                                         'for this material'
        if self.specs.wavelength_is_vacuum:
            n_abs = self.get_n(wavelength)
            return n_abs + self.specs.thermal_dispersion.delta_n_abs(n_abs, wavelength,
                                                                     temperature - self.specs.temperature,
                                                                     self.specs.thermal_dispersion.coefficients)
        else:
            n_rel = self.get_n(wavelength)
            n_abs = dispersion_formulas.relative_to_absolute(n_rel,
                                                             wavelength,
                                                             self.specs.temperature,
                                                             0.101325)
            n_abs += self.specs.thermal_dispersion.delta_n_abs(n_abs,
                                                               wavelength,
                                                               temperature - self.specs.temperature,
                                                               *self.specs.thermal_dispersion.coefficients)
            return dispersion_formulas.absolute_to_relative(n_abs,
                                                            wavelength,
                                                            temperature,
                                                            P)

@dataclass
class TabulatedIndexData:
    wl: np.ndarray | list[float]
    n_or_k: np.ndarray | list[float]
    ip: callable = field(init=False)
    use_interpolation: bool = field(default=True)
    bounds_error: bool = field(default=True)

    def __post_init__(self):
        if self.use_interpolation:
            self.ip = interp1d(self.wl, self.n_or_k, bounds_error=self.bounds_error)

    def get_n_or_k(self, wavelength):
        return self.ip(wavelength)


@dataclass
class FormulaIndexData:
    formula: callable
    coefficients: np.array
    min_wl: float = field(default=-np.inf)
    max_wl: float = field(default=np.inf)

    def get_n_or_k(self, wavelength):
        return self.formula(wavelength, self.coefficients)


@dataclass
class YAMLRefractiveIndexData:
    data_type: str
    wavelength_range: str = field(default="")
    coefficients: str = field(default="")
    data: str = field(default="")


@dataclass
class YAMLMaterialData:
    n_data: YAMLRefractiveIndexData
    k_data: YAMLRefractiveIndexData
    comments: str = field(default="")
    references: str = field(default="")


@dataclass
class YAMLLibraryData:
    name: str
    lib_data: str
    lib_shelf: str = field(default="")
    lib_book: str = field(default="")
    lib_page: str = field(default="")

    def __str__(self):
        return f"{self.lib_shelf:10}, {self.lib_book:15}, {self.lib_page}, {self.name}, {self.lib_data}"


@dataclass
class RefractiveIndexLibrary:
    path_to_library: Path = field(default=Path(__file__).absolute().parent.parent.joinpath('database/library.yml'))
    auto_upgrade: bool = field(default=True)
    materials: list[YAMLLibraryData] = field(default_factory=list, init=False)
    github_sha: str = field(default='', init=False)

    def __str__(self):
        for m in self.materials:
            print(m)

    def _is_library_outdated(self) -> bool:
        if self.path_to_library.parent.joinpath('.local_sha').exists():
            # get local commit hash
            with open(self.path_to_library.parent.joinpath('.local_sha'), 'r') as file:
                local_sha = file.readline()
            # get current commit hash on GitHub
            self.github_sha = \
                requests.get(
                    'https://api.github.com/repos/polyanskiy/refractiveindex.info-database/commits/master').json()[
                    'sha']
            return not (self.github_sha == local_sha)
        else:
            return True

    def _download_latest_commit(self) -> bool:
        if self._is_library_outdated():
            print('Download new library version...')
            zip_url = f"https://api.github.com/repos/polyanskiy/refractiveindex.info-database/zipball"
            response = requests.get(zip_url)

            with open(self.path_to_library.with_suffix('.zip'), 'wb') as file:
                file.write(response.content)
            with zipfile.ZipFile(self.path_to_library.with_suffix('.zip'), 'r') as file:
                file_list = file.namelist()
                subfolder_files = [file for file in file_list if file.startswith(f'{file_list[0]}database/data')
                                   and file.endswith('.yml')]
                subfolder_files.append(f'{file_list[0]}database/library.yml')
                for fn in subfolder_files:
                    print(fn)
                    # create a new Path object for the file to extract
                    extract_path = self.path_to_library.parent / Path('/'.join(Path(fn).parts[2:]))
                    extract_path.parent.mkdir(parents=True, exist_ok=True)
                    # open the file in the zipfile and write it to disk
                    with file.open(fn) as zf, extract_path.open('wb') as of:
                        of.write(zf.read())

            with open(self.path_to_library.parent.joinpath('.local_sha'), 'w') as file:
                file.write(self.github_sha)
            return True
        else:
            return False

    def _load_from_yaml(self):
        print('load from yaml')
        with open(self.path_to_library) as f:
            d = yaml.safe_load(f)

        for s in d:
            for book in s["content"]:
                if "BOOK" in book:
                    for page in book["content"]:
                        if "PAGE" in page:
                            self.materials.append(
                                YAMLLibraryData(
                                    name=page["name"],
                                    lib_page=page["PAGE"],
                                    lib_book=book["BOOK"],
                                    lib_shelf=s["SHELF"],
                                    lib_data=page["data"],
                                )
                            )
        with open(self.path_to_library.with_suffix('.pickle'), 'wb') as f:
            pickle.dump(self.materials, f, pickle.HIGHEST_PROTOCOL)

    def _load_from_pickle(self):
        print('load from pickle')
        with open(self.path_to_library.with_suffix('.pickle'), 'rb') as f:
            self.materials = pickle.load(f)

    def __post_init__(self):
        upgraded = False
        if self.auto_upgrade:
            upgraded = self._download_latest_commit()
        if self.path_to_library.exists():
            if self.path_to_library.with_suffix('.pickle').exists() and not upgraded:
                self._load_from_pickle()
            else:
                self._load_from_yaml()
        else:
            warnings.warn('Path to library does not exist! Please check path or activate auto_upgrade to download.')

    def search_pages(self, term, exact_match=False) -> Material | list[Material] | None:
        materials = []
        if exact_match:
            for m in self.materials:
                if term == m.name:
                    materials.append(
                        yaml_to_material(self.path_to_library.parent.joinpath('data').joinpath(m.lib_data)))
        else:
            for m in self.materials:
                if term in m.name:
                    materials.append(
                        yaml_to_material(self.path_to_library.parent.joinpath('data').joinpath(m.lib_data)))
        return materials[0] if len(materials) == 1 else materials if len(materials) > 1 else None

    def get_material(self, shelf, book, page) -> Material | list[Material] | None:
        materials = []
        for m in self.materials:
            if page == m.lib_page:
                if book == m.lib_book:
                    if shelf == m.lib_shelf:
                        materials.append(
                            yaml_to_material(self.path_to_library.parent.joinpath('data').joinpath(m.lib_data)))
        return materials[0] if len(materials) == 1 else materials if len(materials) > 1 else None


def yaml_to_material(filepath: str | Path) -> Material:
    if isinstance(filepath, str):
        filepath = Path(filepath)

    def fill_variables_from_data_dict(data):
        data_type = data["type"]
        _wl_min = _wl_max = _wl = _n = _k = _coefficients = _formula = None
        if data_type == "tabulated nk":
            _wl, _n, _k = np.loadtxt(data["data"].split("\n")).T
            _wl_min = np.min(_wl)
            _wl_max = np.max(_wl)
        if data_type == "tabulated n":
            (
                _wl,
                _n,
            ) = np.loadtxt(data["data"].split("\n")).T
            _wl_min = np.min(_wl)
            _wl_max = np.max(_wl)
        if data_type == "tabulated k":
            (
                _wl,
                _k,
            ) = np.loadtxt(data["data"].split("\n")).T
            _wl_min = np.min(_wl)
            _wl_max = np.max(_wl)
        if "formula" in data_type:
            _wl_range = data.get("wavelength_range") or data.get("range")
            _wl_min, _wl_max = _wl_range if _wl_range is not None else None, None
            _coefficients = np.array([float(c) for c in data["coefficients"].split()])
            _formula = data["type"].split()[1]
        return _wl, _wl_min, _wl_max, _n, _k, _coefficients, _formula

    with open(filepath) as f:
        n_class = k_class = None
        try:
            d = yaml.safe_load(f)

            # fill core data from yaml file
            wl_n, wl_min_n, wl_max_n, n, k, coefficients, formula = fill_variables_from_data_dict(d["DATA"][0])
            if formula:
                n_class = FormulaIndexData(getattr(dispersion_formulas, f'formula_{formula}'), coefficients, wl_min_n,
                                           wl_max_n)
            elif n is not None:
                n_class = TabulatedIndexData(wl_n, n, True, True)

            wl_k, wl_min_k, wl_max_k, _, k, coefficients_k, formula_k = fill_variables_from_data_dict(
                next(iter(d["DATA"][1:2]), {"type": ""}))
            if formula_k:
                k_class = FormulaIndexData(getattr(dispersion_formulas, f'formula_{formula}'), coefficients_k, wl_min_k,
                                           wl_max_k)
            elif k is not None:
                k_class = TabulatedIndexData(wl_k, k, True, True)

            # fill additional spec data if present
            specs = d.get('SPECS', None)
            if specs is not None:
                specs = Specs.read_specs_from_yaml(specs)
        except ScannerError:
            print(f"data in {filepath} can not be read")
        except ValueError:
            print(f"data in {filepath} can not be read due to bad data")

    return Material(n_class, k_class, specs)

def fit_tabulated(tid: TabulatedIndexData, formula: callable, coefficients: list | np.ndarray) -> FormulaIndexData:
    from scipy.optimize import least_squares
    def fit_func(x, wl, n):
        return formula(wl, *x) - n
    res = least_squares(fit_func, coefficients, args=(tid.wl, tid.n_or_k))
    return FormulaIndexData(formula, res.x)


if __name__ == "__main__":
    import time

    t1 = time.time()
    db = RefractiveIndexLibrary(auto_upgrade=False)
    # bacd16 = db.search_pages('K7', True)
    bacd16 = db.get_material('glass', 'SCHOTT-BK', 'N-BK7')

    print(bacd16)
    print(bacd16.n.coefficients)
    print(bacd16.get_n(0.5875618))
    print(bacd16.get_n_at_temperature(0.5875618, 50))

    print(dispersion_formulas.relative_to_absolute(bacd16.get_n(0.5875618), 0.5875618, 50))
    print(dispersion_formulas.relative_to_absolute(bacd16.get_n_at_temperature(0.5875618, 50), 0.5875618, 50))

    # Ag = db.get_material('glass', 'HOYA-BaCD', 'BACD16')
    # n = Ag.get_n(np.linspace(0.4, 0.9, 1000))
    # print(n)
    # t2 = time.time()
    # k = Ag.get_k(np.linspace(0.399, 0.9, 1000))
    # t3 = time.time()
    # print(t2 - t1)
    # print(t3-t2)
