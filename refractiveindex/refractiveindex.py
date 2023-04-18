from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests as requests
import yaml
from yaml.scanner import ScannerError
from scipy.interpolate import interp1d
import dispersion_formulas

@dataclass
class Material:
    n: TabulatedIndexData | FormulaIndexData | None = field(default=None)
    k: TabulatedIndexData | FormulaIndexData | None = field(default=None)

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
    path_to_library: Path = field(default=Path(__file__).absolute().parent.joinpath('database/library.yml'))
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
            # get current commit hash on github
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
            zip_url = f"https://github.com/polyanskiy/refractiveindex.info-database/archive/{self.github_sha}.zip"
            response = requests.get(zip_url)

            with open(self.path_to_library.with_suffix('.zip'), 'wb') as file:
                file.write(response.content)
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

    def search_pages(self, term, exact_match=False) -> Material | list[Material] | None:
        materials = []
        if exact_match:
            for m in self.materials:
                if term == m.name:
                    materials.append(yaml_to_material(self.path_to_library.parent.joinpath('data').joinpath(m.lib_data)))
        else:
            for m in self.materials:
                if term in m.name:
                    materials.append(yaml_to_material(self.path_to_library.parent.joinpath('data').joinpath(m.lib_data)))
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
            wl_n, wl_min_n, wl_max_n, n, k, coefficients, formula = fill_variables_from_data_dict(d["DATA"][0])
            if formula:
                n_class = FormulaIndexData(getattr(dispersion_formulas, f'formula_{formula}'), coefficients, wl_min_n, wl_max_n)
            elif n is not None:
                n_class = TabulatedIndexData(wl_n, n, True, True)

            wl_k, wl_min_k, wl_max_k, _, k, coefficients_k, formula_k = fill_variables_from_data_dict(
                next(iter(d["DATA"][1:2]), {"type": ""}))
            if formula_k:
                k_class = FormulaIndexData(getattr(dispersion_formulas, f'formula_{formula}'), coefficients_k, wl_min_k, wl_max_k)
            elif k is not None:
                k_class = TabulatedIndexData(wl_k, k, True, True)

        except ScannerError:
            print(f"data in {filepath} can not be read")
        except ValueError:
            print(f"data in {filepath} can not be read due to bad data")

    return Material(n_class, k_class)


if __name__ == "__main__":
    import time
    import pickle

    t1 = time.time()
    db = RefractiveIndexLibrary(auto_upgrade=False)
    bacd16 = db.search_pages('BACD16')
    print(bacd16.n.coefficients)
    print(bacd16)
    # Ag = db.get_material('glass', 'HOYA-BaCD', 'BACD16')
    # n = Ag.get_n(np.linspace(0.4, 0.9, 1000))
    # print(n)
    # t2 = time.time()
    # k = Ag.get_k(np.linspace(0.399, 0.9, 1000))
    # t3 = time.time()
    # print(t2 - t1)
    # print(t3-t2)
