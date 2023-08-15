from __future__ import annotations

import logging
import pickle
import time
import warnings
import zipfile
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import requests as requests
import yaml
from scipy.interpolate import interp1d
from yaml.scanner import ScannerError

from refractiveindex import dispersion_formulas

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RefractiveIndex")

TemperatureRange = namedtuple("TemperatureRange", ["min", "max"])


@dataclass
class ThermalDispersion:
    """Thermal Dispersion
    
    Deals with thermal dispersion of material. For a given formula_type and coefficients, delta_n_abs points to a
    function that is called with the arguments (n_ref, wavelength, coefficients*), where n_ref is the
    refractive index at reference temperature, wavelength the wavelength(s) value(s) and coefficients the array of the
    thermal dispersion coefficients.
    For speed reasons, the function delta_n_abs is outsourced to the dispersion_formulas.py file and speed up by numba.
    
    Attributes:
        formula_type: name of the formula. Supported are 'Schott formula'
        coefficients: array with thermal dispersion coefficients (lengths depends on formula type)
        delta_n_abs: function called with the arguments (n_ref, wavelength, coefficients*)
    
    """

    formula_type: str | None = None
    coefficients: np.ndarray | None = None
    delta_n_abs: callable = field(init=False)

    def __post_init__(self):
        if self.formula_type == "Schott formula":
            self.delta_n_abs = getattr(
                dispersion_formulas, "delta_absolute_temperature"
            )
        else:
            logger.warning("Thermal Dispersion formula not implemented yet")


@dataclass
class ThermalExpansion:
    """Thermal expansion
    Deals with thermal expansion of the material.
    Attributes:
        temperature_range: temperature range where thermal expansion coefficient is considered valid
        coefficient: thermal expansion coefficient [1/K]
    """

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
        if "thermal_dispersion" in specs_dict:
            for td_dict in specs_dict["thermal_dispersion"]:
                td = ThermalDispersion(
                    formula_type=td_dict["type"],
                    coefficients=np.array(
                        [float(val) for val in td_dict["coefficients"].split()],
                        dtype=float,
                    )
                    if td_dict.get("coefficients")
                    else None,
                )
                thermal_dispersion_list.append(td)
        if len(thermal_dispersion_list) > 1:
            warnings.warn(
                "There are multiple values given for the thermal dispersion. This is not supported at the moment."
            )

        # Parse TemperatureRange objects
        temperature_range_list = []
        if "thermal_expansion" in specs_dict:
            for tr_dict in specs_dict["thermal_expansion"]:
                if isinstance(tr_dict, dict):
                    if tr_dict.get("temperature_range"):
                        tr = TemperatureRange(
                            tr_dict["temperature_range"][0],
                            tr_dict["temperature_range"][1],
                        )
                    else:
                        tr = None
                else:
                    tr = None
                temperature_range_list.append(tr)
        temperature = specs_dict.get("temperature")
        if temperature is not None:
            temperature = float(temperature.split()[0])
        # Create Specs dataclass instance
        specs = Specs(
            n_is_absolute=specs_dict.get("n_is_absolute"),
            wavelength_is_vacuum=specs_dict.get("wavelength_is_vacuum"),
            temperature=temperature,
            thermal_dispersion=thermal_dispersion_list[0]
            if thermal_dispersion_list
            else None,
            nd=specs_dict.get("nd"),
            Vd=specs_dict.get("Vd"),
            glass_code=specs_dict.get("glass_code"),
            glass_status=specs_dict.get("glass_status"),
            density=float(specs_dict.get("density").replace(" g/cm<sup>3</sup>", ""))
            if specs_dict.get("density")
            else None,
            thermal_expansion=[
                ThermalExpansion(
                    temperature_range=tr,
                    coefficient=te_dict["coefficient"]
                    if isinstance(te_dict, dict)
                    else None,
                )
                for tr, te_dict in zip(
                    temperature_range_list, specs_dict["thermal_expansion"]
                )
            ]
            if specs_dict.get("thermal_expansion")
            else None,
            climatic_resistance=specs_dict.get("climatic_resistance"),
            stain_resistance=specs_dict.get("stain_resistance"),
            acid_resistance=specs_dict.get("acid_resistance"),
            alkali_resistance=specs_dict.get("alkali_resistance"),
            phosphate_resistance=specs_dict.get("phosphate_resistance"),
        )

        return specs

    def get_coefficient_of_thermal_expansion(self, temperature: float) -> float:
        """Returns the coefficient of thermal expansion for a given temperature."""
        if self.thermal_expansion is not None:
            # sort by total temperature range and return coefficient for smallest range
            self.thermal_expansion.sort(
                key=lambda exp: exp.temperature_range.max - exp.temperature_range.min
            )
            for expansion in self.thermal_expansion:
                if (
                    expansion.temperature_range.min
                    <= temperature
                    <= expansion.temperature_range.max
                ):
                    return expansion.coefficient
            # if temperature is outside any temperature range, return the value for the closest temperature range
            self.thermal_expansion.sort(
                key=lambda exp, t: min(
                    abs(t - exp.temperature_range.max),
                    abs(t - exp.temperature_range.min),
                )
            )

            warnings.warn(
                "Temperature is outside any temperature range, returning closest value as coefficient"
            )
            return self.thermal_expansion[0].coefficient
        else:
            warnings.warn("No thermal expansion data. Returning 0.0")
            return 0.0


@dataclass
class Material:
    n: TabulatedIndexData | FormulaIndexData | None = field(default=None)
    k: TabulatedIndexData | FormulaIndexData | None = field(default=None)
    specs: Specs | None = field(default=None)
    yaml_data: YAMLLibraryData = field(default=None)

    def get_n(self, wavelength):
        if self.n is None:
            print("WARNING: no n data, returning 0")
            return np.zeros_like(wavelength)
        return self.n.get_n_or_k(wavelength)

    def get_k(self, wavelength):
        if self.k is None:
            print("WARNING: no k data, returning 0")
            return np.zeros_like(wavelength)
        return self.k.get_n_or_k(wavelength)

    def get_nk(self, wavelength):
        return self.get_n(wavelength), self.get_k(wavelength)

    def get_n_at_temperature(
        self, wavelength: float | np.ndarray, temperature: float, P: float = 0.10133
    ) -> float | np.ndarray:
        assert self.specs.thermal_expansion is not None, (
            "There is no thermal dispersion formula available " "for this material"
        )
        if self.specs.wavelength_is_vacuum:
            n_abs = self.get_n(wavelength)
            return n_abs + self.specs.thermal_dispersion.delta_n_abs(
                n_abs,
                wavelength,
                temperature - self.specs.temperature,
                self.specs.thermal_dispersion.coefficients,
            )
        else:
            rel_wavelength = (
                wavelength
                * dispersion_formulas.n_air(wavelength, temperature, P)
                / dispersion_formulas.n_air(wavelength)
            )
            n_rel = self.get_n(rel_wavelength)
            n_abs = dispersion_formulas.relative_to_absolute(
                n_rel, rel_wavelength, self.specs.temperature, 0.10133
            )
            n_abs += self.specs.thermal_dispersion.delta_n_abs(
                n_abs,
                rel_wavelength,
                temperature - self.specs.temperature,
                *self.specs.thermal_dispersion.coefficients,
            )
            return dispersion_formulas.absolute_to_relative(
                n_abs, rel_wavelength, temperature, P
            )

    def __str__(self):
        return self.yaml_data.name

    def __repr__(self):
        return self.yaml_data.name


@dataclass
class TabulatedIndexData:
    wl: np.ndarray | list[float]
    n_or_k: np.ndarray | list[float]
    ip: callable = field(init=False)
    use_interpolation: bool = field(default=True)
    bounds_error: bool = field(default=True)

    def __post_init__(self):
        if self.use_interpolation:
            self.ip = interp1d(
                np.atleast_1d(self.wl),
                np.atleast_1d(self.n_or_k),
                bounds_error=self.bounds_error,
            )

    def get_n_or_k(self, wavelength):
        return self.ip(wavelength)


@dataclass
class FormulaIndexData:
    formula: callable
    coefficients: np.array
    min_wl: float = field(default=-np.inf)
    max_wl: float = field(default=np.inf)

    def get_n_or_k(self, wavelength):
        if isinstance(wavelength, float) or isinstance(wavelength, np.ndarray):
            return self.formula(wavelength, self.coefficients)
        elif isinstance(wavelength, list):
            return self.formula(np.array(wavelength), self.coefficients)
        else:
            raise ValueError(f"The datatype {type(wavelength)} is not supported.")


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
    lib_path: Path = field(default="")

    def __str__(self):
        return (
            f"{self.lib_shelf:10}, "
            f"{self.lib_book:15}, "
            f"{self.lib_page}, "
            f"{self.name}, "
            f"{self.lib_data}, "
            f"{self.lib_path}"
        )


@dataclass
class RefractiveIndexLibrary:
    path_to_library: Path = field(
        default=Path(__file__)
        .absolute()
        .parent.parent.joinpath("database/catalog-nk.yml")
    )
    auto_upgrade: bool = field(default=False)
    force_upgrade: bool = field(default=False)
    materials_yaml: list[YAMLLibraryData] = field(default_factory=list, init=False)
    materials: dict[str, dict[str, dict[str, Material]]] = field(
        default_factory=dict, init=False
    )
    materials_list: list[Material] = field(default_factory=list, init=False)
    github_sha: str = field(default="", init=False)

    def _is_library_outdated(self) -> bool:
        """Checks if local library is outdated"""
        if self.path_to_library.parent.joinpath(".local_sha").exists():
            # get local commit hash
            with open(self.path_to_library.parent.joinpath(".local_sha"), "r") as file:
                local_sha = file.readline()
            # get current commit hash on GitHub
            try:
                self.github_sha = requests.get(
                    "https://api.github.com/repos/polyanskiy/refractiveindex.info-database/commits/master"
                ).json()["sha"]
            except KeyError:
                logger.warning(
                    "Couldn't get SHA commit hash on GitHub. Database can not be updated."
                )
                self.github_sha = ""
                return False
            return not (self.github_sha == local_sha)
        else:
            logger.info("No local library exists.")
            return True

    def _download_latest_commit(self) -> bool:
        """Download latest library from GitHub."""
        if self._is_library_outdated() or self.force_upgrade:
            logger.info("New Library available... Downloading...")
            zip_url = f"https://api.github.com/repos/polyanskiy/refractiveindex.info-database/zipball"
            response = requests.get(zip_url)

            with open(self.path_to_library.with_suffix(".zip"), "wb") as file:
                file.write(response.content)
            with zipfile.ZipFile(self.path_to_library.with_suffix(".zip"), "r") as file:
                file_list = file.namelist()
                subfolder_files = [
                    file
                    for file in file_list
                    if file.startswith(f"{file_list[0]}database/data-nk")
                    and file.endswith(".yml")
                ]
                subfolder_files.append(f"{file_list[0]}database/catalog-nk.yml")
                for fn in subfolder_files:
                    print(fn)
                    # create a new Path object for the file to extract
                    extract_path = self.path_to_library.parent / Path(
                        "/".join(Path(fn).parts[2:])
                    )
                    extract_path.parent.mkdir(parents=True, exist_ok=True)
                    # open the file in the zipfile and write it to disk
                    with file.open(fn) as zf, extract_path.open("wb") as of:
                        of.write(zf.read())

            with open(self.path_to_library.parent.joinpath(".local_sha"), "w") as file:
                file.write(self.github_sha)
            return True
        else:
            return False

    def _load_from_yaml(self):
        logger.info("load from yaml")
        with open(self.path_to_library) as f:
            yaml_data = yaml.safe_load(f)

        for s in yaml_data:
            for book in s["content"]:
                if "BOOK" in book:
                    for page in book["content"]:
                        if "PAGE" in page:
                            # fill yaml list
                            self.materials_yaml.append(
                                YAMLLibraryData(
                                    name=page["name"],
                                    lib_page=page["PAGE"],
                                    lib_book=book["BOOK"],
                                    lib_shelf=s["SHELF"],
                                    lib_data=page["data"],
                                    lib_path=self.path_to_library.parent.joinpath(
                                        "data-nk"
                                    ).joinpath(page["data"]),
                                )
                            )
                            print(
                                s["SHELF"],
                                book["BOOK"],
                                page["PAGE"],
                                self.path_to_library.parent.joinpath(
                                    "data-nk"
                                ).joinpath(page["data"]),
                            )
                            # fill materials dict
                            if s["SHELF"] in self.materials:
                                if book["BOOK"] in self.materials[s["SHELF"]]:
                                    self.materials[s["SHELF"]][book["BOOK"]].update(
                                        {
                                            page["PAGE"]: yaml_to_material(
                                                self.path_to_library.parent.joinpath(
                                                    "data-nk"
                                                ).joinpath(page["data"]),
                                                s["SHELF"],
                                                book["BOOK"],
                                                page["PAGE"],
                                                page["name"],
                                            )
                                        }
                                    )
                                else:
                                    self.materials[s["SHELF"]][book["BOOK"]] = {
                                        page["PAGE"]: yaml_to_material(
                                            self.path_to_library.parent.joinpath(
                                                "data-nk"
                                            ).joinpath(page["data"]),
                                            s["SHELF"],
                                            book["BOOK"],
                                            page["PAGE"],
                                            page["name"],
                                        )
                                    }
                            else:
                                self.materials[s["SHELF"]] = {
                                    book["BOOK"]: {
                                        page["PAGE"]: yaml_to_material(
                                            self.path_to_library.parent.joinpath(
                                                "data-nk"
                                            ).joinpath(page["data"]),
                                            s["SHELF"],
                                            book["BOOK"],
                                            page["PAGE"],
                                            page["name"],
                                        )
                                    }
                                }

                            self.materials_list.append(
                                self.materials[s["SHELF"]][book["BOOK"]][page["PAGE"]]
                            )

        with open(self.path_to_library.with_suffix(".pickle"), "wb") as f:
            pickle.dump(self.materials_yaml, f, pickle.HIGHEST_PROTOCOL)
        with open(self.path_to_library.with_suffix(".pickle2"), "wb") as f:
            pickle.dump(self.materials, f, pickle.HIGHEST_PROTOCOL)

    def _load_from_pickle(self):
        logger.info("load from pickle")
        t1 = time.time()
        with open(self.path_to_library.with_suffix(".pickle"), "rb") as f:
            self.materials_yaml = pickle.load(f)
        with open(self.path_to_library.with_suffix(".pickle2"), "rb") as f:
            self.materials = pickle.load(f)

        for sd in self.materials.values():
            for bd in sd.values():
                for mat in bd.values():
                    self.materials_list.append(mat)
        logger.info(f"... done after{t1-time.time()}s")

    def __post_init__(self):
        upgraded = False
        # create database folder if it doesn't exist
        if not self.path_to_library.parent.is_dir():
            self.path_to_library.parent.mkdir()
        if self.auto_upgrade or self.force_upgrade:
            upgraded = self._download_latest_commit()
        if self.path_to_library.exists():
            if self.path_to_library.with_suffix(".pickle").exists() and not upgraded:
                self._load_from_pickle()
            else:
                self._load_from_yaml()
        else:
            warnings.warn(
                "Path to library does not exist! Please check path or activate auto_upgrade to download."
            )

    def search_material_by_page_name(
        self, page_name: str, exact_match: bool = False
    ) -> Material | list[Material] | None:
        """Search Material by name

        Search a Material by page name as given at refractiveindex.info.
        Sometimes, the name is not unique, so the function returns either a single Material or a list of Materials
        or None if it doesn't find a match.

        Args:
            page_name:
            exact_match:

        Returns:
            Material or list of Materials matching the Name

        Examples:
            >>> db = RefractiveIndexLibrary()
            >>> bk7 = db.search_material_by_page_name('N-BK7')[0]  # returns a list of different BK7 glasses
            >>> print(bk7.get_n(0.5875618))
            1.5168000345005885
        """
        materials = []
        if exact_match:
            for m in self.materials_list:
                if page_name == m.yaml_data.name:
                    materials.append(m)
        else:
            for m in self.materials_list:
                if page_name in m.yaml_data.name:
                    materials.append(m)
        return (
            materials[0]
            if len(materials) == 1
            else materials
            if len(materials) > 1
            else None
        )

    def search_material_by_n(
        self,
        n: float,
        wl: float = 0.5875618,
        filter_shelf: str | None = None,
        filter_book: str | None = None,
    ) -> list[Material]:
        """Search Material by refractive index

        Look for a material with a specific refractive index at a certain wavelength.
        In return, you get a sorted list of materials with index [0] being the closest to input n.

        Args:
            n: refractive index
            wl: wavelength
            filter_shelf: if given, only materials containing this string in their shelf name are considered
            filter_book: if given, only materials containing this string in their book name are considered

        Examples:
            >>> db = RefractiveIndexLibrary()
            >>> # get 3 closest OHARA glasses with n=1.5 at 0.55microns:
            >>> materials = db.search_material_by_n(1.5, wl=0.55, filter_book="ohara")[:3]
            >>> print(materials[0].yaml_data.name, materials[0].get_n(0.55))
            BSL3 1.4999474387027893
            >>> print(materials[1].yaml_data.name, materials[1].get_n(0.55))
            S-FPL51Y 1.498313496038896
            >>> print(materials[2].yaml_data.name, materials[2].get_n(0.55))
            S-FPL51 1.498303051383454

        Returns:
            sorted list of materials matching search criteria

        """
        materials = []
        materials_n_distance = []
        for shelf_m, d in self.materials.items():
            if not (shelf_m == filter_shelf or filter_shelf is None):
                continue
            for book_name, book_m in d.items():
                if not (
                    filter_book.lower() in book_name.lower() or filter_book is None
                ):
                    continue
                for mat in book_m.values():
                    materials.append(mat)
                    try:
                        materials_n_distance.append(abs(mat.get_n(wl) - n))
                    except ValueError:
                        materials_n_distance.append(99)

        return [
            x
            for _, x in sorted(
                zip(materials_n_distance, materials), key=lambda pair: pair[0]
            )
        ]

    def get_material(self, shelf: str, book: str, page: str) -> Material:
        """Get Material by shelf, book, page name

        Select Material by specifying shelf, book and page as given on refractiveindex.info

        Args:
            shelf: shelf name
            book: book name
            page: page name

        Returns:
            Material object

        Examples:
            >>> db = RefractiveIndexLibrary()
            >>> bk7 = db.get_material("glass", "SCHOTT-BK", "N-BK7")
            >>> print(bk7.get_n(0.5875618))
            1.5168000345005885
        """
        return self.materials[shelf][book][page]


def yaml_to_material(
    filepath: str | Path, lib_shelf: str, lib_book: str, lib_page: str, lib_name: str
) -> Material:
    """Converts RefractiveIndex.info YAML to Material

    Reads a yaml file of the refractiveindex database and converts it to a Material object.

    Args:
        filepath: path to yaml file
        lib_shelf: RefractiveIndex.info shelf name
        lib_book: RefractiveIndex.info book name
        lib_page: RefractiveIndex.info page name
        lib_name: RefractiveIndex.info material name

    Returns:
        Material object
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    def fill_variables_from_data_dict(data):
        """Helper function to split data in yaml into Material attributes"""
        data_type = data["type"]
        _wl_min = _wl_max = _wl = _n = _k = _coefficients = _formula = None
        if data_type == "tabulated nk":
            _wl, _n, _k = np.loadtxt(data["data"].split("\n")).T
            _wl_min = np.min(_wl)
            _wl_max = np.max(_wl)
        elif data_type == "tabulated n":
            (
                _wl,
                _n,
            ) = np.loadtxt(data["data"].split("\n")).T
            _wl_min = np.min(_wl)
            _wl_max = np.max(_wl)
        elif data_type == "tabulated k":
            (
                _wl,
                _k,
            ) = np.loadtxt(data["data"].split("\n")).T
            _wl_min = np.min(_wl)
            _wl_max = np.max(_wl)
        elif "formula" in data_type:
            _wl_range = data.get("wavelength_range") or data.get("range")
            _wl_min, _wl_max = _wl_range if _wl_range is not None else None, None
            _coefficients = np.array([float(c) for c in data["coefficients"].split()])
            _formula = data["type"].split()[1]
        # else:
        #     logger.warning(f"Could not convert YAML data for datatype {data_type}")
        return _wl, _wl_min, _wl_max, _n, _k, _coefficients, _formula

    with open(filepath) as f:
        n_class = k_class = None
        try:
            d = yaml.safe_load(f)

            # fill core data from yaml file
            (
                wl_n,
                wl_min_n,
                wl_max_n,
                n,
                k,
                coefficients,
                formula,
            ) = fill_variables_from_data_dict(d["DATA"][0])
            if formula:
                n_class = FormulaIndexData(
                    getattr(dispersion_formulas, f"formula_{formula}"),
                    coefficients,
                    wl_min_n,
                    wl_max_n,
                )
            elif n is not None:
                n_class = TabulatedIndexData(wl_n, n, True, True)

            (
                wl_k,
                wl_min_k,
                wl_max_k,
                _,
                k,
                coefficients_k,
                formula_k,
            ) = fill_variables_from_data_dict(next(iter(d["DATA"][1:2]), {"type": ""}))
            if formula_k:
                k_class = FormulaIndexData(
                    getattr(dispersion_formulas, f"formula_{formula}"),
                    coefficients_k,
                    wl_min_k,
                    wl_max_k,
                )
            elif k is not None:
                k_class = TabulatedIndexData(wl_k, k, True, True)

            # fill additional spec data if present
            specs = d.get("SPECS", None)
            if specs is not None:
                specs = Specs.read_specs_from_yaml(specs)
        except ScannerError:
            print(f"data in {filepath} can not be read")
        except ValueError:
            print(f"data in {filepath} can not be read due to bad data")

    return Material(
        n_class,
        k_class,
        specs,
        YAMLLibraryData(
            lib_name, yaml.dump(d), lib_shelf, lib_book, lib_page, filepath
        ),
    )


def fit_tabulated(
    tid: TabulatedIndexData, formula: callable, coefficients: list | np.ndarray
) -> FormulaIndexData:
    from scipy.optimize import least_squares

    def fit_func(x, wl, n):
        return formula(wl, *x) - n

    res = least_squares(fit_func, coefficients, args=(tid.wl, tid.n_or_k))
    return FormulaIndexData(formula, res.x)


if __name__ == "__main__":
    db = RefractiveIndexLibrary()
    # for shelf, m in db.materials.items():
    #     for book, mm in m.items():
    #         for page, mmm in mm.items():
    #             try:
    #                 if "formula_1" in str(mmm.get_n):
    #                     print(mmm.get_n)
    #             except AttributeError:
    #                 pass
    materials = db.search_material_by_n(1.5, wl=0.55, filter_book="ohara")[:5]
    for m in materials:
        print(m.yaml_data.name, m.get_n(0.55))
    # print(materials)
    # print(materials[0].get_n(0.55))