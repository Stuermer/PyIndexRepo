from __future__ import annotations

import logging
import os
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
from scipy.interpolate import interp1d
from yaml.scanner import ScannerError

from pyindexrepo import dispersion_formulas

# from ruamel.yaml import YAML

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
    coefficients: np.ndarray | float | None = None
    delta_n_abs: callable = field(init=False)

    def __post_init__(self):
        if self.formula_type in ["Schott formula", "formula A"]:
            self.delta_n_abs = getattr(
                dispersion_formulas, "delta_absolute_temperature"
            )
        elif self.formula_type == "dn/dT":
            self.delta_n_abs = getattr(
                dispersion_formulas, "n_absolute_with_given_dndt"
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
class Conditions:
    """
    A dataclass representing the environmental conditions.

    Attributes:
        temperature (float, optional): The temperature in Celsius. None if unspecified.
        pressure (float, optional): The pressure in MPa. None if unspecified.
    """

    temperature: float | None = None
    pressure: float | None = None

    @staticmethod
    def parse_temperature(temperature):
        """Parse temperature string to Celsius."""
        if isinstance(temperature, str):
            if "K" in temperature:
                return float(temperature.split()[0]) - 273.15
            elif "C" in temperature:
                return float(temperature.split()[0])
            else:
                warnings.warn("Temperature unit not recognized. Assuming Kelvin.")
                return float(temperature.split()[0]) - 273.15
        return Specs.parse_float(temperature)

    @staticmethod
    def parse_pressure(pressure):
        """Parse pressure string to MPa."""
        if isinstance(pressure, str):
            return Specs.parse_float(pressure.replace(" MPa", ""))
        return Specs.parse_float(pressure)

    @staticmethod
    def read_conditions_from_yaml(conditions_dict):
        """Read conditions from a YAML dictionary and create a Conditions object."""
        return Conditions(
            temperature=Conditions.parse_temperature(conditions_dict.get("temperature")),
            pressure=Conditions.parse_pressure(conditions_dict.get("pressure")),
        )

@dataclass
class Specs:
    """
    A dataclass representing material specifications.

    Attributes:
        n_is_absolute (bool, optional): Indicates whether the refractive index is given in absolute units.
            True if it is, False if not, and None if unspecified.
        wavelength_is_vacuum (bool, optional): Specifies whether the wavelength is given in vacuum.
            True if it is, False if not, and None if unspecified.

        thermal_dispersion (ThermalDispersion, optional): An instance of the ThermalDispersion class representing thermal dispersion information.
            None if unspecified.
        nd (float, optional): The refractive index (n) of the material. None if unspecified.
        Vd (float, optional): The Abbe number (Vd) of the material. None if unspecified.
        glass_code (float, optional): The glass code associated with the material. None if unspecified.
        glass_status (str, optional): The status or classification of the glass material as a string. None if unspecified.
        density (float, optional): The density of the material. None if unspecified.
        thermal_expansion (List[ThermalExpansion], optional): A list of instances of the ThermalExpansion class representing thermal expansion properties.
            None if unspecified.
        climatic_resistance (float, optional): The material's resistance to climatic conditions. None if unspecified.
        stain_resistance (float, optional): The material's resistance to staining. None if unspecified.
        acid_resistance (float, optional): The material's resistance to acids. None if unspecified.
        alkali_resistance (float, optional): The material's resistance to alkalis. None if unspecified.
        phosphate_resistance (float, optional): The material's resistance to phosphates. None if unspecified.
    """

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
    def parse_float(value, default=None):
        """Convert value to float if possible, otherwise return default."""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                return float(value.split()[0])
        except (ValueError, AttributeError):
            pass
        return default

    @staticmethod
    def parse_coefficients(coefficients):
        """Parse coefficients string into a numpy array."""
        if not coefficients:
            return None
        try:
            return np.array([float(val) for val in coefficients.split()], dtype=float)
        except (ValueError, AttributeError):
            return None



    @staticmethod
    def parse_density(density):
        """Parse density string to float."""
        if isinstance(density, str):
            return Specs.parse_float(density.replace(" g/cm<sup>3</sup>", ""))
        return Specs.parse_float(density)

    @staticmethod
    def parse_thermal_dispersion(td_dict):
        """Parse a ThermalDispersion object from dictionary."""
        formula_type = td_dict.get("type")
        if formula_type in ["Schott formula", "formula A"]:
            coefficients = Specs.parse_coefficients(td_dict.get("coefficients"))
            return ThermalDispersion(formula_type=formula_type, coefficients=coefficients)
        elif formula_type == "dn/dT":
            value = Specs.parse_float(td_dict.get("value"))
            return ThermalDispersion(formula_type=formula_type, coefficients=value)
        else:
            warnings.warn(f"Thermal Dispersion formula {formula_type} not implemented yet.")
            return None

    @staticmethod
    def parse_temperature_range(tr_dict):
        """Parse a TemperatureRange object from dictionary."""
        if not tr_dict or "temperature_range" not in tr_dict:
            return None
        try:
            min_temp, max_temp = map(float, tr_dict["temperature_range"].split())
            return TemperatureRange(min=min_temp, max=max_temp)
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def read_specs_from_yaml(specs_dict):
        """Read specs from a YAML dictionary and create a Specs object."""
        # Thermal Dispersion
        thermal_dispersion_list = [
            Specs.parse_thermal_dispersion(td)
            for td in specs_dict.get("thermal_dispersion", [])
        ]
        if len(thermal_dispersion_list) > 1:
            warnings.warn("Multiple thermal dispersion values found. Only the first will be used.")

        # Temperature Range
        temperature_range_list = [
            Specs.parse_temperature_range(tr)
            for tr in specs_dict.get("thermal_expansion", [])
        ]

        # Density
        density = Specs.parse_density(specs_dict.get("density"))

        # Thermal Expansion
        thermal_expansion = None
        if specs_dict.get("thermal_expansion"):
            thermal_expansion = [
                ThermalExpansion(
                    temperature_range=tr,
                    coefficient=Specs.parse_float(te_dict.get("coefficient") or te_dict.get("value"))
                )
                for tr, te_dict in zip(temperature_range_list, specs_dict["thermal_expansion"])
            ]

        # Create Specs object
        return Specs(
            n_is_absolute=specs_dict.get("n_is_absolute"),
            wavelength_is_vacuum=specs_dict.get("wavelength_is_vacuum"),
            thermal_dispersion=thermal_dispersion_list[0] if thermal_dispersion_list else None,
            nd=specs_dict.get("nd"),
            Vd=specs_dict.get("Vd"),
            glass_code=specs_dict.get("glass_code"),
            glass_status=specs_dict.get("glass_status"),
            density=density,
            thermal_expansion=thermal_expansion,
            climatic_resistance=specs_dict.get("climatic_resistance"),
            stain_resistance=specs_dict.get("stain_resistance"),
            acid_resistance=specs_dict.get("acid_resistance"),
            alkali_resistance=specs_dict.get("alkali_resistance"),
            phosphate_resistance=specs_dict.get("phosphate_resistance"),
        )

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
    """
    A dataclass representing a material's properties.

    Attributes:
        n (TabulatedIndexData | FormulaIndexData | None, optional): The refractive index (n) data for the material.
            It can be an instance of TabulatedIndexData or FormulaIndexData, or None if unspecified.
        k (TabulatedIndexData | FormulaIndexData | None, optional): The extinction coefficient (k) data for the material.
            It can be an instance of TabulatedIndexData or FormulaIndexData, or None if unspecified.
        specs (Specs | None, optional): An instance of the Specs dataclass representing material specifications.
            None if unspecified.
        conditions: (Conditions | None, optional): An instance of the Conditions dataclass representing environmental conditions.
        yaml_data (YAMLLibraryData, optional): An instance of the YAMLLibraryData class representing YAML library data.
            None if unspecified.
        name (str, optional): The name of the material. Normally extracted from the YAML data, but can be overridden. Empty string if unspecified.
    """

    n: TabulatedIndexData | FormulaIndexData | None = field(default=None)
    k: TabulatedIndexData | FormulaIndexData | None = field(default=None)
    specs: Specs | None = field(default=None)
    conditions: Conditions | None = field(default=None)
    yaml_data: YAMLLibraryData = field(default=None)
    name: str = field(default=yaml_data.name if yaml_data else "")

    def get_n(self, wavelength: float | np.ndarray) -> float | np.ndarray:
        """Get refractive index n for a given wavelength or array of wavelengths.

        Args:
            wavelength: Wavelength or array of wavelengths in micrometers.

        Returns:
            float or array of floats: Refractive index n for the given wavelength
        """
        if self.n is None:
            warnings.warn("No n data. Returning 0")
            return np.zeros_like(wavelength)
        return self.n.get_n_or_k(wavelength)

    def get_k(self, wavelength: float | np.ndarray) -> float | np.ndarray:
        """Get extinction coefficient k for a given wavelength or array of wavelengths.

        Args:
            wavelength: Wavelength or array of wavelengths in micrometers.

        Returns:
            float or array of floats: Extinction coefficient k for the given wavelength
        """
        if self.k is None:
            warnings.warn("No k data. Returning 0")
            return np.zeros_like(wavelength)
        return self.k.get_n_or_k(wavelength)

    def get_nk(self, wavelength: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Get refractive index n and extinction coefficient k for a given wavelength or array of wavelengths.

        Args:
            wavelength: Wavelength or array of wavelengths in micrometers.

        Returns:
            tuple of floats or arrays of floats: Refractive index n and extinction coefficient k for the given wavelength
        """
        return self.get_n(wavelength), self.get_k(wavelength)

    def get_n_at_temperature(
        self, wavelength: float | np.ndarray, temperature: float, P: float = 0.10133
    ) -> float | np.ndarray:
        """Get refractive index n at a given temperature for a given wavelength or array of wavelengths.

        Args:
            wavelength: Wavelength or array of wavelengths in micrometers.
            temperature: Temperature in degrees
            P: Pressure in MPa. Default is 0.10133 MPa (1 atm).

        Returns:
            float or array of floats: Refractive index n at the given temperature for the given wavelength
        """
        assert self.specs is not None, "There are no specs available for this material"
        assert self.specs.thermal_expansion is not None, (
            "There is no thermal dispersion formula available " "for this material"
        )

        if self.specs.wavelength_is_vacuum:
            n_abs = self.get_n(wavelength)
            return n_abs + self.specs.thermal_dispersion.delta_n_abs(
                n_abs,
                wavelength,
                temperature - self.conditions.temperature + 273.15,
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
                n_rel, rel_wavelength, self.conditions.temperature - 273.15, 0.10133
            )
            n_abs += self.specs.thermal_dispersion.delta_n_abs(
                n_abs,
                rel_wavelength,
                temperature - self.conditions.temperature + 273.15,
                *self.specs.thermal_dispersion.coefficients,
            )
            return dispersion_formulas.absolute_to_relative(
                n_abs, rel_wavelength, temperature, P
            )

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


@dataclass
class TabulatedIndexData:
    """
    A dataclass representing tabulated index data.

    Attributes:
        wl (np.ndarray | list[float]): An array or list containing wavelength values (in microns).
        n_or_k (np.ndarray | list[float]): An array or list containing refractive index (n) or extinction coefficient (k) values.
        ip (callable, read-only): A callable property to perform interpolation on the data.
        interpolation_func (str, optional): Indicates what interpolation function to use. Default is 'interp1d'. Can also be 'spline'.
        bounds_error (bool, optional): Indicates whether to raise an error for out-of-bounds queries.
            True to raise an error, False to suppress errors. Default is True.
    """

    wl: np.ndarray | list[float]
    n_or_k: np.ndarray | list[float]
    ip: callable = field(init=False)
    interpolation_func: str = field(default="interp1d")
    bounds_error: bool = field(default=True)

    def __post_init__(self):
        if self.interpolation_func == 'interp1d':
            self.ip = interp1d(
                np.atleast_1d(self.wl),
                np.atleast_1d(self.n_or_k),
                bounds_error=self.bounds_error,
            )
        elif self.interpolation_func == 'spline':
            self.ip = interp1d(
                np.atleast_1d(self.wl),
                np.atleast_1d(self.n_or_k),
                kind='cubic',
                bounds_error=self.bounds_error,
            )
        else:
            raise ValueError(f"Interpolation function {self.interpolation_func} not supported.")

    def get_n_or_k(self, wavelength):
        return self.ip(wavelength)


@dataclass
class FormulaIndexData:
    """
    A dataclass representing formula-based index data.

    Attributes:
        formula (callable): A callable function that computes the refractive index (n) or extinction coefficient (k) for a given wavelength.
        coefficients (np.array): An array of coefficients required by the formula function.
        min_wl (float, optional): The minimum wavelength (in microns) for which the formula is valid. Default is negative infinity.
        max_wl (float, optional): The maximum wavelength (in microns) for which the formula is valid. Default is positive infinity.
    """

    formula: callable | str
    coefficients: np.array
    min_wl: float = field(default=-np.inf)
    max_wl: float = field(default=np.inf)

    def __post_init__(self):
        self.coefficients = np.array(self.coefficients, dtype=float)
        if isinstance(self.formula, str):
            self.formula = getattr(dispersion_formulas, self.formula)

    def get_n_or_k(self, wavelength: float | np.ndarray) -> float | np.ndarray:
        if isinstance(wavelength, float) or isinstance(wavelength, np.ndarray):
            return self.formula(wavelength, self.coefficients)
        elif isinstance(wavelength, list):
            return self.formula(np.array(wavelength), self.coefficients)
        else:
            raise ValueError(f"The datatype {type(wavelength)} is not supported.")


@dataclass
class YAMLRefractiveIndexData:
    """
    A dataclass representing refractive index data in YAML format.

    Attributes:
        data_type (str): The type of refractive index data, such as 'tabulated', 'formula', etc.
        wavelength_range (str, optional): The range of wavelengths for which the data is valid.
            Default is an empty string, indicating unspecified range.
        coefficients (str, optional): A string containing coefficients or formula details for formula-based data.
            Default is an empty string, indicating unspecified coefficients.
        data (str, optional): A string containing the actual data in YAML format.
            Default is an empty string, indicating unspecified data.
    """

    data_type: str
    wavelength_range: str = field(default="")
    coefficients: str = field(default="")
    data: str = field(default="")


@dataclass
class YAMLMaterialData:
    """
    A dataclass representing material data in YAML format.

    Attributes:
        n_data (YAMLRefractiveIndexData): An instance of YAMLRefractiveIndexData containing refractive index (n) data.
        k_data (YAMLRefractiveIndexData): An instance of YAMLRefractiveIndexData containing extinction coefficient (k) data.
        comments (str, optional): Additional comments or notes related to the material data.
            Default is an empty string, indicating no comments.
        references (str, optional): References or sources of information for the material data.
            Default is an empty string, indicating no references.
    """

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
    """
    The dataclass representing the refractive index library using data from RefractiveIndex.info.

    Attributes:
        path_to_library (Path, optional): The path to the refractive index library YAML file.
            Default is a path pointing to a default library file.
        auto_upgrade (bool, optional): Automatically upgrade the library when initialized if set to True.
            Default is False.
        force_upgrade (bool, optional): Forcefully upgrade the library even if not necessary if set to True.
            Default is False.
        materials_yaml (List[YAMLLibraryData], read-only): A list of YAML library data instances representing materials.
        materials_dict (Dict[str, Dict[str, Dict[str, Material]]], read-only): A dictionary of materials organized by catalog, category, and name.
        materials_list (List[Material], read-only): A list of all materials contained in the library.
        github_sha (str, read-only): The GitHub SHA corresponding to the version of the library data.
    """

    path_to_library: Path = field(
        default=Path(__file__)
        .absolute()
        .parent.parent.joinpath("database/catalog-nk.yml")
    )
    auto_upgrade: bool = field(default=False)
    force_upgrade: bool = field(default=False)
    materials_yaml: list[YAMLLibraryData] = field(default_factory=list, init=False)
    materials_dict: dict[str, dict[str, dict[str, Material | YAMLLibraryData]]] = field(
        default_factory=dict, init=False
    )
    materials_list: list[Material | YAMLLibraryData] = field(default_factory=list, init=False)
    github_sha: str = field(default="", init=False)

    def _is_library_outdated(self) -> bool:
        """Checks if local library is outdated"""
        if self.path_to_library.parent.joinpath(".local_sha").exists():
            # get local release tag
            with open(self.path_to_library.parent.joinpath(".local_sha"), "r") as file:
                local_tag = file.readline().strip()
            # get current release tag on GitHub
            try:
                release_url = "https://api.github.com/repos/polyanskiy/refractiveindex.info-database/releases/latest"
                response = requests.get(release_url)
                self.github_sha = response.json()["tag_name"]
            except KeyError:
                logger.warning(
                    "Couldn't get the latest release tag on GitHub. Database cannot be updated."
                )
                self.github_sha = ""
                return False
            return not (self.github_sha == local_tag)
        else:
            logger.info("No local library exists.")
            return True

    def _download_latest_commit(self) -> bool:
        """Download latest library from GitHub.

        Downloads the latest library from the refractiveindex.info GitHub repository
         and extracts the necessary data files.

        Returns:
            bool: True if the library was successfully downloaded, False otherwise.
        """
        if self._is_library_outdated() or self.force_upgrade:
            logger.info("New Library available... Downloading...")
            release_url = "https://api.github.com/repos/polyanskiy/refractiveindex.info-database/releases/latest"
            response = requests.get(release_url)
            release_data = response.json()
            zip_url = release_data["zipball_url"]
            response = requests.get(zip_url)

            with open(self.path_to_library.with_suffix(".zip"), "wb") as file:
                file.write(response.content)
            with zipfile.ZipFile(self.path_to_library.with_suffix(".zip"), "r") as file:
                file_list = file.namelist()
                subfolder_files = [
                    file
                    for file in file_list
                    if file.startswith(f"{file_list[0]}database/data")
                    and file.endswith(".yml")
                ]
                subfolder_files.append(f"{file_list[0]}database/catalog-nk.yml")
                for fn in subfolder_files:
                    logger.debug(fn)
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
        """Load data from yaml file for internal use

        Returns:
            None
        """
        logger.info("load from yaml")
        with open(self.path_to_library, encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)

            for s in yaml_data:
                for book in s.get("content", []):
                    if "BOOK" not in book:
                        continue
                    for page in book.get("content", []):
                        if "PAGE" not in page:
                            continue
                        self.materials_yaml.append(
                            YAMLLibraryData(
                                name=page["name"],
                                lib_page=page["PAGE"],
                                lib_book=book["BOOK"],
                                lib_shelf=s["SHELF"],
                                lib_data=page["data"],
                                lib_path=self.path_to_library.parent.joinpath("data", page["data"]),
                            )
                        )

    def _convert_to_material_dict(self):
        """Convert yaml data to Material objects

        Returns:
            None
        """
        for m in self.materials_yaml:
            # try to load material from yaml
            mat = yaml_to_material(self.path_to_library.parent.joinpath("data").joinpath(m.lib_data), m.lib_shelf,
                               m.lib_book, m.lib_page, m.name)
            if mat:
                # add material to dict, use shelf, book and page as keys
                self.materials_dict.setdefault(m.lib_shelf, {}).setdefault(m.lib_book, {})[m.lib_page] = mat
                self.materials_list.append(mat)
                # Save each material to a separate pickle file
                material_pickle_path = self.path_to_library.parent.joinpath(
                    f"pickled/{m.lib_shelf}_{m.lib_book}_{m.lib_page}.pkl")
                # create pickled folder if it doesn't exist
                if not material_pickle_path.parent.is_dir():
                    material_pickle_path.parent.mkdir()
                with open(material_pickle_path, "wb") as f:
                    pickle.dump(mat, f, pickle.HIGHEST_PROTOCOL)

        with open(self.path_to_library.with_suffix(".pickle"), "wb") as f:
            pickle.dump(self.materials_yaml, f, pickle.HIGHEST_PROTOCOL)
        # with open(self.path_to_library.with_suffix(".pickle2"), "wb") as f:
        #     pickle.dump(self.materials_dict, f, pickle.HIGHEST_PROTOCOL)

    def _load_from_pickle(self):
        logger.info("load from pickle")
        with open(self.path_to_library.with_suffix(".pickle"), "rb") as f:
            self.materials_yaml = pickle.load(f)
        for m in self.materials_yaml:
            self.materials_dict.setdefault(m.lib_shelf, {}).setdefault(m.lib_book, {})[m.lib_page] = m
            self.materials_list.append(m)
        #
        # for sd in self.materials_dict.values():
        #     for bd in sd.values():
        #         for mat in bd.values():
        #             self.materials_list.append(mat)
        logger.info("... done.")

    def __post_init__(self):
        upgraded = False
        # create database folder if it doesn't exist
        try:
            if not self.path_to_library.parent.is_dir():
                self.path_to_library.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            warnings.warn(f"Failed to create directory: {e}")
            return

        # check if the folder is empty or upgrade is needed
        try:
            if self.auto_upgrade or self.force_upgrade or not os.listdir(self.path_to_library.parent):
                upgraded = self._download_latest_commit()
        except Exception as e:
            warnings.warn(f"Failed to check directory contents or download latest commit: {e}")
            return

        # load data from the appropriate source
        try:
            if self.path_to_library.exists():
                if self.path_to_library.with_suffix(".pickle").exists() and not upgraded:
                    self._load_from_pickle()
                else:
                    self._load_from_yaml()
                    self._convert_to_material_dict()
            else:
                warnings.warn(
                    "Path to library does not exist! Please check path or activate auto_upgrade to download."
                )
        except Exception as e:
            warnings.warn(f"Failed to load data: {e}")

    def search_material_by_page_name(
        self, page_name: str, exact_match: bool = False
    ) -> Material | list[Material] | None:
        """Search Material by name

        Search a Material by page name as given at refractiveindex.info.
        Sometimes, the name is not unique, so the function returns either a single Material or a list of Materials
        or None if it doesn't find a match.

        Args:
            page_name: name of the material as given as page name on refractiveindex.info
            exact_match: if True, only exact matches are considered. Default is False. e.g. if False,
            'BK7' will return all BK7 glasses, including N-BK7, K-BK7, etc.

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
                    if isinstance(m, Material):
                        materials.append(m)
                    elif isinstance(m, YAMLLibraryData):
                        materials.append(self.get_material(m.lib_shelf, m.lib_book, m.lib_page))
                    else:
                        warnings.warn("Unknown material type.")
        else:
            for m in self.materials_list:
                if isinstance(m, YAMLLibraryData):
                    if page_name in m.name:
                        materials.append(self.get_material(m.lib_shelf, m.lib_book, m.lib_page))
                elif isinstance(m, Material):
                    if page_name in m.yaml_data.name:
                        materials.append(m)
                else:
                    warnings.warn("Unknown material type.")
        return (
            materials[0]
            if len(materials) == 1
            else materials if len(materials) > 1 else None
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
        for shelf_m, d in self.materials_dict.items():
            if not (shelf_m == filter_shelf or filter_shelf is None):
                continue
            for book_name, book_m in d.items():
                if filter_book is not None:
                    if filter_book.lower() not in book_name.lower():
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
            >>> bk7 = db.get_material("specs", "SCHOTT-optical", "N-BK7")
            >>> print(bk7.get_n(0.5875618))
            1.5168000345005885
        """
        if shelf not in self.materials_dict:
            raise ValueError(f"Shelf {shelf} not found in database.")
        if book not in self.materials_dict[shelf]:
            raise ValueError(f"Book {book} not found in database.")
        if page not in self.materials_dict[shelf][book]:
            raise ValueError(f"Page {page} not found in database.")
        if not self.materials_dict[shelf][book][page]:
            raise ValueError(f"Material {shelf}/{book}/{page} not found in database.")
        # check if material is a string or YAMLLibraryData object, if so load it and replace it in the dict
        if isinstance(self.materials_dict[shelf][book][page], YAMLLibraryData):
            # check if pickled material exists
            material_pickle_path = self.path_to_library.parent.joinpath(
                f"pickled/{shelf}_{book}_{page}.pkl"
            )
            if material_pickle_path.exists():
                with open(material_pickle_path, "rb") as f:
                    self.materials_dict[shelf][book][page] = pickle.load(f)
            else:

                self.materials_dict[shelf][book][page] = yaml_to_material(
                    self.materials_dict[shelf][book][page].lib_path,
                    shelf,
                    book,
                    page,
                    self.materials_dict[shelf][book][page].name,
                )

        return self.materials_dict[shelf][book][page]

    def get_material_by_path(self, yaml_path: str) -> Material:
        """Get material by path

        Args:
            yaml_path: path as shown on refractive index when hovered over 'CSV - comma separated data'

        Returns:
            Material object
        """
        mat_found = [
            m
            for m in self.materials_list
            if str(m.yaml_data.lib_path).lower().endswith(yaml_path.lower() + ".yml")
        ]
        return mat_found[0] if mat_found else None


def yaml_to_material(filepath: str | Path, lib_shelf: str, lib_book: str, lib_page: str,
                     lib_name: str) -> Material | None:
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
    filepath = Path(filepath) if isinstance(filepath, str) else filepath
    def fill_variables_from_data_dict(data):
        """Helper function to split data in yaml into Material attributes"""
        _wl_min = _wl_max = _wl = _n = _k = _coefficients = _formula = None
        data_type = data["type"]

        if "tabulated" in data_type:
            # Load tabulated data
            raw_data = np.loadtxt(data["data"].split("\n"))

            # Ensure data is 2D even for a single row
            if raw_data.ndim == 1:
                raw_data = raw_data[np.newaxis, :]  # Convert 1D array to 2D

            _wl = raw_data[:, 0]  # Wavelength is always the first column
            if "nk" in data_type:
                _n, _k = raw_data[:, 1], raw_data[:, 2]  # n and k values
            elif "n" in data_type:
                _n = raw_data[:, 1]  # Only n values
            elif "k" in data_type:
                _k = raw_data[:, 1]  # Only k values

            _wl_min, _wl_max = np.min(_wl), np.max(_wl)
        elif "formula" in data_type:
            _wl_range = data.get("wavelength_range") or data.get("range")
            try:
                _wl_min, _wl_max = [float(w) for w in _wl_range.split()]
            except ValueError:
                _wl_min, _wl_max = (None, None)
            _coefficients = np.array([float(c) for c in data["coefficients"].split()])
            _formula = data_type.split()[1]
        return _wl, _wl_min, _wl_max, _n, _k, _coefficients, _formula

    try:
        with open(filepath, encoding='utf-8') as f:
            yaml_content = f.read()  # Read the file once
            # yaml_parser = YAML()
            d = yaml.safe_load(yaml_content)  # Parse the YAML content

            data_blocks = d.get("DATA", [])
            specs = Specs.read_specs_from_yaml(d.get("PROPERTIES", None)) if "PROPERTIES" in d else None
            conditions = Conditions.read_conditions_from_yaml(d.get("CONDITIONS", None)) if "CONDITIONS" in d else None

            n_class, k_class = None, None
            if data_blocks:
                if isinstance(data_blocks, list):
                    n_data = data_blocks[0]
                    wl_n, wl_min_n, wl_max_n, n, k, coefficients, formula = fill_variables_from_data_dict(n_data)
                    if formula:
                        n_class = FormulaIndexData(getattr(dispersion_formulas, f"formula_{formula}"), coefficients,
                                                   wl_min_n, wl_max_n)
                    elif n is not None:
                        n_class = TabulatedIndexData(wl_n, n, 'interp1d', True)
                        k_class = TabulatedIndexData(wl_n, k, 'interp1d', True) if k is not None else None

                    k_data = next((item for item in data_blocks[1:] if item), {"type": ""})
                    wl_k, wl_min_k, wl_max_k, _, k, coefficients_k, formula_k = fill_variables_from_data_dict(k_data)
                    if formula_k:
                        k_class = FormulaIndexData(getattr(dispersion_formulas, f"formula_{formula_k}"), coefficients_k,
                                                   wl_min_k, wl_max_k)
                    elif k is not None:
                        k_class = TabulatedIndexData(wl_k, k, 'interp1d', True)
                else:
                    raise NotImplementedError("Dict data blocks not implemented yet.")

            return Material(
                n_class,
                k_class,
                specs,
                conditions,
                YAMLLibraryData(lib_name, yaml_content, lib_shelf, lib_book, lib_page, filepath),
            )
    except Exception as e:
        logger.warning(f"Could not convert/load data in {filepath}: {e}")
        return None


def load_material(m, path_to_library):
    """Helper function to load a material."""
    mat = yaml_to_material(
        path_to_library.parent.joinpath("data").joinpath(m.lib_data),
        m.lib_shelf, m.lib_book, m.lib_page, m.name
    )
    if mat:
        return m.lib_shelf, m.lib_book, m.lib_page, mat
    return None

def fit_tabulated(
        tid: TabulatedIndexData, formula: callable, coefficients: list | np.ndarray, debug: bool = False
) -> FormulaIndexData:
    from scipy.optimize import least_squares

    def fit_func(x, wl, n):
        return formula(wl, x) - n

    res = least_squares(fit_func, coefficients, args=(np.array(tid.wl, dtype=float), np.array(tid.n_or_k, dtype=float)))
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(tid.wl, tid.n_or_k, label="data")
        wl = np.linspace(min(tid.wl), max(tid.wl), 1000)
        plt.plot(wl, formula(wl, res.x), label="fit")
        plt.legend()
        plt.show()
    return FormulaIndexData(formula, res.x)
