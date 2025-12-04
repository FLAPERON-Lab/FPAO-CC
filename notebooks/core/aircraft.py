import os
import pandas as pd
from functools import cache
from core import atmos
import numpy as np
import polars as pl
import marimo as mo
from core import plot_utils


class AircraftBase:
    RENAME = {"CLmax_ld": "CLmax"}

    def __init__(self, selection):
        data = selection.to_dict().copy()

        # Apply renaming rules
        for old, new in self.RENAME.items():
            if old in data:
                data[new] = data.pop(old)

        # Store all attributes dynamically
        self.__dict__.update(data)

        # Derived performance quantities
        self.CL_E = np.sqrt(self.CD0 / self.K)
        self.CL_P = np.sqrt(3 * self.CD0 / self.K)

        self.CL_array = np.linspace(0, self.CLmax, plot_utils.meshgrid_n + 1)[1:]

        self.E_max = self.CL_E / (self.CD0 + self.K * self.CL_E**2)
        self.E_P = self.CL_P / (self.CD0 + self.K * self.CL_P**2)
        self.E_S = self.CLmax / (self.CD0 + self.K * self.CLmax**2)
        self.E_array = self.CL_array / (self.CD0 + self.K * self.CL_array**2)


"""

    def update_mass_properties(self, mass_selected):
        self.drag_curve = mass_selected / self.E_array
        self.velocity_stall_harray = np.sqrt(
            2 * mass_selected / (self.rho_array * self.S * self.CLmax)
        )

        CL_a0 = self.OEM * atmos.g0 * 2 / (atmos.rho0 * self.S * self.a_0**2)

        self.drag_yrange = (
            1 * self.OEM * atmos.g0 * (self.CD0 + self.K * CL_a0**2) / CL_a0
        )
        self.power_yrange = 0.5 * self.drag_yrange * self.a_0 / 1e3
        """


class SimplifiedAircraft:
    def __init__(self, database: AircraftBase):
        self.aircraft = database
        self._init_variables()

    def _init_variables(self):
        self.h_array = np.linspace(0, atmos.hmax, plot_utils.meshgrid_n)
        self.rho_array = atmos.rho(self.h_array)
        self.dT_array = np.linspace(0, 1, plot_utils.meshgrid_n)

    def compute_drag_curve(self, W):
        return W / self.aircraft.E_array

    def compute_velocity_array(self, W, h):
        rho = atmos.rho(h)
        return np.sqrt(W * 2 / (rho * self.aircraft.S * self.aircraft.CL_array))

    def compute_stall_envelope(self, W):
        return np.sqrt(2 * W / (self.rho_array * self.aircraft.S * self.aircraft.CLmax))

    def compute_velocity(self, W, h, CL):
        return np.sqrt(2 * W / (atmos.rho(h) * self.aircraft.S * CL))

    # ===== Shared API, overridden by subclasses ===== #

    def compute_thrust(self, h, velocity=None):
        raise NotImplementedError("Use Jet or Prop subclass.")

    def compute_power(self, h, velocity):
        raise NotImplementedError("Use Jet or Prop subclass.")


class ModelSimplifiedJet(SimplifiedAircraft):
    def compute_thrust(self, h, velocity=None):
        rho_ratio = atmos.rhoratio(h)
        return (self.aircraft.Ta0 * rho_ratio**self.aircraft.beta) * 1e3

    def compute_power(self, h, velocity):
        return self.compute_thrust(h) * velocity


class ModelSimplifiedProp(SimplifiedAircraft):
    def compute_power(self, h, velocity):
        rho_ratio = atmos.rhoratio(h)
        return (self.aircraft.Pa0 * rho_ratio**self.aircraft.beta) * 1e3

    def compute_thrust(self, h, velocity):
        return self.compute_power(h, velocity) / velocity


# Compute velocity as a function of C_L
def velocity(W, h, CL, S, cap=True, vertical_equilibrium=True):
    numerator = 2 * W  # scalar or array
    denominator = atmos.rho(h) * S * CL
    if vertical_equilibrium:
        vel = np.sqrt(
            np.divide(
                numerator,
                denominator,
                out=np.zeros_like(denominator),
                where=CL != 0,
            )
        )

        vel = np.where(vel == 0, np.nan, vel)
    if cap:
        return np.where(vel > atmos.a(h), np.nan, vel)
    else:
        return vel


def power(h, S, CD0, K, CL, V):
    D = drag(h, S, CD0, K, CL, V)

    return D * V


def drag(h, S, CD0, K, CL, V):
    rho = atmos.rho(h)

    CD = CD0 + K * CL**2

    return 0.5 * rho * V**2 * S * CD


def horizontal_constraint(
    W, h, CD0, K, CL, plant_parameter, beta, V=0, S=0, D=0, type="jet"
):
    """
    Returns the deltaT values using the horizontal constraint, the plant parameter is either Ta0 or Pa0 in SI units depending on the specified type
    """
    # Sigma ratio from rhoratio
    sigma = atmos.rhoratio(h)

    # Rewrite the jet section
    if type == "jet":
        Ta0 = plant_parameter
        deltaT = np.divide(
            W * (CD0 + K * CL**2) / (Ta0 * sigma**beta),
            CL,
            out=np.zeros_like(CL),
            where=CL != 0,
        )
    elif type == "propeller":
        Pa0 = plant_parameter
        deltaT = np.divide(
            D * V,
            Pa0 * sigma**beta,
            out=np.zeros_like(CL),
            where=V != 0,
        )

    return deltaT


def endurance(K, CD0, type_end):
    if type_end == "max":
        out = np.sqrt(1 / (4 * K * CD0))

    return out


def available_aircrafts(data_dir, verbose=False, round=True, ac_type=None):
    """Return the available aircrafts"""

    # Load the data
    data = pl.read_csv(data_dir).to_pandas()

    if ac_type:
        data = data[data["type"] == f"Simplified {ac_type}"]

    if round:
        cols_round = [
            "CD0",
            "K",
            "beta",
            "CLmax_cl",
            "CLmax_to",
            "CLmax_ld",
            "cT",
            "cP",
            "MMO",
        ]
        data[cols_round] = data[cols_round].round(4)

        other_cols = data.columns.difference(cols_round)
        data[other_cols] = data[other_cols].round(1)

    if not verbose and ac_type == "Jet":
        data = data[
            [
                "full_name",
                "ID",
                "type",
                "b",
                "S",
                "CD0",
                "K",
                "Ta0",
                "CLmax_ld",
                "MTOM",
                "OEM",
                "beta",
            ]
        ]
    elif not verbose and ac_type == "Propeller":
        data = data[
            [
                "full_name",
                "ID",
                "type",
                "b",
                "S",
                "CD0",
                "K",
                "Pa0",
                "CLmax_ld",
                "MTOM",
                "OEM",
                "beta",
            ]
        ]

    elif not verbose:
        data = data[
            [
                "full_name",
                "ID",
                "type",
                "b",
                "S",
                "CD0",
                "K",
                "CLmax_ld",
                "MTOM",
                "OEM",
                "beta",
            ]
        ]

    return data[data["CD0"].notna() & data["K"].notna()].reset_index(drop=True)


class Aircraft:
    def __init__(self, selection, type):
        self.__dict__.update(selection.to_dict())

        # Rename awkward ones
        self.CLmax = self.__dict__.pop("CLmax_ld")

        # Convert units
        if type == "Jet":
            self.Ta0 *= 1e3  # to Watts
        elif type == "Prop":
            self.Ta0 *= 1e3  # to Watts
        else:
            raise TypeError("Not a supported aircraft type")

        # Compute derived values
        self.CL_E = np.sqrt(self.CD0 / self.K)
        self.CL_P = np.sqrt(3 * self.CD0 / self.K)

        # Aerodyamic Efficiency
        self.E_max = self.CL_E / (self.CD0 + self.K * self.CL_E**2)
        self.E_P = self.CL_P / (self.CD0 + self.K * self.CL_P**2)
        self.E_S = self.CLmax / (self.CD0 + self.K * self.CLmax**2)

        self.dT_array = np.linspace(0, 1, plot_utils.meshgrid_n)  # -
        self.h_array = np.linspace(0, 20e3, plot_utils.meshgrid_n)  # meters

        self.CL_array = np.linspace(0, self.CLmax, plot_utils.meshgrid_n + 1)[1:]
        self.E_array = self.CL_array / (self.CD0 + self.K * self.CL_array**2)

        # Database cell
        self.a_0 = atmos.a(0)

        self.rho_array = atmos.rho(self.h_array)
        self.sigma_array = atmos.rhoratio(self.h_array)
        self.min_sigma = atmos.rhoratio(atmos.hmax)
        self.a_harray = atmos.a(self.h_array)

        self.ranges = [
            plot_utils.xy_lowerbound,
            self.CLmax + 0.05,
            plot_utils.xy_lowerbound,
            1 + 0.05,
            plot_utils.xy_lowerbound,
            self.a_0,
            plot_utils.xy_lowerbound,
            20,
        ]

        self.axes = (self.CL_array, self.dT_array)

    def update_CL_slider(self, selected_value):
        self.idx_CL = int(
            (selected_value - self.CL_array[0]) / (self.CL_array[2] - self.CL_array[1])
        )

    def update_mass_properties(self, mass_selected):
        self.drag_curve = mass_selected / self.E_array
        self.velocity_stall_harray = np.sqrt(
            2 * mass_selected / (self.rho_array * self.S * self.CLmax)
        )

        CL_a0 = self.OEM * atmos.g0 * 2 / (atmos.rho0 * self.S * self.a_0**2)

        self.drag_yrange = (
            1 * self.OEM * atmos.g0 * (self.CD0 + self.K * CL_a0**2) / CL_a0
        )
        self.power_yrange = 0.5 * self.drag_yrange * self.a_0 / 1e3

    def update_h_slider(self, selected_value):
        self.idx_h = int(
            (selected_value - self.h_array[0]) / (self.h_array[2] - self.h_array[1])
        )

        self.a_selected = atmos.a(selected_value)

        self.sigma_selected = atmos.rhoratio(selected_value)

        self.rho_selected = atmos.rho(selected_value)

    def update_context(self, selected_mass, selected_altitude):
        self.velocity_CL_array = velocity_CLarray = np.sqrt(
            2 * selected_mass / (self.rho_selected * self.S * self.CL_array)
        )
        self.velocity_CL_E = velocity_CLarray[-1] * np.sqrt(self.CLmax / self.CL_E)
        self.velocity_CL_P = velocity_CLarray[-1] * np.sqrt(self.CLmax / self.CL_P)

    def thrust(self, V, h, deltaT):
        beta = self.ac_data["beta"]
        if self.ac_type == "Simplified Jet":
            Ta0 = self.ac_data["Ta0"].item()
            Ta = np.full_like(V, Ta0 * atmos.rhoratio(h) ** beta)
            T = deltaT * Ta
            return Ta, T

        elif self.ac_type == "Simplified Propeller":
            Pa, P = self.power(V, h, deltaT)

            Ta = np.full_like(V, np.nan, dtype=float)

            mask = V != 0
            Ta[mask] = P[mask] / V[mask]

            return None, Ta

    def power(self, V, h, deltaT):
        beta = self.ac_data["beta"]
        if self.ac_type == "Simplified Jet":
            Ta, T = self.thrust(V, h, deltaT)
            Pa = T * V
            return None, Pa

        elif self.ac_type == "Simplified Propeller":
            Pa0 = self.ac_data["Pa0"].item()
            Pa = np.full_like(V, Pa0 * atmos.rhoratio(h) ** beta)
            P = deltaT * Pa

            return Pa, P

    def drag_polar(self, CL):
        cd0 = self.ac_data["CD0"].item()
        k = self.ac_data["K"].item()
        return cd0 + k * CL**2

    def fuel_flow(self, V, h, deltaT):
        if self.ac_type == "Simplified Jet":
            cT = self.ac_data["cT"].item()
            FF = cT * self.thrust(V, h, deltaT)[1]
        elif self.ac_type == "Simplified Prop":
            cP = self.ac_data["cP"].item()
            FF = cP * self.power(V, h, deltaT)[1]
        return FF
