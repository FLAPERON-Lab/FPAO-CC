import os
import pandas as pd
from functools import cache
from core import atmos
import numpy as np
import polars as pl
import marimo as mo
from core import plot_utils
from core.plot_utils import InitialFig


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

        # Plotting variables
        self.h_array = np.linspace(0, atmos.hmax, plot_utils.meshgrid_n)
        self.rho_array = atmos.rho(self.h_array)
        self.rhoratio_array = atmos.rhoratio(self.h_array)
        self.dT_array = np.linspace(0, 1, plot_utils.meshgrid_n)


class OptimumCondition:
    def compute_optimal(self, W, h, Model, equality=False):
        if not equality:
            self.hopt_array = Model.aircraft.h_array[self.condition]

        self.cond = 1 if h in self.hopt_array else np.nan
        self.V_envelope = Model.compute_velocity(W, self.hopt_array, self.CLopt)
        self.V_selected = Model.compute_velocity(W, h, self.CLopt_selected) * self.cond
        self.power_selected = (
            Model.compute_drag(W, self.CLopt_selected) * self.V_selected
        )

        self.h_selected = h / 1e3
        self.CLopt_selected *= self.cond
        self.hopt_array /= 1e3


class SimplifiedAircraft:
    def __init__(self, database: AircraftBase):
        self.aircraft = database

    def compute_drag_curve(self, W):
        return W / self.aircraft.E_array

    def compute_drag(self, W, CL):
        if CL == 0:
            return float("inf")
        return W / CL * (self.aircraft.CD0 + self.aircraft.K * CL**2)

    def compute_velocity(self, W, h, CL):
        return np.sqrt(2 * W / (atmos.rho(h) * self.aircraft.S * CL))

    def update_mass_dependency(self, W):
        """Compute all the values dependent only on the mass changing"""
        self.drag_curve = self.compute_drag_curve(W)
        self.Vstall_envelope = self.compute_velocity(
            W, self.aircraft.h_array, self.aircraft.CLmax
        )

        CL_a0 = (
            self.aircraft.OEM
            * atmos.g0
            * 2
            / (atmos.rho0 * self.aircraft.S * atmos.a(0) ** 2)
        )

        self.drag_ylim = (
            1
            * self.aircraft.OEM
            * atmos.g0
            * (self.aircraft.CD0 + self.aircraft.K * CL_a0**2)
            / CL_a0
        )
        self.power_ylim = 0.5 * self.drag_ylim * atmos.a(0) / 1e3

    def plot_optimum(
        self, surface, Condition, equality=False, constraint=True, factor=2
    ):
        configTraces = plot_utils.configTraces(
            self, surface, constraint=constraint, factor=factor
        )

        return plot_utils.OptimumGridViewNew(self, configTraces, Condition, equality)

    def plot_grid(self, condition, plot_options):
        return plot_utils.OptimumGridView(self, condition, plot_options)

    def plot_initial(self, plot_options, selected):
        return InitialFig(self, plot_options, selected)

    # ===== Shared API, overridden by subclasses =====

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

    def update_altitude_dependency(self, h):
        self.thrust = np.repeat(self.compute_thrust(h), plot_utils.meshgrid_n)
        self.idx_h = int(
            (h - self.aircraft.h_array[0])
            / (self.aircraft.h_array[2] - self.aircraft.h_array[1])
        )
        self.rhoratio_selected = atmos.rhoratio(h)

    def update_context(self, W, h):
        self.V_CLarray = self.compute_velocity(W, h, self.aircraft.CL_array)
        self.power_available = self.compute_power(h, self.V_CLarray)
        self.power_required = self.drag_curve * self.V_CLarray
        self.equilibrium_dT = self.drag_curve / self.thrust
        self.V_CLP = self.compute_velocity(W, h, self.aircraft.CL_P)
        self.V_CLE = self.compute_velocity(W, h, self.aircraft.CL_E)


class ModelSimplifiedProp(SimplifiedAircraft):
    def compute_power(self, h, velocity=None):
        rho_ratio = atmos.rhoratio(h)
        return (self.aircraft.Pa0 * rho_ratio**self.aircraft.beta) * 1e3

    def compute_thrust(self, h, velocity):
        return self.compute_power(h, velocity) / velocity

    def update_altitude_dependency(self, h):
        self.power_available = np.repeat(self.compute_power(h), plot_utils.meshgrid_n)
        self.idx_h = int(
            (h - self.aircraft.h_array[0])
            / (self.aircraft.h_array[2] - self.aircraft.h_array[1])
        )
        self.rhoratio_selected = atmos.rhoratio(h)

    def update_context(self, W, h):
        self.V_CLarray = self.compute_velocity(W, h, self.aircraft.CL_array)
        self.power_required = self.drag_curve * self.V_CLarray
        self.thrust = self.compute_thrust(h, self.V_CLarray)
        self.equilibrium_dT = self.drag_curve / self.thrust
        self.V_CLP = self.compute_velocity(W, h, self.aircraft.CL_P)
        self.V_CLE = self.compute_velocity(W, h, self.aircraft.CL_E)


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
