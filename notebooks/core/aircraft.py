import os
import pandas as pd
from functools import cache
from core import atmos
import numpy as np
import polars as pl


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
    def __init__(self, data_dir, ac_ID):
        df_aircrafts = pl.read_csv(data_dir).to_pandas()

        self.ac_data = df_aircrafts[df_aircrafts["ID"] == ac_ID]
        self.ac_ID = ac_ID
        self.ac_type = self.ac_data["type"].values

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
