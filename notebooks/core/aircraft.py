import os
import pandas as pd
from functools import cache
from core import atmos
import numpy as np

# Data file paths
curr_path = os.path.dirname(os.path.realpath(__file__))
dir_aircraft = os.path.join(os.path.dirname(os.getcwd()), "data", "aircraft")
simplified_dir = os.path.join(dir_aircraft, "AircraftDB_Standard.ssv")


@cache
def available_aircrafts(ac_type=None):
    """Return the available aircrafts"""

    # Load the data
    simplified_aircrafts = pd.read_csv(simplified_dir, sep=";")

    if ac_type:
        return simplified_aircrafts[simplified_aircrafts["type"] == ac_type]

    return simplified_aircrafts


@cache
class Aircraft:
    def __init__(self, ac_ID):
        df_aircrafts = pd.read_csv(simplified_dir, sep=";")

        self.ac_data = df_aircrafts[df_aircrafts["ID"] == ac_ID]
        self.ac_ID = ac_ID
        self.ac_type = self.ac_data["type"].values

    def thrust(self, V=None, beta=None, h=None, deltaT=None):
        if self.ac_type == "Simplified Jet":
            Ta0 = self.ac_data["Ta0"].item()
            Ta = np.full_like(V, Ta0 * atmos.rhoratio(h) ** beta)
            T = deltaT * Ta
            return Ta, T

        elif self.ac_type == "Simplified Propeller":
            Pa, P = self.power(V, beta, h, deltaT)

            Ta = np.full_like(V, np.nan, dtype=float)

            mask = V != 0
            Ta[mask] = Pa[mask] / V[mask]

            return Ta, None

    def power(self, V=None, beta=None, h=None, deltaT=None):
        if self.ac_type == "Simplified Jet":
            Ta, T = self.thrust(V, beta, h, deltaT)
            Pa = Ta * V
            return Pa, None

        elif self.ac_type == "Simplified Propeller":
            Pa0 = self.ac_data["Pa0"].item()
            Pa = np.full_like(V, Pa0 * atmos.rhoratio(h) ** beta)
            P = deltaT * Pa

            return Pa, P

    def drag_polar(self, CL):
        cd0 = self.ac_data["cd0"].item()
        k = self.ac_data["k"].item()
        raise cd0 + k * CL**2

    def fuel_flow(self, V=None, beta=None, h=None, deltaT=None):
        if self.ac_type == "Simplified Jet":
            cT = self.ac_data["cT"].item()
            FF = cT * self.thrust(V, beta, h, deltaT)
        elif self.ac_type == "Simplified Prop":
            cP = self.ac_data["cP"].item()
            FF = cP * self.power(V, beta, h, deltaT)
        return FF
