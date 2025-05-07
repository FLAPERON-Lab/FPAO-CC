import os
import pandas as pd
from functools import cache
from core import atmos
import numpy as np

# Data file paths
curr_path = os.path.dirname(os.path.realpath(__file__))
dir_aircraft = os.path.join(os.path.dirname(os.getcwd()), "data", "aircraft")
simplifiedProps_dir = os.path.join(dir_aircraft, "AircraftDB_Standard_Props.ssv")
simplifiedJets_dir = os.path.join(dir_aircraft, "AircraftDB_Standard_Jets.ssv")


@cache
def available_aircrafts(ac_type=None, data=False):
    """Return the available aircrafts"""

    # Load the data
    simplified_props = pd.read_csv(simplifiedProps_dir, sep=";")

    simplified_jets = pd.read_csv(simplifiedJets_dir, sep=";")

    aircraft_map = {
        "Simplified Propeller": simplified_props,
        "Simplified Jet": simplified_jets,
        "Any": pd.concat([simplified_props, simplified_jets], ignore_index=True),
    }

    if data:
        return aircraft_map[ac_type]

    return list(aircraft_map.get(ac_type, [])["name"])


@cache
class Aircraft:
    def __init__(self, ac_name, ac_type):
        file_map = {
            "Simplified Jet": simplifiedJets_dir,
            "Simplified Propeller": simplifiedProps_dir,
        }
        if ac_type not in file_map:
            return None

        df_aircrafts = pd.read_csv(file_map[ac_type], sep=";")

        self.ac_data = df_aircrafts[df_aircrafts["name"] == ac_name]
        self.ac_name = ac_name
        self.ac_type = ac_type

    def thrust(self, V=None, beta=None, h=None, deltaT=None):
        if self.ac_type == "Simplified Jet":
            Ta0 = self.ac_data["Ta0"].item()
            Ta = Ta0 * atmos.rhoratio(h) ** beta
            T = deltaT * Ta
            return Ta, T

        elif self.ac_type == "Simplified Propeller":
            Pa, P = self.power(V, beta, h, deltaT)

            if V == 0:
                raise ZeroDivisionError("Velocity must be non-zero")

            Ta = Pa / V
            return Pa, None

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
