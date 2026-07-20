# SPDX-FileCopyrightText: 2026 Carmine Varriale <C.varriale@tudelft.nl>
# SPDX-FileCopyrightText: 2026 Federico Angioni <F.angioni@student.tudelft.nl>
# SPDX-FileCopyrightText: 2026 Maarten van Hoven <M.B.vanHoven@tudelft.nl>
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np

# Constants
T0 = 288.15  # Sea level standard temperature in Kelvin
p0 = 101325  # Sea level standard pressure in Pascals
rho0 = 1.225  # Sea level standard density in kg/m^3
Tlapse = -0.0065  # Temperature lapse rate in K/m
R = 287.05  # Specific gas constant for dry air in J/(kg·K)
g0 = 9.80665  # Standard gravity in m/s^2
gamma = 1.4  # Ratio of specific heats for air

h11 = 11000  # Altitude at which the temperature lapse rate changes (11 km)
T11 = T0 + Tlapse * h11  # Temperature at h11
p11 = p0 * (T11 / T0) ** (-g0 / (Tlapse * R))  # Pressure at h11
rho11 = rho0 * (T11 / T0) ** (-(g0 / (Tlapse * R) + 1))  # Density at h11

hmax = 20000  # Maximum altitude for the model (20 km)


def T(h):
    """Calculate temperature at a given altitude."""
    T = np.where(h <= h11, T0 + Tlapse * h, T0 + Tlapse * h11)
    if np.any(h > hmax):
        raise ValueError("Altitude out of range (0-20000 meters)")
    return T


def p(h):
    """Calculate pressure at a given altitude."""
    T_vals = T(h)
    p = np.where(
        h <= h11,
        p0 * (T_vals / T0) ** (-g0 / (Tlapse * R)),
        p11 * np.exp(-g0 * (h - h11) / (R * T11)),
    )
    if np.any(h > hmax):
        raise ValueError("Altitude out of range (0-20000 meters)")
    return p


def rho(h):
    """Calculate density at a given altitude."""
    T_vals = T(h)
    rho = np.where(
        h <= h11,
        rho0 * (T_vals / T0) ** (-(g0 / (Tlapse * R) + 1)),
        rho11 * np.exp(-g0 * (h - h11) / (R * T11)),
    )
    if np.any(h > hmax):
        raise ValueError("Altitude out of range (0-20000 meters)")
    return rho


def a(h):
    """Calculate speed of sound at a given altitude."""
    T_vals = T(h)
    a = np.sqrt(gamma * R * T_vals)  # Speed of sound in m/s
    if np.any(h > hmax):
        raise ValueError("Altitude out of range (0-20000 meters)")
    return a


def Tratio(h):
    """Calculate the ratio of temperature to sea level temperature."""
    return T(h) / T0


def pratio(h):
    """Calculate the ratio of pressure to sea level pressure."""
    return p(h) / p0


def rhoratio(h):
    """Calculate the ratio of density to sea level density."""
    return rho(h) / rho0


def altitude(_rhoratio):

    _rho = _rhoratio * rho0

    exponent = -(g0 / Tlapse / R) - 1
    h = np.where(
        _rho > rho(h11),
        T0 / Tlapse * ((_rhoratio) ** (1 / exponent) - 1),
        h11 - R * T11 / g0 * np.log(_rho / rho(h11)),
    )

    return h


# Example usage
if __name__ == "__main__":
    altitudes = input("Enter altitudes in meters (comma-separated): ")
    altitudes = np.array([float(h) for h in altitudes.split(",")])
    try:
        print(f"At altitudes {altitudes}:")
        print(f"Temperature: {T(altitudes)} K")
        print(f"Pressure: {p(altitudes)} Pa")
        print(f"Density: {rho(altitudes)} kg/m^3")
        print(f"Speed of Sound: {a(altitudes)} m/s")
        print(f"Temperature Ratio: {Tratio(altitudes)}")
        print(f"Pressure Ratio: {pratio(altitudes)}")
        print(f"Density Ratio: {rhoratio(altitudes)}")
    except ValueError as e:
        print(e)
