# SPDX-FileCopyrightText: 2026 Carmine Varriale <C.varriale@tudelft.nl>
# SPDX-FileCopyrightText: 2026 Federico Angioni <F.angioni@student.tudelft.nl>
# SPDX-FileCopyrightText: 2026 Maarten van Hoven <M.B.vanHoven@tudelft.nl>
#
# SPDX-License-Identifier: Apache-2.0
import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))

    # Initialization code that runs before all other cells
    import marimo as mo
    from core import _defaults

    _defaults.FILEURL = _defaults.get_url()

    _defaults.set_plotly_template()


@app.cell
def _():
    _defaults.set_sidebar()
    return


@app.cell
def _():
    mo.md(r"""# Flight Controls""")
    return


@app.cell
def _():
    mo.md(
        r"""
    The aircraft motion, described by the EoMs, is controlled by the pilot and/or the autopilot system by acting on the  _flight controls_.

    One may think of the flight controls as the _inputs_ to the EoMs, which can be chosen in a more or less arbitrary way to achieve desired performance.
    """
    )
    return


@app.cell
def _():
    mo.md("""The flight controls drive the evolution of the aircraft as a dynamic system.   
    Their values can and should be selected at every time instant to achieve desired and/or optimal performance.""").callout(
        kind="success"
    ).style({"width": "75%", "text-align": "center"}).center()
    return


@app.cell
def _():
    mo.md(
        r"""
    The most intuitive flight controls that come to mind are the aircraft control surfaces.
    The pilot deflects control surfaces to change the distribution of  aerodynamic forces and moments acting on the aircraft body, and therefore influence the aircraft motion.

    > Examples:
    >
    > - The elevator is deflected to change the pitch angle of the aircraft, and therefore its angle of attack and lift coefficient.
    > - The ailerons are deflected to change the roll angle of the aircraft, and therefore its turn radius.
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
    In the scope of CAPO, the aircraft is a point-mass and therefore has no geometry or spatial extension. 
    It is therefore necessary to abstract the concept of flight controls away from the actual control surfaces, and towards a set of higher-level parameters.

    Because the ultimate effect of deflecting control surfaces is to change the magnitude and orientation of the external forces acting on the aircraft, in the scope of CAPO we can use the following set of Flight Controls:

    | <div style="width:120px">Flight Controls</div> | <div style="width:250px">**Description** | <div style="width:250px">**Bounds** |
    |:--|:--|:-----|
    | $\delta_T$ | Throttle | $[0, 1]$ |
    | $C_L$ | Lift coefficient  | $[0, C_{L_\mathrm{max}}]$ |
    | $\mu$ | Aerodynamic roll angle  | $[0, 90^\circ)$ |
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""In the same way as control surface deflections are bounded between an upper and lower limit, any chosen control variable should also be bounded to assume a confined set of reasonable values."""
    )
    return


@app.cell
def _():
    _defaults.nav_footer(
        "Problem_Formulation/FlightConstraints.py",
        "Flight Constraints",
        "Optimization_Methodology/PreambleMethodologies.py",
        "Preamble Methodologies",
    )
    return


if __name__ == "__main__":
    app.run()
