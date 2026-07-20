# SPDX-FileCopyrightText: 2026 Carmine Varriale <C.varriale@tudelft.nl>
# SPDX-FileCopyrightText: 2026 Federico Angioni <F.angioni@student.tudelft.nl>
# SPDX-FileCopyrightText: 2026 Maarten van Hoven <M.B.vanHoven@tudelft.nl>
#
# SPDX-License-Identifier: Apache-2.0

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")

with app.setup:
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
    mo.md("""
    # Nomenclature
    """)
    return


@app.cell
def _():
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(r"""## Atmospheric and flight state """),
                    mo.md(r"""
            | Symbol | Description | Units |
            |--------|-------------|-------|
            | $h$ | Altitude | m |
            | $\rho$ | Air density | kg/m³ |
            | $\rho_0$ | Sea-level air density | kg/m³ |
            | $\sigma$ | Density ratio | — |
            | $V$ | True airspeed | m/s |
            """),
                ]
            ),
            mo.vstack(
                [
                    mo.md(r"""## Forces and power"""),
                    mo.md(r"""        
                | Symbol | Description | Units |
                |--------|-------------|-------|
                | $L$ | Lift force | N |
                | $D$ | Drag force | N |
                | $T$ | Thrust force | N |
                | $W$ | Weight | N |
                | $P$ | Power | W |
                """),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(r"""## Aerodynamic model"""),
                    mo.md(r"""
                | Symbol | Description | Units |
                |--------|-------------|-------|
                | $C_L$ | Lift coefficient | — |
                | $C_D$ | Drag coefficient | — |
                | $C_{D_0}$ | Zero-lift drag coefficient | — |
                | $K$ | Induced drag factor | — |
                | $C_{L_\mathrm{max}}$ | Max. lift coefficient | — |
                | $C_{L_E}$ | Lift coefficient for max. aerodynamic efficiency | — |
                | $C_{L_P}$ | Lift coefficient for min. required power | — |
                | $E$ | Aerodynamic efficiency (lift-to-drag ratio) | — |
                | $E_{\mathrm{max}}$ | Max. aerodynamic efficiency | — |
                | $E_P$ | Aerodynamic efficiency at min. required power | — |
                | $E_S$ | Aerodynamic efficiency at stall | — |
                """),
                ]
            ),
            mo.vstack(
                [
                    mo.md(r"""## Propulsion model"""),
                    mo.md("""
                | Symbol | Description | Units |
                |--------|-------------|-------|
                | $\delta_T$ | Throttle setting | — |
                | $T_{a0}$ | Sea-level available thrust | N |
                | $P_{a0}$ | Sea-level available shaft power | kW |
                | $\\beta$ | Thrust / power lapse rate exponent | — |
                """),
                    mo.md(r"""<br> """),
                    mo.md(r"""## Aircraft geometry"""),
                    mo.md("""
                | Symbol | Description | Units |
                |--------|-------------|-------|
                | $S$ | Wing reference area | m² |
                """),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(r"""## Special airspeeds"""),
                    mo.md(r"""
                | Symbol | Description | Units |
                |--------|-------------|-------|
                | $V_s$ | Stall speed | m/s |
                | $V_{s0}$ | Sea-level stall speed | m/s |
                """),
                ]
            ),
            mo.vstack(
                [
                    mo.md(r"""## Optimization"""),
                    mo.md(r"""
                | Symbol | Description | Units |
                |--------|-------------|-------|
                | $\mathcal{L}$ | Lagrangian function | — |
                | $\lambda_i$ | Lagrange multipliers (equality constraints) | — |
                | $\mu_j$ | KKT multipliers (inequality constraints) | — |
                """),
                ]
            ),
        ]
    )
    return


@app.cell
def _():
    _defaults.nav_footer(
        "Scope.py", "Scope", "Models_Library/Atmosphere.py", "Atmosphere"
    )
    return


if __name__ == "__main__":
    app.run()
