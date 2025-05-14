import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import _defaults

    _defaults.set_plotly_template()


@app.cell
def _():
    _defaults.set_sidebar()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Simplified Aircraft Models

    Using simplified aero-propulsive models to characterize the performance of an aircraft keeps the analytical derivations manageable and preserves their didactic value.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Assumptions

    These are standard assumptions in the field of FPAO.

    |<div style="width:250px">Assumption</div> | <div style="width:150px">Jet aircraft</div> | <div style="width:150px">Propeller aircraft</div> |
    |:-|:----------|:----------|
    | Parabolic drag polar | $C_D = C_{D0} + K C_L^2$ | $C_D = C_{D_0} + K C_L^2$ |  
    | Proportional throttle command | $T=\delta_T T_a$ | $P=\delta_T P_a$ |
    | Thrust/power lapse  | $T_a(h) = T_{a_0}\sigma^\beta$ | $P_a(h) = P_{a_0}\sigma^\beta$ |
    | Available power | $P_a =TV$ | $P_a(V)=\mathit{const}$ |
    | Available thrust | $T_a(V)=\mathit{const}$ | $\displaystyle{T_a = \frac{P_a}{V}}$ |
    | Power-Specific Fuel Consumption| | $c_{P}=\mathit{const}$ |
    | Thrust-Specific Fuel Consumption| $c_{T}=\mathit{const}$ | |
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Visualization""")
    return


@app.cell
def _():
    _defaults.nav_footer(
        "Atmosphere.py", "Atmosphere", "AircraftCustom.py", "Custom Aircraft Models"
    )
    return


if __name__ == "__main__":
    app.run()
