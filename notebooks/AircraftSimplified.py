

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import _defaults as defs

    defs.set_plotly_template()
    mo.sidebar(
        defs.sidebar,
        width="300px",
        # footer=mo.md(""),
    )
    return defs, mo


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
    from core import aircraft as ac

    ac_type_dropdown = mo.ui.dropdown(
        options=["Simplified Jet", "Simplified Propeller"], value="Simplified Jet"
    )
    return ac, ac_type_dropdown


@app.cell
def _(ac, ac_type_dropdown, mo):
    availables = ac.available_aircrafts(ac_type=ac_type_dropdown.value)

    ac_name_dropdown = mo.ui.dropdown(options=availables, value=availables[0])

    mo.hstack(
        [
            mo.md("Select the aero-propulsive model type:"),
            ac_type_dropdown,
            mo.md("Select the corresponding aircraft:"),
            ac_name_dropdown,
        ]
    )
    return (ac_name_dropdown,)


@app.cell
def _(ac, ac_name_dropdown, ac_type_dropdown):
    aircraft = ac.Aircraft(
        ac_type=ac_type_dropdown.value, ac_name=ac_name_dropdown.value
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Visualization""")
    return


@app.cell
def _():
    return


@app.cell
def _(defs, mo):
    nav_foot = mo.nav_menu(
        {
            f"{defs._fileurl}Atmosphere.py": f"{mo.icon('lucide:arrow-big-left')} Atmosphere",
            f"{defs._fileurl}AircraftCustom.py": f"Custom Aircraft Models {mo.icon('lucide:arrow-big-right')}",
        }
    ).center()
    nav_foot
    return


if __name__ == "__main__":
    app.run()
