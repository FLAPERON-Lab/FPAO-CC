import marimo

__generated_with = "0.13.4"
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
def _():
    mo.md(
        r"""
    # Simplified Aircraft Models

    Using simplified aero-propulsive models to characterize the performance of an aircraft keeps the analytical derivations manageable and preserves their didactic value.
    """
    )
    return


@app.cell
def _():
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
def _():
    ac_type_dropdown = mo.ui.dropdown(
        options=["Simplified Jet", "Simplified Propeller"], value="Simplified Jet"
    )
    return (ac_type_dropdown,)


@app.cell
def _(ac, ac_type_dropdown):
    availables = ac.available_aircrafts(ac_type=ac_type_dropdown.value)[
        "full_name"
    ].values

    ac_name_dropdown = mo.ui.dropdown(options=availables, value=availables[0])

    mo.hstack(
        [
            mo.md("Select the aero-propulsive model type:"),
            ac_type_dropdown,
            mo.md("Select the corresponding aircraft:"),
            ac_name_dropdown,
        ]
    )
    return


@app.cell
def _(atmos, h_slider, np, pd):
    h = np.array([h_slider.value * 1000])

    atmos_data = pd.DataFrame(
        {
            "h": h,
            "T": atmos.T(h),
            "p": atmos.p(h),
            "rho": atmos.rho(h),
            "a": atmos.a(h),
        }
    )
    return


@app.cell
def _():
    mo.md(r"""## Visualization""")
    return


@app.cell
def _(ac):
    ac_table = mo.ui.table(data=ac.available_aircrafts())
    return (ac_table,)


@app.cell
def _(ac_table):
    ac_table
    return


@app.cell
def _(ac, ac_table):
    aircraft_list = ac_table.value["ID"].tolist()

    fleet = {ID: ac.Aircraft(ac_ID=ID) for ID in aircraft_list}
    return (fleet,)


@app.cell
def _(make_subplots):
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True)
    return (fig,)


@app.cell
def _(fig, fleet, go, h_slider, np, px):
    fig.data = []
    velocities = np.linspace(0, 200, 250)

    colors = px.colors.qualitative.Vivid
    color_map = {id: colors[i % len(colors)] for i, id in enumerate(fleet.keys())}

    for index, (id, obj) in enumerate(fleet.items()):
        power_value = obj.power(
            V=velocities, beta=0.85, h=h_slider.value * 1000, deltaT=0.5
        )[0]
        thrust_value = obj.thrust(
            V=velocities, beta=0.85, h=h_slider.value * 1000, deltaT=0.5
        )[0]

        fig.add_trace(
            go.Scatter(
                x=velocities,
                y=power_value,
                mode="lines",
                legendgroup=id,
                name=id,
                line=dict(width=2, color=color_map[id]),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=velocities,
                y=thrust_value,
                mode="lines",
                legendgroup=id,
                line=dict(width=2, color=color_map[id]),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_yaxes(
        title="Power (kW)",
        row=1,
        col=1,
    ).update_yaxes(
        title="Thrust (kN)",
        row=1,
        col=2,
    ).update_xaxes(title="Velocity (m/s)")

    fig
    return


@app.cell
def _():
    h_slider = mo.ui.slider(
        start=0,
        stop=14,
        label=r"Altitude (km)",
        value=10,
        show_value=True,
    )

    h_slider
    return (h_slider,)


@app.cell
def _():
    _defaults.nav_footer(
        "Atmosphere.py", "Atmosphere", "AircraftCustom.py", "Custom Aircraft Models"
    )
    return


@app.cell
def _():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import numpy as np
    from core import aircraft as ac
    import pandas as pd
    from core import atmos
    return ac, atmos, go, make_subplots, np, pd, px


if __name__ == "__main__":
    app.run()
