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
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
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
    ).center()
    return


@app.cell
def _():
    mo.md(r"# Visualizations").center()
    return


@app.cell
def _():
    mo.md("""Single selection allows to visualise only one aircraft, while in the multiple selection tab a database is provided to allow selection of multiple aircrafts.""")
    return


@app.cell
def _():
    mo.md(
        r"""
    /// admonition | Heads up!

     Don't forget to press **submit** in the "multiple selection" tab! or **clear** if you want to erase all the lines.
    ///
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
    return (ac_name_dropdown,)


@app.cell
def _(ac_name_dropdown, ac_type_dropdown):
    single_selection_ui = mo.hstack(
        [
            mo.md("Select the aero-propulsive model type:"),
            ac_type_dropdown,
            mo.md("Select the corresponding aircraft:"),
            ac_name_dropdown,
        ]
    )
    return (single_selection_ui,)


@app.cell
def _(ac):
    data = ac.available_aircrafts()

    ac_table = mo.ui.table(
        data=data,
        pagination=True,
        freeze_columns_left=["full_name"],
        show_column_summaries=False,
    ).form(show_clear_button=True)
    return ac_table, data


@app.cell
def _():
    tabs = mo.ui.tabs(
        {
            "Single Selection": "",
            "Multiple Selection": "",
        }
    )
    return (tabs,)


@app.cell
def _(tabs):
    tabs.center()
    return


@app.cell
def _(ac_name_dropdown, ac_table, data, fig, go, single_selection_ui, tabs):
    aircraft_list = []
    if tabs.value == "Single Selection":
        fig.data = []
        show = single_selection_ui
        aircraft_list = data[data["full_name"] == ac_name_dropdown.value][
            "ID"
        ].values.tolist()

    elif tabs.value == "Multiple Selection":
        fig.data = []
        show = ac_table
        if ac_table.value is not None and ac_table.value.any().any():
            aircraft_list = ac_table.value["ID"]

        else:
            aircraft_list = []
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(0,0,0,0)"),  # Transparent line
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(0,0,0,0)"),
            ),
            row=1,
            col=2,
        )

    show
    return (aircraft_list,)


@app.cell
def _():
    fix_yaxis = mo.ui.checkbox(label="Fix the y-axis range", value=False)
    return (fix_yaxis,)


@app.cell
def _():
    mo.md("""In the following graph it is possible to fix the y-axis range by ticking the checkmark, this is useful to understand the behaviour of the different curves with the changing of the parameters. You can change the different parameters at the bottom of the graphs, through the use of sliders.""")
    return


@app.cell
def _(fix_yaxis):
    fix_yaxis.right()
    return


@app.cell
def _(ac, aircraft_list):
    fleet = {ID: ac.Aircraft(ac_ID=ID) for ID in aircraft_list}
    return (fleet,)


@app.cell
def _(make_subplots):
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True)
    return (fig,)


@app.cell
def _():
    if "axis_limits" not in globals():
        axis_limits = {"power": 0, "thrust": 0}
    return (axis_limits,)


@app.cell
def _(
    atmos,
    axis_limits,
    delta_t,
    fig,
    fix_yaxis,
    fleet,
    go,
    h_slider,
    np,
    px,
    show_available,
    show_required,
):
    global axis_limits
    fig.data = []
    velocities = np.linspace(15, 200, 250)

    colors = px.colors.qualitative.Vivid
    color_map = {id: colors[i % len(colors)] for i, id in enumerate(fleet.keys())}

    h = h_slider.value * 1000

    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            showlegend=False,
            line=dict(color="rgba(0,0,0,0)"),  # Transparent line
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            showlegend=False,
            line=dict(color="rgba(0,0,0,0)"),
        ),
        row=1,
        col=2,
    )

    yaxis1 = 0
    yaxis2 = 0
    print(delta_t.value)
    for index, (id, obj) in enumerate(fleet.items()):
        if show_available.value:
            power_value = obj.power(V=velocities, beta=0.85, h=h, deltaT=delta_t.value)[1]

            thrust_value = obj.thrust(V=velocities, beta=0.85, h=h, deltaT=delta_t.value)[1]

            yaxis1 = max(yaxis1, max(power_value))
            yaxis2 = max(yaxis2, max(thrust_value))

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
                    legendgroup=(r"$T_a$" + id),
                    line=dict(width=2, color=color_map[id]),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
        if show_required.value:

            CL = (obj.ac_data["MTOM"].values / obj.ac_data["S"].values) * (2 / atmos.rho(h)) * 1 / (velocities**2)
            cd = obj.drag_polar(CL=CL)

            drag = cd * 0.5 * atmos.rho(h) * velocities**2 * obj.ac_data["S"].values / 1e3

            power_required = drag * velocities / 1e3

            fig.add_trace(
                go.Scatter(
                    x=velocities,
                    y=power_required,
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
                    y=drag,
                    mode="lines",
                    legendgroup=id,
                    name=id,
                    line=dict(width=2, color=color_map[id]),
                    showlegend=True,
                ),
                row=1,
                col=2,
            )

    # Update axis_limits only if not fixing y-axis
    if not fix_yaxis.value:
        axis_limits["power"] = yaxis1
        axis_limits["thrust"] = yaxis2

    fig.update_yaxes(
        title="Power (kW)",
        row=1,
        col=1,
        range=[0, axis_limits["power"]] if fix_yaxis.value else None,
    ).update_yaxes(
        title="Thrust (kN)",
        row=1,
        col=2,
        range=[0, axis_limits["thrust"]] if fix_yaxis.value else None,
    ).update_xaxes(title="Velocity (m/s)")

    fig
    return


@app.cell
def _():
    show_required = mo.ui.checkbox(label="Required")
    show_available = mo.ui.checkbox(label="Available")
    return show_available, show_required


@app.cell
def _(show_available, show_required):
    mo.hstack(["Select what to plot: ", show_required, show_available]).left()
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

    speed = mo.ui.dropdown(
        options=["TAS", "EAS", "M", "CAS"], value="TAS", label=r"Speed"
    )

    delta_t = mo.ui.slider(start = 0, stop= 1, label= r"$\delta_T$", show_value= True, step  = 0.1)

    mo.hstack([h_slider, speed, delta_t])
    return delta_t, h_slider


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
    return ac, atmos, go, make_subplots, np, px


if __name__ == "__main__":
    app.run()
