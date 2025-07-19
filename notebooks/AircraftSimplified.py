import marimo

__generated_with = "0.13.8"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    from core import _defaults
    from pathlib import Path

    _defaults.FILEURL = _defaults.get_url()

    _defaults.set_plotly_template()
    data_dir = Path(mo.notebook_location()) / "public" / "AircraftDB_Standard.csv"


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
    mo.md(r"""# Visualizations""")
    return


@app.cell
def _():
    mo.md(r"""Here it is possible to select multiple aircrafts to visualise their thrust and power behaviour with respect to speed, visualising the standard assumptions mentioned above.""")
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


@app.cell(hide_code=True)
def _(ac):
    data = ac.available_aircrafts(data_dir).round(decimals=4)

    cols_4dec = [
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

    data[cols_4dec] = data[cols_4dec].round(4)

    other_cols = data.columns.difference(cols_4dec)
    data[other_cols] = data[other_cols].round(1)

    ac_table = mo.ui.table(
        data=data,
        pagination=True,
        freeze_columns_left=["full_name"],
        show_column_summaries=False,
    ).form(show_clear_button=True)
    return (ac_table,)


@app.cell
def _(ac_table, fig):
    aircraft_list = []

    fig.data = []
    if ac_table.value is not None and ac_table.value.any().any():
        aircraft_list = ac_table.value["ID"]

    ac_table
    return (aircraft_list,)


@app.cell
def _():
    fix_yaxis = mo.ui.checkbox(label="Fix the y-axis range", value=False)
    return (fix_yaxis,)


@app.cell
def _():
    mo.md("""In the following graph it is possible to fix the y-axis range by ticking the checkmark, this is useful to understand the behaviour of the different curves with the changing of the parameters. You can change the different parameters through the use of sliders.""")
    return


@app.cell
def _(ac, aircraft_list):
    fleet = {ID: ac.Aircraft(data_dir, ac_ID=ID) for ID in aircraft_list}
    return (fleet,)


@app.cell
def _(make_subplots):
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True)
    return (fig,)


@app.cell
def _(axis_limits):
    global axis_limits
    return


@app.cell
def _():
    if "axis_limits" not in globals():
        axis_limits = {"power": 0, "thrust": 0}
    return (axis_limits,)


@app.cell(hide_code=True)
def _(fix_yaxis):
    h_slider = mo.ui.slider(
        start=0,
        stop=20,
        label=r"Altitude (km)",
        value=10,
        show_value=True,
    )

    m_slider = mo.ui.slider(start=0, stop=1, step=0.1, label=r"", show_value=True)

    speed = mo.ui.dropdown(
        options=["CAS", "TAS", "EAS", "M"], value="CAS", label=r"Speed"
    )

    drag_condition = mo.ui.dropdown(
        options=["Cruise", "Landing", "Take Off"],
        value="Cruise",
        label="Flight Phase",
    )

    delta_t = mo.ui.slider(
        start=0,
        stop=1,
        label=r"$\delta_T$",
        show_value=True,
        step=0.1,
        value=0.5,
    )

    mass_stack = mo.hstack(
        [mo.md("**OEW**"), m_slider, mo.md("**MTOW**")],
        align="start",
        justify="start",
    )
    mo.vstack(
        [
            mo.hstack([h_slider, speed, delta_t]),
            mo.hstack([drag_condition, fix_yaxis.right()]),
        ]
    )
    return delta_t, drag_condition, h_slider, m_slider, mass_stack, speed


@app.cell
def _(drag_condition, mass_stack):
    show = None
    if drag_condition.value == "Cruise":
        show = mo.hstack([mass_stack.center()]).center()
    show
    return


@app.cell
def _(show_available, show_required):
    mo.hstack(["Select what to plot: ", show_required, show_available])
    return


@app.cell(hide_code=True)
def _(
    atmos,
    axis_limits,
    delta_t,
    drag_condition,
    fig,
    fix_yaxis,
    fleet,
    go,
    h_slider,
    m_slider,
    np,
    px,
    show_available,
    show_required,
    speed,
):
    fig.data = []
    TAS = np.linspace(30, 340, 250)

    h = h_slider.value * 1000
    rho = atmos.rho(h)
    p = atmos.p(h)
    rho0 = atmos.rho0
    p0 = atmos.p0

    qdyn = p * ((1.0 + rho * TAS * TAS / (7.0 * p)) ** 3.5 - 1.0)
    CAS = np.sqrt(7.0 * p0 / rho0 * ((qdyn / p0 + 1.0) ** (2.0 / 7.0) - 1.0))

    if speed.value == "CAS":
        x_axis = CAS
    elif speed.value == "TAS":
        x_axis = TAS
    elif speed.value == "EAS":
        x_axis = TAS * np.sqrt(rho / atmos.rho0)
    elif speed.value == "M":
        x_axis = TAS / atmos.a(h)

    colors = px.colors.qualitative.Vivid
    color_map_available = {
        id: colors[i % len(colors)] for i, id in enumerate(fleet.keys())
    }
    colors = px.colors.qualitative.Safe
    color_map_required = {
        id: colors[i % len(colors)] for i, id in enumerate(fleet.keys())
    }

    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            showlegend=False,
            line=dict(color="rgba(0,0,0,0)"),
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

    for index, (id, obj) in enumerate(fleet.items()):
        full_name = str(obj.ac_data["full_name"].values[0])
        if show_available.value:
            power_value = obj.power(V=TAS, h=h, deltaT=delta_t.value)[1]

            thrust_value = obj.thrust(V=TAS, h=h, deltaT=delta_t.value)[1]

            yaxis1 = max(yaxis1, max(power_value))
            yaxis2 = max(yaxis2, max(thrust_value))

            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=power_value,
                    mode="lines",
                    legendgroup="Available",
                    legendgrouptitle_text="Available",
                    name=full_name,
                    line=dict(width=2, color=color_map_available[id]),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=thrust_value,
                    mode="lines",
                    legendgroup="Available",
                    line=dict(width=2, color=color_map_available[id]),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
        if show_required.value:
            mass = (
                obj.ac_data["OEM"].values
                + (obj.ac_data["MTOM"].values - obj.ac_data["OEM"].values)
                * m_slider.value
            )
            if drag_condition.value == "Cruise":
                CL = (
                    (mass * 9.80665 / obj.ac_data["S"].values)
                    * (2 / atmos.rho(h))
                    * 1
                    / (TAS**2)
                )
            elif drag_condition.value == "Take Off":
                CL = obj.ac_data["CLmax_to"].values
            elif drag_condition.value == "Landing":
                CL = obj.ac_data["CLmax_ld"].values

            CD = obj.drag_polar(CL=CL)

            drag = CD * 0.5 * atmos.rho(h) * TAS**2 * obj.ac_data["S"].values / 1e3

            power_required = drag * TAS

            yaxis1 = max(yaxis1, max(power_required))
            yaxis2 = max(yaxis2, max(drag))

            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=power_required,
                    mode="lines",
                    legendgrouptitle_text="Required",
                    legendgroup="Required",
                    name=full_name,
                    line=dict(width=2, color=color_map_required[id]),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=drag,
                    mode="lines",
                    legendgroup="Required",
                    name=id,
                    line=dict(width=2, color=color_map_required[id]),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

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
    ).update_xaxes(title="Velocity (m/s)", range=[0, max(x_axis)])

    fig
    return


@app.cell
def _():
    mo.md(
        r"""The assumptions that come with using **simplified** aero-propulsive models inherently bring unrealistic estimations near stall speed and maximum operating speed! 

        Asymptotic behaviour in the region of zero velocity has in fact no physical meaning, however, as mentioned previously, these assumptions keep the flight performance optimization derivations manageable.""",
    ).callout(kind="warn")
    return


@app.cell
def _():
    show_required = mo.ui.checkbox(label="Required")
    show_available = mo.ui.checkbox(label="Available", value=True)
    return show_available, show_required


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
    from core import atmos
    import polars as pl

    return ac, atmos, go, make_subplots, np, px


if __name__ == "__main__":
    app.run()
