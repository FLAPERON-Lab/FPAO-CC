import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    from core import _defaults
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import numpy as np
    from core import aircraft as ac

    _defaults.FILEURL = _defaults.get_url()

    _defaults.set_plotly_template()
    data_dir = str(mo.notebook_location() / "public" / "AircraftDB_Standard.csv")


@app.cell
def _():
    _defaults.set_sidebar()
    return


@app.cell
def _():
    mo.md(r"""
    # Minimum airspeed
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Unconstrained optimization problem
    """)
    return


@app.cell
def _():
    mo.callout(
        mo.md(
            r"""
        Find the minimum airspeed by changing the lift coefficient and throttle within certain limits:

    $$
    \begin{aligned}
        \min_{C_L, \delta_T} 
        & \quad V \\
        % \text{subject to} 
        % & \quad \bm{c}_\mathrm{eq}(\bm{x},\bm{u}; \bm{p}) = 0 \\
        % & \quad \bm{c_\mathrm{ineq}}(\bm{x},\bm{u}; \bm{p}) \le 0 \\
        \text{for } 
        & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1]
    \end{aligned}
    $$
        """
        )
    ).center().style({"text-align": "center"})
    return


@app.cell
def _():
    mo.md(r"""
    This problem is ill posed, and it does not make sense to solve it.

    There is no functional relation between the objective function $V$ and the controls $C_L, \delta_T$.
    In other words, there is no equation that specifies how $V$ can change with respect to the controls.
    It does not make sense to optimize Flight Performance if the flight dynamics is not controlled.

    For example, the minimum airspeed achievable could be 0, if the aircraft is standing still on the runway.
    It could even be negative, if someone is pushing the aircraft back, or there is tailwind.

    A relation must be introduced with constraint equations, starting from the EoMs.
    These will define the problem properly.
    """)
    return


@app.cell(hide_code=True)
def _():
    data = ac.available_aircrafts(data_dir)

    ac_table = mo.ui.table(
        data=data,
        pagination=True,
        freeze_columns_left=["full_name"],
        show_column_summaries=False,
        selection="single",
        initial_selection=[0],
        page_size=4,
    )

    ac_table
    return (ac_table,)


@app.cell(hide_code=True)
def _(CL_maxld, CL_slider, ac_table, dT_slider):
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "Scatter"}, {"type": "Surface"}]]
    )

    if ac_table.value is not None and ac_table.value.any().any():
        title_text = str(ac_table.value.full_name.values[0])
    else:
        title_text = ""

    fig.data = []

    fig.add_trace(
        go.Scatter(
            x=[float(CL_slider.value)],
            y=[float(dT_slider.value)],
            showlegend=False,
            marker_color="#EF553B",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[float(CL_slider.value), float(CL_slider.value)],
            y=[float(dT_slider.value), float(dT_slider.value)],
            z=[0, 1],
            mode="lines",
            line=dict(color="#EF553B", width=4),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        xaxis=dict(title=r"C<sub>L</sub> (-)"),
        yaxis=dict(title=r"δ<sub>T</sub> (-)"),
        scene1=dict(
            xaxis=dict(title=r"C<sub>L</sub> (-)"),
            yaxis=dict(title=r"δ<sub>T</sub> (-)"),
            zaxis=dict(title=r"V (m/s)"),
        ),
    )
    fig.update_xaxes(range=[-0.5, CL_maxld], row=1, col=1)
    fig.update_yaxes(range=[-0.25, 1], row=1, col=1)
    fig.update_layout(
        scene1=dict(
            xaxis=dict(range=[-0.5, CL_maxld]),
            yaxis=dict(range=[-0.25, 1]),
            zaxis=dict(range=[0, 1]),
        )
    )
    fig.update_layout(
        title_text=title_text,
        title_x=0.5,
    )
    mo.output.clear()
    return (fig,)


@app.cell(hide_code=True)
def _(ac_table):
    if ac_table.value is not None and ac_table.value.any().any():
        CL_maxld = float(ac_table.value.CLmax_ld.values[0])
    else:
        CL_maxld = 3

    CL_slider = mo.ui.slider(start=0, stop=CL_maxld, step=0.1, label=r"$C_L$")

    dT_slider = mo.ui.slider(start=0, stop=1, step=0.05, label=r"$\delta_T$")
    return CL_maxld, CL_slider, dT_slider


@app.cell
def _(CL_slider, dT_slider, fig):
    mo.vstack([mo.hstack([CL_slider, dT_slider]), fig])
    return


@app.cell
def _():
    mo.md(r"""
    ## Constrained optimization problem
    """)
    return


@app.cell
def _():
    mo.callout(
        mo.md(r"""
        Find the minimum airspeed that can be maintained in Steady Level Flight by changing the lift coefficient and throttle within certain limits

    $$
    \begin{aligned}
        \min_{C_L, \delta_T} 
        & \quad V \\
        \text{subject to} 
        & \quad c_1^\mathrm{eq} = L-W = \frac{1}{2}\rho V^2 S C_L - W = 0 \\
        & \quad c_2^\mathrm{eq} = T-D = \delta_T T_a(V,h) - \frac{1}{2} \rho V^2 S (C_{D_0}+K C_L^2) =0 \\
        \text{for } 
        & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1]
    \end{aligned}
    $$
        """)
    ).center().style({"text-align": "center"})
    return


@app.cell
def _():
    mo.md(r"""
    The introduction of the constraints for vertical ($c_1^\mathrm{eq}$) and horizontal equilibrium ($c_2^\mathrm{eq}$) restricts the scope to only a certain type of optimal speeds we are looking for.

    The constraint equations introduce a functional dependency between the objective function and the controls.
    We are going to use them to reformulate the problem in order to analyse its properties.

    Before that, we notice that the expression of $c_2^\mathrm{eq}$ depends on the type of powertrain of the aircraft, and therefore we must proceed diffently for each powertrain architecture.

    1. [Simplified Jet -  Karush-Kuhn-Tucker Analyis](/?file=MinSpeed_Jet_KKT.py)
    2. [Simplified Piston-Prop -  Karush-Kuhn-Tucker Analysis](/?file=MinSpeed_Prop_KKT.py)
    """)
    return


@app.cell
def _():
    _defaults.nav_footer(
        "MinPower.py",
        "Minimum Power",
        "MaxSpeed.py",
        "Maximum Speed",
    )
    return


if __name__ == "__main__":
    app.run()
