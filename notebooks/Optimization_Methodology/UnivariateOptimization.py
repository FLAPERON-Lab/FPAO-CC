import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))

    # Initialization code that runs before all other cells
    import marimo as mo

    # Import dependencies
    from core import _defaults
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    from core import aircraft as ac

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Data directory
    data_dir = str(
        mo.notebook_location().parent.parent / "data" / "AircraftDB_Standard.csv"
    )


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell
def _():
    mo.md(r"""
    # Univariate optimization

    In this case, we are going to assume that the drag coefficient $C_D$ is exclusively a function of the lift coefficient $C_L$ according to the very classic _parabolic drag polar_ model.

    $$ C_D = C_D(C_L) = C_{D_0} + K C_L^2 $$

    where $K$ and $C_{D_0}$ are constants defined by the aircraft geometry and aerodynamic characteristics.

    The aerodynamic efficiency is therefore also a function only of the lift coefficient:

    $$ E = \frac{C_L}{C_D(C_L)} = E(C_L) $$

    Its expression can be calculated explicitly very easily:

    $$ E = \frac{C_L}{C_D} = \frac{C_L}{C_{D_0} + K C_L^2} $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Optimization

    The optimization problem can then be formulated as:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Find the maximum aerodynamic efficiency by changing the lift coefficient within certain limits

    $$
    \begin{aligned}
        \max_{C_L} 
        & \quad E=\frac{C_L}{C_D}=\frac{C_L}{C_{D_0}+KC_L^2} \\
        % \text{subject to} 
        % & \quad \bm{c}_\mathrm{eq}(\bm{x},\bm{u}; \bm{p}) = 0 \\
        % & \quad \bm{c_\mathrm{ineq}}(\bm{x},\bm{u}; \bm{p}) \le 0 \\
        \text{for } 
        & \quad C_L \in [0, C_{L_\mathrm{max}}]
    \end{aligned}
    $$

    """).callout().center().style({"text-align": "center"})
    return


@app.cell
def _():
    mo.md(r"""
    $E$ is a non-monotonic continuous function of $C_L$, and therefore it may have extreme values in:

    - stationary points
    - boundary points

    We are going to have to calculate the values of $E$ (the objective function), in all of these points if we want to determine its maximum correctly.

    In the compact domain $[0, C_{L_\mathrm{max}}]$, the existence of extreme values of $E$ is actually guaranteed by the (Weierstrass) Extreme Value Theorem (also called the Min-Max theorem, in the case of univariate functions).
    So we know that, if we search exhaustively, we are going to find the maximum (and also the minimum) value of the aerodynamic efficiency.

    Because $E$ is a continuous function, there are no singular points. In general, they would also have to be evaluated.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Stationary values

    The necessary condition for an interior point to be stationary is given by the fact that the gradient of the objective function with respect to the decision variable is zero in that point.
    Therefore, interior stationary points can be found by equating the derivative of the objecitive function to zero.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $$ \frac{\mathrm{d}E}{\mathrm{d}C_L} = \frac{C_{D_0}-KC_L^2}{C_{D_0}+KC_L^2} = 0
    \quad \text{for} \quad C_L^* = C_{L_E} = \sqrt{\frac{C_{D_0}}{K}} $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It can be easily verified that this is indeed a maximum by looking at the convexity of the objective function.
    A sufficient condition to prove it is actually a maximum is to show that the second derivative is negative at the stationary point.
    For us, a graphical verification will be enough in this context.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The value of the total drag coefficient in this condition is twice the value of the zero-lift one, meaning that zero-lift and induced drag have the same contribution to total drag.

    $$ C_D^* = 2C_{D_0} $$

    The corresponding stationary value of the aerodynamic efficiency is therefore:

    $$ E^* = \frac{C_{L}^*}{C_{D_0}+KC_L^{*2}} = \frac{\sqrt{\frac{C_{D_0}}{K}}}{2C_{D_0}} = \sqrt{\frac{1}{4KC_{D_0}}}$$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Boundary values
    The values of the aerodynamic efficiency at the boundary of its domain are immediate to calculate:

    $$ E(C_L = 0) = 0$$

    $$ E(C_L = C_{L_\mathrm{max}}) = E_S = \frac{C_{L_\mathrm{max}}}{C_{D_0}+K C_{L_\mathrm{max}}^2}$$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Conclusion

    To determine the maximum aerodynamic efficiency, we therefore need to compare the values of the objective function in the stationary point and in the boundary points:

    - If $E^* > E_S$, the maximum aerodynamic efficiency is achieved at the stationary point $C_{L_E}$
    - If $E^* \le E_S$, the maximum aerodynamic efficiency is achieved at the boundary point $C_{L_\mathrm{max}}$

    It can be easily verified that the first inequality is verified in all cases, as long as $C_{L_\mathrm{max}} \ne C_{L_E}$.
    And in the equality case, the two maxima would coincide anyway.

    Therefore, the maximum aerodynamic efficiency is always achieved at the stationary point:

    $$ E_{\mathrm{max}} = E^* = \sqrt{\frac{1}{4KC_{D_0}}} \quad \text{for} \quad C_L = C_{L_E} = \sqrt{\frac{C_{D_0}}{K}} $$

    You can verify this graphically by selecting any aircraft in the following menu.
    """)
    return


@app.cell
def _():
    data = ac.available_aircrafts(data_dir).loc[
        :,
        [
            "full_name",
            "ID",
            "type",
            "b",
            "S",
            "CD0",
            "K",
            "CLmax_ld",
            "MTOM",
            "OEM",
        ],
    ]
    return (data,)


@app.cell(hide_code=True)
def _(data):
    # Database cell (1)

    ac_table = mo.ui.table(
        data=data,
        pagination=True,
        show_column_summaries=False,
        selection="single",
        initial_selection=[0],
        page_size=4,
        show_data_types=False,
    )

    ac_table
    return (ac_table,)


@app.cell
def _(ac_table, data):
    # Variables declared
    meshgrid_n = 101
    xy_lowerbound = -0.1

    CL_buffer = 1
    # Handle deselected row from table
    if ac_table.value is not None and ac_table.value.any().any():
        active_selection = ac_table.value.iloc[0]
    else:
        active_selection = data.iloc[0]

    CLmax = active_selection["CLmax_ld"]
    CD0 = active_selection["CD0"]
    K = active_selection["K"]

    CL_array = np.linspace(0, CLmax + CL_buffer, meshgrid_n)
    CD_array = CD0 + K * CL_array**2
    E_array = CL_array / (CD0 + K * CL_array**2)
    CL_E = np.sqrt(CD0 / K)
    CD_E = CD0 + K * CL_E**2
    E_max = CL_E / (CD0 + K * CL_E**2)
    E_max_line = CD_E / CL_E * CL_array
    return (
        CD_E,
        CD_array,
        CL_E,
        CL_array,
        CLmax,
        E_array,
        E_max,
        E_max_line,
        active_selection,
    )


@app.cell
def _(
    CD_E,
    CD_array,
    CL_E,
    CL_array,
    CLmax,
    E_array,
    E_max,
    E_max_line,
    active_selection,
):
    fig_endurance = make_subplots(
        rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    fig_endurance.add_traces(
        [
            go.Scatter(
                x=CL_array,
                y=CD_array,
                showlegend=True,
                name=r"$C_D$",
                line=dict(color="rgba(0, 114, 178, 1)"),
            ),
            go.Scatter(
                x=CL_array,
                y=E_max_line,
                showlegend=False,
                line=dict(color="rgba(230, 159, 0, 1)", dash="dot"),
            ),
            go.Scatter(
                x=[CL_E],
                y=[CD_E],
                line=dict(color="rgba(240, 228, 66, 1)"),
                mode="markers+text",
                text=[r"${E_{\mathrm{max}}}$"],
                showlegend=False,
                textposition="top left",
            ),
        ],
        cols=1,
        rows=1,
    )

    fig_endurance.add_traces(
        [
            go.Scatter(x=CL_array, y=E_array, name=r"$E$"),
            go.Scatter(
                x=[CL_E],
                y=[E_max],
                name=r"$E_{\mathrm{max}}$",
                marker=dict(color="rgba(240, 228, 66, 1)"),
                text=[r"${E_{\mathrm{max}}}$"],
                mode="markers+text",
                textposition="top left",
            ),
        ],
        cols=2,
        rows=1,
    )

    fig_endurance.add_vline(
        x=CLmax,
        line_dash="dot",
        line_color="red",
        annotation=dict(text=r"$C_{L_{\mathrm{max}}}$", xshift=10, yshift=-10),
        row=1,
        col=1,
    )

    fig_endurance.add_shape(
        type="line",
        x0=CL_E,
        x1=CL_E,
        y0=0,
        y1=CD_E,
        line_dash="dot",
        line_color="rgba(0, 158, 115, 1)",
        row=1,
        col=1,
    )

    fig_endurance.add_shape(
        type="line",
        x0=CL_E,
        x1=CL_E,
        y0=0,
        y1=E_max,
        line_dash="dot",
        line_color="rgba(0, 158, 115, 1)",
        row=1,
        col=2,
    )

    fig_endurance.add_annotation(
        x=CL_E - 0.1, y=0, text=r"$C_{L_E}$", showarrow=False, col=1, row=1
    )
    fig_endurance.add_annotation(
        x=CL_E - 0.1, y=0, text=r"$C_{L_E}$", showarrow=False, col=2, row=1
    )

    fig_endurance.add_annotation(
        x=CL_array[-10],
        y=E_max_line[-10] + 0.09,
        text=r"$\left(\frac{C_D}{C_L}\right)_{\mathrm{max}}$",
        showarrow=False,
        col=1,
        row=1,
    )

    fig_endurance.update_xaxes(title_text=r"$C_L \; (-)$", col=1, row=1)
    fig_endurance.update_yaxes(title_text=r"$C_D \; (-)$", col=1, row=1)
    fig_endurance.update_xaxes(title_text=r"$C_L \; (-)$", col=2, row=1)
    fig_endurance.update_yaxes(title_text=r"$E \; (-)$", col=2, row=1)

    fig_endurance.update_layout(
        title_text=active_selection["full_name"],
        title_x=0.5,
    )
    mo.output.clear()
    return (fig_endurance,)


@app.cell
def _(fig_endurance):
    fig_endurance
    return


@app.cell
def _():
    _defaults.nav_footer(
        "PreambleMethodologies.py",
        "Preamble Methodologies",
        "BivariateOptimization.py",
        "Bivariate Optimization",
    )
    return


if __name__ == "__main__":
    app.run()
