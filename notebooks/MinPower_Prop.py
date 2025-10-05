import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo

    # Import dependencies
    from core import _defaults
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    from core import atmos
    from core import aircraft as ac
    from core.aircraft import (
        velocity,
        horizontal_constraint,
        power,
        drag,
        endurance,
    )

    # Set local/online filepath
    _defaults.FILEURL = _defaults.get_url()

    # Plotly dark mode template
    _defaults.set_plotly_template()

    # Data directory
    data_dir = str(mo.notebook_location() / "public" / "AircraftDB_Standard.csv")


@app.cell(hide_code=True)
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell(hide_code=True)
def _(active_selection, h_slider, m_slider):
    # Variables declared
    meshgrid_n = 101
    xy_lowerbound = -0.1

    CL_array = np.linspace(0, active_selection["CLmax_ld"], meshgrid_n)  # -
    dT_array = np.linspace(0, 1, meshgrid_n)  # -
    h_array = np.linspace(0, 20e3, meshgrid_n)  # meters
    # Retrieve selected values
    # Compute selected weight
    W_selected = (
        active_selection["OEM"]
        + (active_selection["MTOM"] - active_selection["OEM"]) * m_slider.value
    ) * atmos.g0  # Netwons

    h_selected = int(h_slider.value * 1e3)  # meters
    step = h_array[1] - h_array[0]  # here it's 200
    idx_selected = int((h_selected - h_array[0]) / step)

    a = atmos.a(h_selected)
    a_harray = atmos.a(h_array)
    CD0 = active_selection["CD0"]
    S = active_selection["S"]
    K = active_selection["K"]
    CLmax = active_selection["CLmax_ld"]
    Pa0 = active_selection["Pa0"] * 1e3  # Watts
    beta = active_selection["beta"]
    CL_P = np.sqrt(3 * CD0 / K)
    CL_E = np.sqrt(CD0 / K)
    E_max = endurance(K, CD0, "max")
    E_P = (np.sqrt(3) / 2) * E_max
    E_S = CLmax / (CD0 + K * CLmax)
    velocity_stall_harray = velocity(W_selected, h_array, CLmax, S)
    return (
        CD0,
        CL_P,
        CL_array,
        CLmax,
        E_P,
        E_S,
        K,
        Pa0,
        S,
        W_selected,
        a_harray,
        beta,
        dT_array,
        h_array,
        h_selected,
        idx_selected,
        velocity_stall_harray,
        xy_lowerbound,
    )


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Minimum Power Required: simplified jet aircraft

    $$
    \begin{aligned}
        \min_{C_L, \delta_T} 
        & \quad P = DV = \frac{1}{2}\rho V^2S(C_{D_0}+KC_L^2)V \\
        \text{subject to} 
        & \quad c_1^\mathrm{eq} = L-W = \frac{1}{2}\rho V^2 S C_L - W = 0 \\
        & \quad c_2^\mathrm{eq} = T-D = \delta_T T_a(V,h) - \frac{1}{2} \rho V^2 S (C_{D_0}+K C_L^2) =0 \\
        \text{for } 
        & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1] \\
        \text{with } 
        & \quad T_a(V,h) = \frac{P_a(h)}{V} = \frac{P_{a0}\sigma^\beta}{V} \\
    \end{aligned}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    # Database cell (1)

    data = ac.available_aircrafts(data_dir, ac_type="Propeller")

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
    return ac_table, data


@app.cell(hide_code=True)
def _(ac_table, data):
    # Interactive elements

    # Handle deselected row from table
    if ac_table.value is not None and ac_table.value.any().any():
        active_selection = ac_table.value.iloc[0]
    else:
        active_selection = data.iloc[0]

    # Interactive CL and \delta_T sliders
    CL_slider = mo.ui.slider(
        start=0,
        stop=active_selection["CLmax_ld"],
        step=0.2,
        label=r"$C_L$",
        value=0.5,
    )

    dT_slider = mo.ui.slider(start=0, stop=1, step=0.1, label=r"$\delta_T$", value=0.5)

    m_slider = mo.ui.slider(start=0, stop=1, step=0.1, label=r"", show_value=True)

    h_slider = h_slider = mo.ui.slider(
        start=0,
        stop=20,
        label=r"Altitude (km)",
        value=10,
        show_value=True,
    )

    # Create stacks
    mass_stack = mo.hstack(
        [mo.md("**OEW**"), m_slider, mo.md("**MTOW**")],
        align="start",
        justify="start",
    )

    variables_stack = mo.hstack([mass_stack, h_slider])
    return (
        CL_slider,
        active_selection,
        dT_slider,
        h_slider,
        m_slider,
        mass_stack,
        variables_stack,
    )


@app.cell
def _(CD0, CL_array, CL_slider, K, Pa0, S, W_selected, beta, h_selected):
    velocity_CLarray = velocity(W_selected, h_selected, CL_array, S, False)

    velocity_CLarray = np.where(
        np.isnan(velocity_CLarray), np.nanmax(velocity_CLarray), velocity_CLarray
    )

    velocity_CLarray_capped = velocity(W_selected, h_selected, CL_array, S, True)

    velocity_user_selected = velocity(W_selected, h_selected, CL_slider.value, S)
    power_user_selected = power(
        h_selected,
        S,
        CD0,
        K,
        CL_slider.value,
        velocity_user_selected,
    )

    drag_curve = drag(
        h_selected,
        S,
        CD0,
        K,
        CL_array,
        velocity_CLarray,
    )

    power_curve_a_capped = power(
        h_selected, S, CD0, K, CL_array, velocity_CLarray_capped
    )

    drag_curve_a_capped = drag(
        h_selected,
        S,
        CD0,
        K,
        CL_array,
        velocity_CLarray_capped,
    )

    constraint = horizontal_constraint(
        W_selected,
        h_selected,
        CD0,
        K,
        CL_array,
        Pa0,
        beta,
        V=velocity_CLarray,
        S=S,
        D=drag_curve,
        type="propeller",
    )

    power_curve = np.where(
        ~np.isnan(constraint),
        power(h_selected, S, CD0, K, CL_array, velocity_CLarray),
        np.nan,
    )

    power_surface = np.tile(power_curve, (len(CL_array), 1))

    max_colorbar = np.nanmin(power_curve_a_capped) * 1.5 / 1e3

    min_colorbar = np.nanmin(power_curve_a_capped) / 1e3
    return (
        constraint,
        max_colorbar,
        min_colorbar,
        power_surface,
        power_user_selected,
    )


@app.cell
def _(
    CL_array,
    CL_slider,
    active_selection,
    constraint,
    dT_array,
    dT_slider,
    max_colorbar,
    min_colorbar,
    power_surface,
    power_user_selected,
    xy_lowerbound,
):
    # Create go.Figure() object
    fig_initial = go.Figure()

    # Minimum velocity surface
    fig_initial.add_traces(
        [
            go.Surface(
                x=CL_array,
                y=dT_array,
                z=power_surface / 1e3,
                opacity=0.9,
                name="Power",
                colorscale="viridis",
                cmax=max_colorbar,
                cmin=min_colorbar,
                colorbar={"title": "Power (kW)"},
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint,
                z=power_surface[0] / 1e3,
                opacity=1,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[-15]],
                y=[constraint[-15]],
                z=[power_surface[0, -15] / 1e3 + 250],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
                textfont=dict(size=14, family="Arial"),
            ),
            go.Scatter3d(
                x=[CL_slider.value],
                y=[dT_slider.value],
                z=[power_user_selected / 1e3],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="Design Point",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>P: %{z}<extra>%{fullData.name}</extra>",
            ),
        ]
    )

    camera = dict(eye=dict(x=1.35, y=1.35, z=1.35))

    fig_initial.update_layout(
        scene=dict(
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[xy_lowerbound, active_selection["CLmax_ld"]],
            ),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(title="P (kW)", range=[0, max_colorbar]),
        ),
    )

    fig_initial.update_layout(
        scene_camera=camera,
        title={
            "text": f"Minimum power domain for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    mo.output.clear()
    return (fig_initial,)


@app.cell(hide_code=True)
def _(CL_slider, dT_slider):
    mo.md(rf"""Here you can modify the control variables to understand how it affects the design: {mo.hstack([dT_slider, CL_slider])}""")
    return


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_initial):
    fig_initial
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## KKT formulation
    To be reconducted in the standard KKT analysis format, the objective function is expressed in terms of the controls by direct elimination of $c_1^\mathrm{eq}$. The velocity $V$ can be expressed as: 

    $$
    V = \sqrt{\frac{2}{\rho}\frac{W}{S}\frac{1}{C_L}}
    $$

    Moreover, in previous analyses we found $\delta_T=C_L=0$ does not correspond to a sensible solution, thus we can write:

    $$
    0\lt \delta_T \le 1 \quad \land \quad  0\lt C_L\le C_{L_{\mathrm{max}}}
    $$

    Notice the open interval in the lower bounds.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    The KKT formulation can now be written: 

    $$
    \begin{aligned}
        \min_{C_L, \delta_T} 
        & \quad P = DV = W \left(\frac{C_{D_0} +K C_L^2}{C_L}\right)\sqrt{\frac{2}{\rho}\frac{W}{S}\frac{1}{C_L}}=\sqrt{\frac{2W^3}{\rho S}}\left(\frac{C_{D_0}+K C_L^2}{C_L^{3/2}}\right) = \sqrt{\frac{2W^3}{\rho S}}\left(C_{D_0} C_L^{-3/2}+K C_L^{1/2}\right)\\
        \text{subject to} 
        & \quad g_1 = T - W\, \frac{1}{E}  =\frac{\delta_T P_{a0}\sigma^\beta}{V} - W\frac{C_{D_0} + K C_L^2}{C_L} = 0 \quad \Rightarrow \quad \delta_T P_{a0}\sigma^\beta - \sqrt{\frac{2W^{3}}{\rho S}} \left(C_{D_0} C_L^{-3/2}+K C_L^{1/2}\right) = 0\\
        & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
        & \quad h_2 = \delta_T - 1 \le 0 \\
    \end{aligned}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Lagrangian function and KKT conditions

    The Lagrangian function combines the objective function with eqaulity constraints using Lagrange multipliers ($\lambda_i$) and inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(C_L, \delta_T, \lambda_1, \mu_1, \mu_2) = & P + \lambda_1 \left[T - D\right]+ \mu_1 (C_L - C_{L_\mathrm{max}}) +\mu_2 (\delta_T - 1)\\ 
    =&\quad \sqrt{\frac{2W^3}{\rho S}}\left(C_{D_0} C_L^{-3/2}+K C_L^{1/2}\right)(1 - \lambda_1) +\\
    & + \lambda_1 \delta_T P_{a0}\sigma^\beta \\
    & + \mu_1 (C_L - C_{L_\mathrm{max}}) + \\
    & + \mu_2 (\delta_T - 1) +\\
    \end{aligned}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    A necessary condition for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist, the multipliers $\lambda_1, \mu_1, \mu_2$ have to meet the following conditions:

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = \sqrt{\frac{2W^3}{\rho S}}\left(-\frac{3}{2}C_{D_0}C_L^{-5/2} + \frac{1}{2} K C_L^{-1/2}\right)(1-\lambda_1) + \mu_1= 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 P_{a0}\sigma^\beta+ \mu_2= 0$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T P_{a0}\sigma^\beta - \sqrt{\frac{2W^{3}}{\rho S}} \left(C_{D_0} C_L^{-3/2}+K C_L^{1/2}\right) = 0$
    4.  $C_L - C_{L_\mathrm{max}} \le 0$
    5.  $\delta_T - 1 \le 0$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    6.  $\mu_1, \mu_2 \ge 0$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    7.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    8. $\mu_2 (\delta_T - 1) = 0$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## KKT analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active 
    or inactive.

    ### _Interior solutions_ 

    In this case: $C_L \lt C_{L_{\mathrm{max}}}$, $\delta_T \lt 1$, $\mu_1=\mu_2= 0$

    from stationarity condition (2): $\lambda_1 = 0$

    from stationarity condition (1): 

    $$
    -\frac{3}{2}C_{D_0} C_L^{-5/2}+\frac{1}{2}KC_L^{-1/2}= 0 \quad \Rightarrow \quad KC_L^2 = 3C_{D_0} \quad \Rightarrow \quad C_L^* = \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Before finding the corresponding $\delta_T$ value find the velocity associated with $C_{L_P}$:

    $$
    V_P = \sqrt{\frac{2W}{\rho S}}\sqrt[4]{\frac{K}{3C_{D_0}}}
    $$


    The optimum $\delta_T$ value is obtained from primal feasibility constraint (3), using the velocity for minimum power we find:

    $$
    \delta_T^* = \frac{W}{E_P}\frac{V_P}{P_{a0}\sigma^\beta}
    $$

    Where: $\displaystyle E_{\mathrm{P}} = \frac{\sqrt{3}}{2}E_{\mathrm{max}}$

    This is valid for:  

    $$
    \delta_T^*\lt 1 \Leftrightarrow \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt P_{a0} E_P \sqrt{\frac{\rho_0SC_{L_P}}{2}} \;  = P_{a0} E_P \sqrt{\frac{\rho_0 S}{2}} \sqrt[4]{\frac{3 C_{D_0}}{K}}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Finally, the optimum for the interior of the domain is thus:

    $$
    \boxed{C_L^* = \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E}} \quad \land \quad \boxed{\delta_T^* = \frac{W}{E_P}\frac{V_P}{P_{a0}\sigma^\beta}} \qquad \mathrm{with} \quad V_P = \sqrt{\frac{2W}{\rho S}}\sqrt[4]{\frac{K}{3C_{D_0}}} \qquad \mathrm{for} \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt P_{a0} E_P \sqrt{\frac{\rho_0 S}{2}} \sqrt[4]{\frac{3 C_{D_0}}{K}}
    $$
    """
    )
    return


@app.cell
def _(S):
    def interior_condition(W, h, beta, CD0, K, Pa0, E_P):
        sigma = atmos.rhoratio(h)
        CL_P = np.sqrt(3 * CD0 / K)
        condition = ((W ** (1.5)) / (sigma ** (beta + 0.5))) < np.sqrt(
            0.5 * atmos.rho0 * S * CL_P
        ) * Pa0 * E_P

        return condition
    return (interior_condition,)


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell(hide_code=True)
def _(
    CD0,
    CL_P,
    E_P,
    K,
    Pa0,
    S,
    W_selected,
    beta,
    h_array,
    idx_selected,
    interior_condition,
):
    interior_mask = interior_condition(W_selected, h_array, beta, CD0, K, Pa0, E_P)

    CLopt_interior = np.where(interior_mask, CL_P, np.nan)

    velocity_interior_harray = velocity(W_selected, h_array, CLopt_interior, S)

    dTopt_interior = (
        W_selected
        / E_P
        * velocity_interior_harray
        / Pa0
        / (atmos.rhoratio(h_array) ** beta)
    )

    CLopt_interior_selected = CLopt_interior[idx_selected]
    dTopt_interior_selected = dTopt_interior[idx_selected]

    velocity_interior_selected = velocity_interior_harray[idx_selected]

    power_interior_harray = power(
        h_array, S, CD0, K, CLopt_interior, velocity_interior_harray
    )
    power_interior_selected = power_interior_harray[idx_selected]
    return (
        CLopt_interior_selected,
        dTopt_interior_selected,
        power_interior_selected,
        velocity_interior_harray,
        velocity_interior_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_interior_selected,
    a_harray,
    active_selection,
    constraint,
    dT_array,
    dTopt_interior_selected,
    h_array,
    h_selected,
    max_colorbar,
    min_colorbar,
    power_interior_selected,
    power_surface,
    velocity_interior_harray,
    velocity_interior_selected,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_interior_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot, first four are template
    fig_interior_optimum.add_traces(
        [
            go.Heatmap(
                x=CL_array,
                y=dT_array,
                z=power_surface / 1e3,
                opacity=0.9,
                name="Power",
                colorscale="viridis",
                zsmooth="best",
                zmin=min_colorbar,
                zmax=max_colorbar,
                colorbar={"title": "Power (kW)"},
            ),
            go.Scatter(
                x=CL_array,
                y=constraint,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter(
                x=[CL_array[-15]],
                y=[constraint[-15]],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
                textfont=dict(size=14, family="Arial"),
            ),
            go.Scatter(
                x=[CLopt_interior_selected],
                y=[dTopt_interior_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=10,
                    color="#FFFFFF",
                    symbol="circle",
                ),
                name="P<sub>min</sub>",
                customdata=[power_interior_selected / 1e3],
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub>: 1 <br>P: %{customdata}<extra></extra>",
            ),
        ],
        cols=1,
        rows=1,
    )

    # Traces on the flight envelope, first four traces are template
    fig_interior_optimum.add_traces(
        [
            go.Scatter(
                x=velocity_stall_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=1, color="rgba(255, 0, 0, 1)", dash="dash"),
                name="V<sub>stall</sub>",
                showlegend=False,
            ),
            go.Scatter(
                x=[velocity_stall_harray[-8]],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(255, 0, 0, 0)"),
                text=["V<sub>stall</sub>"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=a_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(color="rgba(255, 180, 90, 1)", width=2, dash="dash"),
                name="M1.0",
                showlegend=False,
            ),
            go.Scatter(
                x=[a_harray[-8] - 5],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                text=["M1.0"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=velocity_interior_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(
                    width=3,
                    color="rgb(232,158,184)",
                ),
                showlegend=False,
                name="P<sub>min</sub>",
            ),
            go.Scatter(
                x=[velocity_interior_selected],
                y=[h_selected / 1e3],
                mode="markers+text",
                marker=dict(size=10, color="#FFFFFF"),
                name="P<sub>min</sub>",
                showlegend=False,
            ),
        ],
        cols=2,
        rows=1,
    )

    fig_interior_optimum.update_xaxes(
        title_text=r"$C_L\:(\text{-})$",
        range=[xy_lowerbound, active_selection["CLmax_ld"] + 0.05],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=1,
    )
    fig_interior_optimum.update_yaxes(
        title_text=r"$\delta_T \:(\text{-})$",
        range=[xy_lowerbound, 1 + 0.05],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=1,
    )

    # Second subplot: V vs h
    fig_interior_optimum.update_xaxes(
        title_text=r"$V \: \text{(m/s)}$",
        range=[xy_lowerbound, atmos.a(0) + 15],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=2,
    )
    fig_interior_optimum.update_yaxes(
        title_text=r"$h \: 	\text{(km)}$",
        range=[xy_lowerbound, 20],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=2,
    )

    fig_interior_optimum.update_layout(
        title={
            "text": f"Interior minimum power for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        }
    )

    mo.output.clear()
    return (fig_interior_optimum,)


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_interior_optimum):
    fig_interior_optimum
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### _Lift-limited solutions (stall)_

    In this case: $C_L = C_{L_{\mathrm{max}}}$, $\delta_T \lt 1$, $\mu_1 \gt 0$, $\mu_2= 0$

    from stationarity condition (2): $\lambda_1 = 0$

    from stationarity condition (1):

    $$
    \mu_1 = \sqrt{\frac{2W^3}{\rho S}}\left(\frac{3}{2}C_{D_0}C_{L_{\mathrm{max}}}^{-5/2} - \frac{1}{2} K C_{L_{\mathrm{max}}}^{-1/2}\right) \gt 0
    $$

    $$
    \Rightarrow 3C_{D_0}C_{L_{\mathrm{max}}}^{-2} - K \gt 0 \quad  \Rightarrow \quad C_{L_{\mathrm{max}}} \lt \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    This means that, if we design an aircraft such that its $C_{L_P}$ is lower than its stall lift coefficient, then we  the minimum power required will be obtained at stall, because the aircraft is not able to fly at $C_{L_P}$ in steady level flight.
    In other words, an aircraft so designed would only be able to fly on the right branch of the power performance diagram, because the stall speed would be higher than the speed for minimum power. Therefore, the effective minimum power flyable in steady level flight would be obtained at the stall speed.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    We can now calculate the optimal $\delta_T^*$. As before, define the velocity at which the aircraft is flying for a cleaner solution. Note that $C_L = C_{L_{\mathrm{max}}}$ thus the aircraft is flying at stall speed $V_S$: 

    $$
    V_S= \sqrt{\frac{2W}{\rho S C_{L_{\mathrm{max}}}}}
    $$

    The correrponding $\delta_T^*$, found from the primal feasibility constraint (3): 

    $$
    \delta_T^* = \frac{W}{E_S}\frac{V_S}{P_{a0}\sigma^\beta}
    $$

    This is valid for: 

    $$
    \delta_T^*\lt 1 \Leftrightarrow \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt  \; P_{a0} \,E_S\sqrt{\frac{1}{2}\rho_0SC_{L_{\mathrm{max}}}}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Finally, the optimum for the lift-limited case is:

    $$
    \boxed{C_L^* = C_{L_{\mathrm{max}}}} \quad \land \quad \boxed{\delta_T^* = \frac{W}{E_S}\frac{V_S}{P_{a0}\sigma^\beta}} \qquad \mathrm{with} \quad V_S= \sqrt{\frac{2W}{\rho S C_{L_{\mathrm{max}}}}} \qquad \mathrm{for} \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} \lt  \; P_{a0} \,E_S\sqrt{\frac{1}{2}\rho_0SC_{L_{\mathrm{max}}}}
    $$
    """
    )
    return


@app.function
def maxlift_condition(W, h, beta, S, Pa0, CLmax, E_S, CL_P):
    sigma = atmos.rhoratio(h)
    condition = (
        ((W**1.5) / (sigma ** (beta + 0.5)))
        < (np.sqrt(0.5 * atmos.rho0 * S * CLmax) * Pa0 * E_S)
    ) & (CLmax < CL_P)

    return condition


@app.cell(hide_code=True)
def _(
    CD0,
    CL_P,
    CLmax,
    E_S,
    K,
    Pa0,
    S,
    W_selected,
    beta,
    h_array,
    idx_selected,
):
    maxlift_mask = maxlift_condition(
        W_selected, h_array, beta, S, Pa0, CLmax, E_S, CL_P
    )

    CLopt_maxlift = np.where(maxlift_mask, CLmax, np.nan)

    velocity_maxlift_harray = velocity(W_selected, h_array, CLopt_maxlift, S)

    dTopt_maxlift = (
        W_selected
        / E_S
        * velocity_maxlift_harray
        / Pa0
        / (atmos.rhoratio(h_array) ** beta)
    )

    CLopt_maxlift_selected = CLopt_maxlift[idx_selected]
    dTopt_maxlift_selected = dTopt_maxlift[idx_selected]

    velocity_maxlift_selected = velocity_maxlift_harray[idx_selected]

    power_maxlift_harray = power(
        h_array, S, CD0, K, CLopt_maxlift, velocity_maxlift_harray
    )
    power_maxlift_selected = power_maxlift_harray[idx_selected]
    return (
        CLopt_maxlift_selected,
        dTopt_maxlift_selected,
        power_maxlift_selected,
        velocity_maxlift_harray,
        velocity_maxlift_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxlift_selected,
    a_harray,
    active_selection,
    constraint,
    dT_array,
    dTopt_maxlift_selected,
    h_array,
    h_selected,
    max_colorbar,
    min_colorbar,
    power_maxlift_selected,
    power_surface,
    velocity_maxlift_harray,
    velocity_maxlift_selected,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_maxlift_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot
    fig_maxlift_optimum.add_traces(
        [
            go.Heatmap(
                x=CL_array,
                y=dT_array,
                z=power_surface / 1e3,
                opacity=0.9,
                name="Power",
                colorscale="viridis",
                zsmooth="best",
                zmin=min_colorbar,
                zmax=max_colorbar,
                colorbar={"title": "Power (kW)"},
            ),
            go.Scatter(
                x=CL_array,
                y=constraint,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter(
                x=[CL_array[-15]],
                y=[constraint[-15]],
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
                textfont=dict(size=14, family="Arial"),
            ),
            go.Scatter(
                x=[CLopt_maxlift_selected],
                y=[dTopt_maxlift_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=10,
                    color="#FFFFFF",
                    symbol="circle",
                ),
                name="P<sub>min</sub>",
                customdata=[power_maxlift_selected / 1e3],
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub>: 1 <br>P: %{customdata}<extra></extra>",
            ),
        ],
        cols=1,
        rows=1,
    )

    # Traces on the flight envelope
    fig_maxlift_optimum.add_traces(
        [
            go.Scatter(
                x=velocity_stall_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=1, color="rgba(255, 0, 0, 1)", dash="dash"),
                name="V<sub>stall</sub>",
                showlegend=False,
            ),
            go.Scatter(
                x=[velocity_stall_harray[-8]],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(255, 0, 0, 0)"),
                text=["V<sub>stall</sub>"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=a_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(color="rgba(255, 180, 90, 1)", width=2, dash="dash"),
                name="M1.0",
                showlegend=False,
            ),
            go.Scatter(
                x=[a_harray[-8] - 5],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                text=["M1.0"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=velocity_maxlift_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=3, color="rgb(232,158,184)"),
                showlegend=False,
                name="P<sub>min</sub>",
            ),
            go.Scatter(
                x=[velocity_maxlift_selected],
                y=[h_selected / 1e3],
                mode="markers+text",
                marker=dict(size=10, color="#FFFFFF"),
                name="P<sub>min</sub>",
                showlegend=False,
            ),
        ],
        cols=2,
        rows=1,
    )

    fig_maxlift_optimum.update_xaxes(
        title_text=r"$C_L\:(\text{-})$",
        range=[xy_lowerbound, active_selection["CLmax_ld"] + 0.05],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=1,
    )
    fig_maxlift_optimum.update_yaxes(
        title_text=r"$\delta_T \:(\text{-})$",
        range=[xy_lowerbound, 1 + 0.05],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=1,
    )

    # Second subplot: V vs h
    fig_maxlift_optimum.update_xaxes(
        title_text=r"$V \: \text{(m/s)}$",
        range=[xy_lowerbound, atmos.a(0) + 15],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=2,
    )
    fig_maxlift_optimum.update_yaxes(
        title_text=r"$h \: 	\text{(km)}$",
        range=[xy_lowerbound, 20],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=2,
    )

    fig_maxlift_optimum.update_layout(
        title={
            "text": f"Lift-limited minimum power for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        }
    )

    mo.output.clear()
    return (fig_maxlift_optimum,)


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_maxlift_optimum):
    fig_maxlift_optimum
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### _Thrust limited solutions_

    In this case: $C_L \lt C_{L_{\mathrm{max}}}$, $\delta_T = 1$, $\mu_1 = 0$, $\mu_2 \gt 0$

    from stationarity condition (2): $\displaystyle \lambda_1 = -\frac{\mu_2}{P_{a0}\sigma^\beta} \quad \Rightarrow \quad \lambda_1 \lt 0$

    Thus, from stationarity condition (1): 

    $$
    \sqrt{\frac{2W^3}{\rho S}}\left(-\frac{3}{2}C_{D_0}C_L^{-5/2} + \frac{1}{2} K C_L^{-1/2}\right)(1-\lambda_1) = 0 \quad \text{with } \quad 1-\lambda_1 \gt 0
    $$

    $$
    \Rightarrow -3C_{D_0}C_L^{-2} + K = 0
    $$


    $$
    C_L^* = \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    The condition for which this is true is found using the primal feasibility constraint (3). 

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0} E_P \sqrt{\frac{\rho_0SC_{L_P}}{2}} \;  = P_{a0} E_P \sqrt{\frac{\rho_0 S}{2}} \sqrt[4]{\frac{3 C_{D_0}}{K}}
    $$

    This can be compared with what we found in the interior of the domain, showing the thrust limited case represents the limit case of the interior optima.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Finally, the optimum for the thrust-limited case is:

    $$
    \boxed{C_L^* = C_{L_P}} \quad \land \quad \boxed{\delta_T^* = 1} \qquad  \mathrm{for} \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} = \sqrt{\frac{3}{2}\rho_0 S \frac{C_{D_0}}{K}} \; P_{a0} \,E_P
    $$
    """
    )
    return


@app.function
def maxthrust_altitude(W, beta, Pa0, CD0, K, S, E_P):
    CL_P = np.sqrt(3 * CD0 / K)
    sigma_exp = (W**1.5) / Pa0 / E_P / np.sqrt(0.5 * atmos.rho0 * S * CL_P)

    sigma = sigma_exp ** (1 / (beta + 0.5))

    h = atmos.altitude(sigma)
    return np.where(h > 0, h, np.nan)


@app.cell
def _(CD0, CL_P, CL_array, E_P, K, Pa0, S, W_selected, beta):
    # This cell is to revise
    maxthrust_h = maxthrust_altitude(W_selected, beta, Pa0, CD0, K, S, E_P)

    CLopt_maxthrust = CL_P

    velocity_maxthrust_selected = velocity(
        W_selected, maxthrust_h, CLopt_maxthrust, S, cap=False
    )

    velocity_CLarray_maxthrust_h = velocity(
        W_selected, maxthrust_h, CL_array, S, cap=False
    )

    dTopt_maxthrust = 1

    drag_maxthrust_h_curve = drag(
        maxthrust_h, S, CD0, K, CL_array, velocity_CLarray_maxthrust_h
    )

    constraint_maxthrust = horizontal_constraint(
        W_selected,
        maxthrust_h,
        CD0,
        K,
        CL_array,
        Pa0,
        beta,
        velocity_CLarray_maxthrust_h,
        S,
        drag_maxthrust_h_curve,
        type="propeller",
    )

    power_maxthrust_curve = np.where(
        ~np.isnan(constraint_maxthrust),
        power(maxthrust_h, S, CD0, K, CL_array, velocity_CLarray_maxthrust_h),
        np.nan,
    )

    power_maxthrust_surface = np.tile(power_maxthrust_curve, (len(CL_array), 1))

    power_maxthrust_selected = power(
        maxthrust_h, S, CD0, K, CLopt_maxthrust, velocity_maxthrust_selected
    )

    if np.all(np.isnan(power_maxthrust_surface)):
        power_maxthrust_surface[0, 0] = 1e-10
    return (
        CLopt_maxthrust,
        constraint_maxthrust,
        dTopt_maxthrust,
        maxthrust_h,
        power_maxthrust_selected,
        power_maxthrust_surface,
        velocity_maxthrust_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxthrust,
    a_harray,
    active_selection,
    constraint_maxthrust,
    dT_array,
    dTopt_maxthrust,
    h_array,
    maxthrust_h,
    power_maxthrust_selected,
    power_maxthrust_surface,
    velocity_maxthrust_selected,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_maxthrust_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot
    fig_maxthrust_optimum.add_traces(
        [
            go.Heatmap(
                x=CL_array,
                y=dT_array,
                z=power_maxthrust_surface / 1e3,
                opacity=0.9,
                name="Power",
                colorscale="viridis",
                zsmooth="best",
                zmin=np.nanmin(power_maxthrust_surface) / 1e3,
                zmax=np.nanmin(power_maxthrust_surface) * 1.5 / 1e3,
                colorbar={"title": "Power (kW)"},
            ),
            go.Scatter(
                x=CL_array,
                y=constraint_maxthrust,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter(
                x=[CL_array[-15]],
                y=[constraint_maxthrust[-15]],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
                textfont=dict(size=14, family="Arial"),
            ),
            go.Scatter(
                x=[CLopt_maxthrust],
                y=[dTopt_maxthrust],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=10,
                    color="#FFFFFF",
                    symbol="circle",
                ),
                name="P<sub>min</sub>",
                customdata=[power_maxthrust_selected / 1e3],
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub>: 1 <br>P: %{customdata}<extra></extra>",
            ),
        ],
        cols=1,
        rows=1,
    )

    # Traces on the flight envelope
    fig_maxthrust_optimum.add_traces(
        [
            go.Scatter(
                x=velocity_stall_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=1, color="rgba(255, 0, 0, 1)", dash="dash"),
                name="V<sub>stall</sub>",
                showlegend=False,
            ),
            go.Scatter(
                x=[velocity_stall_harray[-8]],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(255, 0, 0, 0)"),
                text=["V<sub>stall</sub>"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=a_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(color="rgba(255, 180, 90, 1)", width=2, dash="dash"),
                name="M1.0",
                showlegend=False,
            ),
            go.Scatter(
                x=[a_harray[-8] - 5],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                text=["M1.0"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=[velocity_maxthrust_selected],
                y=[maxthrust_h / 1e3],
                mode="markers",
                marker=dict(size=10, color="#FFFFFF"),
                name="P<sub>min</sub>",
                showlegend=False,
            ),
        ],
        cols=2,
        rows=1,
    )

    fig_maxthrust_optimum.update_xaxes(
        title_text=r"$C_L\:(\text{-})$",
        range=[xy_lowerbound, active_selection["CLmax_ld"] + 0.05],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=1,
    )
    fig_maxthrust_optimum.update_yaxes(
        title_text=r"$\delta_T \:(\text{-})$",
        range=[xy_lowerbound, 1 + 0.05],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=1,
    )

    # Second subplot: V vs h
    fig_maxthrust_optimum.update_xaxes(
        title_text=r"$V \: \text{(m/s)}$",
        range=[xy_lowerbound, atmos.a(0) + 15],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=2,
    )
    fig_maxthrust_optimum.update_yaxes(
        title_text=r"$h \: 	\text{(km)}$",
        range=[xy_lowerbound, 20],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=2,
    )

    fig_maxthrust_optimum.update_layout(
        title={
            "text": f"Thrust-limited minimum power for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        }
    )

    mo.output.clear()
    return (fig_maxthrust_optimum,)


@app.cell
def _(mass_stack):
    mass_stack
    return


@app.cell
def _(fig_maxthrust_optimum):
    fig_maxthrust_optimum
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### _Lift- and thrust- limited optimum_

    In this case: $C_L = C_{L_{\mathrm{max}}}$, $\delta_T = 1$, $\mu_1 \gt 0$, $\mu_2 \gt 0$

    from stationarity condition (2): $\displaystyle \lambda_1 = -\frac{\mu_2}{P_{a0}\sigma^\beta} \quad \Rightarrow \quad \lambda_1 \lt 0$

    Thus, from stationarity condition (1), since $1-\lambda_1 \gt 0$: 

    $$
    \mu_1 = \sqrt{\frac{2W^3}{\rho S}}\left(\frac{3}{2}C_{D_0}C_{L_{\mathrm{max}}}^{-5/2} - \frac{1}{2} K C_{L_{\mathrm{max}}}^{-1/2}\right)(1-\lambda_1)\gt 0
    $$

    $$
    \Rightarrow 3 C_{D_0}C_{L_{\mathrm{max}}}^{-2} - K \gt 0
    $$


    $$
    \Rightarrow C_{L_{\mathrm{max}}} \lt \sqrt{\frac{3C_{D_0}}{K}} = \sqrt{3}C_{L_E} = C_{L_P}
    $$

    which shows once again that the necessary condition to obtain minimum power in stall conditions and maximum throttle. If it was otherwise ($C_{L_{\mathrm{max}}} > C_{L_P}$), it would be impossible to minimise power at stall and maximum thrust as the aircraft would reach the unconstrained minium power before stalling.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    The condition for which this is true is found using the primal feasibility constraint (3). 

    $$
    \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0}E_S \sqrt{\frac{1}{2}\rho_0SC_{L_{\mathrm{max}}}}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    Finally, the optimum for the thrust-limited and lift-limited case is:

    $$
    \boxed{C_L^* = C_{L_\mathrm{max}}} \quad \land \quad \boxed{\delta_T^* = 1} \qquad  \mathrm{for} \quad \frac{W^{3/2}}{\sigma^{\beta+1/2}} = P_{a0}E_S \sqrt{\frac{1}{2}\rho_0SC_{L_{\mathrm{max}}}}
    $$
    """
    )
    return


@app.function
def maxlift_thrust_altitude(W, beta, Pa0, S, CD0, K, CLmax, E_S):
    sigma_exp = (W**1.5) / Pa0 / E_S / np.sqrt(0.5 * atmos.rho0 * S * CLmax)

    sigma = sigma_exp ** (1 / (beta + 0.5))

    h = atmos.altitude(sigma)
    return np.where(((h > 0) & (CLmax < np.sqrt(3 * CD0 / K))), h, np.nan)


@app.cell
def _(CD0, CL_array, CLmax, E_S, K, Pa0, S, W_selected, beta):
    # This cell is to revise
    maxlift_thrust_h = maxlift_thrust_altitude(
        W_selected, beta, Pa0, S, CD0, K, CLmax, E_S
    )

    CLopt_maxlift_thrust = CLmax

    velocity_maxlift_thrust_selected = velocity(
        W_selected, maxlift_thrust_h, CLopt_maxlift_thrust, S, cap=False
    )

    velocity_CLarray_maxlift_thrust_h = velocity(
        W_selected, maxlift_thrust_h, CL_array, S, cap=False
    )

    dTopt_maxlift_thrust = 1

    drag_maxlift_thrust_h_curve = drag(
        maxlift_thrust_h, S, CD0, K, CL_array, velocity_CLarray_maxlift_thrust_h
    )

    constraint_maxlift_thrust = horizontal_constraint(
        W_selected,
        maxlift_thrust_h,
        CD0,
        K,
        CL_array,
        Pa0,
        beta,
        velocity_CLarray_maxlift_thrust_h,
        S,
        drag_maxlift_thrust_h_curve,
        type="propeller",
    )

    power_maxlift_thrust_curve = np.where(
        ~np.isnan(constraint_maxlift_thrust),
        power(maxlift_thrust_h, S, CD0, K, CL_array, velocity_CLarray_maxlift_thrust_h),
        np.nan,
    )

    power_maxlift_thrust_surface = np.tile(
        power_maxlift_thrust_curve, (len(CL_array), 1)
    )

    power_maxlift_thrust_selected = power(
        maxlift_thrust_h,
        S,
        CD0,
        K,
        CLopt_maxlift_thrust,
        velocity_maxlift_thrust_selected,
    )

    if np.all(np.isnan(power_maxlift_thrust_surface)):
        power_maxlift_thrust_surface[0, 0] = 1e-10
    return (
        CLopt_maxlift_thrust,
        constraint_maxlift_thrust,
        dTopt_maxlift_thrust,
        maxlift_thrust_h,
        power_maxlift_thrust_selected,
        power_maxlift_thrust_surface,
        velocity_maxlift_thrust_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxlift_thrust,
    a_harray,
    active_selection,
    constraint_maxlift_thrust,
    dT_array,
    dTopt_maxlift_thrust,
    h_array,
    maxlift_thrust_h,
    power_maxlift_thrust_selected,
    power_maxlift_thrust_surface,
    velocity_maxlift_thrust_selected,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_maxlift_thrust_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot
    fig_maxlift_thrust_optimum.add_traces(
        [
            go.Heatmap(
                x=CL_array,
                y=dT_array,
                z=power_maxlift_thrust_surface / 1e3,
                opacity=0.9,
                name="Power",
                colorscale="viridis",
                zsmooth="best",
                zmin=np.nanmin(power_maxlift_thrust_surface) / 1e3,
                zmax=np.nanmin(power_maxlift_thrust_surface) * 1.5 / 1e3,
                colorbar={"title": "Power (kW)"},
            ),
            go.Scatter(
                x=CL_array,
                y=constraint_maxlift_thrust,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter(
                x=[CL_array[-15]],
                y=[constraint_maxlift_thrust[-15]],
                opacity=1,
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
                textfont=dict(size=14, family="Arial"),
            ),
            go.Scatter(
                x=[CLopt_maxlift_thrust],
                y=[dTopt_maxlift_thrust],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=10,
                    color="#FFFFFF",
                    symbol="circle",
                ),
                name="P<sub>min</sub>",
                customdata=[power_maxlift_thrust_selected],
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub>: 1 <br>P: %{customdata}<extra></extra>",
            ),
        ],
        cols=1,
        rows=1,
    )

    # Traces on the flight envelope
    fig_maxlift_thrust_optimum.add_traces(
        [
            go.Scatter(
                x=velocity_stall_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=1, color="rgba(255, 0, 0, 1)", dash="dash"),
                name="V<sub>stall</sub>",
                showlegend=False,
            ),
            go.Scatter(
                x=[velocity_stall_harray[-8]],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(255, 0, 0, 0)"),
                text=["V<sub>stall</sub>"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=a_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(color="rgba(255, 180, 90, 1)", width=2, dash="dash"),
                name="M1.0",
                showlegend=False,
            ),
            go.Scatter(
                x=[a_harray[-8] - 5],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                text=["M1.0"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=[velocity_maxlift_thrust_selected],
                y=[maxlift_thrust_h / 1e3],
                mode="markers",
                marker=dict(size=10, color="#FFFFFF"),
                showlegend=False,
                name="P<sub>min</sub>",
            ),
        ],
        cols=2,
        rows=1,
    )

    fig_maxlift_thrust_optimum.update_xaxes(
        title_text=r"$C_L\:(\text{-})$",
        range=[xy_lowerbound, active_selection["CLmax_ld"] + 0.05],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=1,
    )
    fig_maxlift_thrust_optimum.update_yaxes(
        title_text=r"$\delta_T \:(\text{-})$",
        range=[xy_lowerbound, 1 + 0.05],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=1,
    )

    # Second subplot: V vs h
    fig_maxlift_thrust_optimum.update_xaxes(
        title_text=r"$V \: \text{(m/s)}$",
        range=[xy_lowerbound, atmos.a(0) + 15],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=2,
    )
    fig_maxlift_thrust_optimum.update_yaxes(
        title_text=r"$h \: 	\text{(km)}$",
        range=[xy_lowerbound, 20],
        showgrid=True,
        gridcolor="#515151",
        gridwidth=1,
        row=1,
        col=2,
    )

    fig_maxlift_thrust_optimum.update_layout(
        title={
            "text": f"Thrust-lift limited minimum power for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        }
    )

    mo.output.clear()
    return (fig_maxlift_thrust_optimum,)


@app.cell
def _(mass_stack):
    mass_stack
    return


@app.cell
def _(fig_maxlift_thrust_optimum):
    fig_maxlift_thrust_optimum
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""Summarizing all the flight envelopes derived so far we obtain:""")
    return


@app.cell
def _():
    mo.md(r"""Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for minimum speed moves in the graph.""")
    return


@app.cell
def _(
    h_array,
    maxlift_thrust_h,
    maxthrust_h,
    velocity_interior_harray,
    velocity_maxlift_harray,
    velocity_maxlift_selected,
    velocity_maxthrust_selected,
):
    V = np.where(
        np.isnan(velocity_interior_harray),
        velocity_maxlift_harray,
        velocity_interior_harray,
    )

    # remove NaNs from merged velocities
    mask = ~np.isnan(V)
    h_clean = h_array[mask]
    V_clean = V[mask]

    # sort original points by altitude
    sort_idx = np.argsort(h_clean)
    h_sorted = h_clean[sort_idx]
    V_sorted = V_clean[sort_idx]

    final_envelope_h = np.append(h_sorted, [maxthrust_h, maxlift_thrust_h])
    final_envelope_velocity = np.append(
        V_sorted, [velocity_maxthrust_selected, velocity_maxlift_selected]
    )
    return final_envelope_h, final_envelope_velocity


@app.cell
def _(
    a_harray,
    active_selection,
    final_envelope_h,
    final_envelope_velocity,
    h_array,
    maxthrust_h,
    velocity_maxlift_thrust_selected,
    velocity_maxthrust_selected,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_final_flightenv = go.Figure()

    fig_final_flightenv.add_traces(
        [
            go.Scatter(
                x=velocity_stall_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=1, color="rgba(255, 0, 0, 1)", dash="dash"),
                name="V<sub>stall</sub>",
                showlegend=False,
            ),
            go.Scatter(
                x=[velocity_stall_harray[-8]],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(255, 0, 0, 0)"),
                text=["V<sub>stall</sub>"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=a_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(color="rgba(255, 180, 90, 1)", width=2, dash="dash"),
                name="M1.0",
                showlegend=False,
            ),
            go.Scatter(
                x=[a_harray[-8] - 5],
                y=[h_array[-8] / 1e3],
                mode="markers+text",
                marker=dict(size=1, color="rgba(0, 0, 0, 0.0)"),
                text=["M1.0"],
                hoverinfo="skip",
                textposition="top left",
                showlegend=False,
            ),
            go.Scatter(
                x=final_envelope_velocity,
                y=final_envelope_h / 1e3,
                mode="lines",
                line=dict(width=3, color="rgb(232,158,184)"),
                showlegend=False,
                name="P<sub>min</sub>",
            ),
            go.Scatter(
                x=[velocity_maxthrust_selected],
                y=[maxthrust_h / 1e3],
                mode="markers",
                marker=dict(size=10, color="rgb(232,158,184)"),
                name="P<sub>min</sub>",
                showlegend=False,
            ),
            go.Scatter(
                x=[velocity_maxlift_thrust_selected],
                y=[maxthrust_h / 1e3],
                mode="markers",
                marker=dict(size=10, color="rgb(232,158,184)"),
                name="P<sub>min</sub>",
                showlegend=False,
            ),
        ],
    )

    fig_final_flightenv.update_layout(
        xaxis=dict(
            title=r"$V \: \text{(m/s)}$",
            range=[xy_lowerbound, atmos.a(0) + 15],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        yaxis=dict(
            title=r"$h \: 	\text{(km)}$",
            range=[xy_lowerbound, 20],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
    )
    fig_final_flightenv.update_layout(
        title={
            "text": f"Flight envelope for minimum power for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        }
    )

    mo.output.clear()
    return (fig_final_flightenv,)


@app.cell
def _(fig_final_flightenv):
    fig_final_flightenv
    return


if __name__ == "__main__":
    app.run()
