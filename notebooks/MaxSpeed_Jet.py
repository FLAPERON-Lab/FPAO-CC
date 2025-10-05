import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
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


    def CL_from_horizontal_constraint(W, h, S, CD0, K, Ta0, beta, ac_type):
        E_max = endurance(K, CD0, "max")
        sigma = atmos.rhoratio(h)

        plus_solution = np.full_like(sigma, np.nan, dtype=float)
        minus_solution = np.full_like(sigma, np.nan, dtype=float)

        if ac_type == "jet":
            # validity condition
            condition = (W / (sigma**beta)) < (Ta0 * E_max)

            # compute safe argument for sqrt
            arg = 1 - (W / (Ta0 * sigma**beta * E_max)) ** 2
            arg = np.where(arg >= 0, arg, np.nan)  # mask negatives

            root = np.sqrt(arg)
            multiplier = Ta0 * sigma**beta / (2 * K * W)

            plus_solution = np.where(condition, multiplier * (1 + root), np.nan)
            minus_solution = np.where(condition, multiplier * (1 - root), np.nan)

        return [plus_solution, minus_solution]
    return (
        CL_from_horizontal_constraint,
        ac,
        atmos,
        data_dir,
        drag,
        endurance,
        go,
        horizontal_constraint,
        make_subplots,
        mo,
        np,
        velocity,
    )


@app.cell
def _():
    # Set navbar on the right
    _defaults.set_sidebar()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Maximum airspeed: simplfied jet aircraft

    $$
    \begin{aligned}
        \max_{C_L, \delta_T} 
        & \quad V \\
        \text{subject to} 
        & \quad c_1^\mathrm{eq} = L-W = \frac{1}{2}\rho V^2 S C_L - W = 0 \\
        & \quad c_2^\mathrm{eq} = T-D = \delta_T T_a(V,h) - \frac{1}{2} \rho V^2 S (C_{D_0}+K C_L^2) =0 \\
        \text{for } 
        & \quad C_L \in [0, C_{L_\mathrm{max}}] \\
        & \quad \delta_T \in [0, 1] \\
        \text{with } 
        & \quad T_a(V,h) = T_a(h) = T_{a0}\sigma^\beta \\
    \end{aligned}
    $$
    """
    )
    return


@app.cell
def _(ac_table, data, mo):
    # Interactive elements (1)

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

    dT_slider = mo.ui.slider(
        start=0, stop=1, step=0.1, label=r"$\delta_T$", value=0.5
    )

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
def _(active_selection, atmos, endurance, h_slider, m_slider, np, velocity):
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
    Ta0 = active_selection["Ta0"] * 1e3  # Watts
    beta = active_selection["beta"]
    CL_P = np.sqrt(3 * CD0 / K)
    CL_E = np.sqrt(CD0 / K)
    E_max = endurance(K, CD0, "max")
    E_P = (np.sqrt(3) / 2) * E_max
    E_S = CLmax / (CD0 + K * CLmax**2)
    velocity_stall_harray = velocity(W_selected, h_array, CLmax, S)
    return (
        CD0,
        CL_array,
        CLmax,
        E_S,
        E_max,
        K,
        S,
        Ta0,
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


@app.cell
def _(
    CD0,
    CL_array,
    CL_slider,
    K,
    S,
    Ta0,
    W_selected,
    beta,
    drag,
    h_selected,
    horizontal_constraint,
    np,
    velocity,
):
    # Computation cell (1)
    velocity_CLarray = velocity(W_selected, h_selected, CL_array, S, cap=False)

    velocity_CLarray = np.where(
        np.isnan(velocity_CLarray), np.nanmax(velocity_CLarray), velocity_CLarray
    )

    velocity_user_selected = velocity(
        W_selected, h_selected, CL_slider.value, S, cap=False
    )

    drag_curve = drag(
        h_selected,
        S,
        CD0,
        K,
        CL_array,
        velocity_CLarray,
    )

    # Calculate the c2_eq constraint curve
    constraint = horizontal_constraint(
        W_selected,
        h_selected,
        CD0,
        K,
        CL_array,
        Ta0,
        beta,
        V=velocity_CLarray,
        S=S,
        D=drag_curve,
        type="jet",
    )

    velocity_surface = np.tile(velocity_CLarray, (len(CL_array), 1))

    min_colorbar = 1 / np.max(velocity_surface)
    max_colorbar = min_colorbar * 10
    return (
        constraint,
        max_colorbar,
        min_colorbar,
        velocity_surface,
        velocity_user_selected,
    )


@app.cell
def _(
    CL_array,
    CL_slider,
    active_selection,
    constraint,
    dT_array,
    dT_slider,
    go,
    max_colorbar,
    min_colorbar,
    mo,
    velocity_surface,
    velocity_user_selected,
    xy_lowerbound,
):
    # Initial Figure
    fig_initial = go.Figure()

    # Minimum velocity surface
    fig_initial.add_traces(
        [
            go.Surface(
                x=CL_array,
                y=dT_array,
                z=1 / velocity_surface,
                opacity=0.9,
                name="1/Velocity",
                colorscale="viridis",
                cmin=min_colorbar,
                cmax=max_colorbar,
                colorbar={"title": "V<sup>-1</sup> (s/m)"},
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint,
                z=1 / velocity_surface[0],
                opacity=0.7,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[30]],
                y=[constraint[30]],
                z=[1 / velocity_surface[0, 30] - 0.0003],
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
                z=[1 / velocity_user_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="Design Point",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>1/V: %{z}<extra>%{fullData.name}</extra>",
            ),
        ]
    )
    # Set the camera to show the end of both axes
    camera = dict(eye=dict(x=-1.35, y=-1.35, z=1.35))

    fig_initial.update_layout(
        scene=dict(
            xaxis=dict(
                title="C<sub>L</sub> (-)",
                range=[xy_lowerbound, active_selection["CLmax_ld"]],
            ),
            yaxis=dict(title="δ<sub>T</sub> (-)", range=[xy_lowerbound, 1]),
            zaxis=dict(
                title="V<sup>-1</sup> (s/m)", range=[min_colorbar, max_colorbar]
            ),
        ),
    )

    fig_initial.update_layout(
        scene_camera=camera,
        title={
            "text": f"Maximum airspeed domain for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    mo.output.clear()
    return (fig_initial,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## KKT formulation
    To be reconducted in the standard KKT analysis format, the objective function is expressed in terms of the controls by direct elimination of $c_1^\mathrm{eq}$.
    Also, maximizing $V$ is equivalent to minimizing its inverse, $1/V$.
    Therefore, to simplify the calculations, the problem is rewritten as follows:
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    $$
    \begin{aligned}
        \min_{C_L, \delta_T} 
        & \quad \frac{1}{V} = \sqrt{\frac{\rho S C_L}{2W}} \\
        \text{subject to} 
        & \quad g_1 = \delta_T T_{a0}\sigma^\beta - W \left(\frac{C_{D_0} + KC_L^2}{C_L}\right) = 0 \\
        & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
        & \quad h_2 = \delta_T - 1 \le 0 \\
    \end{aligned}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""In the interactive graph below, select a simplified jet aircraft of your choice and experiment in finding an optimum by changing the control variables, $C_L$ and $\delta_T$. The design point is marked in white in the 3D velocity surface."""
    )
    return


@app.cell
def _(ac, data_dir, mo):
    # Database cell (1)

    data = ac.available_aircrafts(data_dir, ac_type="Jet")

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


@app.cell
def _(CL_slider, dT_slider, mo):
    mo.md(
        f"""Here you can modify the control variables to understand how it affects the design: {mo.hstack([dT_slider, CL_slider])}"""
    )
    return


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_initial):
    fig_initial
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Lagrangian function and KKT conditions

    The Lagrangian function combines the objective function with equality constraints using Lagrange multipliers ($\lambda_i$) and inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(C_L, \delta_T, \lambda_1, \mu_1, \mu_2) = 
    \quad \sqrt{\frac{\rho S C_L}{2W}}
    & + \\
    & + \lambda_1 \left[\delta_T T_{a0}\sigma^\beta - W \left(\frac{C_{D_0} + KC_L^2}{C_L}\right)\right] + \\
    & + \mu_1 (C_L - C_{L_\mathrm{max}}) + \\
    & + \mu_2 (\delta_T - 1)\\
    \end{aligned}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    A necessary condition for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist, the multipliers $\lambda_1, \mu_1, \mu_2$ have to meet the following conditions:

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = \frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}}C_L^{-1/2} - \lambda_1 W\left(\frac{KC_L^2 - C_{D_0}}{C_L^2}\right) + \mu_1 = 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 T_{a0}\sigma^\beta + \mu_2 = 0$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \delta_T T_{a0}\sigma^\beta - W \left(\frac{C_{D_0} + KC_L^2}{C_L}\right) = 0$
    4.  $C_L - C_{L_\mathrm{max}} \le 0$
    5.  $\delta_T - 1 \le 0$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    6.  $\mu_1, \mu_2\ge 0$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    7.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    8. $\mu_2 (\delta_T - 1) = 0$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## KKT analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active or inactive.

    ### _Interior solutions_ 

    Assuming that that $C_L < C_{L_\mathrm{max}}$ and $\delta_T < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1,\mu_2=0$. 

    From stationarity condition (2): $\lambda_1 = 0$.

    It can now be seen that stationarity condition (1) is never verified.

    It can be concluded that the maximum speed cannot be achieved in the interior of the domain. The maximum speed must lie on at least one of the boundaries defined by $C_L = C_{L_\mathrm{max}}$ or $\delta_T = 1$.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## _Thrust-limited maximum airspeed_

    $\delta_T=1 \quad \Rightarrow \quad \mu_2 > 0$

    $C_L < C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 = 0$

    From stationarity condition (2): 

    $$
    \lambda_1 = -\frac{\mu_2}{T_{a0}\sigma^\beta} \lt 0
    $$

    Stationarity condition (1) then becomes:

    $$
    \begin{align}
    \mu_1 &= -\frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}}C_L^{-1/2} + \lambda_1 W\left(\frac{KC_L^2 - C_{D_0}}{C_L^2}\right) = 0 \nonumber \\
    \Leftrightarrow \quad \lambda_1 &= \frac{\frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}}C_L^{-1/2}}{W\left(\frac{KC_L^2 - C_{D_0}}{C_L^2}\right)} \lt 0 \quad \Leftrightarrow \frac{1}{KC_L^2 - C_{D_0}} \lt 0 \quad \Leftrightarrow \quad C_L \lt \sqrt{\frac{C_{D_0}}{K}} = C_{L_E} \nonumber
    \end{align}
    $$

    This shows that maximum speed is obtained, intuitively, on the positive (right-hand side) branch of the performance diagram.

    Note: both $C_L \lt C_{L_E}$ and $C_L\lt C_{L_\mathrm{max}}$ can be true in either case of $C_{L_\mathrm{max}} \geq C_{L_E}$. The loosest, and best-case design is of course when $C_{L_E} \lt C_{L_\mathrm{max}}$, meaning that the aircraft is able to fly on the induced (left-hand side) branch of the performance diagram.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The corresponding optimum value of the $C_L$ is obtained by solving the primal feasibiliy condition (3), resulting in the well known:

    $$
    C_{L_{1, 2}}^* = \frac{T_{a0}\sigma^\beta}{2KW} \left[1\pm\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]
    $$

    Which exists for:

    $$
    1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2 \ge 0
    \quad \Leftrightarrow \quad \frac{W}{\sigma^\beta} \le  T_{a0} E_\mathrm{max}
    $$

    Where we are interested in the lower value, with the - sign. In the case where $C_{L_E} \lt C_{L_\mathrm{max}}$, this value is feasible when:

    $$
    C_L^* \lt C_{L_E} \quad \Leftrightarrow \quad \frac{W}{\sigma^\beta} \lt  T_{a0} E_\mathrm{max}
    $$

    Meaning that the minimum drag at current altitude and weight is less then the available thrust.

    Thus the optimal values are:

    $$
    \delta_T^* = 1, \quad C_L^* = \frac{T_{a0}\sigma^\beta}{2KW} \left[1-\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right], \quad \text{for} \:\:\frac{W}{\sigma^\beta} \lt  T_{a0} E_\mathrm{max}, \quad\text{if}\:\: C_{L_\mathrm{max}} \gt \sqrt{\frac{C_{D_0}}{K}}
    $$
    """
    )
    return


@app.cell
def _(atmos, np):
    def maxthrust_condition(W, h, E_max, Ta0, beta, CLstar, CLmax, CD0, K):
        sigma = atmos.rhoratio(h)
        condition = ((W / (sigma**beta)) < (E_max * Ta0)) & (
            CLmax > np.sqrt(CD0 / K)
        )

        return condition
    return (maxthrust_condition,)


@app.cell
def _(
    CD0,
    CL_from_horizontal_constraint,
    CLmax,
    E_max,
    K,
    S,
    Ta0,
    W_selected,
    beta,
    h_array,
    idx_selected,
    maxthrust_condition,
    np,
    velocity,
):
    CLstar_maxthrust = CL_from_horizontal_constraint(
        W_selected, h_array, S, CD0, K, Ta0, beta, "jet"
    )[1]  # select the solution with the minus

    maxthrust_mask = maxthrust_condition(
        W_selected, h_array, E_max, Ta0, beta, CLstar_maxthrust, CLmax, CD0, K
    )

    CLopt_maxthrust = np.where(maxthrust_mask, CLstar_maxthrust, np.nan)

    velocity_maxthrust_harray = velocity(
        W_selected, h_array, CLopt_maxthrust, S, cap=False
    )

    dTopt_maxthrust = np.where(
        maxthrust_mask,
        1,
        np.nan,
    )

    CLopt_maxthrust_selected = CLopt_maxthrust[idx_selected]
    dTopt_maxthrust_selected = dTopt_maxthrust[idx_selected]

    velocity_maxthrust_selected = velocity_maxthrust_harray[idx_selected]
    return (
        CLopt_maxthrust_selected,
        dTopt_maxthrust_selected,
        velocity_maxthrust_harray,
        velocity_maxthrust_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxthrust_selected,
    a_harray,
    active_selection,
    atmos,
    constraint,
    dT_array,
    dTopt_maxthrust_selected,
    go,
    h_array,
    h_selected,
    make_subplots,
    max_colorbar,
    min_colorbar,
    mo,
    np,
    velocity_maxthrust_harray,
    velocity_maxthrust_selected,
    velocity_stall_harray,
    velocity_surface,
    xy_lowerbound,
):
    fig_maxthrust_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot, first four are template
    fig_maxthrust_optimum.add_traces(
        [
            go.Heatmap(
                x=CL_array,
                y=dT_array,
                z=1 / velocity_surface,
                opacity=0.9,
                name="1/Velocity",
                colorscale="viridis",
                zsmooth="best",
                zmin=min_colorbar,
                zmax=max_colorbar,
                colorbar={"title": "V<sup>-1</sup> (s/m)"},
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
                x=[CL_array[30]],
                y=[constraint[30] - 0.07],
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
                textfont=dict(size=14, family="Arial"),
            ),
            go.Scatter(
                x=[CLopt_maxthrust_selected],
                y=[dTopt_maxthrust_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=10,
                    color="#FFFFFF",
                    symbol="circle",
                ),
                name="V<sup>-1</sup><sub>min</sub>",
                customdata=[1 / velocity_maxthrust_selected],
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub>: 1 <br>1/V: %{customdata}<extra></extra>",
            ),
        ],
        cols=1,
        rows=1,
    )

    # Traces on the flight envelope, first four traces are template
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
                x=velocity_maxthrust_harray,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=3, color="rgb(232,158,184)"),
                showlegend=False,
                name="V<sub>max</sub>",
            ),
            go.Scatter(
                x=[velocity_maxthrust_selected],
                y=[h_selected / 1e3],
                mode="markers+text",
                marker=dict(size=10, color="#FFFFFF"),
                name="V<sub>max</sub>",
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
        range=[
            xy_lowerbound,
            max(atmos.a(0), np.nanmax(velocity_maxthrust_harray)) + 15,
        ],
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
            "text": f"Thrust-limited maximum airspeed for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        }
    )

    mo.output.clear()
    return (fig_maxthrust_optimum,)


@app.cell
def _(variables_stack):
    variables_stack
    return


@app.cell
def _(fig_maxthrust_optimum):
    fig_maxthrust_optimum
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ###_Lift-limited minimum airspeed_

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$ 

    $0 < \delta_T < 1 \quad \Rightarrow \quad \mu_2 = 0$.

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1): 

    $$
    \mu_1 = -\frac{1}{2}\sqrt{\frac{\rho_0 S \sigma }{2WC_{L_\mathrm{max}}}}>0, \quad \mathrm{for} \quad C_{L_\mathrm{max}} \lt 0, \mathrm{impossible}
    $$

    The solution cannot be obtained at $C_{L_\mathrm{max}}$, ehivh is intuitive. As a matter of fact: 

    $$
    \min_{C_L} \frac{1}{V} = \sqrt{\frac{\rho S C_L}{2W}} \quad \Leftrightarrow \quad \min_{C_L} \sqrt{C_L} \quad \Leftrightarrow \quad \min_{C_L} C_L
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### _Thrust- and lift-limited minimum speed_

    $\delta_T = 1 \quad \Rightarrow \quad \mu_2 > 0$

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$.

    From the stationary conditions (2):

    $$
    \lambda_1 = -\frac{\mu_2}{T_{a0}\sigma^\beta} < 0
    $$

    From stationary condition (1): 

    $$
    \mu_1 = \lambda_1 W\left(\frac{KC_{L_\mathrm{max}}^2 - C_{D_0}}{C_{L_\mathrm{max}}^2}\right) -\frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}\frac{1}{C_{L_\mathrm{max}}}} \gt 0
    $$

    $$
    \frac{\frac{1}{2}\sqrt{\rho_0\frac{S}{2}\frac{\sigma}{W}\frac{1}{C_{L_\mathrm{max}}}}}{W\left(\frac{KC_{L_\mathrm{max}}^2 - C_{D_0}}{C_{L_\mathrm{max}}^2}\right)} \lt \lambda_1 \lt 0 \quad \Leftarrow \quad C_{L_\mathrm{max}} \lt \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$


    In other words this condition is verified only if the aircraft would not be able to fly in the condition of maximum aerodynamic efficiency (or minimum drag in steady level flight) because it woudl stall at a higher speed.

    From (3), the same derivation as the previous case results in

    $$
    C_L^* = C_{L_\mathrm{max}}, \quad \delta_T^*=1, \quad \frac{W}{\sigma^\beta} = T_{a0}E_S, \quad \mathrm{if} \quad C_{L_\mathrm{max}} \lt \sqrt{\frac{C_{D_0}}{K}}
    $$
    """
    )
    return


@app.cell
def _(atmos, np):
    def maxlift_thrust_altitude(W, beta, CLmax, CD0, K, Ta0, E_S):
        sigma_exp = W / Ta0 / E_S

        sigma = sigma_exp ** (1 / (beta))

        h = atmos.altitude(sigma)

        return np.where((h > 0) & (CLmax < np.sqrt(CD0 / K)), h, np.nan)
    return (maxlift_thrust_altitude,)


@app.cell
def _(
    CD0,
    CL_array,
    CLmax,
    E_S,
    K,
    S,
    Ta0,
    W_selected,
    beta,
    drag,
    horizontal_constraint,
    maxlift_thrust_altitude,
    np,
    velocity,
):
    maxlift_thrust_h = maxlift_thrust_altitude(
        W_selected, beta, CLmax, CD0, K, Ta0, E_S
    )

    CLopt_maxlift_thrust = CLmax

    velocity_maxlift_thrust_selected = velocity(
        W_selected, maxlift_thrust_h, CLopt_maxlift_thrust, S, cap=True
    )

    velocity_CLarray_maxlift_thrust_h = velocity(
        W_selected, maxlift_thrust_h, CL_array, S, cap=True
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
        Ta0,
        beta,
        velocity_CLarray_maxlift_thrust_h,
        S,
        drag_maxlift_thrust_h_curve,
        type="jet",
    )

    velocity_maxlift_thrust_surface = np.tile(
        velocity_CLarray_maxlift_thrust_h, (len(CL_array), 1)
    )

    min_colorbar_maxlift_thrust = 1 / np.max(velocity_maxlift_thrust_selected)
    max_colorbar_maxlift_thrust = min_colorbar_maxlift_thrust * 10

    if np.isnan(velocity_maxlift_thrust_selected):
        CLopt_maxlift_thrust = np.nan
    return (
        CLopt_maxlift_thrust,
        constraint_maxlift_thrust,
        dTopt_maxlift_thrust,
        max_colorbar_maxlift_thrust,
        maxlift_thrust_h,
        min_colorbar_maxlift_thrust,
        velocity_maxlift_thrust_selected,
        velocity_maxlift_thrust_surface,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxlift_thrust,
    a_harray,
    active_selection,
    atmos,
    constraint_maxlift_thrust,
    dT_array,
    dTopt_maxlift_thrust,
    go,
    h_array,
    make_subplots,
    max_colorbar_maxlift_thrust,
    maxlift_thrust_h,
    min_colorbar_maxlift_thrust,
    mo,
    np,
    velocity_maxlift_thrust_selected,
    velocity_maxlift_thrust_surface,
    velocity_stall_harray,
    xy_lowerbound,
):
    fig_maxlift_thrust_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot, first four are template
    fig_maxlift_thrust_optimum.add_traces(
        [
            go.Heatmap(
                x=CL_array,
                y=dT_array,
                z=1 / velocity_maxlift_thrust_surface,
                opacity=0.9,
                name="1/Velocity",
                colorscale="viridis",
                zsmooth="best",
                zmin=min_colorbar_maxlift_thrust,
                zmax=max_colorbar_maxlift_thrust,
                colorbar={"title": "V<sup>-1</sup> (s/m)"},
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
                x=[CL_array[30]],
                y=[constraint_maxlift_thrust[30] - 0.07],
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
                name="V<sup>-1</sup><sub>min</sub>",
                customdata=[1 / velocity_maxlift_thrust_selected],
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub>: 1 <br>1/V: %{customdata}<extra></extra>",
            ),
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                showlegend=False,
                marker=dict(color="rgba(0,0,0,0)"),
            ),
        ],
        cols=1,
        rows=1,
    )

    # Traces on the flight envelope, first four traces are template
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
                name="V<sub>max</sub>",
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
        range=[
            xy_lowerbound,
            max(atmos.a(0), np.nanmax(1 / velocity_maxlift_thrust_selected)) + 15,
        ],
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
            "text": f"Thrust-lift limited maximum airspeed for {active_selection.full_name}",
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


@app.cell
def _(mo):
    mo.md(
        r"""Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for maximum speed moves in the graph."""
    )
    return


@app.cell
def _(velocity_maxthrust_harray):
    # Merge lines to have a continuous line showing up in the final flight envelope

    final_velocity_flightenvelope = velocity_maxthrust_harray
    return (final_velocity_flightenvelope,)


@app.cell
def _(
    a_harray,
    active_selection,
    atmos,
    final_velocity_flightenvelope,
    go,
    h_array,
    maxlift_thrust_h,
    mo,
    np,
    velocity_maxlift_thrust_selected,
    velocity_maxthrust_harray,
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
                x=final_velocity_flightenvelope,
                y=h_array / 1e3,
                mode="lines",
                line=dict(width=3, color="rgb(232,158,184)"),
                showlegend=False,
                name="V<sub>max</sub>",
            ),
            go.Scatter(
                x=[velocity_maxlift_thrust_selected],
                y=[maxlift_thrust_h / 1e3],
                mode="markers",
                marker=dict(size=10, color="rgb(232,158,184)"),
                name="V<sub>max</sub>",
                showlegend=False,
            ),
        ],
    )

    fig_final_flightenv.update_layout(
        xaxis=dict(
            title=r"$V \: \text{(m/s)}$",
            range=[
                xy_lowerbound,
                max(atmos.a(0), np.nanmax(velocity_maxthrust_harray)) + 15,
            ],
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
            "text": f"Flight envelope for maximum speed for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        }
    )

    mo.output.clear()
    return (fig_final_flightenv,)


@app.cell
def _(mass_stack):
    mass_stack
    return


@app.cell
def _(fig_final_flightenv):
    fig_final_flightenv
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary

    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $V^*$ |
    |:-|:----------|:-------:|:------------:|:------|
    |Thrust and Lift-limited    | $\displaystyle \frac{W}{\sigma^\beta} =  T_{a0} E_S$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle V_s =\sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}$ |
    |Thrust-limited    | $\displaystyle \frac{W}{\sigma^\beta} \lt  T_{a0} E_\mathrm{max}$ | $\displaystyle \frac{T_{a0}\sigma^\beta}{2KW} \left[1-\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]$ | $1$ | $\displaystyle V_s \sqrt{\frac{2KWC_{L_\mathrm{max}}/T_{a0}\sigma^\beta}{1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}}}$ |
    """
    )
    return


@app.cell
def _():
    _defaults.nav_footer(
        after_file="MaxSpeed_Prop.py",
        after_title="Maximum Speed Simplified Propeller",
        above_file="MaxSpeed.py",
        above_title="Maximum Speed Homepage",
        above_before=True,
    )
    return


if __name__ == "__main__":
    app.run()
