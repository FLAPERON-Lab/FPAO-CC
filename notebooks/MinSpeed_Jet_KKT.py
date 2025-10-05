import marimo

__generated_with = "0.15.2"
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Minimum airspeed: simplfied jet aircraft

    $$
    \begin{aligned}
        \min_{C_L, \delta_T} 
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


@app.cell(hide_code=True)
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
        a,
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

    # Calculate the g1_eq constraint curve
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
    return constraint, velocity_surface, velocity_user_selected


@app.cell
def _(
    CL_array,
    CL_slider,
    a,
    active_selection,
    constraint,
    dT_array,
    dT_slider,
    go,
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
                z=velocity_surface,
                opacity=0.9,
                name="Velocity",
                colorscale="viridis",
                cmax=a,
                cmin=0,
                colorbar={"title": "Velocity (m/s)"},
            ),
            go.Scatter3d(
                x=CL_array,
                y=constraint,
                z=velocity_surface[0],
                opacity=1,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter3d(
                x=[CL_array[50] - 0.1],
                y=[constraint[50] - 0.1],
                z=[velocity_surface[0, 50] - 0.1 + 10],
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
                z=[velocity_user_selected],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=3,
                    color="white",
                    symbol="circle",
                ),
                name="Design Point",
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub> : %{y}<br>V: %{z}<extra>%{fullData.name}</extra>",
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
            zaxis=dict(title="V (m/s)", range=[0, a]),
        ),
    )

    fig_initial.update_layout(
        scene_camera=camera,
        title={
            "text": f"Minimum airspeed domain for {active_selection.full_name}",
            "font": {"size": 25},
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    mo.output.clear()
    return (fig_initial,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## KKT formulation
    To be reconducted in the standard KKT analysis format, the objective function is expressed in terms of the controls by direct elimination of $c_1^\mathrm{eq}$.
    Also, minimizing $V$ is equivalent to minimizing $V^2$, because the square power function is monotonically increasing.
    Therefore, to simplify the calculations, the problem is rewritten as follows:
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \begin{aligned}
        \min_{C_L, \delta_T} 
        & \quad V^2 = \frac{2W}{\rho S C_L} \\
        \text{subject to} 
        & \quad g_1 = \frac{T}{W} - \frac{1}{E}  =\frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L} = 0 \\
        & \quad h_1 = C_L - C_{L_\mathrm{max}} \le 0 \\
        & \quad h_2 = -C_L \le 0 \\
        & \quad h_3 = \delta_T - 1 \le 0 \\
        & \quad h_4 = -\delta_T \le 0 \\
    \end{aligned}
    $$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""In the interactive graph below, select a simplified jet aircraft of your choice and experiment in finding an optimum by changing the control variables, $C_L$ and $\delta_T$. The design point is marked in white in the 3D velocity surface.""")
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(CL_slider, dT_slider, mo):
    mo.md(f"""Here you can modify the control variables to understand how it affects the design: {mo.hstack([dT_slider, CL_slider])}""")
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
def _(mo):
    mo.md(
        r"""
    ### Lagrangian function and KKT conditions

    The Lagrangian function combines the objective function with equality constraints using Lagrange multipliers ($\lambda_i$) and inequality constraints using KKT multipliers ($\mu_j$).

    $$
    \begin{aligned}
    \mathcal{L}(C_L, \delta_T, \lambda_1, \mu_1, \mu_2, \mu_3, \mu_4) = 
    \quad \frac{2W}{\rho S C_L}
    & + \\
    & + \lambda_1 \left[\frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L}\right] + \\
    & + \mu_1 (C_L - C_{L_\mathrm{max}}) + \\
    & + \mu_2 (-C_L) + \\
    & + \mu_3 (\delta_T - 1) +\\
    & + \mu_4 (-\delta_T)
    \end{aligned}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    A necessary condition for an optimal solution of the optimization problem $(C_L^*, \delta_T^*)$ to exist, the multipliers $\lambda_1, \mu_1, \mu_2, \mu_3, \mu_4$ have to meet the following conditions:

    **A. Stationarity ($\nabla L = 0$):** the gradient of the Lagrangian with respect to each decision variable must be zero

    1. $\displaystyle \frac{\partial \mathcal{L}}{\partial C_L} = -\frac{2W}{\rho S C_L^2} + \lambda_1 \left(\frac{C_{D_0}- KC_L^2}{C_L^2}\right) + \mu_1 - \mu_2 = 0$

    2.  $\displaystyle \frac{\partial \mathcal{L}}{\partial \delta_T} = \lambda_1 \frac{T_{a0}\sigma^\beta}{W} + \mu_3 - \mu_4 = 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **B. Primal feasibility: constraints are satisfied**

    3.  $\displaystyle \frac{\delta_T T_{a0}\sigma^\beta}{W} - \frac{C_{D_0} + K C_L^2}{C_L} = 0$
    4.  $C_L - C_{L_\mathrm{max}} \le 0$
    5.  $-C_L \le 0$
    6.  $\delta_T - 1 \le 0$
    7.  $-\delta_T \le 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **C. Dual feasibility: KKT multipliers for inequalities must be non-negative**

    8.  $\mu_1, \mu_2, \mu_3, \mu_4 \ge 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **D. Complementary slackness ($\mu_j h_j = 0$)**: inactive inequality constraint have null multipliers, as they do not contribute to the objective function. Active inequality constraints have positive multipliers, as they make the objective function worse.

    9.  $\mu_1 (C_L - C_{L_\mathrm{max}}) = 0$
    10. $\mu_2 (-C_L) = 0$
    11. $\mu_3 (\delta_T - 1) = 0$
    12. $\mu_4 (-\delta_T) = 0$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## KKT analysis

    We can now proceed to systematically examine the conditions where various inequality constraints are active or inactive.

    ### _Interior solutions_ 

    Assuming that that $0 < C_L < C_{L_\mathrm{max}}$ and $0 < \delta_T < 1$ is equivalent to consider all inequality constraints as inactive.

    Therefore: $\mu_1,\mu_2,\mu_3,\mu_4=0$. 

    From stationarity condition (2): $\lambda_1 = 0$.

    It can now be seen that stationarity condition (1) is never verified.

    It can be concluded that the minimum speed cannot be achieved in the interior of the domain. 
    The minimum must lie on at least one of the boundaries defined by $C_L = C_{L_\mathrm{max}}$ or $\delta_T = 1$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### _Lower boundary solutions_
    The case where $C_L=0$ and the case where $\delta_T=0$ can be immediately discaded because of the primal feasibility conditions.
    This means that $\mu_2=\mu_4=0$ in all cases.

    We can then proceed with the analysis of the cases where the boundaries $C_L = C_{L_\mathrm{max}}$ and $\delta_T = 1$ are active in any of the three possible combinations.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### _Thrust-limited minimum airspeed_

    $\delta_T=1 \quad \Rightarrow \quad \mu_3 > 0$

    $C_L < C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 = 0$

    From stationarity condition (2): 

    $$
    \lambda_1 = -\mu_3\frac{W}{T_{a0}\sigma^\beta} \quad \Rightarrow \quad \lambda_1 < 0
    $$

    Stationarity condition (1) then becomes:

    $$
    \frac{2T_{a0}\sigma^\beta}{\rho S C_L^2} + \mu_3\left( \frac{C_{D_0}-KC_L^2}{C_L^2}\right) = 0
    \quad \text{and } \quad 
    \mu_3>0 
    \quad \Rightarrow \quad 
    C_L > \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    and implies that the thrust-limited minimum airspeed is obtained strictly on the left branch of the drag performance diagram, at a lift-coefficient strictly higher than the one for maximum aerodynamic efficiency.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The corresponding optimum value of the $C_L$ is obtained by solving the primal feasibiliy condition (3) and taking the highest of the two solutions:

    $$
    C_L^* = \frac{T_{a0}\sigma^\beta}{2KW} \left[1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]
    $$

    It has still to be verified that $C_L^* < C_{L_\mathrm{max}}$, which depends on the numerical values of the design parameters, and on the current values of the weight and altitude.

    First, this optimum value of the lift-coefficient is achievable for 

    $$
    1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2 \ge 0
    \quad \Rightarrow \quad 
    \frac{W}{\sigma^\beta} \le  T_{a0} E_\mathrm{max}
    $$

    The limit equality can be used to calculate the corresponding limit altitude at which the minimum speed is limited by thrust, for a given weight. This is called the _theoretcal ceiling_.

    Second, the optimum value is lower than $C_{L_\mathrm{max}}$ if

    $$
    \frac{W}{\sigma^\beta} > T_{a0} E_\mathrm{S}
    $$

    If both of these conditions are verified, the corresponding minimum airspeed is:

    $$
    V^* = 
    \sqrt{\frac{4KW^2/\rho S T_{a0}\sigma^\beta}{1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}}}
    = V_s \sqrt{\frac{2KWC_{L_\mathrm{max}}/T_{a0}\sigma^\beta}{1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}}}
    $$
    """
    )
    return


@app.cell
def _(atmos):
    def maxthrust_condition(W, h, E_max, Ta0, beta, CLstar, CLmax):
        sigma = atmos.rhoratio(h)
        condition = ((W / (sigma**beta)) <= (E_max * Ta0)) & (CLstar < CLmax)

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
    constraint,
    h_array,
    idx_selected,
    maxthrust_condition,
    np,
    velocity,
    velocity_surface,
):
    CLstar_maxthrust = CL_from_horizontal_constraint(
        W_selected, h_array, S, CD0, K, Ta0, beta, "jet"
    )[0]

    maxthrust_mask = maxthrust_condition(
        W_selected, h_array, E_max, Ta0, beta, CLstar_maxthrust, CLmax
    )

    CLopt_maxthrust = np.where(maxthrust_mask, CLstar_maxthrust, np.nan)

    velocity_maxthrust_harray = velocity(W_selected, h_array, CLopt_maxthrust, S)

    dTopt_maxthrust = np.where(
        maxthrust_mask,
        1,
        np.nan,
    )

    CLopt_maxthrust_selected = CLopt_maxthrust[idx_selected]
    dTopt_maxthrust_selected = dTopt_maxthrust[idx_selected]

    velocity_maxthrust_selected = velocity_maxthrust_harray[idx_selected]

    constraint_on_surface = np.where(
        np.isnan(velocity_surface[0, :]), np.nan, constraint
    )
    return (
        CLopt_maxthrust_selected,
        constraint_on_surface,
        dTopt_maxthrust_selected,
        velocity_maxthrust_harray,
        velocity_maxthrust_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxthrust_selected,
    a,
    a_harray,
    active_selection,
    atmos,
    constraint,
    constraint_on_surface,
    dT_array,
    dTopt_maxthrust_selected,
    go,
    h_array,
    h_selected,
    make_subplots,
    mo,
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
                z=velocity_surface,
                opacity=0.9,
                name="Velocity",
                colorscale="viridis",
                zsmooth="best",
                zmin=0,
                zmax=a,
                colorbar={"title": "Velocity (m/s)"},
            ),
            go.Scatter(
                x=CL_array,
                y=constraint_on_surface,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter(
                x=[CL_array[50]],
                y=[constraint[50] - 0.1],
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                textfont=dict(size=14, family="Arial"),
                showlegend=False,
                name="g1 constraint",
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
                name="V<sub>min</sub>",
                customdata=[velocity_maxthrust_selected],
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub>: 1 <br>V: %{customdata}<extra></extra>",
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
                name="V_min",
            ),
            go.Scatter(
                x=[velocity_maxthrust_selected],
                y=[h_selected / 1e3],
                mode="markers+text",
                marker=dict(size=10, color="#FFFFFF"),
                name="V<sub>min</sub>",
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
            "text": f"Thrust-limited minimum airspeed for {active_selection.full_name}",
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ###_Lift-limited minimum airspeed_

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$ 

    $0 < \delta_T < 1 \quad \Rightarrow \quad \mu_3 = 0$.

    From stationarity condition (2): $\lambda_1 = 0$.

    From stationarity condition (1): $\mu_1 = \frac{2W}{\rho S C_{L_\mathrm{max}}^2}>0$, which does not depend on the value of $\delta_T$, and is always verified.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The corresponding value of the throttle is calculated from the primal feasibility condition (3):

    $$
    \delta_T 
    = \frac{W}{T_{a0}\sigma^\beta} \frac{C_{D_0} + K C^2_{L_\mathrm{max}}}{C_{L_\mathrm{max}}} 
    = \frac{W}{T_{a0}\sigma^\beta} \frac{1}{E_S} 
    $$

    This is valid only if the calculated $\delta_T$ is strictly lower than the maximum allowed, which corresponds to:

    $$
    \frac{W}{\sigma^\beta} < T_{a0} E_S
    $$

    The limit equality can be used to calculate the corresponding limit altitude at which the minimum speed is limited by lift, for a given weight.

    The corresponding minimum airspeed is called the _stall speed_.

    $$
    V^* = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}
    $$
    """
    )
    return


@app.cell
def _(atmos):
    def maxlift_condition(W, h, E_s, Ta0, beta):
        sigma = atmos.rhoratio(h)
        condition = (W / (sigma**beta)) < (E_s * Ta0)

        return condition
    return (maxlift_condition,)


@app.cell
def _(
    CLmax,
    E_S,
    S,
    Ta0,
    W_selected,
    atmos,
    beta,
    h_array,
    idx_selected,
    maxlift_condition,
    np,
    velocity,
):
    maxlift_mask = maxlift_condition(W_selected, h_array, E_S, Ta0, beta)

    CLopt_maxlift = np.where(maxlift_mask, CLmax, np.nan)

    velocity_maxlift_harray = velocity(W_selected, h_array, CLopt_maxlift, S)

    dTopt_maxlift = np.where(
        maxlift_mask,
        W_selected / Ta0 / (atmos.rhoratio(h_array) ** beta) / E_S,
        np.nan,
    )

    CLopt_maxlift_selected = CLopt_maxlift[idx_selected]
    dTopt_maxlift_selected = dTopt_maxlift[idx_selected]

    velocity_maxlift_selected = velocity_maxlift_harray[idx_selected]
    return (
        CLopt_maxlift_selected,
        dTopt_maxlift_selected,
        velocity_maxlift_harray,
        velocity_maxlift_selected,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxlift_selected,
    a,
    a_harray,
    active_selection,
    atmos,
    constraint,
    constraint_on_surface,
    dT_array,
    dTopt_maxlift_selected,
    go,
    h_array,
    h_selected,
    make_subplots,
    mo,
    velocity_maxlift_harray,
    velocity_maxlift_selected,
    velocity_stall_harray,
    velocity_surface,
    xy_lowerbound,
):
    fig_maxlift_optimum = make_subplots(
        rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    # Traces on the 3D plot, first four are template
    fig_maxlift_optimum.add_traces(
        [
            go.Heatmap(
                x=CL_array,
                y=dT_array,
                z=velocity_surface,
                opacity=0.9,
                name="Velocity",
                zsmooth="best",
                colorscale="viridis",
                zmin=0,
                zmax=a,
                colorbar={"title": "Velocity (m/s)"},
            ),
            go.Scatter(
                x=CL_array,
                y=constraint_on_surface,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter(
                x=[CL_array[50]],
                y=[constraint[50] - 0.1],
                textposition="middle left",
                mode="markers+text",
                text=["g<sub>1</sub>"],
                textfont=dict(size=14, family="Arial"),
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
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
                name="V<sub>min</sub>",
                customdata=[velocity_maxlift_selected],
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub>: 1 <br>V: %{customdata}<extra></extra>",
            ),
        ],
        cols=1,
        rows=1,
    )

    # Traces on the flight envelope, first four traces are template
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
                name="V<sub>min</sub>",
            ),
            go.Scatter(
                x=[velocity_maxlift_selected],
                y=[h_selected / 1e3],
                mode="markers+text",
                marker=dict(size=10, color="#FFFFFF"),
                name="V<sub>min</sub>",
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
        title_text=r"$V \text{(m/s)}$",
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
            "text": f"Lift-limited minimum airspeed for {active_selection.full_name}",
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
def _(mo):
    mo.md(
        r"""
    ### _Thrust- and lift-limited minimum speed_

    $\delta_T = 1 \quad \Rightarrow \quad \mu_3 > 0$

    $C_L = C_{L_\mathrm{max}} \quad \Rightarrow \quad \mu_1 > 0$.

    From the stationary conditions (2):

    $$
    \lambda_1 = -\frac{\mu_3}{T_{a0}\sigma^\beta} \quad \Rightarrow \quad \lambda_1 < 0
    $$

    From stationary condition (1): 

    $$
    \mu_1 = \frac{2W}{\rho S C_{L_\mathrm{max}}^2} + \mu_3\frac{W}{T_{a0}\sigma^\beta}\left(\frac{C_{D_0} - K C_{L_\mathrm{max}}^2}{C_{L_\mathrm{max}}^2}\right) > 0 
    \quad \text{if } \quad
    C_{L_\mathrm{max}} < \sqrt{\frac{C_{D_0}}{K}} = C_{L_E}
    $$

    In other words, this condition is reached only if the aircraft is designed in such a way that its maximum lift coefficient is lower than the one for maximum aerodynamic efficiency. 
    It is obvious then that, for the same combination of weight and altitude, its stall speed will be higher than the speed for maximum efficiency (and minimum drag), which would then be unreachable for the aircraft in Steady Level Flight.
    This is of course an undesired situation to be in, and should not be resulting out of good aerodynamic design.

    The primal feasibility equation (3) returns the expression of the condition where the minimum speed is limited by both thrust and lift capabilities of the aircraft.

    $$
    \frac{W}{\sigma^\beta} = T_{a0} E_S
    $$

    The corresponding value of the airspeed is once again

    $$
    V^* = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}
    $$
    """
    )
    return


@app.cell
def _(atmos, np):
    def maxlift_thrust_altitude(W, beta, Ta0, E_S):
        sigma_exp = W / Ta0 / E_S

        sigma = sigma_exp ** (1 / (beta))

        h = atmos.altitude(sigma)
        return np.where(h > 0, h, np.nan)
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
    maxlift_thrust_h = maxlift_thrust_altitude(W_selected, beta, Ta0, E_S)

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

    constraint_maxlift_on_surface = np.where(
        np.isnan(velocity_CLarray_maxlift_thrust_h),
        np.nan,
        constraint_maxlift_thrust,
    )
    return (
        CLopt_maxlift_thrust,
        constraint_maxlift_on_surface,
        dTopt_maxlift_thrust,
        maxlift_thrust_h,
        velocity_maxlift_thrust_selected,
        velocity_maxlift_thrust_surface,
    )


@app.cell
def _(
    CL_array,
    CLopt_maxlift_thrust,
    a,
    a_harray,
    active_selection,
    atmos,
    constraint_maxlift_on_surface,
    dT_array,
    dTopt_maxlift_thrust,
    go,
    h_array,
    make_subplots,
    maxlift_thrust_h,
    mo,
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
                z=velocity_maxlift_thrust_surface,
                opacity=0.9,
                zsmooth="best",
                name="Velocity",
                colorscale="viridis",
                zmin=0,
                zmax=a,
                colorbar={"title": "Velocity (m/s)"},
            ),
            go.Scatter(
                x=CL_array,
                y=constraint_maxlift_on_surface,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(255, 0, 0, 0.35)", width=10),
                name="g1 constraint",
            ),
            go.Scatter(
                x=[CL_array[50]],
                y=[constraint_maxlift_on_surface[50] - 0.1],
                textposition="middle left",
                mode="markers+text",
                textfont=dict(size=14, family="Arial"),
                text=["g<sub>1</sub>"],
                marker=dict(size=1, color="rgba(255, 0, 0, 0.0)"),
                showlegend=False,
                name="g1 constraint",
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
                name="V<sub>min</sub>",
                customdata=[velocity_maxlift_thrust_selected],
                hovertemplate="C<sub>L</sub>: %{x}<br>δ<sub>T</sub>: 1 <br>V: %{customdata}<extra></extra>",
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
                mode="markers+text",
                marker=dict(size=10, color="#FFFFFF"),
                name="V<sub>min</sub>",
                showlegend=False,
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
        title_text=r"$V \text{(m/s)}$",
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
            "text": f"Thrust-lift limited minimum airspeed for {active_selection.full_name}",
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
    mo.md(r"""Now after deriving all the optima for each condition we can summarize the flight envelopes in one graph, as shown below. Experiment with the weight of the aircrarft to understand how the theoretical ceiling for minimum speed moves in the graph.""")
    return


@app.cell
def _(np, velocity_maxlift_harray, velocity_maxthrust_harray):
    # Merge lines to have a continuous line showing up in the final flight envelope

    final_velocity_flightenvelope = np.where(
        np.isnan(velocity_maxlift_harray),
        velocity_maxthrust_harray,
        velocity_maxlift_harray,
    )
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
    velocity_maxlift_thrust_selected,
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
                name="V<sub>min</sub>",
            ),
            go.Scatter(
                x=[velocity_maxlift_thrust_selected],
                y=[maxlift_thrust_h / 1e3],
                mode="markers",
                marker=dict(size=10, color="rgb(232,158,184)"),
                name="V<sub>min</sub>",
                showlegend=False,
            ),
        ],
    )

    fig_final_flightenv.update_layout(
        xaxis=dict(
            title=r"$V \text{(m/s)}$",
            range=[xy_lowerbound, atmos.a(0) + 15],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
        yaxis=dict(
            title=r"$h \: \text{(km)}$",
            range=[xy_lowerbound, 20],
            showgrid=True,
            gridcolor="#515151",
            gridwidth=1,
        ),
    )

    fig_final_flightenv.update_layout(
        title={
            "text": f"Flight envelope for minimum airspeed for {active_selection.full_name}",
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Summary

    | Name | Condition | $C_L^*$ | $\delta_T^*$ | $V^*$ |
    |:- |:----------|:-------:|:------------:|:------|
    |Lift-limited    | $\displaystyle \frac{W}{\sigma^\beta} < T_{a0} E_S$ | $C_{L_\mathrm{max}}$ | $\displaystyle \frac{W}{T_{a0}\sigma^\beta} \frac{1}{E_S}$ | $\displaystyle V_s = \sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}$ |
    |Thrust and Lift-limited    | $\displaystyle \frac{W}{\sigma^\beta} =  T_{a0} E_S$, $C_{L_\mathrm{max}} < \sqrt{\frac{C_{D_0}}{K}}$ | $C_{L_\mathrm{max}}$ | $1$ | $\displaystyle V_s =\sqrt{\frac{2W}{\rho S C_{L_\mathrm{max}}}}$ |
    |Thrust-limited    | $\displaystyle T_{a0} E_\mathrm{S} < \frac{W}{\sigma^\beta} \le  T_{a0} E_\mathrm{max}$ | $\displaystyle \frac{T_{a0}\sigma^\beta}{2KW} \left[1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}\right]$ | $1$ | $\displaystyle V_s \sqrt{\frac{2KWC_{L_\mathrm{max}}/T_{a0}\sigma^\beta}{1+\sqrt{1-\left(\frac{W}{E_\mathrm{max}T_{a0}\sigma^\beta}\right)^2}}}$ |
    """
    )
    return


if __name__ == "__main__":
    app.run()
